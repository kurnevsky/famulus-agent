//! Session persistence: every conversation is an
//! append-only JSONL file in one global directory, created lazily on the first
//! persisted message, and resumable later.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Local};
use rig_core::completion::Message;
use rig_core::message::UserContent;
use serde::{Deserialize, Serialize};

use crate::compaction;

/// One line of a session file.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Record {
  /// First line of every file.
  Header {
    id: String,
    created: DateTime<Local>,
    cwd: String,
    model: String,
  },
  /// A transcript message appended after a turn (or recovered from an abort).
  Message { message: Message },
  /// Compaction replaced the history with a summary plus the kept tail.
  Compaction { summary: String, history: Vec<Message> },
  /// The user rewound the session; the history is cut to `len` messages.
  ///
  /// A length rather than the messages themselves: replay rebuilds the same
  /// history this counted, so the two always agree and the file stays small.
  Rewind { len: usize },
  /// The user named the session.
  Name { name: String },
}

/// The global directory holding all session files.
#[derive(Clone, Debug)]
pub struct Store {
  dir: PathBuf,
}

/// Metadata for the session picker.
#[derive(Clone, Debug)]
pub struct SessionInfo {
  pub path: PathBuf,
  pub id: String,
  pub name: Option<String>,
  pub cwd: String,
  pub modified: DateTime<Local>,
  pub message_count: usize,
  pub first_message: String,
}

impl SessionInfo {
  /// Name if set, otherwise the first user message.
  pub fn title(&self) -> &str {
    match &self.name {
      Some(name) => name,
      None if self.first_message.is_empty() => "(empty session)",
      None => &self.first_message,
    }
  }

  /// Compact relative age: `now`, `5m`, `3h`, `2d`, `1w`, `4mo`, `1y`.
  pub fn age(&self) -> String {
    let secs = (Local::now() - self.modified).num_seconds().max(0);
    let (mins, hours, days) = (secs / 60, secs / 3600, secs / 86400);
    if mins < 1 {
      "now".into()
    } else if mins < 60 {
      format!("{mins}m")
    } else if hours < 24 {
      format!("{hours}h")
    } else if days < 7 {
      format!("{days}d")
    } else if days < 30 {
      format!("{}w", days / 7)
    } else if days < 365 {
      format!("{}mo", days / 30)
    } else {
      format!("{}y", days / 365)
    }
  }
}

impl Store {
  /// `$FA_SESSIONS_DIR`, else `$XDG_DATA_HOME/fa/sessions`, else
  /// `~/.local/share/fa/sessions`.
  pub fn default_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("FA_SESSIONS_DIR").filter(|d| !d.is_empty()) {
      return PathBuf::from(dir);
    }
    let data_home = std::env::var_os("XDG_DATA_HOME")
      .filter(|d| !d.is_empty())
      .map(PathBuf::from)
      .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".local/share")))
      .unwrap_or_else(|| PathBuf::from("."));
    data_home.join("fa").join("sessions")
  }

  pub fn new(dir: PathBuf) -> Self {
    Self { dir }
  }

  /// All sessions, most recently modified first. Unreadable files are skipped.
  pub fn list(&self) -> Vec<SessionInfo> {
    let Ok(entries) = std::fs::read_dir(&self.dir) else {
      return Vec::new();
    };
    let mut sessions: Vec<SessionInfo> = entries
      .flatten()
      .map(|e| e.path())
      .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
      .filter_map(|p| read_info(&p).ok())
      .collect();
    sessions.sort_by_key(|s| std::cmp::Reverse(s.modified));
    sessions
  }

  pub fn most_recent(&self) -> Option<SessionInfo> {
    self.list().into_iter().next()
  }

  /// Resolve `--session`: a file path, or a session id (or unique id prefix).
  pub fn find(&self, needle: &str) -> Option<PathBuf> {
    let as_path = Path::new(needle);
    if as_path.is_file() {
      return Some(as_path.to_path_buf());
    }
    let candidates: Vec<SessionInfo> = self.list().into_iter().filter(|s| s.id.starts_with(needle)).collect();
    match candidates.as_slice() {
      [one] => Some(one.path.clone()),
      _ => None,
    }
  }
}

fn read_info(path: &Path) -> Result<SessionInfo> {
  let file = File::open(path)?;
  let modified: DateTime<Local> = file.metadata()?.modified()?.into();
  let mut lines = BufReader::new(file).lines();
  let header = lines.next().context("empty session file")??;
  let Record::Header { id, cwd, .. } = serde_json::from_str(&header)? else {
    bail!("missing session header");
  };
  let mut info = SessionInfo {
    path: path.to_path_buf(),
    id,
    name: None,
    cwd,
    modified,
    message_count: 0,
    first_message: String::new(),
  };
  for line in lines {
    let Ok(record) = serde_json::from_str::<Record>(&line?) else {
      continue;
    };
    match record {
      Record::Message { message } => {
        info.message_count += 1;
        if info.first_message.is_empty()
          && let Some(text) = user_text(&message)
        {
          info.first_message = first_line(&text);
        }
      }
      Record::Compaction { .. } | Record::Rewind { .. } => {}
      Record::Name { name } => info.name = Some(name),
      Record::Header { .. } => {}
    }
  }
  Ok(info)
}

/// Text of a plain user message (not a tool result or compaction summary).
///
/// Which is also what makes a message a point the session can be rewound to:
/// the things the user typed, and nothing the loop put there itself.
pub fn user_text(message: &Message) -> Option<String> {
  let Message::User { content } = message else {
    return None;
  };
  let text: Vec<&str> = content
    .iter()
    .filter_map(|c| match c {
      UserContent::Text(t) => Some(t.text.as_str()),
      _ => None,
    })
    .collect();
  if text.is_empty() {
    return None;
  }
  let text = text.join("\n");
  if text.starts_with(compaction::SUMMARY_PREFIX) {
    return None;
  }
  Some(text)
}

fn first_line(text: &str) -> String {
  text
    .lines()
    .map(str::trim)
    .find(|l| !l.is_empty())
    .unwrap_or("")
    .chars()
    .take(200)
    .collect()
}

/// The live conversation and, when persistence is on, its file.
pub struct Session {
  pub id: String,
  pub name: Option<String>,
  pub cwd: String,
  pub model: String,
  pub created: DateTime<Local>,
  pub history: Vec<Message>,
  /// Where new sessions get their file; `None` disables persistence.
  dir: Option<PathBuf>,
  /// Open once the first record is written.
  file: Option<(PathBuf, File)>,
}

impl Session {
  pub fn new(store: Option<&Store>, cwd: &Path, model: &str) -> Self {
    Self {
      id: uuid::Uuid::new_v4().to_string(),
      name: None,
      cwd: cwd.display().to_string(),
      model: model.to_string(),
      created: Local::now(),
      history: Vec::new(),
      dir: store.map(|s| s.dir.clone()),
      file: None,
    }
  }

  /// Replay a session file. Appending continues in the same file.
  pub fn load(path: &Path) -> Result<Self> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().with_context(|| format!("{} is empty", path.display()))??;
    let Record::Header {
      id,
      created,
      cwd,
      model,
    } = serde_json::from_str(&header).context("invalid session header")?
    else {
      bail!("{} has no session header", path.display());
    };
    let mut session = Self {
      id,
      name: None,
      cwd,
      model,
      created,
      history: Vec::new(),
      dir: path.parent().map(Path::to_path_buf),
      file: None,
    };
    for (n, line) in lines.enumerate() {
      let line = line?;
      if line.trim().is_empty() {
        continue;
      }
      let record: Record =
        serde_json::from_str(&line).with_context(|| format!("{}: bad record on line {}", path.display(), n + 2))?;
      match record {
        Record::Message { message } => session.history.push(message),
        Record::Compaction { history, .. } => session.history = history,
        Record::Rewind { len } => session.history.truncate(len),
        Record::Name { name } => session.name = Some(name),
        Record::Header { .. } => {}
      }
    }
    let file = OpenOptions::new().append(true).open(path)?;
    session.file = Some((path.to_path_buf(), file));
    Ok(session)
  }

  /// Stop writing to disk (used for `--session` combined with `--no-session`).
  pub fn disable_persistence(&mut self) {
    self.dir = None;
    self.file = None;
  }

  pub fn persistent(&self) -> bool {
    self.dir.is_some()
  }

  /// The session file, once it exists.
  pub fn path(&self) -> Option<&Path> {
    self.file.as_ref().map(|(p, _)| p.as_path())
  }

  /// Extend the history and persist the new messages.
  pub fn append(&mut self, messages: Vec<Message>) -> Result<()> {
    let records: Vec<Record> = messages
      .iter()
      .cloned()
      .map(|message| Record::Message { message })
      .collect();
    self.history.extend(messages);
    self.write_all(&records)
  }

  /// Replace the history after compaction and persist the checkpoint.
  pub fn compacted(&mut self, history: Vec<Message>, summary: &str) -> Result<()> {
    self.history = history.clone();
    self.write_all(&[Record::Compaction {
      summary: summary.to_string(),
      history,
    }])
  }

  /// Cut the history back to its first `len` messages and record the cut.
  ///
  /// The file stays append-only: what was written before is still there, and
  /// replay drops it again on the way past. Nothing is written when the cut
  /// would change nothing, so an accidental rewind to the end leaves no trace.
  pub fn rewind(&mut self, len: usize) -> Result<()> {
    if len >= self.history.len() {
      return Ok(());
    }
    self.history.truncate(len);
    self.write_all(&[Record::Rewind { len }])
  }

  pub fn rename(&mut self, name: &str) -> Result<()> {
    let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
    self.name = Some(name.clone());
    self.write_all(&[Record::Name { name }])
  }

  fn write_all(&mut self, records: &[Record]) -> Result<()> {
    let Some(dir) = &self.dir else {
      return Ok(());
    };
    if self.file.is_none() {
      std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
      let path = dir.join(format!(
        "{}_{}.jsonl",
        self.created.format("%Y-%m-%dT%H-%M-%S"),
        self.id
      ));
      let mut file = OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("cannot create {}", path.display()))?;
      let header = Record::Header {
        id: self.id.clone(),
        created: self.created,
        cwd: self.cwd.clone(),
        model: self.model.clone(),
      };
      serde_json::to_writer(&mut file, &header)?;
      file.write_all(b"\n")?;
      self.file = Some((path, file));
    }
    let (_, file) = self.file.as_mut().expect("opened above");
    for record in records {
      serde_json::to_writer(&mut *file, record)?;
      file.write_all(b"\n")?;
    }
    file.flush()?;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn temp_store(name: &str) -> Store {
    let dir = std::env::temp_dir().join(format!("fa-sessions-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    Store::new(dir)
  }

  #[test]
  fn append_load_compaction_and_listing_round_trip() {
    let store = temp_store("roundtrip");
    let mut session = Session::new(Some(&store), Path::new("/work"), "mock");
    assert!(session.path().is_none(), "file is created lazily");
    assert!(store.list().is_empty());

    session
      .append(vec![
        Message::user("first question\nmore"),
        Message::assistant("answer"),
      ])
      .unwrap();
    let path = session.path().unwrap().to_path_buf();
    assert!(path.starts_with(&store.dir));
    session.rename("  my   session ").unwrap();
    let compacted = vec![
      compaction::summary_message("summary"),
      Message::user("second"),
      Message::assistant("reply"),
    ];
    session.compacted(compacted.clone(), "summary").unwrap();
    session.append(vec![Message::user("third")]).unwrap();

    let loaded = Session::load(&path).unwrap();
    assert_eq!(loaded.id, session.id);
    assert_eq!(loaded.name.as_deref(), Some("my session"));
    assert_eq!(loaded.cwd, "/work");
    let mut expected = compacted;
    expected.push(Message::user("third"));
    assert_eq!(loaded.history, expected);

    let infos = store.list();
    assert_eq!(infos.len(), 1);
    let info = &infos[0];
    assert_eq!(info.title(), "my session");
    assert_eq!(info.first_message, "first question");
    assert_eq!(info.message_count, 3);
    assert_eq!(info.age(), "now");
    assert_eq!(store.find(&session.id[..8]).as_deref(), Some(path.as_path()));
    assert_eq!(store.find(path.to_str().unwrap()).as_deref(), Some(path.as_path()));
    assert_eq!(store.most_recent().unwrap().id, session.id);

    // A resumed session keeps appending to the same file.
    let mut loaded = loaded;
    loaded.append(vec![Message::assistant("more")]).unwrap();
    assert_eq!(Session::load(&path).unwrap().history.len(), 5);
    std::fs::remove_dir_all(&store.dir).unwrap();
  }

  #[test]
  fn rewind_cuts_the_history_and_survives_a_reload() {
    let store = temp_store("rewind");
    let mut session = Session::new(Some(&store), Path::new("/work"), "mock");
    session
      .append(vec![
        Message::user("first"),
        Message::assistant("answer"),
        Message::user("second"),
        Message::assistant("reply"),
      ])
      .unwrap();
    let path = session.path().unwrap().to_path_buf();

    // Back to just before "second", which the caller puts back in the input.
    session.rewind(2).unwrap();
    assert_eq!(session.history, [Message::user("first"), Message::assistant("answer")]);
    // The file is still append-only: replay drops the cut messages again.
    assert_eq!(Session::load(&path).unwrap().history, session.history);

    // The same file keeps taking new messages after the cut.
    session.append(vec![Message::user("instead")]).unwrap();
    let reloaded = Session::load(&path).unwrap();
    assert_eq!(reloaded.history.len(), 3);
    assert_eq!(reloaded.history[2], Message::user("instead"));

    // A rewind that would change nothing writes nothing.
    let before = std::fs::read_to_string(&path).unwrap();
    session.rewind(3).unwrap();
    session.rewind(9).unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    std::fs::remove_dir_all(&store.dir).unwrap();
  }

  #[test]
  fn disabled_persistence_keeps_history_in_memory_only() {
    let mut session = Session::new(None, Path::new("/work"), "mock");
    session.append(vec![Message::user("hi")]).unwrap();
    assert_eq!(session.history.len(), 1);
    assert!(!session.persistent());
    assert!(session.path().is_none());
  }
}
