//! Session persistence: every conversation is an
//! append-only JSONL file in one global directory, created lazily on the first
//! persisted message, and resumable later. Deleting a branch is the one thing
//! that rewrites a file rather than adding to it.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Local};
use rig_core::completion::Message;
use rig_core::message::{AssistantContent, ToolResult, UserContent};
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
    /// The session this one was forked from, for provenance. Absent on
    /// sessions that were started rather than branched, and on files written
    /// before forking existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
  },
  /// A transcript message appended after a turn (or recovered from an abort).
  ///
  /// `id` and `parent` are what make the file a tree rather than a list: a
  /// message written after going back names the entry it was written under,
  /// and the path that was left keeps its own entries.
  Message {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
    /// What the transcript does not say about the tool results in this
    /// message, by the call each answers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    outcomes: Vec<Outcome>,
    message: Message,
  },
  /// Compaction replaced everything above it with a summary. The turns it
  /// kept verbatim follow as entries of their own, so they stay places the
  /// conversation can go back to.
  Compaction {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
    summary: String,
  },
  /// The conversation moved to an existing entry; what follows branches
  /// there. `None` moves back before the first message.
  Leaf { id: Option<String> },
  /// The user named the session.
  Name { name: String },
  /// The session changed model part-way through. The header keeps the one it
  /// opened on, so what it ran on last is written where it happened.
  Model { model: String },
}

/// How a tool result went, which its transcript does not record.
///
/// A failed result looks exactly like a successful one on the wire, and the
/// diff an `edit` produced is not in the transcript at all — it was worked
/// out while the edit ran. Both were on screen at the time, so both are kept
/// here, or reopening the session would quietly lose them.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Outcome {
  /// The call this answers, as the result names it.
  pub call: String,
  #[serde(default, skip_serializing_if = "std::ops::Not::not")]
  pub failed: bool,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub diff: Option<String>,
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

  /// Delete a session file, for good.
  ///
  /// Only a file of this store's: the picker hands back a path it read from
  /// the directory, and anything else is not this store's to remove.
  pub fn delete(&self, path: &Path) -> Result<()> {
    if path.parent() != Some(self.dir.as_path()) {
      bail!("{} is not a session of {}", path.display(), self.dir.display());
    }
    std::fs::remove_file(path).with_context(|| format!("cannot delete {}", path.display()))
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
  // Enough of the tree to count the branch the session is on. A file that
  // has been gone back through holds more entries than the conversation does,
  // and it is the conversation the picker is describing.
  let mut parents: HashMap<String, Option<String>> = HashMap::new();
  let mut checkpoints: HashSet<String> = HashSet::new();
  let mut leaf: Option<String> = None;
  for line in lines {
    let Ok(record) = serde_json::from_str::<Record>(&line?) else {
      continue;
    };
    match record {
      Record::Message {
        id, parent, message, ..
      } => {
        if info.first_message.is_empty()
          && let Some(text) = user_text(&message)
        {
          info.first_message = first_line(&text);
        }
        parents.insert(id.clone(), parent);
        leaf = Some(id);
      }
      Record::Compaction { id, parent, .. } => {
        parents.insert(id.clone(), parent);
        checkpoints.insert(id.clone());
        leaf = Some(id);
      }
      Record::Leaf { id } => leaf = id,
      Record::Name { name } => info.name = Some(name),
      // Neither says anything the picker shows.
      Record::Header { .. } | Record::Model { .. } => {}
    }
  }
  let mut at = leaf;
  // A checkpoint ends the walk, standing for everything above it. The bound
  // is against a parent link that loops: the file is only as good as what
  // last wrote it.
  while let Some(id) = at.filter(|_| info.message_count <= parents.len()) {
    info.message_count += 1;
    if checkpoints.contains(&id) {
      break;
    }
    at = parents.get(&id).cloned().flatten();
  }
  Ok(info)
}

/// Text of a plain user message (not a tool result or compaction summary).
///
/// Which is also what makes a message a point the session can be rewound to:
/// the things the user typed, and nothing the loop put there itself.
///
/// A prompt that attached images is one text part, then a note and an image
/// for each. Only the first is the user's: the notes were written to tell the
/// model what it was being given, and a prompt handed back to the input box
/// has to be the sentence that was typed, `@tokens` and all — which is what
/// attaches the images again when it is sent again.
pub fn user_text(message: &Message) -> Option<String> {
  let Message::User { content } = message else {
    return None;
  };
  let attachments = content.iter().any(|c| matches!(c, UserContent::Image(_)));
  let mut text: Vec<&str> = content
    .iter()
    .filter_map(|c| match c {
      UserContent::Text(t) => Some(t.text.as_str()),
      _ => None,
    })
    .collect();
  if attachments {
    text.truncate(1);
  }
  if text.is_empty() {
    return None;
  }
  let text = text.join("\n");
  if text.starts_with(compaction::SUMMARY_PREFIX) {
    return None;
  }
  Some(text)
}

/// Every way a tool result names the call it answers.
///
/// Rig mints its own handle for a call and keeps the provider's alongside it
/// when there was one, and which of the two a hook reports depends on the
/// provider. Offering both is what keeps the match from depending on that.
pub fn result_ids(result: &ToolResult) -> impl Iterator<Item = String> + '_ {
  [
    Some(result.call.as_str().to_string()),
    result.provider.as_ref().map(|p| p.call_id.clone()),
  ]
  .into_iter()
  .flatten()
}

/// The same for every tool result in a message.
fn call_ids(message: &Message) -> impl Iterator<Item = String> + '_ {
  let content = match message {
    Message::User { content } => Some(content),
    _ => None,
  };
  content
    .into_iter()
    .flatten()
    .filter_map(|c| match c {
      UserContent::ToolResult(result) => Some(result),
      _ => None,
    })
    .flat_map(result_ids)
}

/// Write the session file at `path` again without the entries in `gone`, and
/// without the moves to them.
///
/// The lines that stay are copied as they are rather than written anew, so
/// nothing but the removal changes. The copy is made beside the file and moved
/// over it, so a failure part-way leaves the file as it was.
///
/// The replayed leaf comes out the same: it is set by the last message or move
/// in the file, and one of those that is dropped named an entry in `gone`,
/// which the session is not on.
fn rewrite_without(path: &Path, gone: &HashSet<String>) -> Result<()> {
  let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
  let mut kept = String::with_capacity(text.len());
  for line in text.lines() {
    let id = match serde_json::from_str::<Record>(line) {
      Ok(Record::Message { id, .. } | Record::Compaction { id, .. } | Record::Leaf { id: Some(id) }) => Some(id),
      _ => None,
    };
    if id.is_none_or(|id| !gone.contains(&id)) {
      kept.push_str(line);
      kept.push('\n');
    }
  }
  let temp = path.with_extension("jsonl.tmp");
  std::fs::write(&temp, kept).with_context(|| format!("cannot write {}", temp.display()))?;
  std::fs::rename(&temp, path).with_context(|| format!("cannot replace {}", path.display()))
}

/// The compaction checkpoint as it travels in a history: a user message
/// wearing the summary markers.
fn is_summary(message: &Message) -> bool {
  let Message::User { content } = message else {
    return false;
  };
  content.iter().any(|c| match c {
    UserContent::Text(t) => t.text.starts_with(compaction::SUMMARY_PREFIX),
    _ => false,
  })
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

/// One entry of the session tree.
///
/// Going back does not delete anything: it moves the leaf, and what the
/// conversation said down the path it left stays here as a sibling branch,
/// which is what makes it possible to walk back into it later.
pub struct Node {
  pub id: String,
  pub parent: Option<String>,
  pub kind: NodeKind,
}

pub enum NodeKind {
  Message(Message),
  /// A compaction checkpoint: it stands for everything above it, so a branch
  /// that reaches one needs nothing further up. What the compaction kept
  /// verbatim follows it as entries of its own.
  Checkpoint {
    summary: String,
  },
}

/// A chain of entries as a conversation, the checkpoints in it wearing the
/// summary markers they travel in.
fn messages(chain: Vec<&Node>) -> Vec<Message> {
  chain
    .into_iter()
    .map(|node| match &node.kind {
      NodeKind::Message(message) => message.clone(),
      NodeKind::Checkpoint { summary } => compaction::summary_message(summary),
    })
    .collect()
}

/// The live conversation and, when persistence is on, its file.
pub struct Session {
  pub id: String,
  pub name: Option<String>,
  pub cwd: String,
  pub model: String,
  pub created: DateTime<Local>,
  /// The branch ending at `leaf`, rebuilt whenever the leaf moves. Kept here
  /// rather than walked on demand because every turn reads it.
  pub history: Vec<Message>,
  /// Every entry ever written, in the order it was written.
  nodes: Vec<Node>,
  /// Where the conversation currently ends; `None` is before the first entry.
  leaf: Option<String>,
  /// Counter behind `mint`, past every id already in the file.
  ids: u64,
  /// How each tool result went, by the call it answers.
  outcomes: HashMap<String, Outcome>,
  /// The session this one was forked from, written into its header.
  parent: Option<String>,
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
      nodes: Vec::new(),
      leaf: None,
      ids: 0,
      outcomes: HashMap::new(),
      parent: None,
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
      parent,
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
      nodes: Vec::new(),
      leaf: None,
      ids: 0,
      outcomes: HashMap::new(),
      parent,
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
      let (id, parent, kind) = match record {
        Record::Message {
          id,
          parent,
          outcomes,
          message,
        } => {
          session
            .outcomes
            .extend(outcomes.into_iter().map(|outcome| (outcome.call.clone(), outcome)));
          (id, parent, NodeKind::Message(message))
        }
        Record::Compaction { id, parent, summary } => (id, parent, NodeKind::Checkpoint { summary }),
        Record::Leaf { id } => {
          session.leaf = id;
          continue;
        }
        Record::Name { name } => {
          session.name = Some(name);
          continue;
        }
        Record::Model { model } => {
          session.model = model;
          continue;
        }
        Record::Header { .. } => continue,
      };
      session.remember(&id);
      session.leaf = Some(id.clone());
      session.nodes.push(Node { id, parent, kind });
    }
    session.history = session.branch(session.leaf.as_deref());
    let file = OpenOptions::new().append(true).open(path)?;
    session.file = Some((path.to_path_buf(), file));
    Ok(session)
  }

  /// Where the conversation currently ends.
  pub fn leaf(&self) -> Option<&str> {
    self.leaf.as_deref()
  }

  pub fn nodes(&self) -> &[Node] {
    &self.nodes
  }

  fn node(&self, id: &str) -> Option<&Node> {
    self.nodes.iter().find(|node| node.id == id)
  }

  /// The entry a cut at `id` would leave the conversation ending at.
  pub fn parent_of(&self, id: &str) -> Option<&str> {
    self.node(id).and_then(|node| node.parent.as_deref())
  }

  /// The entries from the root down to `leaf`, oldest first.
  ///
  /// `whole` is what a compaction checkpoint means to the caller. The
  /// conversation the model is given stops at one, since the checkpoint
  /// already carries everything above it; the transcript walks past it, because
  /// what it summarized was on screen when it happened and reopening the
  /// session should not lose it.
  fn ancestry(&self, leaf: Option<&str>, whole: bool) -> Vec<&Node> {
    let mut chain = Vec::new();
    let mut at = leaf;
    // A parent link that goes nowhere, or round in a circle, would otherwise
    // loop forever; the file is only as good as what last wrote it.
    while let Some(node) = at.and_then(|id| self.node(id)) {
      chain.push(node);
      let stop = !whole && matches!(node.kind, NodeKind::Checkpoint { .. });
      if stop || chain.len() > self.nodes.len() {
        break;
      }
      at = node.parent.as_deref();
    }
    chain.reverse();
    chain
  }

  /// The ids from the root down to `leaf`, oldest first.
  ///
  /// `whole` as in `ancestry`: the conversation the model is sent stops at a
  /// checkpoint, while the path the session is *on* — which is what the tree
  /// marks and the transcript draws — carries on above it.
  pub fn lineage(&self, leaf: Option<&str>, whole: bool) -> Vec<&str> {
    self.ancestry(leaf, whole).iter().map(|node| node.id.as_str()).collect()
  }

  /// The conversation ending at `leaf`, as the model is given it.
  pub fn branch(&self, leaf: Option<&str>) -> Vec<Message> {
    messages(self.ancestry(leaf, false))
  }

  /// Everything said on the way down to `leaf`, as it was on screen.
  ///
  /// The same walk as `branch`, except that it does not stop at a compaction
  /// checkpoint: the turns a checkpoint stands for are still in the file, and
  /// a resumed session shows them above the summary, where they were.
  pub fn transcript(&self, leaf: Option<&str>) -> Vec<Message> {
    messages(self.ancestry(leaf, true))
  }

  /// How long that conversation is, without building it.
  pub fn branch_len(&self, leaf: Option<&str>) -> usize {
    self.ancestry(leaf, false).len()
  }

  /// The same for the transcript, which a compaction leaves longer.
  pub fn transcript_len(&self, leaf: Option<&str>) -> usize {
    self.ancestry(leaf, true).len()
  }

  /// A fresh id, past anything the file already holds.
  fn mint(&mut self) -> String {
    self.ids += 1;
    self.ids.to_string()
  }

  /// Keep `mint` clear of an id read from the file.
  fn remember(&mut self, id: &str) {
    if let Ok(n) = id.parse::<u64>() {
      self.ids = self.ids.max(n);
    }
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

  /// Extend the conversation, each message a child of the one before it.
  pub fn append(&mut self, messages: Vec<Message>) -> Result<()> {
    self.append_with(messages, &HashMap::new())
  }

  /// The same, told how the tool calls these messages answer went, so the
  /// transcript can be drawn the same way when it is reopened. `outcomes` may
  /// describe calls from any turn; only the ones these messages actually
  /// answer are written.
  pub fn append_with(&mut self, messages: Vec<Message>, outcomes: &HashMap<String, Outcome>) -> Result<()> {
    let mut records = Vec::with_capacity(messages.len());
    for message in messages {
      let id = self.mint();
      let answered: Vec<Outcome> = call_ids(&message).filter_map(|id| outcomes.get(&id).cloned()).collect();
      for outcome in &answered {
        self.outcomes.insert(outcome.call.clone(), outcome.clone());
      }
      records.push(Record::Message {
        id: id.clone(),
        parent: self.leaf.clone(),
        outcomes: answered,
        message: message.clone(),
      });
      self.history.push(message.clone());
      self.nodes.push(Node {
        id: id.clone(),
        parent: self.leaf.take(),
        kind: NodeKind::Message(message),
      });
      self.leaf = Some(id);
    }
    self.write_all(&records)
  }

  /// How the tool result answering `call` went, if anything was recorded.
  pub fn outcome(&self, call: &str) -> Option<&Outcome> {
    self.outcomes.get(call)
  }

  /// Replace the history after compaction and persist the checkpoint.
  ///
  /// The checkpoint is one entry standing for everything above it; the turns
  /// the compaction kept verbatim are written after it as entries of their
  /// own, so each stays a place the conversation can go back to.
  pub fn compacted(&mut self, history: Vec<Message>, summary: &str) -> Result<()> {
    let id = self.mint();
    let record = Record::Compaction {
      id: id.clone(),
      parent: self.leaf.clone(),
      summary: summary.to_string(),
    };
    self.nodes.push(Node {
      id: id.clone(),
      parent: self.leaf.take(),
      kind: NodeKind::Checkpoint {
        summary: summary.to_string(),
      },
    });
    self.leaf = Some(id);
    self.history = vec![compaction::summary_message(summary)];
    self.write_all(&[record])?;
    // The summary the checkpoint already stands for is not written twice.
    let kept = history.into_iter().skip_while(is_summary).collect();
    self.append(kept)
  }

  /// Move the end of the conversation to `leaf`, and record the move.
  ///
  /// Nothing is deleted: what was said down the path being left stays in the
  /// file as a branch of its own, and this can walk back into it later. What
  /// follows becomes a child of `leaf`, which is where the branching happens.
  pub fn go_to(&mut self, leaf: Option<String>) -> Result<()> {
    if leaf == self.leaf {
      return Ok(());
    }
    self.leaf = leaf.clone();
    self.history = self.branch(leaf.as_deref());
    self.write_all(&[Record::Leaf { id: leaf }])
  }

  /// Remove the entry `id` and everything under it, answering how many
  /// entries went.
  ///
  /// This is the one thing that takes an entry out of the file rather than
  /// adding to it, so it is the one thing written by rewriting the file: an
  /// entry that is only marked as gone would still be there to read. The
  /// conversation the session is on is not something it removes — the next
  /// turn would be written under an entry that is no longer there — so going
  /// somewhere else comes first.
  ///
  /// An assistant turn that called tools and has nothing under it but the
  /// entry being removed goes with it: with its results gone it is a call no
  /// provider will accept an answer to, and it is no place to stop either.
  pub fn delete_branch(&mut self, id: &str) -> Result<usize> {
    let mut root = self.node(id).with_context(|| format!("no entry {id}"))?;
    while let Some(parent) = root.parent.as_deref().and_then(|p| self.node(p)) {
      let calls = matches!(&parent.kind, NodeKind::Message(Message::Assistant { content, .. })
        if content.iter().any(|c| matches!(c, AssistantContent::ToolCall(_))));
      let alone = self
        .nodes
        .iter()
        .filter(|n| n.parent.as_deref() == Some(&parent.id))
        .count()
        == 1;
      if !calls || !alone {
        break;
      }
      root = parent;
    }
    let root = root.id.clone();
    if self.lineage(self.leaf(), true).contains(&root.as_str()) {
      bail!("entry {root} is on the conversation the session is on");
    }
    // Everything under the root, found by sweeping the entries until a sweep
    // adds nothing: a child is written after its parent, so it is usually one.
    let mut gone: HashSet<String> = HashSet::from([root]);
    loop {
      let before = gone.len();
      for node in &self.nodes {
        if node.parent.as_ref().is_some_and(|p| gone.contains(p)) {
          gone.insert(node.id.clone());
        }
      }
      if gone.len() == before {
        break;
      }
    }
    if let Some((path, _)) = &self.file {
      rewrite_without(path, &gone)?;
      let file = OpenOptions::new().append(true).open(path)?;
      self.file = Some((path.clone(), file));
    }
    for node in self.nodes.iter().filter(|node| gone.contains(&node.id)) {
      if let NodeKind::Message(message) = &node.kind {
        for call in call_ids(message) {
          self.outcomes.remove(&call);
        }
      }
    }
    self.nodes.retain(|node| !gone.contains(&node.id));
    Ok(gone.len())
  }

  /// A new session carrying the conversation that ends at `leaf`.
  ///
  /// Where `go_to` branches inside this file, this starts another one: this
  /// session is left exactly as it is, and the new one names it as its parent.
  /// The messages are copied rather than referenced, so the fork stands on its
  /// own if the original is deleted.
  ///
  /// What is copied is the one path down to `leaf`, not the tree around it:
  /// the branches beside it belong to the conversation being left behind, and
  /// the fork starts as a straight line with nowhere of its own to go back to.
  pub fn fork(&self, leaf: Option<&str>) -> Result<Self> {
    let mut forked = Self {
      id: uuid::Uuid::new_v4().to_string(),
      name: None,
      cwd: self.cwd.clone(),
      model: self.model.clone(),
      created: Local::now(),
      history: Vec::new(),
      nodes: Vec::new(),
      leaf: None,
      ids: 0,
      // The fork draws the conversation it copied the same way this one did.
      outcomes: self.outcomes.clone(),
      parent: self.path().map(|path| path.display().to_string()),
      dir: self.dir.clone(),
      file: None,
    };
    // Forking from the very start leaves an empty session, which — like any
    // other empty session — gets its file once it has something to say.
    let history = self.branch(leaf);
    if !history.is_empty() {
      forked.append(history)?;
    }
    Ok(forked)
  }

  /// Record that the conversation carries on against another model.
  ///
  /// Nothing already said is touched: what the file is for is saying what
  /// happened, and a model chosen mid-session is something that happened.
  pub fn set_model(&mut self, model: &str) -> Result<()> {
    if self.model == model {
      return Ok(());
    }
    self.model = model.to_string();
    // Before the first record there is no file, and no header to disagree
    // with: the one written when the file opens says this model already.
    match self.file.is_some() {
      true => self.write_all(&[Record::Model {
        model: model.to_string(),
      }]),
      false => Ok(()),
    }
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
        parent: self.parent.clone(),
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

  /// The ids of the branch the session is on, oldest first.
  fn ids(session: &Session) -> Vec<String> {
    session
      .lineage(session.leaf(), false)
      .iter()
      .map(|s| s.to_string())
      .collect()
  }

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
    // The conversation is the checkpoint, the two turns it kept, and "third".
    assert_eq!(info.message_count, 4);
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
  fn deleting_takes_a_session_off_the_listing_and_only_this_store_s_files() {
    let store = temp_store("delete");
    let mut session = Session::new(Some(&store), Path::new("/work"), "mock");
    session.append(vec![Message::user("first")]).unwrap();
    let path = session.path().unwrap().to_path_buf();
    let mut other = Session::new(Some(&store), Path::new("/work"), "mock");
    other.append(vec![Message::user("second")]).unwrap();
    assert_eq!(store.list().len(), 2);

    // A file of somebody else's is not this store's to remove.
    let elsewhere = std::env::temp_dir().join(format!("fa-not-a-session-{}.jsonl", std::process::id()));
    std::fs::write(&elsewhere, "{}\n").unwrap();
    assert!(store.delete(&elsewhere).is_err());
    assert!(elsewhere.is_file());
    std::fs::remove_file(&elsewhere).unwrap();

    store.delete(&path).unwrap();
    assert!(!path.exists());
    let left = store.list();
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].id, other.id);
    std::fs::remove_dir_all(&store.dir).unwrap();
  }

  #[test]
  fn going_back_branches_rather_than_deletes_and_the_old_path_stays_reachable() {
    let store = temp_store("tree");
    let mut session = Session::new(Some(&store), Path::new("/work"), "mock");
    session
      .append(vec![Message::user("first"), Message::assistant("one")])
      .unwrap();
    let path = session.path().unwrap().to_path_buf();
    let prompt = ids(&session)[0].clone();
    let answered = session.leaf().unwrap().to_string();

    // Back to just after the prompt, then a different answer under it.
    session.go_to(Some(prompt)).unwrap();
    assert_eq!(session.history, [Message::user("first")]);
    session.append(vec![Message::assistant("two")]).unwrap();
    assert_eq!(session.history, [Message::user("first"), Message::assistant("two")]);

    // The answer we walked away from was not deleted, and is still a place
    // the conversation can go — which is the whole point of a tree.
    session.go_to(Some(answered.clone())).unwrap();
    assert_eq!(session.history, [Message::user("first"), Message::assistant("one")]);
    assert_eq!(session.nodes().len(), 3, "both answers, and the prompt they share");

    // All of which survives a reload, leaf and branches alike.
    let loaded = Session::load(&path).unwrap();
    assert_eq!(loaded.leaf(), Some(answered.as_str()));
    assert_eq!(loaded.history, session.history);
    assert_eq!(loaded.nodes().len(), 3);

    // Going where the session already is writes nothing.
    let before = std::fs::read_to_string(&path).unwrap();
    session.go_to(Some(answered)).unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);

    // And going back before the first entry empties the conversation
    // without losing it.
    session.go_to(None).unwrap();
    assert!(session.history.is_empty());
    assert_eq!(Session::load(&path).unwrap().nodes().len(), 3);
    std::fs::remove_dir_all(&store.dir).unwrap();
  }

  #[test]
  fn deleting_a_branch_takes_it_out_of_the_file_and_leaves_the_rest() {
    let store = temp_store("delete-branch");
    let mut session = Session::new(Some(&store), Path::new("/work"), "mock");
    session.append(vec![Message::user("first")]).unwrap();
    let prompt = session.leaf().unwrap().to_string();
    session.append(vec![Message::assistant("one")]).unwrap();
    let left = session.leaf().unwrap().to_string();
    session.go_to(Some(prompt.clone())).unwrap();
    session.append(vec![Message::assistant("two")]).unwrap();
    let path = session.path().unwrap().to_path_buf();

    // The path the session is on is not something to remove.
    assert!(session.delete_branch(&prompt).is_err());
    assert_eq!(session.nodes().len(), 3);

    // The one it left is, and the file no longer holds it — the move to it
    // included.
    assert_eq!(session.delete_branch(&left).unwrap(), 1);
    assert_eq!(session.nodes().len(), 2);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(!text.contains("\"one\""), "{text}");
    let loaded = Session::load(&path).unwrap();
    assert_eq!(loaded.nodes().len(), 2);
    assert_eq!(loaded.leaf(), session.leaf());
    assert_eq!(loaded.history, [Message::user("first"), Message::assistant("two")]);

    // And the file is still the one being appended to.
    session.append(vec![Message::user("again")]).unwrap();
    assert_eq!(Session::load(&path).unwrap().nodes().len(), 3);
    std::fs::remove_dir_all(&store.dir).unwrap();
  }

  #[test]
  fn deleting_a_tool_result_takes_the_call_it_answers_along() {
    let mut session = Session::new(None, Path::new("/work"), "mock");
    session.append(vec![Message::user("look")]).unwrap();
    let prompt = session.leaf().unwrap().to_string();
    let call = Message::Assistant {
      id: None,
      content: vec![AssistantContent::tool_call("1", "read", serde_json::json!({}))],
    };
    session.append(vec![call, tool_result("1", "fn main() {}")]).unwrap();
    let result = session.leaf().unwrap().to_string();
    session.go_to(Some(prompt)).unwrap();
    session.append(vec![Message::assistant("no need")]).unwrap();

    // With its result gone the call would answer nothing, so it goes too.
    assert_eq!(session.delete_branch(&result).unwrap(), 2);
    assert_eq!(session.nodes().len(), 2);
  }

  #[test]
  fn a_compaction_keeps_what_it_kept_addressable() {
    let store = temp_store("compaction-tree");
    let mut session = Session::new(Some(&store), Path::new("/work"), "mock");
    session
      .append(vec![Message::user("old"), Message::assistant("older")])
      .unwrap();
    let compacted = vec![
      compaction::summary_message("summary"),
      Message::user("second"),
      Message::assistant("reply"),
    ];
    session.compacted(compacted.clone(), "summary").unwrap();
    assert_eq!(session.history, compacted);

    // The checkpoint stands for what came before it, and the turns it kept
    // are entries of their own rather than a lump inside it.
    let branch = ids(&session);
    assert_eq!(branch.len(), 3);
    session.go_to(Some(branch[1].clone())).unwrap();
    assert_eq!(session.history, compacted[..2]);

    let path = session.path().unwrap().to_path_buf();
    assert_eq!(Session::load(&path).unwrap().history, compacted[..2]);
    std::fs::remove_dir_all(&store.dir).unwrap();
  }

  #[test]
  fn the_transcript_keeps_what_the_checkpoint_stands_for() {
    let store = temp_store("transcript");
    let mut session = Session::new(Some(&store), Path::new("/work"), "mock");
    let old = vec![Message::user("old"), Message::assistant("older")];
    session.append(old.clone()).unwrap();
    let kept = vec![compaction::summary_message("summary"), Message::user("second")];
    session.compacted(kept.clone(), "summary").unwrap();

    // The model is given the summary in place of the turns above it. The
    // transcript is given both: they were on screen when the compaction
    // happened, and reopening the session should not lose them.
    assert_eq!(session.branch(session.leaf()), kept);
    assert_eq!(session.branch_len(session.leaf()), 2);
    let whole: Vec<Message> = old.into_iter().chain(kept).collect();
    assert_eq!(session.transcript(session.leaf()), whole);
    assert_eq!(session.transcript_len(session.leaf()), 4);

    // Which is what the file is for: it says as much when reopened.
    let path = session.path().unwrap().to_path_buf();
    let loaded = Session::load(&path).unwrap();
    assert_eq!(loaded.history, session.history);
    assert_eq!(loaded.transcript(loaded.leaf()), whole);
    std::fs::remove_dir_all(&store.dir).unwrap();
  }

  /// A tool result message, as a turn that ran `bash` would leave behind.
  fn tool_result(call: &str, output: &str) -> Message {
    use rig_core::message::ToolResultContent;
    Message::User {
      content: vec![UserContent::tool_result(
        call,
        "bash",
        vec![ToolResultContent::text(output)],
      )],
    }
  }

  #[test]
  fn how_a_tool_result_went_survives_a_reload() {
    let store = temp_store("outcomes");
    let mut session = Session::new(Some(&store), Path::new("/work"), "mock");
    let messages = vec![
      tool_result("a", "all good"),
      tool_result("b", "Command exited with code 1"),
      tool_result("c", "Successfully replaced 1 block(s)."),
    ];
    // None of that is in a transcript: a failed result looks like any other,
    // and the diff an edit produced is not there at all.
    let outcomes = HashMap::from([
      (
        "b".to_string(),
        Outcome {
          call: "b".into(),
          failed: true,
          diff: None,
        },
      ),
      (
        "c".to_string(),
        Outcome {
          call: "c".into(),
          failed: false,
          diff: Some(" 1 before\n-2 old\n+2 new".into()),
        },
      ),
    ]);
    session.append_with(messages, &outcomes).unwrap();

    let path = session.path().unwrap().to_path_buf();
    for (which, session) in [("now", &session), ("reopened", &Session::load(&path).unwrap())] {
      assert!(session.outcome("a").is_none(), "{which}: nothing to say about a");
      assert!(session.outcome("b").unwrap().failed, "{which}: b failed");
      assert!(!session.outcome("c").unwrap().failed, "{which}: c did not");
      assert_eq!(
        session.outcome("c").unwrap().diff.as_deref(),
        Some(" 1 before\n-2 old\n+2 new"),
        "{which}: c keeps its diff"
      );
    }

    // A fork draws the conversation it copied the same way.
    let forked = session.fork(session.leaf()).unwrap();
    assert!(forked.outcome("b").unwrap().failed);
    assert!(forked.outcome("c").unwrap().diff.is_some());
    std::fs::remove_dir_all(&store.dir).unwrap();
  }

  #[test]
  fn a_fork_takes_the_one_path_and_not_the_branches_beside_it() {
    let store = temp_store("fork-path");
    let mut session = Session::new(Some(&store), Path::new("/work"), "mock");
    session
      .append(vec![Message::user("first"), Message::assistant("one")])
      .unwrap();
    let prompt = ids(&session)[0].clone();
    // A second answer under the same prompt, so the tree forks in two.
    session.go_to(Some(prompt)).unwrap();
    session.append(vec![Message::assistant("two")]).unwrap();
    assert_eq!(session.nodes().len(), 3);

    // The fork is the conversation as it reads from here — one path down the
    // tree. The answer on the branch beside it is not part of that
    // conversation, so it does not come along.
    let forked = session.fork(session.leaf()).unwrap();
    assert_eq!(forked.history, [Message::user("first"), Message::assistant("two")]);
    assert_eq!(forked.nodes().len(), 2, "the path, and nothing beside it");
    // And it starts life as a straight line, with no branch to go back to.
    assert_eq!(forked.lineage(forked.leaf(), false).len(), 2);
    std::fs::remove_dir_all(&store.dir).unwrap();
  }

  #[test]
  fn a_fork_carries_the_front_of_the_history_and_leaves_the_original_alone() {
    let store = temp_store("fork");
    let mut session = Session::new(Some(&store), Path::new("/work"), "mock");
    let history = vec![
      Message::user("first"),
      Message::assistant("answer"),
      Message::user("second"),
      Message::assistant("reply"),
    ];
    session.append(history.clone()).unwrap();
    let path = session.path().unwrap().to_path_buf();
    let before = std::fs::read_to_string(&path).unwrap();

    let forked = session.fork(Some(&ids(&session)[1])).unwrap();
    assert_eq!(forked.history, history[..2]);
    assert_ne!(forked.id, session.id);
    // The branch it came from is untouched, and still says everything it did.
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    assert_eq!(session.history, history);

    // The fork stands on its own, and names where it came from.
    let forked_path = forked.path().unwrap();
    assert_ne!(forked_path, path);
    assert_eq!(Session::load(forked_path).unwrap().history, history[..2]);
    assert_eq!(forked.parent.as_deref(), Some(path.to_str().unwrap()));
    assert_eq!(Session::load(forked_path).unwrap().parent, forked.parent);

    // Forking from the very start is an empty session, with no file yet.
    let empty = session.fork(None).unwrap();
    assert!(empty.history.is_empty());
    assert!(empty.path().is_none());
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

  #[test]
  fn a_prompt_that_attached_an_image_hands_back_only_what_was_typed() {
    // The shape a prompt with an attachment travels in: the sentence, then
    // a note and the image it labels.
    let message = Message::User {
      content: vec![
        UserContent::text("what is @shot.png"),
        UserContent::text("[Attached @shot.png — image/png]"),
        UserContent::image_base64("aGk=".to_string(), None, None),
      ],
    };
    // The note is the loop's, not the user's: `/tree` and `Up` hand back the
    // sentence alone, whose `@token` attaches the image again when it is
    // sent again.
    assert_eq!(user_text(&message).as_deref(), Some("what is @shot.png"));
  }

  #[test]
  fn a_prompt_of_several_text_parts_and_no_images_is_still_joined() {
    let message = Message::User {
      content: vec![UserContent::text("first"), UserContent::text("second")],
    };
    assert_eq!(user_text(&message).as_deref(), Some("first\nsecond"));
  }
}
