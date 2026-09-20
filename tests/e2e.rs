//! End-to-end tests: the real binary, driven through a real terminal.
//!
//! A unit test can say what a function returns; it cannot say what the
//! transcript looked like, which is where most of this program's behaviour
//! lives. So these run `fa` under tmux against a mock of an OpenAI-compatible
//! server, type at it, and read the screen back.
//!
//! tmux does the terminal emulation, so what `capture-pane` returns is what a
//! person would have seen — wrapping, overwriting and all. Without tmux
//! installed the tests skip rather than fail, like the mock-server test in
//! `src/agent.rs`.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ------------------------------------------------------------ mock provider

/// What the model does on one turn.
#[derive(Clone)]
enum Turn {
  /// Say this, then call this tool with these arguments.
  Call {
    say: &'static str,
    tool: &'static str,
    args: serde_json::Value,
  },
  /// Say this and stop.
  Say(&'static str),
}

/// A scripted OpenAI-compatible server.
///
/// Which turn it is on is worked out from the request rather than counted, so
/// a resumed session picks up where the last one left off: a request whose
/// history already holds *n* tool results is the *n*th turn.
struct Provider {
  port: u16,
  /// Every request body, for asserting on what fa told the model.
  seen: Arc<Mutex<Vec<String>>>,
}

impl Provider {
  fn start(script: Vec<Turn>) -> Self {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a port to listen on");
    let port = listener.local_addr().expect("an address").port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let provider = Self {
      port,
      seen: seen.clone(),
    };
    std::thread::spawn(move || {
      for stream in listener.incoming().flatten() {
        let (script, seen) = (script.clone(), seen.clone());
        std::thread::spawn(move || {
          let _ = serve(stream, &script, &seen);
        });
      }
    });
    provider
  }

  fn base_url(&self) -> String {
    format!("http://127.0.0.1:{}/v1", self.port)
  }

  /// Whether any request carried `needle` — what fa told the model.
  fn sent(&self, needle: &str) -> bool {
    self.seen.lock().expect("lock").iter().any(|body| body.contains(needle))
  }
}

fn serve(mut stream: TcpStream, script: &[Turn], seen: &Mutex<Vec<String>>) -> std::io::Result<()> {
  let mut reader = BufReader::new(stream.try_clone()?);
  let mut length = 0;
  loop {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 || line == "\r\n" {
      break;
    }
    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
      length = value.trim().parse().unwrap_or(0);
    }
  }
  let mut body = vec![0; length];
  std::io::Read::read_exact(&mut reader, &mut body)?;
  let body = String::from_utf8_lossy(&body).into_owned();
  seen.lock().expect("lock").push(body.clone());

  // The turn to play is the number of answers the conversation already holds.
  let done = body.matches("\"tool_call_id\"").count();
  let turn = script.get(done).cloned().unwrap_or(Turn::Say("Nothing left to do."));

  stream.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")?;
  let mut chunk = |delta: serde_json::Value, finish: Option<&str>| -> std::io::Result<()> {
    let payload = serde_json::json!({
      "id": "1", "object": "chat.completion.chunk", "created": 0, "model": "mock",
      "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    });
    stream.write_all(format!("data: {payload}\n\n").as_bytes())?;
    stream.flush()
  };
  match turn {
    Turn::Say(text) => {
      chunk(serde_json::json!({ "role": "assistant", "content": text }), None)?;
      chunk(serde_json::json!({}), Some("stop"))?;
    }
    Turn::Call { say, tool, args } => {
      if !say.is_empty() {
        chunk(serde_json::json!({ "role": "assistant", "content": say }), None)?;
      }
      let args = args.to_string();
      chunk(
        serde_json::json!({
          "tool_calls": [{ "index": 0, "id": format!("call_{done}"), "type": "function",
                           "function": { "name": tool, "arguments": "" } }]
        }),
        None,
      )?;
      // A few characters at a time, as a provider streams them — which is
      // what the transcript draws as the call being written.
      for part in args.as_bytes().chunks(8) {
        chunk(
          serde_json::json!({
            "tool_calls": [{ "index": 0, "function": { "arguments": String::from_utf8_lossy(part) } }]
          }),
          None,
        )?;
        std::thread::sleep(Duration::from_millis(15));
      }
      chunk(serde_json::json!({}), Some("tool_calls"))?;
    }
  }
  stream.write_all(b"data: [DONE]\n\n")?;
  stream.flush()
}

// ------------------------------------------------------------ terminal

/// A running `fa`, in a terminal of its own.
struct Term {
  name: String,
  dir: PathBuf,
}

impl Term {
  /// Start `fa` against `provider`, in a working directory of its own.
  fn start(test: &str, provider: &Provider, args: &[&str]) -> Self {
    let dir = std::env::temp_dir().join(format!("fa-e2e-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    Self::open(dir, provider, args)
  }

  /// Close this one and open another on the same working directory, which is
  /// how a test reopens the session it just wrote.
  fn reopen(self, provider: &Provider, args: &[&str]) -> Self {
    let dir = self.dir.clone();
    self.kill();
    std::mem::forget(self);
    Self::open(dir, provider, args)
  }

  fn open(dir: PathBuf, provider: &Provider, args: &[&str]) -> Self {
    std::fs::create_dir_all(dir.join("sessions")).expect("a working directory");
    let name = format!("fa-e2e-{}", uuid_ish());
    let _ = Command::new("tmux").args(["kill-session", "-t", &name]).output();

    let command = format!(
      "cd {} && FA_SESSIONS_DIR={} {} --base-url {} -m mock {}",
      shell(&dir),
      shell(&dir.join("sessions")),
      shell(Path::new(env!("CARGO_BIN_EXE_fa"))),
      provider.base_url(),
      args.join(" "),
    );
    // A fixed size, so what wraps where does not depend on the terminal the
    // suite happens to run in. `-f /dev/null` keeps a developer's own tmux
    // configuration out of it.
    let started = Command::new("tmux")
      .args([
        "-f",
        "/dev/null",
        "new-session",
        "-d",
        "-s",
        &name,
        "-x",
        "80",
        "-y",
        "30",
        &command,
      ])
      .status()
      .expect("tmux starts");
    assert!(started.success(), "tmux could not start a session");
    let term = Self { name, dir };
    // The input box is the last thing drawn, so its border means fa is up.
    term.wait_for("╭");
    term
  }

  fn type_in(&self, keys: &str) {
    let sent = Command::new("tmux")
      .args(["send-keys", "-t", &self.name, keys])
      .status()
      .expect("tmux sends keys");
    assert!(sent.success(), "tmux could not send keys");
  }

  /// Type a line and submit it. The command popup takes the first `Enter`
  /// when the line is a slash command, so it is given two.
  fn submit(&self, line: &str) {
    self.type_in(line);
    self.type_in("Enter");
    if line.starts_with('/') {
      self.type_in("Enter");
    }
  }

  /// The screen as a person would see it.
  fn screen(&self) -> String {
    self.capture(&[])
  }

  /// The screen with its colours, for asserting on what is red or green.
  fn coloured(&self) -> String {
    self.capture(&["-e"])
  }

  fn capture(&self, extra: &[&str]) -> String {
    let out = Command::new("tmux")
      .args(["capture-pane", "-p"])
      .args(extra)
      .args(["-t", &self.name])
      .output()
      .expect("tmux captures the pane");
    String::from_utf8_lossy(&out.stdout).into_owned()
  }

  /// Wait until the screen says `needle`, or give up and show what it said
  /// instead — a timeout with no screen in it is a test that teaches nothing.
  fn wait_for(&self, needle: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut screen = String::new();
    while Instant::now() < deadline {
      screen = self.screen();
      if screen.contains(needle) {
        return screen;
      }
      std::thread::sleep(Duration::from_millis(50));
    }
    panic!("waited for {needle:?}, screen was:\n{screen}");
  }

  /// Wait for a line to settle before reading it, for assertions about what
  /// is *not* there yet.
  fn settle(&self) {
    std::thread::sleep(Duration::from_millis(400));
  }

  fn session_file(&self) -> String {
    let dir = self.dir.join("sessions");
    let file = std::fs::read_dir(&dir)
      .expect("a sessions directory")
      .flatten()
      .map(|entry| entry.path())
      .find(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
      .unwrap_or_else(|| panic!("a session file in {}", dir.display()));
    std::fs::read_to_string(file).expect("a readable session")
  }
}

impl Term {
  fn kill(&self) {
    let _ = Command::new("tmux").args(["kill-session", "-t", &self.name]).output();
  }
}

impl Drop for Term {
  fn drop(&mut self) {
    self.kill();
    let _ = std::fs::remove_dir_all(&self.dir);
  }
}

/// A name no other session in this run will have.
fn uuid_ish() -> String {
  static NEXT: AtomicU32 = AtomicU32::new(0);
  format!("{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst))
}

fn shell(path: &Path) -> String {
  format!("'{}'", path.display().to_string().replace('\'', r"'\''"))
}

/// Whether this machine can run these at all.
fn have_tmux() -> bool {
  let installed = Command::new("tmux").arg("-V").output().is_ok();
  if !installed {
    eprintln!("tmux not installed; skipping");
  }
  installed
}

// ------------------------------------------------------------ tests

#[test]
fn a_command_is_written_then_run_and_its_output_lands_under_it() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "Let me look.",
      tool: "bash",
      args: serde_json::json!({ "command": "echo written-and-run" }),
    },
    Turn::Say("That is all."),
  ]);
  let term = Term::start("written", &provider, &["--no-session"]);
  term.submit("do it");

  // The call is drawn as the model writes it, with a cursor at the end,
  // before it has run at all.
  let writing = term.wait_for("▌");
  let pending = writing
    .lines()
    .find(|l| l.contains('▌'))
    .expect("the line being written");
  assert!(
    pending.contains("⚙ bash"),
    "the call is what is being written: {pending:?}"
  );

  // Its output only lands once it has run, and lands under it.
  let screen = term.wait_for("That is all.");
  let lines: Vec<&str> = screen.lines().map(str::trim_end).filter(|l| !l.is_empty()).collect();
  let call = lines.iter().position(|l| l.contains("⚙ bash")).expect("the call");
  assert!(
    lines[call].contains("echo written-and-run"),
    "the command is on its own line: {lines:?}"
  );
  // Its output is under it, not somewhere else in the transcript.
  assert!(
    lines[call + 1].contains("written-and-run"),
    "output under its call: {lines:?}"
  );
}

/// How the terminal's own green and red arrive as a background: indexed, so
/// they follow whatever palette the terminal is wearing.
const GREEN_BG: &str = "\u{1b}[48;5;2m";
const RED_BG: &str = "\u{1b}[48;5;1m";

#[test]
fn a_failed_command_is_red_and_a_finished_one_green() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "bash",
      args: serde_json::json!({ "command": "echo fine" }),
    },
    Turn::Call {
      say: "",
      tool: "bash",
      args: serde_json::json!({ "command": "exit 3" }),
    },
    Turn::Say("Both tried."),
  ]);
  let term = Term::start("colours", &provider, &["--no-session"]);
  term.submit("run both");
  term.wait_for("Both tried.");

  // The stripe is beside the output, not on the command's own line, and it
  // is a background rather than recoloured text.
  let screen = term.coloured();
  let striped = |bg: &str, text: &str| screen.lines().any(|line| line.contains(bg) && line.contains(text));
  assert!(striped(GREEN_BG, "fine"), "green beside what worked:\n{screen:?}");
  assert!(
    striped(RED_BG, "Command exited with code 3"),
    "red beside what did not:\n{screen:?}"
  );
}

#[test]
fn an_aborted_run_keeps_its_work_and_can_carry_on() {
  if !have_tmux() {
    return;
  }
  // The third call never returns, so there is something to abort in the
  // middle of — with two turns of real work already behind it.
  let provider = Provider::start(vec![
    Turn::Call {
      say: "First. ",
      tool: "bash",
      args: serde_json::json!({ "command": "echo one" }),
    },
    Turn::Call {
      say: "Second. ",
      tool: "bash",
      args: serde_json::json!({ "command": "echo two" }),
    },
    Turn::Call {
      say: "Third. ",
      tool: "bash",
      args: serde_json::json!({ "command": "sleep 60" }),
    },
    Turn::Say("Carried on."),
  ]);
  let term = Term::start("abort", &provider, &[]);
  term.submit("do three things");
  term.wait_for("⚙ bash sleep 60");
  term.type_in("Escape");
  term.wait_for("Aborted.");

  // A run only hands back its transcript when it finishes, so this is the
  // work that would otherwise be lost with it.
  let session = term.session_file();
  for kept in ["echo one", "echo two", "sleep 60"] {
    assert!(session.contains(kept), "{kept:?} kept in the session");
  }
  assert!(
    session.contains("Aborted by the user."),
    "the interrupted call is answered, or the history cannot be sent again"
  );

  // And the conversation it left behind is one the model can be given back.
  let term = term.reopen(&provider, &["-c"]);
  term.wait_for("Resumed session");
  term.submit("/continue");
  term.wait_for("Carried on.");
  assert!(
    provider.sent("Aborted by the user."),
    "the interrupted call went back with its answer"
  );
}

#[test]
fn what_a_session_shows_is_what_it_showed_before_it_was_closed() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "bash",
      args: serde_json::json!({ "command": "echo kept" }),
    },
    Turn::Call {
      say: "",
      tool: "bash",
      args: serde_json::json!({ "command": "exit 4" }),
    },
    Turn::Say("Done."),
  ]);
  let term = Term::start("reload", &provider, &[]);
  term.submit("go");
  term.wait_for("Done.");
  term.settle();
  let before = term.coloured();

  // Reopened, the same conversation has to read the same way — which takes
  // the session remembering what a transcript does not say, such as which
  // command failed.
  let term = term.reopen(&provider, &["-c"]);
  term.wait_for("Resumed session");
  term.settle();
  let after = term.coloured();

  // Everything above the input box, minus the two things a reopened session
  // has no way to know: that it was reopened, and how long a command took —
  // which is measured while it runs and is not part of the transcript.
  let body = |screen: &str| {
    screen
      .lines()
      .take_while(|line| !line.contains('╭'))
      .map(str::trim_end)
      .filter(|line| !line.is_empty())
      .filter(|line| !line.contains("Resumed session") && !line.contains("Took "))
      .map(str::to_string)
      .collect::<Vec<_>>()
  };
  assert_eq!(body(&before), body(&after), "\nbefore:\n{before}\nafter:\n{after}");
}
