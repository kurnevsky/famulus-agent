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
  /// Think aloud first, then say this — what a local reasoning server sends.
  Think { thought: &'static str, say: &'static str },
  /// Think aloud, then call a tool: a reasoning model working, rather than
  /// answering. What the turn thought is part of it, and a conversation that
  /// carries on past it has to carry that too.
  ThinkCall {
    thought: &'static str,
    say: &'static str,
    tool: &'static str,
    args: serde_json::Value,
  },
  /// Answer with the question, so the two ways a conversation went can be
  /// told apart on screen by what was asked down each.
  Echo,
}

/// A phrase from the summarizer's own preamble, which is how a request for a
/// summary is told from a turn of the conversation.
const SUMMARIZING: &str = "context summarization assistant";

/// A phrase from the prompt the beginning of a split turn is asked for with,
/// which is how the second summarization request is told from the first.
const SUMMARIZING_TURN: &str = "This is the PREFIX of a turn";

/// What the mock always summarizes a conversation into. Distinctive enough to
/// find again in a later request, and it says nothing the conversation said.
const SUMMARY: &str = "## Goal\\nFruit was discussed.";

/// And what it summarizes the beginning of a split turn into.
const TURN_SUMMARY: &str = "## Original Request\\nSomething about fruit.";

/// A scripted OpenAI-compatible server.
///
/// Which turn it is on is worked out from the request rather than counted, so
/// a resumed session picks up where the last one left off: a request whose
/// history already holds *n* tool results is the *n*th turn.
struct Provider {
  port: u16,
  /// Every request body, for asserting on what fa told the model.
  seen: Arc<Mutex<Vec<String>>>,
  /// How many times fa has asked what models there are, for asserting on
  /// what it asked for without being asked to.
  listings: Arc<AtomicU32>,
}

impl Provider {
  fn start(script: Vec<Turn>) -> Self {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a port to listen on");
    let port = listener.local_addr().expect("an address").port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let listings = Arc::new(AtomicU32::new(0));
    let provider = Self {
      port,
      seen: seen.clone(),
      listings: listings.clone(),
    };
    std::thread::spawn(move || {
      for stream in listener.incoming().flatten() {
        let (script, seen, listings) = (script.clone(), seen.clone(), listings.clone());
        std::thread::spawn(move || {
          let _ = serve(stream, &script, &seen, &listings);
        });
      }
    });
    provider
  }

  fn base_url(&self) -> String {
    format!("http://127.0.0.1:{}/v1", self.port)
  }

  /// How many times it was asked for its models.
  fn listings(&self) -> u32 {
    self.listings.load(Ordering::SeqCst)
  }

  /// Whether any request carried `needle` — what fa told the model.
  fn sent(&self, needle: &str) -> bool {
    self.seen.lock().expect("lock").iter().any(|body| body.contains(needle))
  }

  /// Every request body, in the order they arrived.
  fn bodies(&self) -> Vec<String> {
    self.seen.lock().expect("lock").clone()
  }

  /// The first request that carried `needle`, for asking what else was in it
  /// — which is how "when was this sent" is answered.
  fn request(&self, needle: &str) -> String {
    let seen = self.seen.lock().expect("lock");
    seen
      .iter()
      .find(|body| body.contains(needle))
      .unwrap_or_else(|| panic!("no request carried {needle:?}; {} were sent", seen.len()))
      .clone()
  }
}

fn serve(
  mut stream: TcpStream,
  script: &[Turn],
  seen: &Mutex<Vec<String>>,
  listings: &AtomicU32,
) -> std::io::Result<()> {
  let mut reader = BufReader::new(stream.try_clone()?);
  let mut length = 0;
  let mut request_line = String::new();
  reader.read_line(&mut request_line)?;
  loop {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 || line == "\r\n" {
      break;
    }
    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
      length = value.trim().parse().unwrap_or(0);
    }
  }
  // What `/model` asks for, and nothing else does: fa comes up on the model
  // the command line named and leaves the provider alone until the picker is
  // opened. Not a turn of the conversation and not recorded as one — what the
  // tests assert on is what fa told the model.
  if request_line
    .split_whitespace()
    .nth(1)
    .is_some_and(|path| path.ends_with("/models"))
  {
    listings.fetch_add(1, Ordering::SeqCst);
    let body = serde_json::json!({
      "object": "list",
      "data": [
        { "id": "mock", "object": "model", "owned_by": "fa-tests" },
        { "id": "mock-mini", "object": "model", "owned_by": "fa-tests" },
        { "id": "other-model", "object": "model", "owned_by": "fa-tests" },
      ],
    })
    .to_string();
    return stream.write_all(
      format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
      )
      .as_bytes(),
    );
  }
  let mut body = vec![0; length];
  std::io::Read::read_exact(&mut reader, &mut body)?;
  let body = String::from_utf8_lossy(&body).into_owned();
  seen.lock().expect("lock").push(body.clone());

  // The turn to play is the number of answers the conversation already holds.
  // A request for a summary is not a turn of the conversation at all: it is
  // the summarizer, asking with a preamble of its own.
  let done = body.matches("\"tool_call_id\"").count();
  let turn = if body.contains(SUMMARIZING_TURN) {
    // Both summarizations speak with the summarizer's preamble, so the one
    // for a split turn is known by what it asks for, and answered with
    // something the other would never say.
    Turn::Say(TURN_SUMMARY)
  } else if body.contains(SUMMARIZING) {
    Turn::Say(SUMMARY)
  } else {
    script.get(done).cloned().unwrap_or(Turn::Say("Nothing left to do."))
  };

  // Only the conversation is streamed; the summarizer asks outright, and an
  // event stream is not an answer to that.
  if !body.contains("\"stream\":true") {
    let text = match turn {
      Turn::Say(text) => text.to_string(),
      Turn::Think { say, .. } => say.to_string(),
      Turn::Echo => answer_to(&body),
      Turn::Call { say, .. } | Turn::ThinkCall { say, .. } => say.to_string(),
    };
    return answer_outright(&mut stream, &text);
  }

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
    Turn::Think { thought, say } => {
      chunk(
        serde_json::json!({ "role": "assistant", "reasoning_content": thought }),
        None,
      )?;
      chunk(serde_json::json!({ "content": say }), None)?;
      chunk(serde_json::json!({}), Some("stop"))?;
    }
    Turn::Say(_) | Turn::Echo => {
      let text = match turn {
        Turn::Say(text) => text.to_string(),
        _ => answer_to(&body),
      };
      chunk(serde_json::json!({ "role": "assistant", "content": text }), None)?;
      chunk(serde_json::json!({}), Some("stop"))?;
    }
    Turn::Call { say, tool, args } => {
      call(&mut chunk, None, say, tool, &args, done)?;
    }
    Turn::ThinkCall {
      thought,
      say,
      tool,
      args,
    } => {
      call(&mut chunk, Some(thought), say, tool, &args, done)?;
    }
  }
  // What the turn cost, in the usage-only chunk a provider sends last when
  // the request asked for one — which rig's does. Every turn answers the
  // same for what it wrote, so a run of two says twice as much as a run of
  // one, which is how a test can tell when it was counted.
  let prompt_tokens = conversation_tokens(&body);
  let payload = serde_json::json!({
    "id": "1", "object": "chat.completion.chunk", "created": 0, "model": "mock",
    "choices": [],
    "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": SPENT,
               "total_tokens": prompt_tokens + SPENT },
  });
  stream.write_all(format!("data: {payload}\n\n").as_bytes())?;
  stream.write_all(b"data: [DONE]\n\n")?;
  stream.flush()
}

/// Stream a turn that calls a tool, with whatever it thought on the way there.
fn call(
  chunk: &mut impl FnMut(serde_json::Value, Option<&str>) -> std::io::Result<()>,
  thought: Option<&str>,
  say: &str,
  tool: &str,
  args: &serde_json::Value,
  done: usize,
) -> std::io::Result<()> {
  if let Some(thought) = thought {
    chunk(
      serde_json::json!({ "role": "assistant", "reasoning_content": thought }),
      None,
    )?;
  }
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
  // A few characters at a time, as a provider streams them — which is what
  // the transcript draws as the call being written.
  for part in args.as_bytes().chunks(8) {
    chunk(
      serde_json::json!({
        "tool_calls": [{ "index": 0, "function": { "arguments": String::from_utf8_lossy(part) } }]
      }),
      None,
    )?;
    std::thread::sleep(Duration::from_millis(15));
  }
  chunk(serde_json::json!({}), Some("tool_calls"))
}

/// What the mock says every turn costs to write.
const SPENT: u64 = 7;

/// How big the conversation in a request is, in the mock's own tokens: a
/// quarter of the JSON its messages take, leaving out the system prompt —
/// which is the same size whatever has been said, and is not something
/// compacting can bring down.
fn conversation_tokens(body: &str) -> u64 {
  let body: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
  body["messages"]
    .as_array()
    .into_iter()
    .flatten()
    .filter(|message| message["role"] != "system")
    .map(|message| message.to_string().len() as u64 / 4)
    .sum()
}

/// One whole completion, for the requests that are not streamed.
fn answer_outright(stream: &mut TcpStream, text: &str) -> std::io::Result<()> {
  let payload = serde_json::json!({
    "id": "1", "object": "chat.completion", "created": 0, "model": "mock",
    "choices": [{ "index": 0, "finish_reason": "stop",
                  "message": { "role": "assistant", "content": text } }],
    "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
  })
  .to_string();
  stream.write_all(
    format!(
      "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{payload}",
      payload.len()
    )
    .as_bytes(),
  )?;
  stream.flush()
}

/// The last thing the user said in a request, as an answer naming it.
fn answer_to(body: &str) -> String {
  let body: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
  let asked = body["messages"]
    .as_array()
    .into_iter()
    .flatten()
    .rfind(|message| message["role"] == "user")
    .map(|message| match &message["content"] {
      serde_json::Value::String(text) => text.clone(),
      // Some providers take a user message as parts rather than a string.
      content => content
        .as_array()
        .map(|parts| {
          parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join(" ")
        })
        .unwrap_or_default(),
    })
    .unwrap_or_default();
  format!("Answer to {asked}.")
}

// ------------------------------------------------------------ mcp server

/// An MCP server over stdin and stdout, in as little as it takes: the three
/// requests a client makes of one, answered by hand — and one it makes of the
/// client, when `book` needs the user to say something. `grow` changes what it
/// offers, and says so the way the protocol has it. Started with `resources`,
/// it has a note to read as well; with `prompts`, two prompts to send; with
/// `completes`, it says what their arguments and its template's hole could be.
#[cfg(feature = "mcp")]
const MCP_SERVER: &str = r#"
import json, sys

TOOLS = [{
  "name": "weather",
  "description": "What the weather is somewhere.",
  "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}, {
  "name": "forecast",
  "description": "What the weather will be.",
  "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}, {
  "name": "flood",
  "description": "More than anyone asked for.",
  "inputSchema": {"type": "object", "properties": {"lines": {"type": "integer"}}, "required": ["lines"]},
}, {
  "name": "book",
  "description": "Book seats, asking the user how many.",
  "inputSchema": {"type": "object", "properties": {}},
}, {
  "name": "grow",
  "description": "Offer other tools than these.",
  "inputSchema": {"type": "object", "properties": {}},
}, {
  "name": "jot",
  "description": "Write tomorrow's note.",
  "inputSchema": {"type": "object", "properties": {}},
}]

# What `grow` leaves it offering: itself gone, and two it did not have.
GROWN = [tool for tool in TOOLS if tool["name"] != "grow"] + [{
  "name": "radar",
  "description": "Where the rain is now.",
  "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}, {
  "name": "tide",
  "description": "When the sea comes in.",
  "inputSchema": {"type": "object", "properties": {}},
}]

FORM = {
    "mode": "form",
    "message": "How many seats, and where?",
    "requestedSchema": {
        "type": "object",
        "properties": {
            "seats": {"type": "integer", "title": "Seats", "minimum": 1},
            "window": {"type": "boolean", "title": "Window seat"},
        },
        "required": ["seats"],
    },
}

def send(message):
    sys.stdout.write(json.dumps(dict(message, jsonrpc="2.0")) + "\n")
    sys.stdout.flush()

def read():
    while True:
        line = sys.stdin.readline()
        if not line:
            sys.exit(0)
        if line.strip():
            return json.loads(line)

RESOURCES = "resources" in sys.argv[1:]
NOTE = {"uri": "note://today", "name": "today", "mimeType": "text/plain", "description": "What is\n  on today."}
DAY = {"uriTemplate": "note://{day}", "name": "day", "description": "The note for a day."}
LATER = {"uri": "note://tomorrow", "name": "tomorrow", "mimeType": "text/plain"}
NOTES = [NOTE]

PROMPTS = "prompts" in sys.argv[1:]
REVIEW = {
    "name": "review",
    "description": "Review a change.",
    "arguments": [{"name": "pr", "required": True}, {"name": "focus"}],
}
RECAP = {"name": "recap", "description": "Say where things stand."}
COMPLETES = "completes" in sys.argv[1:]

def completed(ref, argument, context):
    if ref.get("type") == "ref/resource":
        choices = ["2026-09-22", "2026-09-23"]
    elif argument["name"] == "pr":
        choices = ["12", "123", "7"]
    else:
        choices = ["tests of %s" % context.get("pr"), "docs"]
    return [c for c in choices if c.startswith(argument["value"])]

def written(name, arguments):
    if name == "review":
        focus = arguments.get("focus") or "everything"
        text = "Review PR %s, looking at %s." % (arguments.get("pr"), focus)
        return [{"role": "user", "content": {"type": "text", "text": text}}]
    return [
        {"role": "user", "content": {"type": "text", "text": "Where were we?"}},
        {"role": "assistant", "content": {"type": "text", "text": "On the parser."}},
        {"role": "user", "content": {"type": "resource", "resource": {
            "uri": "note://today", "mimeType": "text/plain", "text": "Buy milk."}}},
    ]

asks = False
while True:
    message = read()
    if "id" not in message:
        continue  # a notification: nothing to answer
    method, params = message.get("method"), message.get("params") or {}
    if method == "initialize":
        asks = "form" in (params.get("capabilities") or {}).get("elicitation", {})
        capabilities = {"tools": {"listChanged": True}}
        if RESOURCES:
            capabilities["resources"] = {}
        if PROMPTS:
            capabilities["prompts"] = {}
        if COMPLETES:
            capabilities["completions"] = {}
        result = {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
            "capabilities": capabilities,
            "serverInfo": {"name": "mock-weather", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "prompts/list":
        result = {"prompts": [REVIEW, RECAP]}
    elif method == "completion/complete":
        context = (params.get("context") or {}).get("arguments") or {}
        values = completed(params["ref"], params["argument"], context)
        result = {"completion": {"values": values, "hasMore": False}}
    elif method == "prompts/get":
        result = {"messages": written(params.get("name"), params.get("arguments") or {})}
    elif method == "resources/list":
        result = {"resources": NOTES}
    elif method == "resources/templates/list":
        result = {"resourceTemplates": [DAY]}
    elif method == "resources/read":
        if params.get("uri") != NOTE["uri"]:
            send({"id": message["id"], "error": {"code": -32002, "message": "no such note"}})
            continue
        result = {"contents": [{"uri": NOTE["uri"], "mimeType": "text/plain", "text": "Buy milk."}]}
    elif method == "tools/call":
        arguments = params.get("arguments") or {}
        if params.get("name") == "book":
            if not asks:
                answer = "this client cannot be asked anything"
            else:
                send({"id": "form", "method": "elicitation/create", "params": FORM})
                reply = read()
                while reply.get("id") != "form":
                    reply = read()
                answer = json.dumps(reply.get("result"))
        elif params.get("name") == "grow":
            TOOLS = GROWN
            send({"method": "notifications/tools/list_changed"})
            # The client asks for the new list, and has it before this call
            # comes back: what the model is asked next knows of the change.
            ask = read()
            while ask.get("method") != "tools/list":
                ask = read()
            send({"id": ask["id"], "result": {"tools": TOOLS}})
            answer = "grown"
        elif params.get("name") == "jot":
            NOTES = [NOTE, LATER]
            send({"method": "notifications/resources/list_changed"})
            answer = "jotted"
        elif params.get("name") == "flood":
            # A server under no obligation to be brief.
            answer = "\n".join("line %d" % i for i in range(1, arguments["lines"] + 1))
        else:
            city = arguments.get("city", "nowhere")
            # Structure, as one long line — which is how a server answers.
            answer = json.dumps({"city": city, "rain": True, "hours": [1, 2]})
        result = {"content": [{"type": "text", "text": answer}], "isError": False}
    else:
        result = {}
    send({"id": message["id"], "result": result})
"#;

/// A directory of this test's own, emptied first.
#[cfg(feature = "mcp")]
fn scratch(test: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("fa-e2e-{}-{test}-files", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).expect("a directory to put files in");
  dir
}

#[cfg(feature = "mcp")]
fn have_python() -> bool {
  let installed = Command::new("python3").arg("-V").output().is_ok();
  if !installed {
    eprintln!("python3 not installed; skipping");
  }
  installed
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
    // An empty configuration directory, pointed at by both halves of the XDG
    // search path: a developer with servers of their own in `mcp.toml` would
    // otherwise have them started by every test, and what the transcript says
    // is not supposed to depend on the machine the suite runs on. The test
    // that wants a server names its own file with `--mcp-config`, which is
    // not a search at all.
    let config = dir.join("config");
    std::fs::create_dir_all(&config).expect("a configuration directory");
    // The session's name is its socket's too: one server per terminal.
    let name = format!("fa-e2e-{}", uuid_ish());
    let command = format!(
      "cd {} && FA_SESSIONS_DIR={} XDG_CONFIG_HOME={} XDG_CONFIG_DIRS={} {} --base-url {} -m mock {}",
      shell(&dir),
      shell(&dir.join("sessions")),
      shell(&config),
      shell(&config),
      shell(Path::new(env!("CARGO_BIN_EXE_fa"))),
      provider.base_url(),
      args.join(" "),
    );
    // A fixed size, so what wraps where does not depend on the terminal the
    // suite happens to run in. `-f /dev/null` keeps a developer's own tmux
    // configuration out of it.
    let started = tmux(
      &name,
      &[
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
      ],
    )
    .status()
    .expect("tmux starts");
    assert!(started.success(), "tmux could not start a session");
    // Let OSC 52 reach a buffer of this server's, so a test can read back
    // what was copied. Left alone, tmux passes the escape outwards to a
    // terminal there is nobody sitting at.
    let _ = tmux(&name, &["set-option", "-s", "set-clipboard", "on"]).status();
    let term = Self { name, dir };
    // The input box is the last thing drawn, so its border means fa is up.
    term.wait_for("╭");
    term
  }

  fn type_in(&self, keys: &str) {
    let sent = tmux(&self.name, &["send-keys", "-t", &self.name, keys])
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

  /// Hand the program bytes as though the terminal had sent them, which is
  /// how a mouse report gets in: tmux types keys, and a mouse is not one.
  fn send_raw(&self, bytes: &str) {
    let sent = tmux(&self.name, &["send-keys", "-t", &self.name, "-l", bytes])
      .status()
      .expect("tmux sends bytes");
    assert!(sent.success(), "tmux could not send bytes");
  }

  /// Press the left button on a cell, drag to another, and let go — the SGR
  /// mouse reports a terminal sends for it, which is what `fa` turns mouse
  /// capture on to read. Cells are 1-based, as they are on the wire.
  fn drag(&self, from: (usize, usize), to: (usize, usize)) {
    self.drag_through(&[from, to]);
  }

  /// The same, through every cell in turn: what a drag that stops at the edge
  /// and stays there looks like on the wire.
  fn drag_through(&self, cells: &[(usize, usize)]) {
    self.hold_through(cells);
    self.let_go(cells[cells.len() - 1]);
  }

  /// Press and drag without letting go, for reading the screen mid-drag —
  /// which is the only time there is a selection on it to read.
  fn hold_through(&self, cells: &[(usize, usize)]) {
    let (col, row) = cells[0];
    self.send_raw(&format!("\x1b[<0;{col};{row}M"));
    for (col, row) in &cells[1..] {
      self.send_raw(&format!("\x1b[<32;{col};{row}M"));
    }
    self.settle();
  }

  fn let_go(&self, (col, row): (usize, usize)) {
    self.send_raw(&format!("\x1b[<0;{col};{row}m"));
    self.settle();
  }

  /// A notch of the wheel, which is a press of a button of its own.
  fn wheel_up(&self, (col, row): (usize, usize)) {
    self.send_raw(&format!("\x1b[<64;{col};{row}M"));
  }

  /// Where `needle` is on screen, as the 1-based cell its first character is
  /// drawn on.
  fn cell_of(&self, needle: &str) -> (usize, usize) {
    let screen = self.screen();
    screen
      .lines()
      .enumerate()
      .find_map(|(row, line)| {
        let at = line.find(needle)?;
        // Columns are cells, and what is before the needle may not be ascii.
        Some((line[..at].chars().count() + 1, row + 1))
      })
      .unwrap_or_else(|| panic!("{needle:?} on screen:\n{screen}"))
  }

  /// What the program has put on the terminal's clipboard, which for tmux is
  /// its most recent buffer.
  fn clipboard(&self) -> String {
    let out = tmux(&self.name, &["show-buffer"]).output().expect("tmux answers");
    String::from_utf8_lossy(&out.stdout).into_owned()
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
    let out = tmux(&self.name, &["capture-pane", "-p"])
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

  /// Wait for the box at the bottom of the screen to hold `query`, which is
  /// where a list being narrowed says what it is being narrowed by.
  fn wait_for_query(&self, query: &str) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
      if self.typed() == query {
        return;
      }
      std::thread::sleep(Duration::from_millis(50));
    }
    panic!("waited for the query {query:?}, the box held {:?}", self.typed());
  }

  /// Wait for a line to settle before reading it, for assertions about what
  /// is *not* there yet.
  fn settle(&self) {
    std::thread::sleep(Duration::from_millis(400));
  }

  /// The rows of the open overlay, in order, and which one the cursor is on.
  fn overlay(&self) -> (Vec<String>, usize) {
    let screen = self.screen();
    let lines: Vec<&str> = screen.lines().collect();
    // Every overlay says how to leave it, on its top border.
    let top = lines
      .iter()
      .position(|line| line.contains("Esc cancel"))
      .unwrap_or_else(|| panic!("an open overlay:\n{screen}"));
    let bottom = lines[top..]
      .iter()
      .position(|line| line.starts_with('╰'))
      .map_or(lines.len(), |at| top + at);
    // A toast in the corner is drawn over the rows; each row ends where its
    // frame begins, so what it says is not read as part of the list.
    let rows: Vec<String> = lines[top + 1..bottom]
      .iter()
      .map(|line| {
        let row = line.strip_prefix('│').unwrap_or(line);
        let end = row.find(['│', '╭', '╰']).unwrap_or(row.len());
        format!("│{}", &row[..end])
      })
      .collect();
    let on = rows
      .iter()
      .position(|row| row.contains('›'))
      .unwrap_or_else(|| panic!("a selected row:\n{screen}"));
    (rows, on)
  }

  /// Walk the overlay's cursor onto the row that says `needle`.
  fn point_at(&self, needle: &str) {
    for _ in 0..30 {
      let (rows, on) = self.overlay();
      let at = rows
        .iter()
        .position(|row| row.contains(needle))
        .unwrap_or_else(|| panic!("a row saying {needle:?}: {rows:?}"));
      if at == on {
        return;
      }
      self.type_in(if at < on { "Up" } else { "Down" });
      std::thread::sleep(Duration::from_millis(60));
    }
    panic!("could not put the cursor on {needle:?}");
  }

  /// Walk the overlay's cursor onto the row that says `needle` and take it.
  fn choose(&self, needle: &str) {
    self.point_at(needle);
    self.type_in("Enter");
  }

  /// Whether the terminal has been rung since anyone last looked at it. tmux
  /// keeps the flag for a window nobody is watching, which is the case a bell
  /// is rung for in the first place.
  fn rang(&self) -> bool {
    let out = tmux(
      &self.name,
      &["display-message", "-p", "-t", &self.name, "#{window_bell_flag}"],
    )
    .output()
    .expect("tmux answers");
    String::from_utf8_lossy(&out.stdout).trim() == "1"
  }

  /// What is in the input box, or nothing when it is empty.
  fn typed(&self) -> String {
    let screen = self.screen();
    let lines: Vec<&str> = screen.lines().collect();
    // The lowest box on the screen, which is the one above the footer: with a
    // list open there is one over the transcript as well, and the box being
    // typed in is the one at the bottom either way.
    let top = lines.iter().rposition(|line| line.contains('╭')).unwrap_or(0);
    lines
      .get(top + 1)
      .map(|line| line.trim_matches(|c| c == '│' || c == ' ').to_string())
      .unwrap_or_default()
  }

  /// Every session file this terminal has written, with what is in it.
  fn session_files(&self) -> Vec<(PathBuf, String)> {
    let dir = self.dir.join("sessions");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
      .unwrap_or_else(|_| panic!("a sessions directory at {}", dir.display()))
      .flatten()
      .map(|entry| entry.path())
      .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
      .collect();
    files.sort();
    files
      .into_iter()
      .map(|path| {
        let text = std::fs::read_to_string(&path).expect("a readable session");
        (path, text)
      })
      .collect()
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
    let _ = tmux(&self.name, &["kill-server"]).output();
  }
}

impl Drop for Term {
  fn drop(&mut self) {
    self.kill();
    let _ = std::fs::remove_dir_all(&self.dir);
  }
}

/// A tmux command on `socket`, which is a server of this test's own rather
/// than the one the developer is working in: a test sets server options and
/// reads back the clipboard, and neither belongs in somebody's own session —
/// or in another test's, which is why it is one server per terminal rather
/// than one for the suite. It starts with the `new-session` and ends with it.
fn tmux(socket: &str, args: &[&str]) -> Command {
  let mut command = Command::new("tmux");
  command.args(["-L", socket]).args(args);
  command
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
  // And the line it was being written on is gone: the call it became is
  // the only one left, not a second copy under everything else.
  assert!(
    !screen.contains('\u{258C}'),
    "nothing is still being written once it has run:\n{screen}"
  );
  assert_eq!(
    lines.iter().filter(|l| l.contains("\u{2699} bash")).count(),
    1,
    "the call is drawn once: {lines:?}"
  );
}

#[test]
fn a_script_of_several_lines_is_drawn_on_all_of_them() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "bash",
      args: serde_json::json!({ "command": "for word in one two; do\n  echo $word\ndone\n" }),
    },
    Turn::Say("That is all."),
  ]);
  let term = Term::start("script", &provider, &["--no-session"]);
  term.submit("do it");

  let screen = term.wait_for("That is all.");
  let lines: Vec<&str> = screen.lines().map(str::trim_end).filter(|l| !l.is_empty()).collect();
  let call = lines.iter().position(|l| l.contains("⚙ bash")).expect("the call");
  // Every line of the script, under the one before it and indented to where
  // the first one starts — and the trailing newline is not a line of it.
  assert!(
    lines[call].ends_with("⚙ bash for word in one two; do"),
    "the script starts on the call's line: {lines:?}"
  );
  assert_eq!(
    lines[call + 1..call + 3],
    ["         echo $word", "       done"],
    "the rest of the script is under it: {lines:?}"
  );
}

/// How the terminal's own green and red arrive as a background: indexed, so
/// they follow whatever palette the terminal is wearing.
const GREEN_BG: &str = "\u{1b}[48;5;2m";
const RED_BG: &str = "\u{1b}[48;5;1m";
/// Red text, as against red behind text.
const RED: &str = "\u{1b}[38;5;1m";

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
fn an_image_a_tool_read_is_drawn_where_it_was_read() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "read",
      args: serde_json::json!({ "path": "red.png" }),
    },
    Turn::Say("A red square."),
  ]);
  let term = Term::start("image", &provider, &["--no-session"]);
  let red = image::ImageBuffer::from_pixel(16, 16, image::Rgb([220u8, 20, 60]));
  image::DynamicImage::ImageRgb8(red)
    .save(term.dir.join("red.png"))
    .expect("an image to read");
  term.submit("look at red.png");
  term.wait_for("A red square.");

  // Sixteen pixels down is eight lines of half-blocks, sixteen cells across
  // — the image at its own size, since the transcript is wider than it is.
  let screen = term.screen();
  let drawn: Vec<&str> = screen.lines().filter(|line| line.contains('▄')).collect();
  assert_eq!(drawn.len(), 8, "the image is drawn as half-blocks:\n{screen}");
  assert!(
    drawn.iter().all(|line| line.matches('▄').count() == 16),
    "each line is the image's own width:\n{screen}"
  );
  // And drawn in colour, whatever the terminal rounds it to.
  let coloured = term.coloured();
  assert!(
    coloured
      .lines()
      .any(|line| line.contains('▄') && line.contains("\u{1b}[38;")),
    "the blocks carry the image's colours:\n{coloured:?}"
  );
}

#[test]
fn a_tall_image_folds_like_any_other_output_and_ctrl_o_unfolds_it() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "read",
      args: serde_json::json!({ "path": "tall.png" }),
    },
    Turn::Say("A tall one."),
  ]);
  let term = Term::start("tall-image", &provider, &["--no-session"]);
  // Twenty-four across and sixty down: thirty lines drawn, of which the
  // preview keeps sixteen.
  let tall = image::ImageBuffer::from_pixel(24, 60, image::Rgb([30u8, 90, 200]));
  image::DynamicImage::ImageRgb8(tall)
    .save(term.dir.join("tall.png"))
    .expect("an image to read");
  term.submit("look at tall.png");
  term.wait_for("A tall one.");

  let screen = term.wait_for("14 more lines (ctrl+o)");
  let folded = screen.lines().filter(|line| line.contains('▄')).count();
  assert_eq!(folded, 16, "the preview keeps the top of the image:\n{screen}");

  // And ctrl+o shows the rest of it, the same drawing rather than a bigger
  // one — so the lines already on screen are the lines still on screen.
  term.type_in("C-o");
  let unfolded = poll(|| {
    let screen = term.screen();
    (!screen.contains("more lines (ctrl+o)")).then_some(screen)
  });
  let shown = unfolded.lines().filter(|line| line.contains('▄')).count();
  assert!(shown > folded, "ctrl+o unfolds the image:\n{unfolded}");
  assert!(
    unfolded
      .lines()
      .all(|line| !line.contains('▄') || line.matches('▄').count() == 24),
    "unfolding does not redraw it wider:\n{unfolded}"
  );
}

#[test]
fn a_folded_line_is_cut_to_the_width_and_ctrl_o_gives_it_back() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "bash",
      // Printed rather than written out, so the only long line on screen is
      // the output and not the command that made it.
      args: serde_json::json!({ "command": "printf 'x%.0s' $(seq 200); echo" }),
    },
    Turn::Say("That is all."),
  ]);
  let term = Term::start("long-line", &provider, &["--no-session"]);
  term.submit("do it");
  term.wait_for("That is all.");

  // Folded, a line is a row: cut to the terminal, with the rest of it marked.
  let screen = term.wait_for("xxxx");
  let rows: Vec<&str> = screen
    .lines()
    .map(str::trim_end)
    .filter(|l| l.contains("xxxx"))
    .collect();
  assert_eq!(rows.len(), 1, "a folded line takes one row:\n{screen}");
  assert!(rows[0].ends_with('…'), "and says there is more:\n{screen}");

  // Unfolded, it is whole again, wrapped over as many rows as it takes.
  term.type_in("C-o");
  let unfolded = poll(|| {
    let screen = term.screen();
    (screen.lines().filter(|line| line.contains("xxxx")).count() > 1).then_some(screen)
  });
  let shown: usize = unfolded
    .lines()
    .filter(|line| line.contains("xxxx"))
    .map(|line| line.matches('x').count())
    .sum();
  assert_eq!(shown, 200, "ctrl+o gives back every column of it:\n{unfolded}");
}

/// Wait for the screen to say something the `wait_for` needle cannot — that
/// a fold note has gone, say.
fn poll(mut ready: impl FnMut() -> Option<String>) -> String {
  let deadline = Instant::now() + Duration::from_secs(15);
  while Instant::now() < deadline {
    if let Some(screen) = ready() {
      return screen;
    }
    std::thread::sleep(Duration::from_millis(50));
  }
  panic!("waited for the screen to settle");
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
  // Something waiting behind a run that is killed is not sent afterwards —
  // Esc stops everything — but it was typed, so it is still on the screen.
  term.submit("and this later");
  term.settle();
  term.type_in("Escape");
  term.wait_for("Aborted.");
  term.wait_for("Not sent: and this later");

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
  assert!(
    !provider.sent("and this later"),
    "what Esc stopped stays stopped, including what was waiting"
  );
}

/// Esc keeps the conversation as it stood up to the turn it landed in.
///
/// Nothing can ask a killed run what it got through — the task running it is
/// gone — so the turn Esc lands in is this side's reckoning of it, pieced
/// back together from what went past on screen. The turns behind that one
/// were finished and asked about, though, and the run said what they came to
/// in its own words each time it asked. Keeping those is the difference
/// between carrying on from an abort and re-reading the conversation from the
/// top to do it.
#[test]
fn an_abort_leaves_the_turns_behind_it_as_the_model_gave_them() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::ThinkCall {
      thought: "Count them first.",
      say: "First. ",
      tool: "bash",
      args: serde_json::json!({ "command": "echo one" }),
    },
    Turn::ThinkCall {
      thought: "Now the slow one.",
      say: "Second. ",
      tool: "bash",
      args: serde_json::json!({ "command": "sleep 60" }),
    },
    Turn::Say("Carried on."),
  ]);
  let term = Term::start("abort-prefix", &provider, &["--no-session"]);
  term.submit("do two things");
  // The second turn is under way, so the first is behind the abort.
  term.wait_for("⚙ bash sleep 60");
  let asked = wait_bodies(&provider, 2).last().expect("the second request").clone();
  term.type_in("Escape");
  term.wait_for("Aborted.");
  term.settle();
  term.submit("never mind, carry on");
  term.wait_for("Carried on.");
  term.settle();

  let messages = |body: &str| -> Vec<serde_json::Value> {
    let body: serde_json::Value = serde_json::from_str(body).expect("a JSON request");
    body["messages"].as_array().expect("messages").clone()
  };
  let (asked, after) = (messages(&asked), messages(&provider.request("never mind")));
  assert!(after.len() > asked.len(), "the abort left the turn it landed in behind");
  for (at, (before, now)) in asked.iter().zip(&after).enumerate() {
    assert_eq!(
      before,
      now,
      "message {at} of {} was rewritten by the abort",
      asked.len()
    );
  }

  // The turn it landed in is kept too — with the call it was in the middle
  // of answered, or the conversation is one no provider would take back.
  let added: Vec<&serde_json::Value> = after.iter().skip(asked.len()).collect();
  assert!(
    added.iter().any(|m| m.to_string().contains("Aborted by the user.")),
    "the interrupted call is answered: {:?}",
    added.iter().map(|m| &m["role"]).collect::<Vec<_>>()
  );
}

/// A call being written is taken off the screen by the call it becomes.
///
/// Rig mints its own id for a call so the fragments of one stay followable
/// before the provider has named it, and that is the id the half-written
/// line is keyed by — not the id the finished call carries. Dispatch has to
/// say both, or every call the run makes leaves its draft behind it.
/// A command's output, while it is still running, lands under the call that
/// asked for it.
///
/// Nothing tells the command which call it is. It reports down a channel the
/// dispatcher bound to that call before handing it over, so getting this
/// wrong would show the output adrift rather than under its own line.
#[test]
fn live_output_lands_under_the_call_that_asked_for_it() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "First. ",
      tool: "bash",
      args: serde_json::json!({ "command": "echo alpha; sleep 30" }),
    },
    Turn::Say("Done."),
  ]);
  let term = Term::start("live-output", &provider, &["--no-session"]);
  term.submit("go");
  // Its output, while the command it came from is still running: the only
  // window in which the live channel is what put it there.
  term.wait_for("alpha");
  term.settle();
  let screen = term.screen();
  let lines: Vec<&str> = screen.lines().map(str::trim_end).filter(|l| !l.is_empty()).collect();
  let call = lines
    .iter()
    .position(|line| line.contains("\u{2699} bash"))
    .expect("the call");
  assert!(
    lines[call + 1].contains("alpha"),
    "output under the call that asked for it, not adrift: {lines:?}"
  );
}

#[test]
fn a_call_being_written_is_replaced_by_the_call_it_becomes() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "One. ",
      tool: "bash",
      args: serde_json::json!({ "command": "echo one" }),
    },
    Turn::Call {
      say: "Two. ",
      tool: "bash",
      args: serde_json::json!({ "command": "sleep 60" }),
    },
    Turn::Say("Done."),
  ]);
  let term = Term::start("written-once", &provider, &["--no-session"]);
  term.submit("go");
  // Mid-run, with one turn finished and the next one's call running: the
  // moment a draft left behind would be visible.
  term.wait_for("\u{2699} bash sleep 60");
  term.settle();
  let screen = term.screen();
  let lines: Vec<&str> = screen.lines().map(str::trim_end).filter(|l| !l.is_empty()).collect();
  assert!(
    !screen.contains('\u{258C}'),
    "nothing is still being written once it has run:\n{screen}"
  );
  for command in ["echo one", "sleep 60"] {
    assert_eq!(
      lines.iter().filter(|line| line.contains(command)).count(),
      1,
      "{command:?} is drawn once, not as a call and a draft of one: {lines:?}"
    );
  }
}

#[test]
fn a_message_typed_mid_run_waits_at_the_bottom_and_goes_at_the_next_turn() {
  if !have_tmux() {
    return;
  }
  // A file long enough that it is still arriving, line by line, while the
  // test types at it — which is the run at its most in-progress.
  let file = (1..=400).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
  let provider = Provider::start(vec![
    Turn::Call {
      say: "Working. ",
      tool: "write",
      args: serde_json::json!({ "path": "notes.txt", "content": file }),
    },
    Turn::Say("Answered them both."),
  ]);
  let term = Term::start("queue", &provider, &["--no-session"]);
  term.submit("start something slow");
  term.wait_for("⚙ write notes.txt");
  term.submit("and this too");
  term.settle();

  // Where it waits is where it will be sent from: under the call still being
  // written, not above the file arriving under it.
  let screen = term.screen();
  let lines: Vec<&str> = screen.lines().map(str::trim_end).filter(|l| !l.is_empty()).collect();
  let call = lines.iter().position(|l| l.contains("⚙ write")).expect("the call");
  let writing = lines
    .iter()
    .rposition(|l| l.contains('▌'))
    .unwrap_or_else(|| panic!("the file still arriving: {lines:?}"));
  let queued = lines
    .iter()
    .position(|l| l.contains("Queued: and this too"))
    .unwrap_or_else(|| panic!("the message waiting its turn: {lines:?}"));
  assert!(queued > call, "a waiting message sits under the run: {lines:?}");
  assert!(
    queued > writing,
    "under what is still being written, not above it: {lines:?}"
  );

  // Ctrl+O says how much of a tool's output to show. What is waiting to be
  // sent is not the run's output, and is not the run's to take away.
  term.type_in("C-o");
  term.settle();
  let screen = term.screen();
  assert!(
    screen.contains('▌') && screen.contains("Queued: and this too"),
    "still waiting after the view changed under it:\n{screen}"
  );
  term.type_in("C-o");

  // And it goes at the next turn, not after the whole answer: the request
  // carrying it holds the command's result and nothing the model said after.
  term.wait_for("Answered them both.");
  let request = provider.request("and this too");
  assert!(
    request.contains("tool_call_id"),
    "sent once the running call was answered, keeping its work: {request}"
  );
  assert!(
    !request.contains("Answered them both."),
    "sent before the model answered the first message, not after: {request}"
  );
  let screen = term.screen();
  let lines: Vec<&str> = screen.lines().map(str::trim_end).filter(|l| !l.is_empty()).collect();
  assert!(
    !lines.iter().any(|l| l.contains("Queued:")),
    "nothing is left waiting once it is sent: {lines:?}"
  );
  let call = lines.iter().position(|l| l.contains("⚙ write")).expect("the call");
  let prompt = lines
    .iter()
    .position(|l| l.contains("❯ and this too"))
    .expect("the prompt it became");
  assert!(prompt > call, "it becomes a prompt where it waited: {lines:?}");
}

/// A run cut short for a waiting message leaves the conversation exactly as it
/// stood, so the request carrying that message is the last one with more on
/// the end — and everything before it is a prefix the provider has already
/// weighed and cached.
///
/// The run being stopped is the one place the conversation is not the model's
/// own words handed back: there is no final response to take them from. Piece
/// the run back together from what the screen showed and the turns come out
/// almost right — no reasoning on them, the text of a turn run together, the
/// results of one turn split across several messages — and almost right is a
/// cache miss on every token of the conversation from the first turn of that
/// run onwards.
#[test]
fn a_message_sent_mid_run_leaves_everything_before_it_untouched() {
  if !have_tmux() {
    return;
  }
  // Two turns that think before they work, so the run is stopped with turns
  // behind it that the screen alone could not put back.
  let file = (1..=400).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
  let provider = Provider::start(vec![
    Turn::ThinkCall {
      thought: "Notes first.",
      say: "Working. ",
      tool: "write",
      args: serde_json::json!({ "path": "notes.txt", "content": "one" }),
    },
    Turn::ThinkCall {
      thought: "And the long one.",
      say: "Still working. ",
      tool: "write",
      args: serde_json::json!({ "path": "more.txt", "content": file }),
    },
    Turn::Say("Answered them both."),
  ]);
  let term = Term::start("steer-prefix", &provider, &["--no-session"]);
  term.submit("start something slow");
  // The second turn is under way, so the first is behind the run and has to
  // survive it.
  term.wait_for("⚙ write more.txt");
  let stopped = wait_bodies(&provider, 2).last().expect("the second request").clone();
  term.submit("and this too");
  term.wait_for("Answered them both.");
  term.settle();

  // What the stopped run had asked for, and what the message waiting behind
  // it asked for once it went.
  let steered = provider.request("and this too");
  let messages = |body: &str| -> Vec<serde_json::Value> {
    let body: serde_json::Value = serde_json::from_str(body).expect("a JSON request");
    body["messages"].as_array().expect("messages").clone()
  };
  let (stopped, steered) = (messages(&stopped), messages(&steered));
  assert!(
    steered.len() > stopped.len(),
    "the message and the turn it waited for are on the end: {} then {}",
    stopped.len(),
    steered.len()
  );
  for (at, (before, after)) in stopped.iter().zip(&steered).enumerate() {
    assert_eq!(
      before,
      after,
      "message {at} of {} was rewritten by the run being stopped",
      stopped.len()
    );
  }

  // And the turn it was stopped in the middle of is on the end whole, thought
  // and all, followed by the message that stopped it.
  let added: Vec<&serde_json::Value> = steered.iter().skip(stopped.len()).collect();
  let reasoned = added.iter().any(|m| {
    m["reasoning_content"]
      .as_str()
      .is_some_and(|r| r.contains("And the long one."))
  });
  assert!(
    reasoned,
    "the stopped turn kept what it thought: {:?}",
    added.iter().map(|m| &m["role"]).collect::<Vec<_>>()
  );
  let last = added.last().expect("the message that was waiting");
  assert!(
    last.to_string().contains("and this too"),
    "the waiting message is the last thing asked: {last}"
  );
}

/// Two typed while the run was going both go, in the order they were typed
/// and in the one request — they were said before the turn that reads them,
/// so that is where the model sees them.
#[test]
fn everything_typed_while_a_run_went_goes_at_the_next_turn_in_order() {
  if !have_tmux() {
    return;
  }
  let file = (1..=400).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
  let provider = Provider::start(vec![
    Turn::Call {
      say: "Working. ",
      tool: "write",
      args: serde_json::json!({ "path": "notes.txt", "content": file }),
    },
    Turn::Say("Answered them all."),
  ]);
  let term = Term::start("queue-two", &provider, &["--no-session"]);
  term.submit("start something slow");
  term.wait_for("\u{2699} write notes.txt");
  term.submit("first extra");
  term.submit("second extra");
  term.wait_for("Queued: second extra");
  term.wait_for("Answered them all.");
  term.settle();

  let request = provider.request("first extra");
  assert!(
    request.contains("second extra"),
    "both go in the one request: {request}"
  );
  let (first, second) = (
    request.find("first extra").expect("the first"),
    request.find("second extra").expect("the second"),
  );
  assert!(first < second, "in the order they were typed");
  assert!(
    !request.contains("Answered them all."),
    "read at the next turn, not after the answer: {request}"
  );
  let screen = term.screen();
  assert!(
    !screen.contains("Queued:"),
    "nothing is left waiting once the run has read it:\n{screen}"
  );
  for typed in ["first extra", "second extra"] {
    assert_eq!(
      screen.lines().filter(|line| line.contains(typed)).count(),
      1,
      "{typed:?} is drawn once, not queued and sent:\n{screen}"
    );
  }
}

#[test]
fn a_waiting_message_can_be_taken_back_and_fixed() {
  if !have_tmux() {
    return;
  }
  let file = (1..=400).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
  let provider = Provider::start(vec![
    Turn::Call {
      say: "Working. ",
      tool: "write",
      args: serde_json::json!({ "path": "notes.txt", "content": file }),
    },
    Turn::Say("Answered them both."),
  ]);
  let term = Term::start("unqueue", &provider, &["--no-session"]);
  term.submit("start something slow");
  term.wait_for("⚙ write notes.txt");
  term.submit("second questionX");
  term.wait_for("Queued: second questionX");

  // Alt+Up hands the last one waiting back to the input box, typo and all.
  term.type_in("M-Up");
  term.settle();
  let screen = term.screen();
  assert!(
    !screen.contains("Queued:"),
    "taken back out of the queue, not copied out of it:\n{screen}"
  );
  let box_line = screen
    .lines()
    .find(|line| line.contains("second questionX"))
    .unwrap_or_else(|| panic!("the message back in the box:\n{screen}"));
  assert!(box_line.starts_with('│'), "in the input box: {box_line:?}");

  // Fixed there and sent again — what goes to the model is the corrected one.
  term.type_in("BSpace");
  term.type_in("Enter");
  term.wait_for("Answered them both.");
  assert!(provider.sent("second question"), "the fixed message went");
  assert!(
    !provider.sent("second questionX"),
    "and the one it was taken back from did not"
  );
}

/// Two prompts and their answers, each answer naming its question.
fn asked_twice(test: &str, provider: &Provider, args: &[&str]) -> Term {
  let term = Term::start(test, provider, args);
  term.submit("apple");
  term.wait_for("Answer to apple.");
  term.submit("pear");
  term.wait_for("Answer to pear.");
  term
}

#[test]
fn up_walks_back_through_the_prompts_and_a_resumed_session_brings_its_own() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = asked_twice("history", &provider, &[]);

  term.type_in("unsent");
  // From the middle of a line, the first Up only goes to the start of it:
  // reaching the top of something being typed is not also leaving it.
  term.type_in("Up");
  term.settle();
  assert_eq!(term.typed(), "unsent", "still what was being typed");

  term.type_in("Up");
  term.settle();
  assert_eq!(term.typed(), "pear", "the prompt before it");
  term.type_in("Up");
  term.settle();
  assert_eq!(term.typed(), "apple");
  // The oldest is as far back as it goes.
  term.type_in("Up");
  term.settle();
  assert_eq!(term.typed(), "apple");

  term.type_in("Down");
  term.settle();
  assert_eq!(term.typed(), "pear");
  term.type_in("Down");
  term.settle();
  assert_eq!(term.typed(), "unsent", "and what was being typed comes back");

  // The prompts are the session's, so reopening it brings them back — there
  // is no history file of our own, only the conversation.
  let term = term.reopen(&provider, &["-c"]);
  term.wait_for("Resumed session");
  term.type_in("Up");
  term.settle();
  assert_eq!(term.typed(), "pear", "the resumed session's last prompt");
}

/// The session picker is typed at the same way the model picker is: letters
/// narrow it to the sessions whose titles they match, and `Enter` resumes
/// what is left standing under the cursor.
#[test]
fn typing_at_the_session_picker_narrows_it_to_what_was_typed() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = Term::start("session-filter", &provider, &[]);
  // The rows a list has drawn something on: the box is as tall as the
  // screen whatever is in it, and the blank ones are not sessions.
  let listed = |rows: &[String]| {
    rows
      .iter()
      .filter(|row| !row.trim_matches(|c| c == '│' || c == ' ').is_empty())
      .count()
  };
  term.submit("apple");
  term.wait_for("Answer to apple.");
  term.submit("/new");
  term.wait_for("New session.");
  term.submit("pear");
  term.wait_for("Answer to pear.");

  term.submit("/resume");
  term.wait_for("Esc cancel");
  let (rows, _) = term.overlay();
  assert_eq!(listed(&rows), 2, "both to start with: {rows:?}");

  // A fuzzy match, as everywhere else in fa: "apl" is a-p-p-l-e.
  term.type_in("apl");
  term.wait_for_query("apl");
  let (rows, on) = term.overlay();
  assert_eq!(listed(&rows), 1, "only what matches is left: {rows:?}");
  assert!(rows[on].contains("apple"), "and it is the one meant: {rows:?}");

  // Backspace widens it again, and a query nothing matches says so rather
  // than leaving an empty box.
  term.type_in("BSpace");
  term.type_in("BSpace");
  term.type_in("BSpace");
  term.type_in("zzz");
  term.wait_for("No session matches.");
  term.type_in("BSpace");
  term.type_in("BSpace");
  term.type_in("BSpace");

  // What the filter left is what `Enter` resumes, rather than whichever row
  // stood in that place before anything was typed.
  term.type_in("apl");
  term.wait_for_query("apl");
  term.type_in("Enter");
  term.wait_for("Resumed session");
  term.type_in("Up");
  term.settle();
  assert_eq!(term.typed(), "apple", "the filtered-to session's own prompt");
}

/// The query is typed into the box at the bottom of the screen, and that box
/// is a box: the cursor moves through what has been typed the way it moves
/// through a prompt, and the list follows whatever the editing leaves.
#[test]
fn the_query_is_edited_with_the_keys_that_edit_a_prompt() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = Term::start("query-keys", &provider, &[]);
  let listed = |rows: &[String]| {
    rows
      .iter()
      .filter(|row| !row.trim_matches(|c| c == '│' || c == ' ').is_empty())
      .count()
  };
  term.submit("apple");
  term.wait_for("Answer to apple.");
  term.submit("/new");
  term.wait_for("New session.");
  term.submit("pear");
  term.wait_for("Answer to pear.");

  term.submit("/resume");
  term.wait_for("Esc cancel");
  term.type_in("aple");
  term.wait_for_query("aple");

  // The arrows walk back into the query rather than steering the list, which
  // is what the letter typed between two others proves: dropped at the end
  // instead, "aplep" would match nothing.
  term.type_in("Left");
  term.type_in("Left");
  term.type_in("p");
  term.wait_for_query("apple");
  let (rows, on) = term.overlay();
  assert_eq!(listed(&rows), 1, "still the one session: {rows:?}");
  assert!(rows[on].contains("apple"), "and it is the one meant: {rows:?}");

  // Home goes to the start of the query, where a letter narrows it to
  // nothing.
  term.type_in("Home");
  term.type_in("z");
  term.wait_for_query("zapple");
  term.wait_for("No session matches.");

  // And Delete is the query's too: it takes the letter back rather than
  // asking about the session under the cursor, which is Ctrl+D's question.
  term.type_in("Home");
  term.type_in("DC");
  term.wait_for_query("apple");
  let screen = term.screen();
  assert!(
    !screen.contains("delete?"),
    "Delete edits the query and asks nothing:\n{screen}"
  );

  // End goes back to the far end of it, so Backspace can empty it out and
  // leave the whole list standing again.
  term.type_in("End");
  for _ in 0..5 {
    term.type_in("BSpace");
  }
  term.settle();
  let (rows, _) = term.overlay();
  assert_eq!(listed(&rows), 2, "both sessions again: {rows:?}");
}

/// A session file is the only copy of the conversation in it, so the picker
/// asks before it removes one — and the session on screen, which is still
/// writing to its file, is not one it will remove at all.
#[test]
fn the_picker_deletes_a_session_once_it_has_asked_about_it() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = Term::start("delete", &provider, &[]);
  term.submit("apple");
  term.wait_for("Answer to apple.");
  // A second session, so the one being deleted is not the one on screen.
  term.submit("/new");
  term.wait_for("New session.");
  term.submit("pear");
  term.wait_for("Answer to pear.");
  assert_eq!(term.session_files().len(), 2);

  // `C-d` is what tmux calls Ctrl+D.
  term.submit("/resume");
  term.wait_for("Esc cancel");
  term.point_at("pear");
  term.type_in("C-d");
  term.wait_for("delete? Ctrl+D to confirm");
  term.type_in("C-d");
  term.wait_for("start another");
  term.settle();
  let (rows, _) = term.overlay();
  assert!(
    rows.iter().any(|row| row.contains("pear")),
    "the session on screen stays: {rows:?}"
  );
  assert_eq!(term.session_files().len(), 2, "and so does its file");

  // Anything but a second Delete answers no, and only puts the question away.
  term.point_at("apple");
  term.type_in("C-d");
  term.wait_for("delete? Ctrl+D to confirm");
  term.type_in("Up");
  term.settle();
  let screen = term.screen();
  assert!(
    !screen.contains("delete? Ctrl+D to confirm"),
    "the question is answered:\n{screen}"
  );
  assert_eq!(term.session_files().len(), 2, "and nothing is deleted");

  // The row deleted is the one the filter left under the cursor, rather than
  // whichever row stood in that place before anything was typed.
  term.type_in("aple");
  term.wait_for_query("aple");
  term.type_in("C-d");
  term.wait_for("delete? Ctrl+D to confirm");
  term.type_in("C-d");
  term.wait_for("Deleted session apple.");
  term.wait_for("No session matches.");
  for _ in 0..4 {
    term.type_in("BSpace");
  }
  term.settle();
  let (rows, _) = term.overlay();
  assert!(
    !rows.iter().any(|row| row.contains("apple")),
    "the row goes with the file: {rows:?}"
  );
  let files = term.session_files();
  assert_eq!(files.len(), 1);
  assert!(files[0].1.contains("pear"), "the one that was kept: {files:?}");
}

/// Which row of an overlay says `needle`.
fn row_at(rows: &[String], needle: &str) -> usize {
  rows
    .iter()
    .position(|row| row.contains(needle))
    .unwrap_or_else(|| panic!("a row saying {needle:?}: {rows:?}"))
}

/// How far that row is indented, which is what the tree draws its shape with.
fn indent_of(rows: &[String], needle: &str) -> usize {
  let row = &rows[row_at(rows, needle)];
  row[..row.find(needle).expect("the row")].chars().count()
}

#[test]
fn going_back_leaves_a_branch_that_can_be_walked_into_again() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = asked_twice("tree", &provider, &[]);

  // Going back to a prompt hands it to the input box and ends the
  // conversation before it.
  term.submit("/tree");
  term.wait_for("Esc cancel");
  term.choose("❯ pear");
  term.wait_for("Moved to 2 messages.");
  assert_eq!(term.typed(), "pear", "the prompt comes back to be asked again");
  let screen = term.screen();
  assert!(
    !screen.contains("Answer to pear."),
    "the conversation ends before it now:\n{screen}"
  );

  // Asked differently, what was there before is not gone — it is a branch,
  // and the list says so: the way we are on first, the way we left indented
  // under the point the two part.
  for _ in 0..4 {
    term.type_in("BSpace");
  }
  term.submit("plum");
  term.wait_for("Answer to plum.");
  term.submit("/tree");
  term.wait_for("Esc cancel");
  let (rows, _) = term.overlay();
  assert!(
    row_at(&rows, "❯ plum") < row_at(&rows, "❯ pear"),
    "the way we are on comes first: {rows:?}"
  );
  assert!(
    indent_of(&rows, "❯ plum") > indent_of(&rows, "❯ apple")
      && indent_of(&rows, "❯ pear") > indent_of(&rows, "❯ apple"),
    "both ways are indented under where they part: {rows:?}"
  );

  // And the answer that was left behind is a place to go back to.
  term.choose("Answer to pear.");
  term.wait_for("Moved to 4 messages.");
  let screen = term.screen();
  assert!(
    screen.contains("Answer to pear."),
    "walked into the branch that was left:\n{screen}"
  );
  assert!(
    !screen.contains("Answer to plum."),
    "which is now the one left behind:\n{screen}"
  );
  // One file held both ways the whole time.
  let session = term.session_file();
  for said in ["apple", "pear", "plum"] {
    assert!(session.contains(said), "{said:?} kept in the one file");
  }

  // So reopening it brings back the tree and not just the branch it was left
  // on: every way the conversation went is still a place it can go.
  let term = term.reopen(&provider, &["-c"]);
  term.wait_for("Resumed session");
  term.submit("/tree");
  term.wait_for("Esc cancel");
  let (rows, _) = term.overlay();
  for said in ["❯ apple", "❯ pear", "❯ plum"] {
    assert!(
      rows.iter().any(|row| row.contains(said)),
      "{said:?} is still somewhere to go: {rows:?}"
    );
  }
  assert!(
    indent_of(&rows, "❯ plum") > indent_of(&rows, "❯ apple")
      && indent_of(&rows, "❯ pear") > indent_of(&rows, "❯ apple"),
    "and the shape of the tree came back with it: {rows:?}"
  );
}

/// The tree is typed at the same way the session picker is: letters narrow it
/// to the points whose rows they match, and `Enter` goes to what is left
/// standing under the cursor rather than to whatever stood in that place
/// before anything was typed.
#[test]
fn typing_at_the_tree_narrows_it_to_what_was_typed() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = asked_twice("tree-filter", &provider, &[]);
  // The rows a list has drawn something on: the box is as tall as the screen
  // whatever is in it, and the blank ones are not points.
  let listed = |rows: &[String]| {
    rows
      .iter()
      .filter(|row| !row.trim_matches(|c| c == '│' || c == ' ').is_empty())
      .count()
  };

  term.submit("/tree");
  term.wait_for("Esc cancel");
  let (rows, _) = term.overlay();
  assert_eq!(listed(&rows), 4, "both prompts and both answers: {rows:?}");

  // A fuzzy match, as everywhere else in fa: "aple" is a-p-p-l-e, and it
  // leaves the prompt and the answer that say it.
  term.type_in("aple");
  term.wait_for_query("aple");
  let (rows, _) = term.overlay();
  assert_eq!(listed(&rows), 2, "only what matches is left: {rows:?}");
  assert!(
    !rows.iter().any(|row| row.contains("pear")),
    "and nothing that does not: {rows:?}"
  );

  // Backspace widens it again, and a query nothing matches says so rather
  // than leaving an empty box.
  for _ in 0..4 {
    term.type_in("BSpace");
  }
  term.type_in("zzz");
  term.wait_for("No point matches.");
  for _ in 0..3 {
    term.type_in("BSpace");
  }

  // What the filter left is what `Enter` goes to: the row taken is the point
  // it stands for, not the place it sits in the narrowed list.
  term.type_in("pear");
  term.wait_for_query("pear");
  term.choose("❯ pear");
  term.wait_for("Moved to 2 messages.");
  assert_eq!(term.typed(), "pear", "the filtered-to prompt comes back");
}

/// A branch left behind can be deleted from the tree the way a session is
/// from the picker: `Ctrl+D` asks, a second one removes it from the file. The
/// conversation on screen is not one it will take out from under itself.
#[test]
fn the_tree_deletes_a_branch_once_it_has_asked_about_it() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = asked_twice("tree-delete", &provider, &[]);
  // Back under the second prompt and a different one asked, which leaves
  // `pear` and its answer a branch of their own.
  term.submit("/tree");
  term.wait_for("Esc cancel");
  term.choose("❯ pear");
  term.wait_for("Moved to 2 messages.");
  for _ in 0..4 {
    term.type_in("BSpace");
  }
  term.submit("plum");
  term.wait_for("Answer to plum.");

  // Where the session is stays put, asked or not.
  term.submit("/tree");
  term.wait_for("Esc cancel");
  term.point_at("Answer to plum.");
  term.type_in("C-d");
  term.wait_for("delete? Ctrl+D to confirm");
  term.type_in("C-d");
  term.wait_for("go somewhere else");
  term.settle();
  let (rows, _) = term.overlay();
  assert!(
    rows.iter().any(|row| row.contains("Answer to plum.")),
    "the conversation on screen stays: {rows:?}"
  );

  // Anything else answers no.
  term.point_at("❯ pear");
  term.type_in("C-d");
  term.wait_for("delete? Ctrl+D to confirm");
  term.type_in("Escape");
  term.settle();
  let (rows, _) = term.overlay();
  assert!(
    rows.iter().any(|row| row.contains("pear")),
    "nothing is deleted: {rows:?}"
  );

  // The branch is the prompt and everything after it.
  term.type_in("C-d");
  term.wait_for("delete? Ctrl+D to confirm");
  term.type_in("C-d");
  term.wait_for("Deleted ❯ pear, 2 messages in all.");
  term.settle();
  let (rows, _) = term.overlay();
  assert!(
    !rows.iter().any(|row| row.contains("pear")),
    "the prompt and its answer go together: {rows:?}"
  );
  assert!(
    rows.iter().any(|row| row.contains("Answer to plum.")),
    "and the rest stays: {rows:?}"
  );
  let session = term.session_file();
  assert!(!session.contains("pear"), "the file forgets it too:\n{session}");
  assert!(session.contains("plum"));
}

#[test]
fn forking_starts_a_session_of_its_own_and_leaves_the_first_alone() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = asked_twice("fork", &provider, &[]);
  let before = term.session_files();
  let (original, was) = before.first().expect("a session file").clone();
  assert_eq!(before.len(), 1, "one session so far");

  term.submit("/fork");
  term.wait_for("Esc cancel");
  // Only prompts are offered, and this one keeps the turn before it.
  let (rows, _) = term.overlay();
  assert!(
    !rows.iter().any(|row| row.contains("Answer to")),
    "a fork starts from a question: {rows:?}"
  );
  // This list is typed at like the others, and the row the query leaves is
  // the one forked from.
  term.type_in("pea");
  term.wait_for_query("pea");
  let (rows, _) = term.overlay();
  assert!(
    !rows.iter().any(|row| row.contains("apple")),
    "only the prompts that match are left: {rows:?}"
  );
  term.choose("❯ pear");
  term.wait_for("Forked, 2 messages kept.");
  assert_eq!(term.typed(), "pear", "the prompt comes back to be asked again");
  let screen = term.screen();
  assert!(
    screen.contains("Answer to apple.") && !screen.contains("Answer to pear."),
    "the fork holds the conversation up to that point:\n{screen}"
  );

  let after = term.session_files();
  assert_eq!(after.len(), 2, "the fork is a file of its own: {after:?}");
  let (_, now) = after.iter().find(|(path, _)| path == &original).expect("the original");
  assert_eq!(now, &was, "the session forked from is left exactly as it was");
  let (_, forked) = after.iter().find(|(path, _)| path != &original).expect("the fork");
  assert!(
    forked.contains(&format!("\"parent\":\"{}\"", original.display())),
    "the fork says where it came from: {forked}"
  );
  assert!(
    !forked.contains("Answer to pear."),
    "and carries only the path it was forked at: {forked}"
  );

  // What is said next belongs to the fork alone.
  term.type_in("Enter");
  term.wait_for("Answer to pear.");
  let after = term.session_files();
  let (_, now) = after.iter().find(|(path, _)| path == &original).expect("the original");
  assert_eq!(now, &was, "still untouched once the fork is talked to");
}

#[test]
fn a_compacted_conversation_reaches_the_model_as_its_summary() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = Term::start("compact", &provider, &["--no-session"]);
  term.submit("remember the kumquat");
  term.wait_for("Answer to remember the kumquat.");
  term.submit("and the pomelo");
  term.wait_for("Answer to and the pomelo.");

  term.submit("/compact");
  term.wait_for("Compacted 2 messages into a summary; kept the last 2 messages.");
  let screen = term.screen();
  assert!(
    screen.contains("▤ Context summary") && screen.contains("Fruit was discussed."),
    "the summary is shown as one:\n{screen}"
  );
  // The kept turn is kept whole, so there is no beginning of one left over
  // to summarize: one summarization request, and nothing joined onto it.
  let summarizing = provider
    .bodies()
    .iter()
    .filter(|body| body.contains(SUMMARIZING))
    .count();
  assert_eq!(summarizing, 1, "a whole turn is summarized once");
  assert!(!screen.contains("Turn Context"), "with nothing about a split turn");
  // The room a compaction made is the thing to see, so the footer says how
  // much there is rather than going blank until the next call comes back.
  assert!(
    screen.contains("ctx 0%"),
    "the context is estimated until a call has weighed it:\n{screen}"
  );

  // What the model is given next is the summary in place of what it stands
  // for — the point of compacting at all.
  term.submit("what now");
  term.wait_for("Answer to what now.");
  let request = provider.request("what now");
  assert!(
    request.contains("Fruit was discussed."),
    "the summary goes in the history: {request}"
  );
  assert!(
    !request.contains("kumquat"),
    "in place of the turns it summarized: {request}"
  );
  assert!(
    request.contains("and the pomelo"),
    "while the recent turn stays verbatim: {request}"
  );
}

/// The cut can fall inside a turn, and then the half of it that is dropped is
/// summarized a second time — in terms of the half still there, rather than of
/// the conversation, which is what pi does. Unless the session says not to.
#[test]
fn the_start_of_a_split_turn_is_summarized_on_its_own() {
  if !have_tmux() {
    return;
  }
  // A budget of one token keeps only the last answer, so the cut falls on it
  // — inside the turn that asked for it, whose question is left over.
  let compacted = |turn_summary: bool| -> (String, Vec<String>) {
    let provider = Provider::start(vec![Turn::Echo]);
    let mut args = vec!["--no-session", "--keep-recent-tokens", "1"];
    if !turn_summary {
      args.push("--no-turn-summary");
    }
    let term = Term::start(if turn_summary { "split" } else { "unsplit" }, &provider, &args);
    term.submit("remember the kumquat");
    term.wait_for("Answer to remember the kumquat.");
    term.submit("and the pomelo");
    term.wait_for("Answer to and the pomelo.");
    term.submit("/compact");
    let screen = term.wait_for("Compacted 3 messages into a summary; kept the last 1 message.");
    let asked = provider
      .bodies()
      .into_iter()
      .filter(|body| body.contains(SUMMARIZING))
      .collect();
    (screen, asked)
  };

  let (screen, asked) = compacted(true);
  assert_eq!(asked.len(), 2, "the conversation, then the turn: {asked:?}");
  assert!(
    asked[0].contains("remember the kumquat") && !asked[0].contains("and the pomelo"),
    "the first asks about the turns before the one being split: {}",
    asked[0]
  );
  assert!(
    asked[1].contains(SUMMARIZING_TURN) && asked[1].contains("and the pomelo"),
    "the second about the beginning of that turn, by its own prompt: {}",
    asked[1]
  );
  // Both come back as the one message the conversation keeps.
  assert!(
    screen.contains("Fruit was discussed.")
      && screen.contains("**Turn Context (split turn):**")
      && screen.contains("Something about fruit."),
    "joined into one summary:\n{screen}"
  );

  // Told not to, it is one request again and the turn's beginning goes into
  // the checkpoint with everything else.
  let (screen, asked) = compacted(false);
  assert_eq!(asked.len(), 1, "one summarization only: {asked:?}");
  assert!(
    asked[0].contains("remember the kumquat") && asked[0].contains("and the pomelo"),
    "covering everything before the cut: {}",
    asked[0]
  );
  assert!(!screen.contains("Turn Context"), "and nothing joined on:\n{screen}");
}

/// A summary is what the conversation before it now *is*, so it has to
/// survive the session being closed: what is on disk is a checkpoint, and
/// what comes back out of it is the message the model was given.
#[test]
fn a_compacted_session_still_reaches_the_model_as_its_summary_when_reopened() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = Term::start("compact-reopen", &provider, &[]);
  term.submit("remember the kumquat");
  term.wait_for("Answer to remember the kumquat.");
  term.submit("and the pomelo");
  term.wait_for("Answer to and the pomelo.");
  term.submit("/compact");
  term.wait_for("Compacted 2 messages into a summary; kept the last 2 messages.");

  // Closed and opened again on the same session.
  let term = term.reopen(&provider, &["--continue"]);
  let screen = term.wait_for("Resumed session");
  assert!(
    screen.contains("▤ Context summary") && screen.contains("Fruit was discussed."),
    "the checkpoint is drawn as one again:\n{screen}"
  );
  // And what it stands for is still above it, as it was on screen when the
  // compaction happened — the summary is what the model reads, not all the
  // session has to show.
  assert!(
    screen.contains("remember the kumquat") && screen.contains("Answer to remember the kumquat."),
    "the compacted turns are drawn above the summary:\n{screen}"
  );

  term.submit("what now");
  term.wait_for("Answer to what now.");
  let request = provider.request("what now");
  assert!(
    request.contains("Fruit was discussed."),
    "and is what the model is given: {request}"
  );
  assert!(
    !request.contains("kumquat"),
    "in place of what it stands for: {request}"
  );
}

/// A run is many calls to the model, and what each cost is known the moment
/// it comes back. Waiting until the run is over to say so leaves the footer
/// standing still through the long ones — which are the ones worth watching.
#[test]
fn what_a_call_cost_is_counted_when_it_comes_back_not_when_the_run_ends() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "bash",
      args: serde_json::json!({ "command": "sleep 2; echo ok" }),
    },
    Turn::Say("all done"),
  ]);
  let term = Term::start("usage", &provider, &["--no-session"]);
  term.submit("go");
  // The first call has been paid for while the command it asked for is
  // still running — the run has not reported anything yet.
  let screen = term.wait_for(&format!("{SPENT}↓"));
  assert!(
    !screen.contains("all done"),
    "counted mid-run, before the answer: {screen}"
  );
  term.wait_for("all done");
  let screen = term.screen();
  assert!(
    screen.contains(&format!("{}↓", SPENT * 2)),
    "the second call adds to the first: {screen}"
  );
}

/// A window filling up is worth seeing before it is full, so the footer's
/// share of it is coloured rather than dim once there is little left.
#[test]
fn a_context_window_with_little_left_in_it_says_so_in_colour() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  // Nothing makes room here, so the figure stays where the first answer put
  // it: a window of 100 tokens and a message that will not fit in it.
  let term = Term::start(
    "ctx-colour",
    &provider,
    &["--no-session", "--no-compaction", "--context-window", "100"],
  );
  let screen = term.screen();
  assert!(
    screen.contains("ctx 0%"),
    "an empty conversation weighs nothing, and the footer says so rather than \
     going blank: {screen}"
  );

  term.submit(&format!("remember the kumquat {}", "x".repeat(600)));
  // What the answer cost is written at the same time as what the request
  // came to, so waiting for the one waits for the other — and the figure
  // under test is the provider's rather than the estimate standing in.
  term.wait_for(&format!("{SPENT}↓"));
  let screen = term.coloured();
  let footer = screen.lines().find(|line| line.contains("ctx ")).expect("a footer");
  assert!(
    footer.contains(&format!("{RED}ctx ")),
    "a window this far past full is red: {footer:?}"
  );
}

/// The context window fills up mid-run, and the run makes room and carries
/// on by itself: the user asked for the work, not for a conversation about
/// how much of it fits.
#[test]
fn a_run_that_fills_the_context_window_compacts_and_picks_itself_back_up() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "bash",
      // Slow enough that what the run does after compacting cannot be
      // mistaken for something it had already done before.
      args: serde_json::json!({ "command": "sleep 1; echo ok" }),
    },
    Turn::Say("all done"),
  ]);
  // A window of 1000 tokens with 700 held back for the answer leaves 300 to
  // say everything in, and a single message of 400 fills it.
  let term = Term::start(
    "overflow",
    &provider,
    &[
      "--no-session",
      "--context-window",
      "1000",
      "--reserve-tokens",
      "700",
      "--keep-recent-tokens",
      "1",
    ],
  );
  term.submit(&format!("remember the kumquat {}", "x".repeat(1600)));

  // The run stops at the turn after the one that overflowed. What was asked
  // becomes the summary; the call it led to and what came back stay, so the
  // run has its own work in front of it when it goes on — which it does,
  // without anything being typed at it.
  term.wait_for("Compacted 1 message into a summary; kept the last 2 messages.");
  let screen = term.wait_for("all done");
  let at = |needle: &str| {
    screen
      .lines()
      .position(|line| line.contains(needle))
      .unwrap_or_else(|| panic!("{needle:?} on screen:\n{screen}"))
  };
  assert!(
    at("Compacted 1 message") < at("all done"),
    "the room was made before the answer, not after it:\n{screen}"
  );

  // The turn it finished on was asked over the summary, not over what the
  // summary stands for — and over the work it was in the middle of.
  //
  // The cut fell inside the only turn there was, so the summary is the
  // beginning of that turn and nothing else: there was no conversation in
  // front of it to make a checkpoint of.
  let bodies = provider.bodies();
  let last = bodies
    .iter()
    .rfind(|body| !body.contains(SUMMARIZING))
    .expect("a request that was not the summarizer's");
  assert!(
    last.contains("No prior history.") && last.contains("Something about fruit.") && !last.contains("kumquat"),
    "the run carried on over the compacted history: {last}"
  );
  assert!(
    last.contains("tool_call_id"),
    "with the call it had made still in front of it: {last}"
  );
}

/// When the one turn the context is full of is the turn it is full of, there
/// is no room to be made: carrying on regardless would fill the window again,
/// ask for the same summary again, and never stop. So it stops, and says so.
#[test]
fn a_context_full_of_a_single_turn_stops_rather_than_compacting_forever() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "bash",
      args: serde_json::json!({ "command": "echo ok" }),
    },
    Turn::Say("all done"),
  ]);
  // The same full window as above, but keeping the recent turns verbatim —
  // and one turn is all there is, so the summary would have nothing to say.
  let term = Term::start(
    "stuck",
    &provider,
    &["--no-session", "--context-window", "1000", "--reserve-tokens", "700"],
  );
  term.submit(&format!("remember the kumquat {}", "x".repeat(1600)));

  term.wait_for("nothing left to compact");
  term.settle();
  let screen = term.screen();
  assert!(
    !screen.contains("all done"),
    "the run is left where it stopped, for the user to decide: {screen}"
  );
  // And `/continue` is the deciding: one more turn goes out, full window or
  // not, and it is the model's own next step rather than another summary.
  term.submit("/continue");
  term.wait_for("all done");
}

/// MCP is a feature, and a build without it is a build with four tools and no
/// servers to bring more — so this is a test of the feature, not of fa.
#[test]
#[cfg(feature = "mcp")]
fn a_tool_from_an_mcp_server_is_offered_called_and_drawn_like_any_other() {
  if !have_tmux() || !have_python() {
    return;
  }
  let dir = scratch("mcp");
  let server = dir.join("server.py");
  std::fs::write(&server, MCP_SERVER).expect("a server to run");
  let config = dir.join("mcp.toml");
  std::fs::write(
    &config,
    // A line of shell, as it would be typed — the server is started by the
    // command that starts it in a terminal. It offers two tools; this session
    // wants one of them.
    format!(
      "[weather]\ncommand = \"python3 '{}'\"\ntimeout = 10\nexcept = [\"forecast\", \"flood\", \"book\", \"grow\", \"jot\"]\n",
      server.display()
    ),
  )
  .expect("a config to read");

  let provider = Provider::start(vec![
    Turn::Call {
      say: "Let me look. ",
      tool: "weather",
      args: serde_json::json!({ "city": "Berlin" }),
    },
    Turn::Say("Take a coat."),
  ]);
  let term = Term::start("mcp", &provider, &["--no-session", "--mcp-config", &shell(&config)]);

  // What came up is said before anything else, since there is nowhere else
  // to say it: the terminal did not exist yet. The footer keeps saying it,
  // since a session with servers is a session with more than four tools.
  term.wait_for("MCP weather: 1 tool");
  term.wait_for("1 mcp, 1 tool");
  term.submit("what is the weather in Berlin");
  term.wait_for("Take a coat.");

  // The model was offered the tool by the name its server gave it, and not
  // the one this session asked the server to keep.
  assert!(
    provider.sent(r#""name":"weather""#),
    "the server's tool went to the model with the other four"
  );
  assert!(
    !provider.sent("forecast"),
    "a tool a server was asked to keep is never offered"
  );
  // A server with no resources brings no tools for reading them.
  assert!(
    !provider.sent("read_resource"),
    "nothing to read, so nothing to read it with"
  );
  // And it reads in the transcript like any other tool: the call, then what
  // came back under it.
  let screen = term.screen();
  let lines: Vec<&str> = screen.lines().map(str::trim_end).filter(|l| !l.is_empty()).collect();
  let call = lines
    .iter()
    .position(|line| line.contains("⚙ weather"))
    .unwrap_or_else(|| panic!("the call: {lines:?}"));
  assert!(
    lines[call].contains("Berlin"),
    "its arguments are on its line: {lines:?}"
  );
  // What the server answered, under the call — and laid out as the structure
  // it is rather than left as the one long line it arrived as.
  assert!(
    lines[call + 1].trim_start().starts_with('{'),
    "under its call: {lines:?}"
  );
  assert!(
    lines[call + 2].contains(r#""city": "Berlin","#),
    "a field to a line, with room to read it: {lines:?}"
  );
  // And read as JSON, not as a block of dim text.
  #[cfg(feature = "lang-json")]
  {
    let coloured = term.coloured();
    let field = coloured
      .lines()
      .find(|line| line.contains("Berlin"))
      .expect("the field on screen");
    assert!(field.contains("\u{1b}[38;5;"), "highlighted: {field:?}");
  }
  let _ = std::fs::remove_dir_all(&dir);
}

/// A server with resources has the model offered two tools for them: one to
/// list what there is, templates and all, and one to read it.
#[test]
#[cfg(feature = "mcp")]
fn a_server_with_resources_has_them_listed_and_read() {
  if !have_tmux() || !have_python() {
    return;
  }
  let dir = scratch("mcp-resources");
  let server = dir.join("server.py");
  std::fs::write(&server, MCP_SERVER).expect("a server to run");
  let config = dir.join("mcp.toml");
  std::fs::write(
    &config,
    format!(
      "[notes]\ncommand = \"python3 '{}' resources\"\ntimeout = 10\ntools = [\"weather\"]\n",
      server.display()
    ),
  )
  .expect("a config to read");

  let provider = Provider::start(vec![
    Turn::Call {
      say: "Looking. ",
      tool: "list_resources",
      args: serde_json::json!({}),
    },
    Turn::Call {
      say: "Reading. ",
      tool: "read_resource",
      args: serde_json::json!({ "server": "notes", "uri": "note://today" }),
    },
    Turn::Call {
      say: "And another. ",
      tool: "read_resource",
      args: serde_json::json!({ "server": "notes", "uri": "note://never" }),
    },
    Turn::Say("Milk it is."),
  ]);
  let term = Term::start(
    "mcp-resources",
    &provider,
    &["--no-session", "--mcp-config", &shell(&config)],
  );
  term.wait_for("MCP notes: 1 tool");
  term.submit("what is on today");
  term.wait_for("Milk it is.");

  let last = provider.bodies().last().cloned().expect("a request");
  let request: serde_json::Value = serde_json::from_str(&last).expect("JSON");
  let offered: Vec<&str> = request["tools"]
    .as_array()
    .expect("tools")
    .iter()
    .filter_map(|tool| tool["function"]["name"].as_str())
    .collect();
  assert!(offered.contains(&"list_resources"), "{offered:?}");
  assert!(offered.contains(&"read_resource"), "{offered:?}");
  // What the listing and the reads came back as, as the model was told.
  let told: Vec<String> = request["messages"]
    .as_array()
    .expect("messages")
    .iter()
    .filter(|message| message["role"] == "tool")
    .map(|message| message["content"].to_string())
    .collect();
  assert!(
    told[0].contains("note://today — today (text/plain): What is on today."),
    "a resource to a line: {told:?}"
  );
  assert!(
    told[0].contains("note://{day} — day: The note for a day."),
    "and its templates: {told:?}"
  );
  assert!(told[1].contains("Buy milk."), "what it read: {told:?}");
  assert!(
    told[2].contains("no such note"),
    "and what the server said when it could not: {told:?}"
  );
  let screen = term.screen();
  let call = screen.find("⚙ read_resource").expect("the call on screen");
  assert!(screen[call..].contains("Buy milk."), "under it: {screen}");
  let _ = std::fs::remove_dir_all(&dir);
}

/// What a server has to read can be named in a prompt as `&server:uri`,
/// completed from what it lists by the same fuzzy match a path is, and sent
/// as the reference it is written as, for the model to read. An `&` word
/// whose first letter begins no server's name is left alone. A server whose list
/// changes has the new one completed from.
#[test]
#[cfg(feature = "mcp")]
fn a_resource_is_completed_after_an_ampersand_and_sent_as_a_reference() {
  if !have_tmux() || !have_python() {
    return;
  }
  let dir = scratch("mcp-attach");
  let server = dir.join("server.py");
  std::fs::write(&server, MCP_SERVER).expect("a server to run");
  let config = dir.join("mcp.toml");
  std::fs::write(
    &config,
    format!(
      "[notes]\ncommand = \"python3 '{}' resources\"\ntimeout = 10\ntools = [\"jot\"]\n",
      server.display()
    ),
  )
  .expect("a config to read");
  // Answering, the model writes tomorrow's note, which changes what the
  // server has to read.
  let provider = Provider::start(vec![
    Turn::Call {
      say: "Jotting. ",
      tool: "jot",
      args: serde_json::json!({}),
    },
    Turn::Say("Milk it is."),
  ]);
  let term = Term::start(
    "mcp-attach",
    &provider,
    &["--no-session", "--mcp-config", &shell(&config)],
  );
  term.wait_for("MCP notes: 1 tool");

  // Everything the servers have, at a bare `&`; then matched the way a path
  // is — the letters needing only to come in order, the first beginning
  // the server's name.
  term.type_in("what is on &");
  term.wait_for("notes:note://today");
  term.type_in("nday");
  term.wait_for("notes:note://today");
  term.wait_for("template");
  term.type_in("Enter");
  term.wait_for("│what is on &notes:note://today ");
  // And code, which is no server's.
  term.type_in(" for &mut self");
  term.type_in("Enter");
  let screen = term.wait_for("Milk it is.");
  assert!(
    screen.contains("❯ what is on &notes:note://today for &mut self"),
    "the prompt as typed: {screen}"
  );
  assert!(!screen.contains("Not attached"), "nothing to attach: {screen}");

  let body = provider.bodies().last().cloned().expect("a request");
  let request: serde_json::Value = serde_json::from_str(&body).expect("JSON");
  let user = request["messages"]
    .as_array()
    .expect("messages")
    .iter()
    .find(|message| message["role"] == "user")
    .expect("the prompt")
    .to_string();
  assert!(
    user.contains("what is on &notes:note://today for &mut self"),
    "the prompt as typed: {user}"
  );
  assert!(!user.contains("Buy milk."), "a reference, not what it names: {user}");

  // A server whose list changes is completed from the new one.
  term.type_in("&notes:tom");
  term.wait_for("note://tomorrow");
  let _ = std::fs::remove_dir_all(&dir);
}

/// A server that says its tools have changed has them fetched again and put
/// in place of the ones it had: the next request offers the new ones and not
/// the old, the transcript says what changed, and the footer counts them.
/// What the session was told about tools holds for the new ones too.
#[test]
#[cfg(feature = "mcp")]
fn a_server_that_changes_its_tools_has_the_new_ones_offered() {
  if !have_tmux() || !have_python() {
    return;
  }
  let dir = scratch("mcp-changed");
  let server = dir.join("server.py");
  std::fs::write(&server, MCP_SERVER).expect("a server to run");
  let config = dir.join("mcp.toml");
  std::fs::write(
    &config,
    format!(
      "[weather]\ncommand = \"python3 '{}'\"\ntimeout = 10\ntools = [\"weather\", \"grow\", \"radar\", \"tide\"]\n",
      server.display()
    ),
  )
  .expect("a config to read");

  let provider = Provider::start(vec![
    Turn::Call {
      say: "Growing. ",
      tool: "grow",
      args: serde_json::json!({}),
    },
    Turn::Call {
      say: "Looking. ",
      tool: "radar",
      args: serde_json::json!({ "city": "Oslo" }),
    },
    Turn::Say("Rain in Oslo."),
  ]);
  // `radar` and `tide` are not there yet when the session starts, and the
  // rule against `tide` is still a rule when they are.
  let term = Term::start(
    "mcp-changed",
    &provider,
    &["--no-session", "--mcp-config", &shell(&config), "--no-tools", "tide"],
  );
  term.wait_for("MCP weather: 2 tools");
  term.wait_for("1 mcp, 2 tools");
  term.submit("grow, then look for rain");
  term.wait_for("Rain in Oslo.");
  term.wait_for("MCP weather: now 3 tools, new: radar, tide, gone: grow");
  term.wait_for("1 mcp, 3 tools");
  // What the last request offered, apart from the calls its history holds.
  let body = provider.bodies().last().cloned().expect("a request");
  let request: serde_json::Value = serde_json::from_str(&body).expect("JSON");
  let offered: Vec<&str> = request["tools"]
    .as_array()
    .expect("tools")
    .iter()
    .filter_map(|tool| tool["function"]["name"].as_str())
    .collect();
  let last = offered
    .iter()
    .map(|name| format!(r#""name":"{name}""#))
    .collect::<String>();
  assert!(last.contains(r#""name":"radar""#), "the new tool is offered: {last}");
  assert!(
    !last.contains(r#""name":"grow""#),
    "the one it took back is not: {last}"
  );
  assert!(
    !last.contains(r#""name":"tide""#),
    "nor one the session refused: {last}"
  );
  assert!(last.contains(r#""name":"weather""#), "and the rest stay: {last}");
  let screen = term.screen();
  let call = screen.find("⚙ radar").expect("called like any other");
  assert!(screen[call..].contains("Oslo"), "and answered: {screen}");
  let _ = std::fs::remove_dir_all(&dir);
}

/// A server may stop in the middle of a tool call to ask the user something.
/// It is shown the same dialog the model's own questions are, one tab per
/// field, and what the user typed is checked against what the server asked for
/// before it is sent.
#[test]
#[cfg(feature = "mcp")]
fn a_server_asks_the_user_in_a_form_and_gets_what_they_answered() {
  if !have_tmux() || !have_python() {
    return;
  }
  let dir = scratch("mcp-elicit");
  let server = dir.join("server.py");
  std::fs::write(&server, MCP_SERVER).expect("a server to run");
  let config = dir.join("mcp.toml");
  std::fs::write(
    &config,
    format!(
      "[booking]\ncommand = \"python3 '{}'\"\ntimeout = 60\ntools = [\"book\"]\n",
      server.display()
    ),
  )
  .expect("a config to read");

  let provider = Provider::start(vec![
    Turn::Call {
      say: "Booking. ",
      tool: "book",
      args: serde_json::json!({}),
    },
    Turn::Say("Booked."),
  ]);
  let term = Term::start(
    "mcp-elicit",
    &provider,
    &["--no-session", "--mcp-config", &shell(&config)],
  );
  term.wait_for("MCP booking: 1 tool");
  term.submit("book me in");

  // Who is asking, what they want, and the first field ready to type in.
  let screen = term.wait_for("booking is asking");
  assert!(screen.contains("How many seats, and where?"), "{screen}");
  assert!(screen.contains("Seats (required)"), "{screen}");

  // Something that is not a number goes as far as the submit tab, and no
  // further: it is said above the form, and nothing is sent.
  term.type_in("many");
  term.type_in("Enter");
  term.wait_for("1. Yes");
  term.type_in("Enter");
  term.wait_for("Review your answers");
  term.type_in("Enter");
  term.wait_for("Seats must be a whole number");

  // Put right where it was typed, it goes.
  term.type_in("Tab");
  term.type_in("C-u");
  term.type_in("3");
  term.type_in("Enter");
  term.type_in("Enter");
  term.wait_for("Review your answers");
  term.type_in("Enter");
  term.wait_for("Booked.");

  // The server was given the answer as the schema has it — a number and a
  // boolean, not the text they were typed and chosen as.
  assert!(provider.sent(r#"\"action\": \"accept\""#), "accepted");
  assert!(provider.sent(r#"\"seats\": 3"#), "a number");
  assert!(provider.sent(r#"\"window\": true"#), "a boolean");
  let _ = std::fs::remove_dir_all(&dir);
}

/// A server's prompts are offered by the `/` popup as `/server:name`, with
/// what they take, and sent as what the server writes out of them: from the
/// arguments typed after the name, or from a form asking for the required
/// ones left out. What the server writes may be a conversation of its own,
/// and it goes to the model as one.
#[test]
#[cfg(feature = "mcp")]
fn a_servers_prompt_is_a_command_sent_as_what_the_server_writes_out() {
  if !have_tmux() || !have_python() {
    return;
  }
  let dir = scratch("mcp-prompts");
  let server = dir.join("server.py");
  std::fs::write(&server, MCP_SERVER).expect("a server to run");
  let config = dir.join("mcp.toml");
  std::fs::write(
    &config,
    format!(
      "[notes]\ncommand = \"python3 '{}' prompts\"\ntimeout = 10\ntools = [\"weather\"]\n",
      server.display()
    ),
  )
  .expect("a config to read");

  // Which turn it is is counted in tool results, and none of these call a
  // tool: every one is answered the same, and counted on screen.
  let provider = Provider::start(vec![Turn::Say("Noted.")]);
  let term = Term::start(
    "mcp-prompts",
    &provider,
    &["--no-session", "--mcp-config", &shell(&config)],
  );
  term.wait_for("MCP notes: 2 prompts");
  let answered = |n: usize| poll(|| (term.screen().matches("Noted.").count() >= n).then(String::new));

  // Offered with the commands, with what it takes and does.
  term.type_in("/notes:rev");
  let screen = term.wait_for("/notes:review");
  assert!(screen.contains("<pr> [focus] Review a change."), "{screen}");
  term.type_in("Tab");
  poll(|| (term.typed() == "/notes:review").then(String::new));

  // Its arguments typed after it: a word, and the rest of the line.
  term.type_in("12 the tests");
  term.type_in("Enter");
  answered(1);
  assert!(provider.sent("Review PR 12, looking at the tests."), "written out");
  let screen = term.screen();
  assert!(
    screen.contains("Review PR 12, looking at the tests."),
    "drawn as sent: {screen}"
  );

  // A required one left out is asked for, and the rest may stay blank.
  term.submit("/notes:review");
  let screen = term.wait_for("pr (required)");
  assert!(screen.contains("Review a change."), "{screen}");
  term.type_in("7");
  term.type_in("Enter");
  term.type_in("Enter");
  term.wait_for("Review your answers");
  term.type_in("Enter");
  answered(2);
  assert!(provider.sent("Review PR 7, looking at everything."), "from the form");

  // A conversation of its own, and a resource carried whole.
  term.submit("/notes:recap");
  answered(3);
  assert!(term.screen().contains("On the parser."), "drawn as the server wrote it");
  let last = provider.bodies().last().cloned().expect("a request");
  let request: serde_json::Value = serde_json::from_str(&last).expect("JSON");
  let said: Vec<(String, String)> = request["messages"]
    .as_array()
    .expect("messages")
    .iter()
    .map(|m| (m["role"].as_str().unwrap_or("").to_string(), m["content"].to_string()))
    .collect();
  let at = said
    .iter()
    .position(|(_, content)| content.contains("Where were we?"))
    .unwrap_or_else(|| panic!("the recap: {said:?}"));
  assert_eq!(said[at + 1].0, "assistant", "{said:?}");
  assert!(said[at + 1].1.contains("On the parser."), "{said:?}");
  assert_eq!(said[at + 2].0, "user", "{said:?}");
  assert!(
    said[at + 2]
      .1
      .contains("[Resource &notes:note://today — text/plain]\\nBuy milk."),
    "{said:?}"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// A server that completes values is asked for them as they are typed: an
/// argument of its prompt, told the ones before it, and a hole in one of its
/// resource templates. Taking one before the last goes on to the next.
#[test]
#[cfg(feature = "mcp")]
fn a_server_completes_its_prompts_arguments_and_its_templates_holes() {
  if !have_tmux() || !have_python() {
    return;
  }
  let dir = scratch("mcp-completes");
  let server = dir.join("server.py");
  std::fs::write(&server, MCP_SERVER).expect("a server to run");
  let config = dir.join("mcp.toml");
  std::fs::write(
    &config,
    format!(
      "[notes]\ncommand = \"python3 '{}' resources prompts completes\"\ntimeout = 10\ntools = [\"weather\"]\n",
      server.display()
    ),
  )
  .expect("a config to read");

  let provider = Provider::start(vec![Turn::Say("Noted.")]);
  let term = Term::start(
    "mcp-completes",
    &provider,
    &["--no-session", "--mcp-config", &shell(&config)],
  );
  term.wait_for("MCP notes: 2 prompts");
  let typed = |want: &str| poll(|| (term.typed() == want).then(String::new));

  // The command taken, what its first argument could be is offered at once.
  term.type_in("/notes:review");
  term.wait_for("Review a change.");
  term.type_in("Tab");
  term.wait_for("123");
  term.type_in("12");
  typed("/notes:review 12");
  // Taken, it goes on to the next, which the server is told the first for.
  term.type_in("Tab");
  typed("/notes:review 12");
  term.wait_for("tests of 12");
  term.type_in("Tab");
  typed("/notes:review 12 tests of 12");
  term.type_in("Enter");
  poll(|| (term.screen().matches("Noted.").count() >= 1).then(String::new));
  assert!(
    provider.sent("Review PR 12, looking at tests of 12."),
    "sent as completed"
  );

  // A template's hole, completed in its token.
  term.type_in("&notes:note://2026-09-2");
  term.wait_for("2026-09-23");
  term.type_in("Down");
  term.type_in("Tab");
  typed("&notes:note://2026-09-23");
  let _ = std::fs::remove_dir_all(&dir);
}

/// A built-in tool stops itself; a server answers with whatever it likes, and
/// the answer is what the context window is spent on. So the reply is cut to
/// the size `read` and `bash` keep to — and cut once, before the transcript
/// and the model are told, so the terminal shows the copy the model was given.
#[test]
#[cfg(feature = "mcp")]
fn a_reply_longer_than_a_tool_may_answer_with_is_cut_to_its_head() {
  if !have_tmux() || !have_python() {
    return;
  }
  let dir = scratch("mcp-long");
  let server = dir.join("server.py");
  std::fs::write(&server, MCP_SERVER).expect("a server to run");
  let config = dir.join("mcp.toml");
  std::fs::write(
    &config,
    format!(
      "[deluge]\ncommand = \"python3 '{}'\"\ntimeout = 10\ntools = [\"flood\"]\n",
      server.display()
    ),
  )
  .expect("a config to read");

  let provider = Provider::start(vec![
    Turn::Call {
      say: "Bracing. ",
      tool: "flood",
      args: serde_json::json!({ "lines": 3000 }),
    },
    Turn::Say("That was a lot."),
  ]);
  let term = Term::start(
    "mcp-long",
    &provider,
    &["--no-session", "--mcp-config", &shell(&config)],
  );
  term.wait_for("MCP deluge: 1 tool");
  term.submit("open the floodgates");
  term.wait_for("That was a lot.");

  // The model was given the head of it and a note saying where the rest is.
  let note = "[Showing lines 1-2000 of 3000. Full output: ";
  let request = provider.request(note);
  assert!(
    request.contains("line 1\\nline 2\\n"),
    "the front of it: {request:.400}"
  );
  assert!(!request.contains("line 2001"), "and not a line past where it was cut");

  // The whole of it is on disk, for a model that wants more than the head.
  let path = request
    .split_once(note)
    .and_then(|(_, rest)| rest.split_once(']'))
    .map(|(path, _)| path.to_string())
    .expect("the note says where the rest went");
  let full = std::fs::read_to_string(&path).expect("the whole reply, written down");
  assert_eq!(full.lines().count(), 3000);
  assert!(full.ends_with("line 3000"));
  let _ = std::fs::remove_file(&path);

  // And the transcript shows that same cut copy, not the one that arrived.
  let screen = term.screen();
  assert!(screen.contains("line 1"), "what it said, under its call:\n{screen}");
  assert!(!screen.contains("line 2001"), "nothing past the cut:\n{screen}");
  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_session_can_be_held_to_some_of_its_tools() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Echo]);
  let term = Term::start(
    "read-only",
    &provider,
    &["--no-session", "--no-tools", "write,edit,bash"],
  );
  // Asking is not changing anything, so a read-only session keeps it.
  let screen = term.wait_for("Tools this session: read, ask.");
  // And nothing about MCP in the footer of a session that has no servers.
  assert!(!screen.contains("mcp"), "no servers, nothing said:\n{screen}");
  term.submit("what is here");
  term.wait_for("Answer to what is here.");

  let request = provider.request("what is here");
  assert!(
    request.contains(r#""name":"read""#) && request.contains(r#""name":"ask""#),
    "what is left is offered: {request}"
  );
  for gone in [r#""name":"write""#, r#""name":"edit""#, r#""name":"bash""#] {
    assert!(!request.contains(gone), "{gone} is not offered: {request}");
  }
  // And the model is not told about what it cannot call, which it would
  // otherwise try and report being refused.
  assert!(
    request.contains("- read: Read file contents") && !request.contains("Use bash for"),
    "the prompt speaks only for the tools there are: {request}"
  );
  // A name nothing answers to is worth saying, since it leaves nothing out.
  let term = Term::start("typo", &provider, &["--no-session", "--no-tools", "bahs"]);
  term.wait_for("No tool named bahs");
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

#[test]
fn a_question_the_model_asked_is_answered_in_a_dialog_and_read_back() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "I need to know two things.",
      tool: "ask",
      args: serde_json::json!({
        "questions": [
          {
            "question": "Which cache?",
            "header": "Cache",
            "options": [
              { "label": "In memory", "description": "fast, lost on restart" },
              { "label": "On disk", "description": "survives a restart" },
            ],
          },
          {
            "question": "Which tests?",
            "header": "Tests",
            "multiSelect": true,
            "options": [
              { "label": "Unit", "description": "the functions on their own" },
              { "label": "Integration", "description": "the pieces together" },
            ],
          },
        ],
      }),
    },
    Turn::Say("Understood."),
  ]);
  let term = Term::start("asking", &provider, &["--no-session"]);
  assert!(!term.rang(), "nothing has been asked yet");
  term.submit("add caching");

  // The first question is up, with its options and the row to type in.
  // Waiting on a row rather than on the question, which the line announcing
  // the call already says.
  let screen = term.wait_for("› 1. In memory");
  // And the terminal was rung, since a run nobody is watching has just
  // stopped and is waiting on an answer.
  assert!(term.rang(), "the bell rang when the question went up");
  assert!(screen.contains("survives a restart"), "and what each means:\n{screen}");
  assert!(screen.contains("3. Type something."), "{screen}");
  assert!(screen.contains("Tab to switch questions"), "{screen}");

  // Answering it moves on to the next, which has boxes rather than a choice.
  term.type_in("Down");
  term.type_in("Enter");
  let screen = term.wait_for("[ ] Unit");
  assert!(screen.contains("■ Cache"), "the answered tab is filled in:\n{screen}");

  // Space ticks, and the Next row commits — which lands on the review.
  term.type_in("Space");
  term.type_in("Down");
  term.type_in("Down");
  term.type_in("Down");
  term.type_in("Enter");
  let screen = term.wait_for("Review your answers");
  assert!(screen.contains("→ On disk"), "what was answered:\n{screen}");
  assert!(screen.contains("→ Unit"), "{screen}");
  assert!(screen.contains("Ready to submit"), "nothing left blank:\n{screen}");

  term.type_in("Enter");
  let screen = term.wait_for("Understood.");
  // The transcript keeps what was asked and what was answered, with the
  // dialog gone.
  assert!(screen.contains("⚙ ask Which cache? (+1 more)"), "{screen}");
  assert!(
    screen.contains("User has answered your questions"),
    "the answer is under the call:\n{screen}"
  );
  // What the model was told is the extension's own envelope.
  assert!(
    provider.sent(
      "User has answered your questions: \\\"Which cache?\\\"=\\\"On disk\\\". \
       \\\"Which tests?\\\"=\\\"Unit\\\". You can now continue with the user's answers in mind."
    ),
    "the answers went back: {}",
    provider.request("Which cache?")
  );
}

#[test]
fn a_question_nobody_answers_is_a_decline_and_the_run_carries_on() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "ask",
      args: serde_json::json!({
        "questions": [{
          "question": "Which cache?",
          "header": "Cache",
          "options": [
            { "label": "In memory", "description": "fast, lost on restart" },
            { "label": "On disk", "description": "survives a restart" },
          ],
        }],
      }),
    },
    Turn::Say("I will guess then."),
  ]);
  let term = Term::start("declining", &provider, &["--no-session", "--no-bell"]);
  term.submit("add caching");
  let screen = term.wait_for("› 1. In memory");
  // One question has no tabs to switch between and nothing to review.
  assert!(!screen.contains("Tab to switch questions"), "{screen}");
  assert!(
    screen.contains("Cache"),
    "the header, where there is no tab strip:\n{screen}"
  );

  // Started with the bell turned off, the question goes up in silence.
  assert!(!term.rang(), "--no-bell keeps it quiet:\n{screen}");

  term.type_in("Escape");
  term.wait_for("I will guess then.");
  assert!(
    provider.sent("User declined to answer questions"),
    "the decline went back: {}",
    provider.request("Which cache?")
  );
}

#[test]
fn an_answer_of_ones_own_is_typed_into_the_row_that_offers_it() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![
    Turn::Call {
      say: "",
      tool: "ask",
      args: serde_json::json!({
        "questions": [{
          "question": "Which cache?",
          "header": "Cache",
          "options": [
            { "label": "In memory", "description": "fast, lost on restart" },
            { "label": "On disk", "description": "survives a restart" },
          ],
        }],
      }),
    },
    Turn::Say("Redis it is."),
  ]);
  let term = Term::start("typing", &provider, &["--no-session"]);
  term.submit("add caching");
  term.wait_for("› 1. In memory");

  // The row above the first is the last one, which is the one typed into.
  term.type_in("Up");
  term.type_in("redis");
  let screen = term.wait_for("redis█");
  assert!(
    screen.contains("Alt+Enter for newline"),
    "the hint follows the keys:\n{screen}"
  );

  term.type_in("Enter");
  term.wait_for("Redis it is.");
  assert!(
    provider.sent("\\\"Which cache?\\\"=\\\"redis\\\""),
    "what was typed went back: {}",
    provider.request("Which cache?")
  );
}

/// Wait until the provider has taken `n` requests, and hand back every one.
fn wait_bodies(provider: &Provider, n: usize) -> Vec<String> {
  let deadline = Instant::now() + Duration::from_secs(15);
  while Instant::now() < deadline {
    let bodies = provider.bodies();
    if bodies.len() >= n {
      return bodies;
    }
    std::thread::sleep(Duration::from_millis(50));
  }
  panic!("waited for {n} requests, only {} arrived", provider.bodies().len());
}

/// A reopened session asks the model exactly what an unbroken one would.
///
/// Every server worth talking to keeps the prompt it has already read and
/// starts again where the next one parts company with it. So a reload that
/// changes any byte ahead of the new message — the system prompt, the tools,
/// an id inside the transcript — is the whole conversation read again before
/// a token comes back, however faithfully the screen was redrawn.
///
/// The check is the request itself, against the one the same conversation
/// sends with nothing closed in the middle of it: same prompt, same tools,
/// same transcript, byte for byte. What goes into it is everything a reload
/// has to rebuild rather than remember — an image a tool read, a call and its
/// answer, what the model thought, and a turn that was interrupted and kept.
#[test]
fn a_reopened_session_asks_for_byte_for_byte_what_it_would_have_asked_for() {
  if !have_tmux() {
    return;
  }
  let script = vec![
    Turn::Call {
      say: "",
      tool: "read",
      args: serde_json::json!({ "path": "red.png" }),
    },
    Turn::Call {
      say: "Looking. ",
      tool: "bash",
      args: serde_json::json!({ "command": "sleep 60" }),
    },
    Turn::Think {
      thought: "A red one.",
      say: "A red square.",
    },
  ];

  // The same conversation twice, once with the session closed and reopened
  // partway through it, and what each asked the model for afterwards.
  let asked = |reopened: bool| -> String {
    let provider = Provider::start(script.clone());
    let term = Term::start("reopened-request", &provider, &[]);
    let red = image::ImageBuffer::from_pixel(16, 16, image::Rgb([220u8, 20, 60]));
    image::DynamicImage::ImageRgb8(red)
      .save(term.dir.join("red.png"))
      .expect("an image to read");
    // A turn the user stopped partway through: its work is kept, and it is
    // part of the conversation from here on.
    term.submit("look at red.png");
    term.wait_for("⚙ bash sleep 60");
    term.type_in("Escape");
    term.wait_for("Aborted.");
    term.settle();
    term.submit("and again");
    term.wait_for("A red square.");
    term.settle();

    let so_far = provider.bodies().len();
    let term = match reopened {
      true => {
        let term = term.reopen(&provider, &["-c"]);
        term.wait_for("Resumed session");
        term
      }
      false => term,
    };
    term.submit("carry on");
    let asked = wait_bodies(&provider, so_far + 1).last().expect("a request").clone();
    term.settle();
    asked
  };

  same_request(&asked(false), &asked(true));
}

/// The same, for a session a compaction has been through.
///
/// A checkpoint is the one thing on disk that is not a message: the turns
/// above it are still in the file, and what the model is given in their place
/// has to be built back out of the checkpoint alone. Getting that wrong is
/// quiet — the screen redraws correctly either way — and costs either the
/// whole summarized history read again, or the summary and the history both.
#[test]
fn a_reopened_compacted_session_asks_for_byte_for_byte_what_it_would_have_asked_for() {
  if !have_tmux() {
    return;
  }
  // A tool call in the first turn, which is the turn the compaction takes
  // away: a walk that does not stop at the checkpoint puts it back.
  let script = vec![
    Turn::Call {
      say: "",
      tool: "bash",
      args: serde_json::json!({ "command": "echo kumquat" }),
    },
    Turn::Echo,
  ];

  let asked = |reopened: bool| -> String {
    let provider = Provider::start(script.clone());
    let term = Term::start("reopened-compacted", &provider, &[]);
    term.submit("remember the kumquat");
    term.wait_for("Answer to remember the kumquat.");
    term.submit("and the pomelo");
    term.wait_for("Answer to and the pomelo.");
    term.submit("/compact");
    term.wait_for("into a summary");
    term.settle();

    let so_far = provider.bodies().len();
    let term = match reopened {
      true => {
        let term = term.reopen(&provider, &["-c"]);
        term.wait_for("Resumed session");
        term
      }
      false => term,
    };
    term.submit("carry on");
    let asked = wait_bodies(&provider, so_far + 1).last().expect("a request").clone();
    term.settle();
    asked
  };

  let unbroken = asked(false);
  let reopened = asked(true);
  // Both ask for the summary in place of what it stands for — a comparison
  // of two requests that were each wrong the same way would pass otherwise.
  assert!(
    reopened.contains("Fruit was discussed.") && !reopened.contains("echo kumquat"),
    "the checkpoint stands where the turns it summarized were: {reopened}"
  );
  same_request(&unbroken, &reopened);
}

/// Assert two requests are the same request.
///
/// Only the window around the parting is worth printing: the whole of either
/// is the system prompt and every tool schema again. Where the two part
/// company is a byte, which need not be where a character starts.
fn same_request(unbroken: &str, reopened: &str) {
  let common = unbroken
    .as_bytes()
    .iter()
    .zip(reopened.as_bytes())
    .take_while(|(one, other)| one == other)
    .count();
  let window = |text: &str| {
    let boundary = |mut at: usize| {
      while !text.is_char_boundary(at) {
        at -= 1;
      }
      at
    };
    let from = boundary(common.min(text.len()));
    text[from..boundary((from + 200).min(text.len()))].to_string()
  };
  assert!(
    unbroken == reopened,
    "the reopened session asks for something else from byte {common}:\n  unbroken: {}\n  reopened: {}",
    window(unbroken),
    window(reopened),
  );
}

#[test]
fn a_drag_over_the_transcript_selects_it_and_copies_what_it_covered() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Say("Kiwi and quince.\n\nAlso persimmon.")]);
  let term = Term::start("select", &provider, &["--no-session"]);
  term.submit("fruit?");
  term.wait_for("Also persimmon.");

  // A drag across one word takes that word, ends included: at a cell's own
  // resolution, a drag over a word means the word.
  let (col, row) = term.cell_of("quince");
  term.hold_through(&[(col, row), (col + 5, row)]);
  // While the button is down it is on screen as selected, which is the
  // terminal's own reverse video rather than a colour of ours.
  let coloured = term.coloured();
  let shown = coloured
    .lines()
    .find(|line| line.contains("quince"))
    .expect("the line it is on");
  assert!(
    shown.contains("\u{1b}[7mquince\u{1b}[0m"),
    "the word is drawn reversed and the rest of the line is not: {shown:?}"
  );
  // Letting go copies it and lets go of it: the highlight was the drag, and
  // the drag is over.
  term.let_go((col + 5, row));
  assert_eq!(term.clipboard(), "quince");
  let coloured = term.coloured();
  let shown = coloured
    .lines()
    .find(|line| line.contains("quince"))
    .expect("the line it is on");
  // Reverse video where the cursor is, in the input box, is not the
  // transcript's — the line the selection was on carries none of it.
  assert!(!shown.contains("\u{1b}[7m"), "nothing is left selected: {shown:?}");

  // A drag over several lines takes all of them, in the shape they are drawn
  // in — including the blank line between two paragraphs.
  let (start, first) = term.cell_of("Kiwi");
  let (end, last) = term.cell_of("Also persimmon.");
  term.drag((start, first), (end + 14, last));
  assert_eq!(term.clipboard(), "Kiwi and quince.\n\nAlso persimmon.");

  // A click is not a selection, and copies nothing over what was copied.
  term.drag((start, first), (start, first));
  assert_eq!(term.clipboard(), "Kiwi and quince.\n\nAlso persimmon.");
}

#[test]
fn a_selection_dragged_off_the_top_scrolls_the_transcript_and_keeps_going() {
  if !have_tmux() {
    return;
  }
  // More transcript than screen, so there is something above to reach for.
  let long: &'static str = Box::leak(
    (1..=40)
      .map(|i| format!("- item {i:02}\n"))
      .collect::<String>()
      .into_boxed_str(),
  );
  let provider = Provider::start(vec![Turn::Say(long)]);
  let term = Term::start("autoscroll", &provider, &["--no-session"]);
  term.submit("list?");
  term.wait_for("item 40");

  // Notches compose, rather than each one scrolling from where the screen
  // still is: three of them are three notches back, drawn or not.
  for _ in 0..3 {
    term.wheel_up((40, 10));
  }
  term.settle();
  let footer = term.screen();
  assert!(footer.contains("↑ 9 lines"), "three notches back:\n{footer}");

  // A drag that reaches the top row and stays there keeps scrolling, so the
  // selection runs on above what was on screen when it started.
  let (col, row) = term.cell_of("item 30");
  let mut cells = vec![(col + 8, row)];
  cells.extend((0..6).map(|_| (col, 1)));
  term.drag_through(&cells);
  let copied = term.clipboard();
  assert!(
    copied.ends_with("item 30"),
    "the selection still ends where it started: {copied:?}"
  );
  assert!(
    copied.lines().count() > 6,
    "it ran further than the rows it was dragged over: {copied:?}"
  );
}

#[test]
fn an_image_attached_to_a_prompt_is_sent_with_it_and_drawn_under_it() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Say("A green rectangle.")]);
  let term = Term::start("attach", &provider, &["--no-session"]);
  // Sixteen pixels down is eight lines of half-blocks, sixteen across.
  let green = image::ImageBuffer::from_pixel(16, 16, image::Rgb([20u8, 200, 90]));
  image::DynamicImage::ImageRgb8(green)
    .save(term.dir.join("shot.png"))
    .expect("an image to attach");

  // The border says what the token found, before anything is sent.
  term.type_in("what is @shot.png");
  term.wait_for("▣ shot.png 16×16");

  term.type_in("Enter");
  term.wait_for("A green rectangle.");

  // The prompt went as typed, token and all, and the image went with it as
  // a part of its own rather than as anything written into the text.
  let request = provider.request("what is @shot.png");
  assert!(
    request.contains("image_url") && request.contains("data:image/png;base64,"),
    "the image travels as its own content part:\n{request}"
  );
  assert!(
    request.contains("[Attached @shot.png"),
    "and behind a note naming it:\n{request}"
  );

  // And it is drawn under the prompt that attached it.
  let screen = term.screen();
  let drawn: Vec<&str> = screen.lines().filter(|line| line.contains('▄')).collect();
  assert_eq!(drawn.len(), 8, "the image is drawn as half-blocks:\n{screen}");
  assert!(
    drawn.iter().all(|line| line.matches('▄').count() == 16),
    "each line is the image's own width:\n{screen}"
  );
}

#[test]
fn the_at_popup_completes_a_path_and_a_full_path_attaches_too() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Say("Seen.")]);
  let term = Term::start("attach-popup", &provider, &["--no-session"]);
  std::fs::create_dir_all(term.dir.join("shots")).expect("a directory");
  let red = image::ImageBuffer::from_pixel(8, 8, image::Rgb([220u8, 20, 60]));
  image::DynamicImage::ImageRgb8(red)
    .save(term.dir.join("shots/red.png"))
    .expect("an image to complete to");

  // A bare `@` lists the working directory at once, as `/` lists commands.
  term.type_in("look at @");
  term.wait_for("shots/");
  // The popup lists what a half-typed token could mean, sizes and all.
  term.type_in("shots/r");
  term.wait_for("red.png");
  term.wait_for("8×8");
  // Tab takes the row, and the border then says it resolved.
  term.type_in("Tab");
  term.wait_for("▣ red.png 8×8");
  term.type_in("Enter");
  term.wait_for("Seen.");
  assert!(
    provider.sent("@shots/red.png"),
    "the completed token is what was sent:\n{:?}",
    provider.bodies()
  );
}

/// `/model` lists what the provider has and runs the session on what is
/// picked from it, mid-conversation.
#[test]
fn a_model_picked_from_the_list_is_what_the_session_runs_on() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Say("Picked, then answered.")]);
  let term = Term::start("model-picker", &provider, &["--no-session"]);

  term.submit("/model");
  term.wait_for("Esc cancel");
  let (rows, _) = term.overlay();
  assert!(
    rows.iter().any(|row| row.contains("mock-mini")),
    "what the provider listed is on it: {rows:?}"
  );

  term.choose("mock-mini");
  term.wait_for("Model mock-mini");
  term.submit("go");
  let screen = term.wait_for("Picked, then answered.");
  assert!(
    screen.contains("openai/mock-mini"),
    "the footer says what it settled on:\n{screen}"
  );
}

/// `/model` takes a name the provider never listed: the list is what it
/// admits to, not the whole of what it answers to.
#[test]
fn a_model_can_be_named_outright() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Say("Answered anyway.")]);
  let term = Term::start("model-named", &provider, &["--no-session"]);

  term.submit("/model unlisted-model");
  term.wait_for("Model unlisted-model");
  term.submit("go");
  let screen = term.wait_for("Answered anyway.");
  assert!(
    screen.contains("openai/unlisted-model"),
    "named rather than picked, and used all the same:\n{screen}"
  );
}

/// The picker is typed at: letters narrow the list to what they match, and
/// `Enter` takes what is left standing under the cursor.
#[test]
fn typing_at_the_model_picker_narrows_it_to_what_was_typed() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Say("Ran on the filtered one.")]);
  let term = Term::start("model-filter", &provider, &["--no-session"]);
  // The rows a list has drawn something on: the box is as tall as the
  // screen whatever is in it, and the blank ones are not models.
  let listed = |rows: &[String]| {
    rows
      .iter()
      .filter(|row| !row.trim_matches(|c| c == '│' || c == ' ').is_empty())
      .count()
  };
  term.submit("/model");
  term.wait_for("Esc cancel");
  let (rows, _) = term.overlay();
  assert_eq!(listed(&rows), 3, "all three to start with: {rows:?}");

  // A fuzzy match, as everywhere else in fa: "omd" is o-ther-m-o-d-el.
  term.type_in("omd");
  term.wait_for_query("omd");
  let (rows, on) = term.overlay();
  assert_eq!(listed(&rows), 1, "only what matches is left: {rows:?}");
  assert!(rows[on].contains("other-model"), "and it is the one meant: {rows:?}");

  // The letters it matched are picked out where they are, as the `/` popup
  // picks out its own — so the row is drawn letter by letter, and what is
  // left unhighlighted is what sits between them.
  let coloured = term.coloured();
  let row = coloured.lines().find(|line| line.contains("ther-")).expect("the row");
  // The terminal's own cyan, indexed as everything else here is, and the
  // underline that goes with it.
  assert!(
    row.contains("\u{1b}[4m") && row.contains("\u{1b}[38;5;6m"),
    "matched letters are cyan and underlined: {row:?}"
  );

  // Backspace widens it again, and a query nothing matches says so rather
  // than leaving an empty box.
  term.type_in("BSpace");
  term.type_in("BSpace");
  term.type_in("BSpace");
  term.type_in("zzz");
  term.wait_for("No model matches.");
  term.type_in("BSpace");
  term.type_in("BSpace");
  term.type_in("BSpace");

  // `q` is a letter here, not the key that closes the other lists.
  term.type_in("mini");
  term.wait_for_query("mini");
  term.type_in("Enter");
  term.wait_for("Model mock-mini");
  term.submit("go");
  let screen = term.wait_for("Ran on the filtered one.");
  assert!(
    screen.contains("openai/mock-mini"),
    "the one the filter left is what it runs on:\n{screen}"
  );
}

/// The model named on the command line is the whole of what a session needs,
/// so nothing is asked of the provider until `/model` asks it: a session that
/// never opens the picker never lists anything.
#[test]
fn nothing_is_listed_until_model_asks_for_it() {
  if !have_tmux() {
    return;
  }
  let provider = Provider::start(vec![Turn::Say("Answered.")]);
  let term = Term::start("model-unasked", &provider, &["--no-session"]);
  term.submit("go");
  let screen = term.wait_for("Answered.");
  assert!(
    screen.contains("openai/mock"),
    "it runs on what it was told to: {screen}"
  );
  assert_eq!(
    provider.listings(),
    0,
    "nothing was asked of the provider that the command line had not already answered"
  );

  term.submit("/model");
  term.wait_for("Esc cancel");
  assert_eq!(provider.listings(), 1, "asked once, when asked to");
}
