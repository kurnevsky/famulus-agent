//! A provider to test the agent against: OpenAI's chat completions and
//! Gemini's `generateContent`, answered from a script on a port of its own.
//!
//! What to say is read off the request rather than counted, so a test can
//! hand it any history it likes:
//!
//! - a conversation that already holds a tool's answer gets a thought and a
//!   reply about it;
//! - `run it` gets a `bash` call that echoes `hello from mock`;
//! - `please fail` gets a `read` of a file that is not there;
//! - `image <path>` gets a `read` of that path;
//! - anything else gets a greeting counting the messages it came with;
//! - a request that is not streamed is the summarizer, and gets a summary.
//!
//! It refuses what the real provider would: an image inside a tool message,
//! which OpenAI does not take.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

use serde_json::{Value, json};

/// What every summary says, for a test to find.
pub const SUMMARY: &str = "## Goal\nMock summary of the conversation.";

/// What the greeting costs: the context it reports is the two together.
const PROMPT_TOKENS: u64 = 12;
const COMPLETION_TOKENS: u64 = 7;

pub struct Mock {
  port: u16,
}

impl Mock {
  pub fn start() -> Self {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a port to listen on");
    let port = listener.local_addr().expect("an address").port();
    std::thread::spawn(move || {
      for stream in listener.incoming().flatten() {
        std::thread::spawn(move || {
          let _ = serve(stream);
        });
      }
    });
    Self { port }
  }

  /// Where an OpenAI-compatible client is pointed.
  pub fn openai_url(&self) -> String {
    format!("http://127.0.0.1:{}/v1", self.port)
  }

  /// Where a Gemini client is pointed: it adds `/v1beta` itself.
  pub fn gemini_url(&self) -> String {
    format!("http://127.0.0.1:{}", self.port)
  }
}

fn serve(mut stream: TcpStream) -> std::io::Result<()> {
  let mut reader = BufReader::new(stream.try_clone()?);
  let mut request_line = String::new();
  reader.read_line(&mut request_line)?;
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
  reader.read_exact(&mut body)?;
  let body: Value = serde_json::from_slice(&body).unwrap_or_default();
  let path = request_line.split_whitespace().nth(1).unwrap_or_default();

  if path.contains(":streamGenerateContent") {
    let reply = gemini(&body);
    return stream_events(&mut stream, std::iter::once(reply), false);
  }
  if path.contains(":generateContent") {
    return whole(&mut stream, 200, &gemini_text(SUMMARY));
  }
  if path.ends_with("/chat/completions") {
    return openai(&mut stream, &body);
  }
  whole(
    &mut stream,
    404,
    &json!({ "error": { "message": format!("no such path: {path}") } }),
  )
}

// ------------------------------------------------------------ openai

fn openai(stream: &mut TcpStream, body: &Value) -> std::io::Result<()> {
  let messages = body["messages"].as_array().cloned().unwrap_or_default();
  let image_in_tool = messages.iter().any(|message| {
    message["role"] == "tool"
      && message["content"]
        .as_array()
        .is_some_and(|parts| parts.iter().any(|part| part["type"] == "image_url"))
  });
  if image_in_tool {
    let error = json!({ "error": { "message": "Image URLs are only allowed for messages with role 'user'" } });
    return whole(stream, 400, &error);
  }

  if body["stream"] != true {
    let reply = json!({
      "id": "1", "object": "chat.completion", "created": 0, "model": "mock",
      "choices": [{ "index": 0, "finish_reason": "stop",
                    "message": { "role": "assistant", "content": SUMMARY } }],
      "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
    });
    return whole(stream, 200, &reply);
  }

  let chunk = |delta: Value, finish: Option<&str>| {
    json!({
      "id": "1", "object": "chat.completion.chunk", "created": 0, "model": "mock",
      "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    })
  };
  let usage = |prompt: u64, completion: u64| {
    json!({
      "id": "1", "object": "chat.completion.chunk", "created": 0, "model": "mock", "choices": [],
      "usage": { "prompt_tokens": prompt, "completion_tokens": completion,
                 "total_tokens": prompt + completion },
    })
  };
  let call = |name: &str, args: Value| {
    vec![
      chunk(
        json!({ "role": "assistant", "tool_calls": [{
          "index": 0, "id": "call_1", "type": "function",
          "function": { "name": name, "arguments": args.to_string() },
        }] }),
        None,
      ),
      chunk(json!({}), Some("tool_calls")),
      usage(1, 1),
    ]
  };

  let events = if messages.iter().any(|message| message["role"] == "tool") {
    vec![
      chunk(
        json!({ "role": "assistant", "reasoning_content": "The tool has answered." }),
        None,
      ),
      chunk(json!({ "content": "That is what it said." }), None),
      chunk(json!({}), Some("stop")),
      usage(1, 1),
    ]
  } else {
    let asked = last_user_text(&messages);
    if asked == "run it" {
      call("bash", json!({ "command": "echo hello from mock" }))
    } else if asked == "please fail" {
      call("read", json!({ "path": "/nonexistent/file" }))
    } else if let Some(path) = asked.strip_prefix("image ") {
      call("read", json!({ "path": path }))
    } else {
      let greeting = format!("Hello there! I see {} messages in history.", messages.len());
      vec![
        chunk(json!({ "role": "assistant", "content": greeting }), None),
        chunk(json!({}), Some("stop")),
        usage(PROMPT_TOKENS, COMPLETION_TOKENS),
      ]
    }
  };
  stream_events(stream, events, true)
}

/// What the user said last, whether it came as a string or as parts.
fn last_user_text(messages: &[Value]) -> String {
  let Some(message) = messages.iter().rfind(|message| message["role"] == "user") else {
    return String::new();
  };
  match &message["content"] {
    Value::String(text) => text.clone(),
    content => content
      .as_array()
      .into_iter()
      .flatten()
      .filter_map(|part| part["text"].as_str())
      .collect::<Vec<_>>()
      .join(" "),
  }
}

// ------------------------------------------------------------ gemini

/// One streamed reply: a `read` for `image <path>`, and once a function has
/// answered, what the answer carried — which is how a test sees whether the
/// image travelled inside the `functionResponse` rather than beside it.
fn gemini(body: &Value) -> Value {
  let parts: Vec<&Value> = body["contents"]
    .as_array()
    .into_iter()
    .flatten()
    .flat_map(|content| content["parts"].as_array().into_iter().flatten())
    .collect();
  if let Some(response) = parts.iter().rev().find_map(|part| part.get("functionResponse")) {
    let inline = response["parts"]
      .as_array()
      .into_iter()
      .flatten()
      .find_map(|part| part["inlineData"]["mimeType"].as_str());
    return gemini_text(&match inline {
      Some(mime) => format!("inline {mime} in functionResponse"),
      None => "no image in functionResponse".to_string(),
    });
  }
  let asked = parts
    .iter()
    .rev()
    .find_map(|part| part["text"].as_str())
    .unwrap_or_default();
  match asked.strip_prefix("image ") {
    Some(path) => gemini_reply(json!({ "functionCall": { "id": "call_1", "name": "read", "args": { "path": path } } })),
    None => gemini_text("Hello there!"),
  }
}

fn gemini_text(text: &str) -> Value {
  gemini_reply(json!({ "text": text }))
}

fn gemini_reply(part: Value) -> Value {
  json!({
    "candidates": [{ "index": 0, "finishReason": "STOP",
                     "content": { "role": "model", "parts": [part] } }],
    "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2 },
    "modelVersion": "mock",
  })
}

// ------------------------------------------------------------ http

fn whole(stream: &mut TcpStream, status: u16, body: &Value) -> std::io::Result<()> {
  let body = body.to_string();
  let reason = match status {
    200 => "OK",
    400 => "Bad Request",
    _ => "Not Found",
  };
  stream.write_all(
    format!(
      "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
      body.len()
    )
    .as_bytes(),
  )?;
  stream.flush()
}

/// An event stream, then the connection closed. OpenAI marks the end with
/// `[DONE]` first; Gemini's end is the end of the file.
fn stream_events(stream: &mut TcpStream, events: impl IntoIterator<Item = Value>, done: bool) -> std::io::Result<()> {
  stream.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n")?;
  for event in events {
    stream.write_all(format!("data: {event}\n\n").as_bytes())?;
  }
  if done {
    stream.write_all(b"data: [DONE]\n\n")?;
  }
  stream.flush()
}
