//! Calling a tool an MCP server offers, and hearing how it is getting on.
//!
//! Every request to a server carries a progress token, and a server doing
//! something long may send notes against it as it goes: so far, out of how
//! much, and what it is doing. A call listens for the notes sent against its
//! own token and shows the newest one under the call's line, the way a
//! command's output is shown while it runs. The answer then takes its place.
//!
//! A note is also word that the server is still at it: the time a call may
//! take is counted from the last thing heard from the server, not from when
//! the call was made, so a long job that keeps saying how far along it is
//! runs to the end.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rig_agent::tool::{DynamicTool, ToolContext, ToolExecutionError, ToolOutput};
use rig_core::message::{ImageMediaType, MimeType, ToolResultContent};
use rmcp::ServiceError;
use rmcp::model::{
  CallToolRequest, CallToolRequestParams, CallToolResult, ClientRequest, ContentBlock, ProgressNotificationParam,
  ProgressToken, ResourceContents, ServerResult,
};
use rmcp::service::{PeerRequestOptions, ServerSink};

use crate::tools::Output;

// ---------------------------------------------------------------- progress

/// How many notes to hold for tokens no call listens to yet. A server may
/// send one against any request it is given — a listing, or a call that has
/// already been answered — and nobody ever comes for those, so they are
/// thrown out together once there are this many.
const UNCLAIMED: usize = 32;

/// The progress notes one server sends, handed to the calls they are about.
///
/// The token a request carries is only known once it has gone out, so a
/// server quick to say it has started can be heard from before the call is
/// listening. The newest such note is held for the call to find.
#[derive(Clone, Default)]
pub struct Progress(Arc<Mutex<HashMap<ProgressToken, Heard>>>);

enum Heard {
  /// A note nobody is listening for yet.
  Early(ProgressNotificationParam),
  /// A call listening, and how far along the last note it was shown said it
  /// was.
  Listening { sink: Output, shown: Option<f64> },
}

impl Progress {
  /// A note the server sent.
  pub fn heard(&self, note: ProgressNotificationParam) {
    let mut heard = self.lock();
    match heard.get_mut(&note.progress_token) {
      Some(Heard::Listening { sink, shown }) => show(sink, shown, &note),
      Some(early @ Heard::Early(_)) => *early = Heard::Early(note),
      None => {
        if heard.values().filter(|h| matches!(h, Heard::Early(_))).count() >= UNCLAIMED {
          heard.retain(|_, h| matches!(h, Heard::Listening { .. }));
        }
        heard.insert(note.progress_token.clone(), Heard::Early(note));
      }
    }
  }

  /// Show the notes sent against `token` on `sink`, for as long as what
  /// this returns is kept — starting with one already heard, if there is.
  pub fn listen(&self, token: ProgressToken, sink: Output) -> Listening {
    let mut heard = self.lock();
    let mut shown = None;
    if let Some(Heard::Early(note)) = heard.remove(&token) {
      show(&sink, &mut shown, &note);
    }
    heard.insert(token.clone(), Heard::Listening { sink, shown });
    Listening {
      progress: self.clone(),
      token,
    }
  }

  fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<ProgressToken, Heard>> {
    // Nothing holding it panics; a poisoned lock is still the map it was.
    self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
  }
}

/// A call listening for its notes; dropped once it is answered.
pub struct Listening {
  progress: Progress,
  token: ProgressToken,
}

impl Drop for Listening {
  fn drop(&mut self) {
    self.progress.lock().remove(&self.token);
  }
}

/// Show `note` on `sink`, unless it is behind one already shown. Progress
/// only goes up, but notes are handled as they come in, each on its own, so
/// two sent close together can arrive the other way round — and the older
/// one would be the call going backwards.
fn show(sink: &Output, shown: &mut Option<f64>, note: &ProgressNotificationParam) {
  if shown.is_some_and(|shown| note.progress < shown) {
    return;
  }
  *shown = Some(note.progress);
  (sink.0)(said(note));
}

/// A note as one line: how far along, and what the server says it is doing.
///
/// A total makes a percentage, and a count as well when both are whole
/// numbers — `3/10 (30%)` — but not out of a hundred, which is the
/// percentage again. Without a total the amount means little on its own, so
/// only the message is shown, when there is one.
fn said(note: &ProgressNotificationParam) -> String {
  let message = note
    .message
    .as_deref()
    .map(crate::resources::one_line)
    .filter(|m| !m.is_empty());
  let amount = match note.total {
    Some(total) if total > 0.0 => {
      let percent = (note.progress / total * 100.0).round();
      match note.progress.fract() == 0.0 && total.fract() == 0.0 && total != 100.0 {
        true => format!("{}/{total} ({percent}%)", note.progress),
        false => format!("{percent}%"),
      }
    }
    _ if message.is_some() => String::new(),
    _ => note.progress.to_string(),
  };
  match (amount.is_empty(), message) {
    (false, Some(message)) => format!("{amount}: {message}"),
    (true, Some(message)) => message,
    (_, None) => amount,
  }
}

// ---------------------------------------------------------------- the tool

/// One of a server's tools, as the agent calls it: waited on for up to
/// `timeout` of silence from the server, with what it says of its progress
/// shown under the call's line.
pub fn tool(tool: &rmcp::model::Tool, peer: ServerSink, timeout: Option<Duration>, progress: Progress) -> DynamicTool {
  let name = tool.name.to_string();
  let description = tool.description.as_deref().unwrap_or_default().to_string();
  DynamicTool::new(
    name.clone(),
    description,
    tool.schema_as_json_value(),
    move |context: &mut ToolContext, args: serde_json::Value| {
      let (name, peer, progress) = (name.clone(), peer.clone(), progress.clone());
      let sink = context.get::<Output>().cloned();
      Box::pin(async move {
        let mut params = CallToolRequestParams::new(name.clone());
        match args {
          serde_json::Value::Object(args) => params = params.with_arguments(args),
          serde_json::Value::Null => {}
          // Anything else is not a call's arguments, and sent without them it
          // would be a different call from the one the model asked for.
          other => {
            return Err(ToolExecutionError::invalid_args(format!(
              "MCP tool '{name}' takes an object of arguments, not {other}"
            )));
          }
        }
        let result = call(&peer, params, timeout, &progress, sink)
          .await
          .map_err(|err| failed(&name, err))?;
        let output = answer(&result)?;
        match result.is_error == Some(true) {
          true => Err(
            ToolExecutionError::other(format!("MCP tool '{name}' reported an execution error"))
              .with_model_output(output),
          ),
          false => Ok(output),
        }
      })
    },
  )
}

/// Why a call came to nothing, for the model to make what it can of.
fn failed(name: &str, err: ServiceError) -> ToolExecutionError {
  match err {
    ServiceError::Timeout { timeout } => ToolExecutionError::timeout(format!(
      "MCP tool '{name}' timed out: nothing from the server for {}s",
      timeout.as_secs_f64()
    )),
    err => ToolExecutionError::provider(format!("MCP tool '{name}' request failed: {err}")),
  }
}

/// Ask the server to run the tool, and wait for it to say what came of it:
/// for up to `timeout` after the last word from it, whether the call going
/// out or a note on how far along it is. Its notes go to `sink`.
async fn call(
  peer: &ServerSink,
  params: CallToolRequestParams,
  timeout: Option<Duration>,
  progress: &Progress,
  sink: Option<Output>,
) -> Result<CallToolResult, ServiceError> {
  let options = match timeout {
    Some(limit) => PeerRequestOptions::with_timeout(limit).reset_timeout_on_progress(),
    None => PeerRequestOptions::no_options(),
  };
  let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
  // Going out waits for room among what else is going to the server, which
  // is time the call takes too.
  let handle = crate::resources::bounded(timeout, peer.send_cancellable_request(request, options)).await?;
  let _listening = sink.map(|sink| progress.listen(handle.progress_token.clone(), sink));
  match handle.await_response().await? {
    ServerResult::CallToolResult(result) => Ok(result),
    _ => Err(ServiceError::UnexpectedResponse),
  }
}

// ---------------------------------------------------------------- the answer

/// The answer as the model is given it: text as text, an image as an image,
/// and anything else as the block the server sent, whole, so nothing it said
/// about it is lost.
///
/// Structured content goes first, as it is. A server that sends it also
/// sends it written out as text, for clients that only read text; that
/// block is the same thing twice, so it is where the structured content goes
/// instead, rather than in front of it.
fn answer(result: &CallToolResult) -> Result<ToolOutput, ToolExecutionError> {
  let structured = result.structured_content.as_ref();
  // Written out however the server likes to write JSON — flat, or indented
  // for whoever reads it — so it is the value that is compared, not the text.
  let copy = structured.and_then(|structured| {
    result.content.iter().position(|block| match block {
      ContentBlock::Text(text) => serde_json::from_str::<serde_json::Value>(&text.text).is_ok_and(|v| &v == structured),
      _ => false,
    })
  });
  let mut content = result
    .content
    .iter()
    .enumerate()
    .map(|(at, block)| match (Some(at) == copy, structured) {
      (true, Some(structured)) => Ok(ToolResultContent::json(structured.clone())),
      _ => content(block),
    })
    .collect::<Result<Vec<_>, _>>()?;
  if let (Some(structured), None) = (structured, copy) {
    content.insert(0, ToolResultContent::json(structured.clone()));
  }
  // An answer with nothing in it is still an answer to give.
  if content.is_empty() {
    return Ok(ToolOutput::text(match result.is_error == Some(true) {
      true => "the MCP tool reported an error",
      false => "",
    }));
  }
  ToolOutput::content(content)
}

/// One block of an answer.
fn content(block: &ContentBlock) -> Result<ToolResultContent, ToolExecutionError> {
  let image = |data: &str, mime: Option<&str>| {
    let media = mime.and_then(ImageMediaType::from_mime_type)?;
    Some(ToolResultContent::image_base64(data.to_string(), Some(media), None))
  };
  let shown = match block {
    ContentBlock::Text(text) => Some(ToolResultContent::text(text.text.clone())),
    ContentBlock::Image(picture) => image(&picture.data, Some(&picture.mime_type)),
    ContentBlock::Resource(resource) => match &resource.resource {
      ResourceContents::BlobResourceContents { mime_type, blob, .. } => image(blob, mime_type.as_deref()),
      _ => None,
    },
    _ => None,
  };
  match shown {
    Some(shown) => Ok(shown),
    None => serde_json::to_value(block)
      .map(ToolResultContent::json)
      .map_err(|err| ToolExecutionError::provider(format!("an MCP answer that could not be kept: {err}"))),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rmcp::model::NumberOrString;

  fn note(token: i64, progress: f64, total: Option<f64>, message: Option<&str>) -> ProgressNotificationParam {
    let mut note = ProgressNotificationParam::new(ProgressToken(NumberOrString::Number(token)), progress);
    note.total = total;
    note.message = message.map(str::to_string);
    note
  }

  fn collecting() -> (Output, Arc<Mutex<Vec<String>>>) {
    let lines = Arc::new(Mutex::new(Vec::new()));
    let into = lines.clone();
    (Output(Arc::new(move |line| into.lock().unwrap().push(line))), lines)
  }

  #[test]
  fn a_note_is_how_far_along_and_what_it_is_doing() {
    assert_eq!(said(&note(0, 3.0, Some(10.0), None)), "3/10 (30%)");
    assert_eq!(said(&note(0, 3.0, Some(10.0), Some("files"))), "3/10 (30%): files");
    // A fraction, or a percentage out of a hundred, is only the percentage.
    assert_eq!(said(&note(0, 0.25, Some(1.0), None)), "25%");
    assert_eq!(said(&note(0, 40.0, Some(100.0), None)), "40%");
    // Without a total, the message says more than the amount would.
    assert_eq!(said(&note(0, 7.0, None, Some(" Indexing\n src/ "))), "Indexing src/");
    assert_eq!(said(&note(0, 7.0, None, None)), "7");
    assert_eq!(said(&note(0, 7.0, Some(0.0), Some(""))), "7");
  }

  #[test]
  fn a_call_is_shown_its_own_notes_and_never_goes_backwards() {
    let progress = Progress::default();
    let token = |n| ProgressToken(NumberOrString::Number(n));
    // Said before the call was listening: held for it.
    progress.heard(note(1, 1.0, Some(4.0), None));
    let (sink, lines) = collecting();
    let listening = progress.listen(token(1), sink);
    progress.heard(note(1, 3.0, Some(4.0), None));
    progress.heard(note(1, 2.0, Some(4.0), None));
    progress.heard(note(2, 1.0, Some(4.0), None));
    progress.heard(note(1, 3.0, Some(4.0), Some("still")));
    assert_eq!(
      *lines.lock().unwrap(),
      ["1/4 (25%)", "3/4 (75%)", "3/4 (75%): still"],
      "the late 2/4 is behind, and token 2 is another call's"
    );

    // Answered: what comes after is nobody's, and is not shown.
    drop(listening);
    progress.heard(note(1, 4.0, Some(4.0), None));
    assert_eq!(lines.lock().unwrap().len(), 3);

    // Notes nobody comes for do not pile up, and a call listening is kept.
    let (sink, lines) = collecting();
    let _listening = progress.listen(token(100), sink);
    for n in 200..300 {
      progress.heard(note(n, 1.0, None, None));
    }
    assert!(progress.lock().len() <= UNCLAIMED + 1);
    progress.heard(note(100, 1.0, None, Some("here")));
    assert_eq!(*lines.lock().unwrap(), ["here"]);
  }

  #[test]
  fn an_answer_is_its_blocks_with_structured_content_once() {
    let text = |t: &str| ContentBlock::text(t);
    let texts = |output: &ToolOutput| -> Vec<String> {
      output
        .as_content()
        .iter()
        .map(|c| match c {
          ToolResultContent::Text(t) => t.text.clone(),
          ToolResultContent::Json { value } => format!("json {value}"),
          other => format!("{other:?}"),
        })
        .collect()
    };
    let plain = CallToolResult::success(vec![text("a"), text("b")]);
    assert_eq!(texts(&answer(&plain).unwrap()), ["a", "b"]);

    // The text written for clients that only read text is the structured
    // content again, so it is replaced by it rather than kept beside it.
    let value = serde_json::json!({"n": 1});
    let structured = CallToolResult::structured(value.clone());
    assert_eq!(texts(&answer(&structured).unwrap()), [format!("json {value}")]);
    // However it was written out: indented is as much a copy as flat.
    let mut pretty = CallToolResult::success(vec![text("{\n  \"n\": 1\n}")]);
    pretty.structured_content = Some(value.clone());
    assert_eq!(texts(&answer(&pretty).unwrap()), [format!("json {value}")]);
    // Other text is the server's own, and stays, after the structured content.
    let mut both = CallToolResult::success(vec![text("note")]);
    both.structured_content = Some(value.clone());
    assert_eq!(texts(&answer(&both).unwrap()), [format!("json {value}"), "note".into()]);

    // An image is an image; one of a type no model takes is kept as sent.
    let png = ContentBlock::image("AAAA", "image/png");
    let tiff = ContentBlock::image("AAAA", "image/tiff");
    let shown = answer(&CallToolResult::success(vec![png, tiff])).unwrap();
    assert!(matches!(shown.as_content()[0], ToolResultContent::Image(_)));
    assert!(matches!(&shown.as_content()[1], ToolResultContent::Json { value } if value["mimeType"] == "image/tiff"));

    // Nothing at all is still something to give the model.
    assert_eq!(texts(&answer(&CallToolResult::success(vec![])).unwrap()), [""]);
    assert_eq!(
      texts(&answer(&CallToolResult::error(vec![])).unwrap()),
      ["the MCP tool reported an error"]
    );
  }

  // ------------------------------------------------------------ a server

  /// A client that hears notes and nothing else.
  struct Hears(Progress);

  impl rmcp::ClientHandler for Hears {
    async fn on_progress(
      &self,
      note: ProgressNotificationParam,
      _context: rmcp::service::NotificationContext<rmcp::service::RoleClient>,
    ) {
      self.0.heard(note);
    }
  }

  /// A server at the other end of `io` with one tool, `slow`, that says it
  /// has started, then how far along it is every `pause`, and is done after
  /// four of them — and another, `silent`, that never answers at all.
  async fn server(io: tokio::io::DuplexStream, pause: Duration) {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    let (read, mut write) = tokio::io::split(io);
    let mut lines = tokio::io::BufReader::new(read).lines();
    let mut send = async move |message: serde_json::Value| {
      let line = format!("{message}\n");
      write.write_all(line.as_bytes()).await.expect("the client is there");
    };
    while let Ok(Some(line)) = lines.next_line().await {
      let message: serde_json::Value = serde_json::from_str(&line).expect("JSON-RPC");
      // What the client only says, rather than asks, needs no answer.
      let Some(id) = message.get("id").cloned() else {
        continue;
      };
      let answer = |result| serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result });
      match (message["method"].as_str(), message["params"]["name"].as_str()) {
        (Some("initialize"), _) => {
          send(answer(serde_json::json!({
            "protocolVersion": message["params"]["protocolVersion"],
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "slow", "version": "0" },
          })))
          .await
        }
        (Some("tools/call"), Some("slow")) => {
          let token = message["params"]["_meta"]["progressToken"].clone();
          let mut note = async |progress: u32, message: Option<&str>| {
            let params = serde_json::json!({
              "progressToken": token, "progress": progress, "total": 4, "message": message,
            });
            send(serde_json::json!({ "jsonrpc": "2.0", "method": "notifications/progress", "params": params })).await;
          };
          note(0, Some("starting")).await;
          for step in 1..4 {
            tokio::time::sleep(pause).await;
            note(step, None).await;
          }
          tokio::time::sleep(pause).await;
          send(answer(
            serde_json::json!({ "content": [{ "type": "text", "text": "done" }] }),
          ))
          .await;
        }
        (Some("tools/call"), _) => {}
        _ => panic!("not asked for here: {message}"),
      }
    }
  }

  /// A connection to a server of our own, and what its calls hear.
  async fn connect(
    pause: Duration,
  ) -> (
    rmcp::service::RunningService<rmcp::service::RoleClient, Hears>,
    Progress,
  ) {
    use rmcp::ServiceExt;

    let (ours, theirs) = tokio::io::duplex(64 * 1024);
    tokio::spawn(server(theirs, pause));
    let progress = Progress::default();
    let running = Hears(progress.clone()).serve(ours).await.expect("connected");
    (running, progress)
  }

  #[tokio::test]
  async fn a_long_call_is_shown_how_far_along_it_is_and_is_not_cut_short_while_it_says() {
    // Four pauses are longer than the call may take, but each is well
    // within it, and each ends with word from the server.
    let pause = Duration::from_millis(150);
    let (running, progress) = connect(pause).await;
    let (sink, lines) = collecting();
    let result = call(
      running.peer(),
      CallToolRequestParams::new("slow"),
      Some(Duration::from_millis(400)),
      &progress,
      Some(sink),
    )
    .await
    .expect("answered in the end");
    assert_eq!(texts_of(&result), ["done"]);
    assert_eq!(
      *lines.lock().unwrap(),
      ["0/4 (0%): starting", "1/4 (25%)", "2/4 (50%)", "3/4 (75%)"]
    );
    assert!(progress.lock().is_empty(), "nothing left listening once it is answered");
  }

  #[tokio::test]
  async fn a_call_the_server_says_nothing_about_times_out() {
    let (running, progress) = connect(Duration::ZERO).await;
    let err = call(
      running.peer(),
      CallToolRequestParams::new("silent"),
      Some(Duration::from_millis(100)),
      &progress,
      None,
    )
    .await
    .expect_err("no answer");
    assert!(matches!(err, ServiceError::Timeout { .. }), "{err}");
    let said = failed("silent", err).to_string();
    assert!(said.contains("nothing from the server for 0.1s"), "{said}");
  }

  fn texts_of(result: &CallToolResult) -> Vec<String> {
    result
      .content
      .iter()
      .filter_map(|block| block.as_text().map(|text| text.text.clone()))
      .collect()
  }
}
