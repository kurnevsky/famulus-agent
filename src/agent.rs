//! Agent construction and the streaming run loop. The TUI never touches rig
//! directly; it receives `AgentEvent`s over a channel and can abort a run.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use futures::StreamExt;
use rig_agent::agent::{
  Agent, AgentBuilder, AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, MultiTurnStreamItem,
  RequestPatch, StreamingError, ToolCall, ToolCallAction, ToolResultAction, ToolResultEvent,
};
use rig_agent::client::AgentClientExt;
use rig_agent::completion::PromptError;
use rig_agent::streaming::StreamingPrompt;
use rig_agent::tool::{Tool, ToolOutput};
use rig_core::client::completion::CompletionClient;
use rig_core::completion::{
  CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Message, ProviderCapabilities, Usage,
};
use rig_core::message::{ToolResultContent, UserContent};
use rig_core::providers::{gemini, openai};
use rig_core::streaming::{StreamedAssistantContent, StreamingCompletionResponse, ToolCallDeltaContent};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::compaction::{self, Compacted, Settings};
use crate::tools::{BashTool, CALL_ARG, EditDiff, EditTool, ReadTool, WriteTool};

/// Events streamed from a run to the UI.
#[derive(Debug)]
pub enum AgentEvent {
  Text(String),
  Reasoning(String),
  ToolCall {
    name: String,
    args: serde_json::Value,
    /// The call, so its output and result find it again even when several
    /// tools are in flight at once.
    call: String,
    /// The same call as the stream named it while the model wrote it, which
    /// is the id the half-written line is keyed by.
    internal: String,
  },
  /// A tool call being written, before it is run. `args` is the JSON as far
  /// as it has arrived, which is usually not yet parseable; `name` is empty
  /// until the provider has said it.
  ToolCallDelta {
    /// Correlates the fragments of one call, so two calls in the same turn
    /// do not run into each other.
    id: String,
    name: String,
    args: String,
  },
  /// Live output of the running tool (bash), replacing earlier snapshots.
  ToolOutput {
    call: String,
    text: String,
  },
  ToolResult {
    name: String,
    output: String,
    is_error: bool,
    /// The call this answers, as the transcript will name it. Whether a tool
    /// failed is not something a transcript records — a failed result looks
    /// like any other on the wire — so a session that wants to redraw it
    /// later has to keep the flag against this id itself.
    call: String,
    /// Numbered diff for `edit`, shown in place of the output text.
    diff: Option<String>,
  },
  /// The run finished. `messages` holds only this run's new transcript
  /// messages (prompt, tool calls/results, final answer); append them to the
  /// history. `context_tokens` is the size of the last completion request
  /// as reported by the provider (0 if it reported nothing).
  Done {
    messages: Vec<Message>,
    usage: Usage,
    context_tokens: u64,
  },
  /// The run ended without a final response (after an `Error`).
  Ended,
  /// Compaction finished; `None` means there was nothing to compact.
  Compacted(Option<Compacted>),
  Error(String),
}

pub struct Agents {
  pub agent: Arc<Agent>,
  /// Tool-less agent used to summarize history during compaction.
  pub summarizer: Arc<Agent>,
  /// Raised while a message the user typed is waiting behind the run.
  pub waiting: Waiting,
}

/// Whether something the user typed is waiting to be sent. The UI raises it,
/// the run reads it at each turn boundary, so a message that arrives mid-run
/// is taken at the next opportunity rather than after the whole answer.
pub type Waiting = Arc<AtomicBool>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
  /// OpenAI Chat Completions and compatible servers.
  OpenAi,
  /// Google Gemini (generateContent API).
  Gemini,
}

pub struct Config {
  pub provider: Provider,
  /// Provider endpoint root; `None` uses the provider's default.
  pub base_url: Option<String>,
  pub api_key: String,
  pub model: String,
  pub system_prompt: Option<String>,
  pub max_turns: usize,
  pub compaction: Settings,
  /// Whether the model accepts image input.
  pub vision: bool,
  /// The tools to offer the model, or `None` to offer every one there is.
  /// Every tool is registered either way; this is what the model is shown,
  /// and rig refuses a call to anything left out of it.
  pub tools: Option<Vec<String>>,
}

/// The id of the call a hook is reporting.
///
/// Rig's handle is always there — minted when the provider issued none — and
/// the `Option` is only in the shape of the event, so there is no absent case
/// to have an answer for.
fn call_id(id: Option<&str>) -> String {
  id.unwrap_or_default().to_string()
}

/// Reports tool calls and results to the UI. Both go through the hook path so
/// they stay ordered, and the result carries an accurate error flag, which the
/// transcript-level stream items do not.
struct UiHook {
  tx: mpsc::UnboundedSender<AgentEvent>,
  waiting: Waiting,
  /// Narrows what each request advertises, when a session asked for less than
  /// everything. Per turn is the only place rig takes it, so it is said again
  /// every turn.
  tools: Option<Vec<String>>,
}

impl AgentHook for UiHook {
  /// Stop before the next model call when the user has said something since
  /// the run started, so their message is the next thing the model reads.
  ///
  /// Between turns is the only place a run can be cut without leaving a call
  /// unanswered: every tool the turn just ended asked for has come back. The
  /// run's work is kept the way an abort keeps it, and the waiting message is
  /// sent as a fresh run over that history — which is what the model would
  /// have seen had it been typed a moment earlier.
  ///
  /// Never before the first call, where the run has done nothing yet: the
  /// message it stopped for would start a run that stops for the next one.
  async fn on_completion_call(&self, _ctx: &HookContext, event: CompletionCallEvent<'_>) -> CompletionCallAction {
    if event.turn > 1 && self.waiting.load(Ordering::Relaxed) {
      // The reason is rig's to carry, not anything this program reads: the
      // stop comes back as a `PromptCancelled` and `start_run` knows it by
      // that, not by what it says. It is here for a stack trace to say.
      return CompletionCallAction::Stop("a message is waiting to be sent".into());
    }
    match &self.tools {
      Some(tools) => CompletionCallAction::Patch(RequestPatch::new().active_tools(tools.clone())),
      None => CompletionCallAction::Continue,
    }
  }

  async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
    let args = serde_json::from_str(event.args).unwrap_or_else(|_| serde_json::Value::String(event.args.to_string()));
    let _ = self.tx.send(AgentEvent::ToolCall {
      name: event.tool_name.to_string(),
      args: args.clone(),
      call: call_id(event.tool_call_id),
      internal: event.internal_call_id.to_string(),
    });
    // A command reports its output while it runs, and nothing in rig tells a
    // tool which call it is. Rewriting the arguments is the one channel from
    // here into the body, so the id goes down it — see `tools::CALL_ARG`.
    match (event.tool_name, args) {
      (BashTool::NAME, serde_json::Value::Object(mut args)) => {
        args.insert(CALL_ARG.to_string(), call_id(event.tool_call_id).into());
        ToolCallAction::Rewrite(args.into())
      }
      _ => ToolCallAction::Run,
    }
  }

  async fn on_tool_result(&self, _ctx: &HookContext, event: ToolResultEvent<'_>) -> ToolResultAction {
    let _ = self.tx.send(AgentEvent::ToolResult {
      name: event.tool_name.to_string(),
      call: call_id(event.tool_call_id),
      diff: event.tool_context.result::<EditDiff>().map(|d| d.diff.clone()),
      output: render_output(event.presentation),
      is_error: event.raw_result.is_error() || event.raw_result.is_refused(),
    });
    ToolResultAction::Keep
  }
}

pub fn build_agents(
  cfg: &Config,
  cwd: &Path,
  tx: mpsc::UnboundedSender<AgentEvent>,
  servers: &crate::mcp::Servers,
) -> Result<Agents> {
  let preamble = match &cfg.system_prompt {
    Some(p) => p.clone(),
    None => default_system_prompt(cwd, cfg.tools.as_deref()),
  };
  let base_url = cfg.base_url.as_deref().map(|u| u.trim_end_matches('/'));

  // The two providers differ only in the model handed to the builder. Gemini
  // accepts images inside function responses, so it gets the raw model; the
  // chat completions format needs the relay wrapper.
  let (agent, summarizer) = match cfg.provider {
    Provider::OpenAi => {
      let mut builder = openai::CompletionsClient::builder().api_key(cfg.api_key.as_str());
      if let Some(url) = base_url {
        builder = builder.base_url(url);
      }
      let client = builder.build().context("failed to build OpenAI-compatible client")?;
      let model = ToolImageRelay {
        inner: client.completion_model(cfg.model.clone()),
      };
      (AgentBuilder::new(model), client.agent(cfg.model.clone()))
    }
    Provider::Gemini => {
      let mut builder = gemini::Client::builder().api_key(cfg.api_key.as_str());
      if let Some(url) = base_url {
        builder = builder.base_url(url);
      }
      let client = builder.build().context("failed to build Gemini client")?;
      (
        AgentBuilder::new(client.completion_model(cfg.model.clone())),
        client.agent(cfg.model.clone()),
      )
    }
  };

  let output_tx = tx.clone();
  let waiting = Waiting::default();
  let agent = agent
    .preamble(&preamble)
    .default_max_turns(cfg.max_turns)
    .add_hook(UiHook {
      tx,
      waiting: waiting.clone(),
      tools: cfg.tools.clone(),
    })
    .tool(ReadTool {
      cwd: cwd.to_path_buf(),
      vision: cfg.vision,
    })
    .tool(WriteTool { cwd: cwd.to_path_buf() })
    .tool(EditTool { cwd: cwd.to_path_buf() })
    .tool(BashTool {
      cwd: cwd.to_path_buf(),
      on_output: Some(Arc::new(move |call, text| {
        let _ = output_tx.send(AgentEvent::ToolOutput { call, text });
      })),
    });
  // Whatever the session's MCP servers offer, alongside the four the agent
  // brought: a tool is a tool, and the transcript draws them all the same.
  let agent = crate::mcp::attach(agent, servers).build();
  let summarizer = summarizer
    .preamble(compaction::SYSTEM_PROMPT)
    .default_max_turns(1)
    .build();
  Ok(Agents {
    agent: Arc::new(agent),
    summarizer: Arc::new(summarizer),
    waiting,
  })
}

/// Text shown in the UI for a tool result; images become a placeholder.
fn render_output(output: &ToolOutput) -> String {
  output
    .as_content()
    .iter()
    .map(|c| match c {
      ToolResultContent::Text(t) => t.text.clone(),
      ToolResultContent::Json { value } => value.to_string(),
      ToolResultContent::Image(_) => "[image]".to_string(),
    })
    .collect::<Vec<_>>()
    .join("\n")
}

/// The OpenAI chat completions API only accepts text in tool messages, so this
/// wrapper strips images out of tool results and re-sends them in a user
/// message right after, so `read` can return screenshots.
struct ToolImageRelay<M> {
  inner: M,
}

impl<M: CompletionModel> CompletionModel for ToolImageRelay<M> {
  async fn completion(&self, mut request: CompletionRequest) -> Result<CompletionResponse, CompletionError> {
    relay_tool_images(&mut request.chat_history);
    self.inner.completion(request).await
  }

  async fn stream(&self, mut request: CompletionRequest) -> Result<StreamingCompletionResponse, CompletionError> {
    relay_tool_images(&mut request.chat_history);
    self.inner.stream(request).await
  }

  fn capabilities(&self) -> ProviderCapabilities {
    self.inner.capabilities()
  }
}

fn relay_tool_images(history: &mut Vec<Message>) {
  let has_image = |message: &Message| {
    matches!(message, Message::User { content } if content.iter().any(|c| {
        matches!(c, UserContent::ToolResult(r) if r.content.iter().any(|c| matches!(c, ToolResultContent::Image(_))))
    }))
  };
  if !history.iter().any(has_image) {
    return;
  }
  let mut out = Vec::with_capacity(history.len() + 1);
  for message in history.drain(..) {
    if !has_image(&message) {
      out.push(message);
      continue;
    }
    let Message::User { mut content } = message else {
      unreachable!()
    };
    let mut images = Vec::new();
    for item in &mut content {
      let UserContent::ToolResult(result) = item else {
        continue;
      };
      let (imgs, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut result.content)
        .into_iter()
        .partition(|c| matches!(c, ToolResultContent::Image(_)));
      result.content = if rest.is_empty() {
        vec![ToolResultContent::text("(see attached image)")]
      } else {
        rest
      };
      images.extend(imgs.into_iter().filter_map(|c| match c {
        ToolResultContent::Image(image) => Some(UserContent::Image(image)),
        _ => None,
      }));
    }
    out.push(Message::User { content });
    let mut relay = vec![UserContent::text("Attached image(s) from tool result:")];
    relay.extend(images);
    out.push(Message::User { content: relay });
  }
  *history = out;
}

/// The lines of the built-in prompt that speak for a tool, by the tool they
/// speak for: a session without one should not be told to use it.
/// A whole word, so `read` does not take `already` with it — and the plural
/// too, since the rules speak of a tool's arguments as `edits[]`.
fn mentions(line: &str, tool: &str) -> bool {
  line
    .split(|c: char| !c.is_ascii_alphanumeric())
    .any(|word| word == tool || word.strip_suffix('s') == Some(tool))
}

/// Drop what the prompt says about tools this session does not have.
///
/// A model told about `bash` and then refused it does not quietly do without:
/// it tries, is refused, and says so instead of using what it does have.
fn for_tools(prompt: &str, tools: Option<&[String]>) -> String {
  let Some(tools) = tools else {
    return prompt.to_string();
  };
  let gone: Vec<&str> = crate::tools::BUILT_IN
    .iter()
    .copied()
    .filter(|name| !tools.iter().any(|tool| tool == name))
    .collect();
  if gone.is_empty() {
    return prompt.to_string();
  }
  prompt
    .lines()
    .filter(|line| !gone.iter().any(|tool| mentions(line, tool)))
    .collect::<Vec<_>>()
    .join("\n")
}

fn default_system_prompt(cwd: &Path, tools: Option<&[String]>) -> String {
  let mut prompt = for_tools(
    "You are an expert coding assistant operating inside a minimal terminal coding agent. \
         You help users by reading files, executing commands, editing code, and writing new files.\n\n\
         <tools>\n\
         - read: Read file contents\n\
         - bash: Execute bash commands (ls, grep, find, etc.)\n\
         - edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call\n\
         - write: Create or overwrite files\n\
         </tools>\n\n\
         <rules>\n\
         - Use bash for file operations like ls, rg, find\n\
         - Use read to examine files instead of cat or sed.\n\
         - Use edit for precise changes (edits[].oldText must match exactly)\n\
         - When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls\n\
         - Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.\n\
         - Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.\n\
         - Use write only for new files or complete rewrites.\n\
         - Be concise in your responses\n\
         - Show file paths clearly when working with files\n\
         </rules>\n",
    tools,
  );

  let mut context_files = Vec::new();
  for name in ["AGENTS.md", "CLAUDE.md"] {
    let path = cwd.join(name);
    if let Ok(content) = std::fs::read_to_string(&path) {
      context_files.push((path.display().to_string(), content));
      break;
    }
  }
  if !context_files.is_empty() {
    prompt.push_str("\n<project_context>\nProject-specific instructions and guidelines:\n\n");
    for (path, content) in &context_files {
      prompt.push_str(&format!(
        "<project_instructions path=\"{path}\">\n{content}\n</project_instructions>\n"
      ));
    }
    prompt.push_str("</project_context>\n");
  }

  prompt.push_str(&format!("\n<cwd>\n{}\n</cwd>", cwd.display()));
  prompt
}

/// Start one agent run in the background. Dropping/aborting the handle cancels
/// the HTTP stream and any running tool process. `prompt` is the last message
/// of the request: usually a new user message, but `/continue` re-sends the
/// last message of the history to resume the loop without one.
pub fn start_run(
  agent: Arc<Agent>,
  history: Vec<Message>,
  prompt: Message,
  tx: mpsc::UnboundedSender<AgentEvent>,
) -> JoinHandle<()> {
  tokio::spawn(async move {
    let mut stream = agent.stream_prompt(prompt).history(history).await;
    let mut sent_done = false;
    // The stream stopped for a reason the UI has already been told, so there
    // is nothing left to say about it when the loop falls out.
    let mut explained = false;
    // Arguments arrive a few characters at a time and each fragment carries
    // only what is new, so the run holds what has arrived per call and sends
    // the whole of it. The UI then has nothing to reassemble.
    let mut writing: HashMap<String, (String, String)> = HashMap::new();
    while let Some(item) = stream.next().await {
      let event = match item {
        Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => match content {
          StreamedAssistantContent::Text(text) => Some(AgentEvent::Text(text.text)),
          StreamedAssistantContent::ReasoningDelta { reasoning, .. } => Some(AgentEvent::Reasoning(reasoning)),
          StreamedAssistantContent::Reasoning { reasoning, .. } => {
            let text = reasoning.display_text();
            (!text.is_empty()).then_some(AgentEvent::Reasoning(text))
          }
          StreamedAssistantContent::ToolCallDelta {
            internal_call_id,
            content,
          } => {
            let (name, args) = writing.entry(internal_call_id.clone()).or_default();
            match content {
              ToolCallDeltaContent::Name(part) => name.push_str(&part),
              ToolCallDeltaContent::Delta(part) => args.push_str(&part),
            }
            Some(AgentEvent::ToolCallDelta {
              id: internal_call_id,
              name: name.clone(),
              args: args.clone(),
            })
          }
          // The call is written but not yet run — `UiHook` reports it when it
          // starts. Showing it whole in the meantime is what the finished
          // line will say, so nothing jumps when the two swap over.
          StreamedAssistantContent::ToolCall {
            tool_call,
            internal_call_id,
          } => Some(AgentEvent::ToolCallDelta {
            id: internal_call_id,
            name: tool_call.function.name,
            args: tool_call.function.arguments.to_string(),
          }),
          _ => None,
        },
        // Tool calls and results are reported by `UiHook`.
        Ok(MultiTurnStreamItem::FinalResponse(response)) => {
          sent_done = true;
          Some(AgentEvent::Done {
            messages: response.messages.unwrap_or_default(),
            usage: response.usage,
            context_tokens: response
              .completion_calls
              .last()
              .map(|call| call.usage.input_tokens + call.usage.output_tokens)
              .unwrap_or(0),
          })
        }
        Ok(_) => None,
        // A cancelled run is this app cutting in with the message the user
        // typed while it ran, not something that went wrong, so it ends the
        // way an abort does — quietly, keeping what it got through.
        Err(StreamingError::Prompt(err)) if matches!(*err, PromptError::PromptCancelled { .. }) => {
          explained = true;
          None
        }
        Err(err) => {
          explained = true;
          Some(AgentEvent::Error(err.to_string()))
        }
      };
      if let Some(event) = event
        && tx.send(event).is_err()
      {
        return;
      }
    }
    if !sent_done {
      if !explained {
        let _ = tx.send(AgentEvent::Error("stream ended without a final response".into()));
      }
      let _ = tx.send(AgentEvent::Ended);
    }
  })
}

/// Summarize older history in the background; the result arrives as
/// `AgentEvent::Compacted` (or `AgentEvent::Error` followed by `Ended`).
pub fn start_compaction(
  summarizer: Arc<Agent>,
  history: Vec<Message>,
  settings: Settings,
  tx: mpsc::UnboundedSender<AgentEvent>,
) -> JoinHandle<()> {
  tokio::spawn(async move {
    let event = match compaction::compact(&summarizer, history, &settings).await {
      Ok(result) => AgentEvent::Compacted(result),
      Err(err) => {
        let _ = tx.send(AgentEvent::Error(format!("compaction failed: {err}")));
        AgentEvent::Ended
      }
    };
    let _ = tx.send(event);
  })
}

#[cfg(test)]
mod tests {
  //! Runs against a mock OpenAI-compatible server when `FA_TEST_BASE_URL` is set.
  use super::*;

  async fn collect(
    agent: &Arc<Agent>,
    history: Vec<Message>,
    prompt: &str,
    rx: &mut mpsc::UnboundedReceiver<AgentEvent>,
    tx: &mpsc::UnboundedSender<AgentEvent>,
  ) -> Vec<AgentEvent> {
    let handle = start_run(agent.clone(), history, Message::user(prompt), tx.clone());
    let mut events = Vec::new();
    loop {
      let ev = tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
        .await
        .expect("event within 10s")
        .expect("channel open");
      let done = matches!(ev, AgentEvent::Done { .. });
      events.push(ev);
      if done {
        break;
      }
    }
    handle.await.unwrap();
    events
  }

  const TEST_SETTINGS: Settings = Settings {
    enabled: true,
    context_window: 1000,
    reserve_tokens: 100,
    keep_recent_tokens: 1,
  };

  #[tokio::test]
  async fn mock_end_to_end() {
    let Ok(base_url) = std::env::var("FA_TEST_BASE_URL") else {
      eprintln!("FA_TEST_BASE_URL not set; skipping");
      return;
    };
    let cfg = Config {
      provider: Provider::OpenAi,
      base_url: Some(base_url),
      api_key: "test".into(),
      model: "mock".into(),
      system_prompt: None,
      max_turns: 5,
      compaction: TEST_SETTINGS,
      vision: true,
      tools: None,
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let agent = build_agents(&cfg, Path::new("/tmp"), tx.clone(), &Default::default())
      .unwrap()
      .agent;

    // 1. Plain streamed text.
    let events = collect(&agent, vec![], "hi", &mut rx, &tx).await;
    let text: String = events
      .iter()
      .filter_map(|e| match e {
        AgentEvent::Text(t) => Some(t.as_str()),
        _ => None,
      })
      .collect();
    assert_eq!(text, "Hello there! I see 2 messages in history.");
    let Some(AgentEvent::Done {
      messages: history,
      usage,
      context_tokens,
    }) = events.last()
    else {
      panic!("no Done")
    };
    assert_eq!(history.len(), 2, "messages = user + assistant, got {history:?}");
    assert_eq!(usage.output_tokens, 7);
    assert_eq!(*context_tokens, 19);

    // 2. Tool call -> tool result -> reasoning + final text, continuing the history.
    let events = collect(&agent, history.clone(), "run it", &mut rx, &tx).await;
    let names: Vec<String> = events
      .iter()
      .map(|e| match e {
        AgentEvent::Text(_) => "text".into(),
        AgentEvent::Reasoning(_) => "reasoning".into(),
        AgentEvent::ToolCall { name, .. } => format!("call:{name}"),
        AgentEvent::ToolCallDelta { .. } => "writing".into(),
        AgentEvent::ToolOutput { .. } => "output".into(),
        AgentEvent::ToolResult { is_error, .. } => {
          format!("result:{}", if *is_error { "err" } else { "ok" })
        }
        AgentEvent::Done { .. } => "done".into(),
        AgentEvent::Error(e) => format!("error:{e}"),
        AgentEvent::Ended => "ended".into(),
        AgentEvent::Compacted(_) => "compacted".into(),
      })
      .collect();
    assert!(names.contains(&"call:bash".to_string()), "{names:?}");
    assert!(names.contains(&"result:ok".to_string()), "{names:?}");
    assert!(names.contains(&"reasoning".to_string()), "{names:?}");
    assert!(
      names.iter().position(|n| n == "call:bash") < names.iter().position(|n| n == "result:ok"),
      "{names:?}"
    );
    let out = events
      .iter()
      .find_map(|e| match e {
        AgentEvent::ToolResult { output, .. } => Some(output.clone()),
        _ => None,
      })
      .unwrap();
    assert_eq!(out.trim(), "hello from mock");
    let Some(AgentEvent::Done { messages: history, .. }) = events.last() else {
      panic!("no Done")
    };
    let roles: Vec<String> = history
      .iter()
      .map(|m| match m {
        Message::System { .. } => "system".into(),
        Message::User { content } => format!(
          "user[{}]",
          content
            .iter()
            .map(|c| match c {
              rig_core::message::UserContent::Text(_) => "text",
              rig_core::message::UserContent::ToolResult(_) => "tool_result",
              _ => "other",
            })
            .collect::<Vec<_>>()
            .join(",")
        ),
        Message::Assistant { content, .. } => format!(
          "assistant[{}]",
          content
            .iter()
            .map(|c| match c {
              rig_core::message::AssistantContent::Text(_) => "text",
              rig_core::message::AssistantContent::ToolCall(_) => "tool_call",
              rig_core::message::AssistantContent::Reasoning(_) => "reasoning",
              _ => "other",
            })
            .collect::<Vec<_>>()
            .join(",")
        ),
      })
      .collect();
    assert_eq!(
      roles,
      [
        "user[text]",
        "assistant[tool_call]",
        "user[tool_result]",
        "assistant[reasoning,text]"
      ],
      "only this run's messages are returned"
    );

    // 3. A failing tool is flagged as an error and fed back to the model.
    let events = collect(&agent, vec![], "please fail", &mut rx, &tx).await;
    let failure = events.iter().find_map(|e| match e {
      AgentEvent::ToolResult {
        is_error: true, output, ..
      } => Some(output.clone()),
      _ => None,
    });
    let failure = failure.unwrap_or_else(|| panic!("no failed tool result in {events:?}"));
    assert!(
      failure.contains("/nonexistent/file") && failure.contains("No such file"),
      "model should see the real reason: {failure}"
    );
  }

  #[tokio::test]
  async fn mock_compaction() {
    let Ok(base_url) = std::env::var("FA_TEST_BASE_URL") else {
      eprintln!("FA_TEST_BASE_URL not set; skipping");
      return;
    };
    let cfg = Config {
      provider: Provider::OpenAi,
      base_url: Some(base_url),
      api_key: "test".into(),
      model: "mock".into(),
      system_prompt: None,
      max_turns: 5,
      compaction: TEST_SETTINGS,
      vision: true,
      tools: None,
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let agents = build_agents(&cfg, Path::new("/tmp"), tx.clone(), &Default::default()).unwrap();
    let history = vec![
      Message::user("first question"),
      Message::assistant("first answer"),
      Message::user("second question"),
      Message::assistant("second answer"),
    ];
    start_compaction(agents.summarizer.clone(), history, TEST_SETTINGS, tx.clone())
      .await
      .unwrap();
    let compacted = match rx.recv().await {
      Some(AgentEvent::Compacted(Some(compacted))) => compacted,
      other => panic!("expected Compacted event, got {other:?}"),
    };
    assert!(compacted.summary.contains("Mock summary"), "{}", compacted.summary);
    assert_eq!((compacted.summarized, compacted.kept), (2, 2));
    assert_eq!(compacted.history.len(), 3);
    assert_eq!(compacted.history[0], compaction::summary_message(&compacted.summary));
    assert_eq!(compacted.history[1], Message::user("second question"));

    // With a single turn left after the summary and a budget it fits in,
    // there is nothing to compact.
    let roomy = Settings {
      keep_recent_tokens: 1000,
      ..TEST_SETTINGS
    };
    start_compaction(agents.summarizer.clone(), compacted.history, roomy, tx.clone())
      .await
      .unwrap();
    let Some(AgentEvent::Compacted(None)) = rx.recv().await else {
      panic!("expected nothing to compact");
    };

    // A tight budget forces the remaining turn into the existing summary:
    // the update path, which feeds the previous summary back to the model.
    let history = vec![
      compaction::summary_message("## Goal\nOld summary"),
      Message::user("third question"),
      Message::assistant("third answer"),
    ];
    start_compaction(agents.summarizer.clone(), history, TEST_SETTINGS, tx.clone())
      .await
      .unwrap();
    let compacted = match rx.recv().await {
      Some(AgentEvent::Compacted(Some(compacted))) => compacted,
      other => panic!("expected Compacted event, got {other:?}"),
    };
    assert_eq!((compacted.summarized, compacted.kept), (2, 0));
    assert_eq!(compacted.history.len(), 1, "one fresh summary, not nested");
  }

  #[test]
  fn the_prompt_stops_speaking_for_a_tool_the_session_does_not_have() {
    let all = default_system_prompt(Path::new("/work"), None);
    for tool in crate::tools::BUILT_IN {
      assert!(all.contains(&format!("- {tool}:")), "{tool} is introduced by default");
    }

    // A model told about `bash` and then refused it tries anyway and reports
    // being refused, instead of using what it does have.
    let reading = default_system_prompt(Path::new("/work"), Some(&["read".to_string()]));
    assert!(reading.contains("- read: Read file contents"));
    for gone in [
      "- bash:",
      "- edit:",
      "- write:",
      "Use bash for",
      "Use edit for",
      "Use write only",
      // Including where the rules speak of a tool's arguments rather than of
      // the tool by name.
      "edits[].oldText",
    ] {
      assert!(!reading.contains(gone), "{gone:?} is not this session's: {reading}");
    }
    // What does not speak for a tool stays, and so does the rest of it.
    assert!(reading.contains("- Be concise in your responses"));
    assert!(reading.contains("Use read to examine files"));
    assert!(reading.contains("<cwd>\n/work\n</cwd>"));

    // A list that leaves the built-in four alone changes nothing, however
    // many other tools it names.
    let with_mcp = default_system_prompt(
      Path::new("/work"),
      Some(crate::tools::BUILT_IN.map(str::to_string).as_ref()),
    );
    assert_eq!(with_mcp, all);
  }

  #[test]
  fn relay_moves_tool_images_into_a_following_user_message() {
    use rig_core::message::{ToolCallId, ToolResult};
    let result = |content: Vec<ToolResultContent>| {
      UserContent::ToolResult(ToolResult {
        call: ToolCallId::mint(),
        provider: None,
        name: "read".into(),
        content,
      })
    };
    let mut history = vec![
      Message::user("look"),
      Message::assistant("reading"),
      Message::User {
        content: vec![
          result(vec![ToolResultContent::text("plain")]),
          result(vec![ToolResultContent::image_base64(
            "AAAA",
            Some(rig_core::message::ImageMediaType::PNG),
            None,
          )]),
        ],
      },
      Message::assistant("done"),
    ];
    let untouched = history[..2].to_vec();
    relay_tool_images(&mut history);
    assert_eq!(history.len(), 5);
    assert_eq!(&history[..2], &untouched[..]);
    let Message::User { content } = &history[2] else {
      panic!("tool results stay a user message");
    };
    let texts: Vec<Option<&str>> = content
      .iter()
      .map(|c| match c {
        UserContent::ToolResult(r) => r.content[0].as_text(),
        _ => None,
      })
      .collect();
    assert_eq!(texts, [Some("plain"), Some("(see attached image)")]);
    let Message::User { content } = &history[3] else {
      panic!("relay message follows the tool results");
    };
    assert!(matches!(&content[0], UserContent::Text(t) if t.text == "Attached image(s) from tool result:"));
    assert!(matches!(&content[1], UserContent::Image(_)));
    assert_eq!(history[4], Message::assistant("done"));

    // Nothing to do for text-only histories.
    let mut plain = vec![Message::user("a"), Message::assistant("b")];
    let before = plain.clone();
    relay_tool_images(&mut plain);
    assert_eq!(plain, before);
  }

  #[tokio::test]
  async fn mock_image_read() {
    let Ok(base_url) = std::env::var("FA_TEST_BASE_URL") else {
      eprintln!("FA_TEST_BASE_URL not set; skipping");
      return;
    };
    let path = std::env::temp_dir().join("fa-test-image.png");
    let img = image::ImageBuffer::from_fn(8, 8, |x, _| image::Rgb([x as u8 * 30, 0, 0]));
    img.save(&path).unwrap();

    let cfg = Config {
      provider: Provider::OpenAi,
      base_url: Some(base_url),
      api_key: "test".into(),
      model: "mock".into(),
      system_prompt: None,
      max_turns: 5,
      compaction: TEST_SETTINGS,
      vision: true,
      tools: None,
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let agent = build_agents(&cfg, Path::new("/tmp"), tx.clone(), &Default::default())
      .unwrap()
      .agent;
    // The mock turns "image <path>" into a `read` tool call for that path,
    // then answers the follow-up request with plain text. If the relay did
    // not work, rig would reject the image in the tool message and the run
    // would end with an error instead of a Done.
    let events = collect(&agent, vec![], &format!("image {}", path.display()), &mut rx, &tx).await;
    let result = events
      .iter()
      .find_map(|e| match e {
        AgentEvent::ToolResult { output, is_error, .. } => Some((output.clone(), *is_error)),
        _ => None,
      })
      .expect("tool result");
    assert_eq!(result, ("Read image file [image/png]\n[image]".to_string(), false));
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Error(_))), "{events:?}");
    let Some(AgentEvent::Done { messages, .. }) = events.last() else {
      panic!("no Done")
    };
    assert_eq!(messages.len(), 4, "{messages:?}");
  }

  /// Same scenario as `mock_image_read`, but through the Gemini provider,
  /// where the image travels inside the function response with no relay.
  #[tokio::test]
  async fn mock_gemini_image_read() {
    let Ok(base_url) = std::env::var("FA_TEST_GEMINI_BASE_URL") else {
      eprintln!("FA_TEST_GEMINI_BASE_URL not set; skipping");
      return;
    };
    let path = std::env::temp_dir().join("fa-test-image-gemini.png");
    let img = image::ImageBuffer::from_fn(8, 8, |x, _| image::Rgb([0, x as u8 * 30, 0]));
    img.save(&path).unwrap();

    let cfg = Config {
      provider: Provider::Gemini,
      base_url: Some(base_url),
      api_key: "test".into(),
      model: "mock".into(),
      system_prompt: None,
      max_turns: 5,
      compaction: TEST_SETTINGS,
      vision: true,
      tools: None,
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let agent = build_agents(&cfg, Path::new("/tmp"), tx.clone(), &Default::default())
      .unwrap()
      .agent;
    let events = collect(&agent, vec![], &format!("image {}", path.display()), &mut rx, &tx).await;
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Error(_))), "{events:?}");
    let text: String = events
      .iter()
      .filter_map(|e| match e {
        AgentEvent::Text(t) => Some(t.as_str()),
        _ => None,
      })
      .collect();
    // The mock answers with what it found inside the functionResponse.
    assert_eq!(text, "inline image/png in functionResponse");
    let Some(AgentEvent::Done { messages, .. }) = events.last() else {
      panic!("no Done")
    };
    assert_eq!(messages.len(), 4, "{messages:?}");
  }
}
