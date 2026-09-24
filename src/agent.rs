//! The run loop, and what it is built from.
//!
//! The TUI never touches rig directly: it starts a run, reads `AgentEvent`s
//! off a channel, and stops or steers the run through a [`Control`].
//!
//! The loop here owns the conversation as it grows. That is the whole of the
//! design: a run that is interrupted, or added to while it goes, leaves the
//! messages behind the interruption exactly as the model gave them, because
//! they were never anywhere else to be rebuilt from. Rig supplies the two
//! hard parts — the provider wires, and the tools — and nothing in between.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use rig_agent::ModelHandle;
use rig_agent::tool::ToolContext;
use rig_agent::tool::server::{ToolServer, ToolServerHandle};
use rig_core::client::Nothing;
use rig_core::client::completion::CompletionClient;
use rig_core::client::model_listing::ModelListingClient;
use rig_core::completion::{CompletionError, CompletionModel, Message, ToolDefinition, Usage};
use rig_core::message::{AssistantContent, ToolCall, ToolResultContent, UserContent};
use rig_core::providers::{
  anthropic, cohere, deepseek, doubleword, gemini, groq, hyperbolic, llamafile, mira, mistral, ollama, openai,
  openrouter, perplexity, together, venice, xai,
};
use rig_core::streaming::{StreamedAssistantContent, StreamingCompletionResponse, ToolCallDeltaContent};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use crate::attach::Prompt;
use crate::compaction::{self, Compacted, Settings};
use crate::modal::{Host, Modal};
use crate::tools::{AskTool, BUILT_IN, BashTool, EditDiff, EditTool, Output, ReadTool, WriteTool};

/// What a tool result says when the run was stopped before it could be run.
///
/// The model asked for the call, so the conversation owes an answer — a call
/// left hanging is one no provider will take back, which would leave the
/// session unable to carry on from what it just kept.
pub const ABORTED: &str = "Aborted by the user.";

/// Events streamed from a run to the UI.
#[derive(Debug)]
pub enum AgentEvent {
  Text(String),
  Reasoning(String),
  /// A message the user typed mid-run, read by the run at the top of a turn.
  /// The UI has been drawing it as waiting; this is where it becomes part of
  /// the conversation.
  Steered {
    text: String,
    /// Images it attached, for the transcript to draw under it.
    images: Vec<Vec<u8>>,
    /// What an MCP server wrote out from it, which is what was sent.
    expanded: Vec<Message>,
  },
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
    /// Images the tool answered with, drawn under its output.
    images: Vec<Vec<u8>>,
    is_error: bool,
    /// The call this answers, as the transcript will name it.
    call: String,
    /// Numbered diff for `edit`, shown in place of the output text.
    diff: Option<String>,
  },
  /// What one model call cost, sent as soon as that call comes back rather
  /// than when the run ends: a run of twenty turns moves the counters twenty
  /// times, and a run that never reaches an answer still says what it spent.
  /// `context_tokens` is how big that request was — the provider's own count
  /// when it gives one, and what the request was weighed at before it went
  /// out when it does not.
  Usage {
    usage: Usage,
    context_tokens: u64,
  },
  /// The run reached its answer. `messages` is everything it added to the
  /// conversation: the prompt, every turn, and the answer.
  Done {
    messages: Vec<Message>,
  },
  /// Something wants the user to answer it — the `ask` tool, or an MCP
  /// server. The modal already knows where its answer goes; dropping it
  /// unfinished is no answer, which is what an aborted run leaves behind.
  Show(Box<dyn Modal>),
  /// The same, for the UI itself rather than a run: shown whether or not
  /// one is going.
  #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
  Ask(Box<dyn Modal>),
  /// The run stopped short of an answer — cancelled, out of room, or after
  /// an `Error`. `messages` holds the same thing `Done` carries: everything
  /// it got through, as it was sent.
  Ended {
    messages: Vec<Message>,
  },
  /// Compaction finished; `None` means there was nothing to compact.
  Compacted(Option<Compacted>),
  /// What the provider says it offers, or why it would not say. Sent by the
  /// fetch `/model` starts, which is the only thing that asks.
  Models(Result<Vec<ModelInfo>, String>),
  /// What sending an MCP server's prompt, typed as `text`, came to: the
  /// messages the server wrote out, `None` when the user put away the form
  /// that asked for its arguments, or why it could not be had.
  #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
  Expanded {
    text: String,
    result: Result<Option<Vec<Message>>, String>,
  },
  /// What a server offered for a value being typed, asked for by the popup.
  #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
  Suggested {
    asking: crate::mcp::Completing,
    result: Result<crate::mcp::Suggestions, String>,
  },
  /// Something about an MCP server while the session ran — its tools
  /// changed, or what it holds could not be listed: what to say about it.
  #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
  Mcp(String),
  Error(String),
}

/// What a session runs on: one model with its tools, and the handle the UI
/// stops and steers it by.
///
/// Everything in the runtime but the model — the tools, the preamble, the
/// queue — outlives whichever model is in front of it, which is what lets
/// `/model` swap one for another mid-session.
pub struct Agents {
  pub runtime: Arc<Runtime>,
  pub control: Control,
}

impl Agents {
  /// Point the session at `cfg.model`. The runtime is not replaced until the
  /// new model is built, so one that cannot be reached leaves the session on
  /// the one it already had.
  pub fn use_model(&mut self, cfg: &Config) -> Result<()> {
    let old = &self.runtime;
    self.runtime = Arc::new(Runtime::new(
      cfg,
      old.tools.clone(),
      old.preamble.clone(),
      old.relay_images,
    )?);
    Ok(())
  }
}

/// One model a provider offers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
  pub id: String,
  /// The provider's own display name for it, when it gives one.
  pub name: Option<String>,
  /// The context window it reports, which most providers do not.
  pub context_length: Option<u64>,
}

// ---------------------------------------------------------------- control

/// What the UI says to a run while it is going.
///
/// Cancelling and steering are the two things it has to say, and both have
/// to reach a run that is in the middle of something — so each is something
/// the loop reads at its next opportunity, and cancelling can also be waited
/// on by the parts that cannot wait for whatever the loop is blocked on.
#[derive(Clone, Default)]
pub struct Control(Arc<Signals>);

#[derive(Default)]
struct Signals {
  /// Esc, as a value the run can wait to see change rather than only read.
  cancelled: watch::Sender<bool>,
  /// Raised by the run when the conversation outgrew the window. The UI
  /// takes it back down once it has made the room.
  overflow: AtomicBool,
  /// What the user typed while the run was going, for the run to read at
  /// the top of its next turn. Whole prompts rather than their text: an
  /// attachment is read when Enter is pressed, not whenever the run gets
  /// round to it.
  steer: Mutex<VecDeque<Prompt>>,
}

impl Control {
  /// Start a run with nothing held against it — but keep anything typed
  /// while the last one was ending, which was meant for this one.
  fn begin(&self) {
    self.0.cancelled.send_replace(false);
  }

  /// Stop the run at the first thing it can stop in the middle of.
  pub fn cancel(&self) {
    self.0.cancelled.send_replace(true);
  }

  pub fn cancelled(&self) -> bool {
    *self.0.cancelled.borrow()
  }

  /// Resolves once the run should stop what it is doing — at once, when it
  /// already should.
  async fn stopped(&self) {
    // The sender lives as long as `self`, so the wait can only end by the
    // value turning true.
    let _ = self.0.cancelled.subscribe().wait_for(|&cancelled| cancelled).await;
  }

  /// Hand the run something the user typed while it was going.
  ///
  /// This is where a waiting message lives — there is no second copy of it
  /// anywhere. Whoever gets to it first takes it: the run, at the top of
  /// its next turn, or the UI once there is no run left to give it to.
  pub fn steer(&self, prompt: Prompt) {
    self.held().push_back(prompt);
  }

  /// Take the newest back, for the input box to have it again.
  pub fn unsteer(&self) -> Option<Prompt> {
    self.held().pop_back()
  }

  /// Take the oldest, for the UI to send as a run of its own.
  pub fn take_next(&self) -> Option<Prompt> {
    self.held().pop_front()
  }

  /// Take everything, in the order it was typed: the run reads them into
  /// the turn it is about to ask for, and Esc takes them out of its way.
  pub fn take(&self) -> Vec<Prompt> {
    self.held().drain(..).collect()
  }

  pub fn steering(&self) -> bool {
    !self.held().is_empty()
  }

  /// What is waiting, for the screen to say so.
  pub fn waiting(&self) -> Vec<String> {
    self.held().iter().map(|prompt| prompt.text.clone()).collect()
  }

  fn held(&self) -> std::sync::MutexGuard<'_, VecDeque<Prompt>> {
    self.0.steer.lock().expect("a queue nobody panicked holding")
  }

  fn overflowed_now(&self) {
    self.0.overflow.store(true, Ordering::Relaxed);
  }

  /// Whether the run found the context window full — and clear the mark,
  /// since answering it is the UI's half of the bargain.
  pub fn overflowed(&self) -> bool {
    self.0.overflow.swap(false, Ordering::Relaxed)
  }
}

// ------------------------------------------------------------------ model

/// The tools a session offers, as they are now: the five it was built with,
/// and whatever its MCP servers offer at the moment.
///
/// A server that changes what it offers has the whole set built again and
/// put in place of this one. A run takes the set as it is when it asks, so a
/// call already under way finishes on the set it started with — which calls
/// the same servers.
#[derive(Clone)]
pub struct Tools(Arc<std::sync::RwLock<ToolServerHandle>>);

impl Tools {
  pub fn new(handle: ToolServerHandle) -> Self {
    Self(Arc::new(std::sync::RwLock::new(handle)))
  }

  fn now(&self) -> ToolServerHandle {
    self.0.read().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
  }

  fn replace(&self, handle: ToolServerHandle) {
    *self.0.write().unwrap_or_else(|poisoned| poisoned.into_inner()) = handle;
  }
}

/// Everything a run needs that does not change between runs.
pub struct Runtime {
  /// Whichever provider it came from: the rest of the file does not care.
  model: ModelHandle,
  tools: Tools,
  preamble: String,
  compaction: Settings,
  /// Which tools this session was held to.
  rules: crate::tools::Rules,
  /// Chat completions will not carry an image inside a tool message, so a
  /// `read` that answers with a screenshot has it relayed after the result.
  relay_images: bool,
  /// Cap on one answer, left to the provider when `None`.
  max_tokens: Option<u64>,
}

impl Runtime {
  fn new(cfg: &Config, tools: Tools, preamble: String, relay_images: bool) -> Result<Self> {
    Ok(Self {
      model: build_model(cfg)?,
      tools,
      preamble,
      compaction: cfg.compaction,
      rules: cfg.tools.clone(),
      relay_images,
      max_tokens: cfg.max_tokens,
    })
  }

  /// One question to the model with no tools and the summarizer's preamble,
  /// and its answer: how compaction summarizes. Nothing here is a
  /// conversation.
  pub async fn ask(&self, prompt: String) -> Result<String, CompletionError> {
    let response = self
      .model
      .completion_request(Message::user(prompt))
      .preamble(compaction::SYSTEM_PROMPT.to_string())
      .max_tokens_opt(self.max_tokens)
      .send()
      .await?;
    Ok(
      response
        .choice
        .iter()
        .filter_map(|content| match content {
          AssistantContent::Text(text) => Some(text.text.as_str()),
          _ => None,
        })
        .collect::<Vec<_>>()
        .join(""),
    )
  }
}

/// Which API to speak. The names are the `--provider` values, and what the
/// provider is called on screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Provider {
  /// OpenAI Chat Completions and compatible servers
  #[value(name = "openai")]
  OpenAi,
  /// OpenRouter
  #[value(name = "openrouter")]
  OpenRouter,
  /// Ollama, on http://localhost:11434 by default
  Ollama,
  /// Google Gemini
  Gemini,
  /// Anthropic
  Anthropic,
  /// Cohere
  Cohere,
  /// DeepSeek
  #[value(name = "deepseek")]
  DeepSeek,
  /// Doubleword
  Doubleword,
  /// Groq
  Groq,
  /// Hyperbolic
  Hyperbolic,
  /// A llamafile server, on http://localhost:8080 by default
  Llamafile,
  /// Mira
  Mira,
  /// Mistral
  Mistral,
  /// Perplexity
  Perplexity,
  /// Together AI
  Together,
  /// Venice
  Venice,
  /// xAI
  #[value(name = "xai")]
  XAi,
}

impl Provider {
  /// What this provider is called on screen: its `--provider` value.
  pub fn label(self) -> String {
    clap::ValueEnum::to_possible_value(&self)
      .expect("no provider is skipped")
      .get_name()
      .to_string()
  }

  /// The environment variable the key is read from when `--api-key` is not
  /// given.
  pub fn key_env(self) -> String {
    format!("{}_API_KEY", self.label().to_uppercase())
  }

  /// The key to use when none is given. Ollama and llamafile want no key at
  /// all — rig leaves the header off for an empty one, which is what a local
  /// server expects — while a hosted endpoint that ignores auth is happy with
  /// anything.
  pub fn no_key(self) -> &'static str {
    match self {
      Self::Ollama | Self::Llamafile => "",
      _ => "none",
    }
  }
}

#[derive(Clone)]
pub struct Config {
  pub provider: Provider,
  /// Provider endpoint root; `None` uses the provider's default.
  pub base_url: Option<String>,
  pub api_key: String,
  /// The model to send to, as `--model` named it or `/model` chose it.
  pub model: String,
  pub system_prompt: Option<String>,
  /// Cap on what one answer may come to, or `None` for the provider's own.
  pub max_tokens: Option<u64>,
  pub compaction: Settings,
  /// Whether the model accepts image input.
  pub vision: bool,
  /// Which tools this session offers.
  pub tools: crate::tools::Rules,
}

/// Whether this provider takes an image inside a tool result.
///
/// Gemini takes one inside a function response and Anthropic inside a tool
/// result; the rest refuse them there and have them relayed after it.
fn relays_images(provider: Provider) -> bool {
  !matches!(provider, Provider::Gemini | Provider::Anthropic)
}

/// A provider's client, with the configured key — or the one given, since
/// llamafile's client takes no key at all and says so in its type — and the
/// base URL when there is one. Each builder is a type of its own, so this is
/// spelled once here rather than once per provider.
macro_rules! client {
  ($cfg:expr, $builder:expr, $what:literal) => {
    client!($cfg, $builder, $what, $cfg.api_key.as_str())
  };
  ($cfg:expr, $builder:expr, $what:literal, $key:expr) => {{
    let mut builder = $builder.api_key($key);
    if let Some(url) = $cfg.base_url.as_deref() {
      builder = builder.base_url(url.trim_end_matches('/'));
    }
    builder.build().context(concat!("failed to build ", $what, " client"))?
  }};
}

/// The handle `cfg.model` is reached through, whichever provider offers it.
///
/// Summarizing asks the same model the same way, so it is the same handle —
/// what makes that call a summary is the preamble it carries, which belongs
/// to the request rather than to the model.
fn build_model(cfg: &Config) -> Result<ModelHandle> {
  macro_rules! model {
    ($($client:tt)*) => {
      ModelHandle::new(client!(cfg, $($client)*).completion_model(cfg.model.clone()))
    };
  }

  Ok(match cfg.provider {
    Provider::OpenAi => model!(openai::CompletionsClient::builder(), "OpenAI-compatible"),
    Provider::OpenRouter => model!(openrouter::Client::builder(), "OpenRouter"),
    Provider::Ollama => model!(ollama::Client::builder(), "Ollama"),
    Provider::Gemini => model!(gemini::Client::builder(), "Gemini"),
    // Anthropic refuses a request that names no `max_tokens`, so the model is
    // built through `with_model` rather than `completion_model`: it fills in
    // what the named model allows, and falls back to a small cap for one it
    // does not know — which is what `--max-tokens` is for, since a figure on
    // the request wins over the model's own.
    Provider::Anthropic => ModelHandle::new(anthropic::completion::CompletionModel::with_model(
      client!(cfg, anthropic::Client::builder(), "Anthropic"),
      &cfg.model,
    )),
    Provider::Cohere => model!(cohere::Client::builder(), "Cohere"),
    Provider::DeepSeek => model!(deepseek::Client::builder(), "DeepSeek"),
    Provider::Doubleword => model!(doubleword::Client::builder(), "Doubleword"),
    Provider::Groq => model!(groq::Client::builder(), "Groq"),
    Provider::Hyperbolic => model!(hyperbolic::Client::builder(), "Hyperbolic"),
    Provider::Llamafile => model!(llamafile::Client::builder(), "llamafile", Nothing),
    Provider::Mira => model!(mira::Client::builder(), "Mira"),
    Provider::Mistral => model!(mistral::Client::builder(), "Mistral"),
    Provider::Perplexity => model!(perplexity::Client::builder(), "Perplexity"),
    Provider::Together => model!(together::Client::builder(), "Together"),
    Provider::Venice => model!(venice::Client::builder(), "Venice"),
    Provider::XAi => model!(xai::Client::builder(), "xAI"),
  })
}

/// Ask the provider what models it offers, alphabetically.
///
/// Only four of them report a context window — Gemini, Groq, Mistral and
/// OpenRouter — and the rest answer with names alone, which is why the window
/// a model is given falls back to the configured one.
pub async fn list_models(cfg: &Config) -> Result<Vec<ModelInfo>> {
  macro_rules! list {
    ($builder:expr, $what:literal) => {
      client!(cfg, $builder, $what)
        .list_models()
        .await
        .context(concat!($what, " would not say what models it has"))?
    };
  }

  let models = match cfg.provider {
    Provider::OpenAi => list!(openai::CompletionsClient::builder(), "OpenAI-compatible"),
    Provider::OpenRouter => list!(openrouter::Client::builder(), "OpenRouter"),
    Provider::Ollama => list!(ollama::Client::builder(), "Ollama"),
    Provider::Gemini => list!(gemini::Client::builder(), "Gemini"),
    Provider::Anthropic => list!(anthropic::Client::builder(), "Anthropic"),
    Provider::DeepSeek => list!(deepseek::Client::builder(), "DeepSeek"),
    Provider::Groq => list!(groq::Client::builder(), "Groq"),
    Provider::Mira => list!(mira::Client::builder(), "Mira"),
    Provider::Mistral => list!(mistral::Client::builder(), "Mistral"),
    Provider::Venice => list!(venice::Client::builder(), "Venice"),
    // The rest have no listing endpoint rig speaks. Refused here rather than
    // by a second list of which providers have an arm above: this is the
    // list, and it answers without a request going out.
    other => bail!("{} does not list its models — name one with /model <id>", other.label()),
  };

  let mut models: Vec<ModelInfo> = models
    .into_iter()
    .map(|model| ModelInfo {
      // A name that only repeats the id is no more than the id.
      name: model.name.filter(|name| name != &model.id),
      id: model.id,
      context_length: model.context_length.map(u64::from),
    })
    .collect();
  // A provider's own order is whatever it is — newest first, or the order
  // they were added. One long list is easier to walk in a known one.
  models.sort_by(|a, b| a.id.cmp(&b.id));
  models.dedup_by(|a, b| a.id == b.id);
  Ok(models)
}

pub fn build_agents(cfg: &Config, cwd: &Path, host: &Host, servers: &crate::mcp::Servers) -> Result<Agents> {
  let preamble = match &cfg.system_prompt {
    Some(p) => p.clone(),
    None => default_system_prompt(cwd, &cfg.tools),
  };

  let catalog = servers.catalog();
  let (cwd, vision, host) = (cwd.to_path_buf(), cfg.vision, host.clone());
  #[cfg(feature = "mcp")]
  let servers = catalog.clone();
  let built_in = move || {
    let tools = ToolServer::new()
      .tool(ReadTool {
        cwd: cwd.clone(),
        vision,
      })
      .tool(WriteTool { cwd: cwd.clone() })
      .tool(EditTool { cwd: cwd.clone() })
      .tool(BashTool { cwd: cwd.clone() })
      .tool(AskTool { host: host.clone() });
    // Reading what the servers hold, when one of them holds anything: a
    // session with no such server is not offered tools with nothing behind
    // them.
    #[cfg(feature = "mcp")]
    let tools = match servers.has_resources() {
      true => tools
        .tool(crate::resources::ListResources {
          catalog: servers.clone(),
        })
        .tool(crate::resources::ReadResource {
          catalog: servers.clone(),
          vision,
        }),
      false => tools,
    };
    tools
  };
  // Whatever the session's MCP servers offer, alongside the five the agent
  // brought: a tool is a tool, and the transcript draws them all the same.
  let tools = Tools::new(catalog.attach(built_in()).run());
  // And whatever they offer later, in place of what they offered before.
  let again = tools.clone();
  catalog.on_change(move |catalog| again.replace(catalog.attach(built_in()).run()));

  Ok(Agents {
    runtime: Arc::new(Runtime::new(cfg, tools, preamble, relays_images(cfg.provider))?),
    control: Control::default(),
  })
}

// ------------------------------------------------------------------- loop

/// What the last answered request cost, and how much of the conversation
/// that figure covers.
///
/// A provider counts the request it was sent, so its figure is the truth
/// about everything up to the answer it gave — and says nothing about what a
/// tool has returned since. Holding the two apart is what keeps the guessing
/// down to the few messages nobody has counted yet.
#[derive(Default)]
struct Weigh {
  /// The provider's own count for the request it has answered, or 0 while
  /// it has answered none — or reports nothing.
  reported: u64,
  /// How many messages that count covers: everything the request carried,
  /// and the answer its output tokens paid for.
  counted: usize,
}

impl Weigh {
  /// What the request about to go out comes to: the provider's own count of
  /// everything it has already weighed, plus a chars/4 estimate of what has
  /// been said since. Nothing it has counted is guessed at, which is as
  /// close as a client gets without a tokenizer of its own.
  fn request(&self, chat: &[Message]) -> u64 {
    self.reported + compaction::estimate_tokens(&chat[self.counted.min(chat.len())..])
  }

  /// Take the provider's word for what the answered request held. One that
  /// reports nothing leaves the last figure standing.
  fn answered(&mut self, usage: &Usage, sent: usize) {
    let reported = weight(usage);
    if reported > 0 {
      self.reported = reported;
      self.counted = sent + 1;
    }
  }
}

/// What a provider says the request it answered came to, prompt and answer
/// together.
///
/// `input_tokens` is only the part of the prompt a provider charged as new.
/// Anthropic counts what it read back from its cache, and what it wrote
/// there, under fields of their own — so on a conversation that is mostly
/// cached, which is what a long one becomes, the prompt and the answer added
/// up come to a small fraction of what was actually sent. `total_tokens` is
/// the figure the provider reports, or the one rig adds up from the parts for
/// the providers that report only parts, and it is the whole request wherever
/// there is one. The sum stands in where there is not.
fn weight(usage: &Usage) -> u64 {
  usage.total_tokens.max(usage.input_tokens + usage.output_tokens)
}

/// Why a run stopped short of an answer.
enum Stop {
  /// Esc. What it got through is kept, and the calls it was in the middle
  /// of are answered.
  Cancelled,
  /// The conversation outgrew the window. The UI makes room and picks the
  /// run back up where it left off.
  Overflow,
  /// Something the user should be told: the turn budget, or a provider or
  /// tool that would not.
  Failed(String),
}

/// Start one run in the background.
///
/// `prompt` is the message the run opens with, or `None` for `/continue`,
/// which picks the loop back up over the history as it stands — an
/// unanswered message is answered, a half-written turn is carried on.
/// Everything the run adds, that prompt included, comes back in the `Done`
/// or `Ended` that ends it, and nothing that was already in the history
/// does: what is handed back is only ever new.
pub fn start_run(
  runtime: Arc<Runtime>,
  control: Control,
  history: Vec<Message>,
  mut made: Vec<Message>,
  tx: mpsc::UnboundedSender<AgentEvent>,
) -> JoinHandle<()> {
  tokio::spawn(async move {
    control.begin();
    let event = match run(&runtime, &control, &history, &mut made, &tx).await {
      None => AgentEvent::Done { messages: made },
      Some(stop) => {
        if let Stop::Failed(err) = stop {
          let _ = tx.send(AgentEvent::Error(err));
        }
        AgentEvent::Ended { messages: made }
      }
    };
    let _ = tx.send(event);
  })
}

/// The loop itself. `made` grows with everything the run adds, and is the
/// caller's however it ends.
async fn run(
  rt: &Runtime,
  control: &Control,
  history: &[Message],
  made: &mut Vec<Message>,
  tx: &mpsc::UnboundedSender<AgentEvent>,
) -> Option<Stop> {
  let mut weigh = Weigh::default();

  let mut first = true;

  loop {
    // Anything typed while the last turn ran is read before this one, so it
    // lands where the user meant it rather than after the whole answer.
    for prompt in control.take() {
      let _ = tx.send(AgentEvent::Steered {
        text: prompt.text.clone(),
        images: prompt.preview(),
        expanded: prompt.expanded.clone(),
      });
      made.extend(prompt.messages());
    }
    if control.cancelled() {
      return Some(Stop::Cancelled);
    }

    let mut chat = Vec::with_capacity(history.len() + made.len());
    chat.extend_from_slice(history);
    chat.extend(made.iter().cloned());
    if rt.relay_images {
      relay_tool_images(&mut chat);
    }

    // Weighed before it is sent rather than once the answer comes back: a
    // tool can return a file the size of the window, and the request
    // carrying it is the one that would be refused.
    let context = weigh.request(&chat);
    if compaction::should_compact(context, &rt.compaction) {
      control.overflowed_now();
      // Never before the first call, where the run has done nothing yet:
      // it would be handed the same conversation again and stop again.
      if !first {
        return Some(Stop::Overflow);
      }
    }
    first = false;

    // Asked again every turn, since an MCP server can change what it offers
    // in the middle of a run — as the answer to one of its own tools, even.
    let definitions = match rt.definitions().await {
      Ok(definitions) => definitions,
      Err(err) => return Some(Stop::Failed(err)),
    };
    let sent = chat.len();
    // The builder takes the newest message apart from the rest.
    let Some(prompt) = chat.pop() else {
      return Some(Stop::Failed("there is nothing to answer".into()));
    };
    let request = rt
      .model
      .completion_request(prompt)
      .preamble(rt.preamble.clone())
      .messages(chat)
      .tools(definitions)
      .max_tokens_opt(rt.max_tokens);
    let mut stream = match request.stream().await {
      Ok(stream) => stream,
      Err(err) => return Some(Stop::Failed(err.to_string())),
    };

    let mut partial = Partial::default();
    loop {
      let stopped = tokio::select! {
        biased;
        () = control.stopped() => None,
        item = stream.next() => Some(item),
      };
      // Nothing of this turn has run, so the half of it that was written is
      // kept for the transcript's sake and no more.
      let Some(item) = stopped else {
        made.extend(cut_off(&mut stream).await);
        return Some(Stop::Cancelled);
      };
      let Some(item) = item else { break };
      let content = match item {
        Ok(content) => content,
        Err(err) => {
          made.extend(cut_off(&mut stream).await);
          return Some(Stop::Failed(err.to_string()));
        }
      };
      if let Some(event) = partial.saw(content) {
        let _ = tx.send(event);
      }
    }

    // Which line each call was written on, for the dispatch to take away.
    let written = partial.written;
    // The turn as the provider gave it, aggregated by rig: the text, the
    // reasoning, and the calls, in the order they were said. This is the
    // copy that goes into the conversation and the copy the next request
    // replays — there is only ever the one.
    let choice = std::mem::take(&mut stream.choice);
    let usage = stream.usage();
    weigh.answered(&usage, sent);
    let _ = tx.send(AgentEvent::Usage {
      usage,
      context_tokens: weigh.reported.max(context),
    });
    if choice.is_empty() {
      return Some(Stop::Failed("the model answered with nothing".into()));
    }
    let calls: Vec<ToolCall> = choice
      .iter()
      .filter_map(|content| match content {
        AssistantContent::ToolCall(call) => Some(call.clone()),
        _ => None,
      })
      .collect();
    made.push(Message::Assistant {
      id: stream.message_id.clone(),
      content: choice,
    });

    if calls.is_empty() {
      // The answer — unless something was typed while it was being
      // written, which the run reads rather than making it be sent again.
      match control.steering() {
        true => continue,
        false => return None,
      }
    }

    // In the order the model asked for them: a conversation replayed is
    // the same conversation, which is worth more than the overlap.
    let mut results = Vec::with_capacity(calls.len());
    for call in &calls {
      // Biased, so a run already stopped answers the call rather than
      // starting it: the one it was in the middle of and the ones it never
      // reached are answered the same way, and neither is left hanging.
      results.push(tokio::select! {
        biased;
        () = control.stopped() => aborted(call),
        answer = rt.call(call, written.get(call.id.as_str()), tx) => answer,
      });
    }
    made.push(Message::User { content: results });
    if control.cancelled() {
      return Some(Stop::Cancelled);
    }
  }
}

/// What the UI is told about a turn while it streams, and what it has to be
/// told again once the turn's calls are run.
#[derive(Default)]
struct Partial {
  /// Arguments arrive a few characters at a time and each fragment carries
  /// only what is new, so this holds what has arrived per call and the UI is
  /// sent the whole of it, with nothing to reassemble.
  writing: HashMap<String, (String, String)>,
  /// The id rig correlated a call's fragments under, by the id the provider
  /// gave the finished call.
  ///
  /// The two are not the same — rig mints its own so a call stays followable
  /// before the provider has named it — and the line the call was written on
  /// is keyed by rig's. Without the pairing, dispatching the call would
  /// leave that line on screen with nothing to take it away.
  written: HashMap<String, String>,
}

impl Partial {
  /// What the UI should be told about what the stream said.
  fn saw(&mut self, content: StreamedAssistantContent) -> Option<AgentEvent> {
    match content {
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
        let (name, args) = self.writing.entry(internal_call_id.clone()).or_default();
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
      // Written but not yet run — the loop reports it when it starts it.
      // Showing it whole in the meantime is what the finished line will
      // say, so nothing jumps when the two swap over.
      StreamedAssistantContent::ToolCall {
        tool_call,
        internal_call_id,
      } => {
        self
          .written
          .insert(tool_call.id.as_str().to_string(), internal_call_id.clone());
        Some(AgentEvent::ToolCallDelta {
          id: internal_call_id,
          name: tool_call.function.name,
          args: tool_call.function.arguments.to_string(),
        })
      }
      _ => None,
    }
  }
}

/// What a turn stopped part-way had said, if anything, as rig put it
/// together: ending the stream is what has it finish the turn from what had
/// arrived, reasoning signatures and all.
///
/// The calls are left out, finished or not — a call nobody ran is one
/// nothing would answer — and so is text with nothing in it.
async fn cut_off(stream: &mut StreamingCompletionResponse) -> Option<Message> {
  stream.cancel();
  while stream.next().await.is_some() {}
  let content: Vec<AssistantContent> = std::mem::take(&mut stream.choice)
    .into_iter()
    .filter(|content| match content {
      AssistantContent::ToolCall(_) => false,
      AssistantContent::Text(text) => !text.text.trim().is_empty(),
      _ => true,
    })
    .collect();
  // No id: the provider never finished the message it would name, and
  // replaying a half of it under that name is asking to be told it is not one.
  (!content.is_empty()).then_some(Message::Assistant { id: None, content })
}

/// A call answered without being run.
fn aborted(call: &ToolCall) -> UserContent {
  UserContent::tool_result(
    call.id.as_str(),
    &call.function.name,
    vec![ToolResultContent::text(ABORTED)],
  )
}

impl Runtime {
  /// What this session offers the model, in a stable order — a tool set that
  /// shuffled between requests would be a different prompt every time.
  async fn definitions(&self) -> Result<Vec<ToolDefinition>, String> {
    let mut definitions = self
      .tools
      .now()
      .get_tool_defs(None)
      .await
      .map_err(|err| err.to_string())?;
    definitions.retain(|definition| self.rules.permits(&definition.name));
    Ok(definitions)
  }

  /// Run one call and answer it, telling the UI as it goes.
  async fn call(
    &self,
    call: &ToolCall,
    written: Option<&String>,
    tx: &mpsc::UnboundedSender<AgentEvent>,
  ) -> UserContent {
    let name = call.function.name.clone();
    let id = call.id.as_str().to_string();
    let _ = tx.send(AgentEvent::ToolCall {
      name: name.clone(),
      args: call.function.arguments.clone(),
      call: id.clone(),
      internal: written.cloned().unwrap_or_else(|| id.clone()),
    });

    let mut context = ToolContext::new();
    // Where to report output while the call runs, already answering for
    // this call. The loop dispatching it is the one that knows which it
    // is, so nothing has to be smuggled through the arguments to tell the
    // tool — and the tool never has to be told at all.
    let (reporting, under) = (tx.clone(), id.clone());
    context.insert(Output(Arc::new(move |text| {
      let _ = reporting.send(AgentEvent::ToolOutput {
        call: under.clone(),
        text,
      });
    })));
    let result = self
      .tools
      .now()
      .execute(&name, &call.function.arguments.to_string(), &mut context)
      .await;

    // A built-in tool holds itself to a size as it makes its output; a
    // server's reply arrives whole, and as long as the server felt like, so
    // it is cut here — the one place every result passes through.
    let raw = result.output().as_content();
    let content = match BUILT_IN.contains(&name.as_str()) {
      true => None,
      false => crate::tools::cap_reply(raw),
    }
    .unwrap_or_else(|| raw.to_vec());
    let (output, images) = crate::images::split(&content);
    let _ = tx.send(AgentEvent::ToolResult {
      name: name.clone(),
      call: id.clone(),
      diff: context.result::<EditDiff>().map(|diff| diff.diff.clone()),
      output,
      images,
      is_error: result.is_error() || result.is_refused(),
    });
    // What the transcript shows and what the model is given are the same
    // bytes: there is one copy of a result, not a shown one and a sent one.
    UserContent::tool_result(id, name, content)
  }
}

/// The OpenAI chat completions API only accepts text in tool messages, so
/// this strips images out of tool results and re-sends them in a user
/// message right after, so `read` can return screenshots.
fn relay_tool_images(history: &mut Vec<Message>) {
  *history = std::mem::take(history)
    .into_iter()
    .flat_map(|message| {
      let (message, relay) = relay(message);
      std::iter::once(message).chain(relay)
    })
    .collect();
}

/// One message with the images taken out of its tool results, and the
/// message that carries them instead — when it had any to take.
fn relay(message: Message) -> (Message, Option<Message>) {
  let Message::User { mut content } = message else {
    return (message, None);
  };
  let mut images = Vec::new();
  for item in &mut content {
    let UserContent::ToolResult(result) = item else {
      continue;
    };
    let (found, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut result.content)
      .into_iter()
      .partition(|c| matches!(c, ToolResultContent::Image(_)));
    result.content = match found.is_empty() || !rest.is_empty() {
      true => rest,
      false => vec![ToolResultContent::text("(see attached image)")],
    };
    images.extend(found.into_iter().filter_map(|c| match c {
      ToolResultContent::Image(image) => Some(UserContent::Image(image)),
      _ => None,
    }));
  }
  let relay = (!images.is_empty()).then(|| {
    let mut relay = vec![UserContent::text("Attached image(s) from tool result:")];
    relay.extend(images);
    Message::User { content: relay }
  });
  (Message::User { content }, relay)
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
fn for_tools(prompt: &str, tools: &crate::tools::Rules) -> String {
  let gone: Vec<&str> = crate::tools::BUILT_IN
    .iter()
    .copied()
    .filter(|name| !tools.permits(name))
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

fn default_system_prompt(cwd: &Path, tools: &crate::tools::Rules) -> String {
  let mut prompt = for_tools(
    "You are an expert coding assistant operating inside a minimal terminal coding agent. \
         You help users by reading files, executing commands, editing code, and writing new files.\n\n\
         <rules>\n\
         - Use bash for file operations like ls, rg, find\n\
         - Use read to examine files instead of cat or sed.\n\
         - Use edit for precise changes (edits[].oldText must match exactly)\n\
         - When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls\n\
         - Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.\n\
         - Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.\n\
         - Use write only for new files or complete rewrites.\n\
         - Use ask whenever the request is underspecified and you cannot proceed without a concrete decision; do not ask what the code itself can answer\n\
         - Be concise in your responses\n\
         - Show file paths clearly when working with files\n\
         </rules>\n",
    tools,
  );

  let context = ["AGENTS.md", "CLAUDE.md"].iter().find_map(|name| {
    let path = cwd.join(name);
    std::fs::read_to_string(&path).ok().map(|content| (path, content))
  });
  if let Some((path, content)) = context {
    prompt.push_str(&format!(
      "\n<project_context>\nProject-specific instructions and guidelines:\n\n\
       <project_instructions path=\"{}\">\n{content}\n</project_instructions>\n\
       </project_context>\n",
      path.display()
    ));
  }

  prompt.push_str(&format!("\n<cwd>\n{}\n</cwd>", cwd.display()));
  prompt
}

/// Summarize older history in the background; the result arrives as
/// `AgentEvent::Compacted` (or `AgentEvent::Error` followed by `Ended`).
pub fn start_compaction(
  runtime: Arc<Runtime>,
  history: Vec<Message>,
  settings: Settings,
  tx: mpsc::UnboundedSender<AgentEvent>,
) -> JoinHandle<()> {
  tokio::spawn(async move {
    let event = match compaction::compact(&runtime, history, &settings).await {
      Ok(result) => AgentEvent::Compacted(result),
      Err(err) => {
        let _ = tx.send(AgentEvent::Error(format!("compaction failed: {err}")));
        AgentEvent::Ended { messages: Vec::new() }
      }
    };
    let _ = tx.send(event);
  })
}

#[cfg(test)]
mod tests {
  //! Runs against a mock OpenAI-compatible server when `FA_TEST_BASE_URL` is set.
  use super::*;
  use rig_core::completion::CompletionRequest;

  async fn collect(
    runtime: &Arc<Runtime>,
    history: Vec<Message>,
    prompt: &str,
    rx: &mut mpsc::UnboundedReceiver<AgentEvent>,
    tx: &mpsc::UnboundedSender<AgentEvent>,
  ) -> Vec<AgentEvent> {
    let handle = start_run(
      runtime.clone(),
      Control::default(),
      history,
      vec![Message::user(prompt)],
      tx.clone(),
    );
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

  // -------------------------------------------------- driving the loop

  /// A model that answers from a script, so the loop can be put in the
  /// states a provider is too slow and too willing to reach: a turn cut off
  /// half-said, a turn whose calls never all come back.
  struct Scripted {
    turns: Mutex<VecDeque<Vec<rig_core::streaming::RawStreamingChoice>>>,
    /// Leave the stream open after the script runs out, so a turn can be
    /// stopped in the middle of itself rather than ending on its own.
    hang: bool,
  }

  impl CompletionModel for Scripted {
    async fn completion(
      &self,
      _request: CompletionRequest,
    ) -> Result<rig_core::completion::CompletionResponse, CompletionError> {
      Err(CompletionError::ResponseError("the script only streams".into()))
    }

    async fn stream(&self, _request: CompletionRequest) -> Result<StreamingCompletionResponse, CompletionError> {
      let turn = self.turns.lock().expect("a script nobody panicked holding").pop_front();
      let items = futures::stream::iter(turn.unwrap_or_default().into_iter().map(Ok));
      let inner: rig_core::streaming::StreamingResult = match self.hang {
        true => Box::pin(items.chain(futures::stream::pending())),
        false => Box::pin(items),
      };
      Ok(StreamingCompletionResponse::stream("scripted", inner))
    }
  }

  fn said(text: &str) -> rig_core::streaming::RawStreamingChoice {
    rig_core::streaming::RawStreamingChoice::Message(text.to_string())
  }

  fn asks(id: &str, command: &str) -> rig_core::streaming::RawStreamingChoice {
    rig_core::streaming::RawStreamingChoice::ToolCall(rig_core::streaming::RawStreamingToolCall::new(
      id,
      "bash".to_string(),
      serde_json::json!({ "command": command }),
    ))
  }

  /// A runtime whose model reads from `turns` and whose only tool is the
  /// real `bash`, which is the one that can be told to take its time.
  fn scripted(turns: Vec<Vec<rig_core::streaming::RawStreamingChoice>>, hang: bool) -> Arc<Runtime> {
    let tools = ToolServer::new()
      .tool(BashTool {
        cwd: std::env::temp_dir(),
      })
      .run();
    let tools = Tools::new(tools);
    Arc::new(Runtime {
      model: ModelHandle::new(Scripted {
        turns: Mutex::new(turns.into()),
        hang,
      }),
      tools,
      preamble: String::new(),
      compaction: TEST_SETTINGS,
      rules: Default::default(),
      relay_images: false,
      max_tokens: None,
    })
  }

  /// Read events until one of them is what the test was waiting for, then
  /// keep reading until the run says it is over. Returns what it ended with.
  async fn stop_at(
    runtime: Arc<Runtime>,
    control: Control,
    prompt: &str,
    at: impl Fn(&AgentEvent) -> bool,
  ) -> Vec<Message> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let handle = start_run(runtime, control.clone(), Vec::new(), vec![Message::user(prompt)], tx);
    let mut cancelled = false;
    loop {
      let event = tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
        .await
        .expect("an event within 10s")
        .expect("the channel open");
      if !cancelled && at(&event) {
        cancelled = true;
        control.cancel();
      }
      match event {
        AgentEvent::Ended { messages } | AgentEvent::Done { messages } => {
          handle.await.expect("the run to finish");
          return messages;
        }
        _ => {}
      }
    }
  }

  fn shapes(messages: &[Message]) -> Vec<String> {
    messages
      .iter()
      .map(|message| match message {
        Message::User { content } => content
          .iter()
          .map(|c| match c {
            UserContent::Text(text) => format!("user {}", text.text),
            UserContent::ToolResult(result) => format!(
              "result {} {}",
              result.call.as_str(),
              match &result.content[..] {
                [ToolResultContent::Text(text)] => text.text.trim().to_string(),
                _ => String::new(),
              }
            ),
            _ => "?".into(),
          })
          .collect::<Vec<_>>()
          .join(" + "),
        Message::Assistant { content, .. } => content
          .iter()
          .map(|c| match c {
            AssistantContent::Text(text) => format!("said {:?}", text.text),
            AssistantContent::Reasoning(reasoning) => format!("thought {:?}", reasoning.display_text()),
            AssistantContent::ToolCall(call) => format!("call {}", call.id.as_str()),
            _ => "?".into(),
          })
          .collect::<Vec<_>>()
          .join(" + "),
        Message::System { .. } => "system".into(),
      })
      .collect()
  }

  /// A turn stopped while the model was still writing it keeps what was
  /// said and asks for nothing.
  ///
  /// The calls it had begun to write were never dispatched, so carrying
  /// them would leave the conversation owing answers nothing is coming to
  /// give — which is a conversation no provider will take back.
  #[tokio::test]
  async fn a_turn_cut_off_mid_stream_keeps_what_it_said_and_asks_for_nothing() {
    let runtime = scripted(vec![vec![said("Let me look. "), asks("call_1", "echo one")]], true);
    let messages = stop_at(runtime, Control::default(), "look", |event| {
      // Stopped once the call has been written but before the stream that
      // was writing it ever ends.
      matches!(event, AgentEvent::ToolCallDelta { .. })
    })
    .await;
    assert_eq!(shapes(&messages), ["user look", "said \"Let me look. \""]);
  }

  /// What the model was thinking when it was stopped stays with what it had
  /// said, in the order it came.
  #[tokio::test]
  async fn a_turn_cut_off_mid_thought_keeps_the_thought() {
    let thinking = rig_core::streaming::RawStreamingChoice::ReasoningDelta {
      id: rig_core::streaming::StreamPartId::wire("r1"),
      provider_id: None,
      reasoning: "Where to look".to_string(),
    };
    let runtime = scripted(vec![vec![thinking, said("Half ")]], true);
    let messages = stop_at(runtime, Control::default(), "look", |event| {
      matches!(event, AgentEvent::Text(_))
    })
    .await;
    assert_eq!(
      shapes(&messages),
      ["user look", "thought \"Where to look\" + said \"Half \""]
    );
  }

  /// A turn stopped before it had said anything but blanks leaves nothing:
  /// an assistant message with nothing in it is one no provider takes back.
  #[tokio::test]
  async fn a_turn_cut_off_before_it_said_anything_leaves_nothing() {
    let runtime = scripted(vec![vec![said("\n\n")]], true);
    let messages = stop_at(runtime, Control::default(), "look", |event| {
      matches!(event, AgentEvent::Text(_))
    })
    .await;
    assert_eq!(shapes(&messages), ["user look"]);
  }

  /// Every call a stopped run had out is answered, whether it was running
  /// or had not been reached.
  #[tokio::test]
  async fn every_call_a_stopped_run_had_out_is_answered() {
    let runtime = scripted(
      vec![vec![
        said("Three things. "),
        asks("call_1", "echo one"),
        asks("call_2", "sleep 60"),
        asks("call_3", "echo three"),
      ]],
      false,
    );
    let messages = stop_at(runtime, Control::default(), "do three things", |event| {
      // Stopped while the second is running, which leaves the third
      // waiting its turn and never started.
      matches!(event, AgentEvent::ToolCall { call, .. } if call == "call_2")
    })
    .await;
    assert_eq!(
      shapes(&messages),
      [
        "user do three things".to_string(),
        "said \"Three things. \" + call call_1 + call call_2 + call call_3".to_string(),
        // The one that finished answers with what it said; the one it was
        // stopped in the middle of and the one it never reached both
        // answer all the same.
        format!("result call_1 one + result call_2 {ABORTED} + result call_3 {ABORTED}"),
      ]
    );
  }

  /// What was typed while a turn ran is read at the top of the next one,
  /// in the order it was typed, and lands in the conversation before the
  /// turn that answers it rather than after the whole run.
  #[tokio::test]
  async fn what_was_typed_mid_run_is_read_before_the_next_turn() {
    let runtime = scripted(
      vec![
        // The call takes long enough that what is typed while it runs is
        // reliably typed before the turn that reads it.
        vec![said("Working. "), asks("call_1", "sleep 0.5; echo one")],
        vec![said("Answered them all.")],
      ],
      false,
    );
    let control = Control::default();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let handle = start_run(runtime, control.clone(), Vec::new(), vec![Message::user("start")], tx);
    let mut steered = false;
    let messages = loop {
      let event = tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
        .await
        .expect("an event within 10s")
        .expect("the channel open");
      // Typed while the first turn's call is running, which is the run at
      // its least interruptible.
      if !steered && matches!(&event, AgentEvent::ToolCall { call, .. } if call == "call_1") {
        steered = true;
        control.steer(Prompt::text("and this".into()));
        control.steer(Prompt::text("and this too".into()));
      }
      if let AgentEvent::Done { messages } | AgentEvent::Ended { messages } = event {
        handle.await.expect("the run to finish");
        break messages;
      }
    };
    assert_eq!(
      shapes(&messages),
      [
        "user start",
        "said \"Working. \" + call call_1",
        "result call_1 one",
        // Both, in the order they were typed, and before the turn that
        // reads them — not appended after the answer.
        "user and this",
        "user and this too",
        "said \"Answered them all.\"",
      ]
    );
    assert!(!control.steering(), "nothing is left waiting once it has been read");
  }

  const TEST_SETTINGS: Settings = Settings {
    enabled: true,
    context_window: 1000,
    reserve_tokens: 100,
    keep_recent_tokens: 1,
    turn_summary: true,
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
      max_tokens: None,
      compaction: TEST_SETTINGS,
      vision: true,
      tools: Default::default(),
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let agent = build_agents(&cfg, Path::new("/tmp"), &Host::new(tx.clone()), &Default::default())
      .unwrap()
      .runtime;

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
    let Some(AgentEvent::Done { messages: history }) = events.last() else {
      panic!("no Done")
    };
    assert_eq!(history.len(), 2, "messages = user + assistant, got {history:?}");
    // What the call cost is reported by the call, not by the run it belongs
    // to: this one is the whole run, but a run of twenty turns says it
    // twenty times.
    let spent: Vec<(u64, u64)> = events
      .iter()
      .filter_map(|e| match e {
        AgentEvent::Usage { usage, context_tokens } => Some((usage.output_tokens, *context_tokens)),
        _ => None,
      })
      .collect();
    assert_eq!(spent, [(7, 19)]);

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
        AgentEvent::Steered { .. } => "steered".into(),
        AgentEvent::Usage { .. } => "usage".into(),
        AgentEvent::Error(e) => format!("error:{e}"),
        AgentEvent::Ended { .. } => "ended".into(),
        AgentEvent::Compacted(_) => "compacted".into(),
        AgentEvent::Models(_) => "models".into(),
        AgentEvent::Show(_) | AgentEvent::Ask(_) => "asking".into(),
        AgentEvent::Mcp(_) => "mcp".into(),
        AgentEvent::Expanded { .. } => "expanded".into(),
        AgentEvent::Suggested { .. } => "suggested".into(),
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
      max_tokens: None,
      compaction: TEST_SETTINGS,
      vision: true,
      tools: Default::default(),
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let agents = build_agents(&cfg, Path::new("/tmp"), &Host::new(tx.clone()), &Default::default()).unwrap();
    let history = vec![
      Message::user("first question"),
      Message::assistant("first answer"),
      Message::user("second question"),
      Message::assistant("second answer"),
    ];
    start_compaction(agents.runtime.clone(), history, TEST_SETTINGS, tx.clone())
      .await
      .unwrap();
    let compacted = match rx.recv().await {
      Some(AgentEvent::Compacted(Some(compacted))) => compacted,
      other => panic!("expected Compacted event, got {other:?}"),
    };
    assert!(compacted.summary.contains("Mock summary"), "{}", compacted.summary);
    assert_eq!(compacted.summarized, 2);
    assert_eq!(
      compacted.kept,
      [Message::user("second question"), Message::assistant("second answer")]
    );

    // With a single turn left after the summary and a budget it fits in,
    // there is nothing to compact.
    let roomy = Settings {
      keep_recent_tokens: 1000,
      ..TEST_SETTINGS
    };
    let history = std::iter::once(compaction::summary_message(&compacted.summary))
      .chain(compacted.kept)
      .collect();
    start_compaction(agents.runtime.clone(), history, roomy, tx.clone())
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
    start_compaction(agents.runtime.clone(), history, TEST_SETTINGS, tx.clone())
      .await
      .unwrap();
    let compacted = match rx.recv().await {
      Some(AgentEvent::Compacted(Some(compacted))) => compacted,
      other => panic!("expected Compacted event, got {other:?}"),
    };
    assert_eq!(compacted.summarized, 2);
    assert!(compacted.kept.is_empty(), "one fresh summary, not nested");
  }

  /// The provider counts what it was sent; only what has been said since is
  /// guessed at. Nothing a request was charged for is estimated a second
  /// time.
  #[test]
  fn the_context_is_the_providers_own_count_plus_what_it_has_not_seen_yet() {
    let result = |text: &str| Message::User {
      content: vec![UserContent::tool_result(
        "call",
        "bash",
        vec![ToolResultContent::text(text)],
      )],
    };
    let asked = Message::user("x".repeat(400));
    let answered = Message::assistant("y".repeat(400));
    let mut weigh = Weigh::default();

    // Nothing counted yet: the whole request is the estimate, 400 chars of
    // it to the hundred tokens.
    assert_eq!(weigh.request(std::slice::from_ref(&asked)), 100);
    // The provider's figure covers the message it was sent and the answer
    // it gave, so the next request estimates neither: only the result that
    // came back after it.
    weigh.answered(
      &Usage {
        input_tokens: 500,
        output_tokens: 20,
        ..Usage::new()
      },
      1,
    );
    let chat = vec![asked.clone(), answered.clone(), result(&"z".repeat(800))];
    assert_eq!(weigh.request(&chat), 520 + 200);
    // Two results in the same turn: both are estimated, and still nothing
    // before them.
    let chat = vec![
      asked.clone(),
      answered.clone(),
      result(&"z".repeat(800)),
      result(&"z".repeat(400)),
    ];
    assert_eq!(weigh.request(&chat), 520 + 200 + 100);

    // A provider that says nothing leaves the last figure standing rather
    // than reporting the context as empty.
    weigh.answered(&Usage::new(), 9);
    assert_eq!(weigh.request(&chat), 520 + 200 + 100);
  }

  /// A cached prompt is still a prompt: what the provider read back from its
  /// cache was in the request, and a context that leaves it out is the size
  /// of a long conversation short by nearly all of it.
  #[test]
  fn the_context_counts_what_was_cached_as_well_as_what_was_charged() {
    let cached = Usage {
      input_tokens: 12,
      output_tokens: 40,
      cached_input_tokens: 30_000,
      cache_creation_input_tokens: 2_000,
      total_tokens: 32_052,
      ..Usage::new()
    };
    assert_eq!(weight(&cached), 32_052);

    // A provider that reports no total is taken at the sum of its parts.
    let plain = Usage {
      input_tokens: 500,
      output_tokens: 20,
      ..Usage::new()
    };
    assert_eq!(weight(&plain), 520);

    // And one whose total says less than the prompt and the answer it also
    // reported is believed about neither: the larger of the two is the one
    // that leaves nothing out.
    let short = Usage {
      total_tokens: 100,
      ..plain
    };
    assert_eq!(weight(&short), 520);

    assert_eq!(weight(&Usage::new()), 0);
  }

  #[test]
  fn the_prompt_stops_speaking_for_a_tool_the_session_does_not_have() {
    let rules = |allow: &[&str], deny: &[&str]| crate::tools::Rules {
      allow: allow.iter().map(|s| s.to_string()).collect(),
      deny: deny.iter().map(|s| s.to_string()).collect(),
    };
    let all = default_system_prompt(Path::new("/work"), &rules(&[], &[]));
    for rule in ["Use read", "Use bash", "Use edit", "Use write", "Use ask"] {
      assert!(all.contains(rule), "{rule:?} is there by default");
    }

    // A model told about `bash` and then refused it tries anyway and reports
    // being refused, instead of using what it does have.
    let reading = default_system_prompt(Path::new("/work"), &rules(&["read"], &[]));
    for gone in [
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

    // A list that leaves the built-in five alone changes nothing, however
    // many other tools it names.
    let mut five = crate::tools::BUILT_IN.to_vec();
    five.push("fetch");
    assert_eq!(default_system_prompt(Path::new("/work"), &rules(&five, &[])), all);
    assert_eq!(default_system_prompt(Path::new("/work"), &rules(&[], &["fetch"])), all);
    // And refusing one is the same as allowing the rest.
    let no_bash = default_system_prompt(Path::new("/work"), &rules(&[], &["bash"]));
    assert!(!no_bash.contains("Use bash"), "{no_bash}");
    assert!(no_bash.contains("Use read"), "{no_bash}");
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

  /// A cap travels with the run's requests and the summarizer's alike —
  /// Anthropic refuses a request that names none, so it has to reach both.
  #[tokio::test]
  async fn a_token_cap_is_carried_on_every_request() {
    #[derive(Default)]
    struct Recording {
      caps: Mutex<Vec<Option<u64>>>,
    }

    impl CompletionModel for Recording {
      async fn completion(
        &self,
        request: CompletionRequest,
      ) -> Result<rig_core::completion::CompletionResponse, CompletionError> {
        self.caps.lock().unwrap().push(request.max_tokens);
        Err(CompletionError::ResponseError("recorded, nothing to say".into()))
      }

      async fn stream(&self, request: CompletionRequest) -> Result<StreamingCompletionResponse, CompletionError> {
        self.caps.lock().unwrap().push(request.max_tokens);
        let items = futures::stream::iter([Ok(said("done"))]);
        Ok(StreamingCompletionResponse::stream("recording", Box::pin(items)))
      }
    }

    let recording = Arc::new(Recording::default());
    let runtime = Arc::new(Runtime {
      model: ModelHandle::new(recording.clone()),
      tools: Tools::new(ToolServer::new().run()),
      preamble: String::new(),
      compaction: TEST_SETTINGS,
      rules: Default::default(),
      relay_images: false,
      max_tokens: Some(99),
    });
    let (tx, mut rx) = mpsc::unbounded_channel();
    collect(&runtime, Vec::new(), "hello", &mut rx, &tx).await;

    let _ = runtime.ask("summarize".to_string()).await;

    assert_eq!(*recording.caps.lock().unwrap(), vec![Some(99), Some(99)]);
  }

  /// Every provider there is, and whether an image it is sent inside a tool
  /// result has to be relayed after it instead.
  const PROVIDERS: [(Provider, bool); 17] = [
    (Provider::OpenAi, true),
    (Provider::OpenRouter, true),
    (Provider::Ollama, true),
    (Provider::Gemini, false),
    (Provider::Anthropic, false),
    (Provider::Cohere, true),
    (Provider::DeepSeek, true),
    (Provider::Doubleword, true),
    (Provider::Groq, true),
    (Provider::Hyperbolic, true),
    (Provider::Llamafile, true),
    (Provider::Mira, true),
    (Provider::Mistral, true),
    (Provider::Perplexity, true),
    (Provider::Together, true),
    (Provider::Venice, true),
    (Provider::XAi, true),
  ];

  /// Every provider builds, with a key and without one, and only the two
  /// that take an image inside a tool result skip the relay.
  #[test]
  fn each_provider_builds_a_model() {
    for (provider, relay) in PROVIDERS {
      for (key, base_url) in [("", None), ("k", Some("http://127.0.0.1:1/v1/".to_string()))] {
        let cfg = Config {
          provider,
          base_url,
          api_key: key.into(),
          model: "mock".into(),
          system_prompt: None,
          max_tokens: None,
          compaction: TEST_SETTINGS,
          vision: true,
          tools: Default::default(),
        };
        let (tx, _rx) = mpsc::unbounded_channel();
        let agents = build_agents(&cfg, Path::new("/tmp"), &Host::new(tx), &Default::default())
          .unwrap_or_else(|e| panic!("{provider:?} with key {key:?}: {e}"));
        assert_eq!(agents.runtime.relay_images, relay, "{provider:?}");
      }
    }
  }

  /// Every provider answers being asked for its models, with a list or with
  /// a refusal the UI can put on screen — and none of them panics, which a
  /// client built for an endpoint that is not there could.
  #[tokio::test]
  async fn every_provider_answers_being_asked_for_its_models() {
    for (provider, _) in PROVIDERS {
      let cfg = Config {
        provider,
        // Nothing is listening, so a provider with an arm fails to reach it
        // and one without never gets that far.
        base_url: Some("http://127.0.0.1:1/v1".to_string()),
        api_key: "k".into(),
        model: "mock".into(),
        system_prompt: None,
        max_tokens: None,
        compaction: TEST_SETTINGS,
        vision: true,
        tools: Default::default(),
      };
      let err = list_models(&cfg).await.expect_err("nothing is listening");
      assert!(!format!("{err:#}").is_empty(), "{provider:?} said nothing");
    }
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
      max_tokens: None,
      compaction: TEST_SETTINGS,
      vision: true,
      tools: Default::default(),
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let agent = build_agents(&cfg, Path::new("/tmp"), &Host::new(tx.clone()), &Default::default())
      .unwrap()
      .runtime;
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
      max_tokens: None,
      compaction: TEST_SETTINGS,
      vision: true,
      tools: Default::default(),
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let agent = build_agents(&cfg, Path::new("/tmp"), &Host::new(tx.clone()), &Default::default())
      .unwrap()
      .runtime;
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
