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
use rig_agent::tool::server::{ToolServer, ToolServerHandle};
use rig_agent::tool::{ToolContext, ToolExecutionError, ToolResult};
use rig_core::DynModel;
use rig_core::completion::{CompletionRequest, Message, ToolDefinition, Usage};
use rig_core::message::{
  AssistantContent, Issuer, Reasoning, ReasoningContent, ToolCall, ToolResultContent, UserContent,
};
use rig_core::operation::Completion;
use rig_core::providers::{anthropic, cohere, gemini, ollama, openai, xai};
use rig_core::streaming::{CompletionStream, Item, Part, PartKind, StreamEvent};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use crate::attach::Prompt;
use crate::compaction::{self, Compacted, Settings};
use crate::drafts::{Fragment, drafting};
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
  /// The run is making room for itself: the conversation outgrew the window,
  /// so it is being compacted before the run goes on — `resuming` — or,
  /// when it filled up with the answer, before the next prompt is sent into
  /// it. `messages` is everything the run added up to here, which is what
  /// the summary is made from and what `Done` or `Ended` will not carry
  /// again.
  Compacting {
    messages: Vec<Message>,
    resuming: bool,
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
    self.runtime = Arc::new(Runtime::new(cfg, old.tools.clone(), old.preamble.clone())?);
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
/// Cancelling, pausing and steering are the things it has to say, and all
/// have to reach a run that is in the middle of something — so each is
/// something the loop reads at its next opportunity, and cancelling can also
/// be waited on by the parts that cannot wait for whatever the loop is
/// blocked on.
#[derive(Clone, Default)]
pub struct Control(Arc<Signals>);

#[derive(Default)]
struct Signals {
  /// Esc, as a value the run can wait to see change rather than only read.
  cancelled: watch::Sender<bool>,
  /// `/pause`: finish what is in hand, then stop before asking the model
  /// again.
  paused: AtomicBool,
  /// Raised by the run when the conversation outgrew the window, and taken
  /// back down by whatever makes the room.
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
  pub fn begin(&self) {
    self.0.cancelled.send_replace(false);
    self.0.paused.store(false, Ordering::Relaxed);
  }

  /// Stop the run at the first thing it can stop in the middle of.
  pub fn cancel(&self) {
    self.0.cancelled.send_replace(true);
  }

  pub fn cancelled(&self) -> bool {
    *self.0.cancelled.borrow()
  }

  /// Stop the run once the turn it is on is over: the answer it is writing
  /// is finished and the calls it asked for are run, but nothing more is
  /// asked of the model.
  pub fn pause(&self) {
    self.0.paused.store(true, Ordering::Relaxed);
  }

  fn paused(&self) -> bool {
    self.0.paused.load(Ordering::Relaxed)
  }

  /// Resolves once the run should stop what it is doing — at once, when it
  /// already should.
  pub async fn stopped(&self) {
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

  /// Take everything, in the order it was typed, for Esc to take out of
  /// the way.
  pub fn take(&self) -> Vec<Prompt> {
    self.held().drain(..).collect()
  }

  /// Take the messages at the front, in the order they were typed, for the
  /// run to read into the turn it is about to ask for. A command among them
  /// is not the run's to read, so it stops there: the command, and anything
  /// typed after it, wait for the run to end and go in the order they came.
  fn take_said(&self) -> Vec<Prompt> {
    let mut held = self.held();
    let said = held.iter().take_while(|prompt| !prompt.command).count();
    held.drain(..said).collect()
  }

  /// Whether something is waiting that the run has to read.
  pub fn steering(&self) -> bool {
    self.held().front().is_some_and(|prompt| !prompt.command)
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
  /// since whoever asks is the one answering it.
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
  model: DynModel<Completion>,
  tools: Tools,
  preamble: String,
  compaction: Settings,
  /// Which tools this session was held to.
  rules: crate::tools::Rules,
  /// Chat completions will not carry an image inside a tool message, so a
  /// `read` that answers with a screenshot has it relayed after the result.
  relay_images: bool,
  /// Who the readable part of reasoning another service produced is passed
  /// off as, so it is sent here too; `None` where only what this service
  /// signed itself may go back.
  adopt_reasoning: Option<Adopter>,
  /// Cap on one answer, left to the provider when `None`.
  max_tokens: Option<u64>,
}

impl Runtime {
  fn new(cfg: &Config, tools: Tools, preamble: String) -> Result<Self> {
    let (model, relay_images, adopt_reasoning) = build_model(cfg)?;
    Ok(Self {
      model,
      tools,
      preamble,
      compaction: cfg.compaction,
      rules: cfg.tools.clone(),
      relay_images,
      adopt_reasoning,
      max_tokens: cfg.max_tokens,
    })
  }

  /// One question to the model with no tools and the summarizer's preamble,
  /// and its answer: how compaction summarizes. Nothing here is a
  /// conversation.
  pub async fn ask(&self, prompt: String) -> Result<String> {
    let request = CompletionRequest::new(Message::user(prompt))
      .preamble(compaction::SYSTEM_PROMPT)
      .max_tokens(self.max_tokens);
    Ok(self.model.call(request).await?.text())
  }
}

/// Which API to speak. The names are the `--provider` values, and what the
/// provider is called on screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Provider {
  /// OpenAI Chat Completions and compatible servers
  #[value(name = "openai")]
  OpenAi,
  /// OpenAI Responses API
  #[value(name = "openai-responses")]
  OpenAiResponses,
  /// Azure OpenAI, at the resource endpoint `--base-url` names
  Azure,
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
  /// Hugging Face Inference
  #[value(name = "huggingface")]
  HuggingFace,
  /// Hyperbolic
  Hyperbolic,
  /// A llama.cpp server — llama-server, or a llamafile — on
  /// http://localhost:8080 by default
  #[value(name = "llamacpp")]
  LlamaCpp,
  /// MiniMax
  #[value(name = "minimax")]
  MiniMax,
  /// Mira
  Mira,
  /// Mistral
  Mistral,
  /// Moonshot AI (Kimi)
  Moonshot,
  /// Perplexity
  Perplexity,
  /// Together AI
  Together,
  /// Venice
  Venice,
  /// xAI
  #[value(name = "xai")]
  XAi,
  /// Xiaomi MiMo
  #[value(name = "xiaomimimo")]
  XiaomiMimo,
  /// Z.AI
  #[value(name = "zai")]
  ZAi,
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
    match (self, dialect(self)) {
      // The variable rig reads for it, which is the one its own tools use.
      (_, Some(dialect)) => dialect.api_key_env.into(),
      (Self::Anthropic, None) => anthropic::ANTHROPIC.api_key_env.into(),
      (Self::Gemini, None) => gemini::API_KEY_ENV.into(),
      _ => format!("{}_API_KEY", self.label().to_uppercase()),
    }
  }

  /// The key to use when none is given. Ollama and llama.cpp want no key at
  /// all — rig leaves the header off for an empty one, which is what a local
  /// server expects — while a hosted endpoint that ignores auth is happy with
  /// anything.
  pub fn no_key(self) -> &'static str {
    match self {
      Self::Ollama | Self::LlamaCpp => "",
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

/// A provider's client, with the configured key and the base URL when there
/// is one. Every OpenAI-shaped provider is the one client on a dialect of its
/// own; the four that speak something else have clients of their own.
enum Client {
  OpenAi(Box<openai::OpenAI>),
  Anthropic(anthropic::Anthropic),
  Gemini(gemini::Gemini),
  Ollama(ollama::Ollama),
  Cohere(cohere::Cohere),
}

/// The OpenAI-shaped dialect `provider` speaks, for the ones that speak one.
fn dialect(provider: Provider) -> Option<&'static openai::wire::Dialect> {
  use openai::wire::*;
  Some(match provider {
    Provider::OpenAi | Provider::OpenAiResponses => &OPENAI,
    Provider::Azure => &AZURE,
    Provider::OpenRouter => &OPENROUTER,
    Provider::DeepSeek => &DEEPSEEK,
    Provider::Doubleword => &DOUBLEWORD,
    Provider::Groq => &GROQ,
    Provider::HuggingFace => &HUGGINGFACE,
    Provider::Hyperbolic => &HYPERBOLIC,
    Provider::LlamaCpp => &LLAMACPP,
    Provider::MiniMax => &MINIMAX,
    Provider::Mira => &MIRA,
    Provider::Mistral => &MISTRAL,
    Provider::Moonshot => &MOONSHOT,
    Provider::Perplexity => &PERPLEXITY,
    Provider::Together => &TOGETHER,
    Provider::Venice => &VENICE,
    Provider::XAi => &xai::DIALECT,
    Provider::XiaomiMimo => &XIAOMIMIMO,
    Provider::ZAi => &ZAI,
    Provider::Ollama | Provider::Gemini | Provider::Anthropic | Provider::Cohere => return None,
  })
}

fn client(cfg: &Config) -> Result<Client> {
  let key = cfg.api_key.as_str();
  let url = cfg.base_url.as_deref().map(|url| url.trim_end_matches('/'));
  macro_rules! at {
    ($config:expr) => {
      match url {
        Some(url) => $config.with_base_url(url),
        None => $config,
      }
      .client()
    };
  }

  if let Some(dialect) = dialect(cfg.provider) {
    // Azure has no endpoint of its own to fall back to: every resource is
    // reached at its own host. The key goes in the `api-key` header Azure
    // keys are checked under, which is how its dialect sends one.
    if cfg.provider == Provider::Azure && url.is_none() {
      bail!(
        "{} needs --base-url: the resource endpoint, like https://NAME.openai.azure.com",
        cfg.provider.label()
      );
    }
    return Ok(Client::OpenAi(Box::new(at!(openai::OpenAIConfig::with_key(
      dialect, key
    )))));
  }
  Ok(match cfg.provider {
    Provider::Anthropic => Client::Anthropic(at!(anthropic::AnthropicConfig::new(key))),
    Provider::Gemini => Client::Gemini(at!(gemini::GeminiConfig::new(key))),
    Provider::Ollama => Client::Ollama(at!(ollama::OllamaConfig::new().with_api_key(key))),
    Provider::Cohere => Client::Cohere(at!(cohere::CohereConfig::new(key))),
    other => unreachable!("{other:?} speaks an OpenAI dialect"),
  })
}

/// The model `cfg.model` is reached through, whichever provider offers it,
/// whether an image a tool answers with has to be relayed after the result
/// rather than sent inside it, and who reasoning other services produced is
/// passed off as ([`adopt_reasoning`]).
///
/// Summarizing asks the same model the same way, so it is the same model —
/// what makes that call a summary is the preamble it carries, which belongs
/// to the request rather than to the model.
///
/// Gemini takes an image inside a function response, Anthropic inside a tool
/// result, and the Responses API inside a function call's output. Chat
/// Completions refuses one in a tool message unless the server is one rig
/// knows takes it, so which of the two endpoints the model is spoken to
/// through is what decides — and for some providers that is the model's to
/// say, so it is read off the endpoint rig chose.
fn build_model(cfg: &Config) -> Result<(DynModel<Completion>, bool, Option<Adopter>)> {
  let model = cfg.model.clone();
  let chat = |chat: &openai::wire::Chat| (!chat.provider.dialect.quirks.supports_image_tool_results, adopter(chat));
  Ok(match client(cfg)? {
    // The official endpoint answers on either API, and which one is what the
    // two providers are for.
    Client::OpenAi(client) => match cfg.provider {
      Provider::OpenAi => {
        let model = client.chat(model);
        let (relay, adopt) = chat(&model.wire);
        (drafting(model), relay, adopt)
      }
      Provider::OpenAiResponses => (drafting(client.responses(model)), false, None),
      _ => {
        let model = client.completion(model);
        let (relay, adopt) = match &model.wire {
          openai::wire::OpenAiWire::Chat(wire) => chat(wire),
          openai::wire::OpenAiWire::Responses(_) => (false, None),
        };
        (drafting(model), relay, adopt)
      }
    },
    // Anthropic refuses a request that names no `max_tokens`, so rig fills in
    // what the named model allows. One it does not know is given a small cap
    // rather than refused — which is what `--max-tokens` is for, since a
    // figure on the request wins over the model's own.
    Client::Anthropic(client) => {
      let mut model = client.completion(model);
      model.wire.default_max_tokens.get_or_insert(2048);
      (drafting(model), false, None)
    }
    Client::Gemini(client) => (drafting(client.completion(model)), false, None),
    Client::Ollama(client) => (drafting(client.completion(model)), true, None),
    Client::Cohere(client) => (drafting(client.completion(model)), true, None),
  })
}

/// What a Chat Completions endpoint makes of reasoning from elsewhere: who
/// it is passed off as, and whose it sends back as it is already.
pub struct Adopter {
  /// The endpoint's own name, which is always among those it sends reasoning
  /// back for.
  issuer: Issuer,
  /// Whose reasoning goes back untouched, signed and sealed: the endpoint's
  /// own and, behind a gateway, the family of the model asked for.
  native: Vec<Issuer>,
}

/// What reasoning from elsewhere becomes on its way out over `wire`.
///
/// Chat Completions carries a thought as plain text, which any server on it
/// reads whoever thought it, and a server that takes none has it dropped by
/// rig. Claude is the exception, behind a gateway as anywhere: it refuses a
/// thought it did not sign, so it is given only its own. The Responses API
/// and the native wires replay reasoning as items of their own service, and
/// get nothing from elsewhere either.
fn adopter(wire: &openai::wire::Chat) -> Option<Adopter> {
  let dialect = &wire.provider.dialect;
  let issuer = Issuer::from_static(dialect.name);
  let mut native = vec![issuer.clone()];
  if dialect.quirks.upstream_reasoning_issuer {
    let upstream = openai::wire::upstream_reasoning_issuer(dialect.name, &wire.model);
    if upstream == "anthropic" {
      return None;
    }
    native.push(Issuer::from(upstream));
  }
  Some(Adopter { issuer, native })
}

/// Ask the provider what models it offers, alphabetically.
///
/// Only some of them report a context window, and the rest answer with names
/// alone, which is why the window a model is given falls back to the
/// configured one.
pub async fn list_models(cfg: &Config) -> Result<Vec<ModelInfo>> {
  let models = match client(cfg)? {
    Client::OpenAi(client) => client.list_models().await,
    Client::Anthropic(client) => client.list_models().await,
    Client::Gemini(client) => client.list_models().await,
    Client::Ollama(client) => client.list_models().await,
    // Refused without a request going out: there is no listing to ask for.
    Client::Cohere(_) => bail!(
      "{} does not list its models — name one with /model <id>",
      cfg.provider.label()
    ),
  }
  .with_context(|| format!("{} would not say what models it has", cfg.provider.label()))?;

  let mut models: Vec<ModelInfo> = models
    .data
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
    None => default_system_prompt(cwd),
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
    runtime: Arc::new(Runtime::new(cfg, tools, preamble)?),
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
/// Rig counts every token of the prompt as input — what a provider read back
/// from its cache and what it wrote there included — so a conversation that
/// is mostly cached, which is what a long one becomes, still weighs what was
/// actually sent. A counter the provider did not report adds nothing.
fn weight(usage: &Usage) -> u64 {
  usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0)
}

/// Why a run stopped short of an answer.
enum Stop {
  /// Esc. What it got through is kept, and the calls it was in the middle
  /// of are answered.
  Cancelled,
  /// `/pause`, at a turn boundary: everything it got through is whole, and
  /// `/continue` picks it back up.
  Paused,
  /// The conversation outgrew the window. The run makes room and picks
  /// itself back up where it left off.
  Overflow,
  /// Something the user should be told: the turn budget, or a provider or
  /// tool that would not.
  Failed(String),
}

/// Start one run in the background.
///
/// `made` is the message the run opens with, or nothing for `/continue`,
/// which picks the loop back up over the history as it stands — an
/// unanswered message is answered, a half-written turn is carried on.
/// Everything the run adds, that prompt included, comes back in the `Done`
/// or `Ended` that ends it — or in the `Compacting` before it, for what was
/// added before the run made room — and nothing that was already in the
/// history does: what is handed back is only ever new.
pub fn start_run(
  runtime: Arc<Runtime>,
  control: Control,
  history: Vec<Message>,
  made: Vec<Message>,
  tx: mpsc::UnboundedSender<AgentEvent>,
) -> JoinHandle<()> {
  tokio::spawn(async move {
    control.begin();
    let (made, stop) = drive(&runtime, &control, history, made, &tx).await;
    let event = match stop {
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

/// The loop, and the room it needs: whenever it finds the conversation has
/// outgrown the window, the conversation is compacted and the loop carries on
/// over the summary.
///
/// Answers with what was added since the last compaction — what came before
/// went out with `Compacting` — and why it stopped, if it stopped short.
async fn drive(
  rt: &Runtime,
  control: &Control,
  mut history: Vec<Message>,
  mut made: Vec<Message>,
  tx: &mpsc::UnboundedSender<AgentEvent>,
) -> (Vec<Message>, Option<Stop>) {
  loop {
    let stop = run(rt, control, &history, &mut made, tx).await;
    // Esc is the end of it: the next run weighs the window afresh.
    if matches!(stop, Some(Stop::Cancelled)) || !control.overflowed() {
      return (made, stop);
    }
    // Stopped for room, or refused for a request too big to take: it goes
    // on once there is some. An answer or a pause is whole, and the room is
    // only made before the next prompt is sent into it.
    let resuming = matches!(stop, Some(Stop::Overflow | Stop::Failed(_)));
    if let Some(Stop::Failed(err)) = &stop {
      let _ = tx.send(AgentEvent::Error(err.clone()));
    }
    history.extend(made.iter().cloned());
    let _ = tx.send(AgentEvent::Compacting {
      messages: std::mem::take(&mut made),
      resuming,
    });
    let compacted = tokio::select! {
      biased;
      () = control.stopped() => return (Vec::new(), Some(Stop::Cancelled)),
      compacted = compaction::compact(rt, std::mem::take(&mut history), &rt.compaction) => compacted,
    };
    let compacted = match compacted {
      Ok(compacted) => compacted,
      Err(err) => return (Vec::new(), Some(Stop::Failed(format!("compaction failed: {err}")))),
    };
    // Nothing left to summarize but the turn the context is full of.
    // Carrying on regardless would only fill it again and ask for the same
    // summary, so this is where it stops — what it stopped for is already
    // said.
    let Some(compacted) = compacted else {
      let _ = tx.send(AgentEvent::Compacted(None));
      return (Vec::new(), if resuming { Some(Stop::Overflow) } else { stop });
    };
    history = std::iter::once(compaction::summary_message(&compacted.summary))
      .chain(compacted.kept.iter().cloned())
      .collect();
    let _ = tx.send(AgentEvent::Compacted(Some(compacted)));
    if !resuming {
      return (Vec::new(), stop);
    }
    // Paused while the room was made: the run waits for `/continue`, and so
    // does whatever was typed in the meantime.
    if control.paused() {
      return (Vec::new(), Some(Stop::Paused));
    }
  }
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
    for prompt in control.take_said() {
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
    // After the steering is read, so what was typed is in the conversation
    // `/continue` resumes rather than left behind in the queue.
    if control.paused() {
      return Some(Stop::Paused);
    }

    let mut chat = Vec::with_capacity(history.len() + made.len());
    chat.extend_from_slice(history);
    chat.extend(made.iter().cloned());
    if rt.relay_images {
      relay_tool_images(&mut chat);
    }
    if let Some(adopter) = &rt.adopt_reasoning {
      adopt_reasoning(&mut chat, adopter);
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
    let sent = chat.len();
    let request = CompletionRequest::from(chat)
      .preamble(rt.preamble.clone())
      .tools(rt.definitions())
      .max_tokens(rt.max_tokens);
    let mut stream = match rt.model.stream(request) {
      Ok(stream) => stream,
      Err(err) => return Some(Stop::Failed(err.to_string())),
    };

    let mut partial = Partial::new();
    loop {
      let stopped = tokio::select! {
        biased;
        () = control.stopped() => None,
        item = stream.next() => Some(item),
      };
      // Nothing of this turn has run, so the half of it that was written is
      // kept for the transcript's sake and no more.
      let Some(item) = stopped else {
        made.extend(partial.cut_off(&stream));
        return Some(Stop::Cancelled);
      };
      let Some(item) = item else { break };
      let event = match item {
        Ok(Item::Event(event)) => partial.saw(event),
        // A piece of a call being written, or something the provider sent
        // that rig has no word for.
        Ok(Item::Unknown(payload)) => Fragment::from_payload(&payload).map(|piece| partial.drafted(piece)),
        Err(err) => {
          made.extend(partial.cut_off(&stream));
          return Some(Stop::Failed(err.to_string()));
        }
      };
      if let Some(event) = event {
        let _ = tx.send(event);
      }
    }

    // Which line each call was written on, for the dispatch to take away.
    let written = partial.written;
    // The turn as the provider gave it, aggregated by rig: the text, the
    // reasoning, and the calls, in the order they were said. This is the
    // copy that goes into the conversation and the copy the next request
    // replays — there is only ever the one.
    let response = match stream.finish().await {
      Ok(response) => response,
      Err(err) => return Some(Stop::Failed(err.to_string())),
    };
    weigh.answered(&response.usage, sent);
    let _ = tx.send(AgentEvent::Usage {
      usage: response.usage,
      context_tokens: weigh.reported.max(context),
    });
    // Nothing at all, which is as far as a turn cut short by its budget or a
    // filter can get: rig says which, and what to do about it.
    let Some(message) = response.message() else {
      return Some(Stop::Failed(match response.finish_reason() {
        Some(reason) => reason.no_answer_message(),
        None => "the model answered with nothing".into(),
      }));
    };
    let calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
    made.push(message);

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
        answer = rt.call(call, written.get(&call.id.to_string()), tx) => answer,
      });
    }
    made.push(Message::User { content: results });
    if control.cancelled() {
      return Some(Stop::Cancelled);
    }
  }
}

/// What the UI is told about a turn while it streams, what it has to be told
/// again once the turn's calls are run, and what the turn had said if it is
/// stopped before it ends.
struct Partial {
  /// Told apart from every other turn's, since the parts of a turn are
  /// numbered from nought each time and a line still on screen from the last
  /// one would otherwise be taken for this one's.
  turn: uuid::Uuid,
  /// What each part has said, in the order the parts opened.
  parts: Vec<Said>,
  /// The calls being written, by the provider's number for each: rig says
  /// nothing of a call until it is whole, so these come from what the
  /// provider sent, read on the way in.
  drafts: HashMap<u64, Draft>,
  /// Which of those a provider's id names, once it has named one.
  named: HashMap<String, u64>,
  /// The line a call was written on, by the id the finished call was given.
  ///
  /// The line is keyed from the call's first piece, before anything says
  /// what the finished call will be called. Without the pairing, dispatching
  /// the call would leave that line on screen with nothing to take it away.
  written: HashMap<String, String>,
}

/// A call still being written, as far as it has got.
struct Draft {
  line: String,
  name: String,
  args: String,
}

/// One part of a turn, as far as it has got.
enum Said {
  Text(String),
  Thought(String),
  /// A part that ended, as rig finished it.
  Done(Box<AssistantContent>),
  /// A part that says nothing until it ends: a call, or an image.
  Pending,
}

impl Partial {
  fn new() -> Self {
    Self {
      turn: uuid::Uuid::new_v4(),
      parts: Vec::new(),
      drafts: HashMap::new(),
      named: HashMap::new(),
      written: HashMap::new(),
    }
  }

  /// The id the UI keys the line a call is written on under.
  fn line(&self, part: Part) -> String {
    format!("{}:{}", self.turn, part.index())
  }

  /// A piece of a call being written. Arguments arrive a few characters at
  /// a time and each piece carries only what is new, so the UI is sent the
  /// whole of what has arrived, with nothing to reassemble.
  fn drafted(&mut self, piece: Fragment) -> AgentEvent {
    let line = format!("{}:draft:{}", self.turn, piece.index);
    let draft = self.drafts.entry(piece.index).or_insert(Draft {
      line,
      name: String::new(),
      args: String::new(),
    });
    if let Some(name) = piece.name {
      draft.name = name;
    }
    draft.args.push_str(&piece.args);
    if let Some(id) = piece.id {
      self.named.insert(id, piece.index);
    }
    AgentEvent::ToolCallDelta {
      id: draft.line.clone(),
      name: draft.name.clone(),
      args: draft.args.clone(),
    }
  }

  /// The line the finished `call` was being written on: the draft its id
  /// names, or — for a provider that gave it none, and has it named by rig —
  /// the first one of its name still unclaimed.
  fn claim(&mut self, call: &ToolCall) -> Option<String> {
    let index = self
      .named
      .get(&call.id.to_string())
      .copied()
      .filter(|index| self.drafts.contains_key(index))
      .or_else(|| {
        self
          .drafts
          .iter()
          .filter(|(_, draft)| draft.name == call.function.name.as_str())
          .map(|(index, _)| *index)
          .min()
      })?;
    self.drafts.remove(&index).map(|draft| draft.line)
  }

  /// What the UI should be told about what the stream said.
  fn saw(&mut self, event: StreamEvent) -> Option<AgentEvent> {
    match event {
      StreamEvent::Start { kind, .. } => {
        self.parts.push(match kind {
          PartKind::Text => Said::Text(String::new()),
          PartKind::Reasoning => Said::Thought(String::new()),
          PartKind::ToolCall | PartKind::Image => Said::Pending,
        });
        None
      }
      StreamEvent::Text { part, text } => {
        if let Some(Said::Text(said)) = self.parts.get_mut(part.index()) {
          said.push_str(&text);
        }
        Some(AgentEvent::Text(text))
      }
      StreamEvent::Reasoning { part, text } => {
        if let Some(Said::Thought(said)) = self.parts.get_mut(part.index()) {
          said.push_str(&text);
        }
        Some(AgentEvent::Reasoning(text))
      }
      // Rig's own copy of a call's arguments arrives whole, just before the
      // call ends: what the provider sent was drawn as it came.
      StreamEvent::Arguments { .. } => None,
      StreamEvent::End { part, content } => {
        let said = self
          .parts
          .get_mut(part.index())
          .map(|said| std::mem::replace(said, Said::Done(Box::new(content.clone()))));
        match content {
          // Reasoning that arrived whole, with nothing said along the way.
          AssistantContent::Reasoning(reasoning) if matches!(&said, Some(Said::Thought(so_far)) if so_far.is_empty()) =>
          {
            let text = compaction::reasoning_text(&reasoning);
            (!text.is_empty()).then_some(AgentEvent::Reasoning(text))
          }
          // Written but not yet run — the loop reports it when it starts it.
          // Showing it whole in the meantime is what the finished line will
          // say, so nothing jumps when the two swap over.
          AssistantContent::ToolCall(call) => {
            let line = self.claim(&call).unwrap_or_else(|| self.line(part));
            self.written.insert(call.id.to_string(), line.clone());
            Some(AgentEvent::ToolCallDelta {
              id: line,
              name: call.function.name.to_string(),
              args: call.function.arguments.to_string(),
            })
          }
          _ => None,
        }
      }
    }
  }

  /// What a turn stopped part-way had said, if anything: what each part had
  /// finished as, and what had arrived of the ones still going — a thought
  /// cut off is still the thought the text after it came from.
  ///
  /// The calls are left out, finished or not — a call nobody ran is one
  /// nothing would answer — and so is text with nothing in it.
  fn cut_off(self, stream: &CompletionStream) -> Option<Message> {
    let issuer = stream.reasoning_issuer();
    let content: Vec<AssistantContent> = self
      .parts
      .into_iter()
      .filter_map(|said| match said {
        Said::Text(text) => (!text.trim().is_empty()).then(|| AssistantContent::text(text)),
        Said::Thought(text) => {
          (!text.is_empty()).then(|| AssistantContent::Reasoning(Reasoning::new(&text).sealed(issuer.clone())))
        }
        Said::Done(content) => match *content {
          AssistantContent::ToolCall(_) => None,
          AssistantContent::Text(text) if text.text.trim().is_empty() => None,
          content => Some(content),
        },
        Said::Pending => None,
      })
      .collect();
    // No id: the provider never finished the message it would name, and
    // replaying a half of it under that name is asking to be told it is not
    // one.
    (!content.is_empty()).then_some(Message::Assistant { id: None, content })
  }
}

/// A call answered without being run.
fn aborted(call: &ToolCall) -> UserContent {
  UserContent::ToolResult(call.result(vec![ToolResultContent::text(ABORTED)]))
}

impl Runtime {
  /// What this session offers the model, in a stable order — a tool set that
  /// shuffled between requests would be a different prompt every time.
  fn definitions(&self) -> Vec<ToolDefinition> {
    let mut definitions = self.tools.now().static_tool_defs();
    definitions.retain(|definition| self.rules.permits(&definition.name));
    definitions
  }

  /// Run one call and answer it, telling the UI as it goes.
  async fn call(
    &self,
    call: &ToolCall,
    written: Option<&String>,
    tx: &mpsc::UnboundedSender<AgentEvent>,
  ) -> UserContent {
    let name = call.function.name.to_string();
    let id = call.id.to_string();
    let _ = tx.send(AgentEvent::ToolCall {
      name: name.clone(),
      args: call.function.arguments.clone(),
      call: id.clone(),
      internal: written.cloned().unwrap_or_else(|| id.clone()),
    });

    // Where to report output while the call runs, already answering for
    // this call. The loop dispatching it is the one that knows which it
    // is, so nothing has to be smuggled through the arguments to tell the
    // tool — and the tool never has to be told at all.
    let (reporting, under) = (tx.clone(), id.clone());
    let mut context = ToolContext::new().with_scope(Arc::new(Output(Arc::new(move |text| {
      let _ = reporting.send(AgentEvent::ToolOutput {
        call: under.clone(),
        text,
      });
    }))));
    // A tool the session was not offered is not run for being named anyway:
    // the list the model was shown is the boundary, not a hint. It is
    // answered as a tool that is not there, which is all the model was ever
    // told about it.
    let result = match self.rules.permits(&name) {
      true => {
        self
          .tools
          .now()
          .execute(&name, &call.function.arguments.to_string(), &mut context)
          .await
      }
      false => ToolResult::failed(
        ToolExecutionError::not_found(format!("`{name}` is not offered in this session"))
          .with_model_feedback(format!("tool `{name}` not found")),
      ),
    };

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
      diff: context.result::<EditDiff>().ok().flatten().map(|diff| diff.diff),
      output,
      images,
      is_error: result.is_error() || result.is_refused(),
    });
    // What the transcript shows and what the model is given are the same
    // bytes: there is one copy of a result, not a shown one and a sent one.
    UserContent::ToolResult(call.result(content))
  }
}

/// Reasoning another service produced, passed off as the adopter's so that
/// it goes out with the rest rather than being left behind.
///
/// Only what reads as a thought travels: its text and summaries. Signatures,
/// ciphertext and the reasoning's id mean something only to the service that
/// issued them, so they are dropped, and reasoning that was nothing but those
/// stays its issuer's. What the endpoint sends back as it is already is left
/// alone, and so is the transcript: this is the copy the request is made
/// from.
fn adopt_reasoning(history: &mut [Message], adopter: &Adopter) {
  for message in history {
    let Message::Assistant { content, .. } = message else {
      continue;
    };
    for part in content {
      let AssistantContent::Reasoning(sealed) = part else {
        continue;
      };
      if sealed.open_for(&adopter.native).is_some() {
        continue;
      }
      let Some(reasoning) = sealed.open(sealed.issuer()) else {
        continue;
      };
      let readable: Vec<ReasoningContent> = reasoning
        .content
        .iter()
        .filter_map(|block| match block {
          ReasoningContent::Text { text, .. } => Some(ReasoningContent::Text {
            text: text.clone(),
            signature: None,
          }),
          ReasoningContent::Summary(summary) => Some(ReasoningContent::Summary(summary.clone())),
          ReasoningContent::Encrypted(_) | ReasoningContent::Redacted { .. } => None,
        })
        .collect();
      if !readable.is_empty() {
        *sealed = Reasoning {
          id: None,
          content: readable,
        }
        .sealed(adopter.issuer.clone());
      }
    }
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

fn default_system_prompt(cwd: &Path) -> String {
  let mut prompt = String::from(
    "You are a helpful general-purpose assistant. \
     You help with anything: answering questions, research, writing, analysis and software work, \
     using your tools when the task calls for them. \
     Be concise and direct. Say plainly when you are unsure or something failed.\n",
  );

  let path = cwd.join("AGENTS.md");
  if let Ok(content) = std::fs::read_to_string(&path) {
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
  use rig_core::driver::{Exchange, Opened, Opening, Transport};
  use rig_core::test_utils::{MockFrame, MockScript, MockStreamEvent};
  use rig_core::wire::Mode;

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
  #[derive(Clone, Default)]
  struct Scripted {
    turns: Arc<Mutex<VecDeque<Vec<MockStreamEvent>>>>,
    /// Leave the stream open after the script runs out, so a turn can be
    /// stopped in the middle of itself rather than ending on its own.
    hang: bool,
    /// The cap each request carried, in the order they came.
    caps: Arc<Mutex<Vec<Option<u64>>>>,
  }

  impl Transport<MockScript> for Scripted {
    // The error type is the trait's, not ours to make smaller.
    #[allow(clippy::result_large_err)]
    fn send(&self, request: CompletionRequest, exchange: Exchange) -> Opening<MockFrame> {
      self.caps.lock().unwrap().push(request.max_tokens);
      let (turn, hang) = match exchange.mode {
        // Only the summarizer asks without streaming, and it is told the
        // same every time.
        Mode::Unary => (vec![said("Something about fruit.")], false),
        Mode::Streaming => (
          self
            .turns
            .lock()
            .expect("a script nobody panicked holding")
            .pop_front()
            .unwrap_or_default(),
          self.hang,
        ),
      };
      let frames = futures::stream::iter(turn.into_iter().map(|event| Ok(MockFrame::Event(event))));
      let frames: futures::stream::BoxStream<'static, _> = match hang {
        true => Box::pin(frames.chain(futures::stream::pending())),
        false => Box::pin(frames.chain(futures::stream::iter([Ok(MockFrame::Event(
          MockStreamEvent::FinalResponse(Default::default()),
        ))]))),
      };
      Opening::ready(Opened::new(frames))
    }
  }

  impl Scripted {
    fn model(&self) -> DynModel<Completion> {
      rig_core::Model::new(MockScript::default(), self.clone()).erase()
    }
  }

  fn said(text: &str) -> MockStreamEvent {
    MockStreamEvent::Text(text.to_string())
  }

  fn asks(id: &str, command: &str) -> MockStreamEvent {
    MockStreamEvent::ToolCall {
      id: id.to_string(),
      name: "bash".to_string(),
      arguments: serde_json::json!({ "command": command }),
      call_id: None,
    }
  }

  /// A runtime whose model reads from `turns` and whose only tool is the
  /// real `bash`, which is the one that can be told to take its time.
  fn scripted(turns: Vec<Vec<MockStreamEvent>>, hang: bool) -> Arc<Runtime> {
    let tools = ToolServer::new()
      .tool(BashTool {
        cwd: std::env::temp_dir(),
      })
      .run();
    let tools = Tools::new(tools);
    let model = Scripted {
      turns: Arc::new(Mutex::new(turns.into())),
      hang,
      ..Scripted::default()
    };
    Arc::new(Runtime {
      model: model.model(),
      tools,
      preamble: String::new(),
      compaction: TEST_SETTINGS,
      rules: Default::default(),
      relay_images: false,
      adopt_reasoning: None,
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
              result.call,
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
            AssistantContent::Reasoning(reasoning) => format!("thought {:?}", compaction::reasoning_text(reasoning)),
            AssistantContent::ToolCall(call) => format!("call {}", call.id),
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
    let thinking = MockStreamEvent::ReasoningDelta {
      id: "r1".to_string(),
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

  /// Pausing lets the call in hand finish rather than cutting it off, and
  /// stops before the model is asked again — with what was typed meanwhile
  /// already in the conversation, for `/continue` to answer.
  #[tokio::test]
  async fn a_paused_run_finishes_its_turn_and_asks_no_more() {
    let runtime = scripted(
      vec![
        vec![said("Working. "), asks("call_1", "sleep 0.5; echo one")],
        vec![said("Never asked for.")],
      ],
      false,
    );
    let control = Control::default();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let handle = start_run(runtime, control.clone(), Vec::new(), vec![Message::user("start")], tx);
    let mut paused = false;
    let (messages, done) = loop {
      let event = tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
        .await
        .expect("an event within 10s")
        .expect("the channel open");
      if !paused && matches!(&event, AgentEvent::ToolCall { call, .. } if call == "call_1") {
        paused = true;
        control.pause();
        control.steer(Prompt::text("and this".into()));
      }
      match event {
        AgentEvent::Done { messages } => break (messages, true),
        AgentEvent::Ended { messages } => break (messages, false),
        _ => {}
      }
    };
    handle.await.expect("the run to finish");
    assert!(!done, "a paused run ends short of its answer");
    assert_eq!(
      shapes(&messages),
      [
        "user start",
        "said \"Working. \" + call call_1",
        "result call_1 one",
        "user and this"
      ]
    );
    assert!(!control.steering());
  }

  /// A command waiting in the queue is not the run's to read: the run takes
  /// what was typed before it, and leaves it — and everything after it — to
  /// go in order once the run is over.
  #[test]
  fn the_run_reads_up_to_a_waiting_command_and_no_further() {
    let control = Control::default();
    let command = |text: &str| Prompt {
      command: true,
      ..Prompt::text(text.into())
    };
    control.steer(Prompt::text("first".into()));
    control.steer(command("/compact"));
    control.steer(Prompt::text("after".into()));
    assert!(control.steering(), "a message is in front");

    let said: Vec<String> = control.take_said().into_iter().map(|prompt| prompt.text).collect();
    assert_eq!(said, ["first"]);
    assert!(!control.steering(), "a command is not the run's to wait for");
    assert!(control.take_said().is_empty());
    assert_eq!(control.waiting(), ["/compact", "after"]);
    assert_eq!(
      control.take_next().map(|prompt| prompt.text).as_deref(),
      Some("/compact")
    );
  }

  /// A call to a tool the session refused is answered as one to a tool that
  /// is not there, and nothing is run: the model is only kept from a tool if
  /// naming it anyway does not get it.
  #[tokio::test]
  async fn a_refused_tool_is_not_run_for_being_named() {
    let witness = std::env::temp_dir().join(format!("fa-refused-{}", std::process::id()));
    let _ = std::fs::remove_file(&witness);
    let mut runtime = Arc::into_inner(scripted(
      vec![
        vec![asks("call_1", &format!("touch '{}'", witness.display()))],
        vec![said("fine")],
      ],
      false,
    ))
    .expect("the only handle");
    runtime.rules.refused = vec!["bash".into()];
    assert!(runtime.definitions().is_empty(), "bash is not offered");

    let (tx, mut rx) = mpsc::unbounded_channel();
    let handle = start_run(
      Arc::new(runtime),
      Control::default(),
      Vec::new(),
      vec![Message::user("make a file")],
      tx,
    );
    let mut refused = false;
    let messages = loop {
      let event = tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
        .await
        .expect("an event within 10s")
        .expect("the channel open");
      match event {
        AgentEvent::ToolResult { output, is_error, .. } => {
          assert!(is_error, "{output}");
          assert_eq!(output, "tool `bash` not found");
          refused = true;
        }
        AgentEvent::Done { messages } => break messages,
        AgentEvent::Ended { .. } => panic!("the run carried on to its answer"),
        _ => {}
      }
    };
    handle.await.expect("the run to finish");
    assert!(refused, "the call was answered");
    assert!(!witness.exists(), "the command was not run");
    assert_eq!(
      shapes(&messages)[1..],
      ["call call_1", "result call_1 tool `bash` not found", "said \"fine\""]
    );
  }

  /// A run that outgrows the window makes room for itself and carries on:
  /// what it got through goes out with `Compacting` for the session to keep,
  /// the summary takes its place, and the answer is asked for over that.
  #[tokio::test]
  async fn a_run_that_outgrows_the_window_compacts_and_carries_on() {
    let runtime = scripted(vec![vec![asks("call_1", "echo ok")], vec![said("all done")]], false);
    let (tx, mut rx) = mpsc::unbounded_channel();
    // Past the 900 tokens the window leaves, so the turn after the first
    // finds it full.
    let prompt = format!("remember the kumquat {}", "x".repeat(4000));
    let handle = start_run(runtime, Control::default(), Vec::new(), vec![Message::user(prompt)], tx);
    let mut seen = Vec::new();
    let done = loop {
      let event = tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
        .await
        .expect("an event within 10s")
        .expect("the channel open");
      match event {
        AgentEvent::Compacting { messages, resuming } => {
          assert!(resuming, "it stopped for room, so it goes on");
          assert_eq!(shapes(&messages)[1..], ["call call_1", "result call_1 ok"]);
          seen.push("compacting");
        }
        AgentEvent::Compacted(Some(compacted)) => {
          assert!(
            compacted.summary.contains("Something about fruit."),
            "{}",
            compacted.summary
          );
          assert_eq!(shapes(&compacted.kept), ["call call_1", "result call_1 ok"]);
          seen.push("compacted");
        }
        AgentEvent::Done { messages } => break messages,
        AgentEvent::Ended { .. } => panic!("the run carried on to its answer: {seen:?}"),
        AgentEvent::Error(err) => panic!("{err}"),
        _ => {}
      }
    };
    handle.await.expect("the run to finish");
    assert_eq!(seen, ["compacting", "compacted"]);
    // Only what came after the room was made: the rest went out before it.
    assert_eq!(shapes(&done), ["said \"all done\""]);
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
        AgentEvent::Usage { usage, context_tokens } => Some((usage.output_tokens.unwrap_or(0), *context_tokens)),
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
        AgentEvent::Compacting { .. } => "compacting".into(),
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
        rig_core::message::CallId::from_wire("call"),
        rig_core::message::ToolName::new("bash").unwrap(),
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
        input_tokens: Some(500),
        output_tokens: Some(20),
        ..Usage::default()
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
    weigh.answered(&Usage::default(), 9);
    assert_eq!(weigh.request(&chat), 520 + 200 + 100);
  }

  /// A cached prompt is still a prompt: what the provider read back from its
  /// cache was in the request, and rig counts it as input, so the context is
  /// the prompt and the answer — and a counter nobody reported is nothing.
  #[test]
  fn the_context_is_the_prompt_and_the_answer() {
    let cached = Usage {
      input_tokens: Some(32_012),
      output_tokens: Some(40),
      cached_input_tokens: Some(30_000),
      cache_creation_input_tokens: Some(2_000),
      total_tokens: Some(32_052),
      ..Usage::default()
    };
    assert_eq!(weight(&cached), 32_052);

    // A provider that reports only the prompt is taken at its word for that.
    let prompt = Usage {
      input_tokens: Some(500),
      ..Usage::default()
    };
    assert_eq!(weight(&prompt), 500);

    assert_eq!(weight(&Usage::default()), 0);
  }

  #[test]
  fn relay_moves_tool_images_into_a_following_user_message() {
    use rig_core::message::{CallId, LocalCallId, ToolName};
    let result = |content: Vec<ToolResultContent>| {
      UserContent::tool_result(
        CallId::from(LocalCallId::new()),
        ToolName::new("read").unwrap(),
        content,
      )
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
    let recording = Scripted {
      turns: Arc::new(Mutex::new([vec![said("done")]].into())),
      ..Scripted::default()
    };
    let runtime = Arc::new(Runtime {
      model: recording.model(),
      tools: Tools::new(ToolServer::new().run()),
      preamble: String::new(),
      compaction: TEST_SETTINGS,
      rules: Default::default(),
      relay_images: false,
      adopt_reasoning: None,
      max_tokens: Some(99),
    });
    let (tx, mut rx) = mpsc::unbounded_channel();
    collect(&runtime, Vec::new(), "hello", &mut rx, &tx).await;

    let _ = runtime.ask("summarize".to_string()).await;

    assert_eq!(*recording.caps.lock().unwrap(), vec![Some(99), Some(99)]);
  }

  /// Every provider there is, and whether an image it is sent inside a tool
  /// result has to be relayed after it instead.
  const PROVIDERS: [(Provider, bool); 24] = [
    (Provider::OpenAi, true),
    (Provider::OpenAiResponses, false),
    (Provider::Azure, true),
    (Provider::OpenRouter, true),
    (Provider::Ollama, true),
    (Provider::Gemini, false),
    (Provider::Anthropic, false),
    (Provider::Cohere, true),
    (Provider::DeepSeek, true),
    (Provider::Doubleword, true),
    (Provider::Groq, true),
    (Provider::HuggingFace, true),
    (Provider::Hyperbolic, true),
    (Provider::LlamaCpp, false),
    (Provider::MiniMax, true),
    (Provider::Mira, true),
    (Provider::Mistral, true),
    (Provider::Moonshot, true),
    (Provider::Perplexity, true),
    (Provider::Together, true),
    (Provider::Venice, true),
    (Provider::XAi, false),
    (Provider::XiaomiMimo, true),
    (Provider::ZAi, true),
  ];

  /// Every provider builds, with a key and without one, and only the ones
  /// that take an image inside a tool result skip the relay.
  #[test]
  fn each_provider_builds_a_model() {
    for (provider, relay) in PROVIDERS {
      for (key, base_url) in [("", None), ("k", Some("http://127.0.0.1:1/v1/".to_string()))] {
        // Azure has no endpoint to fall back on; that refusal is tested on
        // its own.
        if provider == Provider::Azure && base_url.is_none() {
          continue;
        }
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

  /// Azure is refused without an endpoint, and told what to give instead of
  /// being built against nowhere.
  #[test]
  fn azure_needs_an_endpoint() {
    let cfg = Config {
      provider: Provider::Azure,
      base_url: None,
      api_key: "k".into(),
      model: "mock".into(),
      system_prompt: None,
      max_tokens: None,
      compaction: TEST_SETTINGS,
      vision: true,
      tools: Default::default(),
    };
    let (tx, _rx) = mpsc::unbounded_channel();
    let Err(err) = build_agents(&cfg, Path::new("/tmp"), &Host::new(tx), &Default::default()) else {
      panic!("Azure built with no endpoint");
    };
    assert!(format!("{err:#}").contains("--base-url"), "{err:#}");
  }

  /// Chat Completions takes thoughts from anywhere, as its own; Claude, and
  /// the wires that replay reasoning as items of their own, do not.
  #[test]
  fn chat_completions_adopts_reasoning() {
    for (provider, model, adopter) in [
      (Provider::LlamaCpp, "mock", Some("llamacpp")),
      (Provider::OpenAi, "mock", Some("openai")),
      (Provider::DeepSeek, "deepseek-reasoner", Some("deepseek")),
      (Provider::OpenRouter, "deepseek/deepseek-r1", Some("openrouter")),
      (Provider::OpenRouter, "anthropic/claude-sonnet-4.5", None),
      (Provider::OpenAiResponses, "mock", None),
      (Provider::Anthropic, "claude-sonnet-4-5", None),
      (Provider::Gemini, "mock", None),
      (Provider::Ollama, "mock", None),
    ] {
      let cfg = Config {
        provider,
        base_url: Some("http://127.0.0.1:1/v1".to_string()),
        api_key: "k".into(),
        model: model.into(),
        system_prompt: None,
        max_tokens: None,
        compaction: TEST_SETTINGS,
        vision: true,
        tools: Default::default(),
      };
      let (_, _, adopt) = build_model(&cfg).unwrap();
      assert_eq!(
        adopt.as_ref().map(|adopt| adopt.issuer.as_str()),
        adopter,
        "{provider:?} {model}"
      );
    }
  }

  /// What reads as a thought is passed off as the adopter's, without what only
  /// its issuer could check; the rest, and what the endpoint sends back as it
  /// is already, is left as it was.
  #[test]
  fn adopted_reasoning_keeps_what_reads() {
    let reasoning = |issuer: &'static str, id: Option<&str>, content: Vec<ReasoningContent>| {
      AssistantContent::Reasoning(
        Reasoning {
          id: id.map(String::from),
          content,
        }
        .sealed(issuer),
      )
    };
    let signed = ReasoningContent::Text {
      text: "think".into(),
      signature: Some("sig".into()),
    };
    let foreign = reasoning(
      "llamacpp",
      Some("r1"),
      vec![
        signed.clone(),
        ReasoningContent::Encrypted("blob".into()),
        ReasoningContent::Summary("in short".into()),
      ],
    );
    let sealed_shut = reasoning("anthropic", None, vec![ReasoningContent::Redacted { data: "x".into() }]);
    let upstream = reasoning("openrouter/deepseek", Some("mine"), vec![signed]);
    let mut history = vec![
      Message::user("hi"),
      Message::Assistant {
        id: None,
        content: vec![
          foreign,
          sealed_shut.clone(),
          upstream.clone(),
          AssistantContent::text("done"),
        ],
      },
    ];
    let adopter = Adopter {
      issuer: Issuer::from_static("openrouter"),
      native: vec![
        Issuer::from_static("openrouter"),
        Issuer::from_static("openrouter/deepseek"),
      ],
    };
    adopt_reasoning(&mut history, &adopter);
    let Message::Assistant { content, .. } = &history[1] else {
      panic!("not an answer")
    };
    assert_eq!(
      content[0],
      reasoning(
        "openrouter",
        None,
        vec![
          ReasoningContent::Text {
            text: "think".into(),
            signature: None,
          },
          ReasoningContent::Summary("in short".into()),
        ],
      )
    );
    assert_eq!(content[1], sealed_shut);
    assert_eq!(content[2], upstream);
    assert_eq!(content[3], AssistantContent::text("done"));
  }

  /// The key is read from the variable the provider's own tools use.
  #[test]
  fn each_provider_reads_its_own_key_variable() {
    assert_eq!(Provider::OpenAi.key_env(), "OPENAI_API_KEY");
    assert_eq!(Provider::OpenAiResponses.key_env(), "OPENAI_API_KEY");
    assert_eq!(Provider::XiaomiMimo.key_env(), "XIAOMI_MIMO_API_KEY");
    assert_eq!(Provider::ZAi.key_env(), "ZAI_API_KEY");
    assert_eq!(Provider::HuggingFace.key_env(), "HUGGINGFACE_API_KEY");
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
