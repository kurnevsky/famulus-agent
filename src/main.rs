mod agent;
mod ask;
mod attach;
mod clipboard;
mod compaction;
mod edit;
mod highlight;
mod images;
mod markdown;
mod mcp;
mod session;
mod tools;
mod ui;

use std::io::stdout;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use ratatui::crossterm::event::{
  DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, KeyboardEnhancementFlags,
  PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::supports_keyboard_enhancement;
use tokio::sync::mpsc;

/// famulus-agent: a minimal terminal coding agent for OpenAI-compatible APIs,
/// Anthropic, Gemini, and a dozen more.
#[derive(Parser)]
#[command(name = "fa", version)]
struct Cli {
  /// API flavour to speak
  #[arg(long, env = "FA_PROVIDER", value_enum, default_value_t = ProviderArg::Openai)]
  provider: ProviderArg,

  /// Endpoint root, e.g. http://localhost:8080/v1 for an OpenAI-compatible
  /// server. Defaults to the provider's public API.
  #[arg(long, env = "FA_BASE_URL")]
  base_url: Option<String>,

  /// API key. Falls back to the provider's own environment variable
  /// (OPENAI_API_KEY, OPENROUTER_API_KEY, OLLAMA_API_KEY, GEMINI_API_KEY);
  /// any value works for servers without auth.
  #[arg(long, env = "FA_API_KEY", hide_env_values = true)]
  api_key: Option<String>,

  /// Model name to request. /model changes it later, and lists what the
  /// provider has to change it to
  #[arg(short, long, env = "FA_MODEL")]
  model: String,

  /// Replace the built-in system prompt
  #[arg(long, env = "FA_SYSTEM_PROMPT")]
  system_prompt: Option<String>,

  /// Cap on what one answer may come to, in tokens. Left to the provider when
  /// unset — except for Anthropic, which requires one and is given what the
  /// named model allows, or 2048 for a model rig does not know
  #[arg(long, env = "FA_MAX_TOKENS")]
  max_tokens: Option<u64>,

  /// Context window of the model in tokens; compaction triggers near this
  /// limit. Given here it stands whatever the model is, and left out it is
  /// what the provider reports for the chosen model — or 128000, for the
  /// providers that report nothing
  #[arg(long, env = "FA_CONTEXT_WINDOW")]
  context_window: Option<u64>,

  /// Compact once fewer than this many tokens remain in the context window
  #[arg(long, default_value_t = 16_384)]
  reserve_tokens: u64,

  /// Approximate number of recent tokens kept verbatim when compacting
  #[arg(long, default_value_t = 20_000)]
  keep_recent_tokens: u64,

  /// Disable automatic compaction (/compact still works)
  #[arg(long)]
  no_compaction: bool,

  /// Do not summarize the start of a split turn separately when compacting:
  /// one summarizer call rather than two, and what the turn was for goes into
  /// the checkpoint with everything else
  #[arg(long)]
  no_turn_summary: bool,

  /// The model cannot take images: `read` describes image files but omits their data
  #[arg(long, env = "FA_NO_VISION")]
  no_vision: bool,

  /// Continue the most recent session
  #[arg(short = 'c', long = "continue")]
  continue_: bool,

  /// Pick a session to resume
  #[arg(short = 'r', long, conflicts_with = "continue_")]
  resume: bool,

  /// Resume a specific session by file path or id (prefix)
  #[arg(long, conflicts_with_all = ["continue_", "resume"])]
  session: Option<String>,

  /// Do not save this session to disk
  #[arg(long)]
  no_session: bool,

  /// Directory holding session files (global, not per project)
  #[arg(long, env = "FA_SESSIONS_DIR")]
  sessions_dir: Option<PathBuf>,

  /// Transcript scrollbar: shown briefly while scrolling, always, or never
  #[arg(long, env = "FA_SCROLLBAR", value_enum, default_value_t = ui::ScrollbarMode::Auto)]
  scrollbar: ui::ScrollbarMode,

  /// Do not ring the terminal when the model asks a question
  #[arg(long, env = "FA_NO_BELL")]
  no_bell: bool,

  /// MCP servers to start, instead of the mcp.toml in $XDG_CONFIG_HOME/fa
  #[arg(long, env = "FA_MCP_CONFIG")]
  mcp_config: Option<PathBuf>,

  /// Start no MCP servers this session
  #[arg(long, conflicts_with = "mcp_config")]
  no_mcp: bool,

  /// Offer the model only these tools, by name (comma-separated)
  #[arg(long, env = "FA_TOOLS", value_delimiter = ',')]
  tools: Vec<String>,

  /// Keep these tools from the model, by name (comma-separated)
  #[arg(long, env = "FA_NO_TOOLS", value_delimiter = ',')]
  no_tools: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ProviderArg {
  /// OpenAI Chat Completions and compatible servers
  Openai,
  /// OpenRouter
  Openrouter,
  /// Ollama, on http://localhost:11434 by default
  Ollama,
  /// Google Gemini
  Gemini,
  /// Anthropic
  Anthropic,
  /// Cohere
  Cohere,
  /// DeepSeek
  Deepseek,
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
  Xai,
}

/// Which provider an argument names, which variable its key is read from,
/// and what key to use when neither the flag nor that variable says.
struct Wire {
  provider: agent::Provider,
  /// `None` for llamafile, whose client takes no key at all, so there is no
  /// variable worth reading.
  key_env: Option<&'static str>,
  /// Ollama and llamafile want no key at all — rig leaves the header off for
  /// an empty one, which is what a local server expects — while a hosted
  /// endpoint that ignores auth is happy with anything.
  no_key: &'static str,
}

impl ProviderArg {
  fn wire(self) -> Wire {
    use agent::Provider as P;
    let hosted = |provider, key_env| Wire {
      provider,
      key_env: Some(key_env),
      no_key: "none",
    };
    let local = |provider, key_env| Wire {
      provider,
      key_env,
      no_key: "",
    };
    match self {
      Self::Openai => hosted(P::OpenAi, "OPENAI_API_KEY"),
      Self::Openrouter => hosted(P::OpenRouter, "OPENROUTER_API_KEY"),
      Self::Ollama => local(P::Ollama, Some("OLLAMA_API_KEY")),
      Self::Gemini => hosted(P::Gemini, "GEMINI_API_KEY"),
      Self::Anthropic => hosted(P::Anthropic, "ANTHROPIC_API_KEY"),
      Self::Cohere => hosted(P::Cohere, "COHERE_API_KEY"),
      Self::Deepseek => hosted(P::DeepSeek, "DEEPSEEK_API_KEY"),
      Self::Doubleword => hosted(P::Doubleword, "DOUBLEWORD_API_KEY"),
      Self::Groq => hosted(P::Groq, "GROQ_API_KEY"),
      Self::Hyperbolic => hosted(P::Hyperbolic, "HYPERBOLIC_API_KEY"),
      Self::Llamafile => local(P::Llamafile, None),
      Self::Mira => hosted(P::Mira, "MIRA_API_KEY"),
      Self::Mistral => hosted(P::Mistral, "MISTRAL_API_KEY"),
      Self::Perplexity => hosted(P::Perplexity, "PERPLEXITY_API_KEY"),
      Self::Together => hosted(P::Together, "TOGETHER_API_KEY"),
      Self::Venice => hosted(P::Venice, "VENICE_API_KEY"),
      Self::Xai => hosted(P::XAi, "XAI_API_KEY"),
    }
  }
}

#[tokio::main]
async fn main() -> Result<()> {
  let cli = Cli::parse();
  let wire = cli.provider.wire();
  let api_key = cli
    .api_key
    .or_else(|| {
      wire
        .key_env
        .and_then(|env| std::env::var(env).ok())
        .filter(|k| !k.is_empty())
    })
    .unwrap_or_else(|| wire.no_key.to_string());
  let mut cfg = agent::Config {
    provider: wire.provider,
    base_url: cli.base_url,
    api_key,
    model: cli.model,
    system_prompt: cli.system_prompt,
    max_tokens: cli.max_tokens,
    vision: !cli.no_vision,
    // Worked out below, once the MCP servers have said what they brought.
    tools: None,
    compaction: compaction::Settings {
      enabled: !cli.no_compaction,
      // What the flag says, or the fallback until the provider is asked what
      // the chosen model holds.
      context_window: cli.context_window.unwrap_or(compaction::DEFAULT_CONTEXT_WINDOW),
      reserve_tokens: cli.reserve_tokens,
      keep_recent_tokens: cli.keep_recent_tokens,
      turn_summary: !cli.no_turn_summary,
    },
  };
  let cwd = std::env::current_dir()?;

  let store = (!cli.no_session)
    .then(|| session::Store::new(cli.sessions_dir.clone().unwrap_or_else(session::Store::default_dir)));
  let start = if let Some(wanted) = &cli.session {
    let found = store
      .as_ref()
      .and_then(|s| s.find(wanted))
      .or_else(|| Path::new(wanted).is_file().then(|| PathBuf::from(wanted)));
    match found {
      Some(path) => ui::SessionStart::Path(path),
      None => bail!("session not found: {wanted}"),
    }
  } else if cli.continue_ {
    ui::SessionStart::Continue
  } else if cli.resume {
    ui::SessionStart::Resume
  } else {
    ui::SessionStart::New
  };
  // The servers come up before the terminal does, and stay up as long as this
  // binding: a stdio server is a child process of ours, and closing the
  // connection is what stops it.
  let (servers, notes) = match cli.no_mcp {
    true => (mcp::Servers::default(), Vec::new()),
    false => {
      let files = mcp::files(cli.mcp_config.as_deref());
      let (config, mut notes) = mcp::load(&files, cli.mcp_config.is_some());
      let servers = mcp::connect(config).await;
      notes.extend(servers.notes().iter().cloned());
      (servers, notes)
    }
  };

  // What this session may call: the built-in tools and whatever the servers
  // brought are one list, since the model is offered them as one.
  let available: Vec<String> = tools::BUILT_IN
    .iter()
    .map(|name| name.to_string())
    .chain(servers.tool_names())
    .collect();
  let (allowed, unknown) = tools::choose(&available, &cli.tools, &cli.no_tools);
  let mut notes = notes;
  if !unknown.is_empty() {
    notes.push(format!(
      "No tool named {} — nothing left in or out by it.",
      unknown.join(", ")
    ));
  }
  if let Some(allowed) = &allowed {
    notes.push(match allowed.is_empty() {
      true => "No tools this session: the model can only answer.".to_string(),
      false => format!("Tools this session: {}.", allowed.join(", ")),
    });
  }
  cfg.tools = allowed;

  let (tx, rx) = mpsc::unbounded_channel();
  let agents = agent::build_agents(&cfg, &cwd, tx.clone(), &servers)?;
  let app = ui::App::new(
    agents,
    cfg,
    tx,
    ui::Options {
      cwd,
      context_window: cli.context_window,
      scrollbar: cli.scrollbar,
      store,
      start,
      notes,
      mcp: servers.count(),
      bell: !cli.no_bell,
    },
  );

  let mut terminal = ratatui::init();
  // Mouse wheel scrolls the transcript. Note: while captured, the terminal's
  // own drag-selection usually needs Shift held.
  let _ = execute!(stdout(), EnableMouseCapture);
  // Pasted text arrives in one piece rather than as the keys it is made of,
  // so a paste of several lines lands as several lines instead of sending the
  // first one at its first newline.
  let bracketed = execute!(stdout(), EnableBracketedPaste).is_ok();
  // Lets terminals that speak the kitty keyboard protocol report Shift+Enter.
  let enhanced = matches!(supports_keyboard_enhancement(), Ok(true))
    && execute!(
      stdout(),
      PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    )
    .is_ok();

  let result = app.run(&mut terminal, rx).await;

  if enhanced {
    let _ = execute!(stdout(), PopKeyboardEnhancementFlags);
  }
  if bracketed {
    let _ = execute!(stdout(), DisableBracketedPaste);
  }
  let _ = execute!(stdout(), DisableMouseCapture);
  ratatui::restore();
  result
}
