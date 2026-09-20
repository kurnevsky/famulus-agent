mod agent;
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
  DisableMouseCapture, EnableMouseCapture, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
  PushKeyboardEnhancementFlags,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::supports_keyboard_enhancement;
use tokio::sync::mpsc;

/// fa: a minimal terminal coding agent for OpenAI-compatible APIs and Gemini.
#[derive(Parser)]
#[command(name = "fa", version)]
struct Cli {
  /// API flavour to speak
  #[arg(long, env = "FA_PROVIDER", value_enum, default_value_t = ProviderArg::Openai)]
  provider: ProviderArg,

  /// Endpoint root, e.g. http://localhost:11434/v1 for an OpenAI-compatible
  /// server. Defaults to the provider's public API.
  #[arg(long, env = "OPENAI_BASE_URL")]
  base_url: Option<String>,

  /// API key. Falls back to OPENAI_API_KEY or GEMINI_API_KEY depending on the
  /// provider; any value works for servers without auth.
  #[arg(long, env = "FA_API_KEY", hide_env_values = true)]
  api_key: Option<String>,

  /// Model name to request
  #[arg(short, long, env = "FA_MODEL")]
  model: String,

  /// Replace the built-in system prompt
  #[arg(long, env = "FA_SYSTEM_PROMPT")]
  system_prompt: Option<String>,

  /// Maximum model calls (tool rounds) per user message
  #[arg(long, default_value_t = 50)]
  max_turns: usize,

  /// Context window of the model in tokens; compaction triggers near this limit
  #[arg(long, env = "FA_CONTEXT_WINDOW", default_value_t = 128_000)]
  context_window: u64,

  /// Compact once fewer than this many tokens remain in the context window
  #[arg(long, default_value_t = 16_384)]
  reserve_tokens: u64,

  /// Approximate number of recent tokens kept verbatim when compacting
  #[arg(long, default_value_t = 20_000)]
  keep_recent_tokens: u64,

  /// Disable automatic compaction (/compact still works)
  #[arg(long)]
  no_compaction: bool,

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
  /// Google Gemini
  Gemini,
}

#[tokio::main]
async fn main() -> Result<()> {
  let cli = Cli::parse();
  let (provider, key_env) = match cli.provider {
    ProviderArg::Openai => (agent::Provider::OpenAi, "OPENAI_API_KEY"),
    ProviderArg::Gemini => (agent::Provider::Gemini, "GEMINI_API_KEY"),
  };
  let api_key = cli
    .api_key
    .or_else(|| std::env::var(key_env).ok().filter(|k| !k.is_empty()))
    .unwrap_or_else(|| "none".to_string());
  let mut cfg = agent::Config {
    provider,
    base_url: cli.base_url,
    api_key,
    model: cli.model,
    system_prompt: cli.system_prompt,
    max_turns: cli.max_turns,
    vision: !cli.no_vision,
    // Worked out below, once the MCP servers have said what they brought.
    tools: None,
    compaction: compaction::Settings {
      enabled: !cli.no_compaction,
      context_window: cli.context_window,
      reserve_tokens: cli.reserve_tokens,
      keep_recent_tokens: cli.keep_recent_tokens,
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
  let model_label = format!(
    "{}/{}",
    match cli.provider {
      ProviderArg::Openai => "openai",
      ProviderArg::Gemini => "gemini",
    },
    cfg.model
  );

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
    tx,
    ui::Options {
      model: model_label,
      cwd,
      settings: cfg.compaction,
      scrollbar: cli.scrollbar,
      store,
      start,
      notes,
      mcp: servers.count(),
    },
  );

  let mut terminal = ratatui::init();
  // Mouse wheel scrolls the transcript. Note: while captured, the terminal's
  // own drag-selection usually needs Shift held.
  let _ = execute!(stdout(), EnableMouseCapture);
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
  let _ = execute!(stdout(), DisableMouseCapture);
  ratatui::restore();
  result
}
