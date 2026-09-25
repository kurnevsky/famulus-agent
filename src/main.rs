mod agent;
mod ask;
mod attach;
#[cfg(feature = "mcp")]
mod call;
mod cells;
mod clipboard;
mod compaction;
mod config;
mod edit;
#[cfg(feature = "mcp")]
mod elicit;
mod highlight;
mod images;
#[cfg(feature = "mcp")]
mod keyring;
mod markdown;
mod mcp;
mod modal;
#[cfg(feature = "mcp")]
mod oauth;
mod pdf;
#[cfg(feature = "mcp")]
mod prompts;
#[cfg(feature = "mcp")]
mod resources;
mod session;
mod tools;
mod ui;

use std::io::stdout;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
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
  /// Settings to start from, instead of the config.toml in $XDG_CONFIG_HOME/fa.
  /// A flag given here, or its variable, beats what the file says
  #[arg(long, env = "FA_CONFIG")]
  config: Option<PathBuf>,

  /// API flavour to speak [default: openai]
  #[arg(long, env = "FA_PROVIDER", value_enum)]
  provider: Option<agent::Provider>,

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
  /// provider has to change it to. Required, here or in config.toml
  #[arg(short, long, env = "FA_MODEL")]
  model: Option<String>,

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
  /// [default: 16384]
  #[arg(long)]
  reserve_tokens: Option<u64>,

  /// Approximate number of recent tokens kept verbatim when compacting
  /// [default: 20000]
  #[arg(long)]
  keep_recent_tokens: Option<u64>,

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
  /// [default: auto]
  #[arg(long, env = "FA_SCROLLBAR", value_enum)]
  scrollbar: Option<ui::ScrollbarMode>,

  /// Do not ring the terminal when the model asks a question
  #[arg(long, env = "FA_NO_BELL")]
  no_bell: bool,

  /// MCP servers to start, instead of the mcp.toml in $XDG_CONFIG_HOME/fa
  #[arg(long, env = "FA_MCP_CONFIG")]
  mcp_config: Option<PathBuf>,

  /// Start no MCP servers this session
  #[arg(long, conflicts_with = "mcp_config")]
  no_mcp: bool,

  /// Offer the model only these of fa's own tools, by name (comma-separated)
  #[arg(long, env = "FA_TOOLS", value_delimiter = ',')]
  tools: Vec<String>,

  /// Keep these of fa's own tools from the model, by name (comma-separated)
  #[arg(long, env = "FA_NO_TOOLS", value_delimiter = ',')]
  no_tools: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
  let mut cli = Cli::parse();
  // What the file says fills in whatever the flags and their variables left
  // out, and nothing more: a flag is how one session differs from the rest.
  let file = config::load(
    &config::files(config::FILE, cli.config.as_deref()),
    cli.config.is_some(),
  )?;
  let provider = cli.provider.or(file.provider).unwrap_or(agent::Provider::OpenAi);
  let Some(model) = cli.model.take().or(file.model) else {
    bail!(
      "no model: give one with --model, FA_MODEL, or `model` in {}",
      config::FILE
    );
  };
  // The provider's own variable is a variable like any other, so it beats the
  // file too; the file's command only runs when nothing else gave a key.
  let env_key = std::env::var(provider.key_env()).ok().filter(|k| !k.is_empty());
  let api_key = match cli.api_key.take().or(env_key).or(file.api_key) {
    Some(key) => key,
    None => match &file.api_key_command {
      Some(command) => config::value(command).await.context("api-key-command")?,
      None => provider.no_key().to_string(),
    },
  };
  let context_window = cli.context_window.or(file.context_window);
  let sessions_dir = cli.sessions_dir.take().or(file.sessions_dir);
  // A file named on the command line beats a file that says to start none.
  let no_mcp = cli.no_mcp || (file.no_mcp && cli.mcp_config.is_none());
  let mcp_config = cli.mcp_config.take().or(file.mcp_config);
  let tools = match cli.tools.is_empty() {
    true => file.tools.unwrap_or_default(),
    false => std::mem::take(&mut cli.tools),
  };
  let no_tools = match cli.no_tools.is_empty() {
    true => file.no_tools.unwrap_or_default(),
    false => std::mem::take(&mut cli.no_tools),
  };
  let mut cfg = agent::Config {
    provider,
    base_url: cli.base_url.take().or(file.base_url),
    api_key,
    model,
    system_prompt: cli.system_prompt.take().or(file.system_prompt),
    max_tokens: cli.max_tokens.or(file.max_tokens),
    vision: !(cli.no_vision || file.no_vision),
    // Worked out below, once the MCP servers have said what they brought.
    tools: Default::default(),
    compaction: compaction::Settings {
      enabled: !(cli.no_compaction || file.no_compaction),
      // What the flag says, or the fallback until the provider is asked what
      // the chosen model holds.
      context_window: context_window.unwrap_or(compaction::DEFAULT_CONTEXT_WINDOW),
      reserve_tokens: cli.reserve_tokens.or(file.reserve_tokens).unwrap_or(16_384),
      keep_recent_tokens: cli.keep_recent_tokens.or(file.keep_recent_tokens).unwrap_or(20_000),
      turn_summary: !(cli.no_turn_summary || file.no_turn_summary),
    },
  };
  let cwd = std::env::current_dir()?;

  let store = (!(cli.no_session || file.no_session))
    .then(|| session::Store::new(sessions_dir.unwrap_or_else(session::Store::default_dir)));
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
  // Everything that wants the user's attention reaches the UI down this
  // channel: the run's events, and the questions its tools and servers ask.
  let (tx, rx) = mpsc::unbounded_channel();
  let host = modal::Host::new(tx.clone());

  // The servers come up before the terminal does, and stay up as long as this
  // binding: a stdio server is a child process of ours, and closing the
  // connection is what stops it.
  let (servers, mut notes) = match no_mcp {
    true => (mcp::Servers::default(), Vec::new()),
    false => {
      let files = mcp::files(mcp_config.as_deref());
      let (config, mut notes) = mcp::load(&files, mcp_config.is_some());
      let servers = mcp::connect(config, &host).await;
      notes.extend(servers.notes().iter().cloned());
      (servers, notes)
    }
  };

  // What `--tools` and `--no-tools` choose from: fa's own tools, and nothing
  // a server brought — a server's tools are narrowed in its own table.
  let theirs: Vec<String> = servers.catalog().tool_names();
  let available: Vec<String> = tools::BUILT_IN
    .iter()
    .map(|name| name.to_string())
    .chain(theirs.iter().filter(|name| tools::own(name)).cloned())
    .collect();
  let (allowed, unknown) = tools::choose(&available, &tools, &no_tools);
  let (servers_own, unknown): (Vec<String>, Vec<String>) = unknown.into_iter().partition(|name| theirs.contains(name));
  if !servers_own.is_empty() {
    notes.push(format!(
      "--tools and --no-tools are for fa's own tools, not {} — a server's are narrowed with `tools` or `except` in mcp.toml.",
      servers_own.join(", ")
    ));
  }
  if !unknown.is_empty() {
    notes.push(format!(
      "No tool named {} — nothing left in or out by it.",
      unknown.join(", ")
    ));
  }
  cfg.tools = tools::Rules {
    refused: available.into_iter().filter(|name| !allowed.contains(name)).collect(),
  };

  let agents = agent::build_agents(&cfg, &cwd, &host, &servers)?;
  let app = ui::App::new(
    agents,
    cfg,
    tx,
    ui::Options {
      cwd,
      context_window,
      scrollbar: cli.scrollbar.or(file.scrollbar).unwrap_or(ui::ScrollbarMode::Auto),
      store,
      start,
      notes,
      catalog: servers.catalog().clone(),
      bell: !(cli.no_bell || file.no_bell),
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
  let enhanced = supports_keyboard_enhancement().unwrap_or(false)
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
