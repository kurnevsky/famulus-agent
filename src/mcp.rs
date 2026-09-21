//! Tools from MCP servers.
//!
//! Rig speaks the protocol, through `rmcp`. What is here is the finding and the
//! starting: which servers a session should have, how to reach each one, and
//! handing what they offer to the agent as tools like any other. A server is
//! either a program that talks over its own stdin and stdout, or an endpoint
//! that speaks streamable HTTP.
//!
//! The file is this program's own, so it reads the way the rest of it does: a
//! table per server, named by the name its tools will be called under, and a
//! `command` written as a line of shell, like the one `bash` takes. Where it
//! lives is the XDG search path and nothing else — no dotfile in a home
//! directory, nothing beside the project.
//!
//! ```toml
//! [fetch]
//! command = "uvx mcp-server-fetch"
//!
//! [docs]
//! url = "https://example.com/mcp"
//! headers.Authorization = "Bearer …"
//! timeout = 60
//! ```

#[cfg(feature = "mcp")]
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "mcp")]
use serde::Deserialize;

/// How long a server has to come up and say what it offers. A server that is
/// slower than this is a session that never starts, which is worse than a
/// session without it.
#[cfg(feature = "mcp")]
const START_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// The name servers are declared under, in each configuration directory.
const FILE: &str = "mcp.toml";

/// Servers as the file declares them: one table each, under its own name.
///
/// The names are the file's top level, so the file is nothing but servers.
/// Anything else it says is a mistake worth hearing about rather than a key
/// for a client that is not this one.
#[cfg(feature = "mcp")]
pub type Config = BTreeMap<String, Server>;

/// One server: a command to run, or a URL to call.
///
/// A key that is not one of these is an error, not something to read past —
/// this file is fa's own, and a misspelled `comand` that went quietly would be
/// a server that never came up for no stated reason.
#[cfg(feature = "mcp")]
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Server {
  /// A line of shell run for a server that speaks over its own stdin and
  /// stdout — the same shape `bash` takes, and run the same way, so a command
  /// that works in the terminal works here.
  #[serde(default)]
  pub command: Option<String>,
  /// Added to the environment that command inherits.
  #[serde(default)]
  pub env: BTreeMap<String, String>,
  /// The endpoint of a server that speaks streamable HTTP.
  #[serde(default)]
  pub url: Option<String>,
  /// Sent with every request to that endpoint, which is where a token goes.
  #[serde(default)]
  pub headers: BTreeMap<String, String>,
  /// Seconds one of this server's tools may take before the call comes back
  /// as an error the model can recover from. `0` waits forever.
  #[serde(default)]
  pub timeout: Option<u64>,
  /// Take only these of the tools it offers. Everything it offers, when empty
  /// — which is not the same as naming them all, since a server that grows a
  /// tool later would keep it.
  #[serde(default)]
  pub tools: Vec<String>,
  /// Take everything but these. Named in both lists, a tool is refused.
  #[serde(default)]
  pub except: Vec<String>,
}

/// Where servers are declared, in the order the files are read: the system's
/// first, the user's last, so the nearer file wins the names they share.
///
/// The search is the XDG one — `$XDG_CONFIG_HOME` then `$XDG_CONFIG_DIRS`,
/// each with their spec defaults — and nothing else. No dotfile in a home
/// directory, and none beside the project either: a file in the working
/// directory would be a file whose name depends on where fa was started.
pub fn files(explicit: Option<&Path>) -> Vec<PathBuf> {
  if let Some(path) = explicit {
    return vec![path.to_path_buf()];
  }
  let mut files: Vec<PathBuf> = config_dirs()
    .into_iter()
    .rev()
    .map(|dir| dir.join("fa").join(FILE))
    .collect();
  files.dedup();
  files
}

/// The XDG configuration directories, nearest first.
fn config_dirs() -> Vec<PathBuf> {
  let var = |name| std::env::var_os(name).map(|value| value.to_string_lossy().into_owned());
  search(var("XDG_CONFIG_HOME"), var("HOME"), var("XDG_CONFIG_DIRS"))
}

/// The XDG search path, from the three variables that decide it, so what it
/// works out can be said without an environment to say it in.
///
/// Relative entries are dropped, which the specification asks for, and each
/// variable falls back to the default the specification gives it.
fn search(config_home: Option<String>, home: Option<String>, config_dirs: Option<String>) -> Vec<PathBuf> {
  let absolute = |dir: PathBuf| dir.is_absolute().then_some(dir);
  let first = config_home
    .map(PathBuf::from)
    .and_then(absolute)
    .or_else(|| home.map(|home| PathBuf::from(home).join(".config")));
  let rest = config_dirs.filter(|dirs| !dirs.is_empty()).unwrap_or("/etc/xdg".into());
  first
    .into_iter()
    .chain(rest.split(':').map(PathBuf::from).filter_map(absolute))
    .collect()
}

/// Read every file that is there. A file that cannot be read or makes no sense
/// is a note rather than a failure: the session still has its own five tools,
/// and saying so beats refusing to start.
///
/// `strict` is for a file that was asked for by name, where not being there is
/// worth saying.
#[cfg(feature = "mcp")]
pub fn load(paths: &[PathBuf], strict: bool) -> (Config, Vec<String>) {
  let mut config = Config::default();
  let mut notes = Vec::new();
  for path in paths {
    let text = match std::fs::read_to_string(path) {
      Ok(text) => text,
      Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
        if strict {
          notes.push(format!("MCP: no such file: {}", path.display()));
        }
        continue;
      }
      Err(err) => {
        notes.push(format!("MCP: could not read {}: {err}", path.display()));
        continue;
      }
    };
    match toml::from_str::<Config>(&text) {
      // Later files are nearer the work, so they win the names they share.
      Ok(read) => config.extend(read),
      // Every line and column toml worked out, since a file this small is
      // usually wrong in a way the message alone would not place.
      Err(err) => notes.push(format!("MCP: could not read {}: {err}", path.display())),
    }
  }
  (config, notes)
}

/// Nothing is read without the machinery to act on it, but a file that was
/// meant to do something and cannot is still worth a word.
#[cfg(not(feature = "mcp"))]
#[derive(Default)]
pub struct Config;

#[cfg(not(feature = "mcp"))]
pub fn load(paths: &[PathBuf], strict: bool) -> (Config, Vec<String>) {
  let asked = strict || paths.iter().any(|path| path.exists());
  let notes = match asked {
    true => vec!["MCP: this build has no MCP support (built without the `mcp` feature).".to_string()],
    false => Vec::new(),
  };
  (Config, notes)
}

/// The servers of one session, and what they offer.
///
/// Holding this is what keeps the servers running: a command's child process
/// lives as long as the connection to it, and dropping this closes both.
#[derive(Default)]
pub struct Servers {
  #[cfg(feature = "mcp")]
  running: Vec<rmcp::service::RunningService<rmcp::service::RoleClient, ()>>,
  /// The tools each connection offers, with the connection to call them on
  /// and how long one of its calls may take.
  #[cfg(feature = "mcp")]
  tools: Vec<(Vec<rmcp::model::Tool>, rmcp::service::ServerSink, Option<u64>)>,
  notes: Vec<String>,
}

impl Servers {
  /// What to say in the transcript about the servers this session has: which
  /// came up and with how many tools, and what went wrong with the rest.
  pub fn notes(&self) -> &[String] {
    &self.notes
  }

  /// How many servers came up, and how many tools they brought between them.
  pub fn count(&self) -> (usize, usize) {
    #[cfg(feature = "mcp")]
    return (self.tools.len(), self.tools.iter().map(|(tools, ..)| tools.len()).sum());
    #[cfg(not(feature = "mcp"))]
    (0, 0)
  }

  /// Every tool the session's servers brought, for a list that names tools
  /// without knowing where each came from.
  pub fn tool_names(&self) -> Vec<String> {
    #[cfg(feature = "mcp")]
    return self
      .tools
      .iter()
      .flat_map(|(tools, ..)| tools.iter().map(|tool| tool.name.to_string()))
      .collect();
    #[cfg(not(feature = "mcp"))]
    Vec::new()
  }

  #[cfg(feature = "mcp")]
  fn note(&mut self, note: String) {
    self.notes.push(note);
  }
}

#[cfg(feature = "mcp")]
pub async fn connect(config: Config) -> Servers {
  let mut servers = Servers::default();
  let mut taken: Vec<String> = crate::tools::BUILT_IN.map(str::to_string).to_vec();
  for (name, server) in config {
    let started = tokio::time::timeout(START_TIMEOUT, start(&server));
    let running = match started.await {
      Ok(Ok(running)) => running,
      Ok(Err(err)) => {
        servers.note(format!("MCP {name}: {err:#}"));
        continue;
      }
      Err(_) => {
        servers.note(format!("MCP {name}: no answer in {}s", START_TIMEOUT.as_secs()));
        continue;
      }
    };
    let tools = match running.list_all_tools().await {
      Ok(tools) => tools,
      Err(err) => {
        servers.note(format!("MCP {name}: could not list tools: {err}"));
        continue;
      }
    };
    // What the server offers, narrowed to what was asked of it. A name in
    // neither list is a name nothing answers to — usually a typo, and a typo
    // in a list like this is a tool quietly left in or out.
    let offered: Vec<String> = tools.iter().map(|tool| tool.name.to_string()).collect();
    let (wanted, unknown) = crate::tools::choose(&offered, &server.tools, &server.except);
    if !unknown.is_empty() {
      servers.note(format!("MCP {name}: offers no {}", unknown.join(", ")));
    }
    let tools: Vec<_> = match &wanted {
      Some(wanted) => tools
        .into_iter()
        .filter(|tool| wanted.contains(&tool.name.to_string()))
        .collect(),
      None => tools,
    };
    // A tool cannot be had twice under one name: the model would have no way
    // to say which it meant, and the five the system prompt describes are the
    // ones it was told about.
    let (tools, clashed): (Vec<_>, Vec<_>) = tools
      .into_iter()
      .partition(|tool| !taken.contains(&tool.name.to_string()));
    taken.extend(tools.iter().map(|tool| tool.name.to_string()));
    if !clashed.is_empty() {
      let names: Vec<String> = clashed.iter().map(|tool| tool.name.to_string()).collect();
      servers.note(format!(
        "MCP {name}: not taking {} (name already used)",
        names.join(", ")
      ));
    }
    servers.note(format!(
      "MCP {name}: {}",
      match tools.len() {
        1 => "1 tool".to_string(),
        n => format!("{n} tools"),
      }
    ));
    servers.tools.push((tools, running.peer().clone(), server.timeout));
    servers.running.push(running);
  }
  servers
}

/// Reach one server, however it is reached.
#[cfg(feature = "mcp")]
async fn start(server: &Server) -> anyhow::Result<rmcp::service::RunningService<rmcp::service::RoleClient, ()>> {
  use anyhow::{Context, bail};
  use rmcp::ServiceExt;

  match (&server.command, &server.url) {
    (Some(_), Some(_)) => bail!("declared as both a command and a URL"),
    (Some(command), None) => {
      // A line of shell, run the way the `bash` tool runs one, so a server is
      // started by the command that starts it in a terminal — quoting, `~`,
      // `$HOME` and all.
      let mut process = tokio::process::Command::new("bash");
      process.arg("-c").arg(command).envs(&server.env);
      // A server's own chatter is not this program's to print: the terminal
      // belongs to the transcript.
      process.stderr(std::process::Stdio::null());
      let transport =
        rmcp::transport::TokioChildProcess::new(process).with_context(|| format!("could not run {command}"))?;
      Ok(().serve(transport).await?)
    }
    (None, Some(url)) => {
      let mut config =
        rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(url.as_str());
      config.custom_headers = headers(&server.headers)?;
      let transport = rmcp::transport::StreamableHttpClientTransport::from_config(config);
      Ok(().serve(transport).await?)
    }
    (None, None) => bail!("declared with neither a command to run nor a URL to call"),
  }
}

#[cfg(feature = "mcp")]
fn headers(
  headers: &BTreeMap<String, String>,
) -> anyhow::Result<std::collections::HashMap<http::HeaderName, http::HeaderValue>> {
  use anyhow::Context;

  headers
    .iter()
    .map(|(name, value)| {
      let name: http::HeaderName = name.parse().with_context(|| format!("not a header name: {name}"))?;
      let value: http::HeaderValue = value
        .parse()
        .with_context(|| format!("not a header value for {name}"))?;
      Ok((name, value))
    })
    .collect()
}

/// Hand every tool that came up to the agent being built.
#[cfg(feature = "mcp")]
pub fn attach(server: rig_agent::tool::server::ToolServer, servers: &Servers) -> rig_agent::tool::server::ToolServer {
  servers.tools.iter().fold(server, |server, (tools, sink, timeout)| {
    // Rig bounds a call at five minutes unless told otherwise. A server can
    // say its own number, and zero lets a call take as long as it takes.
    let timeout = match timeout {
      Some(0) => None,
      Some(seconds) => Some(std::time::Duration::from_secs(*seconds)),
      None => Some(rig_agent::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT),
    };
    server.rmcp_tools_with_timeout(tools.clone(), sink.clone(), timeout)
  })
}

#[cfg(not(feature = "mcp"))]
pub async fn connect(_config: Config) -> Servers {
  Servers::default()
}

#[cfg(not(feature = "mcp"))]
pub fn attach(server: rig_agent::tool::server::ToolServer, _servers: &Servers) -> rig_agent::tool::server::ToolServer {
  server
}

#[cfg(test)]
mod tests {
  use super::*;

  #[cfg(feature = "mcp")]
  fn config(text: &str) -> Config {
    toml::from_str(text).expect("a readable config")
  }

  #[cfg(feature = "mcp")]
  #[test]
  fn a_server_is_a_command_or_a_url_and_a_key_that_is_neither_is_a_mistake() {
    let read = config(
      r#"
        # A line of shell, as the terminal would take it.
        [files]
        command = "mcp-files --root ."
        env.TOKEN = "x"

        [docs]
        url = "https://example.com/mcp"
        headers.Authorization = "Bearer k"
        timeout = 60
      "#,
    );
    assert_eq!(read.len(), 2);
    let files = &read["files"];
    assert_eq!(files.command.as_deref(), Some("mcp-files --root ."));
    assert_eq!(files.env["TOKEN"], "x");
    assert_eq!(files.timeout, None, "a server says nothing about time by default");
    let docs = &read["docs"];
    assert_eq!(docs.url.as_deref(), Some("https://example.com/mcp"));
    assert_eq!(docs.headers["Authorization"], "Bearer k");
    assert_eq!(docs.timeout, Some(60));
    // An empty file is a file with no servers in it, not an error.
    assert_eq!(config(""), Config::default());

    // This file is fa's own, so a key it does not know is a typo to be told
    // about rather than another client's business to be read past.
    let err = toml::from_str::<Config>("[files]\ncomand = \"oops\"")
      .expect_err("an unknown key is refused")
      .to_string();
    assert!(err.contains("comand"), "which key it was: {err}");
  }

  #[cfg(feature = "mcp")]
  #[test]
  fn the_nearest_file_has_the_last_word_and_a_bad_one_is_only_a_note() {
    let dir = std::env::temp_dir().join(format!("fa-mcp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory");
    let system = dir.join("system.toml");
    let user = dir.join("user.toml");
    std::fs::write(
      &system,
      "[shared]\ncommand = \"system\"\n\n[only-system]\ncommand = \"s\"",
    )
    .expect("a file");
    std::fs::write(&user, "[shared]\ncommand = \"user\"").expect("a file");

    let (config, notes) = load(&[system.clone(), user.clone()], true);
    assert_eq!(
      config["shared"].command.as_deref(),
      Some("user"),
      "read last, so it wins"
    );
    assert!(config.contains_key("only-system"), "the rest of the other file stays");
    assert!(notes.is_empty(), "{notes:?}");

    // A file that is not there is only worth saying when it was asked for.
    let missing = dir.join("nowhere.toml");
    assert!(load(std::slice::from_ref(&missing), false).1.is_empty());
    assert_eq!(load(&[missing], true).1.len(), 1);

    // One that is there but makes no sense leaves the others alone, and says
    // where in it the trouble is.
    let broken = dir.join("broken.toml");
    std::fs::write(&broken, "[files]\ncommand = oh no").expect("a file");
    let (config, notes) = load(&[user, broken], false);
    assert!(config.contains_key("shared"), "the good file is still read");
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(notes[0].contains("line 2"), "where it went wrong: {notes:?}");
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn the_search_path_is_the_xdg_one_and_only_that() {
    let path = |config_home: Option<&str>, home: Option<&str>, dirs: Option<&str>| {
      search(
        config_home.map(str::to_string),
        home.map(str::to_string),
        dirs.map(str::to_string),
      )
    };
    // Nearest first here; a relative entry is no entry at all.
    assert_eq!(
      path(
        Some("/home/someone/.config"),
        None,
        Some("/etc/xdg:relative/ignored:/opt/xdg")
      ),
      [
        PathBuf::from("/home/someone/.config"),
        PathBuf::from("/etc/xdg"),
        PathBuf::from("/opt/xdg"),
      ]
    );
    // Each variable falls back to what the specification says it means.
    assert_eq!(
      path(None, Some("/home/someone"), None),
      [PathBuf::from("/home/someone/.config"), PathBuf::from("/etc/xdg")]
    );
    assert_eq!(
      path(Some("relative"), Some("/home/someone"), None)[0],
      PathBuf::from("/home/someone/.config")
    );
    // Nowhere to look is not somewhere to look.
    assert_eq!(path(None, None, Some("")), [PathBuf::from("/etc/xdg")]);
  }

  #[test]
  fn the_files_are_read_system_first_and_never_from_the_working_directory() {
    let found = files(None);
    assert!(
      found.iter().all(|path| path.ends_with(Path::new("fa").join(FILE))),
      "only ever that one name under the search path: {found:?}"
    );
    assert!(
      !found.iter().any(|path| path.is_relative()),
      "nothing beside the project: {found:?}"
    );
    // The nearest directory is read last, so its names win.
    let dirs = config_dirs();
    assert_eq!(found.len(), dirs.len(), "one file per directory: {found:?} {dirs:?}");
    if let (Some(nearest), Some(last)) = (dirs.first(), found.last()) {
      assert_eq!(last, &nearest.join("fa").join(FILE));
    }
    // Asked for by name, that is the only one.
    assert_eq!(
      files(Some(Path::new("/tmp/one.json"))),
      [PathBuf::from("/tmp/one.json")]
    );
  }
}
