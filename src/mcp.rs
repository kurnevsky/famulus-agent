//! Tools from MCP servers.
//!
//! `rmcp` speaks the protocol. What is here is the finding and the starting:
//! which servers a session should have, how to reach each one, and handing
//! what they offer to the agent as tools like any other. A server is either a
//! program that talks over its own stdin and stdout, or an endpoint that
//! speaks streamable HTTP. A server that asks the user something while one of
//! its tools runs is answered through `elicit`; what it says of how far along
//! the tool is, is shown through `call`.
//!
//! The file is this program's own, so it reads the way the rest of it does: a
//! table per server, named by the name its tools will be called under, and a
//! `command` written as a line of shell, like the one `bash` takes. Where it
//! lives is the XDG search path and nothing else — no dotfile in a home
//! directory, nothing beside the project.
//!
//! A value that should not sit in a file is written as the line of shell that
//! produces it, under `env-command`, `token-command` or `headers-command`
//! rather than `env`, `token` or `headers`: run once, when the server starts,
//! with its output for the value. A token can also be named by where it sits
//! in the system keyring, under `token-keyring`.
//!
//! An endpoint that wants a login and is given no token gets one through
//! `oauth`, in the browser, and keeps it in the system keyring.
//!
//! ```toml
//! [fetch]
//! command = "uvx mcp-server-fetch"
//!
//! [docs]
//! url = "https://example.com/mcp"
//! token-keyring = { service = "work", account = "mcp" }
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
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Server {
  /// A line of shell run for a server that speaks over its own stdin and
  /// stdout — the same shape `bash` takes, and run the same way, so a command
  /// that works in the terminal works here.
  pub command: Option<String>,
  /// Added to the environment that command inherits.
  pub env: BTreeMap<String, String>,
  /// The same, for a value that should not sit in a file: a line of shell
  /// whose output is what the variable is set to.
  pub env_command: BTreeMap<String, String>,
  /// The endpoint of a server that speaks streamable HTTP.
  pub url: Option<String>,
  /// The bearer token that endpoint is called with, which rmcp sends as the
  /// `Authorization` of every request. A server given one is not signed in
  /// to any other way.
  pub token: Option<String>,
  /// The same, for a token that should not sit in a file: a line of shell
  /// whose output is the token — `pass`, `gh auth token`, `op read`.
  pub token_command: Option<String>,
  /// The same, kept in the system keyring: the attributes of the one item
  /// whose secret is the token.
  pub token_keyring: BTreeMap<String, String>,
  /// Sent with every request to that endpoint.
  pub headers: BTreeMap<String, String>,
  /// The same, each value the output of a line of shell.
  pub headers_command: BTreeMap<String, String>,
  /// Seconds one of this server's tools — or a listing or reading of its
  /// resources, or of its prompts — may take before the call comes back as
  /// an error the model can recover from. `0` waits forever. For a tool, the seconds are
  /// counted from the last word from the server, so one that keeps saying
  /// how far along it is may take as long as it needs.
  pub timeout: Option<u64>,
  /// Take only these of the tools it offers. Everything it offers, when empty
  /// — which is not the same as naming them all, since a server that grows a
  /// tool later would keep it.
  pub tools: Vec<String>,
  /// Take everything but these. Named in both lists, a tool is refused.
  pub except: Vec<String>,
  /// How to sign in to an endpoint that wants a login, where the server
  /// cannot work that out for itself.
  pub oauth: crate::oauth::Settings,
}

/// Where servers are declared, in the order the files are read: the system's
/// first, the user's last, so the nearer file wins the names they share.
pub fn files(explicit: Option<&Path>) -> Vec<PathBuf> {
  crate::config::files(FILE, explicit)
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
  running: Vec<Service>,
  catalog: Catalog,
  notes: Vec<String>,
}

/// A connection to one server, answering what it asks through `elicit`.
#[cfg(feature = "mcp")]
type Service = rmcp::service::RunningService<rmcp::service::RoleClient, Watch>;

impl Servers {
  /// What to say in the transcript about the servers this session has: which
  /// came up and with how many tools, and what went wrong with the rest.
  pub fn notes(&self) -> &[String] {
    &self.notes
  }

  /// What the servers offer, now and as it changes.
  pub fn catalog(&self) -> &Catalog {
    &self.catalog
  }
}

// ---------------------------------------------------------------- the catalog

/// What every server of the session offers at the moment, in the order the
/// file names them — which is the order a name two of them offer goes to.
///
/// A server can say its tools have changed while the session runs, and this
/// is where the new ones are put. Whoever built the agent's tools from it is
/// told, and builds them again.
#[derive(Clone, Default)]
pub struct Catalog(#[cfg_attr(not(feature = "mcp"), allow(dead_code))] std::sync::Arc<Shelves>);

/// Building the agent's tools again, from what is on offer now.
#[cfg(feature = "mcp")]
type Rebuild = Box<dyn Fn(&Catalog) + Send + Sync>;

#[derive(Default)]
struct Shelves {
  #[cfg(feature = "mcp")]
  offers: std::sync::Mutex<Vec<Offer>>,
  /// What to do when an offer changes, once there are agents to tell.
  #[cfg(feature = "mcp")]
  changed: std::sync::OnceLock<Rebuild>,
}

/// What one server offers, and what calling it takes.
#[cfg(feature = "mcp")]
struct Offer {
  server: String,
  /// What it offers, narrowed to what was asked of it.
  tools: Vec<rmcp::model::Tool>,
  peer: rmcp::service::ServerSink,
  /// How long one of its calls may take.
  timeout: Option<u64>,
  /// What its calls hear of how far along they are.
  progress: crate::call::Progress,
  /// What it has to read, for a server that said it has resources as it came
  /// up — whether or not it could list them then.
  resources: Option<crate::resources::ServerResources>,
  /// What it has written out for the user to send, as it last said —
  /// `None` until it has said.
  prompts: Option<Vec<rmcp::model::Prompt>>,
}

#[cfg(feature = "mcp")]
impl Offer {
  /// What `named` — `server:rest` — names on this server: the rest, when
  /// the server is this one.
  fn under<'a>(&self, named: &'a str) -> Option<&'a str> {
    named.strip_prefix(self.server.as_str())?.strip_prefix(':')
  }
}

/// How long a call to a server may take, from the seconds its table says:
/// five minutes, as rig has it, unless told otherwise, and zero lets a call
/// take as long as it takes.
#[cfg(feature = "mcp")]
fn call_timeout(seconds: Option<u64>) -> Option<std::time::Duration> {
  match seconds {
    Some(0) => None,
    Some(seconds) => Some(std::time::Duration::from_secs(seconds)),
    None => Some(rig_agent::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT),
  }
}

impl Catalog {
  /// How many servers came up, and how many tools they offer between them.
  pub fn count(&self) -> (usize, usize) {
    #[cfg(feature = "mcp")]
    {
      let offers = self.offers();
      (offers.len(), offers.iter().map(|offer| offer.tools.len()).sum())
    }
    #[cfg(not(feature = "mcp"))]
    (0, 0)
  }

  /// Every tool the servers offer, and the two that read what they hold
  /// when one of them holds anything, for a list that names tools without
  /// knowing where each came from.
  pub fn tool_names(&self) -> Vec<String> {
    #[cfg(feature = "mcp")]
    {
      let mut names: Vec<String> = self
        .offers()
        .iter()
        .flat_map(|offer| offer.tools.iter().map(|tool| tool.name.to_string()))
        .collect();
      if self.has_resources() {
        names.extend(crate::resources::NAMES.map(str::to_string));
      }
      names
    }
    #[cfg(not(feature = "mcp"))]
    Vec::new()
  }

  /// Every `server:uri` the servers list, resources and then templates,
  /// each with what the popup says beside it: a resource's type, or that it
  /// is a template.
  pub fn completions(&self) -> Vec<(String, String)> {
    #[cfg(feature = "mcp")]
    return self
      .resources()
      .iter()
      .flat_map(|(server, held)| {
        let resources = held
          .resources
          .iter()
          .map(move |r| (format!("{server}:{}", r.uri), r.mime_type.clone().unwrap_or_default()));
        let templates = held
          .templates
          .iter()
          .map(move |t| (format!("{server}:{}", t.uri_template), "template".to_string()));
        resources.chain(templates)
      })
      .collect();
    #[cfg(not(feature = "mcp"))]
    Vec::new()
  }

  /// What each server with resources has to read, as it last said.
  #[cfg(feature = "mcp")]
  pub fn resources(&self) -> BTreeMap<String, crate::resources::ServerResources> {
    self
      .offers()
      .iter()
      .filter_map(|offer| Some((offer.server.clone(), offer.resources.clone()?)))
      .collect()
  }

  /// Whether any server has resources to read, which is when the tools
  /// that read them are offered.
  #[cfg(feature = "mcp")]
  pub fn has_resources(&self) -> bool {
    self.offers().iter().any(|offer| offer.resources.is_some())
  }

  /// The server called `named`, if it is one with resources, and how long
  /// a request to it may take.
  #[cfg(feature = "mcp")]
  pub fn resource_peer(&self, named: &str) -> Option<(rmcp::service::ServerSink, Option<std::time::Duration>)> {
    self
      .offers()
      .iter()
      .find(|offer| offer.server == named && offer.resources.is_some())
      .map(|offer| (offer.peer.clone(), call_timeout(offer.timeout)))
  }

  /// Every prompt the servers offer, as the `/` popup offers it: its
  /// `server:name`, what it takes and does, and whether it takes anything.
  pub fn prompt_commands(&self) -> Vec<(String, String, bool)> {
    #[cfg(feature = "mcp")]
    return self
      .offers()
      .iter()
      .flat_map(|offer| {
        offer.prompts.iter().flatten().map(|prompt| {
          let takes = prompt.arguments.as_ref().is_some_and(|a| !a.is_empty());
          (
            format!("{}:{}", offer.server, prompt.name),
            crate::prompts::described(prompt),
            takes,
          )
        })
      })
      .collect();
    #[cfg(not(feature = "mcp"))]
    Vec::new()
  }

  /// What sending `text` comes to, when it is `/server:name` and a server
  /// offers that prompt: the messages the server writes out from the rest of
  /// the line, asking the user through `host` for what it leaves out. `None`
  /// for anything else, which is sent as it was typed.
  pub fn expansion(&self, text: &str, host: &crate::modal::Host, vision: bool) -> Option<Expansion> {
    #[cfg(feature = "mcp")]
    {
      let typed = text.strip_prefix('/')?;
      let (named, rest) = typed.split_once(char::is_whitespace).unwrap_or((typed, ""));
      let asking = self.prompt(named)?;
      let rest = rest.to_string();
      let host = host.clone();
      Some(Box::pin(async move { asking.expand(&rest, &host, vision).await }))
    }
    #[cfg(not(feature = "mcp"))]
    {
      let _ = (text, host, vision);
      None
    }
  }

  /// The prompt `named`, as `server:name`, and what asking it takes.
  #[cfg(feature = "mcp")]
  fn prompt(&self, named: &str) -> Option<crate::prompts::Asking> {
    self.offers().iter().find_map(|offer| {
      let name = offer.under(named)?;
      let prompt = offer.prompts.iter().flatten().find(|prompt| prompt.name == name)?;
      Some(crate::prompts::Asking {
        server: offer.server.clone(),
        prompt: prompt.clone(),
        peer: offer.peer.clone(),
        timeout: call_timeout(offer.timeout),
      })
    })
  }

  /// The argument of a server's prompt being typed in `text` — a line that
  /// starts `/server:name` — at `cursor`, when that server completes them.
  pub fn argument_completion(&self, text: &str, cursor: usize) -> Option<Completion> {
    #[cfg(feature = "mcp")]
    {
      let typed = text.strip_prefix('/')?;
      let named = typed.split(char::is_whitespace).next().unwrap_or(typed);
      let asking = self.prompt(named).filter(|asking| completes(&asking.peer))?;
      // At the end of what is being typed, as a token's popup is.
      let start = 1 + named.len();
      if cursor <= start || !text[cursor..].chars().next().is_none_or(char::is_whitespace) {
        return None;
      }
      let declared = asking.prompt.arguments.as_deref().unwrap_or_default();
      let typing = crate::prompts::typing(declared, &text[start..cursor])?;
      Some(Completion {
        asking: Completing {
          server: asking.server,
          of: Of::Prompt(asking.prompt.name),
          argument: typing.name,
          value: typing.value,
          context: typing.earlier,
        },
        range: (start + typing.range.0, start + typing.range.1),
        shape: Shape::Argument { last: typing.last },
      })
    }
    #[cfg(not(feature = "mcp"))]
    {
      let _ = (text, cursor);
      None
    }
  }

  /// The hole of a server's template being filled in by an `&` token, at
  /// `range` in the input and saying `typed` — `server:uri`, sigil and
  /// quotes off — when that server completes them.
  pub fn template_completion(&self, range: (usize, usize), typed: &str) -> Option<Completion> {
    #[cfg(feature = "mcp")]
    return self.offers().iter().find_map(|offer| {
      let uri = offer.under(typed)?;
      if !completes(&offer.peer) {
        return None;
      }
      offer.resources.as_ref()?.templates.iter().find_map(|template| {
        let filling = crate::resources::filling(&template.uri_template, uri)?;
        Some(Completion {
          asking: Completing {
            server: offer.server.clone(),
            of: Of::Template(template.uri_template.clone()),
            argument: filling.name,
            value: filling.value,
            context: filling.earlier,
          },
          range,
          shape: Shape::Hole {
            before: format!("{}:{}", offer.server, &uri[..filling.start]),
            after: filling.after,
            more: filling.more,
          },
        })
      })
    });
    #[cfg(not(feature = "mcp"))]
    {
      let _ = (range, typed);
      None
    }
  }

  /// Where taking the template `named` — `server:template` — leaves the
  /// token when its server completes it: at its first hole, for the server
  /// to be asked what goes there. `None` takes the template as it is.
  pub fn template_opening(&self, named: &str) -> Option<String> {
    #[cfg(feature = "mcp")]
    return self.offers().iter().find_map(|offer| {
      let template = offer.under(named)?;
      let listed = &offer.resources.as_ref()?.templates;
      if !completes(&offer.peer) || !listed.iter().any(|t| t.uri_template == template) {
        return None;
      }
      let opening = &template[..template.find('{')?];
      crate::resources::filling(template, opening)?;
      Some(format!("{}:{opening}", offer.server))
    });
    #[cfg(not(feature = "mcp"))]
    {
      let _ = named;
      None
    }
  }

  /// Ask the server what `asking` could be.
  pub fn suggest(&self, asking: &Completing) -> Option<Suggesting> {
    #[cfg(feature = "mcp")]
    {
      let (peer, timeout) = self
        .offers()
        .iter()
        .find(|offer| offer.server == asking.server && completes(&offer.peer))
        .map(|offer| (offer.peer.clone(), call_timeout(offer.timeout)))?;
      let asking = asking.clone();
      Some(Box::pin(async move {
        let context = (!asking.context.is_empty())
          .then(|| rmcp::model::CompletionContext::with_arguments(asking.context.into_iter().collect()));
        let asked = async {
          match asking.of {
            Of::Prompt(name) => {
              peer
                .complete_prompt_argument(name, asking.argument, asking.value, context)
                .await
            }
            Of::Template(template) => {
              peer
                .complete_resource_argument(template, asking.argument, asking.value, context)
                .await
            }
          }
        };
        let offered = crate::resources::bounded(timeout, asked)
          .await
          .map_err(|err| err.to_string())?;
        Ok(Suggestions {
          values: offered.values,
          more: offered.has_more == Some(true),
          total: offered.total,
        })
      }))
    }
    #[cfg(not(feature = "mcp"))]
    {
      let _ = asking;
      None
    }
  }

  /// Hand every tool on offer to the tools being built.
  pub fn attach(&self, server: rig_agent::tool::server::ToolServer) -> rig_agent::tool::server::ToolServer {
    #[cfg(feature = "mcp")]
    return self.offers().iter().fold(server, |server, offer| {
      server.dynamic_tools(
        offer
          .tools
          .iter()
          .map(|tool| {
            crate::call::tool(
              tool,
              offer.peer.clone(),
              call_timeout(offer.timeout),
              offer.progress.clone(),
            )
          })
          .collect(),
      )
    });
    #[cfg(not(feature = "mcp"))]
    server
  }

  /// Have `rebuild` run whenever what is on offer changes. Only the first
  /// caller is heard: there is one set of agents to build tools for.
  pub fn on_change(&self, rebuild: impl Fn(&Catalog) + Send + Sync + 'static) {
    #[cfg(feature = "mcp")]
    let _ = self.0.changed.set(Box::new(rebuild));
    #[cfg(not(feature = "mcp"))]
    drop(rebuild);
  }

  #[cfg(feature = "mcp")]
  fn offers(&self) -> std::sync::MutexGuard<'_, Vec<Offer>> {
    // Nothing holding it panics; a poisoned lock is still the list it was.
    self.0.offers.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
  }

  /// The names a server may not take: the built-in tools', the resource
  /// tools' — kept whether or not any server has resources, so a name does
  /// not change hands with the order servers come up in — and every other
  /// server's.
  #[cfg(feature = "mcp")]
  fn taken(&self, besides: &str) -> Vec<String> {
    let offers = self.offers();
    let others = offers
      .iter()
      .filter(|offer| offer.server != besides)
      .flat_map(|offer| offer.tools.iter().map(|tool| tool.name.to_string()));
    crate::tools::BUILT_IN
      .into_iter()
      .chain(crate::resources::NAMES)
      .map(str::to_string)
      .chain(others)
      .collect()
  }

  /// The tools `server` offers now, in place of what it offered before, and
  /// the names of those it offered before — `None` for a server not on offer
  /// until now.
  #[cfg(feature = "mcp")]
  fn offer(
    &self,
    server: &str,
    tools: Vec<rmcp::model::Tool>,
    peer: rmcp::service::ServerSink,
    timeout: Option<u64>,
    progress: crate::call::Progress,
  ) -> Option<Vec<String>> {
    let before = {
      let mut offers = self.offers();
      match offers.iter_mut().find(|offer| offer.server == server) {
        Some(old) => {
          old.peer = peer;
          old.timeout = timeout;
          old.progress = progress;
          Some(std::mem::replace(&mut old.tools, tools))
        }
        None => {
          offers.push(Offer {
            server: server.to_string(),
            tools,
            peer,
            timeout,
            progress,
            resources: None,
            prompts: None,
          });
          None
        }
      }
    };
    // Outside the lock, since building tools reads what is on offer.
    if let Some(rebuild) = self.0.changed.get() {
      rebuild(self);
    }
    before.map(|tools| tools.iter().map(|tool| tool.name.to_string()).collect())
  }

  /// Put the prompts `server` offers in place of the ones it had, and the
  /// names of those it had — `None` the first time it says.
  #[cfg(feature = "mcp")]
  fn shelve(&self, server: &str, prompts: Vec<rmcp::model::Prompt>) -> Option<Vec<String>> {
    let mut offers = self.offers();
    let offer = offers.iter_mut().find(|offer| offer.server == server)?;
    let before = offer.prompts.replace(prompts)?;
    Some(before.into_iter().map(|prompt| prompt.name).collect())
  }

  /// Put what `server` has to read in place of what it had. A list that
  /// could not be had leaves what was there, or nothing, since the server can
  /// still be read from — marked with why, so it is not taken for all there is.
  #[cfg(feature = "mcp")]
  fn stock(&self, server: &str, listed: Result<crate::resources::ServerResources, String>) {
    if let Some(offer) = self.offers().iter_mut().find(|offer| offer.server == server) {
      match listed {
        Ok(held) => offer.resources = Some(held),
        Err(err) => offer.resources.get_or_insert_default().failed = Some(err),
      }
    }
  }
}

/// What a server is asked to complete: an argument of one of its prompts, or
/// a hole in one of its resource templates — its name, what is in it so far,
/// and what the ones before it were given.
#[derive(Clone, Debug, PartialEq)]
pub struct Completing {
  pub server: String,
  pub of: Of,
  pub argument: String,
  pub value: String,
  pub context: Vec<(String, String)>,
}

/// What is being completed: a prompt, by name, or a template.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(not(feature = "mcp"), allow(dead_code))]
pub enum Of {
  Prompt(String),
  Template(String),
}

/// A value being typed that its server can complete: what to ask it, the
/// bytes of the input a value taken goes in place of, and how it is written.
pub struct Completion {
  pub asking: Completing,
  pub range: (usize, usize),
  #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
  shape: Shape,
}

#[cfg_attr(not(feature = "mcp"), allow(dead_code))]
enum Shape {
  /// A prompt's argument: the last one takes the rest of the line, and one
  /// before it is a word, quoted when it has a space in it.
  Argument { last: bool },
  /// A hole in a template, written into the `&` token around it.
  Hole {
    /// The token's `server:uri` up to the hole.
    before: String,
    /// The template's text after it, up to the next hole.
    after: String,
    /// Whether there is a next hole.
    more: bool,
  },
}

impl Completion {
  /// What taking `value` puts in place of `range`, and whether there is
  /// something after it to complete next: the next argument, or the next
  /// hole.
  pub fn insert(&self, value: &str) -> (String, bool) {
    match &self.shape {
      Shape::Argument { last: true } => (value.to_string(), false),
      Shape::Argument { last: false } => match value.contains(char::is_whitespace) {
        true => (format!("\"{value}\" "), true),
        false => (format!("{value} "), true),
      },
      Shape::Hole { before, after, more } => (crate::attach::written('&', &format!("{before}{value}{after}")), *more),
    }
  }
}

/// What a server offers for a value: at most a hundred of them, and whether
/// it has more, and how many in all when it says.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Suggestions {
  pub values: Vec<String>,
  pub more: bool,
  pub total: Option<u32>,
}

/// Asking a server what a value could be.
pub type Suggesting = futures::future::BoxFuture<'static, Result<Suggestions, String>>;

/// What sending an MCP prompt comes to: the messages the server wrote out,
/// or `None` when the user put away the form that asked for its arguments.
pub type Expansion = futures::future::BoxFuture<'static, Result<Option<Vec<rig_core::completion::Message>>, String>>;

/// What a server offers, narrowed to what its table asks of it and to the
/// names nothing else has taken.
#[cfg(feature = "mcp")]
struct Picked {
  tools: Vec<rmcp::model::Tool>,
  /// Named in `tools` or `except` and offered by no tool.
  unknown: Vec<String>,
  /// Offered, but under a name already in use.
  clashed: Vec<String>,
}

#[cfg(feature = "mcp")]
fn pick(server: &Server, tools: Vec<rmcp::model::Tool>, taken: &[String]) -> Picked {
  // A name in neither list is a name nothing answers to — usually a typo,
  // and a typo in a list like this is a tool quietly left in or out.
  let offered: Vec<String> = tools.iter().map(|tool| tool.name.to_string()).collect();
  let (wanted, unknown) = crate::tools::choose(&offered, &server.tools, &server.except);
  let tools: Vec<_> = match &wanted {
    Some(wanted) => tools
      .into_iter()
      .filter(|tool| wanted.iter().any(|name| *name == tool.name))
      .collect(),
    None => tools,
  };
  // A tool cannot be had twice under one name: the model would have no way
  // to say which it meant, and the five the system prompt describes are the
  // ones it was told about.
  let (tools, clashed): (Vec<_>, Vec<_>) = tools
    .into_iter()
    .partition(|tool| !taken.iter().any(|name| *name == tool.name));
  Picked {
    tools,
    unknown,
    clashed: clashed.iter().map(|tool| tool.name.to_string()).collect(),
  }
}

/// "1 tool", "3 prompts".
#[cfg(feature = "mcp")]
fn counted(n: usize, what: &str) -> String {
  match n {
    1 => format!("1 {what}"),
    n => format!("{n} {what}s"),
  }
}

/// Which of `now` are new since `before`, and which of `before` are gone,
/// as the end of a sentence saying how many there are now.
#[cfg(feature = "mcp")]
fn changes(before: &[String], now: &[String]) -> String {
  let gained: Vec<&str> = now.iter().filter(|n| !before.contains(n)).map(String::as_str).collect();
  let lost: Vec<&str> = before.iter().filter(|n| !now.contains(n)).map(String::as_str).collect();
  let mut said = String::new();
  if !gained.is_empty() {
    said.push_str(&format!(", new: {}", gained.join(", ")));
  }
  if !lost.is_empty() {
    said.push_str(&format!(", gone: {}", lost.join(", ")));
  }
  said
}

/// The client side of one server's connection: what it answers the server
/// with, through `elicit`, and what it needs to put what the server offers on
/// the shelf again, when it says that has changed.
#[cfg(feature = "mcp")]
#[derive(Clone)]
pub struct Watch {
  /// The server's name in the file, which is who a form says is asking.
  pub(crate) server: String,
  /// Its table, for the `tools` and `except` it narrows what it offers by.
  table: std::sync::Arc<Server>,
  catalog: Catalog,
  pub(crate) host: crate::modal::Host,
  /// One refresh at a time: two lists fetched at once could land in either
  /// order, and the older one would win.
  busy: std::sync::Arc<tokio::sync::Mutex<()>>,
  /// What the server says of how far along its calls are, for the calls to
  /// hear. Its own, since a token is only one server's.
  pub(crate) progress: crate::call::Progress,
}

/// Whether a server said, when it came up, that it has resources.
#[cfg(feature = "mcp")]
fn has_resources(peer: &rmcp::service::ServerSink) -> bool {
  peer
    .peer_info()
    .is_some_and(|info| info.capabilities.resources.is_some())
}

/// Whether a server said, when it came up, that it completes values.
#[cfg(feature = "mcp")]
fn completes(peer: &rmcp::service::ServerSink) -> bool {
  peer
    .peer_info()
    .is_some_and(|info| info.capabilities.completions.is_some())
}

/// Whether a server said, when it came up, that it has prompts.
#[cfg(feature = "mcp")]
fn has_prompts(peer: &rmcp::service::ServerSink) -> bool {
  peer.peer_info().is_some_and(|info| info.capabilities.prompts.is_some())
}

#[cfg(feature = "mcp")]
impl Watch {
  /// The server says what it has to read has changed: ask it again. Said
  /// only when it cannot be asked, since a server of files changes its list
  /// whenever a file does, and none of that is news.
  pub async fn resources_changed(&self, peer: &rmcp::service::ServerSink) {
    if let Some(note) = self.stock_resources(peer).await {
      self.host.tell(crate::agent::AgentEvent::Mcp(note));
    }
  }

  /// The server says its prompts have changed: ask it for them, and say
  /// which came and went — they are commands the user types, and one that
  /// appears without a word is one nobody knows to type.
  pub async fn prompts_changed(&self, peer: &rmcp::service::ServerSink) {
    if let Some(note) = self.stock_prompts(peer).await {
      self.host.tell(crate::agent::AgentEvent::Mcp(note));
    }
  }

  /// The server says its tools have changed: ask it for them, and put them
  /// in place of the ones it had.
  pub async fn changed(&self, peer: &rmcp::service::ServerSink) {
    let note = self.stock_tools(peer).await.unwrap_or_else(|err| {
      format!(
        "MCP {}: said its tools changed, but could not list them: {err}",
        self.server
      )
    });
    self.host.tell(crate::agent::AgentEvent::Mcp(note));
  }

  /// Ask the server for its tools and put them on the shelf, and what to say
  /// about it: how many there are the first time, and what changed after.
  async fn stock_tools(&self, peer: &rmcp::service::ServerSink) -> Result<String, rmcp::ServiceError> {
    let _one = self.busy.lock().await;
    let name = &self.server;
    let tools = peer.list_all_tools().await?;
    let picked = pick(&self.table, tools, &self.catalog.taken(name));
    let now: Vec<String> = picked.tools.iter().map(|tool| tool.name.to_string()).collect();
    let mut note = format!("MCP {name}: ");
    let offered = self.catalog.offer(
      name,
      picked.tools,
      peer.clone(),
      self.table.timeout,
      self.progress.clone(),
    );
    match offered {
      None => {
        note.push_str(&counted(now.len(), "tool"));
        // Said once, as it comes up: a typo in the file is not news again
        // every time the server's list changes.
        if !picked.unknown.is_empty() {
          note.push_str(&format!("; offers no {}", picked.unknown.join(", ")));
        }
      }
      Some(before) => {
        note.push_str(&format!("now {}", counted(now.len(), "tool")));
        note.push_str(&changes(&before, &now));
      }
    }
    if !picked.clashed.is_empty() {
      note.push_str(&format!(
        "; not taking {} (name already used)",
        picked.clashed.join(", ")
      ));
    }
    Ok(note)
  }

  /// Ask a server that said it has prompts what they are, for the `/` popup
  /// to offer, and what to say about it: how many there are the first time,
  /// what changed after, and why it could not say, if it could not.
  async fn stock_prompts(&self, peer: &rmcp::service::ServerSink) -> Option<String> {
    if !has_prompts(peer) {
      return None;
    }
    let _one = self.busy.lock().await;
    let name = &self.server;
    let prompts = match crate::prompts::inventory(peer, call_timeout(self.table.timeout)).await {
      Ok(prompts) => prompts,
      Err(err) => return Some(format!("MCP {name}: could not list prompts: {err}")),
    };
    let now: Vec<String> = prompts.iter().map(|prompt| prompt.name.clone()).collect();
    match self.catalog.shelve(name, prompts) {
      None if now.is_empty() => None,
      None => Some(format!("MCP {name}: {}", counted(now.len(), "prompt"))),
      Some(before) => Some(format!(
        "MCP {name}: now {}{}",
        counted(now.len(), "prompt"),
        changes(&before, &now)
      )),
    }
  }

  /// Ask a server that said it has resources what they are, for `&` to
  /// complete from, and why it could not say, if it could not.
  async fn stock_resources(&self, peer: &rmcp::service::ServerSink) -> Option<String> {
    if !has_resources(peer) {
      return None;
    }
    let _one = self.busy.lock().await;
    let listed = crate::resources::inventory(peer, call_timeout(self.table.timeout)).await;
    let listed = listed.map_err(|err| err.to_string());
    let note = listed
      .as_ref()
      .err()
      .map(|err| format!("MCP {}: could not list resources: {err}", self.server));
    self.catalog.stock(&self.server, listed);
    note
  }
}

#[cfg(feature = "mcp")]
pub async fn connect(config: Config, host: &crate::modal::Host) -> Servers {
  let mut servers = Servers::default();
  for (name, server) in config {
    let watch = Watch {
      server: name.clone(),
      table: std::sync::Arc::new(server),
      catalog: servers.catalog.clone(),
      host: host.clone(),
      busy: Default::default(),
      progress: Default::default(),
    };
    let running = match start(&name, &watch.table, watch.clone(), &mut servers.notes).await {
      Ok(running) => running,
      Err(err) => {
        servers.notes.push(format!("MCP {name}: {err:#}"));
        continue;
      }
    };
    match watch.stock_tools(running.peer()).await {
      Ok(note) => servers.notes.push(note),
      Err(err) => {
        servers.notes.push(format!("MCP {name}: could not list tools: {err}"));
        continue;
      }
    }
    servers.notes.extend(watch.stock_resources(running.peer()).await);
    servers.notes.extend(watch.stock_prompts(running.peer()).await);
    servers.running.push(running);
  }
  servers
}

/// Reach one server, however it is reached. What it is worth saying about
/// the way there goes in `notes`.
#[cfg(feature = "mcp")]
async fn start(name: &str, server: &Server, client: Watch, notes: &mut Vec<String>) -> anyhow::Result<Service> {
  use anyhow::{Context, bail};
  use rmcp::ServiceExt;

  match (&server.command, &server.url) {
    (Some(_), Some(_)) => bail!("declared as both a command and a URL"),
    (Some(command), None) => {
      within(async {
        let env = values(&server.env, &server.env_command, "env").await?;
        // A line of shell, run the way the `bash` tool runs one, so a server is
        // started by the command that starts it in a terminal — quoting, `~`,
        // `$HOME` and all.
        let mut process = tokio::process::Command::new("bash");
        process.arg("-c").arg(command).envs(&env);
        // A server's own chatter is not this program's to print: the terminal
        // belongs to the transcript.
        process.stderr(std::process::Stdio::null());
        let transport =
          rmcp::transport::TokioChildProcess::new(process).with_context(|| format!("could not run {command}"))?;
        Ok(client.serve(transport).await?)
      })
      .await
    }
    (None, Some(url)) => endpoint(name, server, url, client, notes).await,
    (None, None) => bail!("declared with neither a command to run nor a URL to call"),
  }
}

/// Reach a server that speaks streamable HTTP: as the file says to, and
/// signed in, if that is what it turns out to want.
#[cfg(feature = "mcp")]
async fn endpoint(
  name: &str,
  server: &Server,
  url: &str,
  client: Watch,
  notes: &mut Vec<String>,
) -> anyhow::Result<Service> {
  use rmcp::ServiceExt;
  use rmcp::transport::StreamableHttpClientTransport;
  use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;

  use crate::oauth;

  let config = within(async {
    let sent = values(&server.headers, &server.headers_command, "header").await?;
    let mut config = StreamableHttpClientTransportConfig::with_uri(url);
    config.custom_headers = headers(&sent)?;
    config.auth_header = token(server).await?.map(|token| bearer(&token).to_string());
    Ok(config)
  })
  .await?;
  // A token in the file is the one this server is called with: what it
  // makes of it is its own business, and no login is started over it.
  let given = config.auth_header.is_some();

  let store = oauth::Store::new(url);
  let signed_in = async |manager| {
    let transport = StreamableHttpClientTransport::with_client(oauth::client(manager), config.clone());
    client.clone().serve(transport).await
  };
  // Signed in before, and it still holds; or never asked to sign in at all.
  let first = within(async {
    let manager = match given {
      true => None,
      false => oauth::resume(url, &server.oauth, &store).await?,
    };
    Ok(match manager {
      Some(manager) => signed_in(manager).await,
      None => {
        let transport = StreamableHttpClientTransport::from_config(config.clone());
        client.clone().serve(transport).await
      }
    })
  })
  .await?;
  let running = match first {
    Ok(running) => Ok(running),
    Err(err) if !given && oauth::wants_login(&err) => {
      // A login that no longer holds is not one to be tried again next time.
      let _ = rmcp::transport::auth::CredentialStore::clear(&store).await;
      let manager = oauth::login(name, url, &server.oauth, &store).await?;
      within(async { Ok(signed_in(manager).await?) }).await
    }
    Err(err) => Err(err.into()),
  };
  if let Some(trouble) = store.trouble() {
    notes.push(format!("MCP {name}: {trouble}"));
  }
  running
}

/// What `work` comes to, if it comes to anything in the time a server has to
/// come up. A server that is slower than that is left out, rather than
/// holding up the session.
#[cfg(feature = "mcp")]
async fn within<T>(work: impl Future<Output = anyhow::Result<T>>) -> anyhow::Result<T> {
  tokio::time::timeout(START_TIMEOUT, work)
    .await
    .map_err(|_| anyhow::anyhow!("no answer in {}s", START_TIMEOUT.as_secs()))?
}

/// One table of values, with the ones written as a command run for.
///
/// A name in both tables is refused rather than resolved by a rule about which
/// wins: a token declared twice is a token one of the two places is wrong
/// about, and quietly using either is how the wrong one goes unnoticed.
#[cfg(feature = "mcp")]
async fn values(
  literal: &BTreeMap<String, String>,
  commands: &BTreeMap<String, String>,
  what: &str,
) -> anyhow::Result<BTreeMap<String, String>> {
  use anyhow::{Context, bail};

  if let Some(name) = commands.keys().find(|name| literal.contains_key(*name)) {
    bail!("{what} {name} is given both a value and a command");
  }
  // Every one of them at once: they are separate programs that wait on
  // separate things, and they are waited on inside the budget the server has
  // to come up at all.
  let run = commands.iter().map(async |(name, command)| {
    let value = crate::config::value(command)
      .await
      .with_context(|| format!("{what} {name}"))?;
    anyhow::Ok((name.clone(), value))
  });
  let mut values = literal.clone();
  values.extend(futures::future::try_join_all(run).await?);
  Ok(values)
}

/// The token a server is given, from wherever the file says it is.
#[cfg(feature = "mcp")]
async fn token(server: &Server) -> anyhow::Result<Option<String>> {
  use anyhow::Context;

  let keyring = !server.token_keyring.is_empty();
  if keyring && (server.token.is_some() || server.token_command.is_some()) {
    anyhow::bail!("token is given more than one way");
  }
  if keyring {
    return crate::keyring::lookup(&server.token_keyring)
      .await
      .context("token-keyring")
      .map(Some);
  }
  one(server.token.as_ref(), server.token_command.as_ref(), "token").await
}

/// One value, written out or as the line of shell that prints it.
#[cfg(feature = "mcp")]
pub async fn one(literal: Option<&String>, command: Option<&String>, what: &str) -> anyhow::Result<Option<String>> {
  use anyhow::Context;

  match (literal, command) {
    (Some(_), Some(_)) => anyhow::bail!("{what} is given both a value and a command"),
    (Some(value), None) => Ok(Some(value.clone())),
    (None, Some(command)) => Ok(Some(
      crate::config::value(command)
        .await
        .with_context(|| format!("{what}-command"))?,
    )),
    (None, None) => Ok(None),
  }
}

/// The token itself, for one written the way the header it goes in is:
/// rmcp puts the scheme in front, and a second would be a token no server
/// takes.
#[cfg(feature = "mcp")]
fn bearer(token: &str) -> &str {
  match token.split_at_checked(7) {
    Some((scheme, rest)) if scheme.eq_ignore_ascii_case("bearer ") => rest.trim_start(),
    _ => token,
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
      let mut value: http::HeaderValue = value
        .parse()
        .with_context(|| format!("not a header value for {name}"))?;
      // A header written in this file is a header that carries a token, near
      // enough, and one marked as such is one nothing prints while looking at
      // the request it went out on.
      value.set_sensitive(true);
      Ok((name, value))
    })
    .collect()
}

#[cfg(not(feature = "mcp"))]
pub async fn connect(_config: Config, _host: &crate::modal::Host) -> Servers {
  Servers::default()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn completion(shape: Shape) -> Completion {
    Completion {
      asking: Completing {
        server: "notes".into(),
        of: Of::Prompt("review".into()),
        argument: "pr".into(),
        value: String::new(),
        context: Vec::new(),
      },
      range: (0, 0),
      shape,
    }
  }

  #[test]
  fn a_value_taken_is_written_the_way_it_is_read_back() {
    // An argument before the last is a word, and the next one starts after it.
    let word = completion(Shape::Argument { last: false });
    assert_eq!(word.insert("12"), ("12 ".to_string(), true));
    assert_eq!(word.insert("a b"), ("\"a b\" ".to_string(), true));
    // The last takes the rest of the line, spaces and all.
    let rest = completion(Shape::Argument { last: true });
    assert_eq!(rest.insert("a b"), ("a b".to_string(), false));
    // A hole is filled in inside its token, with the template's text after
    // it, and the next hole is completed next.
    let hole = |more| {
      completion(Shape::Hole {
        before: "gh:repo://".into(),
        after: "/".into(),
        more,
      })
    };
    assert_eq!(hole(true).insert("me"), ("&gh:repo://me/".to_string(), true));
    assert_eq!(
      hole(false).insert("my fa"),
      ("&\"gh:repo://my fa/\"".to_string(), false)
    );
  }

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
        env-command.KEY = "gh auth token"

        [docs]
        url = "https://example.com/mcp"
        token-keyring = { service = "work", account = "mcp" }
        headers.Authorization = "Bearer k"
        headers-command.X-Api-Key = "pass show work/mcp"
        timeout = 60
      "#,
    );
    assert_eq!(read.len(), 2);
    let files = &read["files"];
    assert_eq!(files.command.as_deref(), Some("mcp-files --root ."));
    assert_eq!(files.env["TOKEN"], "x");
    assert_eq!(files.env_command["KEY"], "gh auth token");
    assert_eq!(files.timeout, None, "a server says nothing about time by default");
    let docs = &read["docs"];
    assert_eq!(docs.url.as_deref(), Some("https://example.com/mcp"));
    assert_eq!(docs.token_keyring["service"], "work");
    assert_eq!(docs.token_keyring["account"], "mcp");
    assert_eq!(docs.headers["Authorization"], "Bearer k");
    assert_eq!(docs.headers_command["X-Api-Key"], "pass show work/mcp");
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
  #[tokio::test]
  async fn a_value_can_be_what_a_command_prints_and_what_went_wrong_is_never_that() {
    let map = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
      pairs
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect()
    };
    let none = BTreeMap::new();

    // What the command printed is the value, without the newline it was
    // printed with, and a value written out is left alone.
    let read = values(
      &map(&[("KEEP", "as written")]),
      &map(&[("TOKEN", "printf 'secret \n'"), ("SUM", "echo $((1 + 1))")]),
      "env",
    )
    .await
    .expect("both kinds of value");
    assert_eq!(read["KEEP"], "as written");
    assert_eq!(read["TOKEN"], "secret");
    assert_eq!(read["SUM"], "2", "a line of shell, not an argv");

    // Nothing here may stop to ask the terminal anything, so a command that
    // reads is given the end of the input rather than the session's keys.
    let read = values(&none, &map(&[("ASKS", "cat")]), "env")
      .await
      .expect("no waiting");
    assert_eq!(read["ASKS"], "");

    // A command that failed is told about as what it said for itself — the
    // value, the command from the file, and the first line of its complaint.
    // Never what it printed: that is the secret it half produced, and a note
    // in the transcript is a note in the session file on disk.
    let err = values(
      &none,
      &map(&[("TOKEN", "echo $((111 * 111)); echo locked >&2; exit 1")]),
      "env",
    )
    .await
    .expect_err("a command that failed");
    let err = format!("{err:#}");
    assert!(err.contains("env TOKEN"), "which value it was: {err}");
    assert!(err.contains("locked"), "and why: {err}");
    assert!(!err.contains("12321"), "but never the output: {err}");

    // Given twice, it is wrong in one of the two places.
    let err = values(
      &map(&[("Authorization", "Bearer k")]),
      &map(&[("Authorization", "true")]),
      "header",
    )
    .await
    .expect_err("declared twice")
    .to_string();
    assert!(err.contains("header Authorization"), "{err}");

    // A token is one or the other too, and is the token, whatever scheme it
    // was written with.
    let text = |value: &str| value.to_string();
    let read = one(Some(&text("k")), None, "token").await.expect("written");
    assert_eq!(read.as_deref(), Some("k"));
    let read = one(None, Some(&text("echo k")), "token").await.expect("printed");
    assert_eq!(read.as_deref(), Some("k"));
    let err = one(Some(&text("k")), Some(&text("echo k")), "token")
      .await
      .expect_err("twice")
      .to_string();
    assert!(err.contains("token is given both"), "{err}");
    assert_eq!(one(None, None, "token").await.expect("none"), None);
    // The keyring is a third way, and not one to give beside the others.
    let both = Server {
      token_command: Some("echo k".into()),
      token_keyring: map(&[("service", "work")]),
      ..Server::default()
    };
    let err = token(&both).await.expect_err("twice").to_string();
    assert!(err.contains("more than one way"), "{err}");
    assert_eq!(bearer("Bearer k"), "k");
    assert_eq!(bearer("bearer  k"), "k");
    assert_eq!(bearer("k"), "k");
    assert_eq!(bearer("Bearerk"), "Bearerk");

    // And a header this file carries is one nothing prints while looking at
    // the request it went out on.
    let built = headers(&map(&[("Authorization", "Bearer k")])).expect("a header");
    let value = &built[&http::HeaderName::from_static("authorization")];
    assert_eq!(value.to_str().expect("text"), "Bearer k");
    assert!(value.is_sensitive(), "a token is not for printing");
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
}
