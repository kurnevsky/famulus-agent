//! fa as an agent an editor drives: the Agent Client Protocol, spoken over
//! stdin and stdout in place of the terminal UI.
//!
//! The run loop is the same one the terminal starts. A prompt starts a run,
//! the run's `AgentEvent`s become `session/update`s as they arrive, and the
//! messages the run hands back are saved to the session the way the terminal
//! saves them — so a session started in an editor can be resumed in the
//! terminal, and the other way round.
//!
//! What the terminal asks the user directly has no way to be asked here:
//! stable ACP has no free-form question for an agent to put to the user, so
//! the `ask` tool is kept from the model, and a form an MCP server brings up
//! goes unanswered.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
  AgentCapabilities, AvailableCommand, AvailableCommandsUpdate, CancelNotification, CloseSessionRequest,
  CloseSessionResponse, ContentBlock, ContentChunk, DeleteSessionRequest, DeleteSessionResponse, Diff,
  EmbeddedResourceResource, Error, ImageContent, Implementation, InitializeRequest, InitializeResponse,
  ListSessionsRequest, ListSessionsResponse, LoadSessionRequest, LoadSessionResponse, McpCapabilities, McpServer,
  NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse, ResumeSessionRequest,
  ResumeSessionResponse, SessionCapabilities, SessionCloseCapabilities, SessionConfigOption,
  SessionConfigOptionCategory, SessionConfigOptionValue, SessionConfigSelectOption, SessionDeleteCapabilities,
  SessionId, SessionInfo, SessionInfoUpdate, SessionListCapabilities, SessionNotification, SessionResumeCapabilities,
  SessionUpdate, SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, StopReason, ToolCall, ToolCallContent,
  ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use agent_client_protocol::{Agent, Client, ConnectionTo, Responder, Stdio};
use anyhow::Result;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rig_agent::tool::Tool;
use rig_core::completion::Message;
use rig_core::message::{AssistantContent, AssistantMessage, ImageMediaType, MimeType, UserContent};
use tokio::sync::mpsc;

use crate::agent::{self, AgentEvent, Agents, Control, ModelInfo, start_compaction, start_run};
use crate::compaction::DEFAULT_CONTEXT_WINDOW;
use crate::modal::Host;
use crate::session::{Outcome, Session, Store};
use crate::{mcp, tools};

/// How long the provider has to say what models it offers before the
/// sessions opened in the meantime are offered only the one in use.
const LISTING_TIMEOUT: Duration = Duration::from_secs(20);

/// The one command a prompt can be instead of something said to the model.
const COMPACT: &str = "compact";

/// What `main` worked out before it knew it was serving an editor: the
/// settings every session starts from, and what each one works out for
/// itself in the directory the editor names.
pub struct Setup {
  pub cfg: agent::Config,
  pub store: Option<Store>,
  /// `--context-window`, which stands whatever the provider says a model holds.
  pub context_window: Option<u64>,
  /// The files MCP servers are declared in, and whether they were named
  /// rather than looked for; `None` starts none.
  pub mcp: Option<(Vec<PathBuf>, bool)>,
  /// `--tools` and `--no-tools`, which are only settled once a session's
  /// servers have said what they bring.
  pub tools: Vec<String>,
  pub no_tools: Vec<String>,
}

/// Serve one editor until it hangs up.
pub async fn serve(setup: Setup) -> Result<()> {
  let fa = Arc::new(Fa {
    setup,
    sessions: Mutex::default(),
    models: tokio::sync::OnceCell::new(),
  });
  // Asked once, as early as possible: the first session to open is the one
  // that would otherwise wait for it.
  tokio::spawn({
    let fa = fa.clone();
    async move {
      fa.models().await;
    }
  });

  let (on_new, on_load, on_resume, on_list, on_delete, on_close, on_config, on_prompt, on_cancel) = (
    fa.clone(),
    fa.clone(),
    fa.clone(),
    fa.clone(),
    fa.clone(),
    fa.clone(),
    fa.clone(),
    fa.clone(),
    fa.clone(),
  );
  Agent
    .builder()
    .name("fa")
    .on_receive_request(
      async move |_: InitializeRequest, responder: Responder<InitializeResponse>, _: ConnectionTo<Client>| {
        responder.respond(fa.initialize())
      },
      agent_client_protocol::on_receive_request!(),
    )
    // Everything that opens a session starts servers or reads files, so it
    // is done beside the connection rather than holding up whatever arrives
    // behind it — a cancel for another session, say.
    .on_receive_request(
      async move |request: NewSessionRequest, responder: Responder<NewSessionResponse>, cx: ConnectionTo<Client>| {
        let fa = on_new.clone();
        tokio::spawn(async move {
          let result = fa.new_session(request).await;
          let opened = result.as_ref().ok().map(|response| response.session_id.clone());
          let _ = responder.respond_with_result(result);
          if let Some(id) = opened {
            announce(&cx, &id);
          }
        });
        Ok(())
      },
      agent_client_protocol::on_receive_request!(),
    )
    .on_receive_request(
      async move |request: LoadSessionRequest, responder: Responder<LoadSessionResponse>, cx: ConnectionTo<Client>| {
        let fa = on_load.clone();
        tokio::spawn(async move {
          let id = request.session_id.clone();
          let result = fa.load_session(request, &cx).await;
          let opened = result.is_ok();
          let _ = responder.respond_with_result(result);
          if opened {
            announce(&cx, &id);
          }
        });
        Ok(())
      },
      agent_client_protocol::on_receive_request!(),
    )
    .on_receive_request(
      async move |request: ResumeSessionRequest,
                  responder: Responder<ResumeSessionResponse>,
                  cx: ConnectionTo<Client>| {
        let fa = on_resume.clone();
        tokio::spawn(async move {
          let id = request.session_id.clone();
          let result = fa.resume_session(request).await;
          let opened = result.is_ok();
          let _ = responder.respond_with_result(result);
          if opened {
            announce(&cx, &id);
          }
        });
        Ok(())
      },
      agent_client_protocol::on_receive_request!(),
    )
    .on_receive_request(
      async move |request: ListSessionsRequest, responder: Responder<ListSessionsResponse>, _: ConnectionTo<Client>| {
        let fa = on_list.clone();
        tokio::task::spawn_blocking(move || {
          let _ = responder.respond(fa.list_sessions(request));
        });
        Ok(())
      },
      agent_client_protocol::on_receive_request!(),
    )
    .on_receive_request(
      async move |request: DeleteSessionRequest,
                  responder: Responder<DeleteSessionResponse>,
                  _: ConnectionTo<Client>| { responder.respond_with_result(on_delete.delete_session(request)) },
      agent_client_protocol::on_receive_request!(),
    )
    .on_receive_request(
      async move |request: CloseSessionRequest, responder: Responder<CloseSessionResponse>, _: ConnectionTo<Client>| {
        on_close.close_session(&request.session_id);
        responder.respond(CloseSessionResponse::new())
      },
      agent_client_protocol::on_receive_request!(),
    )
    .on_receive_request(
      async move |request: SetSessionConfigOptionRequest,
                  responder: Responder<SetSessionConfigOptionResponse>,
                  _: ConnectionTo<Client>| {
        let fa = on_config.clone();
        tokio::spawn(async move {
          let _ = responder.respond_with_result(fa.set_config_option(request).await);
        });
        Ok(())
      },
      agent_client_protocol::on_receive_request!(),
    )
    // A prompt is answered when its run is over, which is a long time for
    // the connection to wait on: it goes on beside it, and a cancel arriving
    // in the meantime is read at once.
    .on_receive_request(
      async move |request: PromptRequest, responder: Responder<PromptResponse>, cx: ConnectionTo<Client>| {
        let fa = on_prompt.clone();
        tokio::spawn(async move {
          let _ = responder.respond_with_result(fa.prompt(request, cx).await);
        });
        Ok(())
      },
      agent_client_protocol::on_receive_request!(),
    )
    .on_receive_notification(
      async move |notification: CancelNotification, _: ConnectionTo<Client>| {
        on_cancel.cancel(&notification.session_id);
        Ok(())
      },
      agent_client_protocol::on_receive_notification!(),
    )
    .connect_to(Stdio::new())
    .await
    .map_err(|err| anyhow::anyhow!("ACP: {err}"))
}

/// Everything the connection's handlers share.
struct Fa {
  setup: Setup,
  /// The sessions the editor has open, by their id.
  sessions: Mutex<HashMap<String, Arc<Live>>>,
  /// What the provider said it offers, or nothing when it would not say.
  models: tokio::sync::OnceCell<Vec<ModelInfo>>,
}

/// One open session.
struct Live {
  /// Held apart from the rest, so a cancel reaches the run without waiting
  /// for the lock the run is holding.
  control: Control,
  /// Taken by whatever is using the session — a prompt, for as long as its
  /// run goes on — so a second one finds it busy rather than interleaved.
  state: tokio::sync::Mutex<State>,
}

struct State {
  session: Session,
  agents: Agents,
  /// The settings this session runs on: the shared ones, held to the tools
  /// its servers left, at the model it was last pointed at.
  cfg: agent::Config,
  cwd: PathBuf,
  /// The session's MCP servers, which run as long as this is held.
  _servers: mcp::Servers,
}

impl Fa {
  fn initialize(&self) -> InitializeResponse {
    let saved = self.setup.store.is_some();
    let mut sessions = SessionCapabilities::new()
      .resume(SessionResumeCapabilities::new())
      .close(SessionCloseCapabilities::new());
    if saved {
      sessions = sessions
        .list(SessionListCapabilities::new())
        .delete(SessionDeleteCapabilities::new());
    }
    let capabilities = AgentCapabilities::new()
      .load_session(saved)
      .prompt_capabilities(
        PromptCapabilities::new()
          .image(self.setup.cfg.vision)
          .embedded_context(true),
      )
      .mcp_capabilities(McpCapabilities::new().http(cfg!(feature = "mcp")))
      .session_capabilities(sessions);
    InitializeResponse::new(ProtocolVersion::V1)
      .agent_capabilities(capabilities)
      .agent_info(Implementation::new("fa", env!("CARGO_PKG_VERSION")).title("famulus-agent"))
  }

  /// What the provider offers, asked for the first time it is wanted.
  async fn models(&self) -> &[ModelInfo] {
    self
      .models
      .get_or_init(|| async {
        let cfg = &self.setup.cfg;
        match tokio::time::timeout(LISTING_TIMEOUT, agent::list_models(cfg)).await {
          Ok(Ok(models)) => models,
          Ok(Err(err)) => {
            eprintln!("{err:#}");
            Vec::new()
          }
          Err(_) => {
            eprintln!(
              "{} did not answer for its models within {} seconds",
              cfg.provider.label(),
              LISTING_TIMEOUT.as_secs()
            );
            Vec::new()
          }
        }
      })
      .await
  }

  /// The window `model` is compacted at: the one `--context-window` named,
  /// or the one the provider reports for it, or the fallback.
  async fn window(&self, model: &str) -> u64 {
    let reported = self
      .models()
      .await
      .iter()
      .find(|info| info.id == model)
      .and_then(|info| info.context_length);
    self.setup.context_window.or(reported).unwrap_or(DEFAULT_CONTEXT_WINDOW)
  }

  async fn config_options(&self, cfg: &agent::Config) -> Vec<SessionConfigOption> {
    model_option(cfg, self.models().await)
  }

  fn live(&self, id: &SessionId) -> Result<Arc<Live>, Error> {
    self
      .sessions
      .lock()
      .expect("a map nobody panicked holding")
      .get(id.0.as_ref())
      .cloned()
      .ok_or_else(|| Error::invalid_params().data(format!("no open session {id}")))
  }

  /// Make `session` one the editor can talk to, in `cwd`, with the servers
  /// fa's own files declare and the ones the editor brought.
  async fn open(&self, cwd: PathBuf, servers: Vec<McpServer>, session: Option<Session>) -> Result<String, Error> {
    if !cwd.is_absolute() {
      return Err(Error::invalid_params().data(format!("{} is not an absolute path", cwd.display())));
    }
    // What a server asks the user has nobody to put it to, so it is put
    // away unanswered — which the server hears as a decline — and what it
    // has to say goes where the editor keeps the agent's log.
    let (tx, mut rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
      while let Some(event) = rx.recv().await {
        if let AgentEvent::Mcp(note) = event {
          eprintln!("{note}");
        }
      }
    });
    let host = Host::new(tx);

    let (mut config, mut notes) = match &self.setup.mcp {
      Some((files, named)) => mcp::load(files, *named),
      None => Default::default(),
    };
    add_servers(&mut config, servers, &mut notes);
    let servers = mcp::connect(config, &host).await;
    notes.extend(servers.notes().iter().cloned());

    let mut cfg = self.setup.cfg.clone();
    cfg.tools = tools::rules(
      &servers.catalog().tool_names(),
      &self.setup.tools,
      &self.setup.no_tools,
      &mut notes,
    );
    cfg.tools.refused.push(<tools::AskTool as Tool>::NAME.to_string());
    for note in notes {
      eprintln!("{note}");
    }

    let mut session = match session {
      Some(session) => session,
      None => Session::new(self.setup.store.as_ref(), &cwd, &label(&cfg)),
    };
    if self.setup.store.is_none() {
      session.disable_persistence();
    }
    cfg.compaction.context_window = self.window(&cfg.model).await;
    let agents = agent::build_agents(&cfg, &cwd, &host, &servers).map_err(failed)?;
    let id = session.id.clone();
    let live = Arc::new(Live {
      control: agents.control.clone(),
      state: tokio::sync::Mutex::new(State {
        session,
        agents,
        cfg,
        cwd,
        _servers: servers,
      }),
    });
    // Opening a session that is already open is opening it again: the one
    // it replaces finishes whatever it was doing on its own.
    self
      .sessions
      .lock()
      .expect("a map nobody panicked holding")
      .insert(id.clone(), live);
    Ok(id)
  }

  async fn new_session(&self, request: NewSessionRequest) -> Result<NewSessionResponse, Error> {
    let id = self.open(request.cwd, request.mcp_servers, None).await?;
    let options = self.options_of(&id).await?;
    Ok(NewSessionResponse::new(id).config_options(options))
  }

  /// A saved session, picked back up: what was said in it is told to the
  /// editor again before the answer, as the protocol has it.
  async fn load_session(
    &self,
    request: LoadSessionRequest,
    cx: &ConnectionTo<Client>,
  ) -> Result<LoadSessionResponse, Error> {
    let session = self.saved(&request.session_id)?;
    let id = self.open(request.cwd, request.mcp_servers, Some(session)).await?;
    let live = self.live(&request.session_id)?;
    {
      let state = live.state.lock().await;
      for update in replay(&state.session, &state.cwd) {
        notify(cx, &request.session_id, update);
      }
    }
    let options = self.options_of(&id).await?;
    Ok(LoadSessionResponse::new().config_options(options))
  }

  /// The same, for an editor that already has the conversation on screen.
  async fn resume_session(&self, request: ResumeSessionRequest) -> Result<ResumeSessionResponse, Error> {
    let session = self.saved(&request.session_id)?;
    let id = self.open(request.cwd, request.mcp_servers, Some(session)).await?;
    let options = self.options_of(&id).await?;
    Ok(ResumeSessionResponse::new().config_options(options))
  }

  async fn options_of(&self, id: &str) -> Result<Vec<SessionConfigOption>, Error> {
    let live = self.live(&SessionId::new(id))?;
    let cfg = live.state.lock().await.cfg.clone();
    Ok(self.config_options(&cfg).await)
  }

  /// The saved session named `id` — that one exactly, not one its id begins
  /// with: the editor is naming a session it was told about.
  fn saved(&self, id: &SessionId) -> Result<Session, Error> {
    let path = self.saved_path(id)?;
    Session::load(&path).map_err(failed)
  }

  fn saved_path(&self, id: &SessionId) -> Result<PathBuf, Error> {
    let Some(store) = &self.setup.store else {
      return Err(Error::invalid_params().data("sessions are not saved (--no-session)"));
    };
    store
      .list()
      .into_iter()
      .find(|info| *info.id == *id.0)
      .map(|info| info.path)
      .ok_or_else(|| Error::resource_not_found(Some(id.to_string())))
  }

  fn list_sessions(&self, request: ListSessionsRequest) -> ListSessionsResponse {
    let Some(store) = &self.setup.store else {
      return ListSessionsResponse::new(Vec::new());
    };
    let sessions = store
      .list()
      .into_iter()
      .filter(|info| request.cwd.as_deref().is_none_or(|cwd| Path::new(&info.cwd) == cwd))
      .map(|info| {
        SessionInfo::new(info.id.clone(), info.cwd.clone())
          .title(info.title().to_string())
          .updated_at(info.modified.to_rfc3339())
      })
      .collect();
    ListSessionsResponse::new(sessions)
  }

  fn delete_session(&self, request: DeleteSessionRequest) -> Result<DeleteSessionResponse, Error> {
    self.close_session(&request.session_id);
    let path = self.saved_path(&request.session_id)?;
    if let Some(store) = &self.setup.store {
      store.delete(&path).map_err(failed)?;
    }
    Ok(DeleteSessionResponse::new())
  }

  fn close_session(&self, id: &SessionId) {
    let live = self
      .sessions
      .lock()
      .expect("a map nobody panicked holding")
      .remove(id.0.as_ref());
    if let Some(live) = live {
      live.control.cancel();
    }
  }

  fn cancel(&self, id: &SessionId) {
    if let Ok(live) = self.live(id) {
      live.control.cancel();
    }
  }

  /// Point a session at another model. Not while it is answering: the run
  /// reads the model it was started on.
  async fn set_config_option(
    &self,
    request: SetSessionConfigOptionRequest,
  ) -> Result<SetSessionConfigOptionResponse, Error> {
    let live = self.live(&request.session_id)?;
    let Ok(mut state) = live.state.try_lock() else {
      return Err(Error::invalid_request().data("the session is busy answering a prompt"));
    };
    let model = match (&*request.config_id.0, &request.value) {
      ("model", SessionConfigOptionValue::ValueId { value }) => value.0.to_string(),
      _ => return Err(Error::invalid_params().data(format!("no option {}", request.config_id.0))),
    };
    let mut cfg = state.cfg.clone();
    cfg.compaction.context_window = self.window(&model).await;
    cfg.model = model;
    state.agents.use_model(&cfg).map_err(failed)?;
    if let Err(err) = state.session.set_model(&label(&cfg)) {
      eprintln!("could not save the session: {err:#}");
    }
    state.cfg = cfg;
    let options = self.config_options(&state.cfg).await;
    Ok(SetSessionConfigOptionResponse::new(options))
  }

  async fn prompt(&self, request: PromptRequest, cx: ConnectionTo<Client>) -> Result<PromptResponse, Error> {
    let live = self.live(&request.session_id)?;
    let Ok(mut state) = live.state.try_lock() else {
      return Err(Error::invalid_request().data("the session is already answering a prompt"));
    };
    let said = Said::from(request.prompt, state.cfg.vision);
    let turn = Turn {
      session: request.session_id,
      cwd: state.cwd.clone(),
      cx,
      window: state.cfg.compaction.context_window,
      calls: HashMap::new(),
      ids: HashMap::new(),
      outcomes: HashMap::new(),
    };
    if said.text.trim() == format!("/{COMPACT}") && said.images.is_empty() {
      return turn.compact(&mut state, &live.control).await;
    }
    if state.session.history.is_empty() {
      let title = said.text.lines().map(str::trim).find(|line| !line.is_empty());
      if let Some(title) = title {
        turn.tell(SessionUpdate::SessionInfoUpdate(
          SessionInfoUpdate::new().title(title.chars().take(200).collect::<String>()),
        ));
      }
    }
    let (tx, rx) = mpsc::unbounded_channel();
    start_run(
      state.agents.runtime.clone(),
      live.control.clone(),
      state.session.history.clone(),
      vec![said.message()],
      tx,
    );
    turn.follow(&mut state, &live.control, rx).await
  }
}

/// Tell the editor what the session it just opened can be asked besides a
/// prompt — once it has the session's id, which is in the answer just sent.
fn announce(cx: &ConnectionTo<Client>, id: &SessionId) {
  let compact = AvailableCommand::new(
    COMPACT,
    "Summarize the conversation so far, to make room in the context",
  );
  notify(
    cx,
    id,
    SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(vec![compact])),
  );
}

fn notify(cx: &ConnectionTo<Client>, id: &SessionId, update: SessionUpdate) {
  let _ = cx.send_notification(SessionNotification::new(id.clone(), update));
}

/// Something that went wrong, said as the error a request answers with.
fn failed(err: impl std::fmt::Display) -> Error {
  let mut error = Error::internal_error();
  error.message = format!("{err:#}");
  error
}

/// What a session's model is called in its file: the provider and the model,
/// the way the terminal names it.
fn label(cfg: &agent::Config) -> String {
  format!("{}/{}", cfg.provider.label(), cfg.model)
}

/// The model picker the editor shows: what the provider offers, with the one
/// in use among them whether or not it was listed.
fn model_option(cfg: &agent::Config, models: &[ModelInfo]) -> Vec<SessionConfigOption> {
  let mut choices: Vec<SessionConfigSelectOption> = models
    .iter()
    .map(|model| {
      SessionConfigSelectOption::new(model.id.clone(), model.name.clone().unwrap_or_else(|| model.id.clone()))
    })
    .collect();
  if !models.iter().any(|model| model.id == cfg.model) {
    choices.insert(0, SessionConfigSelectOption::new(cfg.model.clone(), cfg.model.clone()));
  }
  vec![
    SessionConfigOption::select("model", "Model", cfg.model.clone(), choices)
      .category(SessionConfigOptionCategory::Model),
  ]
}

// ---------------------------------------------------------------- servers

/// The servers the editor brought, alongside the ones fa's own files declare
/// — in place of one of those that has the same name, since the editor's is
/// the nearer word.
#[cfg(feature = "mcp")]
fn add_servers(config: &mut mcp::Config, servers: Vec<McpServer>, notes: &mut Vec<String>) {
  for server in servers {
    let (name, server) = match server {
      McpServer::Stdio(stdio) => {
        let command = std::iter::once(stdio.command.to_string_lossy().into_owned())
          .chain(stdio.args)
          .map(|word| quoted(&word))
          .collect::<Vec<_>>()
          .join(" ");
        let env = stdio.env.into_iter().map(|var| (var.name, var.value)).collect();
        (
          stdio.name,
          mcp::Server {
            command: Some(command),
            env,
            ..Default::default()
          },
        )
      }
      McpServer::Http(http) => {
        let headers = http
          .headers
          .into_iter()
          .map(|header| (header.name, header.value))
          .collect();
        (
          http.name,
          mcp::Server {
            url: Some(http.url),
            headers,
            ..Default::default()
          },
        )
      }
      McpServer::Sse(sse) => {
        notes.push(format!(
          "MCP {}: SSE is not spoken here, only streamable HTTP",
          sse.name
        ));
        continue;
      }
      _ => continue,
    };
    config.insert(name, server);
  }
}

#[cfg(not(feature = "mcp"))]
fn add_servers(_config: &mut mcp::Config, servers: Vec<McpServer>, notes: &mut Vec<String>) {
  if !servers.is_empty() {
    notes.push("MCP: this build has no MCP support (built without the `mcp` feature).".into());
  }
}

/// `word` as one word of a line of shell, whatever it holds.
#[cfg(feature = "mcp")]
fn quoted(word: &str) -> String {
  let plain = |c: char| c.is_ascii_alphanumeric() || "-_./=:,+@%".contains(c);
  match !word.is_empty() && word.chars().all(plain) {
    true => word.to_string(),
    false => format!("'{}'", word.replace('\'', r"'\''")),
  }
}

// ---------------------------------------------------------------- prompts

/// A prompt as the editor sent it, made into what the model is given.
struct Said {
  /// The prose, with what it points at written into it.
  text: String,
  /// Images it attached, each with what it was called.
  images: Vec<(String, String, ImageMediaType)>,
}

impl Said {
  fn from(blocks: Vec<ContentBlock>, vision: bool) -> Self {
    let mut text = String::new();
    // Files the editor sent the text of, which go after the prose that
    // mentions them, the way a file read in is set out.
    let mut context = Vec::new();
    let mut images = Vec::new();
    for block in blocks {
      match block {
        ContentBlock::Text(said) => text.push_str(&said.text),
        ContentBlock::ResourceLink(link) => text.push_str(&mention(&link.name, &link.uri)),
        ContentBlock::Resource(resource) => match resource.resource {
          EmbeddedResourceResource::TextResourceContents(file) => {
            text.push_str(&mention(&name_of(&file.uri), &file.uri));
            context.push(format!(
              "<context ref=\"{}\">\n{}\n</context>",
              file.uri,
              file.text.trim_end_matches('\n')
            ));
          }
          EmbeddedResourceResource::BlobResourceContents(blob) => {
            match blob.mime_type.as_deref().and_then(ImageMediaType::from_mime_type) {
              Some(media) if vision => images.push((blob.uri.clone(), blob.blob, media)),
              _ => text.push_str(&mention(&name_of(&blob.uri), &blob.uri)),
            }
          }
          _ => {}
        },
        ContentBlock::Image(image) => match ImageMediaType::from_mime_type(&image.mime_type) {
          Some(media) if vision => {
            images.push((image.uri.clone().unwrap_or_else(|| "image".into()), image.data, media))
          }
          _ => text.push_str("[an image the model cannot see]"),
        },
        _ => {}
      }
    }
    for block in context {
      text.push_str("\n\n");
      text.push_str(&block);
    }
    Self { text, images }
  }

  /// The prose first, then each image behind a note naming it: the shape a
  /// prompt typed in the terminal takes, which is the shape the rest of fa
  /// reads a prompt back in.
  fn message(&self) -> Message {
    let mut content = vec![UserContent::text(self.text.clone())];
    for (name, data, media) in &self.images {
      content.push(UserContent::text(format!(
        "[Attached {name} — {}]",
        media.to_mime_type()
      )));
      content.push(UserContent::image_base64(data.clone(), Some(media.clone()), None));
    }
    Message::User { content }
  }
}

/// A file the prompt points at, written where it was pointed at — as a path
/// the model's tools can open, when it is one.
fn mention(name: &str, uri: &str) -> String {
  match uri.strip_prefix("file://") {
    Some(path) => format!("@{path}"),
    None => format!("[@{name}]({uri})"),
  }
}

fn name_of(uri: &str) -> String {
  uri.rsplit('/').next().unwrap_or(uri).to_string()
}

// ---------------------------------------------------------------- a turn

/// One prompt's run, as the editor is told about it.
struct Turn {
  session: SessionId,
  cwd: PathBuf,
  cx: ConnectionTo<Client>,
  /// The window the usage is reported against.
  window: u64,
  /// The calls the editor has been told of and not yet told the end of, by
  /// the id it was told them under, with the arguments it was last told.
  calls: HashMap<String, Option<serde_json::Value>>,
  /// Which of those each call is, by the id the run names it with: the
  /// editor first hears of a call while it is being written, before it has
  /// the id it is run under.
  ids: HashMap<String, String>,
  /// How the calls went, for the session file to keep.
  outcomes: HashMap<String, Outcome>,
}

impl Turn {
  fn tell(&self, update: SessionUpdate) {
    notify(&self.cx, &self.session, update);
  }

  /// Relay the run's events until it hands back what it added, and save
  /// that; the stop reason is how it ended.
  async fn follow(
    mut self,
    state: &mut State,
    control: &Control,
    mut rx: mpsc::UnboundedReceiver<AgentEvent>,
  ) -> Result<PromptResponse, Error> {
    let mut errors = Vec::new();
    while let Some(event) = rx.recv().await {
      match event {
        AgentEvent::Done { messages } => {
          self.keep(state, messages);
          return Ok(PromptResponse::new(StopReason::EndTurn));
        }
        AgentEvent::Ended { messages } => {
          self.keep(state, messages);
          self.abandon();
          return match (control.cancelled(), errors.is_empty()) {
            (true, _) => Ok(PromptResponse::new(StopReason::Cancelled)),
            (false, false) => Err(failed(errors.join("\n"))),
            // Out of room with nothing left to compact, which is no
            // error of the editor's request, so it is said as an answer.
            (false, true) => {
              self.say("The context is full and there is nothing left to compact.");
              Ok(PromptResponse::new(StopReason::MaxTokens))
            }
          };
        }
        // What the run got through before it ran out of room is saved
        // before the summary takes its place.
        AgentEvent::Compacting { messages, .. } => self.keep(state, messages),
        AgentEvent::Compacted(Some(compacted)) => {
          report(state.session.compacted(&compacted.summary, compacted.kept));
        }
        AgentEvent::Error(err) => errors.push(err),
        event => self.relay(event),
      }
    }
    Err(failed("the run ended without a word"))
  }

  /// `/compact`.
  async fn compact(self, state: &mut State, control: &Control) -> Result<PromptResponse, Error> {
    if state.session.history.is_empty() {
      self.say("Nothing to compact.");
      return Ok(PromptResponse::new(StopReason::EndTurn));
    }
    control.begin();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let handle = start_compaction(
      state.agents.runtime.clone(),
      state.session.history.clone(),
      state.cfg.compaction,
      tx,
    );
    let mut errors = Vec::new();
    loop {
      let event = tokio::select! {
        biased;
        () = control.stopped() => {
          handle.abort();
          return Ok(PromptResponse::new(StopReason::Cancelled));
        }
        event = rx.recv() => event,
      };
      match event {
        Some(AgentEvent::Compacted(Some(compacted))) => {
          let (summarized, kept) = (compacted.summarized, compacted.kept.len());
          report(state.session.compacted(&compacted.summary, compacted.kept));
          self.say(&format!(
            "Compacted {summarized} messages into a summary; kept the last {kept}."
          ));
          return Ok(PromptResponse::new(StopReason::EndTurn));
        }
        Some(AgentEvent::Compacted(None)) => {
          self.say("Nothing to compact.");
          return Ok(PromptResponse::new(StopReason::EndTurn));
        }
        Some(AgentEvent::Error(err)) => errors.push(err),
        Some(_) => {}
        None if errors.is_empty() => return Err(failed("the compaction ended without a word")),
        None => return Err(failed(errors.join("\n"))),
      }
    }
  }

  fn say(&self, text: &str) {
    self.tell(SessionUpdate::AgentMessageChunk(ContentChunk::new(text.into())));
  }

  /// Save what the run added, with how its calls went.
  fn keep(&mut self, state: &mut State, messages: Vec<Message>) {
    if messages.is_empty() {
      return;
    }
    let outcomes = std::mem::take(&mut self.outcomes);
    report(state.session.append_with(messages, &outcomes));
  }

  /// The calls a stopped run will never answer: written but never run, or
  /// cut off while running.
  fn abandon(&mut self) {
    for (id, _) in self.calls.drain() {
      notify(
        &self.cx,
        &self.session,
        SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
          id,
          ToolCallUpdateFields::new().status(ToolCallStatus::Failed),
        )),
      );
    }
  }

  fn relay(&mut self, event: AgentEvent) {
    match event {
      AgentEvent::Text(text) => self.tell(SessionUpdate::AgentMessageChunk(ContentChunk::new(text.into()))),
      AgentEvent::Reasoning(text) => self.tell(SessionUpdate::AgentThoughtChunk(ContentChunk::new(text.into()))),
      AgentEvent::ToolCallDelta { id, name, args } => self.writing(id, name, &args),
      AgentEvent::ToolCall {
        name,
        args,
        call,
        internal,
      } => {
        self.ids.insert(call, internal.clone());
        let fields = ToolCallUpdateFields::new()
          .title(title(&name, Some(&args)))
          .kind(kind(&name))
          .status(ToolCallStatus::InProgress)
          .locations(any(locations(&name, &args, &self.cwd)))
          .content(any(proposed(&name, &args, &self.cwd)))
          .raw_input(args.clone());
        self.update(internal, Some(args), fields);
      }
      AgentEvent::ToolOutput { call, text } => {
        let Some(id) = self.ids.get(&call).cloned() else {
          return;
        };
        self.tell(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
          id,
          ToolCallUpdateFields::new().content(vec![fenced(&text).into()]),
        )));
      }
      AgentEvent::ToolResult {
        name,
        output,
        images,
        is_error,
        call,
        diff,
      } => {
        if is_error || diff.is_some() {
          self.outcomes.insert(
            call.clone(),
            Outcome {
              call: call.clone(),
              failed: is_error,
              diff,
            },
          );
        }
        let Some(id) = self.ids.remove(&call) else {
          return;
        };
        let args = self.calls.remove(&id).flatten();
        let fields = finished(&name, args.as_ref(), &output, &images, is_error, &self.cwd);
        self.tell(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(id, fields)));
      }
      AgentEvent::Usage { context_tokens, .. } => {
        self.tell(SessionUpdate::UsageUpdate(
          agent_client_protocol::schema::v1::UsageUpdate::new(context_tokens, self.window),
        ));
      }
      // Nothing here asks the user: an editor has no way to answer.
      _ => {}
    }
  }

  /// A call as far as the model has written it. The editor hears of it when
  /// its name is known, and again whenever what has arrived of it reads as
  /// whole arguments — not at every character, which would be a message
  /// for each.
  fn writing(&mut self, id: String, name: String, args: &str) {
    let parsed = match args.trim_end().ends_with('}') {
      true => serde_json::from_str::<serde_json::Value>(args).ok(),
      false => None,
    };
    if let Some(told) = self.calls.get(&id)
      && (parsed.is_none() || *told == parsed)
    {
      return;
    }
    let mut fields = ToolCallUpdateFields::new()
      .title(title(&name, parsed.as_ref()))
      .kind(kind(&name))
      .status(ToolCallStatus::Pending);
    if let Some(args) = &parsed {
      fields = fields
        .locations(any(locations(&name, args, &self.cwd)))
        .raw_input(args.clone());
    }
    self.update(id, parsed, fields);
  }

  /// Tell the editor about a call: all of it, the first time, and what has
  /// changed afterwards.
  fn update(&mut self, id: String, args: Option<serde_json::Value>, fields: ToolCallUpdateFields) {
    let update = ToolCallUpdate::new(id.clone(), fields);
    let told = self.calls.insert(id, args);
    self.tell(match told {
      Some(_) => SessionUpdate::ToolCallUpdate(update),
      None => match ToolCall::try_from(update) {
        Ok(call) => SessionUpdate::ToolCall(call),
        Err(_) => return,
      },
    });
  }
}

/// A list for an update to carry, when there is anything in it: an update
/// that carries one replaces what the editor holds, so an empty one would
/// only clear it.
fn any<T>(items: Vec<T>) -> Option<Vec<T>> {
  (!items.is_empty()).then_some(items)
}

fn report(result: Result<()>) {
  if let Err(err) = result {
    eprintln!("could not save the session: {err:#}");
  }
}

// ---------------------------------------------------------------- tool calls

/// The line a call is listed under: the tool, and what it acts on.
fn title(name: &str, args: Option<&serde_json::Value>) -> String {
  let Some(args) = args else {
    return name.to_string();
  };
  let summary = crate::ui::summarize_args(name, args);
  let first = summary.lines().next().unwrap_or_default();
  match summary.lines().nth(1) {
    Some(_) => format!("{name} {first} …"),
    None => format!("{name} {first}"),
  }
}

fn kind(name: &str) -> ToolKind {
  match name {
    "read" | "list_resources" | "read_resource" => ToolKind::Read,
    "edit" | "write" => ToolKind::Edit,
    "bash" => ToolKind::Execute,
    _ => ToolKind::Other,
  }
}

/// The path a call names, made absolute — which is how the editor wants it.
fn path(args: &serde_json::Value, cwd: &Path) -> Option<PathBuf> {
  let path = args.get("path")?.as_str()?;
  Some(tools::resolve(cwd, path))
}

/// The file a call reads or changes, for the editor to follow it to.
fn locations(name: &str, args: &serde_json::Value, cwd: &Path) -> Vec<ToolCallLocation> {
  if !matches!(name, "read" | "write" | "edit") {
    return Vec::new();
  }
  let Some(path) = path(args, cwd) else {
    return Vec::new();
  };
  let line = match name {
    "read" => args.get("offset").and_then(|offset| offset.as_u64()),
    _ => None,
  };
  vec![ToolCallLocation::new(path).line(line.and_then(|line| u32::try_from(line).ok()))]
}

/// What a call that changes a file is about to change, as the diff the
/// editor draws: each replacement of an edit, or the whole of what a write
/// puts in the file.
fn proposed(name: &str, args: &serde_json::Value, cwd: &Path) -> Vec<ToolCallContent> {
  let Some(path) = path(args, cwd) else {
    return Vec::new();
  };
  let text = |key: &str, value: &serde_json::Value| value.get(key).and_then(|v| v.as_str()).map(str::to_string);
  match name {
    "write" => text("content", args)
      .map(|content| vec![Diff::new(path, content).into()])
      .unwrap_or_default(),
    "edit" => args
      .get("edits")
      .and_then(|edits| edits.as_array())
      .into_iter()
      .flatten()
      .filter_map(|edit| {
        let new = text("newText", edit)?;
        Some(Diff::new(path.clone(), new).old_text(text("oldText", edit)).into())
      })
      .collect(),
    _ => Vec::new(),
  }
}

/// How a call ended, as the editor is told it: the change a file was given,
/// or what the tool said and the images it answered with.
fn finished(
  name: &str,
  args: Option<&serde_json::Value>,
  output: &str,
  images: &[Vec<u8>],
  is_error: bool,
  cwd: &Path,
) -> ToolCallUpdateFields {
  let change = match (is_error, args) {
    (false, Some(args)) => proposed(name, args, cwd),
    _ => Vec::new(),
  };
  let content = match change.is_empty() {
    false => change,
    true => {
      let mut content: Vec<ToolCallContent> = Vec::new();
      if !output.is_empty() {
        content.push(fenced(output).into());
      }
      content.extend(images.iter().map(|bytes| image(bytes).into()));
      content
    }
  };
  ToolCallUpdateFields::new()
    .status(match is_error {
      true => ToolCallStatus::Failed,
      false => ToolCallStatus::Completed,
    })
    .content(content)
    .raw_output(serde_json::Value::String(output.to_string()))
}

/// Text to be shown as it is, rather than read as Markdown: in a fence
/// longer than any run of backticks inside it.
fn fenced(text: &str) -> ContentBlock {
  let mut longest = 0;
  let mut run = 0;
  for c in text.chars() {
    run = if c == '`' { run + 1 } else { 0 };
    longest = longest.max(run);
  }
  let fence = "`".repeat(longest.max(2) + 1);
  format!("{fence}\n{}\n{fence}", text.trim_end_matches('\n')).into()
}

fn image(bytes: &[u8]) -> ContentBlock {
  let mime = image::guess_format(bytes).map_or("image/png", |format| format.to_mime_type());
  ContentBlock::Image(ImageContent::new(STANDARD.encode(bytes), mime))
}

// ---------------------------------------------------------------- replaying

/// A saved session, told as the updates it would have been told in as it
/// happened: what the user said, what the model said and thought, and each
/// call with how it ended.
fn replay(session: &Session, cwd: &Path) -> Vec<SessionUpdate> {
  let history = session.transcript(session.leaf());
  let results: HashMap<String, (String, Vec<Vec<u8>>)> = history
    .iter()
    .filter_map(|message| match message {
      Message::User { content } => Some(content),
      _ => None,
    })
    .flatten()
    .filter_map(|content| match content {
      UserContent::ToolResult(result) => Some((result.call.to_string(), crate::images::split(&result.content))),
      _ => None,
    })
    .collect();

  let mut updates = Vec::new();
  for message in &history {
    // What a compaction summarized is told in full above it.
    if crate::compaction::is_summary(message) {
      continue;
    }
    match message {
      Message::User { content } => {
        if let Some(text) = crate::session::user_text(message) {
          updates.push(SessionUpdate::UserMessageChunk(ContentChunk::new(text.into())));
        }
        for part in content {
          if let UserContent::Image(picture) = part
            && let Some(bytes) = crate::images::source_bytes(&picture.data)
          {
            updates.push(SessionUpdate::UserMessageChunk(ContentChunk::new(image(&bytes))));
          }
        }
      }
      Message::Assistant(AssistantMessage { content, .. }) => {
        for part in content {
          match part {
            AssistantContent::Text(text) => {
              updates.push(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                text.text.clone().into(),
              )));
            }
            AssistantContent::Reasoning(reasoning) => {
              if !reasoning.text.is_empty() {
                updates.push(SessionUpdate::AgentThoughtChunk(ContentChunk::new(
                  reasoning.text.clone().into(),
                )));
              }
            }
            AssistantContent::ToolCall(call) => {
              let (name, args, id) = (
                call.function.name.as_str(),
                &call.function.arguments_value(),
                call.id.to_string(),
              );
              let mut told = ToolCall::new(id.clone(), title(name, Some(args)))
                .kind(kind(name))
                .locations(locations(name, args, cwd))
                .raw_input(args.clone());
              match results.get(&id) {
                Some((output, images)) => {
                  let failed = session.outcome(&id).is_some_and(|outcome| outcome.failed);
                  told.update(finished(name, Some(args), output, images, failed, cwd));
                }
                None => told = told.status(ToolCallStatus::Failed),
              }
              updates.push(SessionUpdate::ToolCall(told));
            }
            _ => {}
          }
        }
      }
      Message::System { .. } => {}
    }
  }
  updates
}

#[cfg(test)]
mod tests {
  use super::*;
  use agent_client_protocol::schema::v1::{EmbeddedResource, ResourceLink, TextContent, TextResourceContents};

  #[test]
  fn a_prompt_reads_its_mentions_where_they_were_made() {
    let said = Said::from(
      vec![
        ContentBlock::Text(TextContent::new("look at ")),
        ContentBlock::ResourceLink(ResourceLink::new("main.rs", "file:///w/src/main.rs")),
        ContentBlock::Text(TextContent::new(" and ")),
        ContentBlock::Resource(EmbeddedResource::new(EmbeddedResourceResource::TextResourceContents(
          TextResourceContents::new("fn x() {}\n", "file:///w/src/x.rs"),
        ))),
      ],
      true,
    );
    assert_eq!(
      said.text,
      "look at @/w/src/main.rs and @/w/src/x.rs\n\n<context ref=\"file:///w/src/x.rs\">\nfn x() {}\n</context>"
    );
    assert!(said.images.is_empty());
  }

  #[test]
  fn an_image_the_model_cannot_see_is_said_rather_than_sent() {
    let block = || ContentBlock::Image(ImageContent::new("aGk=", "image/png"));
    assert_eq!(Said::from(vec![block()], true).images.len(), 1);
    let blind = Said::from(vec![block()], false);
    assert!(blind.images.is_empty());
    assert_eq!(blind.text, "[an image the model cannot see]");
  }

  #[test]
  fn output_is_fenced_past_its_own_backticks() {
    let ContentBlock::Text(text) = fenced("a ```` b\n") else {
      panic!("text");
    };
    assert_eq!(text.text, "`````\na ```` b\n`````");
  }

  #[test]
  fn an_edit_is_shown_as_the_replacements_it_makes() {
    let args = serde_json::json!({
      "path": "src/a.rs",
      "edits": [{"oldText": "one", "newText": "two"}],
    });
    let content = proposed("edit", &args, Path::new("/w"));
    let [ToolCallContent::Diff(diff)] = content.as_slice() else {
      panic!("one diff: {content:?}");
    };
    assert_eq!(diff.path, Path::new("/w/src/a.rs"));
    assert_eq!(diff.old_text.as_deref(), Some("one"));
    assert_eq!(diff.new_text, "two");
  }

  #[cfg(feature = "mcp")]
  #[test]
  fn a_command_the_editor_names_is_quoted_into_a_line_of_shell() {
    assert_eq!(quoted("npx"), "npx");
    assert_eq!(quoted("it's here"), r"'it'\''s here'");
    assert_eq!(quoted(""), "''");
  }
}
