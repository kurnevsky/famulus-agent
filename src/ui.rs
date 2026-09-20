//! Ratatui front-end: a scrolling transcript, a multi-line input box and a
//! one-line footer.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap};
use ratatui::{DefaultTerminal, Frame};
use ratatui_textarea::{CursorMove, TextArea, WrapMode};
use rig_core::completion::{Message, Usage};
use rig_core::message::{AssistantContent, ToolCall, ToolResult, ToolResultContent, UserContent};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::agent::{AgentEvent, Agents, start_compaction, start_run};
use crate::ask::{self, Dialog};
use crate::compaction::{SUMMARY_PREFIX, SUMMARY_SUFFIX, Settings};
use crate::session::{Node, NodeKind, Outcome, Session, SessionInfo, Store};
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32Str};

const MAX_INPUT_LINES: usize = 8;
const TOOL_OUTPUT_LINES: usize = 10;
const DIFF_LINES: usize = 30;
const REASONING_LINES: usize = 6;
/// Lines of an image shown before it folds, as every other block of a tool's
/// output folds.
const IMAGE_LINES: usize = 16;
/// The ceiling on a drawn image, which binds only for the very tall and
/// narrow — an image is drawn at the width it is given, and a 100x5000 one
/// would otherwise be a thousand lines for `Ctrl+O` to unfold.
const IMAGE_MAX_LINES: u16 = 80;
/// Reasoning text is indented under its `· thinking…` header.
const REASONING_INDENT: &str = "  ";
/// Cache tags, so two entry kinds holding the same text stay apart.
const ASSISTANT_KIND: u8 = 0;
const REASONING_KIND: u8 = 1;
const IMAGE_KIND: u8 = 2;
/// Transcript lines moved per mouse wheel notch.
const WHEEL_LINES: usize = 3;
/// How long the `auto` scrollbar stays visible after the last scroll.
const SCROLLBAR_HIDE_DELAY: Duration = Duration::from_millis(1000);
/// How close together two `Esc` presses count as one double press, as in pi.
const DOUBLE_ESC: Duration = Duration::from_millis(500);
/// What a tool call the user stopped is answered with, so the model knows it
/// did not simply go unheard.
const ABORTED: &str = "Aborted by the user.";
/// Rows an overlay list moves per `PageUp`/`PageDown`.
const OVERLAY_PAGE: usize = 10;

/// Transcript scrollbar behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ScrollbarMode {
  /// Shown while scrolling an overflowing transcript, hidden a second later
  Auto,
  /// Always reserve a column for it
  Always,
  /// Never shown
  Hidden,
}
const SPINNER: [&str; 4] = ["◐", "◓", "◑", "◒"];

/// Which session to open at startup.
pub enum SessionStart {
  New,
  /// The most recently modified session, if any.
  Continue,
  /// Open the session picker.
  Resume,
  Path(PathBuf),
}

/// Startup configuration for the UI.
pub struct Options {
  pub model: String,
  pub cwd: PathBuf,
  pub settings: Settings,
  pub scrollbar: ScrollbarMode,
  pub store: Option<Store>,
  pub start: SessionStart,
  /// What happened before the terminal existed — which MCP servers came up,
  /// and what went wrong with the rest — said in the transcript, since there
  /// is nowhere else left to say it.
  pub notes: Vec<String>,
  /// How many MCP servers came up, and how many tools they brought.
  pub mcp: (usize, usize),
  /// Whether a question the model asks rings the terminal.
  pub bell: bool,
}

/// Slash commands offered by the `/` popup: name, description, takes an argument.
const COMMANDS: &[(&str, &str, bool)] = &[
  ("compact", "Manually compact the session context", false),
  ("continue", "Resume the loop without a new message", false),
  ("fork", "Start a new session from an earlier message", false),
  ("name", "Set session display name", true),
  ("new", "Start a new session", false),
  ("resume", "Resume a different session", false),
  ("session", "Show session info and stats", false),
  ("tree", "Go back to an earlier point (or press Esc twice)", false),
  ("quit", "Quit fa", false),
];
/// Rows shown in the command popup.
const COMPLETION_ROWS: usize = 5;

/// One popup row: index into `COMMANDS` plus the matched character positions.
struct Match {
  index: usize,
  highlights: Vec<u32>,
}

/// The `/` command popup, best match first.
struct Completion {
  items: Vec<Match>,
  selected: usize,
}

/// The run in flight, kept so an abort can still record what the user saw.
struct InFlight {
  /// Trailing history messages this run re-sent as its prompt (`/continue`).
  /// They come back in `Done`, where they are dropped rather than stored
  /// twice; an abort leaves them where they already are.
  resumed: usize,
  /// The new user message, absent for `/continue`.
  prompt: Option<Message>,
  /// Turns this run has already finished.
  ///
  /// A run only hands its transcript back when it reaches the end, so a run
  /// that is stopped part-way would otherwise leave nothing behind — while
  /// every file its tools touched stays touched. Kept as it happens, so an
  /// abort keeps the work rather than only the memory of it.
  done: Vec<Message>,
  /// What the model has said in the turn being streamed.
  text: String,
  /// The calls it has made in that turn.
  calls: Vec<AssistantContent>,
  /// Calls made and not yet answered, by the id each will be answered with
  /// and the tool each ran. Held by id rather than counted, so a result is
  /// matched to its own call however many are in flight and whatever order
  /// they come back in.
  pending: Vec<(String, String)>,
}

impl InFlight {
  fn new(prompt: Option<Message>, resumed: usize) -> Self {
    Self {
      resumed,
      prompt,
      done: Vec::new(),
      text: String::new(),
      calls: Vec::new(),
      pending: Vec::new(),
    }
  }

  fn said(&mut self, delta: &str) {
    self.text.push_str(delta);
  }

  fn called(&mut self, call: &str, name: &str, args: serde_json::Value) {
    self.calls.push(AssistantContent::tool_call(call, name, args));
    self.pending.push((call.to_string(), name.to_string()));
  }

  /// A result closes the turn that asked for it: what the model said and the
  /// call it made become a message, and the answer follows as its own.
  fn answered(&mut self, call: &str, name: &str, output: String) {
    self.turn();
    self.pending.retain(|(pending, _)| pending != call);
    self.done.push(Message::User {
      content: vec![UserContent::tool_result(
        call,
        name,
        vec![ToolResultContent::text(output)],
      )],
    });
  }

  /// Close off the turn being streamed, if it said anything at all.
  fn turn(&mut self) {
    let text = std::mem::take(&mut self.text);
    let calls = std::mem::take(&mut self.calls);
    let mut content: Vec<AssistantContent> = Vec::new();
    if !text.trim().is_empty() {
      content.push(AssistantContent::text(text));
    }
    content.extend(calls);
    if !content.is_empty() {
      self.done.push(Message::Assistant { id: None, content });
    }
  }

  /// Everything this run got through, for a run that will not report it
  /// itself.
  ///
  /// A call it was stopped in the middle of is answered as interrupted. The
  /// model asked for it, so the transcript owes an answer — and a call left
  /// hanging is a conversation no provider will take back, which would leave
  /// the session unable to carry on from what it just kept.
  fn recovered(mut self) -> Vec<Message> {
    self.turn();
    for (call, name) in std::mem::take(&mut self.pending) {
      self.done.push(Message::User {
        content: vec![UserContent::tool_result(
          call,
          name,
          vec![ToolResultContent::text(ABORTED)],
        )],
      });
    }
    let mut messages: Vec<Message> = self.prompt.into_iter().collect();
    messages.extend(self.done);
    messages
  }
}

/// A list drawn over the transcript. Moving through one and dismissing it are
/// the same whichever list it is; only the rows and what `Enter` does differ.
struct Overlay {
  list: OverlayList,
  selected: usize,
}

enum OverlayList {
  /// `/resume`: every saved session, most recent first.
  Sessions(Vec<SessionInfo>),
  /// `/tree`: every point this session can go back to, oldest first.
  Tree(Vec<Point>),
  /// `/fork`: the prompts, to start a new session from one of them.
  Fork(Vec<Point>),
  /// What the `ask` tool put to the user, and the channel the answer goes
  /// back down. The dialog keeps its own cursor, so `Overlay::selected` says
  /// nothing about this one.
  Question {
    dialog: Box<Dialog>,
    /// Taken when the questionnaire is answered. Dropping it unanswered is
    /// what tells the tool the user walked away.
    reply: Option<oneshot::Sender<ask::Outcome>>,
  },
}

/// A point the conversation can be moved to.
struct Point {
  /// Where the conversation would end after going there; `None` is before the
  /// first message.
  leaf: Option<String>,
  /// A prompt that going there takes back out of the history and returns to
  /// the input box; `None` for a point kept as the conversation's new end.
  text: Option<String>,
  /// How the row reads in the list.
  label: String,
  /// Branch points crossed to reach it, which is how far the row is indented.
  depth: usize,
  /// Messages the conversation would hold after going there.
  len: usize,
  /// This is where the session already is.
  here: bool,
}

impl Overlay {
  fn points(&self) -> &[Point] {
    match &self.list {
      OverlayList::Tree(points) | OverlayList::Fork(points) => points,
      OverlayList::Sessions(_) | OverlayList::Question { .. } => &[],
    }
  }

  fn len(&self) -> usize {
    match &self.list {
      OverlayList::Sessions(sessions) => sessions.len(),
      OverlayList::Question { .. } => 0,
      _ => self.points().len(),
    }
  }

  fn title(&self) -> &'static str {
    match &self.list {
      OverlayList::Sessions(_) => " Resume session — ↑↓ select · Enter resume · Esc cancel ",
      OverlayList::Tree(_) => " Tree — ↑↓ PgUp/PgDn select · Enter go there · Esc cancel ",
      OverlayList::Fork(_) => " Fork — ↑↓ PgUp/PgDn select · Enter fork · Esc cancel ",
      // The dialog says which keys do what along its own bottom, where the
      // answer to that changes with the question.
      OverlayList::Question { .. } => " The model is asking ",
    }
  }

  /// The questionnaire this overlay is, if it is one.
  fn dialog(&mut self) -> Option<(&mut Dialog, &mut Option<oneshot::Sender<ask::Outcome>>)> {
    match &mut self.list {
      OverlayList::Question { dialog, reply } => Some((dialog, reply)),
      _ => None,
    }
  }
}

/// A tool call whose arguments are still arriving, shown at the end of the
/// transcript so the command can be read as the model writes it.
///
/// It is replaced by a real entry when the call starts running, which is when
/// `UiHook` reports it — not when the last of its text arrives.
struct Writing {
  id: String,
  name: String,
  /// The JSON so far, usually not yet parseable.
  args: String,
}

enum Entry {
  User(String),
  Assistant(String),
  Reasoning(String),
  ToolCall {
    name: String,
    summary: String,
    /// The call this is, so its output finds it again when several tools are
    /// in flight at once and finish in whatever order they finish.
    call: String,
    /// What a `write` put in the file. A write leaves nothing behind but a
    /// sentence, and the file it wrote is worth more than the sentence — so
    /// the transcript takes it from the asking, where it already is.
    wrote: Option<String>,
    started: Instant,
  },
  ToolResult {
    name: String,
    output: String,
    /// Images the tool answered with, drawn under its output as half-blocks.
    /// The bytes as the tool produced them, since what they are drawn as
    /// depends on how wide the transcript is when they are drawn.
    images: Vec<Vec<u8>>,
    is_error: bool,
    call: String,
    /// Still streaming; `output` is a live snapshot.
    running: bool,
    /// Numbered diff for `edit`, rendered instead of `output`.
    diff: Option<String>,
    started: Instant,
    took: Option<Duration>,
  },
  Error(String),
  Info(String),
  /// Compaction checkpoint that replaced older history.
  Summary(String),
}

/// The prompts this conversation has been sent, and where `Up` and `Down`
/// have walked back to among them.
///
/// Not a store of its own: the session already holds every prompt it was
/// told, so resuming one brings its prompts back with it and a new one starts
/// empty — the walk is over what is on screen, not over everything ever
/// typed. Slash commands are kept here as they are sent, since they never
/// reach the session and are worth recalling until it is left.
#[derive(Default)]
struct Prompts {
  /// Oldest first, never two of the same in a row.
  sent: Vec<String>,
  /// The walk `Up` started, while one is under way.
  walk: Option<Walk>,
}

/// Where a walk back through the prompts has got to.
struct Walk {
  /// Which of `sent` is in the input box.
  at: usize,
  /// What was in the box when the walk began, for `Down` to hand back.
  draft: String,
}

impl Prompts {
  /// The prompts of a session, in the order it was told them.
  fn of(session: &Session) -> Self {
    let mut prompts = Self::default();
    for message in &session.history {
      if let Some(text) = crate::session::user_text(message) {
        prompts.add(text);
      }
    }
    prompts
  }

  /// Remember a prompt, and end any walk: it has been sent, so the box is
  /// the user's own again.
  fn add(&mut self, text: String) {
    self.walk = None;
    if text.is_empty() || self.sent.last() == Some(&text) {
      return;
    }
    self.sent.push(text);
  }

  fn walking(&self) -> bool {
    self.walk.is_some()
  }

  fn stop(&mut self) {
    self.walk = None;
  }

  /// The prompt before the one in the box, `draft` being what is in it now.
  /// `None` at the oldest, which leaves the walk standing where it is.
  fn previous(&mut self, draft: &str) -> Option<String> {
    let (at, draft) = match self.walk.take() {
      Some(walk) => match walk.at.checked_sub(1) {
        Some(at) => (at, walk.draft),
        None => {
          self.walk = Some(walk);
          return None;
        }
      },
      None => (self.sent.len().checked_sub(1)?, draft.to_string()),
    };
    let text = self.sent[at].clone();
    self.walk = Some(Walk { at, draft });
    Some(text)
  }

  /// The prompt after it, and past the newest the draft the walk began from.
  /// `None` when there is no walk to come back from.
  fn next(&mut self) -> Option<String> {
    let mut walk = self.walk.take()?;
    match self.sent.get(walk.at + 1).cloned() {
      Some(text) => {
        walk.at += 1;
        self.walk = Some(walk);
        Some(text)
      }
      None => Some(walk.draft),
    }
  }
}

pub struct App {
  agents: Agents,
  settings: Settings,
  scrollbar: ScrollbarMode,
  /// When the transcript was last scrolled, for the `auto` scrollbar.
  last_scroll: Option<Instant>,
  model: String,
  cwd: PathBuf,
  store: Option<Store>,
  /// How many MCP servers came up, and how many tools they brought, for the
  /// footer to say a session has more than the five it was built with.
  mcp: (usize, usize),
  /// Whether a question the model asks rings the terminal.
  bell: bool,
  /// The conversation: history plus its on-disk file.
  session: Session,
  overlay: Option<Overlay>,
  matcher: Matcher,
  completion: Option<Completion>,
  /// Esc closed the popup; stay closed until the input changes.
  completion_dismissed: bool,
  /// The last bare `Esc`, for spotting the second of a double press.
  last_escape: Option<Instant>,
  entries: Vec<Entry>,
  input: TextArea<'static>,
  /// What `Up` and `Down` walk back through from the input box.
  prompts: Prompts,
  tx: mpsc::UnboundedSender<AgentEvent>,
  run: Option<JoinHandle<()>>,
  /// The background task in `run` is a compaction rather than a model turn.
  compacting: bool,
  /// That compaction is making room for a run which stopped short of the
  /// context window, and which carries on once there is some.
  resuming: bool,
  /// The run in flight, kept for abort recovery.
  in_flight: Option<InFlight>,
  /// Tool calls the model is still writing, oldest first.
  writing: Vec<Writing>,
  /// How this run's tool calls went, by the id the transcript names them
  /// with. Handed to the session so a reload draws them the same; the
  /// transcript itself records neither the verdict nor the diff.
  outcomes: HashMap<String, Outcome>,
  queued: VecDeque<String>,
  /// Transcript position. `None` follows new output at the bottom; `Some`
  /// is a fixed offset from the top, so appended text does not move the view.
  anchor: Option<usize>,
  /// Offset and maximum offset used by the last draw, for relative scrolling.
  view: (usize, usize),
  /// Ctrl+T shows reasoning blocks in full instead of their last few lines.
  expand_thinking: bool,
  /// Ctrl+O shows tool output in full instead of the preview.
  expand_tools: bool,
  /// Rendered markdown, keyed by message text and width rather than by entry,
  /// so reloading a session or compacting cannot serve another entry's lines.
  /// Rebuilt each draw by moving live entries across, which evicts the rest.
  markdown: HashMap<(u64, u16, bool), Vec<Line<'static>>>,
  usage: Usage,
  /// Size of the last completion request, for the footer. `None` until a
  /// call has come back: after a compaction the figure is hidden rather than
  /// left saying what the context no longer holds.
  context_tokens: Option<u64>,
  tick: usize,
  quit: bool,
}

impl App {
  pub fn new(agents: Agents, tx: mpsc::UnboundedSender<AgentEvent>, options: Options) -> Self {
    let Options {
      model,
      cwd,
      settings,
      scrollbar,
      store,
      start,
      notes,
      mcp,
      bell,
    } = options;
    let mut input = TextArea::default();
    input.set_cursor_line_style(Style::default());
    // Not a list of commands: `/` opens one that is complete and filters
    // itself, where three names picked out here only ever go stale.
    input.set_placeholder_text("Ask anything. Enter sends, Alt+Enter a newline, / commands, Ctrl+C quits.");
    // The terminal's own grey rather than a dimmed foreground, which some
    // terminals ignore and others render as the text colour proper.
    input.set_placeholder_style(Style::default().fg(Color::DarkGray));
    input.set_wrap_mode(WrapMode::WordOrGlyph);
    let session = Session::new(store.as_ref(), &cwd, &model);
    let mut app = Self {
      agents,
      settings,
      scrollbar,
      last_scroll: None,
      model,
      mcp,
      bell,
      cwd,
      store,
      session,
      overlay: None,
      matcher: Matcher::new(Config::DEFAULT),
      completion: None,
      completion_dismissed: false,
      last_escape: None,
      entries: Vec::new(),
      input,
      prompts: Prompts::default(),
      tx,
      run: None,
      compacting: false,
      resuming: false,
      in_flight: None,
      writing: Vec::new(),
      outcomes: HashMap::new(),
      queued: VecDeque::new(),
      anchor: None,
      view: (0, 0),
      expand_thinking: false,
      expand_tools: false,
      markdown: HashMap::new(),
      usage: Usage::new(),
      context_tokens: None,
      tick: 0,
      quit: false,
    };
    match start {
      SessionStart::New => {}
      SessionStart::Continue => match app.store.as_ref().and_then(Store::most_recent) {
        Some(info) => app.load_session(&info.path),
        None => app.entries.push(Entry::Info("No previous session to continue.".into())),
      },
      SessionStart::Resume => app.open_picker(),
      SessionStart::Path(path) => app.load_session(&path),
    }
    // What happened before the terminal existed, said before whatever the
    // session itself had to say, because that is the order it happened in.
    // In front of the entries rather than after them: resuming a session
    // draws its transcript from the history, which would otherwise leave the
    // servers reported under a conversation that predates them.
    app.entries.splice(0..0, notes.into_iter().map(Entry::Info));
    app
  }

  pub async fn run(
    mut self,
    terminal: &mut DefaultTerminal,
    mut rx: mpsc::UnboundedReceiver<AgentEvent>,
  ) -> Result<()> {
    // Terminal input is blocking; read it on a dedicated thread.
    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<Event>();
    std::thread::spawn(move || {
      while let Ok(ev) = event::read() {
        if input_tx.send(ev).is_err() {
          break;
        }
      }
    });
    let mut ticker = tokio::time::interval(Duration::from_millis(120));

    loop {
      terminal.draw(|f| self.draw(f))?;
      tokio::select! {
          Some(ev) = input_rx.recv() => {
              self.handle_terminal(ev);
              while let Ok(ev) = input_rx.try_recv() {
                  self.handle_terminal(ev);
              }
          }
          Some(ev) = rx.recv() => {
              self.handle_agent(ev);
              while let Ok(ev) = rx.try_recv() {
                  self.handle_agent(ev);
              }
          }
          _ = ticker.tick(), if self.run.is_some() || self.scrollbar_fading() => {
              self.tick = self.tick.wrapping_add(1);
          }
      }
      if self.quit {
        self.abort();
        return Ok(());
      }
    }
  }

  // ------------------------------------------------------------ input

  fn handle_terminal(&mut self, ev: Event) {
    let key = match ev {
      Event::Key(key) => key,
      Event::Mouse(mouse) => {
        match mouse.kind {
          MouseEventKind::ScrollUp => self.scroll_by(WHEEL_LINES as isize),
          MouseEventKind::ScrollDown => self.scroll_by(-(WHEEL_LINES as isize)),
          _ => {}
        }
        return;
      }
      _ => return,
    };
    if key.kind != KeyEventKind::Press {
      return;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    if self.overlay.as_mut().is_some_and(|o| o.dialog().is_some()) {
      self.handle_question_key(key, ctrl);
      return;
    }
    if self.overlay.is_some() {
      self.handle_overlay_key(key.code, ctrl);
      return;
    }
    if self.completion.is_some() && self.handle_completion_key(key.code, ctrl) {
      return;
    }
    match (key.code, ctrl) {
      (KeyCode::Char('c'), true) => {
        if self.run.is_some() {
          self.abort();
        } else {
          self.quit = true;
        }
      }
      (KeyCode::Char('d'), true) if self.input.is_empty() => self.quit = true,
      (KeyCode::Char('t'), true) => {
        self.expand_thinking = !self.expand_thinking;
        // Expanding moves everything below the block, so go back to following
        // the bottom rather than leaving the reader mid-paragraph. Only the
        // view changes: the conversation is the same one, and so is anything
        // waiting to be said to it.
        self.anchor = None;
      }
      (KeyCode::Char('o'), true) => {
        self.expand_tools = !self.expand_tools;
        self.anchor = None;
      }
      (KeyCode::Esc, _) if self.run.is_some() => self.abort(),
      // Esc on its own has nothing to do once there is no run to stop, so a
      // second one within the window opens the tree — pi's shortcut, and its
      // default action. Only from an empty box, where Esc cannot be meant for
      // the text.
      (KeyCode::Esc, _) if self.input_is_blank() => {
        let now = Instant::now();
        let again = self.last_escape.is_some_and(|last| now - last < DOUBLE_ESC);
        self.last_escape = (!again).then_some(now);
        if again {
          self.open_points(false);
        }
      }
      // Alt+Up takes the last message still waiting back out of the queue,
      // to be fixed and sent again. Only from an empty box, where it cannot
      // land on top of something half-typed.
      (KeyCode::Up, _) if key.modifiers.contains(KeyModifiers::ALT) && self.input_is_blank() => self.unqueue(),
      // Up and Down walk back through the prompts already sent, but only from
      // the ends of the box: inside a prompt of several lines they are still
      // the cursor's, which is what the walk hands back.
      (KeyCode::Up, false) if key.modifiers.is_empty() => self.walk_back(),
      (KeyCode::Down, false) if key.modifiers.is_empty() => self.walk_forward(),
      (KeyCode::PageUp, _) => self.scroll_by(10),
      (KeyCode::PageDown, _) => self.scroll_by(-10),
      (KeyCode::Enter, _) if is_newline(&key) => {
        self.prompts.stop();
        self.input.insert_newline();
        self.refresh_completion();
      }
      (KeyCode::Char('j'), true) => {
        self.prompts.stop();
        self.input.insert_newline();
        self.refresh_completion();
      }
      (KeyCode::Enter, _) => self.submit(),
      _ => {
        if self.input.input(key) {
          self.completion_dismissed = false;
          // A recalled prompt that has been edited is the user's text now,
          // and Down is no longer a way back out of it.
          self.prompts.stop();
        }
        self.refresh_completion();
      }
    }
  }

  /// Keys consumed by the `/` popup. Returns false to let the key fall
  /// through to normal editing (which then re-filters the list).
  fn handle_completion_key(&mut self, code: KeyCode, ctrl: bool) -> bool {
    match code {
      KeyCode::Esc => {
        self.completion = None;
        self.completion_dismissed = true;
      }
      KeyCode::Up => self.move_completion(-1),
      KeyCode::Char('p') if ctrl => self.move_completion(-1),
      KeyCode::Down => self.move_completion(1),
      KeyCode::Char('n') if ctrl => self.move_completion(1),
      KeyCode::Tab | KeyCode::Enter => {
        let Some(c) = self.completion.take() else {
          return false;
        };
        let Some(index) = c.items.get(c.selected).map(|m| m.index) else {
          return false;
        };
        let (name, _, takes_arg) = COMMANDS[index];
        self.set_input(&format!("/{name}{}", if takes_arg { " " } else { "" }));
        // The completed command is exact; keep the popup closed until
        // the user edits the text again.
        self.completion_dismissed = true;
      }
      _ => return false,
    }
    true
  }

  fn move_completion(&mut self, delta: isize) {
    if let Some(c) = &mut self.completion {
      let len = c.items.len().max(1);
      c.selected = (c.selected as isize + delta).rem_euclid(len as isize) as usize;
    }
  }

  /// Show the command popup while the input is a single `/word` prefix.
  fn refresh_completion(&mut self) {
    let lines = self.input.lines();
    let query = match lines {
      [line] if line.starts_with('/') && !line.contains(char::is_whitespace) => &line[1..],
      _ => {
        self.completion = None;
        return;
      }
    };
    if self.completion_dismissed {
      return;
    }
    let items = filter_commands(&mut self.matcher, query);
    if items.is_empty() {
      self.completion = None;
      return;
    }
    let selected = self
      .completion
      .as_ref()
      .and_then(|c| c.items.get(c.selected).map(|m| m.index))
      .and_then(|prev| items.iter().position(|m| m.index == prev))
      .unwrap_or(0);
    self.completion = Some(Completion { items, selected });
  }

  /// A key while the model's questionnaire is open. It is the dialog's to
  /// answer — the whole keyboard belongs to it while it is up, which is what
  /// makes typing an answer of one's own possible at all.
  ///
  /// Ctrl+C is the exception: the run the question came from is still in
  /// flight, and stopping it is what that key means everywhere else in fa.
  fn handle_question_key(&mut self, key: KeyEvent, ctrl: bool) {
    if ctrl && key.code == KeyCode::Char('c') {
      self.abort();
      return;
    }
    let Some((dialog, reply)) = self.overlay.as_mut().and_then(Overlay::dialog) else {
      return;
    };
    let Some(outcome) = dialog.key(key) else {
      return;
    };
    if let Some(reply) = reply.take() {
      let _ = reply.send(outcome);
    }
    self.overlay = None;
  }

  fn handle_overlay_key(&mut self, code: KeyCode, ctrl: bool) {
    let len = self.overlay.as_ref().map_or(0, Overlay::len);
    match code {
      KeyCode::Esc | KeyCode::Char('q') => self.overlay = None,
      KeyCode::Char('c') if ctrl => self.quit = true,
      KeyCode::Up | KeyCode::Char('k') => {
        if let Some(o) = &mut self.overlay {
          o.selected = o.selected.saturating_sub(1);
        }
      }
      KeyCode::Down | KeyCode::Char('j') => {
        if let Some(o) = &mut self.overlay {
          o.selected = (o.selected + 1).min(len.saturating_sub(1));
        }
      }
      // A tree of every tool result is long enough to need more than one row
      // at a time.
      KeyCode::PageUp | KeyCode::Home => {
        if let Some(o) = &mut self.overlay {
          o.selected = if code == KeyCode::Home {
            0
          } else {
            o.selected.saturating_sub(OVERLAY_PAGE)
          };
        }
      }
      KeyCode::PageDown | KeyCode::End => {
        if let Some(o) = &mut self.overlay {
          let last = len.saturating_sub(1);
          o.selected = if code == KeyCode::End {
            last
          } else {
            (o.selected + OVERLAY_PAGE).min(last)
          };
        }
      }
      KeyCode::Enter => {
        let Some(Overlay { list, selected }) = self.overlay.take() else {
          return;
        };
        match list {
          OverlayList::Sessions(sessions) => {
            if let Some(info) = sessions.get(selected) {
              self.load_session(&info.path.clone());
            }
          }
          OverlayList::Tree(mut points) if selected < points.len() => self.go_to(points.remove(selected)),
          OverlayList::Fork(mut points) if selected < points.len() => self.fork_to(points.remove(selected)),
          _ => {}
        }
      }
      _ => {}
    }
  }

  /// Scroll the transcript up (positive) or down (negative) by `lines`.
  fn scroll_by(&mut self, lines: isize) {
    let (offset, max_scroll) = self.view;
    let target = offset.saturating_add_signed(-lines);
    // Reaching the bottom re-attaches to the live end of the transcript.
    self.anchor = (target < max_scroll).then_some(target);
    self.last_scroll = Some(Instant::now());
  }

  /// The `auto` scrollbar is showing and will need a redraw to disappear.
  fn scrollbar_fading(&self) -> bool {
    self.scrollbar == ScrollbarMode::Auto && self.last_scroll.is_some_and(|t| t.elapsed() < SCROLLBAR_HIDE_DELAY)
  }

  /// Replace whatever is in the input box, with text from somewhere other
  /// than the prompts being walked through — which is what ends the walk.
  fn set_input(&mut self, text: &str) {
    self.prompts.stop();
    self.fill_input(text);
  }

  /// The replacement itself, without disturbing the yank buffer — the user's
  /// own cut text is theirs, not ours to overwrite.
  fn fill_input(&mut self, text: &str) {
    self.input.select_all();
    self.input.cut();
    self.input.set_yank_text("");
    self.input.insert_str(text);
  }

  /// `Up`: the cursor while it has a line above it, and the prompts already
  /// sent once it is on the first one. From the start of that line, so that
  /// reaching the top of something being typed is not also leaving it — the
  /// first press goes there, a second walks back.
  fn walk_back(&mut self) {
    let cursor = self.input.screen_cursor();
    if cursor.row > 0 {
      self.input.move_cursor(CursorMove::Up);
      return;
    }
    if cursor.col > 0 && !self.prompts.walking() && !self.input_is_blank() {
      self.input.move_cursor(CursorMove::Head);
      return;
    }
    let draft = self.input.lines().join("\n");
    if let Some(text) = self.prompts.previous(&draft) {
      self.fill_input(&text);
      // At the top of the prompt it just handed back, so that holding Up
      // keeps walking rather than reading down the one it landed on.
      self.input.move_cursor(CursorMove::Jump(0, 0));
    }
  }

  /// `Down`: the way back, a prompt at a time from the last line of the box,
  /// ending at whatever was being typed when the walk began.
  fn walk_forward(&mut self) {
    let row = self.input.screen_cursor().row;
    self.input.move_cursor(CursorMove::Down);
    if !self.prompts.walking() || self.input.screen_cursor().row != row {
      return;
    }
    if let Some(text) = self.prompts.next() {
      self.fill_input(&text);
    }
  }

  /// Nothing to send: empty, or only whitespace.
  fn input_is_blank(&self) -> bool {
    self.input.lines().iter().all(|line| line.trim().is_empty())
  }

  fn submit(&mut self) {
    let text = self.input.lines().join("\n").trim().to_string();
    if text.is_empty() {
      return;
    }
    // Everything sent is worth recalling, commands included — the session
    // will only remember the prompts.
    self.prompts.add(text.clone());
    self.set_input("");
    self.completion = None;
    self.anchor = None;

    if self.run.is_some() && !matches!(text.as_str(), "/quit" | "/new") {
      self.queued.push_back(text);
      self.waiting();
      return;
    }
    self.dispatch(text);
  }

  /// Run a prompt or slash command now.
  fn dispatch(&mut self, text: String) {
    match text.as_str() {
      "/quit" => self.quit = true,
      "/new" => self.new_session(),
      "/compact" => self.compact(),
      "/continue" => self.continue_run(),
      "/resume" => {
        if self.run.is_some() {
          self.entries.push(Entry::Info(
            "Finish or abort the current run before resuming another session.".into(),
          ));
        } else {
          self.open_picker();
        }
      }
      "/tree" | "/fork" => {
        if self.run.is_some() {
          self
            .entries
            .push(Entry::Info(format!("Finish or abort the current run before {text}.")));
        } else {
          self.open_points(text == "/fork");
        }
      }
      "/session" => self.session_info(),
      t if t == "/name" || t.starts_with("/name ") => {
        let name = t["/name".len()..].trim();
        if name.is_empty() {
          let current = self.session.name.clone();
          self.entries.push(Entry::Info(match current {
            Some(n) => format!("Session name: {n}"),
            None => "Session has no name. Use /name <name> to set one.".into(),
          }));
        } else {
          let result = self.session.rename(name);
          self.report(result);
          self.entries.push(Entry::Info(format!("Session named \"{name}\".")));
        }
      }
      _ => self.start(text),
    }
  }

  // ------------------------------------------------------------ sessions

  /// Show a persistence failure once; the conversation continues in memory.
  fn report(&mut self, result: anyhow::Result<()>) {
    if let Err(err) = result {
      self.entries.push(Entry::Error(format!("Session not saved: {err:#}")));
    }
  }

  /// Leave behind everything that belonged to the conversation being left:
  /// what it cost, where it was scrolled to, and anything typed at it that
  /// has not been sent.
  fn reset_conversation(&mut self) {
    self.queued.clear();
    self.waiting();
    self.overflowed();
    self.resuming = false;
    self.usage = Usage::new();
    self.context_tokens = None;
    self.anchor = None;
  }

  fn new_session(&mut self) {
    self.abort();
    self.session = Session::new(self.store.as_ref(), &self.cwd, &self.model);
    self.entries.clear();
    self.prompts = Prompts::default();
    self.reset_conversation();
    self.entries.push(Entry::Info("New session.".into()));
  }

  fn load_session(&mut self, path: &Path) {
    match Session::load(path) {
      Ok(mut session) => {
        if self.store.is_none() {
          session.disable_persistence();
        }
        self.entries = entries_from_history(&session);
        let title = session.name.clone().unwrap_or_else(|| session.id.clone());
        self.entries.push(Entry::Info(format!(
          "Resumed session {title} ({} messages).",
          session.history.len()
        )));
        self.session = session;
        // The prompts of the conversation being resumed are the ones Up
        // walks back through in it.
        self.prompts = Prompts::of(&self.session);
        self.reset_conversation();
      }
      Err(err) => self
        .entries
        .push(Entry::Error(format!("Could not resume session: {err:#}"))),
    }
  }

  fn open_picker(&mut self) {
    let Some(store) = &self.store else {
      self
        .entries
        .push(Entry::Info("Sessions are disabled (--no-session).".into()));
      return;
    };
    let sessions = store.list();
    if sessions.is_empty() {
      self.entries.push(Entry::Info("No saved sessions.".into()));
      return;
    }
    self.overlay = Some(Overlay {
      list: OverlayList::Sessions(sessions),
      selected: 0,
    });
  }

  /// List where the conversation can go, with where it is now selected.
  ///
  /// `/tree` offers every point; `/fork` only the prompts, since a fork is
  /// something you re-ask rather than a place you stand.
  fn open_points(&mut self, fork: bool) {
    let mut points = points(&self.session);
    if fork {
      points.retain(|point| point.text.is_some());
    }
    if points.is_empty() {
      self.entries.push(Entry::Info(match fork {
        true => "Nothing to fork from.".into(),
        false => "Nothing to go back to.".into(),
      }));
      return;
    }
    // Start where the session already is, so the way back is one step up.
    let selected = points.iter().rposition(|point| point.here).unwrap_or(points.len() - 1);
    let list = match fork {
      true => OverlayList::Fork(points),
      false => OverlayList::Tree(points),
    };
    self.overlay = Some(Overlay { list, selected });
  }

  /// Move this session's end to `point`.
  ///
  /// Nothing is dropped — what the conversation said down the path being left
  /// stays a branch of its own, and this list can walk back into it. The
  /// transcript is rebuilt from the history rather than edited alongside it,
  /// so the two cannot drift: what is on screen is what the model will be sent.
  fn go_to(&mut self, point: Point) {
    if point.here && point.text.is_none() {
      self.entries.push(Entry::Info("Already there.".into()));
      return;
    }
    let result = self.session.go_to(point.leaf.clone());
    self.report(result);
    self.show_point(&point, format!("Moved to {}.", messages(point.len)));
  }

  /// Start a new session holding the conversation up to `point`, leaving this
  /// one as it is — the branch you came from stays on disk, whole.
  fn fork_to(&mut self, point: Point) {
    match self.session.fork(point.leaf.as_deref()) {
      Ok(session) => {
        self.session = session;
        self.show_point(&point, format!("Forked, {} kept.", messages(point.len)));
      }
      Err(err) => self.entries.push(Entry::Error(format!("Could not fork: {err:#}"))),
    }
  }

  /// Redraw the transcript around a session that has just moved, and hand the
  /// user back the prompt they landed on.
  fn show_point(&mut self, point: &Point, note: String) {
    self.entries = entries_from_history(&self.session);
    // The conversation is another one now, down to which prompts are behind
    // the input box.
    self.prompts = Prompts::of(&self.session);
    self.entries.push(Entry::Info(note));
    // The token counts and the queue belonged to a conversation that is no
    // longer the one we are in.
    self.reset_conversation();
    if let Some(text) = &point.text {
      self.set_input(&text.clone());
    }
  }

  fn session_info(&mut self) {
    let s = &self.session;
    let file = match s.path() {
      Some(p) => p.display().to_string(),
      None if s.persistent() => "not created yet (saved on first message)".into(),
      None => "not saved (--no-session)".into(),
    };
    let info = format!(
      "Session {}\n  name: {}\n  file: {file}\n  cwd: {}\n  created: {}\n  model: {}\n  messages: {}\n  tokens this run: {}↑ {}↓",
      s.id,
      s.name.as_deref().unwrap_or("(none)"),
      s.cwd,
      s.created.format("%Y-%m-%d %H:%M"),
      s.model,
      s.history.len(),
      self.usage.input_tokens,
      self.usage.output_tokens
    );
    self.entries.push(Entry::Info(info));
  }

  /// Hand the newest waiting message back to the input box. Newest first, so
  /// pressing it again after sending walks back through the queue in the
  /// order the messages would have gone out, last one first.
  fn unqueue(&mut self) {
    let Some(text) = self.queued.pop_back() else { return };
    self.waiting();
    self.set_input(&text);
  }

  fn next_queued(&mut self) {
    if let Some(next) = self.queued.pop_front() {
      self.waiting();
      self.dispatch(next);
    }
  }

  /// Tell a run in flight whether anything is waiting behind it, so it can
  /// stop at its next turn rather than finish an answer to a question the
  /// user has already moved past.
  fn waiting(&self) {
    self
      .agents
      .waiting
      .store(!self.queued.is_empty(), std::sync::atomic::Ordering::Relaxed);
  }

  /// Whether the run that just ended found the context window full — and
  /// clear the mark, since answering it is this side's half of the bargain.
  fn overflowed(&self) -> bool {
    self.agents.overflow.swap(false, std::sync::atomic::Ordering::Relaxed)
  }

  fn compact(&mut self) {
    if self.session.history.is_empty() {
      self.entries.push(Entry::Info("Nothing to compact.".into()));
      self.resuming = false;
      self.next_queued();
      return;
    }
    self.entries.push(Entry::Info("Compacting context…".into()));
    self.compacting = true;
    self.run = Some(start_compaction(
      self.agents.summarizer.clone(),
      self.session.history.clone(),
      self.settings,
      self.tx.clone(),
    ));
  }

  fn start(&mut self, prompt: String) {
    self.entries.push(Entry::User(prompt.clone()));
    let prompt = Message::user(prompt);
    let handle = start_run(
      self.agents.agent.clone(),
      self.session.history.clone(),
      prompt.clone(),
      self.tx.clone(),
    );
    self.run = Some(handle);
    self.in_flight = Some(InFlight::new(Some(prompt), 0));
  }

  /// Run the model again with no new user message, to pick the loop back up
  /// where an abort or a compaction left it. The last history message becomes
  /// the prompt of the request, so the model sees exactly the conversation it
  /// already had: an unanswered user message is answered, and a half-written
  /// answer is continued.
  fn continue_run(&mut self) {
    let mut history = self.session.history.clone();
    let Some(prompt) = history.pop() else {
      self.entries.push(Entry::Info("Nothing to continue.".into()));
      self.next_queued();
      return;
    };
    let handle = start_run(self.agents.agent.clone(), history, prompt, self.tx.clone());
    self.run = Some(handle);
    self.in_flight = Some(InFlight::new(None, 1));
  }

  fn abort(&mut self) {
    // A question the run was waiting on has nobody left to answer to; closing
    // it drops the channel, which is how the tool hears that.
    self.close_question();
    let Some(handle) = self.run.take() else {
      return;
    };
    handle.abort();
    // A call the model had not finished writing was never run, and the
    // results of this turn are not the next turn's to record.
    self.writing.clear();
    self.outcomes.clear();
    // Esc is the end of it: a run stopped to be compacted is not resumed
    // afterwards, and the next one starts with the window weighed afresh.
    self.overflowed();
    self.resuming = false;
    if self.compacting {
      self.compacting = false;
      self.entries.push(Entry::Info("Compaction aborted.".into()));
    } else {
      self.recover_in_flight();
      self.finish_running_tool();
      self.entries.push(Entry::Info("Aborted.".into()));
    }
    // Esc stops everything, including what was waiting behind the run — but
    // it was typed, so it is kept in the transcript rather than dropped out
    // of sight.
    for text in std::mem::take(&mut self.queued) {
      self
        .entries
        .push(Entry::Info(format!("Not sent: {}", first_line(&text))));
    }
    self.waiting();
  }

  /// Take down the model's questionnaire, if one is up. What it had been
  /// asked is left unanswered, which the tool reads as a decline.
  fn close_question(&mut self) {
    if self.overlay.as_mut().is_some_and(|o| o.dialog().is_some()) {
      self.overlay = None;
    }
  }

  /// Freeze every live tool output when the run was killed.
  fn finish_running_tool(&mut self) {
    finish_running(&mut self.entries);
  }

  /// The run never reached its final response, so rig did not hand back the
  /// updated transcript. Keep what the user saw. The messages a `/continue`
  /// resumed from are already in the history and stay there.
  fn recover_in_flight(&mut self) {
    let Some(in_flight) = self.in_flight.take() else {
      return;
    };
    let messages = in_flight.recovered();
    if messages.is_empty() {
      return;
    }
    // How those tool calls went goes with them, or the transcript they leave
    // behind would forget which failed and what each changed.
    let outcomes = std::mem::take(&mut self.outcomes);
    let result = self.session.append_with(messages, &outcomes);
    self.report(result);
  }

  // ------------------------------------------------------------ agent events

  fn handle_agent(&mut self, ev: AgentEvent) {
    if self.run.is_none() {
      return; // stale event from an aborted run
    }
    match ev {
      AgentEvent::Text(delta) => {
        if let Some(in_flight) = &mut self.in_flight {
          in_flight.said(&delta);
        }
        match self.entries.last_mut() {
          Some(Entry::Assistant(text)) => text.push_str(&delta),
          _ => self.entries.push(Entry::Assistant(delta)),
        }
      }
      AgentEvent::Reasoning(delta) => match self.entries.last_mut() {
        Some(Entry::Reasoning(text)) => text.push_str(&delta),
        _ => self.entries.push(Entry::Reasoning(delta)),
      },
      AgentEvent::ToolCallDelta { id, name, args } => match self.writing.iter_mut().find(|w| w.id == id) {
        Some(writing) => (writing.name, writing.args) = (name, args),
        None => self.writing.push(Writing { id, name, args }),
      },
      AgentEvent::ToolCall {
        name,
        args,
        call,
        internal,
      } => {
        // This call has stopped being a line the model is typing and become
        // one the transcript keeps — whichever of the pending lines it is.
        self.writing.retain(|writing| writing.id != internal);
        // Part of the turn being streamed, so an abort keeps it.
        if let Some(in_flight) = &mut self.in_flight {
          in_flight.called(&call, &name, args.clone());
        }
        let summary = summarize_args(&name, &args);
        self.entries.push(Entry::ToolCall {
          wrote: wrote_content(&name, &args),
          name,
          summary,
          call,
          started: Instant::now(),
        });
      }
      // Output goes under the call it came from, wherever that call's line
      // has ended up — not under whatever the transcript happens to end with.
      AgentEvent::ToolOutput { call, text } => {
        place_output(&mut self.entries, call, text);
      }
      AgentEvent::ToolResult {
        name,
        output,
        images,
        is_error,
        call,
        diff,
      } => {
        if let Some(in_flight) = &mut self.in_flight {
          in_flight.answered(&call, &name, output.clone());
        }
        // What the transcript will not carry on its own.
        if is_error || diff.is_some() {
          self.outcomes.insert(
            call.clone(),
            Outcome {
              call: call.clone(),
              failed: is_error,
              diff: diff.clone(),
            },
          );
        }
        place_result(
          &mut self.entries,
          Finished {
            name,
            output,
            images,
            is_error,
            call,
            diff,
          },
        );
      }
      // The question takes the screen until it is answered: the run that
      // asked it is waiting on the answer, so there is nothing else to be
      // doing here anyway.
      AgentEvent::AskUser { questions, reply } => {
        if self.bell {
          bell();
        }
        self.overlay = Some(Overlay {
          list: OverlayList::Question {
            dialog: Box::new(Dialog::new(questions)),
            reply: Some(reply),
          },
          selected: 0,
        });
      }
      AgentEvent::Error(err) => {
        self.entries.push(Entry::Error(err));
      }
      // One model call, counted as it happens: a run that takes twenty of
      // them moves the footer twenty times rather than sitting still until
      // it is over.
      AgentEvent::Usage { usage, context_tokens } => {
        self.usage.input_tokens += usage.input_tokens;
        self.usage.output_tokens += usage.output_tokens;
        self.usage.total_tokens += usage.total_tokens;
        self.context_tokens = Some(context_tokens);
      }
      AgentEvent::Done { messages } => {
        self.run = None;
        self.writing.clear();
        self.close_question();
        // A `/continue` run echoes back the messages it resumed from; they
        // are already in the history.
        let resumed = self.in_flight.take().map_or(0, |f| f.resumed);
        let outcomes = std::mem::take(&mut self.outcomes);
        let result = self
          .session
          .append_with(messages.into_iter().skip(resumed).collect(), &outcomes);
        self.report(result);
        // The run ended with its answer, so there is nothing to pick back
        // up; a context that outgrew the window is still made room in,
        // before the next message is sent into it.
        self.resuming = false;
        if self.overflowed() {
          self.compact();
        } else {
          self.next_queued();
        }
      }
      AgentEvent::Ended => {
        self.run = None;
        self.writing.clear();
        self.outcomes.clear();
        self.close_question();
        let full = self.overflowed();
        if self.compacting {
          // A compaction that did not finish is not worth starting again on
          // the next turn: the room it was going to make is not coming.
          self.compacting = false;
          self.resuming = false;
        } else {
          self.recover_in_flight();
          if full {
            // The run stopped at a turn boundary to let this happen, and
            // goes on once there is room again.
            self.resuming = true;
            self.compact();
            return;
          }
        }
        self.next_queued();
      }
      AgentEvent::Compacted(result) => {
        self.run = None;
        self.compacting = false;
        let resuming = std::mem::take(&mut self.resuming);
        match result {
          Some(compacted) => {
            let result = self.session.compacted(compacted.history, &compacted.summary);
            self.report(result);
            self.context_tokens = None;
            self.entries.push(Entry::Info(format!(
              "Compacted {} messages into a summary; kept the last {}.",
              compacted.summarized, compacted.kept
            )));
            self.entries.push(Entry::Summary(compacted.summary));
            // What was cut short to make this room carries on where it
            // stopped — unless the user has said something since, which is
            // what it would have read next anyway.
            if resuming && self.queued.is_empty() {
              self.continue_run();
              return;
            }
          }
          // Nothing left to summarize but the turn the context is full of.
          // Carrying on regardless would only fill it again and ask for the
          // same summary, so this is where it stops and the user decides.
          None => self.entries.push(Entry::Info(match resuming {
            true => "The context is full and there is nothing left to compact — /continue to carry on anyway.".into(),
            false => "Nothing to compact.".to_string(),
          })),
        }
        self.next_queued();
      }
    }
  }

  // ------------------------------------------------------------ drawing

  fn draw(&mut self, f: &mut Frame) {
    let input_height = self.input.lines().len().clamp(1, MAX_INPUT_LINES) as u16 + 2;
    let popup_height = self
      .completion
      .as_ref()
      .map_or(0, |c| c.items.len().min(COMPLETION_ROWS)) as u16;
    let [transcript_area, popup_area, input_area, footer_area] = Layout::vertical([
      Constraint::Fill(1),
      Constraint::Length(popup_height),
      Constraint::Length(input_height),
      Constraint::Length(1),
    ])
    .areas(f.area());
    if popup_height > 0 {
      self.draw_completion(f, popup_area);
    }

    if self.overlay.is_some() {
      self.draw_overlay(f, transcript_area);
    } else {
      self.draw_transcript(f, transcript_area);
    }

    // Input box.
    let border_color = if self.run.is_some() {
      Color::DarkGray
    } else {
      Color::Gray
    };
    self.input.set_block(
      Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color)),
    );
    f.render_widget(&self.input, input_area);
    self.draw_footer(f, footer_area);
  }

  fn draw_transcript(&mut self, f: &mut Frame, transcript_area: Rect) {
    // Pinned to the bottom unless the user scrolled up. In `always` mode
    // the scrollbar gets its own column; in `auto` mode it is
    // overlaid on the transcript's last column while visible.
    let mut content_area = transcript_area;
    if self.scrollbar == ScrollbarMode::Always && content_area.width > 1 {
      content_area.width -= 1;
    }
    // Markdown arrives pre-wrapped to this width, so `Wrap` passes it through
    // untouched and still handles the entries that stay literal.
    let lines = self.transcript_lines(content_area.width);
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    let total = paragraph.line_count(content_area.width);
    let viewport = content_area.height as usize;
    let max_scroll = total.saturating_sub(viewport);
    if self.anchor.is_some_and(|a| a >= max_scroll) {
      self.anchor = None;
    }
    let offset = self.anchor.unwrap_or(max_scroll);
    self.view = (offset, max_scroll);
    f.render_widget(
      paragraph.scroll((offset.min(u16::MAX as usize) as u16, 0)),
      content_area,
    );
    let overflows = total > viewport;
    let show_scrollbar = match self.scrollbar {
      ScrollbarMode::Always => viewport > 0,
      ScrollbarMode::Auto => overflows && self.scrollbar_fading(),
      ScrollbarMode::Hidden => false,
    };
    if show_scrollbar {
      let mut state = ScrollbarState::new(max_scroll)
        .position(offset)
        .viewport_content_length(viewport);
      f.render_stateful_widget(
        Scrollbar::new(ScrollbarOrientation::VerticalRight)
          .begin_symbol(None)
          .end_symbol(None)
          .track_symbol(Some("│"))
          .thumb_symbol("┃")
          .track_style(Style::default().fg(Color::DarkGray))
          .thumb_style(Style::default().fg(Color::Gray)),
        transcript_area,
        &mut state,
      );
    }
  }

  fn draw_completion(&self, f: &mut Frame, area: Rect) {
    let Some(c) = &self.completion else { return };
    let rows = area.height as usize;
    let first = c.selected.saturating_sub(rows.saturating_sub(1));
    let name_width = COMMANDS.iter().map(|(n, ..)| n.len() + 1).max().unwrap_or(0);
    let dim = Style::default().add_modifier(Modifier::DIM);
    let lines: Vec<Line> = c
      .items
      .iter()
      .enumerate()
      .skip(first)
      .take(rows)
      .map(|(i, m)| {
        let (name, description, _) = COMMANDS[m.index];
        let selected = i == c.selected;
        let base = if selected {
          Style::default().bold()
        } else {
          Style::default()
        };
        let hit = base.fg(Color::Cyan).underlined();
        let mut spans = vec![
          Span::styled(
            if selected { "› " } else { "  " },
            Style::default().fg(Color::Cyan).bold(),
          ),
          Span::styled("/", base),
        ];
        // Matched letters get their own styled spans.
        for (pos, ch) in name.chars().enumerate() {
          let style = if m.highlights.contains(&(pos as u32)) {
            hit
          } else {
            base
          };
          spans.push(Span::styled(ch.to_string(), style));
        }
        spans.push(Span::raw(" ".repeat(name_width - name.chars().count())));
        spans.push(Span::styled(description, dim));
        Line::from(spans)
      })
      .collect();
    f.render_widget(Paragraph::new(lines), area);
  }

  fn draw_overlay(&self, f: &mut Frame, area: Rect) {
    let Some(overlay) = &self.overlay else { return };
    let block = Block::default()
      .borders(Borders::ALL)
      .border_type(BorderType::Rounded)
      .border_style(Style::default().fg(Color::Gray))
      .title(overlay.title());
    let inner = block.inner(area);
    f.render_widget(block, area);
    let height = inner.height as usize;
    let width = inner.width as usize;
    if let OverlayList::Question { dialog, .. } = &overlay.list {
      let (lines, focus) = dialog.lines(inner.width);
      // More dialog than screen: scroll it just far enough to keep the row
      // the cursor is on in view, which is the row being answered.
      let first = focus.saturating_sub(height.saturating_sub(1)).min(focus);
      f.render_widget(
        Paragraph::new(lines.into_iter().skip(first).take(height).collect::<Vec<_>>()),
        inner,
      );
      return;
    }
    // The window ends at the selection, so moving down walks off the bottom
    // rather than jumping the list around.
    let first = overlay.selected.saturating_sub(height.saturating_sub(1));
    // Each row is a title and a dim note about it, the title clipped so the
    // note always fits.
    let rows: Vec<(String, String)> = match &overlay.list {
      OverlayList::Sessions(sessions) => sessions
        .iter()
        .map(|s| {
          let note = format!(
            "{}  {} msgs  {}",
            shorten_home(Path::new(&s.cwd)),
            s.message_count,
            s.age()
          );
          (s.title().to_string(), note)
        })
        .collect(),
      // The tree is indented at its branch points, so a conversation that
      // went two ways reads as two ways. Both lists say how long the
      // conversation would be once you got there.
      OverlayList::Tree(points) => points
        .iter()
        .map(|p| {
          let note = match p.here {
            true => "here".to_string(),
            false => messages(p.len),
          };
          (format!("{}{}", "  ".repeat(p.depth), p.label), note)
        })
        .collect(),
      OverlayList::Fork(points) => points
        .iter()
        .map(|p| (p.label.clone(), format!("keeps {}", messages(p.len))))
        .collect(),
      // Drawn above, where it draws itself.
      OverlayList::Question { .. } => Vec::new(),
    };
    let dim = Style::default().add_modifier(Modifier::DIM);
    let mut lines = Vec::new();
    for (i, (title, note)) in rows.into_iter().enumerate().skip(first).take(height) {
      let selected = i == overlay.selected;
      let avail = width.saturating_sub(2 + note.chars().count() + 2);
      let title: String = title.chars().take(avail).collect();
      let pad = width.saturating_sub(2 + title.chars().count() + note.chars().count());
      lines.push(Line::from(vec![
        Span::styled(
          if selected { "› " } else { "  " },
          Style::default().fg(Color::Cyan).bold(),
        ),
        Span::styled(
          title,
          if selected {
            Style::default().bold()
          } else {
            Style::default()
          },
        ),
        Span::raw(" ".repeat(pad)),
        Span::styled(note, dim),
      ]));
    }
    f.render_widget(Paragraph::new(lines), inner);
  }

  fn draw_footer(&self, f: &mut Frame, footer_area: Rect) {
    let cwd = shorten_home(&self.cwd);
    let mut left = vec![
      Span::raw(cwd).dim(),
      Span::raw("  "),
      Span::raw(self.model.clone()).dim(),
    ];
    if let Some(label) = mcp_label(self.mcp.0, self.mcp.1) {
      left.push(Span::raw("  "));
      left.push(Span::raw(label).dim());
    }
    if self.run.is_some() {
      left.push(Span::raw("  "));
      let verb = if self.compacting { "compacting" } else { "working" };
      left.push(Span::raw(format!("{} {verb} — esc to abort", SPINNER[self.tick % SPINNER.len()])).fg(Color::Yellow));
      if !self.queued.is_empty() {
        left.push(Span::raw(format!("  ({} queued)", self.queued.len())).dim());
      }
    } else if let Some(anchor) = self.anchor {
      let behind = self.view.1 - anchor;
      left.push(Span::raw("  "));
      left.push(Span::raw(format!("↑ {behind} lines — pgdn to follow")).dim());
    }
    let context = self
      .context_tokens
      .and_then(|tokens| (tokens * 100).checked_div(self.settings.context_window))
      .map(|pct| (format!("ctx {pct}%  "), pct));
    let tokens = format!("{}↑ {}↓", self.usage.input_tokens, self.usage.output_tokens);
    let width = context.as_ref().map_or(0, |(text, _)| text.chars().count()) + tokens.chars().count();
    let pad = (footer_area.width as usize).saturating_sub(Line::from(left.clone()).width() + width);
    left.push(Span::raw(" ".repeat(pad)));
    if let Some((text, percent)) = context {
      left.push(Span::styled(text, context_style(percent)));
    }
    left.push(Span::raw(tokens).dim());
    f.render_widget(Paragraph::new(Line::from(left)), footer_area);
  }

  /// Builds the transcript for a `width`-column viewport.
  ///
  /// Assistant text is the only markdown here: tool output, diffs and
  /// reasoning are literal, and parsing them as markdown would mangle them.
  fn transcript_lines(&mut self, width: u16) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let dim = Style::default().add_modifier(Modifier::DIM);
    // Thinking is grey and italic rather than dimmed: grey on top of dim
    // reads as noise on terminals that render faint text very faint. Bright
    // black is the palette's own grey, so it tracks the terminal's theme.
    let thinking = Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC);
    let mut cached = std::mem::take(&mut self.markdown);
    let mut live = HashMap::with_capacity(cached.len());
    // Only the last entry can still be growing, and only while a turn is in
    // flight. Everything else is final, and is parsed as written.
    let streaming = self.run.is_some().then(|| self.entries.len().saturating_sub(1));
    for (at, entry) in self.entries.iter().enumerate() {
      match entry {
        Entry::User(text) => {
          lines.push(Line::default());
          for (i, l) in text.lines().enumerate() {
            let prefix = if i == 0 { "❯ " } else { "  " };
            lines.push(Line::from(vec![
              Span::styled(prefix, Style::default().fg(Color::Cyan).bold()),
              Span::styled(l.to_string(), Style::default().bold()),
            ]));
          }
        }
        Entry::Assistant(text) => {
          lines.push(Line::default());
          // Only the streaming entry changes between frames; the rest come
          // back from the cache untouched, so a long transcript costs one
          // parse per message rather than one per draw. `streaming` is part
          // of the key so the completed text is not served its mid-stream
          // rendering, which closes tokens this one should leave literal.
          let streaming = streaming == Some(at);
          let key = (hash(ASSISTANT_KIND, text), width, streaming);
          let rendered = match cached.remove(&key) {
            Some(rendered) => rendered,
            None => crate::markdown::render(text, width, streaming),
          };
          lines.extend(rendered.iter().cloned());
          live.insert(key, rendered);
        }
        Entry::Reasoning(text) => {
          lines.push(Line::default());
          // Counted in lines as drawn, not as the provider happened to break
          // them: a reasoning summary arrives as one long paragraph, which
          // would otherwise count as a single line and never collapse.
          let body = width.saturating_sub(REASONING_INDENT.len() as u16);
          let key = (hash(REASONING_KIND, text), body, false);
          let wrapped = match cached.remove(&key) {
            Some(wrapped) => wrapped,
            None => crate::markdown::wrap_text(text, body, thinking),
          };
          let shown = if self.expand_thinking {
            wrapped.len()
          } else {
            wrapped.len().min(REASONING_LINES)
          };
          let hidden = wrapped.len() - shown;
          let header = match hidden {
            0 => "· thinking…".to_string(),
            1 => "· thinking… (1 earlier line hidden)".to_string(),
            n => format!("· thinking… ({n} earlier lines hidden)"),
          };
          lines.push(Line::styled(header, thinking));
          for line in &wrapped[hidden..] {
            lines.push(prefix(REASONING_INDENT, line.clone()));
          }
          live.insert(key, wrapped);
        }
        Entry::ToolCall { name, summary, .. } => {
          lines.push(Line::default());
          let mut spans = vec![
            Span::styled("⚙ ", Style::default().fg(Color::Yellow)),
            Span::styled(name.clone(), Style::default().fg(Color::Yellow).bold()),
            Span::raw(" "),
          ];
          // A command is code, and reads as code. So is what a tool the agent
          // did not bring is being asked for: nothing here knows what its
          // arguments mean, so they are shown as the JSON they arrived as.
          match name.as_str() {
            "bash" => spans.extend(code_spans("bash", summary, dim)),
            name if !crate::tools::BUILT_IN.contains(&name) => spans.extend(code_spans("json", summary, dim)),
            _ => spans.push(Span::styled(summary.clone(), dim)),
          }
          lines.push(Line::from(spans));
        }
        Entry::ToolResult {
          name,
          output,
          images,
          is_error,
          call,
          running,
          diff,
          started,
          took,
        } => {
          // What the tool has to show. A write shows the file it wrote, an
          // edit what it changed, and anything else what it said — and a
          // diff of nothing is not worth a block of nothing.
          let diff = diff.as_deref().filter(|diff| !diff.trim().is_empty());
          let wrote = wrote_by(&self.entries[..at], call);
          let (body, cap, from_end) = match (wrote, diff) {
            (Some((path, content)), _) => {
              // A file's last newline ends its last line; it does not start
              // another.
              let content = content.strip_suffix('\n').unwrap_or(content);
              (file_lines(path, content), TOOL_OUTPUT_LINES, false)
            }
            (None, Some(diff)) => (marked_lines(diff, true), DIFF_LINES, false),
            // Structure a tool answered with is worth reading as structure.
            (None, None) => match structured(name, output) {
              Some(lines) => (lines, TOOL_OUTPUT_LINES, false),
              // A command's output is most useful at its end and a file's at
              // its start, so each keeps the end that matters.
              None => (marked_lines(output, false), TOOL_OUTPUT_LINES, name == "bash"),
            },
          };
          let stripe = gutter(*running, *is_error);
          // A tool that answered with a picture and nothing else gets no
          // block of nothing above it.
          let silent = body.iter().flatten().all(|span| span.content.trim().is_empty());
          if !(silent && !images.is_empty()) {
            Preview {
              body,
              gutter: stripe.clone(),
              cap,
              expanded: self.expand_tools,
              from_end,
              cursor: false,
            }
            .draw(&mut lines);
          }
          // The picture itself, under whatever was said about it: drawn at
          // the width the transcript has, which is the detail a terminal can
          // hold, and folded like any other block — so `Ctrl+O` shows the
          // rest of it rather than a larger copy of it, and nothing already
          // on screen moves when it does.
          //
          // Scaling one is far more work than a frame has, so a drawn image
          // is kept by its bytes and the width it was drawn at, the way
          // rendered markdown is. The fold is not part of the key: it is how
          // much of the same drawing is shown.
          for image in images {
            // Two columns of indent and the gutter, as every other block of
            // a tool's output is drawn.
            let cols = width.saturating_sub(3);
            let key = (hash_bytes(IMAGE_KIND, image), cols, false);
            let drawn = match cached.remove(&key) {
              Some(drawn) => drawn,
              None => crate::images::blocks(image, cols, IMAGE_MAX_LINES)
                .unwrap_or_else(|| vec![Line::styled("[image could not be drawn]", dim)]),
            };
            Preview {
              body: drawn.iter().map(|line| line.spans.clone()).collect(),
              gutter: stripe.clone(),
              cap: IMAGE_LINES,
              expanded: self.expand_tools,
              from_end: false,
              cursor: false,
            }
            .draw(&mut lines);
            live.insert(key, drawn);
          }
          if name == "bash" && (*running || took.is_some()) {
            let (label, elapsed) = match took {
              Some(took) => ("Took", *took),
              None => ("Elapsed", started.elapsed()),
            };
            lines.push(Line::styled(format!("  {label} {}", format_duration(elapsed)), dim));
          }
        }
        Entry::Error(err) => {
          lines.push(Line::default());
          for (i, l) in err.lines().enumerate() {
            let prefix = if i == 0 { "✗ " } else { "  " };
            lines.push(Line::styled(format!("{prefix}{l}"), Style::default().fg(Color::Red)));
          }
        }
        Entry::Info(text) => {
          lines.push(Line::default());
          for l in text.lines() {
            lines.push(Line::styled(l.to_string(), dim.italic()));
          }
        }
        Entry::Summary(text) => {
          lines.push(Line::default());
          lines.push(Line::styled(
            "▤ Context summary",
            Style::default().fg(Color::Magenta).bold(),
          ));
          for l in text.lines() {
            lines.push(Line::styled(format!("  {l}"), dim));
          }
        }
      }
    }
    // Calls the model is still writing, drawn like the entry each will
    // become so that nothing moves when it does.
    for writing in &self.writing {
      lines.push(Line::default());
      let summary = writing_summary(&writing.name, &writing.args);
      let blocks = writing_body(&writing.name, &writing.args);
      let body: Vec<Vec<Span<'static>>> = match (writing.name.as_str(), blocks.as_slice()) {
        // A file arriving is shown as the file it will be, highlighted the
        // same way — so nothing recolours when the call is finally made.
        ("write", [(_, content)]) => file_lines(&summary, content),
        // Every line of every other block, each carrying the mark it is
        // drawn under: a replacement says which half it is, and anything
        // else says nothing, since the gutter is already saying it.
        _ => blocks
          .iter()
          .flat_map(|(mark, text)| {
            let prefix = match *mark {
              "│" => String::new(),
              mark => format!("{mark} "),
            };
            // Split rather than `lines`, so a body ending in a newline keeps
            // the empty line the model is about to write into.
            text
              .split('\n')
              .map(|line| {
                vec![Span::styled(
                  format!(" {prefix}{line}"),
                  mark_style(mark.chars().next()),
                )]
              })
              .collect::<Vec<_>>()
          })
          .collect(),
      };
      let mut header = vec![
        Span::styled("⚙ ", Style::default().fg(Color::Yellow)),
        Span::styled(writing.name.clone(), Style::default().fg(Color::Yellow).bold()),
        Span::raw(" "),
      ];
      // Highlighted as it is typed, so the line does not recolour under the
      // reader when the call is finally made.
      match writing.name.as_str() {
        "bash" => header.extend(code_spans("bash", &summary, dim)),
        name if !crate::tools::BUILT_IN.contains(&name) => header.extend(code_spans("json", &summary, dim)),
        _ => header.push(Span::styled(summary, dim)),
      }
      // The cursor follows the model: on the first line until there is a body
      // to write into, then at the end of what has arrived.
      if body.is_empty() {
        header.push(Span::styled("▌", Style::default().fg(Color::Yellow)));
      }
      lines.push(Line::from(header));
      Preview {
        body,
        // Nothing has gone right or wrong yet, so the plain gutter.
        gutter: gutter(true, false),
        cap: TOOL_OUTPUT_LINES,
        expanded: self.expand_tools,
        // The tail is what is being written; what came before is already said.
        from_end: true,
        cursor: true,
      }
      .draw(&mut lines);
    }
    // What the user typed while the run was going, at the end because that is
    // where it will be sent from — under everything the run is still saying,
    // not above it.
    if !self.queued.is_empty() {
      lines.push(Line::default());
      for text in &self.queued {
        lines.push(Line::styled(format!("Queued: {}", first_line(text)), dim.italic()));
      }
    }
    self.markdown = live;
    lines
  }
}

/// Identifies a message by content, for the rendered-text cache. `kind` keeps
/// an assistant message and a reasoning block apart when they read the same.
fn hash(kind: u8, text: &str) -> u64 {
  let mut hasher = DefaultHasher::new();
  kind.hash(&mut hasher);
  text.hash(&mut hasher);
  hasher.finish()
}

/// The same, for an image, which is bytes rather than text.
fn hash_bytes(kind: u8, bytes: &[u8]) -> u64 {
  let mut hasher = DefaultHasher::new();
  kind.hash(&mut hasher);
  bytes.hash(&mut hasher);
  hasher.finish()
}

/// Puts `lead` in front of a rendered line, keeping the rest of its spans.
fn prefix(lead: &'static str, line: Line<'static>) -> Line<'static> {
  let mut spans = Vec::with_capacity(line.spans.len() + 1);
  spans.push(Span::raw(lead));
  spans.extend(line.spans);
  Line::from(spans)
}

/// One terminal bell, rung when the model stops and waits on an answer.
///
/// The questionnaire is the one thing in fa that goes nowhere until somebody
/// comes back to it, and a run is usually left to get on with its work. What a
/// bell then does — a sound, a flash of the window, a badge, nothing at all —
/// is the terminal's own business, which is what makes it the right thing to
/// send: every terminal has already been told how its user wants to be
/// interrupted, and fa has not.
fn bell() {
  use std::io::Write;
  let mut out = std::io::stdout();
  let _ = out.write_all(b"\x07");
  let _ = out.flush();
}

/// Alt+Enter and Shift+Enter (where the terminal reports it) insert a newline.
pub fn is_newline(key: &KeyEvent) -> bool {
  key.modifiers.intersects(KeyModifiers::ALT | KeyModifiers::SHIFT)
}

/// Commands matching `query` (the text after `/`), best first. An empty
/// query lists everything in table order.
fn filter_commands(matcher: &mut Matcher, query: &str) -> Vec<Match> {
  if query.is_empty() {
    return (0..COMMANDS.len())
      .map(|index| Match {
        index,
        highlights: Vec::new(),
      })
      .collect();
  }
  let pattern = Pattern::parse(query, CaseMatching::Ignore, Normalization::Smart);
  let mut buf = Vec::new();
  let mut scored: Vec<(u32, Match)> = COMMANDS
    .iter()
    .enumerate()
    .filter_map(|(index, (name, ..))| {
      let mut highlights = Vec::new();
      let score = pattern.indices(Utf32Str::new(name, &mut buf), matcher, &mut highlights)?;
      highlights.sort_unstable();
      highlights.dedup();
      Some((score, Match { index, highlights }))
    })
    .collect();
  scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.index.cmp(&b.1.index)));
  scored.into_iter().map(|(_, m)| m).collect()
}

/// Stop the clock on every command still drawing output, because the run
/// they belonged to is over. Every one of them, since more than one can be in
/// flight and an abort takes them all.
fn finish_running(entries: &mut [Entry]) {
  for entry in entries {
    if let Entry::ToolResult {
      running, took, started, ..
    } = entry
      && *running
    {
      *running = false;
      *took = Some(started.elapsed());
    }
  }
}

/// How much of a block is folded away, and which way — with the key that
/// unfolds it, since a fold nobody knows how to open is just a truncation.
fn fold_note(hidden: usize, earlier: bool) -> String {
  let which = if earlier { "earlier" } else { "more" };
  match hidden {
    1 => format!(" … 1 {which} line (ctrl+o)"),
    n => format!(" … {n} {which} lines (ctrl+o)"),
  }
}

/// What a `write` is putting in the file, from the arguments asking for it.
fn wrote_content(name: &str, args: &serde_json::Value) -> Option<String> {
  (name == "write")
    .then(|| args.get("content")?.as_str().map(str::to_string))
    .flatten()
}

/// The file a write put down, and the name that says how to read it, from the
/// call that asked for it.
fn wrote_by<'a>(entries: &'a [Entry], call: &str) -> Option<(&'a str, &'a str)> {
  entries.iter().rev().find_map(|entry| match entry {
    Entry::ToolCall {
      call: at,
      summary,
      wrote: Some(content),
      ..
    } if at == call => Some((summary.as_str(), content.as_str())),
    _ => None,
  })
}

/// A file as the transcript shows it: highlighted when its name says what
/// language it is, plain when it does not.
///
/// A write is not a change to be marked up — it is the file, so it is shown
/// as the file, with none of a diff's pluses and none of its green.
fn file_lines(path: &str, content: &str) -> Vec<Vec<Span<'static>>> {
  let language = Path::new(path)
    .extension()
    .and_then(|extension| extension.to_str())
    .unwrap_or_default();
  code_lines(language, content)
}

/// Lines of code in `language`, highlighted where there is a grammar for it
/// and plain where there is not.
fn code_lines(language: &str, content: &str) -> Vec<Vec<Span<'static>>> {
  let highlighted = crate::highlight::highlight(language, content);
  content
    .split('\n')
    .enumerate()
    .map(|(i, line)| match highlighted.as_ref().and_then(|lines| lines.get(i)) {
      Some(spans) if !spans.is_empty() => {
        let mut row = vec![Span::raw(" ")];
        row.extend(spans.iter().cloned());
        row
      }
      _ => vec![Span::styled(format!(" {line}"), mark_style(None))],
    })
    .collect()
}

/// What a tool answered, laid out and highlighted as the JSON it is — or
/// nothing, when it did not answer with any.
///
/// Only for the tools the agent did not bring. What `bash` printed is the
/// command's own to lay out, and a file `read` holds should be seen the way it
/// is written, so neither is re-indented for being valid JSON. An MCP server's
/// answer has no such shape of its own: it arrives as one long line, which is
/// the worst way to read a structure.
fn structured(name: &str, output: &str) -> Option<Vec<Vec<Span<'static>>>> {
  if crate::tools::BUILT_IN.contains(&name) {
    return None;
  }
  let value: serde_json::Value = serde_json::from_str(output.trim()).ok()?;
  // A bare string or number is already the shortest way to say itself.
  if !value.is_object() && !value.is_array() {
    return None;
  }
  let pretty = serde_json::to_string_pretty(&value).ok()?;
  Some(code_lines("json", &pretty))
}

/// Lines of a tool's own text. A diff says what each line is with its first
/// character; anything else is the tool talking, and stays out of the way.
fn marked_lines(text: &str, diff: bool) -> Vec<Vec<Span<'static>>> {
  text
    .lines()
    .map(|line| {
      let mark = diff.then(|| line.chars().next()).flatten();
      vec![Span::styled(format!(" {line}"), mark_style(mark))]
    })
    .collect()
}

/// What a `+` and a `-` mean, wherever they are drawn — in the diff a call
/// left behind, or in the replacement it is still writing.
fn mark_style(mark: Option<char>) -> Style {
  match mark {
    Some('+') => Style::default().fg(Color::Green),
    Some('-') => Style::default().fg(Color::Red),
    _ => Style::default().add_modifier(Modifier::DIM),
  }
}

/// A block of a tool's text under its line: what it said, what it changed, or
/// what it is still writing.
///
/// One shape for all three. The block a call is writing and the block it
/// leaves behind are the same thing at different moments, and drawing them
/// from one place is what stops them drifting apart on screen.
struct Preview {
  /// Each line, as the spans it is drawn from — a line at a time rather than
  /// a style at a time, so a highlighted one can carry a colour per word.
  body: Vec<Vec<Span<'static>>>,
  /// Drawn at the head of every line: the verdict stripe once there is one,
  /// the plain gutter until then.
  gutter: Span<'static>,
  cap: usize,
  expanded: bool,
  /// Keep the end rather than the start, for text whose point is its latest.
  from_end: bool,
  /// Mark the last line kept as where the model has got to.
  cursor: bool,
}

impl Preview {
  fn draw(self, out: &mut Vec<Line<'static>>) {
    let Preview {
      body,
      gutter,
      cap,
      expanded,
      from_end,
      cursor,
    } = self;
    let hidden = match expanded {
      true => 0,
      false => body.len().saturating_sub(cap),
    };
    let row = |line: Vec<Span<'static>>, tip: bool| {
      let mut spans = vec![Span::raw("  "), gutter.clone()];
      spans.extend(line);
      if tip {
        spans.push(Span::styled("▌", Style::default().fg(Color::Yellow)));
      }
      Line::from(spans)
    };
    let note = |hidden, earlier| vec![Span::styled(fold_note(hidden, earlier), mark_style(None))];
    if hidden > 0 && from_end {
      out.push(row(note(hidden, true), false));
    }
    let kept: Vec<Vec<Span<'static>>> = match from_end {
      true => body.into_iter().skip(hidden).collect(),
      false => {
        let shown = body.len() - hidden;
        body.into_iter().take(shown).collect()
      }
    };
    let last = kept.len().saturating_sub(1);
    for (i, line) in kept.into_iter().enumerate() {
      out.push(row(line, cursor && i == last));
    }
    if hidden > 0 && !from_end {
      out.push(row(note(hidden, false), false));
    }
  }
}

/// The stripe down the side of a finished tool's output, saying how it went.
///
/// A background rather than coloured text, so the output keeps whatever
/// colours are its own — and one cell wide, which is where the terminal's own
/// red and green are right: saturated enough to read at a glance, and carrying
/// no text to be legible against. A tool still running has no verdict yet, so
/// it keeps the plain gutter.
fn gutter(running: bool, is_error: bool) -> Span<'static> {
  match (running, is_error) {
    (true, _) => Span::styled("│", Style::default().add_modifier(Modifier::DIM)),
    (false, true) => Span::styled(" ", Style::default().bg(Color::Red)),
    (false, false) => Span::styled(" ", Style::default().bg(Color::Green)),
  }
}

/// One line of code, so a command reads as the command it is and a call's
/// arguments as the JSON they are.
///
/// Falls back to the plain line when the grammar was not built in, which is
/// the same text either way.
fn code_spans(language: &str, code: &str, style: Style) -> Vec<Span<'static>> {
  crate::highlight::highlight(language, code)
    .and_then(|lines| lines.into_iter().next())
    .filter(|spans| !spans.is_empty())
    .map(|spans| {
      spans
        .into_iter()
        .map(|span| Span::styled(span.content, style.patch(span.style)))
        .collect()
    })
    .unwrap_or_else(|| vec![Span::styled(code.to_string(), style)])
}

/// Where the call `call` was announced.
///
/// Searched from the end: a reopened session brings its old calls back as
/// entries, and a server that numbers calls from one per turn hands out ids
/// the transcript already holds. The live one is the one written last.
fn call_at(entries: &[Entry], call: &str) -> Option<usize> {
  entries.iter().rposition(|entry| match entry {
    Entry::ToolCall { call: at, .. } => at == call,
    _ => false,
  })
}

/// Where that call's output is being drawn while it runs.
fn running_at(entries: &[Entry], call: &str) -> Option<usize> {
  entries.iter().rposition(|entry| match entry {
    Entry::ToolResult {
      running: true,
      call: at,
      ..
    } => at == call,
    _ => false,
  })
}

/// Draw a running command's latest output under the call it came from.
///
/// Under the call, rather than at the end of the transcript: two commands can
/// be in flight at once, and the one that speaks is not always the one that
/// started last.
fn place_output(entries: &mut Vec<Entry>, call: String, text: String) {
  if let Some(i) = running_at(entries, &call)
    && let Some(Entry::ToolResult { output, .. }) = entries.get_mut(i)
  {
    *output = text;
    return;
  }
  let Some(i) = call_at(entries, &call) else { return };
  let Some(Entry::ToolCall { name, started, .. }) = entries.get(i) else {
    return;
  };
  let (name, started) = (name.clone(), *started);
  entries.insert(
    i + 1,
    Entry::ToolResult {
      name,
      output: text,
      images: Vec::new(),
      is_error: false,
      call,
      running: true,
      diff: None,
      started,
      took: None,
    },
  );
}

/// What a tool finally said, as the transcript takes it in.
struct Finished {
  name: String,
  output: String,
  images: Vec<Vec<u8>>,
  is_error: bool,
  call: String,
  diff: Option<String>,
}

/// Replace a command's live output with what it finally said, in place, or
/// put it under the call that asked for it.
fn place_result(entries: &mut Vec<Entry>, result: Finished) {
  let Finished {
    name,
    output,
    images,
    is_error,
    call,
    diff,
  } = result;
  if let Some(i) = running_at(entries, &call) {
    entries.remove(i);
  }
  let at = call_at(entries, &call).map_or(entries.len(), |i| i + 1);
  let started = match entries.get(at.saturating_sub(1)) {
    Some(Entry::ToolCall { started, .. }) => *started,
    _ => Instant::now(),
  };
  entries.insert(
    at,
    Entry::ToolResult {
      name,
      output,
      images,
      is_error,
      call,
      running: false,
      diff,
      started,
      took: Some(started.elapsed()),
    },
  );
}

/// Every tool result in a history, and which call each answers.
struct Results<'a> {
  results: Vec<&'a ToolResult>,
  /// Both of the ids a result can be claimed by, to its place above.
  by_id: HashMap<String, usize>,
}

impl<'a> Results<'a> {
  fn collect(history: &'a [Message]) -> Self {
    let mut results = Vec::new();
    let mut by_id = HashMap::new();
    for message in history {
      let Message::User { content } = message else { continue };
      for result in content.iter().filter_map(|c| match c {
        UserContent::ToolResult(result) => Some(result),
        _ => None,
      }) {
        for id in crate::session::result_ids(result) {
          by_id.insert(id, results.len());
        }
        results.push(result);
      }
    }
    Self { results, by_id }
  }

  fn index(&self, ids: impl Iterator<Item = String>) -> Option<usize> {
    ids.filter_map(|id| self.by_id.get(&id)).next().copied()
  }

  fn entry(&self, index: usize, session: &Session, now: Instant) -> Entry {
    let result = self.results[index];
    let (output, images) = crate::images::split(&result.content);
    // A failed result reads like any other in the transcript, and the diff an
    // edit produced is not in it at all, so both are things the session
    // remembers — against this result's own call, since one message can
    // answer several calls that went differently.
    let outcome = crate::session::result_ids(result).find_map(|id| session.outcome(&id));
    Entry::ToolResult {
      name: result.name.clone(),
      output,
      images,
      call: result.call.as_str().to_string(),
      is_error: outcome.is_some_and(|outcome| outcome.failed),
      running: false,
      diff: outcome.and_then(|outcome| outcome.diff.clone()),
      started: now,
      took: None,
    }
  }
}

/// Every way a tool call names itself, matching `session::result_ids`.
fn call_ids(call: &ToolCall) -> impl Iterator<Item = String> + '_ {
  [
    Some(call.id.as_str().to_string()),
    call.provider.as_ref().map(|p| p.call_id.clone()),
  ]
  .into_iter()
  .flatten()
}

/// Rebuild the transcript view from a resumed session's history.
///
/// Tool results travel in a message of their own, after the one that asked
/// for them, and a turn can ask for several at once — so taking the history
/// as it comes would read as every call and then every output, in whatever
/// order the provider sent the answers back. Each result is instead put with
/// the call it answers, which is the order the transcript had while it was
/// live.
fn entries_from_history(session: &Session) -> Vec<Entry> {
  let history = &session.history;
  let now = Instant::now();
  let mut entries = Vec::new();
  let results = Results::collect(history);
  let mut answered: HashSet<usize> = HashSet::new();
  for message in history {
    match message {
      Message::System { .. } => {}
      Message::User { content } => {
        for c in content {
          match c {
            UserContent::Text(t) => {
              let summary = t
                .text
                .strip_prefix(SUMMARY_PREFIX)
                .and_then(|r| r.strip_suffix(SUMMARY_SUFFIX));
              entries.push(match summary {
                Some(s) => Entry::Summary(s.to_string()),
                None => Entry::User(t.text.clone()),
              });
            }
            // An output nothing claimed — a result whose call is not in this
            // branch — is still shown, where it was written.
            UserContent::ToolResult(r) => {
              if let Some(i) = results.index(crate::session::result_ids(r))
                && answered.insert(i)
              {
                entries.push(results.entry(i, session, now));
              }
            }
            _ => {}
          }
        }
      }
      Message::Assistant { content, .. } => {
        for c in content {
          match c {
            AssistantContent::Text(t) => entries.push(Entry::Assistant(t.text.clone())),
            AssistantContent::Reasoning(r) => {
              let text = r.display_text();
              if !text.is_empty() {
                entries.push(Entry::Reasoning(text));
              }
            }
            AssistantContent::ToolCall(call) => {
              entries.push(Entry::ToolCall {
                wrote: wrote_content(&call.function.name, &call.function.arguments),
                name: call.function.name.clone(),
                summary: summarize_args(&call.function.name, &call.function.arguments),
                call: call.id.as_str().to_string(),
                started: now,
              });
              // The output belongs under the call that asked for it, not
              // after every call of the turn.
              if let Some(i) = results.index(call_ids(call))
                && answered.insert(i)
              {
                entries.push(results.entry(i, session, now));
              }
            }
            AssistantContent::Image(_) => {}
          }
        }
      }
    }
  }
  entries
}

/// Duration format: `1.2s`, `3m 4s`, `1h 2m 3s`.
fn format_duration(d: Duration) -> String {
  let secs = d.as_secs_f64();
  if secs < 60.0 {
    return format!("{secs:.1}s");
  }
  let total = d.as_secs();
  let (minutes, rem) = (total / 60, total % 60);
  if minutes < 60 {
    format!("{minutes}m {rem}s")
  } else {
    format!("{}h {}m {rem}s", minutes / 60, minutes % 60)
  }
}

/// What a history message is to the tree: where a cut at it lands, and how its
/// row reads. The marks match the ones the transcript puts on the same thing.
enum Kind {
  /// Something the user typed. Going back here takes it out of the history
  /// and returns it to the input box, to be edited and asked again.
  Prompt(String),
  /// An assistant turn that called tools, which is not a place the
  /// conversation can stop: a call with no result behind it is a transcript
  /// no provider will accept. The tool results that answer it are the point
  /// just after, and the message before it the point just before.
  ToolCalls,
  /// An answer, a tool result, a checkpoint — a step a cut keeps as the
  /// conversation's new end.
  Step(String),
}

fn classify(message: &Message) -> Kind {
  match message {
    Message::User { content } => {
      if let Some(text) = crate::session::user_text(message) {
        return Kind::Prompt(text);
      }
      let tools: Vec<&str> = content
        .iter()
        .filter_map(|c| match c {
          UserContent::ToolResult(r) => Some(r.name.as_str()),
          _ => None,
        })
        .collect();
      match tools.is_empty() {
        false => Kind::Step(format!("⚙ {}", tools.join(", "))),
        // What is left is the compaction checkpoint, which travels as a user
        // message but is not one.
        true => Kind::Step("▤ Context summary".into()),
      }
    }
    Message::Assistant { content, .. } => {
      if content.iter().any(|c| matches!(c, AssistantContent::ToolCall(_))) {
        return Kind::ToolCalls;
      }
      let text = content.iter().find_map(|c| match c {
        AssistantContent::Text(t) => Some(first_line(&t.text)),
        AssistantContent::Reasoning(r) => Some(format!("· {}", first_line(&r.display_text()))),
        _ => None,
      });
      Kind::Step(text.unwrap_or_else(|| "(no text)".into()))
    }
    Message::System { .. } => Kind::Step("(system)".into()),
  }
}

/// Every point the conversation can be moved to.
///
/// Depth-first from the roots, and at each branch point the path the session
/// is on comes first — so the conversation you are in reads top to bottom and
/// the ones you left hang off it, where you can walk back into them.
fn points(session: &Session) -> Vec<Point> {
  let mut children: HashMap<Option<&str>, Vec<&Node>> = HashMap::new();
  for node in session.nodes() {
    children.entry(node.parent.as_deref()).or_default().push(node);
  }
  let here: HashSet<&str> = session.lineage(session.leaf()).into_iter().collect();
  let mut out = Vec::new();
  walk(session, &children, &here, None, 0, &mut out);
  out
}

fn walk<'a>(
  session: &'a Session,
  children: &HashMap<Option<&'a str>, Vec<&'a Node>>,
  here: &HashSet<&'a str>,
  parent: Option<&'a str>,
  depth: usize,
  out: &mut Vec<Point>,
) {
  let Some(kids) = children.get(&parent) else {
    return;
  };
  // One child is the conversation carrying on, and reads at the same level.
  // More than one is somewhere it went two ways, which is what the indent is
  // for — and where the path still in use is listed first.
  let branching = kids.len() > 1;
  let mut kids = kids.clone();
  if branching {
    kids.sort_by_key(|node| !here.contains(node.id.as_str()));
  }
  let depth = depth + usize::from(branching);
  for node in kids {
    if let Some(point) = point(session, node, depth) {
      out.push(point);
    }
    // A step that is no place to stop still has children that are.
    walk(session, children, here, Some(&node.id), depth, out);
  }
}

fn point(session: &Session, node: &Node, depth: usize) -> Option<Point> {
  let kind = match &node.kind {
    NodeKind::Message(message) => classify(message),
    NodeKind::Checkpoint { .. } => Kind::Step("▤ Context summary".into()),
  };
  // A prompt is taken back out of the history and handed to the input box, so
  // the conversation ends where it did before. Anything else is kept.
  let (leaf, text, label) = match kind {
    Kind::Prompt(text) => (
      session.parent_of(&node.id).map(str::to_string),
      Some(text.clone()),
      format!("❯ {}", first_line(&text)),
    ),
    Kind::Step(label) => (Some(node.id.clone()), None, label),
    Kind::ToolCalls => return None,
  };
  Some(Point {
    // Where the session already stands, and selecting it would do nothing.
    // The prompt that was answered here ends the conversation in the same
    // place, but hands itself back to be asked again, which is not nothing.
    here: leaf.as_deref() == session.leaf() && text.is_none(),
    len: session.branch_len(leaf.as_deref()),
    leaf,
    text,
    label,
    depth,
  })
}

/// `1 message` or `4 messages`.
fn messages(n: usize) -> String {
  match n {
    1 => "1 message".into(),
    n => format!("{n} messages"),
  }
}

/// What the footer says about this session's MCP servers, or nothing at all
/// when it has none: a session that never asked for a server should not be
/// told it has no servers.
fn mcp_label(servers: usize, tools: usize) -> Option<String> {
  (servers > 0).then(|| format!("{servers} mcp, {tools} tool{}", if tools == 1 { "" } else { "s" }))
}

/// How a context window `percent` full reads in the footer: out of the way
/// while there is room, yellow once it is worth an eye, red when the next
/// answer may not fit. pi's thresholds.
///
/// Red is rare on a session that compacts — the summary comes at around 87%
/// and takes the figure back down with it — so seeing it means the window is
/// filling with nothing being done about it: compaction turned off, or a
/// single turn too big to summarize.
fn context_style(percent: u64) -> Style {
  match percent {
    90.. => Style::default().fg(Color::Red),
    70.. => Style::default().fg(Color::Yellow),
    _ => Style::default().add_modifier(Modifier::DIM),
  }
}

fn first_line(text: &str) -> String {
  let mut it = text.lines();
  let first = it.next().unwrap_or_default().to_string();
  if it.next().is_some() {
    format!("{first} …")
  } else {
    first
  }
}

fn summarize_args(name: &str, args: &serde_json::Value) -> String {
  let get = |k: &str| args.get(k).and_then(|v| v.as_str()).map(str::to_string);
  let summary = match name {
    "bash" => get("command").map(|c| first_line(&c)),
    "read" => get("path").map(|p| {
      match (
        args.get("offset").and_then(|v| v.as_u64()),
        args.get("limit").and_then(|v| v.as_u64()),
      ) {
        (Some(o), Some(l)) => format!("{p}:{o}-{}", o + l - 1),
        (Some(o), None) => format!("{p}:{o}-"),
        (None, Some(l)) => format!("{p}:1-{l}"),
        (None, None) => p,
      }
    }),
    "write" => get("path"),
    "edit" => get("path").map(|p| {
      let n = args.get("edits").and_then(|e| e.as_array()).map_or(0, Vec::len);
      format!("{p} ({n} edit{})", if n == 1 { "" } else { "s" })
    }),
    // What was asked, which is the line the dialog was drawn over and the
    // line the answer under it is an answer to.
    "ask" => args.get("questions").and_then(|q| q.as_array()).map(|questions| {
      let first = questions
        .first()
        .and_then(|q| q.get("question"))
        .and_then(|q| q.as_str())
        .unwrap_or_default();
      match questions.len() {
        0 | 1 => first.to_string(),
        n => format!("{first} (+{} more)", n - 1),
      }
    }),
    _ => None,
  };
  summary.unwrap_or_else(|| first_line(&args.to_string()))
}

/// What to show of a call the model is still writing.
///
/// Once the arguments parse, the finished summary is exact and is used as-is.
/// Until then only the field that summary would lead with is worth showing —
/// the command, or the path — which is the part being typed anyway.
fn writing_summary(name: &str, args: &str) -> String {
  if let Ok(value) = serde_json::from_str::<serde_json::Value>(args) {
    return summarize_args(name, &value);
  }
  let key = match name {
    "bash" => "command",
    "read" | "write" | "edit" => "path",
    _ => return String::new(),
  };
  partial_str(args, key).map(|text| first_line(&text)).unwrap_or_default()
}

/// The text a call is carrying in its arguments, as far as it has arrived,
/// in blocks with the mark each is drawn under.
///
/// `bash` says all it has to say on its one line, but `write` and `edit` put
/// a file's worth of text in their arguments — the slow part of the call, and
/// the part worth watching arrive. An edit's two halves are marked the way
/// the diff it becomes will mark them.
fn writing_body(name: &str, args: &str) -> Vec<(&'static str, String)> {
  // Once the arguments parse, read them as arguments. Scanning the text is
  // only for what is still half-written, and that has to assume the halves of
  // an edit arrive in the order they were written — which stops being true
  // the moment anything re-serializes them, since that sorts the keys.
  if let Ok(value) = serde_json::from_str::<serde_json::Value>(args) {
    let text = |value: &serde_json::Value, key| value.get(key).and_then(|v| v.as_str()).map(str::to_string);
    return match name {
      "write" => text(&value, "content").map(|body| ("│", body)).into_iter().collect(),
      // Every replacement the call makes, not only the one it is on: a call
      // is the whole set of them, and the ones already written are still
      // part of what it will do.
      "edit" => value
        .get("edits")
        .and_then(|edits| edits.as_array())
        .map(|edits| {
          edits
            .iter()
            .flat_map(|edit| {
              [
                text(edit, "oldText").map(|t| ("-", t)),
                text(edit, "newText").map(|t| ("+", t)),
              ]
            })
            .flatten()
            .collect()
        })
        .unwrap_or_default(),
      _ => Vec::new(),
    };
  }
  match name {
    "write" => args
      .find("\"content\"")
      .and_then(|at| value_after(args, at + 9))
      .map(|body| ("│", body))
      .into_iter()
      .collect(),
    "edit" => replacements(args),
    _ => Vec::new(),
  }
}

/// The replacements written so far, in the order they were written, the last
/// of them as far as it has got.
///
/// Read by scanning because there is nothing parseable yet. A replacement
/// whose own text contained `"newText":` would fool it, which costs a wrongly
/// drawn line until the call finishes and is read properly.
fn replacements(args: &str) -> Vec<(&'static str, String)> {
  let mut out = Vec::new();
  let mut from = 0;
  loop {
    let next = [("-", "oldText"), ("+", "newText")]
      .into_iter()
      .filter_map(|(mark, key)| {
        let at = args[from..].find(&format!("\"{key}\""))? + from;
        Some((at, mark, key.len()))
      })
      .min_by_key(|(at, ..)| *at);
    let Some((at, mark, len)) = next else { return out };
    from = at + len + 2;
    // A value that has not opened yet, or a key with nothing after it, is
    // where the writing has got to.
    let Some(text) = value_after(args, from) else {
      return out;
    };
    out.push((mark, text));
  }
}

/// The value of a string field in a JSON object that is still being written,
/// including one whose closing quote has not arrived yet.
///
/// The key is found by text rather than by parsing, since there is nothing
/// parseable yet. A tool argument that itself contained `"command":` would
/// fool it, which costs a wrong half-drawn line and nothing else.
fn partial_str(json: &str, key: &str) -> Option<String> {
  value_after(json, json.find(&format!("\"{key}\""))? + key.len() + 2)
}

/// Reads the string value that follows a key, from `at`.
fn value_after(json: &str, at: usize) -> Option<String> {
  let rest = json.get(at..)?.trim_start().strip_prefix(':')?.trim_start();
  let mut chars = rest.strip_prefix('"')?.chars();
  let mut out = String::new();
  while let Some(c) = chars.next() {
    match c {
      '"' => break,
      '\\' => match chars.next() {
        Some('n') => out.push('\n'),
        Some('t') => out.push('\t'),
        Some('r') => {}
        Some('u') => {
          // Four hex digits, which may not all have arrived.
          let hex: String = chars.by_ref().take(4).collect();
          if let Some(c) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
            out.push(c);
          }
        }
        Some(escaped) => out.push(escaped),
        // The escape itself is only half here; the rest is on its way.
        None => break,
      },
      c => out.push(c),
    }
  }
  Some(out)
}

fn shorten_home(path: &std::path::Path) -> String {
  let s = path.display().to_string();
  match std::env::var("HOME") {
    Ok(home) if !home.is_empty() && s.starts_with(&home) => format!("~{}", &s[home.len()..]),
    _ => s,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn names(matches: &[Match]) -> Vec<&'static str> {
    matches.iter().map(|m| COMMANDS[m.index].0).collect()
  }

  /// The text of a block of spans, line by line.
  fn text(lines: &[Vec<Span<'static>>]) -> Vec<String> {
    lines
      .iter()
      .map(|spans| spans.iter().map(|span| span.content.as_ref()).collect())
      .collect()
  }

  #[test]
  fn a_structured_answer_is_laid_out_as_the_structure_it_is() {
    let answer = r#"{"city":"Berlin","rain":true,"hours":[1,2]}"#;
    let lines = structured("weather", answer).expect("one long line is the worst way to read this");
    assert_eq!(
      text(&lines),
      [
        " {",
        r#"   "city": "Berlin","#,
        r#"   "rain": true,"#,
        r#"   "hours": ["#,
        "     1,",
        "     2",
        "   ]",
        " }",
      ]
    );

    // What a command printed is the command's own to lay out, and a file is
    // to be seen the way it is written — neither is re-indented for being
    // valid JSON.
    assert!(structured("bash", answer).is_none());
    assert!(structured("read", answer).is_none());
    // Nor is anything that is not a structure to begin with.
    assert!(structured("weather", "It rains in Berlin.").is_none());
    assert!(structured("weather", "42").is_none(), "a number says itself");
    assert!(structured("weather", r#""rain""#).is_none(), "and so does a string");
  }

  #[cfg(feature = "lang-json")]
  #[test]
  fn a_structured_answer_is_highlighted_by_what_it_is() {
    let lines = structured("weather", r#"{"city":"Berlin"}"#).expect("a structure");
    let coloured: Vec<&Span<'static>> = lines
      .iter()
      .flatten()
      .filter(|span| span.style.fg.is_some() && !span.content.trim().is_empty())
      .collect();
    assert!(
      coloured.iter().any(|span| span.content.contains("Berlin")),
      "the values are coloured: {lines:?}"
    );
  }

  #[test]
  fn the_footer_says_what_the_servers_brought_and_nothing_when_there_are_none() {
    // A session that never asked for a server should not be told it has none.
    assert_eq!(mcp_label(0, 0), None);
    assert_eq!(mcp_label(1, 1).as_deref(), Some("1 mcp, 1 tool"));
    assert_eq!(mcp_label(2, 14).as_deref(), Some("2 mcp, 14 tools"));
    // A server that came up with nothing to offer still came up.
    assert_eq!(mcp_label(1, 0).as_deref(), Some("1 mcp, 0 tools"));
  }

  #[test]
  fn a_filling_context_window_is_yellow_before_it_is_red() {
    let colour = |percent| context_style(percent).fg;
    // Room to work in: the figure keeps out of the way.
    assert_eq!(colour(0), None);
    assert_eq!(colour(69), None);
    assert!(context_style(69).add_modifier.contains(Modifier::DIM));
    // Worth an eye, then worth acting on.
    assert_eq!(colour(70), Some(Color::Yellow));
    assert_eq!(colour(89), Some(Color::Yellow));
    assert_eq!(colour(90), Some(Color::Red));
    // A request bigger than the window at all is as red as it gets.
    assert_eq!(colour(400), Some(Color::Red));
  }

  #[test]
  fn command_filter_is_fuzzy_and_ranked() {
    let mut matcher = Matcher::new(Config::DEFAULT);
    assert_eq!(filter_commands(&mut matcher, "").len(), COMMANDS.len());
    assert_eq!(names(&filter_commands(&mut matcher, "res")), ["resume"]);
    // Fuzzy: letters in order, not contiguous.
    assert_eq!(names(&filter_commands(&mut matcher, "nm")), ["name"]);
    // Prefix match ranks above a scattered match.
    assert_eq!(names(&filter_commands(&mut matcher, "se"))[0], "session");
    assert!(filter_commands(&mut matcher, "zzz").is_empty());
    // Highlights point at the matched letters: n-a-m-e for "nm".
    let m = &filter_commands(&mut matcher, "nm")[0];
    assert_eq!(m.highlights, [0, 2]);
  }

  /// A history of one prompt, a tool call answered, and a final answer.
  fn tool_history() -> Vec<Message> {
    let call = AssistantContent::tool_call("1", "read", serde_json::json!({ "path": "a.rs" }));
    let result = UserContent::tool_result("1", "read", vec![ToolResultContent::text("fn main() {}")]);
    vec![
      Message::user("look at a.rs"),
      Message::Assistant {
        id: None,
        content: vec![AssistantContent::text("Let me read it."), call],
      },
      Message::User { content: vec![result] },
      Message::assistant("It is a hello world."),
    ]
  }

  /// An unsaved session holding `history`.
  fn session_of(history: Vec<Message>) -> Session {
    let mut session = Session::new(None, Path::new("/work"), "mock");
    session.append(history).unwrap();
    session
  }

  /// Every row as it reads: indented label, resulting length, prompt handed
  /// back, and whether it is where the session is.
  fn rows(session: &Session) -> Vec<(String, usize, Option<String>, bool)> {
    points(session)
      .into_iter()
      .map(|p| (format!("{}{}", "  ".repeat(p.depth), p.label), p.len, p.text, p.here))
      .collect()
  }

  #[test]
  fn a_prompt_is_taken_back_and_anything_else_is_kept() {
    let session = session_of(vec![
      Message::user("first"),
      Message::assistant("answer"),
      Message::user("second"),
      Message::assistant("reply"),
    ]);
    assert_eq!(
      rows(&session),
      [
        // A prompt is taken back out of the history and handed to the input
        // box, so going there ends the conversation before it.
        ("❯ first".into(), 0, Some("first".into()), false),
        // Everything else is kept as the new end.
        ("answer".into(), 2, None, false),
        ("❯ second".into(), 2, Some("second".into()), false),
        ("reply".into(), 4, None, true),
      ]
    );
  }

  #[test]
  fn a_tool_call_is_not_a_point_but_its_result_is() {
    // Stopping straight after the call would leave it unanswered, which is a
    // transcript no provider will take. The result right after it is the
    // point "between the tool calls".
    let session = session_of(tool_history());
    let labels: Vec<String> = rows(&session).into_iter().map(|r| r.0).collect();
    assert_eq!(labels, ["❯ look at a.rs", "⚙ read", "It is a hello world."]);
  }

  #[test]
  fn forking_offers_only_the_prompts() {
    let session = session_of(tool_history());
    let mut points = points(&session);
    points.retain(|point| point.text.is_some());
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].text.as_deref(), Some("look at a.rs"));
  }

  #[test]
  fn the_prompts_walked_back_through_are_the_session_s_own() {
    let mut history = tool_history();
    // A compaction checkpoint travels as a user message, and is no more a
    // prompt than the tool result above it is.
    history.push(Message::user(format!(
      "{SUMMARY_PREFIX}Files were read.{SUMMARY_SUFFIX}"
    )));
    history.push(Message::user("and now b.rs"));
    let prompts = Prompts::of(&session_of(history));
    assert_eq!(prompts.sent, ["look at a.rs", "and now b.rs"]);
  }

  #[test]
  fn up_walks_back_through_the_prompts_and_down_hands_the_draft_back() {
    let mut prompts = Prompts::default();
    prompts.add("first".into());
    prompts.add("second".into());
    // The same thing sent twice running is one entry to walk past.
    prompts.add("second".into());

    assert_eq!(prompts.previous("half-typed").as_deref(), Some("second"));
    assert_eq!(prompts.previous("").as_deref(), Some("first"));
    // The oldest is as far back as it goes; the box keeps what it has.
    assert_eq!(prompts.previous(""), None);

    assert_eq!(prompts.next().as_deref(), Some("second"));
    // Past the newest is what was being typed when the walk began.
    assert_eq!(prompts.next().as_deref(), Some("half-typed"));
    assert!(!prompts.walking());
    assert_eq!(prompts.next(), None);
  }

  #[test]
  fn sending_something_ends_the_walk_it_was_recalled_by() {
    let mut prompts = Prompts::default();
    prompts.add("first".into());
    prompts.previous("");
    assert!(prompts.walking());

    prompts.add("first, edited".into());
    assert!(!prompts.walking());
    // And Up starts again from the newest, which is what was just sent.
    assert_eq!(prompts.previous("").as_deref(), Some("first, edited"));
  }

  #[test]
  fn a_branch_is_indented_under_the_point_it_left_and_the_live_one_comes_first() {
    let mut session = session_of(vec![Message::user("first"), Message::assistant("one")]);
    let prompt = session.lineage(session.leaf())[0].to_string();
    // Go back under the prompt and answer differently, which forks the tree.
    session.go_to(Some(prompt)).unwrap();
    session.append(vec![Message::assistant("two")]).unwrap();

    assert_eq!(
      rows(&session),
      [
        ("❯ first".into(), 0, Some("first".into()), false),
        // Both answers hang off the prompt, the one in use listed first, and
        // the one walked away from still there to walk back into.
        ("  two".into(), 2, None, true),
        ("  one".into(), 2, None, false),
      ]
    );
  }

  #[test]
  fn a_command_reads_back_as_it_is_written() {
    // What the model sends, a few characters at a time. Every prefix of it
    // has to render as the command so far and nothing else.
    let whole = r#"{"command":"cargo test --all"}"#;
    let seen: Vec<String> = (0..=whole.len())
      .map(|n| writing_summary("bash", &whole[..n]))
      .collect();
    assert_eq!(seen.first().unwrap(), "");
    assert_eq!(seen.last().unwrap(), "cargo test --all");
    // It only ever grows, and never shows the JSON around it.
    for pair in seen.windows(2) {
      assert!(pair[1].starts_with(&pair[0]), "{pair:?}");
      assert!(!pair[1].contains('{') && !pair[1].contains('"'), "{pair:?}");
    }
    assert!(seen.contains(&"cargo te".to_string()), "{seen:?}");
  }

  #[test]
  fn a_half_written_command_keeps_its_escapes_whole() {
    let quote = |args: &str| writing_summary("bash", args);
    assert_eq!(quote(r#"{"command":"echo \"hi"#), "echo \"hi");
    // An escape that is itself half here waits rather than showing a stray
    // backslash.
    assert_eq!(quote(r#"{"command":"echo \"#), "echo ");
    // Only the first line, marked as having more, exactly as the finished
    // summary shows the same command.
    assert_eq!(quote(r#"{"command":"one\ntwo"#), "one …");
    assert_eq!(
      quote(r#"{"command":"one\ntwo"}"#),
      summarize_args("bash", &serde_json::json!({ "command": "one\ntwo" }))
    );
  }

  #[test]
  fn a_file_being_written_shows_up_as_it_arrives() {
    let whole = r#"{"path":"a.rs","content":"fn main() {\n    body();\n}"}"#;
    // The path is on the first line from early on; the content grows under it.
    let at = |n: usize| {
      (
        writing_summary("write", &whole[..n]),
        writing_body("write", &whole[..n]),
      )
    };
    assert_eq!(at(whole.find("\"content\"").unwrap()).1, []);
    // Cut in the middle of the second line, where the model has got to.
    let (path, body) = at(whole.find("body();").unwrap() + 4);
    assert_eq!(path, "a.rs");
    assert_eq!(body, [("│", "fn main() {\n    body".to_string())]);
    assert_eq!(at(whole.len()).1, [("│", "fn main() {\n    body();\n}".to_string())]);
    // Every prefix is a prefix of the whole, so the block only ever grows.
    let full = writing_body("write", whole)[0].1.clone();
    for n in 0..=whole.len() {
      if let [(_, text)] = writing_body("write", &whole[..n]).as_slice() {
        assert!(full.starts_with(text), "{text:?}");
      }
    }
  }

  #[test]
  fn an_edit_shows_its_two_halves_marked_as_the_diff_will_mark_them() {
    let one = r#"{"path":"a.rs","edits":[{"oldText":"was","newText":"is"#;
    assert_eq!(
      writing_body("edit", one),
      [("-", "was".to_string()), ("+", "is".to_string())]
    );
    // Before the new text arrives there is only the old.
    let half = r#"{"path":"a.rs","edits":[{"oldText":"wa"#;
    assert_eq!(writing_body("edit", half), [("-", "wa".to_string())]);
  }

  #[test]
  fn an_edit_shows_every_replacement_it_has_written_so_far() {
    // A call is the whole set of replacements, so the ones already written
    // stay on screen while the next is being typed — they are still part of
    // what the call will do.
    let two = r#"{"edits":[{"oldText":"one","newText":"1"},{"oldText":"tw"#;
    assert_eq!(
      writing_body("edit", two),
      [
        ("-", "one".to_string()),
        ("+", "1".to_string()),
        ("-", "tw".to_string()),
      ]
    );
    // And the replacement being written grows in place rather than
    // displacing the ones before it.
    let two = format!("{two}o\",\"newText\":\"2");
    assert_eq!(
      writing_body("edit", &two),
      [
        ("-", "one".to_string()),
        ("+", "1".to_string()),
        ("-", "two".to_string()),
        ("+", "2".to_string()),
      ]
    );
  }

  #[test]
  fn a_command_has_nothing_to_show_below_its_line() {
    // Everything bash has to say fits on the one line; the block is for the
    // tools that carry a file in their arguments.
    assert_eq!(writing_body("bash", r#"{"command":"ls -la"#), []);
  }

  #[test]
  fn a_written_call_reads_the_same_as_the_finished_one() {
    // The live line and the entry it becomes must agree, or the transcript
    // jumps when the call starts running.
    let args = serde_json::json!({ "path": "src/main.rs", "offset": 10, "limit": 5 });
    assert_eq!(
      writing_summary("read", &args.to_string()),
      summarize_args("read", &args)
    );
    // A tool with nothing worth showing early says nothing, rather than
    // guessing.
    assert_eq!(writing_summary("mystery", r#"{"a":"b"#), "");
  }

  /// Each entry as one line, for asserting what the transcript reads like.
  fn shapes(entries: &[Entry]) -> Vec<String> {
    entries
      .iter()
      .map(|entry| match entry {
        Entry::User(text) => format!("user {text}"),
        Entry::Assistant(text) => format!("said {text}"),
        Entry::ToolCall { name, summary, .. } => format!("call {name} {summary}"),
        Entry::ToolResult { name, output, .. } => format!("out {name} {}", first_line(output)),
        _ => "other".into(),
      })
      .collect()
  }

  #[test]
  fn two_commands_in_one_turn_keep_their_own_output_on_reload() {
    // A turn that ran two commands at once, with the provider answering them
    // in the other order — which is allowed, and which taking the history as
    // it comes would render as both commands and then both outputs.
    let call = |id: &str, cmd: &str| AssistantContent::tool_call(id, "bash", serde_json::json!({ "command": cmd }));
    let result = |id: &str, out: &str| UserContent::tool_result(id, "bash", vec![ToolResultContent::text(out)]);
    let session = session_of(vec![
      Message::user("build and test"),
      Message::Assistant {
        id: None,
        content: vec![
          AssistantContent::text("Doing both."),
          call("1", "cargo build"),
          call("2", "cargo test"),
        ],
      },
      Message::User {
        content: vec![result("2", "test output"), result("1", "build output")],
      },
      Message::assistant("Both fine."),
    ]);
    assert_eq!(
      shapes(&entries_from_history(&session)),
      [
        "user build and test",
        "said Doing both.",
        "call bash cargo build",
        "out bash build output",
        "call bash cargo test",
        "out bash test output",
        "said Both fine.",
      ]
    );
  }

  fn announced(summary: &str, call: &str) -> Entry {
    Entry::ToolCall {
      name: "bash".into(),
      summary: summary.into(),
      call: call.into(),
      wrote: None,
      started: Instant::now(),
    }
  }

  fn finished(output: &str, is_error: bool, call: &str) -> Finished {
    Finished {
      name: "bash".into(),
      output: output.into(),
      images: Vec::new(),
      is_error,
      call: call.into(),
      diff: None,
    }
  }

  #[test]
  fn output_finds_its_own_call_when_two_are_in_flight() {
    // Both commands announced, then output arriving in the other order —
    // which taking the last entry would put under the wrong one.
    let mut entries = vec![announced("slow", "a"), announced("quick", "b")];
    place_output(&mut entries, "b".into(), "quick is talking".into());
    place_output(&mut entries, "a".into(), "slow is talking".into());
    assert_eq!(
      shapes(&entries),
      [
        "call bash slow",
        "out bash slow is talking",
        "call bash quick",
        "out bash quick is talking",
      ]
    );

    // More output replaces that call's own line rather than adding another.
    place_output(&mut entries, "a".into(), "slow said more".into());
    assert_eq!(
      shapes(&entries),
      [
        "call bash slow",
        "out bash slow said more",
        "call bash quick",
        "out bash quick is talking",
      ]
    );

    // And the finished results land in the same places, in whatever order
    // the two commands happen to end.
    place_result(&mut entries, finished("quick done", false, "b"));
    place_result(&mut entries, finished("slow failed", true, "a"));
    assert_eq!(
      shapes(&entries),
      [
        "call bash slow",
        "out bash slow failed",
        "call bash quick",
        "out bash quick done",
      ]
    );
    // Each kept its own verdict.
    let failed: Vec<bool> = entries
      .iter()
      .filter_map(|e| match e {
        Entry::ToolResult { is_error, .. } => Some(*is_error),
        _ => None,
      })
      .collect();
    assert_eq!(failed, [true, false]);
  }

  #[test]
  fn an_abort_stops_the_clock_on_every_command_still_running() {
    // An abort takes down the whole run, so no command it killed is left
    // counting up forever — not just the last one to have spoken.
    let mut entries = vec![announced("slow", "a"), announced("quick", "b")];
    place_output(&mut entries, "a".into(), "a is talking".into());
    place_output(&mut entries, "b".into(), "b is talking".into());
    finish_running(&mut entries);
    let stopped: Vec<bool> = entries
      .iter()
      .filter_map(|entry| match entry {
        Entry::ToolResult { running, took, .. } => Some(!running && took.is_some()),
        _ => None,
      })
      .collect();
    assert_eq!(stopped, [true, true]);
  }

  #[test]
  fn a_new_command_does_not_land_on_a_reloaded_one_of_the_same_id() {
    // Reopened sessions bring their old calls back as entries, and a server
    // that numbers calls from one per turn will hand out an id the
    // transcript already holds. The live one is the one still being written,
    // so the search runs from the end.
    let mut entries = vec![
      announced("old command", "call_1"),
      Entry::ToolResult {
        name: "bash".into(),
        output: "old output".into(),
        images: Vec::new(),
        is_error: false,
        call: "call_1".into(),
        running: false,
        diff: None,
        started: Instant::now(),
        took: None,
      },
      announced("new command", "call_1"),
    ];
    place_output(&mut entries, "call_1".into(), "new output".into());
    place_result(&mut entries, finished("new output", false, "call_1"));
    assert_eq!(
      shapes(&entries),
      [
        "call bash old command",
        "out bash old output",
        "call bash new command",
        "out bash new output",
      ]
    );
  }

  #[test]
  fn an_output_whose_call_is_missing_is_still_shown() {
    // A result with nothing to pair against — a branch that kept the answer
    // but not the question — is drawn where it was written rather than lost.
    let session = session_of(vec![Message::User {
      content: vec![UserContent::tool_result(
        "gone",
        "bash",
        vec![ToolResultContent::text("orphan output")],
      )],
    }]);
    assert_eq!(shapes(&entries_from_history(&session)), ["out bash orphan output"]);
  }

  #[test]
  fn an_aborted_run_keeps_the_work_it_got_through() {
    // A run only hands back its transcript when it reaches the end, so a run
    // stopped part-way has to have kept it as it went — otherwise the tools
    // have changed the files and the conversation denies all knowledge.
    let mut run = InFlight::new(Some(Message::user("fix it")), 0);
    run.said("Let me look.");
    run.called("c1", "read", serde_json::json!({ "path": "a.rs" }));
    run.answered("c1", "read", "fn main() {}".into());
    run.said("Now the edit.");
    run.called("c2", "edit", serde_json::json!({ "path": "a.rs" }));
    run.answered("c2", "edit", "Successfully replaced 1 block(s).".into());
    // Stopped here: mid-sentence, with a third call still running.
    run.said("And now I will");
    run.called("c3", "bash", serde_json::json!({ "command": "sleep 30" }));

    let shapes: Vec<String> = run
      .recovered()
      .iter()
      .map(|message| match message {
        Message::User { content } => match &content[..] {
          [UserContent::Text(t)] => format!("user {}", t.text),
          [UserContent::ToolResult(r)] => format!("result {}", r.name),
          _ => "user ?".into(),
        },
        Message::Assistant { content, .. } => content
          .iter()
          .map(|c| match c {
            AssistantContent::Text(t) => format!("said {:?}", t.text),
            AssistantContent::ToolCall(call) => format!("call {}", call.function.name),
            _ => "?".into(),
          })
          .collect::<Vec<_>>()
          .join(" + "),
        Message::System { .. } => "system".into(),
      })
      .collect();
    assert_eq!(
      shapes,
      [
        "user fix it",
        "said \"Let me look.\" + call read",
        "result read",
        "said \"Now the edit.\" + call edit",
        "result edit",
        // The half-finished sentence is kept too, and the call it was in
        // the middle of is answered rather than left hanging.
        "said \"And now I will\" + call bash",
        "result bash",
      ]
    );
  }

  #[test]
  fn every_call_in_flight_is_answered_whichever_came_back() {
    // Two calls out at once, the second answering first. Both have to end up
    // answered — the one that came back with its own result, the one that
    // did not with an interruption — or the history keeps a call nothing
    // ever replied to, which is a conversation no provider will take back.
    let mut run = InFlight::new(Some(Message::user("both")), 0);
    run.called("c1", "bash", serde_json::json!({ "command": "slow" }));
    run.called("c2", "bash", serde_json::json!({ "command": "quick" }));
    run.answered("c2", "bash", "quick done".into());

    let recovered = run.recovered();
    let answered: Vec<String> = recovered
      .iter()
      .filter_map(|message| match message {
        Message::User { content } => match &content[..] {
          [UserContent::ToolResult(r)] => Some(format!(
            "{}: {}",
            r.call.as_str(),
            match &r.content[..] {
              [ToolResultContent::Text(t)] => t.text.clone(),
              _ => String::new(),
            }
          )),
          _ => None,
        },
        _ => None,
      })
      .collect();
    assert_eq!(answered, ["c2: quick done".to_string(), format!("c1: {ABORTED}")]);
  }

  #[test]
  fn a_run_that_did_nothing_leaves_only_what_was_asked() {
    let mut run = InFlight::new(Some(Message::user("hello")), 0);
    run.said("   ");
    assert_eq!(run.recovered(), [Message::user("hello")]);
    // A `/continue` run has no prompt of its own, and nothing yet to keep.
    assert!(InFlight::new(None, 1).recovered().is_empty());
  }

  /// The text of each line, and the colours it carries.
  fn painted(lines: &[Vec<Span<'static>>]) -> Vec<(String, Vec<Option<Color>>)> {
    lines
      .iter()
      .map(|line| {
        let text = line.iter().map(|span| span.content.as_ref()).collect::<String>();
        let colours = line
          .iter()
          .filter(|s| !s.content.trim().is_empty())
          .map(|s| s.style.fg)
          .collect();
        (text, colours)
      })
      .collect()
  }

  #[test]
  fn a_written_file_is_shown_as_the_file_it_is() {
    // Plain: no diff's pluses down the side, and none of its green — a write
    // put the file there, it did not change it.
    let plain = painted(&file_lines("notes.txt", "hello\nthere"));
    assert_eq!(
      plain,
      [(" hello".to_string(), vec![None]), (" there".to_string(), vec![None])]
    );
    assert!(plain.iter().all(|(text, _)| !text.trim_start().starts_with('+')));
  }

  #[test]
  #[cfg(feature = "lang-rust")]
  fn a_written_file_is_highlighted_by_the_name_it_was_written_to() {
    // The extension is all the transcript has to go on, and all it needs.
    let code = file_lines("src/main.rs", "fn main() {}");
    let spans: Vec<(String, Option<Color>)> = code[0]
      .iter()
      .map(|span| (span.content.to_string(), span.style.fg))
      .collect();
    assert!(
      spans.contains(&("fn".to_string(), Some(Color::Magenta))),
      "a keyword is a keyword: {spans:?}"
    );
    assert!(
      spans.contains(&("main".to_string(), Some(Color::Blue))),
      "and a name is a name: {spans:?}"
    );
    // A name that says nothing about its language is shown as it is.
    let unknown = painted(&file_lines("notes", "fn main() {}"));
    assert_eq!(unknown, [(" fn main() {}".to_string(), vec![None])]);
  }

  #[test]
  fn only_a_write_carries_its_file_and_only_from_its_own_call() {
    let args = serde_json::json!({ "path": "a.rs", "content": "fn main() {}" });
    assert_eq!(wrote_content("write", &args).as_deref(), Some("fn main() {}"));
    assert!(wrote_content("read", &args).is_none(), "a read writes nothing");
    assert!(wrote_content("bash", &serde_json::json!({ "command": "ls" })).is_none());

    let entries = vec![
      Entry::ToolCall {
        name: "write".into(),
        summary: "a.rs".into(),
        call: "c1".into(),
        wrote: Some("fn main() {}".into()),
        started: Instant::now(),
      },
      announced("ls", "c2"),
    ];
    assert_eq!(wrote_by(&entries, "c1"), Some(("a.rs", "fn main() {}")));
    assert_eq!(wrote_by(&entries, "c2"), None, "a command wrote no file");
  }

  #[test]
  fn an_empty_session_has_nowhere_to_go() {
    assert!(points(&session_of(vec![])).is_empty());
  }

  #[test]
  fn going_back_leaves_what_came_after_on_the_list_to_go_forward_to() {
    let mut session = session_of(vec![
      Message::user("first"),
      Message::assistant("one"),
      Message::user("second"),
      Message::assistant("two"),
    ]);
    let answered = session.lineage(session.leaf())[1].to_string();
    session.go_to(Some(answered)).unwrap();

    // The rest of the conversation is still listed, and still somewhere to
    // go — forward is the same move as back, in the other direction.
    assert_eq!(
      rows(&session),
      [
        ("❯ first".into(), 0, Some("first".into()), false),
        ("one".into(), 2, None, true),
        // Only one row is where we stand: this prompt ends the conversation
        // in the same place, but hands itself back, which is not nothing.
        ("❯ second".into(), 2, Some("second".into()), false),
        ("two".into(), 4, None, false),
      ]
    );
  }
}
