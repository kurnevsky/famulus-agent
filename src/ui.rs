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
use ratatui_textarea::{TextArea, WrapMode};
use rig_core::completion::{Message, Usage};
use rig_core::message::{AssistantContent, ToolResultContent, UserContent};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::agent::{AgentEvent, Agents, start_compaction, start_run};
use crate::compaction::{self, SUMMARY_PREFIX, SUMMARY_SUFFIX, Settings};
use crate::session::{Node, NodeKind, Session, SessionInfo, Store};
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32Str};

const MAX_INPUT_LINES: usize = 8;
const TOOL_OUTPUT_LINES: usize = 10;
const DIFF_LINES: usize = 30;
const REASONING_LINES: usize = 6;
/// Reasoning text is indented under its `· thinking…` header.
const REASONING_INDENT: &str = "  ";
/// Cache tags, so two entry kinds holding the same text stay apart.
const ASSISTANT_KIND: u8 = 0;
const REASONING_KIND: u8 = 1;
/// Transcript lines moved per mouse wheel notch.
const WHEEL_LINES: usize = 3;
/// How long the `auto` scrollbar stays visible after the last scroll.
const SCROLLBAR_HIDE_DELAY: Duration = Duration::from_millis(1000);
/// How close together two `Esc` presses count as one double press, as in pi.
const DOUBLE_ESC: Duration = Duration::from_millis(500);
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
  /// Assistant text streamed so far.
  partial: String,
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
      OverlayList::Sessions(_) => &[],
    }
  }

  fn len(&self) -> usize {
    match &self.list {
      OverlayList::Sessions(sessions) => sessions.len(),
      _ => self.points().len(),
    }
  }

  fn title(&self) -> &'static str {
    match &self.list {
      OverlayList::Sessions(_) => " Resume session — ↑↓ select · Enter resume · Esc cancel ",
      OverlayList::Tree(_) => " Tree — ↑↓ PgUp/PgDn select · Enter go there · Esc cancel ",
      OverlayList::Fork(_) => " Fork — ↑↓ PgUp/PgDn select · Enter fork · Esc cancel ",
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
    started: Instant,
  },
  ToolResult {
    name: String,
    output: String,
    is_error: bool,
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

pub struct App {
  agents: Agents,
  settings: Settings,
  scrollbar: ScrollbarMode,
  /// When the transcript was last scrolled, for the `auto` scrollbar.
  last_scroll: Option<Instant>,
  model: String,
  cwd: PathBuf,
  store: Option<Store>,
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
  tx: mpsc::UnboundedSender<AgentEvent>,
  run: Option<JoinHandle<()>>,
  /// The background task in `run` is a compaction rather than a model turn.
  compacting: bool,
  /// The run in flight, kept for abort recovery.
  in_flight: Option<InFlight>,
  /// Tool calls the model is still writing, oldest first.
  writing: Vec<Writing>,
  /// Calls in this run whose result was an error, by the id the transcript
  /// will name them with. Handed to the session so a reload draws them red
  /// again; the transcript itself does not record it.
  failing: HashSet<String>,
  queued: VecDeque<String>,
  /// Transcript position. `None` follows new output at the bottom; `Some`
  /// is a fixed offset from the top, so appended text does not move the view.
  anchor: Option<usize>,
  /// Offset and maximum offset used by the last draw, for relative scrolling.
  view: (usize, usize),
  /// Ctrl+T shows reasoning blocks in full instead of their last few lines.
  expand_thinking: bool,
  /// Rendered markdown, keyed by message text and width rather than by entry,
  /// so reloading a session or compacting cannot serve another entry's lines.
  /// Rebuilt each draw by moving live entries across, which evicts the rest.
  markdown: HashMap<(u64, u16, bool), Vec<Line<'static>>>,
  usage: Usage,
  /// Size of the last completion request, for the footer and compaction.
  /// `None` until the provider reports usage for a request: the figure is
  /// hidden after compaction rather than shown as an estimate.
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
    } = options;
    let mut input = TextArea::default();
    input.set_cursor_line_style(Style::default());
    input.set_placeholder_text(
      "Ask anything. Enter sends, Alt+Enter inserts a newline, /new /resume /compact, Ctrl+C quits.",
    );
    input.set_placeholder_style(Style::default().dim());
    input.set_wrap_mode(WrapMode::WordOrGlyph);
    let session = Session::new(store.as_ref(), &cwd, &model);
    let mut app = Self {
      agents,
      settings,
      scrollbar,
      last_scroll: None,
      model,
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
      tx,
      run: None,
      compacting: false,
      in_flight: None,
      writing: Vec::new(),
      failing: HashSet::new(),
      queued: VecDeque::new(),
      anchor: None,
      view: (0, 0),
      expand_thinking: false,
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
        // Expanding moves everything below the block, so keep the view
        // pinned rather than leaving the reader mid-paragraph.
        self.reset_view();
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
      (KeyCode::PageUp, _) => self.scroll_by(10),
      (KeyCode::PageDown, _) => self.scroll_by(-10),
      (KeyCode::Enter, _) if is_newline(&key) => {
        self.input.insert_newline();
        self.refresh_completion();
      }
      (KeyCode::Char('j'), true) => {
        self.input.insert_newline();
        self.refresh_completion();
      }
      (KeyCode::Enter, _) => self.submit(),
      _ => {
        if self.input.input(key) {
          self.completion_dismissed = false;
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

  /// Replace whatever is in the input box, without disturbing the yank buffer
  /// — the user's own cut text is theirs, not ours to overwrite.
  fn set_input(&mut self, text: &str) {
    self.input.select_all();
    self.input.cut();
    self.input.set_yank_text("");
    self.input.insert_str(text);
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
    self.set_input("");
    self.completion = None;
    self.anchor = None;

    if self.run.is_some() && !matches!(text.as_str(), "/quit" | "/new") {
      self.entries.push(Entry::Info(format!("Queued: {}", first_line(&text))));
      self.queued.push_back(text);
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

  fn reset_view(&mut self) {
    self.queued.clear();
    self.usage = Usage::new();
    self.context_tokens = None;
    self.anchor = None;
  }

  fn new_session(&mut self) {
    self.abort();
    self.session = Session::new(self.store.as_ref(), &self.cwd, &self.model);
    self.entries.clear();
    self.reset_view();
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
        self.reset_view();
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
    self.entries.push(Entry::Info(note));
    // The token counts and the queue belonged to a conversation that is no
    // longer the one we are in.
    self.reset_view();
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

  fn next_queued(&mut self) {
    if let Some(next) = self.queued.pop_front() {
      self.dispatch(next);
    }
  }

  fn compact(&mut self) {
    if self.session.history.is_empty() {
      self.entries.push(Entry::Info("Nothing to compact.".into()));
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
    self.in_flight = Some(InFlight {
      resumed: 0,
      prompt: Some(prompt),
      partial: String::new(),
    });
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
    self.in_flight = Some(InFlight {
      resumed: 1,
      prompt: None,
      partial: String::new(),
    });
  }

  fn abort(&mut self) {
    let Some(handle) = self.run.take() else {
      return;
    };
    handle.abort();
    // A call the model had not finished writing was never run, and the
    // results of this turn are not the next turn's to record.
    self.writing.clear();
    self.failing.clear();
    if self.compacting {
      self.compacting = false;
      self.entries.push(Entry::Info("Compaction aborted.".into()));
    } else {
      self.recover_in_flight();
      self.finish_running_tool();
      self.entries.push(Entry::Info("Aborted.".into()));
    }
  }

  /// Freeze a live tool output entry when its command was killed.
  fn finish_running_tool(&mut self) {
    if let Some(Entry::ToolResult {
      running, took, started, ..
    }) = self.entries.last_mut()
      && *running
    {
      *running = false;
      *took = Some(started.elapsed());
    }
  }

  /// The run never reached its final response, so rig did not hand back the
  /// updated transcript. Keep what the user saw. The messages a `/continue`
  /// resumed from are already in the history and stay there.
  fn recover_in_flight(&mut self) {
    let Some(InFlight { prompt, partial, .. }) = self.in_flight.take() else {
      return;
    };
    let mut messages: Vec<Message> = prompt.into_iter().collect();
    if !partial.trim().is_empty() {
      messages.push(Message::assistant(partial));
    }
    if messages.is_empty() {
      return;
    }
    let result = self.session.append(messages);
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
          in_flight.partial.push_str(&delta);
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
      AgentEvent::ToolCall { name, args } => {
        // The oldest call still being written is this one: calls are run in
        // the order they were written, so it stops being a line the model is
        // typing and becomes one the transcript keeps.
        if !self.writing.is_empty() {
          self.writing.remove(0);
        }
        let summary = summarize_args(&name, &args);
        self.entries.push(Entry::ToolCall {
          name,
          summary,
          started: Instant::now(),
        });
      }
      AgentEvent::ToolOutput(text) => match self.entries.last_mut() {
        Some(Entry::ToolResult {
          running: true, output, ..
        }) => *output = text,
        Some(Entry::ToolCall { name, started, .. }) => {
          let (name, started) = (name.clone(), *started);
          self.entries.push(Entry::ToolResult {
            name,
            output: text,
            is_error: false,
            running: true,
            diff: None,
            started,
            took: None,
          });
        }
        _ => {}
      },
      AgentEvent::ToolResult {
        name,
        output,
        is_error,
        call,
        diff,
      } => {
        if is_error && let Some(call) = call {
          self.failing.insert(call);
        }
        if matches!(self.entries.last(), Some(Entry::ToolResult { running: true, .. })) {
          self.entries.pop();
        }
        let started = match self.entries.last() {
          Some(Entry::ToolCall { started, .. }) => *started,
          _ => Instant::now(),
        };
        self.entries.push(Entry::ToolResult {
          name,
          output,
          is_error,
          running: false,
          diff,
          started,
          took: Some(started.elapsed()),
        });
      }
      AgentEvent::Error(err) => {
        self.entries.push(Entry::Error(err));
      }
      AgentEvent::Done {
        messages,
        usage,
        context_tokens,
      } => {
        self.run = None;
        self.writing.clear();
        // A `/continue` run echoes back the messages it resumed from; they
        // are already in the history.
        let resumed = self.in_flight.take().map_or(0, |f| f.resumed);
        let failing = std::mem::take(&mut self.failing);
        let result = self
          .session
          .append_failing(messages.into_iter().skip(resumed).collect(), &failing);
        self.report(result);
        self.usage.input_tokens += usage.input_tokens;
        self.usage.output_tokens += usage.output_tokens;
        self.usage.total_tokens += usage.total_tokens;
        // Fall back to a chars/4 estimate only when the provider reports
        // no usage at all.
        let context_tokens = if context_tokens > 0 {
          context_tokens
        } else {
          compaction::estimate_tokens(&self.session.history)
        };
        self.context_tokens = Some(context_tokens);
        if compaction::should_compact(context_tokens, &self.settings) {
          self.compact();
        } else {
          self.next_queued();
        }
      }
      AgentEvent::Ended => {
        self.run = None;
        self.writing.clear();
        self.failing.clear();
        if self.compacting {
          self.compacting = false;
        } else {
          self.recover_in_flight();
        }
        self.next_queued();
      }
      AgentEvent::Compacted(result) => {
        self.run = None;
        self.compacting = false;
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
          }
          None => self.entries.push(Entry::Info("Nothing to compact.".into())),
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
      .map(|pct| format!("ctx {pct}%  "))
      .unwrap_or_default();
    let right = format!("{context}{}↑ {}↓", self.usage.input_tokens, self.usage.output_tokens);
    let pad = (footer_area.width as usize).saturating_sub(Line::from(left.clone()).width() + right.chars().count());
    left.push(Span::raw(" ".repeat(pad)));
    left.push(Span::raw(right).dim());
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
    for (i, entry) in self.entries.iter().enumerate() {
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
          let streaming = streaming == Some(i);
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
          lines.push(Line::from(vec![
            Span::styled("⚙ ", Style::default().fg(Color::Yellow)),
            Span::styled(name.clone(), Style::default().fg(Color::Yellow).bold()),
            Span::raw(" "),
            Span::styled(summary.clone(), dim),
          ]));
        }
        Entry::ToolResult {
          name,
          output,
          is_error,
          running,
          diff,
          started,
          took,
        } => {
          if let Some(diff) = diff {
            // An edit is shown as a diff: removed red, added green, context
            // dim, capped like other tool output.
            let all: Vec<&str> = diff.lines().collect();
            let shown = all.len().min(DIFF_LINES);
            for l in &all[..shown] {
              let style = match l.chars().next() {
                Some('+') => Style::default().fg(Color::Green),
                Some('-') => Style::default().fg(Color::Red),
                _ => dim,
              };
              lines.push(Line::styled(format!("  {l}"), style));
            }
            if all.len() > shown {
              lines.push(Line::styled(format!("  … {} more lines", all.len() - shown), dim));
            }
            continue;
          }
          // A command is worth reading as pass or fail at a glance, so its
          // output carries the verdict. Other tools stay dim: a file read
          // back in green would be colour for its own sake, and the text of
          // a file has its own reasons to be coloured.
          let style = if *is_error {
            Style::default().fg(Color::Red)
          } else if name == "bash" && !*running {
            // Still running is not yet a verdict.
            Style::default().fg(Color::Green)
          } else {
            dim
          };
          let all: Vec<&str> = output.lines().collect();
          let shown = all.len().min(TOOL_OUTPUT_LINES);
          let hidden = all.len() - shown;
          // Command output is most useful at its end; file contents at the
          // start.
          if name == "bash" {
            if hidden > 0 {
              lines.push(Line::styled(format!("  │ … {hidden} earlier lines"), dim));
            }
            for l in &all[hidden..] {
              lines.push(Line::styled(format!("  │ {l}"), style));
            }
          } else {
            for l in &all[..shown] {
              lines.push(Line::styled(format!("  │ {l}"), style));
            }
            if hidden > 0 {
              lines.push(Line::styled(format!("  │ … {hidden} more lines"), dim));
            }
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
      lines.push(Line::from(vec![
        Span::styled("⚙ ", Style::default().fg(Color::Yellow)),
        Span::styled(writing.name.clone(), Style::default().fg(Color::Yellow).bold()),
        Span::raw(" "),
        Span::styled(writing_summary(&writing.name, &writing.args), dim),
        Span::styled("▌", Style::default().fg(Color::Yellow)),
      ]));
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

/// Puts `lead` in front of a rendered line, keeping the rest of its spans.
fn prefix(lead: &'static str, line: Line<'static>) -> Line<'static> {
  let mut spans = Vec::with_capacity(line.spans.len() + 1);
  spans.push(Span::raw(lead));
  spans.extend(line.spans);
  Line::from(spans)
}

/// Alt+Enter and Shift+Enter (where the terminal reports it) insert a newline.
fn is_newline(key: &KeyEvent) -> bool {
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

/// Rebuild the transcript view from a resumed session's history.
fn entries_from_history(session: &Session) -> Vec<Entry> {
  let history = &session.history;
  let now = Instant::now();
  let mut entries = Vec::new();
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
            UserContent::ToolResult(r) => {
              let output = r
                .content
                .iter()
                .map(|c| match c {
                  ToolResultContent::Text(t) => t.text.clone(),
                  ToolResultContent::Json { value } => value.to_string(),
                  ToolResultContent::Image(_) => "[image]".to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n");
              // A failed tool result reads like any other in the transcript,
              // so whether it was one is something the session remembers —
              // against this result's own call, not the message's, since one
              // message can answer several calls with different luck.
              let is_error = crate::session::result_ids(r).any(|id| session.failed(&id));
              entries.push(Entry::ToolResult {
                name: r.name.clone(),
                output,
                is_error,
                running: false,
                diff: None,
                started: now,
                took: None,
              });
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
            AssistantContent::ToolCall(call) => entries.push(Entry::ToolCall {
              name: call.function.name.clone(),
              summary: summarize_args(&call.function.name, &call.function.arguments),
              started: now,
            }),
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

/// The value of a string field in a JSON object that is still being written,
/// including one whose closing quote has not arrived yet.
///
/// The key is found by text rather than by parsing, since there is nothing
/// parseable yet. A tool argument that itself contained `"command":` would
/// fool it, which costs a wrong half-drawn line and nothing else.
fn partial_str(json: &str, key: &str) -> Option<String> {
  let at = json.find(&format!("\"{key}\""))? + key.len() + 2;
  let rest = json[at..].trim_start().strip_prefix(':')?.trim_start();
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
