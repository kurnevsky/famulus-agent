//! Ratatui front-end: a scrolling transcript, a multi-line input box and a
//! one-line footer.

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use ratatui::crossterm::event::{
  self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};
use ratatui::layout::{Constraint, Layout, Margin, Position, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Padding, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};
use ratatui::{DefaultTerminal, Frame};
use ratatui_textarea::{CursorMove, TextArea, WrapMode};
use rig_core::completion::{Message, Usage};
#[cfg(test)]
use rig_core::message::ToolResultContent;
use rig_core::message::{AssistantContent, ToolCall, ToolResult, UserContent};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::agent::{self, AgentEvent, Agents, ModelInfo, start_compaction, start_run};
use crate::ask::{self, Dialog};
use crate::attach::{self, Prompt, Token};
use crate::compaction::{DEFAULT_CONTEXT_WINDOW, SUMMARY_PREFIX, SUMMARY_SUFFIX, estimate_tokens};
use crate::session::{Node, NodeKind, Outcome, Session, SessionInfo, Store};
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32Str};

/// How long a provider is given to say what models it has. Long enough for a
/// slow endpoint, short enough that a silent one is answered rather than
/// waited on.
const LISTING_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_INPUT_LINES: usize = 8;
const TOOL_OUTPUT_LINES: usize = 10;
const DIFF_LINES: usize = 30;
const REASONING_LINES: usize = 6;
/// Lines of a context summary shown before it folds.
const SUMMARY_LINES: usize = 10;
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
const SUMMARY_KIND: u8 = 3;
/// Transcript lines moved per mouse wheel notch.
const WHEEL_LINES: usize = 3;
/// How long the `auto` scrollbar stays visible after the last scroll.
const SCROLLBAR_HIDE_DELAY: Duration = Duration::from_millis(1000);
/// How long a toast stays in the corner of the transcript: a moment to
/// notice it, and more for each character there is to read, up to a limit.
const TOAST_DELAY: Duration = Duration::from_millis(1500);
const TOAST_PER_CHAR: Duration = Duration::from_millis(40);
const TOAST_MAX_DELAY: Duration = Duration::from_millis(5000);
/// How close together two `Esc` presses count as one double press, as in pi.
const DOUBLE_ESC: Duration = Duration::from_millis(500);
/// What a tool call the user stopped is answered with, so the model knows it
/// did not simply go unheard.
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

/// Where the last draw left the scrollbar, so the mouse can find its thumb:
/// ratatui draws one but does not say where it put it.
#[derive(Clone, Copy)]
struct Thumb {
  /// The single column the whole scrollbar occupies, track included.
  track: Rect,
  /// Rows from the top of the track to the top of the thumb.
  start: u16,
  /// Rows the thumb covers.
  len: u16,
}

/// Where ratatui's `Scrollbar` draws its thumb within a track of `track` rows:
/// the rows from the top of the track to the top of the thumb, and the rows it
/// covers. Mirrors the widget's own arithmetic, which it keeps to itself.
fn thumb_bounds(track: u16, max_scroll: usize, viewport: usize, offset: usize) -> (u16, u16) {
  // The widget rounds to nearest rather than truncating.
  let divide = |n: usize, d: usize| (n + d / 2) / d;
  let track = track as usize;
  // The scrollbar is told `max_scroll` as its content length, and counts
  // positions from the last one rather than from one past it.
  let span = max_scroll.saturating_sub(1) + viewport;
  if track == 0 || span == 0 {
    return (0, track as u16);
  }
  let len = divide(viewport * track, span).clamp(1, track);
  let start = divide(offset.min(max_scroll.saturating_sub(1)) * track, span).min(track - len);
  (start as u16, len as u16)
}

/// What the mouse has picked out of the transcript, in the coordinates of the
/// lines the last draw wrapped it into: which line a cell is on, and which
/// column of it.
///
/// `anchor` is where the press landed and `head` where the cursor has got to,
/// in either order — a selection can be dragged upwards as well as down. Both
/// are cells rather than the boundaries between them, so the cell under the
/// cursor is part of what is selected: at a cell's own resolution, a drag
/// across a word means the word.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Selection {
  anchor: (usize, usize),
  head: (usize, usize),
}

impl Selection {
  /// The two ends in reading order.
  fn ends(self) -> ((usize, usize), (usize, usize)) {
    match self.anchor <= self.head {
      true => (self.anchor, self.head),
      false => (self.head, self.anchor),
    }
  }

  /// The columns of line `at` this selection covers, end exclusive, or `None`
  /// for a line it does not reach. Every line between its ends is covered
  /// whole, however long it turns out to be.
  fn columns(self, at: usize) -> Option<(usize, usize)> {
    let ((first, from), (last, to)) = self.ends();
    if at < first || at > last {
      return None;
    }
    let from = if at == first { from } else { 0 };
    let to = if at == last { to + 1 } else { usize::MAX };
    Some((from, to))
  }

  /// A press that went nowhere: a click, which selects nothing.
  fn is_empty(self) -> bool {
    self.anchor == self.head
  }
}

/// The scroll offset that puts the top of the thumb `start` rows down its
/// track — the inverse of [`thumb_bounds`], mapped so that both ends of the
/// track reach both ends of the transcript.
fn thumb_offset(start: isize, thumb: Thumb, max_scroll: usize) -> usize {
  let travel = (thumb.track.height.saturating_sub(thumb.len) as isize).max(1);
  let start = start.clamp(0, travel) as usize;
  (start * max_scroll + travel as usize / 2) / travel as usize
}

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
  pub cwd: PathBuf,
  /// `--context-window`, when it was given: a figure the user named stands
  /// whatever model is chosen, and only its absence leaves the window to
  /// what the provider says the model holds.
  pub context_window: Option<u64>,
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
  ("goto", "Scroll to an earlier prompt", false),
  ("model", "Choose the model, or name one: /model <id>", true),
  ("name", "Set session display name", true),
  ("new", "Start a new session", false),
  ("resume", "Resume a different session", false),
  ("session", "Show session info and stats", false),
  ("tree", "Go back to an earlier point (or press Esc twice)", false),
  ("quit", "Quit fa", false),
];
/// Rows shown in the command popup.
const COMPLETION_ROWS: usize = 5;
/// Paths gathered for the `@` popup before it is cut to what fits. More than
/// the rows shown, so scrolling the list has somewhere to go, and few enough
/// that a directory of a thousand images is not measured a keystroke.
const PATH_ROWS: usize = 20;

/// One popup row, with the characters the query matched in it picked out.
enum Match {
  /// Index into `COMMANDS`.
  Command { index: usize, highlights: Vec<u32> },
  /// A path an `@token` is reaching for: what the token becomes when it is
  /// taken, how the row reads, and what the right-hand column says about it.
  Path {
    /// The whole token, `@` and quotes and all.
    insert: String,
    /// The part of it the row shows and the query matched against.
    name: String,
    highlights: Vec<u32>,
    /// Dimensions for an image, and nothing for a directory.
    meta: String,
    /// A directory, which is carried on into rather than attached.
    dir: bool,
  },
}

/// The popup under the input box, best match first: `/` commands, or the
/// files an `@` is reaching for.
struct Completion {
  items: Vec<Match>,
  selected: usize,
  /// The byte range accepting a row replaces. `None` for the command list,
  /// which replaces the whole line.
  replacing: Option<(usize, usize)>,
}

/// A list drawn over the transcript. Moving through one and dismissing it are
/// the same whichever list it is; only the rows and what `Enter` does differ.
struct Overlay {
  list: OverlayList,
  selected: usize,
  /// What the list is narrowed by.
  filter: Filter,
  /// `Ctrl+D` has been pressed on the selected row and the deletion is
  /// waiting to be confirmed. What either list deletes is gone for good, so
  /// it is asked about first.
  confirming: bool,
}

/// The query a list is being narrowed by, and what it leaves.
///
/// The query is typed in the box at the bottom of the screen, which is the
/// box a prompt is typed in and the same widget — so a query is edited with
/// every key a prompt is edited with, the arrows and `Home` and `End` and the
/// Ctrl keys included, rather than with the two a filter in a title could
/// spare. The prompt the box was holding is put back the moment the list is
/// dismissed: the box is borrowed, not taken.
struct Filter {
  field: TextArea<'static>,
  /// Which rows of the list the query leaves, best match first, and which
  /// letters of each it matched — for drawing them picked out, as the `/`
  /// popup picks out its own. Worked out when the query changes rather than
  /// while drawing, because matching is what the matcher does and drawing
  /// does not get to borrow it.
  shown: Vec<(usize, Vec<u32>)>,
}

impl Filter {
  /// A filter over a list of `len` rows, with nothing typed yet: every row is
  /// shown, in the order the list was built in, with nothing picked out in
  /// any of them.
  fn new(len: usize) -> Self {
    let mut field = TextArea::default();
    field.set_cursor_line_style(Style::default());
    field.set_placeholder_text("Type to filter.");
    field.set_placeholder_style(Style::default().fg(Color::DarkGray));
    Self {
      field,
      shown: (0..len).map(|at| (at, Vec::new())).collect(),
    }
  }

  /// What has been typed, as the one line it is: `Enter` is the list's, so
  /// there is never a second one.
  fn query(&self) -> String {
    self.field.lines().join("")
  }
}

enum OverlayList {
  /// `/resume`: every saved session, most recent first.
  Sessions(Vec<SessionInfo>),
  /// `/tree`: every point this session can go back to, oldest first.
  Tree(Vec<Point>),
  /// `/fork`: the prompts, to start a new session from one of them.
  Fork(Vec<Point>),
  /// `/model`: what the provider last said it offers.
  Models(Vec<ModelInfo>),
  /// `/goto`: the prompts on screen, to scroll the transcript to one of them.
  Goto(Vec<Mark>),
}

/// A prompt in the transcript, as `/goto` lists it.
struct Mark {
  /// Index into the transcript's entries.
  entry: usize,
  /// The prompt's first line.
  label: String,
}

/// What the `ask` tool put to the user, and the channel the answer goes back
/// down. Drawn over whatever else is on screen, a list included: it takes the
/// keyboard whole until it is answered.
struct Question {
  dialog: Dialog,
  /// Dropping it unanswered is what tells the tool the user walked away.
  reply: oneshot::Sender<ask::Outcome>,
}

/// A point the conversation can be moved to.
struct Point {
  /// The entry the row stands for, which is what deleting it removes.
  id: String,
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
  /// A list the filter has yet to touch, at the row `selected`.
  fn new(list: OverlayList, selected: usize) -> Self {
    let len = match &list {
      OverlayList::Sessions(sessions) => sessions.len(),
      OverlayList::Tree(points) | OverlayList::Fork(points) => points.len(),
      OverlayList::Models(models) => models.len(),
      OverlayList::Goto(marks) => marks.len(),
    };
    Self {
      list,
      selected,
      filter: Filter::new(len),
      confirming: false,
    }
  }

  /// Rows the list is showing, which is what the filter left of it.
  fn len(&self) -> usize {
    self.filter.shown.len()
  }

  /// Which row of the list the cursor is on: an index into the whole of it,
  /// rather than the place in the narrowed list the cursor stands at.
  fn at(&self) -> Option<usize> {
    self.filter.shown.get(self.selected).map(|(at, _)| *at)
  }

  /// The letters of each shown row the query matched, for picking out.
  fn matched(&self) -> &[(usize, Vec<u32>)] {
    &self.filter.shown
  }

  /// Work out again which rows the query leaves. Called when the query
  /// changes, and when a row has gone from under it.
  fn refilter(&mut self, matcher: &mut Matcher) {
    let Overlay { list, filter, .. } = self;
    let query = &filter.query();
    filter.shown = match list {
      OverlayList::Sessions(sessions) => filter_sessions(matcher, sessions, query),
      OverlayList::Tree(points) => filter_points(matcher, points, query),
      // The fork list draws its prompts flat, so that is what is matched:
      // what is picked out has to sit under what was typed.
      OverlayList::Fork(points) => filter_rows(matcher, points.iter().map(|p| p.label.as_str()), query),
      OverlayList::Models(models) => filter_models(matcher, models, query),
      OverlayList::Goto(marks) => filter_rows(matcher, marks.iter().map(|m| m.label.as_str()), query),
    };
  }

  fn title(&self) -> String {
    match &self.list {
      // A deletion waiting to be confirmed says so where the keys are said,
      // since the keys it is waiting for are not the usual ones.
      OverlayList::Sessions(_) if self.confirming => " Delete session? — Ctrl+D confirm · Esc cancel ".into(),
      OverlayList::Tree(_) if self.confirming => " Delete branch? — Ctrl+D confirm · Esc cancel ".into(),
      // What narrows the list is said by the box the query is typed in,
      // which is on screen under the list saying it. What is left to say up
      // here is what the keys the box does not take do.
      OverlayList::Sessions(_) => " Resume — ↑↓ select · Enter resume · Ctrl+D delete · Esc cancel ".into(),
      OverlayList::Tree(_) => " Tree — ↑↓ PgUp/PgDn select · Enter go there · Ctrl+D delete · Esc cancel ".into(),
      OverlayList::Fork(_) => " Fork — ↑↓ PgUp/PgDn select · Enter fork · Esc cancel ".into(),
      OverlayList::Models(_) => " Model — ↑↓ select · Enter use · Esc cancel ".into(),
      OverlayList::Goto(_) => " Go to — ↑↓ PgUp/PgDn select · Enter scroll there · Esc cancel ".into(),
    }
  }
}

/// A tool call whose arguments are still arriving, shown at the end of the
/// transcript so the command can be read as the model writes it.
///
/// It is replaced by a real entry when the call starts running, which is
/// when the run reports it — not when the last of its text arrives.
struct Writing {
  id: String,
  name: String,
  /// The JSON so far, usually not yet parseable.
  args: String,
}

enum Entry {
  User {
    text: String,
    /// Images the prompt attached, drawn under it as half-blocks — the same
    /// bytes the model was given, kept as bytes because what they are drawn
    /// as depends on how wide the transcript is when they are drawn.
    images: Vec<Vec<u8>>,
  },
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
    /// The file an `edit` changed, which is what says how to read the diff it
    /// leaves behind. The summary says it too, but with the edit count after
    /// it; this is the path on its own.
    edited: Option<String>,
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
    // Everything on screen, compacted turns included: a prompt from before a
    // compaction is still one the user can see and ask again.
    for message in &session.transcript(session.leaf()) {
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
  /// What the session runs on, model included — the one copy of it, so that
  /// `/model` rebuilding the agents and the footer drawing what they are
  /// cannot come apart.
  cfg: agent::Config,
  /// `--context-window` as it was given, which outranks anything a provider
  /// says about a model. `None` leaves the window to the provider, and to
  /// the default when it reports none.
  context_window: Option<u64>,
  /// What the provider last said it offers, for `/model` to list. Empty
  /// until the answer arrives, or when it never does.
  models: Vec<ModelInfo>,
  /// Whether a fetch of that list is on its way. Every one of them was asked
  /// for by somebody wanting to pick from it, so its arrival is what opens
  /// the picker.
  listing: bool,
  scrollbar: ScrollbarMode,
  /// When the transcript was last scrolled, for the `auto` scrollbar.
  last_scroll: Option<Instant>,
  /// A short note in the top right corner, and when it goes away: what a
  /// toggle or a copy did, said where it cannot be missed and gone again
  /// without leaving anything in the transcript.
  toast: Option<(String, Instant)>,
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
  question: Option<Question>,
  matcher: Matcher,
  completion: Option<Completion>,
  /// Esc closed the popup; stay closed until the input changes.
  completion_dismissed: bool,
  /// The last bare `Esc`, for spotting the second of a double press.
  last_escape: Option<Instant>,
  entries: Vec<Entry>,
  input: TextArea<'static>,
  /// The `@path` tokens standing in the input box, resolved as they are
  /// typed so the box can say what it found before Enter is pressed.
  attachments: Vec<Token>,
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
  /// Esc has been pressed and the run is on its way to stopping.
  aborting: bool,
  /// Tool calls the model is still writing, oldest first.
  writing: Vec<Writing>,
  /// How this run's tool calls went, by the id the transcript names them
  /// with. Handed to the session so a reload draws them the same; the
  /// transcript itself records neither the verdict nor the diff.
  outcomes: HashMap<String, Outcome>,
  /// Transcript position. `None` follows new output at the bottom; `Some`
  /// is a fixed offset from the top, so appended text does not move the view.
  anchor: Option<usize>,
  /// Offset and maximum offset used by the last draw, for relative scrolling.
  view: (usize, usize),
  /// Where the last draw put the scrollbar thumb, if it drew one at all.
  thumb: Option<Thumb>,
  /// A thumb drag in flight, holding how far down the thumb it was grabbed so
  /// the cursor keeps hold of the same row of it.
  dragging: Option<u16>,
  /// The line each entry starts at in `rendered`, one per entry.
  starts: Vec<usize>,
  /// The transcript as the last draw wrapped it, one line per row on screen.
  /// What is on screen is a window onto this, and what the mouse points at is
  /// a cell of it.
  rendered: Vec<Line<'static>>,
  /// Where the last draw put that window, for the mouse to be mapped back
  /// into the text under it.
  content: Rect,
  /// What the mouse is picking out, while it is: a selection lives from the
  /// press to the release that copies it, and no longer — so nothing has to
  /// notice that the lines under it have moved.
  selection: Option<Selection>,
  /// How much of a reasoning block to show, cycled by Ctrl+T.
  thinking_fold: Fold,
  /// How much of a tool's output to show, cycled by Ctrl+O.
  tools_fold: Fold,
  /// How much of a context summary to show, cycled by Ctrl+S.
  summary_fold: Fold,
  /// Rendered markdown, keyed by message text and width rather than by entry,
  /// so reloading a session or compacting cannot serve another entry's lines.
  /// Rebuilt each draw by moving live entries across, which evicts the rest.
  markdown: RenderCache,
  usage: Usage,
  /// Size of the last completion request, for the footer. `None` until a
  /// call has come back, and again after a compaction: what that call was
  /// weighed at is about a conversation that no longer exists, and what the
  /// footer says instead is an estimate of the one that does.
  context_tokens: Option<u64>,
  tick: usize,
  quit: bool,
}

impl App {
  pub fn new(agents: Agents, cfg: agent::Config, tx: mpsc::UnboundedSender<AgentEvent>, options: Options) -> Self {
    let Options {
      cwd,
      context_window,
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
    // Kept inside eighty columns, which is the narrowest terminal worth
    // drawing for: the hint is no use to anyone if its end is cut off.
    input.set_placeholder_text("Enter sends, Alt+Enter a newline, / commands, @ images, Ctrl+C quits.");
    // The terminal's own grey rather than a dimmed foreground, which some
    // terminals ignore and others render as the text colour proper.
    input.set_placeholder_style(Style::default().fg(Color::DarkGray));
    input.set_wrap_mode(WrapMode::WordOrGlyph);
    let session = Session::new(store.as_ref(), &cwd, &model_label(&cfg));
    let mut app = Self {
      agents,
      cfg,
      context_window,
      models: Vec::new(),
      listing: false,
      scrollbar,
      last_scroll: None,
      toast: None,
      mcp,
      bell,
      cwd,
      store,
      session,
      overlay: None,
      question: None,
      matcher: Matcher::new(Config::DEFAULT),
      completion: None,
      completion_dismissed: false,
      last_escape: None,
      entries: Vec::new(),
      input,
      attachments: Vec::new(),
      prompts: Prompts::default(),
      tx,
      run: None,
      compacting: false,
      resuming: false,
      aborting: false,
      writing: Vec::new(),
      outcomes: HashMap::new(),
      anchor: None,
      view: (0, 0),
      thumb: None,
      dragging: None,
      starts: Vec::new(),
      rendered: Vec::new(),
      content: Rect::ZERO,
      selection: None,
      thinking_fold: Fold::Preview,
      tools_fold: Fold::Preview,
      summary_fold: Fold::Preview,
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
    // The tick branch is disabled while nothing is animating, so the interval
    // goes unpolled for as long as the session sits idle. Without this the
    // backlog of missed ticks is replayed the instant a turn starts and the
    // spinner races through it before settling into its real cadence.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

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
          _ = ticker.tick(), if self.run.is_some() || self.scrollbar_fading() || self.toast.is_some() => {
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
      Event::Paste(text) => {
        self.handle_paste(&text);
        return;
      }
      Event::Mouse(mouse) => {
        match mouse.kind {
          MouseEventKind::ScrollUp => self.scroll_by(WHEEL_LINES as isize),
          MouseEventKind::ScrollDown => self.scroll_by(-(WHEEL_LINES as isize)),
          MouseEventKind::Down(MouseButton::Left) => self.press(mouse.column, mouse.row),
          MouseEventKind::Drag(MouseButton::Left) => self.drag(mouse.column, mouse.row),
          MouseEventKind::Up(MouseButton::Left) => self.release(),
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
    if self.question.is_some() {
      self.handle_question_key(key, ctrl);
      return;
    }
    if self.overlay.is_some() {
      self.handle_overlay_key(key, ctrl);
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
        self.thinking_fold = self.thinking_fold.next();
        self.notify(format!("Thinking {}", self.thinking_fold.label()));
        // Expanding moves everything below the block, so go back to following
        // the bottom rather than leaving the reader mid-paragraph. Only the
        // view changes: the conversation is the same one, and so is anything
        // waiting to be said to it.
        self.anchor = None;
      }
      (KeyCode::Char('o'), true) => {
        self.tools_fold = self.tools_fold.next();
        self.notify(format!("Tool output {}", self.tools_fold.label()));
        self.anchor = None;
      }
      (KeyCode::Char('s'), true) => {
        self.summary_fold = self.summary_fold.next();
        self.notify(format!("Context summary {}", self.summary_fold.label()));
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
        self.input_changed();
      }
      (KeyCode::Char('j'), true) => {
        self.prompts.stop();
        self.input.insert_newline();
        self.input_changed();
      }
      (KeyCode::Enter, _) => self.submit(),
      _ => {
        if self.input.input(key) {
          self.completion_dismissed = false;
          // A recalled prompt that has been edited is the user's text now,
          // and Down is no longer a way back out of it.
          self.prompts.stop();
        }
        self.input_changed();
      }
    }
  }

  /// The input text has changed: resolve its `@tokens` again, and then the
  /// popup, which is chosen by the token the cursor is in.
  fn input_changed(&mut self) {
    self.refresh_attachments();
    self.refresh_completion();
  }

  /// A bracketed paste: text the user never typed, so its newlines are text
  /// too. It goes in whole, and nothing in it is read as a key — the Enter
  /// halfway through a pasted snippet is not a request to send it.
  fn handle_paste(&mut self, text: &str) {
    // Terminals are not of one mind about how a clipboard's line endings
    // reach us; the box keeps `\n`.
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    if text.is_empty() {
      return;
    }
    if let Some(question) = &mut self.question {
      question.dialog.paste(&text);
      return;
    }
    // The other overlays are lists, and text pasted at one narrows it: the
    // box at the bottom of the screen is holding the query, and a path or a
    // title copied from somewhere else is a reasonable thing to look for.
    // Its newlines are not, so it arrives as the one line the query is.
    if let Some(overlay) = &mut self.overlay {
      overlay.filter.field.insert_str(text.replace('\n', " "));
      overlay.refilter(&mut self.matcher);
      overlay.selected = 0;
      return;
    }
    self.prompts.stop();
    // Dropping an image on the terminal pastes its path, which is the
    // gesture people reach for first. Written down as a token, it attaches
    // rather than sitting there as a path nothing reads.
    let text = as_token(&text, &self.cwd).unwrap_or(text);
    self.input.insert_str(&text);
    self.completion_dismissed = false;
    self.input_changed();
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
        let Some(item) = c.items.get(c.selected) else {
          return false;
        };
        match item {
          Match::Command { index, .. } => {
            let (name, _, takes_arg) = COMMANDS[*index];
            self.set_input(&format!("/{name}{}", if takes_arg { " " } else { "" }));
            // The completed command is exact; keep the popup closed until
            // the user edits the text again.
            self.completion_dismissed = true;
          }
          Match::Path { insert, dir, .. } => {
            let Some((from, to)) = c.replacing else {
              return false;
            };
            // A directory is a step on the way rather than an answer, so the
            // popup stays up and lists what is inside it.
            self.completion_dismissed = !dir;
            self.replace_token(from, to, insert.clone());
          }
        }
      }
      _ => return false,
    }
    true
  }

  /// Put `insert` in place of the bytes `from..to` of the input, and leave
  /// the cursor just after it.
  fn replace_token(&mut self, from: usize, to: usize, insert: String) {
    let text = self.input.lines().join("\n");
    let (head, tail) = (&text[..from], &text[to.min(text.len())..]);
    let cursor = head.len() + insert.len();
    let replaced = format!("{head}{insert}{tail}");
    self.set_input(&replaced);
    let before = &replaced[..cursor];
    let row = before.matches('\n').count();
    let col = before.rsplit('\n').next().unwrap_or("").chars().count();
    self.input.move_cursor(CursorMove::Jump(row as u16, col as u16));
    self.input_changed();
  }

  fn move_completion(&mut self, delta: isize) {
    if let Some(c) = &mut self.completion {
      let len = c.items.len().max(1);
      c.selected = (c.selected as isize + delta).rem_euclid(len as isize) as usize;
    }
  }

  /// Show the command popup while the input is a single `/word` prefix, and
  /// the file popup while an `@token` is being typed.
  fn refresh_completion(&mut self) {
    let lines = self.input.lines();
    let commands = match lines {
      [line] if line.starts_with('/') && !line.contains(char::is_whitespace) => Some(line[1..].to_string()),
      _ => None,
    };
    if self.completion_dismissed {
      if commands.is_none() && self.completing_token().is_none() {
        // Whatever was dismissed is no longer being typed, so the next one
        // starts fresh.
        self.completion_dismissed = false;
      }
      self.completion = None;
      return;
    }
    let (items, replacing) = match commands {
      Some(query) => (filter_commands(&mut self.matcher, &query), None),
      None => match self.completing_token() {
        Some(range) => {
          let text = self.input.lines().join("\n");
          let items = self.filter_paths(&text[range.0..range.1]);
          (items, Some(range))
        }
        None => (Vec::new(), None),
      },
    };
    if items.is_empty() {
      self.completion = None;
      return;
    }
    let selected = self
      .completion
      .as_ref()
      .and_then(|c| c.items.get(c.selected).map(key))
      .and_then(|prev| items.iter().position(|m| key(m) == prev))
      .unwrap_or(0);
    self.completion = Some(Completion {
      items,
      selected,
      replacing,
    });
  }

  /// The `@token` the cursor is at the end of, which is the one being typed.
  /// Only that one: an `@` earlier in the sentence is settled.
  ///
  /// A token that already names an image is settled too, popup closed — so
  /// `Enter` on a finished token sends the prompt rather than being taken by
  /// a list offering the file that is already written there.
  fn completing_token(&self) -> Option<(usize, usize)> {
    let at = self.cursor_offset();
    self
      .attachments
      .iter()
      .find(|token| token.range.1 == at && !token.is_image())
      .map(|token| token.range)
  }

  /// Where the cursor is, as a byte offset into the input.
  fn cursor_offset(&self) -> usize {
    let ratatui_textarea::DataCursor(row, col) = self.input.cursor();
    let lines = self.input.lines();
    let before: usize = lines.iter().take(row).map(|line| line.len() + 1).sum();
    let line = lines.get(row).map_or("", |l| l.as_str());
    before + line.char_indices().nth(col).map_or(line.len(), |(i, _)| i)
  }

  /// The paths an `@token` could be reaching for: what is in the directory
  /// it names, narrowed to images and the directories on the way to them.
  fn filter_paths(&mut self, token: &str) -> Vec<Match> {
    // The token as a path, with the sigil and any quoting taken off.
    let typed = token.trim_start_matches('@').trim_matches('"');
    // Everything up to the last separator names the directory to look in;
    // what is left is what the name has to match.
    let (dir, prefix) = match typed.rfind('/') {
      Some(cut) => (&typed[..=cut], &typed[cut + 1..]),
      None => ("", typed),
    };
    let base = match dir.is_empty() {
      true => self.cwd.clone(),
      false => crate::tools::resolve(&self.cwd, dir),
    };
    let Ok(entries) = std::fs::read_dir(&base) else {
      return Vec::new();
    };
    let mut candidates: Vec<(u32, String, bool)> = Vec::new();
    let pattern = Pattern::parse(prefix, CaseMatching::Ignore, Normalization::Smart);
    let mut buf = Vec::new();
    for entry in entries.flatten() {
      let name = entry.file_name().to_string_lossy().into_owned();
      // Hidden files only when they are being asked for by name.
      if name.starts_with('.') && !prefix.starts_with('.') {
        continue;
      }
      let is_dir = entry.file_type().is_ok_and(|t| t.is_dir());
      if !is_dir && !attach::looks_like_image(&entry.path()) {
        continue;
      }
      let Some(score) = pattern.score(Utf32Str::new(&name, &mut buf), &mut self.matcher) else {
        continue;
      };
      candidates.push((score, name, is_dir));
    }
    // Best match first, directories after the images they sit beside at the
    // same score, and then alphabetically so the list does not jitter.
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)));
    candidates.truncate(PATH_ROWS);
    candidates
      .into_iter()
      .map(|(_, name, is_dir)| {
        let mut buf = Vec::new();
        let mut highlights = Vec::new();
        pattern.indices(Utf32Str::new(&name, &mut buf), &mut self.matcher, &mut highlights);
        highlights.sort_unstable();
        highlights.dedup();
        let path = format!("{dir}{name}{}", if is_dir { "/" } else { "" });
        // A path with a space in it has to come back quoted, or the token
        // would end at the space.
        let insert = match path.contains(' ') {
          true => format!("@\"{path}\""),
          false => format!("@{path}"),
        };
        let meta = match is_dir {
          true => String::new(),
          false => match attach::dimensions(&base.join(&name)) {
            Some((w, h)) => format!("{w}×{h}"),
            None => String::new(),
          },
        };
        Match::Path {
          insert,
          name,
          highlights,
          meta,
          dir: is_dir,
        }
      })
      .collect()
  }

  /// Resolve the `@path` tokens in the input box, so the border can say what
  /// they found. Called wherever the text changes.
  fn refresh_attachments(&mut self) {
    let text = self.input.lines().join("\n");
    self.attachments = match text.contains('@') {
      true => attach::tokens(&text, &self.cwd),
      // The overwhelmingly common case, and worth not touching the disk for.
      false => Vec::new(),
    };
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
    let Some(question) = &mut self.question else {
      return;
    };
    let Some(outcome) = question.dialog.key(key) else {
      return;
    };
    if let Some(question) = self.question.take() {
      let _ = question.reply.send(outcome);
    }
  }

  fn handle_overlay_key(&mut self, key: KeyEvent, ctrl: bool) {
    let len = self.overlay.as_ref().map_or(0, Overlay::len);
    if self.handle_delete_key(key.code, ctrl) {
      return;
    }
    match key.code {
      KeyCode::Esc => self.overlay = None,
      KeyCode::Char('c') if ctrl => self.quit = true,
      KeyCode::Up => {
        if let Some(o) = &mut self.overlay {
          o.selected = o.selected.saturating_sub(1);
        }
      }
      KeyCode::Down => {
        if let Some(o) = &mut self.overlay {
          o.selected = (o.selected + 1).min(len.saturating_sub(1));
        }
      }
      // A tree of every tool result is long enough to need more than one row
      // at a time. Home and End, which used to take the list to its ends, are
      // the query's now — they are what moves a cursor through text, and a
      // list can be filtered down to its far end instead.
      KeyCode::PageUp => {
        if let Some(o) = &mut self.overlay {
          o.selected = o.selected.saturating_sub(OVERLAY_PAGE);
        }
      }
      KeyCode::PageDown => {
        if let Some(o) = &mut self.overlay {
          o.selected = (o.selected + OVERLAY_PAGE).min(len.saturating_sub(1));
        }
      }
      KeyCode::Enter => {
        let Some(overlay) = self.overlay.take() else {
          return;
        };
        // The row taken is the one the filter left under the cursor, which is
        // an index into the whole list rather than a place in it.
        let Some(at) = overlay.at() else {
          return;
        };
        match overlay.list {
          OverlayList::Sessions(sessions) => {
            if let Some(info) = sessions.get(at) {
              self.load_session(&info.path.clone());
            }
          }
          OverlayList::Tree(mut points) if at < points.len() => self.go_to(points.remove(at)),
          OverlayList::Fork(mut points) if at < points.len() => self.fork_to(points.remove(at)),
          OverlayList::Models(models) => {
            if let Some(model) = models.get(at) {
              self.set_model(model.id.clone());
            }
          }
          OverlayList::Goto(marks) => {
            if let Some(mark) = marks.get(at) {
              self.goto(mark.entry);
            }
          }
          _ => {}
        }
      }
      // Everything else belongs to the box at the bottom of the screen, which
      // is holding the query.
      _ => self.handle_filter_key(key),
    }
  }

  /// A key the list did not claim, which is the query's.
  ///
  /// The box keeps it: what a key does to text is the box's business, and it
  /// is the same box and the same widget a prompt is typed in. Only a key
  /// that changed what is written there narrows the list again — the arrows
  /// within the query move a cursor and leave the rows alone.
  fn handle_filter_key(&mut self, key: KeyEvent) {
    // Taken as two borrows of two fields rather than through a method, so
    // the matcher is free while the overlay is held.
    let Some(overlay) = &mut self.overlay else {
      return;
    };
    let filter = &mut overlay.filter;
    let before = filter.query();
    filter.field.input(key);
    if filter.query() == before {
      return;
    }
    overlay.refilter(&mut self.matcher);
    // The best match is the one wanted, and the rows under it have been
    // reordered anyway — a cursor left where it stood would mean nothing.
    overlay.selected = 0;
  }

  /// The keys that delete a session from the picker, answering whether this
  /// was one of them.
  ///
  /// `Ctrl+D` asks, and a second `Ctrl+D` goes through with it. Anything
  /// else is not an answer to the question: it puts the question away and
  /// then means what it usually means — apart from `Esc`, which was the
  /// answer no and leaves the picker open.
  ///
  /// `Delete` itself is the query's: it is a key that edits text, and the
  /// query is text. Ctrl+D is what is left, and what a shell deletes with.
  fn handle_delete_key(&mut self, code: KeyCode, ctrl: bool) -> bool {
    let Some(overlay) = &mut self.overlay else {
      return false;
    };
    if !matches!(overlay.list, OverlayList::Sessions(_) | OverlayList::Tree(_)) {
      return false;
    }
    let asked = ctrl && code == KeyCode::Char('d');
    match (asked, overlay.confirming) {
      (true, false) => overlay.confirming = overlay.at().is_some(),
      (true, true) => {
        overlay.confirming = false;
        match overlay.list {
          OverlayList::Tree(_) => self.delete_point(),
          _ => self.delete_session(),
        }
      }
      (false, true) => {
        overlay.confirming = false;
        return code == KeyCode::Esc;
      }
      _ => return false,
    }
    true
  }

  /// Delete the session the picker is on, and take its row out of the list.
  fn delete_session(&mut self) {
    let Some(overlay) = &mut self.overlay else {
      return;
    };
    let at = overlay.at();
    let OverlayList::Sessions(sessions) = &mut overlay.list else {
      return;
    };
    let Some(at) = at else {
      return;
    };
    let Some(info) = sessions.get(at) else {
      return;
    };
    let (path, title) = (info.path.clone(), info.title().to_string());
    // The conversation on screen goes on writing to its file, so deleting it
    // from under itself would only leave a shorter one behind.
    if self.session.path() == Some(path.as_path()) {
      self.notify("That is this session — start another with /new before deleting it.");
      return;
    }
    let Some(store) = &self.store else { return };
    if let Err(err) = store.delete(&path) {
      self
        .entries
        .push(Entry::Error(format!("Could not delete session: {err:#}")));
      return;
    }
    sessions.remove(at);
    let emptied = sessions.is_empty();
    // Every index past the deleted one has moved, so the rows are worked out
    // again rather than patched up — the query is the same, so what is left
    // is the same list one row shorter.
    overlay.refilter(&mut self.matcher);
    overlay.selected = overlay.selected.min(overlay.len().saturating_sub(1));
    self.notify(format!("Deleted session {title}."));
    // Nothing left to pick from is nothing to keep a picker open for.
    if emptied {
      self.overlay = None;
    }
  }

  /// Delete the branch the tree is on — the entry under the cursor and
  /// everything said after it — and draw the tree again without it.
  fn delete_point(&mut self) {
    let Some(overlay) = &mut self.overlay else {
      return;
    };
    let at = overlay.at();
    let OverlayList::Tree(rows) = &overlay.list else {
      return;
    };
    let Some(point) = at.and_then(|at| rows.get(at)) else {
      return;
    };
    let (id, label) = (point.id.clone(), point.label.clone());
    // The next turn is written under the end of the conversation on screen,
    // so that is not something to take out from under it.
    if self.session.lineage(self.session.leaf(), true).contains(&id.as_str()) {
      self.notify("That is on the conversation you are in — go somewhere else before deleting it.");
      return;
    }
    let gone = match self.session.delete_branch(&id) {
      Ok(gone) => gone,
      Err(err) => {
        self
          .entries
          .push(Entry::Error(format!("Could not delete branch: {err:#}")));
        return;
      }
    };
    // The rows past it have moved and the indents where it branched off
    // have changed, so the tree is worked out again rather than patched up.
    let rows = points(&self.session);
    let emptied = rows.is_empty();
    overlay.list = OverlayList::Tree(rows);
    overlay.refilter(&mut self.matcher);
    overlay.selected = overlay.selected.min(overlay.len().saturating_sub(1));
    self.notify(format!("Deleted {label}, {} in all.", messages(gone)));
    if emptied {
      self.overlay = None;
    }
  }

  /// Scroll the transcript up (positive) or down (negative) by `lines`.
  fn scroll_by(&mut self, lines: isize) {
    self.scroll_to(self.view.0.saturating_add_signed(-lines));
  }

  /// Scroll the transcript to `target` lines from its top.
  fn scroll_to(&mut self, target: usize) {
    // Reaching the bottom re-attaches to the live end of the transcript.
    self.anchor = (target < self.view.1).then_some(target);
    // Where the view is now, which the next draw will say for itself — but
    // several scrolls can arrive between two draws, and each of them should
    // move on from where the last one left off rather than from where the
    // screen still is.
    self.view.0 = target.min(self.view.1);
    self.last_scroll = Some(Instant::now());
  }

  /// A left press. On the scrollbar it takes hold of the thumb; in the
  /// transcript it starts a selection.
  fn press(&mut self, column: u16, row: u16) {
    self.grab_thumb(column, row);
    // An overlay is a window over the transcript, not a view of it: what is
    // under it is not what is on screen at that cell.
    if self.dragging.is_some() || self.overlay.is_some() || self.question.is_some() {
      return;
    }
    if !self.content.contains(Position::new(column, row)) {
      return;
    }
    let Some(cell) = self.cell_at(column, row) else {
      return;
    };
    self.selection = Some(Selection {
      anchor: cell,
      head: cell,
    });
  }

  /// The cursor moving with the button down: the thumb follows it, or the
  /// selection grows to it.
  fn drag(&mut self, column: u16, row: u16) {
    if self.dragging.is_some() {
      self.drag_thumb(row);
      return;
    }
    if self.selection.is_none() {
      return;
    }
    // Dragging along either edge scrolls the transcript under the cursor, so
    // a selection can run further than the screen shows. The edge row itself
    // and not past it: the transcript starts at the top of the screen, where
    // there is no row above to reach for.
    if row <= self.content.y {
      self.scroll_by(1);
    } else if row >= self.content.bottom().saturating_sub(1) {
      self.scroll_by(-1);
    }
    if let Some(cell) = self.cell_at(column, row)
      && let Some(selection) = &mut self.selection
    {
      selection.head = cell;
    }
  }

  /// The button coming up, which is the end of the selection as well as of
  /// the drag: it is copied and then let go of, rather than left highlighted
  /// for something else to have to take it back off the screen. A press that
  /// went nowhere was a click, and copies nothing.
  fn release(&mut self) {
    self.dragging = None;
    let Some(selection) = self.selection.take().filter(|selection| !selection.is_empty()) else {
      return;
    };
    let text = self.selected_text(selection);
    // Blank cells are what the transcript pads with rather than anything the
    // user meant to take away with them.
    if text.trim().is_empty() {
      return;
    }
    match crate::clipboard::copy(&text) {
      Ok(()) => self.notify("Copied"),
      Err(note) => self.entries.push(Entry::Info(note)),
    }
  }

  /// Put `text` in the corner, for longer the more of it there is to read.
  fn notify(&mut self, text: impl Into<String>) {
    let text = text.into();
    let reading = TOAST_PER_CHAR * text.chars().count() as u32;
    self.toast = Some((text, Instant::now() + (TOAST_DELAY + reading).min(TOAST_MAX_DELAY)));
  }

  /// The cell of the transcript under the cursor, clamped into the text — so
  /// a drag that wanders off the side, or below the last line, still points
  /// at the end of what it passed over. `None` when there is nothing to point
  /// at.
  fn cell_at(&self, column: u16, row: u16) -> Option<(usize, usize)> {
    if self.content.height == 0 || self.rendered.is_empty() {
      return None;
    }
    let row = row.clamp(self.content.y, self.content.bottom() - 1) - self.content.y;
    let line = (self.view.0 + row as usize).min(self.rendered.len() - 1);
    let column = column.clamp(self.content.x, self.content.right()) - self.content.x;
    Some((line, column as usize))
  }

  /// What a selection covers, as the text it is drawn from: the lines it
  /// touches, each cut to the columns of it that are in the selection.
  ///
  /// Line by line as they are on screen, so a wrapped paragraph comes back
  /// wrapped — what was copied is what was pointed at, and a code block
  /// stays the lines it was written on.
  fn selected_text(&self, selection: Selection) -> String {
    let ((first, _), (last, _)) = selection.ends();
    (first..=last.min(self.rendered.len().saturating_sub(1)))
      .filter_map(|at| {
        let (from, to) = selection.columns(at)?;
        Some(selected(&self.rendered[at], from, to))
      })
      .collect::<Vec<_>>()
      .join("\n")
  }

  /// A left press on the scrollbar. On the thumb it takes hold of it; on the
  /// bare track it pulls the thumb to the cursor first, so either way what
  /// follows is a drag.
  fn grab_thumb(&mut self, column: u16, row: u16) {
    let Some(thumb) = self.thumb else { return };
    if column != thumb.track.x || !(thumb.track.y..thumb.track.bottom()).contains(&row) {
      return;
    }
    let top = thumb.track.y + thumb.start;
    if (top..top + thumb.len).contains(&row) {
      self.dragging = Some(row - top);
    } else {
      self.dragging = Some(thumb.len / 2);
      self.drag_thumb(row);
    }
  }

  /// Follow the cursor with the thumb it is holding. The offset comes from
  /// where the cursor is rather than how far it moved, so a drag that runs off
  /// the end of the track and back finds the transcript where it left it.
  fn drag_thumb(&mut self, row: u16) {
    let (Some(thumb), Some(grab)) = (self.thumb, self.dragging) else {
      return;
    };
    let top = row as isize - thumb.track.y as isize - grab as isize;
    self.scroll_to(thumb_offset(top, thumb, self.view.1));
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
    // Only the tokens: a prompt handed back by `/tree` or walked back to
    // should say what it attaches, but it should not open a popup over a
    // box nobody has typed in yet.
    self.refresh_attachments();
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

    // Read here rather than wherever the prompt is finally sent: the file
    // is what it was when Enter was pressed, not what it becomes while a
    // run works through the queue ahead of it.
    let prompt = self.attach(&text);

    // What a run makes of what is typed at it is a message, which is what
    // was meant by all but these: they were typed at fa, and are answered
    // here whether or not a run has the floor.
    let for_us = matches!(text.as_str(), "/quit" | "/new" | "/model" | "/goto") || text.starts_with("/model ");
    if self.run.is_some() && !for_us {
      // Handed to the run, which reads it at the top of its next turn
      // rather than after the whole answer. It is kept there and nowhere
      // else, so whoever gets to it first is the only one who can.
      self.agents.control.steer(prompt);
      return;
    }
    self.dispatch(prompt);
  }

  /// Load the images the prompt's `@tokens` name, saying in the transcript
  /// what could not be sent and why.
  fn attach(&mut self, text: &str) -> Prompt {
    let (prompt, notes) = attach_images(text, &self.cwd, self.cfg.vision);
    self.entries.extend(notes.into_iter().map(Entry::Info));
    prompt
  }

  /// Run a prompt or slash command now.
  fn dispatch(&mut self, prompt: Prompt) {
    let text = prompt.text.clone();
    match text.as_str() {
      "/quit" => self.quit = true,
      "/new" => self.new_session(),
      "/compact" => self.compact(),
      "/continue" => self.continue_run(),
      "/resume" => {
        if self.run.is_some() {
          self.notify("Finish or abort the current run before resuming another session.");
        } else {
          self.open_picker();
        }
      }
      "/tree" | "/fork" => {
        if self.run.is_some() {
          self.notify(format!("Finish or abort the current run before {text}."));
        } else {
          self.open_points(text == "/fork");
        }
      }
      "/session" => self.session_info(),
      "/goto" => self.open_marks(),
      t if t == "/model" || t.starts_with("/model ") => {
        let named = t["/model".len()..].trim().to_string();
        match named.is_empty() {
          true => self.open_models(),
          // A model named here is used whether or not the provider listed
          // it: the list is what the provider admits to, not the whole of
          // what it will answer to.
          false => self.set_model(named),
        }
      }
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
      _ => self.start(prompt),
    }
  }

  // -------------------------------------------------------------- models

  /// Ask the provider what it offers, in the background: the answer comes
  /// back as an `AgentEvent`, like everything else the UI waits on, and
  /// opens the picker when it does.
  fn list_models(&mut self) {
    // One is already on its way, and it is the same list: `/model` pressed
    // while a session that came up without a model is still waiting for it
    // is answered by the request already out.
    if self.listing {
      return;
    }
    self.listing = true;
    let cfg = self.cfg.clone();
    let tx = self.tx.clone();
    tokio::spawn(async move {
      // An endpoint that takes the request and never answers it would
      // otherwise leave this session waiting on a list forever, with
      // `/model` waiting behind it for the same one.
      let models = match tokio::time::timeout(LISTING_TIMEOUT, agent::list_models(&cfg)).await {
        Ok(models) => models.map_err(|err| format!("{err:#}")),
        Err(_) => Err(format!(
          "{} did not answer for its models within {} seconds",
          cfg.provider.label(),
          LISTING_TIMEOUT.as_secs()
        )),
      };
      let _ = tx.send(AgentEvent::Models(models));
    });
  }

  /// What the fetch came back with.
  fn take_models(&mut self, models: Result<Vec<ModelInfo>, String>) {
    self.listing = false;
    let models = match models {
      Ok(models) => models,
      Err(err) => {
        self.entries.push(Entry::Error(err));
        return;
      }
    };
    self.models = models;
    // The provider has just said what the model in use holds, which is where
    // its context window comes from — so a session that opened `/model` and
    // thought better of it still leaves knowing how big its own model is.
    // Not while a run is going: it reads the window it was built with, and
    // is not to be rebuilt underneath it.
    if self.run.is_none() {
      self.set_model(self.cfg.model.clone());
    }
    self.show_models();
  }

  /// `/model` with nothing after it: ask the provider what it has, and pick
  /// from the answer.
  ///
  /// Asked every time rather than kept: a list fetched when it is wanted is
  /// one that cannot be out of date, and a provider that has gained a model
  /// since fa started has it here without being asked twice.
  fn open_models(&mut self) {
    if self.run.is_some() {
      self.notify("Finish or abort the current run before changing model.");
      return;
    }
    self
      .entries
      .push(Entry::Info("Asking the provider for its models…".into()));
    self.list_models();
  }

  /// Open the picker on the list as it stands, at the model in use.
  fn show_models(&mut self) {
    if self.models.is_empty() {
      self.notify("The provider offered no models — name one with /model <id>.");
      return;
    }
    let selected = self
      .models
      .iter()
      .position(|model| model.id == self.cfg.model)
      .unwrap_or(0);
    self.overlay = Some(Overlay::new(OverlayList::Models(self.models.clone()), selected));
  }

  /// Point the session at `id` and rebuild the agents around it.
  ///
  /// The window it is held to is the one `--context-window` named, or the one
  /// the provider reports for this model, or the fallback — in that order, so
  /// a figure the user gave stands whatever is chosen afterwards.
  ///
  /// A model named rather than picked is used whether or not the provider
  /// listed it: the list is what the provider admits to, not the whole of
  /// what it answers to.
  fn set_model(&mut self, id: String) {
    let reported = self
      .models
      .iter()
      .find(|model| model.id == id)
      .and_then(|model| model.context_length);
    let window = self.context_window.or(reported).unwrap_or(DEFAULT_CONTEXT_WINDOW);
    // Nothing to do, and nothing to say: this is the model already in use,
    // held to the window it is already held to.
    if self.cfg.model == id && self.cfg.compaction.context_window == window {
      return;
    }
    if self.run.is_some() {
      self.notify("Finish or abort the current run before changing model.");
      return;
    }
    let mut cfg = self.cfg.clone();
    cfg.model = id.clone();
    cfg.compaction.context_window = window;
    // The runtime carries a copy of both, so the model and the window it is
    // compacted at only take effect once it has been built again.
    if let Err(err) = self.agents.use_model(&cfg) {
      self.entries.push(Entry::Error(format!("Could not use {id}: {err:#}")));
      return;
    }
    self.cfg = cfg;
    let result = self.session.set_model(&model_label(&self.cfg));
    self.report(result);
    self
      .entries
      .push(Entry::Info(format!("Model {id}, context window {window} tokens.")));
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
    self.agents.control.take();
    self.overflowed();
    self.resuming = false;
    self.usage = Usage::new();
    self.context_tokens = None;
    self.anchor = None;
  }

  fn new_session(&mut self) {
    self.abort();
    self.session = Session::new(self.store.as_ref(), &self.cwd, &model_label(&self.cfg));
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
        // What a compaction summarized is back on screen but not back in the
        // context, so when the two differ both are worth saying.
        let shown = session.transcript_len(session.leaf());
        let context = session.history.len();
        let counts = match shown > context {
          true => format!("{shown} messages shown, {context} in context"),
          false => messages(context),
        };
        self
          .entries
          .push(Entry::Info(format!("Resumed session {title} ({counts}).")));
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
      self.notify("Sessions are disabled (--no-session).");
      return;
    };
    let sessions = store.list();
    if sessions.is_empty() {
      self.notify("No saved sessions.");
      return;
    }
    self.overlay = Some(Overlay::new(OverlayList::Sessions(sessions), 0));
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
      self.notify(match fork {
        true => "Nothing to fork from.",
        false => "Nothing to go back to.",
      });
      return;
    }
    // Start where the session already is, so the way back is one step up.
    let selected = points.iter().rposition(|point| point.here).unwrap_or(points.len() - 1);
    let list = match fork {
      true => OverlayList::Fork(points),
      false => OverlayList::Tree(points),
    };
    self.overlay = Some(Overlay::new(list, selected));
  }

  /// List the prompts in the transcript, the latest selected.
  ///
  /// Only the view moves: where the session is stays where it was, which is
  /// why this one is open while a run is going.
  fn open_marks(&mut self) {
    let marks: Vec<Mark> = self
      .entries
      .iter()
      .enumerate()
      .filter_map(|(entry, e)| match e {
        Entry::User { text, .. } => Some(Mark {
          entry,
          label: first_line(text),
        }),
        _ => None,
      })
      .collect();
    if marks.is_empty() {
      self.notify("No prompts to go to.");
      return;
    }
    let selected = marks.len() - 1;
    self.overlay = Some(Overlay::new(OverlayList::Goto(marks), selected));
  }

  /// Scroll the transcript so the prompt at `entry` is at the top of it.
  ///
  /// The lines are the last draw's, which was at the width the next one will
  /// be: the list is drawn where the transcript was.
  fn goto(&mut self, entry: usize) {
    // A prompt's first line is the blank one above it; the one worth having
    // at the top is the one it is written on.
    if let Some(start) = self.starts.get(entry) {
      self.scroll_to(start + 1);
    }
  }

  /// Move this session's end to `point`.
  ///
  /// Nothing is dropped — what the conversation said down the path being left
  /// stays a branch of its own, and this list can walk back into it. The
  /// transcript is rebuilt from the history rather than edited alongside it,
  /// so the two cannot drift: what is on screen is what the model will be sent.
  fn go_to(&mut self, point: Point) {
    if point.here && point.text.is_none() {
      self.notify("Already there.");
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
    let Some(prompt) = self.agents.control.unsteer() else {
      return;
    };
    // The text is what goes back in the box; its attachments are read again
    // when it is sent again, from the tokens still written in it.
    self.set_input(&prompt.text);
  }

  /// Send what is still waiting once there is no run to hand it to.
  ///
  /// A run reads what was typed at it itself, so this is only ever the
  /// remainder: a message typed in the moment between the run looking and
  /// the run ending, or one typed while a compaction had the floor.
  fn next_queued(&mut self) {
    if let Some(next) = self.agents.control.take_next() {
      self.dispatch(next);
    }
  }

  /// Whether the run that just ended found the context window full — and
  /// clear the mark, since answering it is this side's half of the bargain.
  fn overflowed(&self) -> bool {
    self.agents.control.overflowed()
  }

  fn compact(&mut self) {
    if self.session.history.is_empty() {
      self.notify("Nothing to compact.");
      self.resuming = false;
      self.next_queued();
      return;
    }
    self.entries.push(Entry::Info("Compacting context…".into()));
    self.compacting = true;
    self.run = Some(start_compaction(
      self.agents.runtime.clone(),
      self.session.history.clone(),
      self.cfg.compaction,
      self.tx.clone(),
    ));
  }

  fn start(&mut self, prompt: Prompt) {
    self.entries.push(Entry::User {
      text: prompt.text.clone(),
      images: prompt.preview(),
    });
    let prompt = prompt.message();
    let handle = start_run(
      self.agents.runtime.clone(),
      self.agents.control.clone(),
      self.session.history.clone(),
      Some(prompt),
      self.tx.clone(),
    );
    self.run = Some(handle);
  }

  /// Run the model again with no new user message, to pick the loop back up
  /// where an abort or a compaction left it. The last history message becomes
  /// the prompt of the request, so the model sees exactly the conversation it
  /// already had: an unanswered user message is answered, and a half-written
  /// answer is continued.
  fn continue_run(&mut self) {
    if self.session.history.is_empty() {
      self.notify("Nothing to continue.");
      self.next_queued();
      return;
    }
    let handle = start_run(
      self.agents.runtime.clone(),
      self.agents.control.clone(),
      self.session.history.clone(),
      None,
      self.tx.clone(),
    );
    self.run = Some(handle);
  }

  /// Esc.
  ///
  /// A run is asked to stop rather than killed: it drops what it is in the
  /// middle of, answers the calls it had out so nothing is left hanging,
  /// and hands back everything it got through. So the work is kept exactly
  /// as the model gave it, and `Ended` finishes the job here.
  ///
  /// A compaction has no loop of its own to ask, so it is still killed.
  fn abort(&mut self) {
    // A question the run was waiting on has nobody left to answer to;
    // closing it drops the channel, which is how the tool hears that.
    self.close_question();
    if self.run.is_none() {
      return;
    }
    // Esc is the end of it: a run stopped to be compacted is not resumed
    // afterwards, and the next one starts with the window weighed afresh.
    self.overflowed();
    self.resuming = false;
    if self.compacting {
      if let Some(handle) = self.run.take() {
        handle.abort();
      }
      self.compacting = false;
      self.writing.clear();
      self.outcomes.clear();
      self.entries.push(Entry::Info("Compaction aborted.".into()));
      self.strand_queued();
      return;
    }
    self.aborting = true;
    self.agents.control.cancel();
  }

  /// Esc stops everything, including what was waiting behind the run — but
  /// it was typed, so it is kept in the transcript rather than dropped out
  /// of sight.
  fn strand_queued(&mut self) {
    for text in self.agents.control.take() {
      self
        .entries
        .push(Entry::Info(format!("Not sent: {}", first_line(&text.text))));
    }
  }

  /// Take down the model's questionnaire, if one is up. What it had been
  /// asked is left unanswered, which the tool reads as a decline.
  fn close_question(&mut self) {
    self.question = None;
  }

  /// Record what a run added to the conversation.
  ///
  /// However it ended — with its answer, stopped for room, or cancelled —
  /// these are the messages the run itself held and sent, not a reckoning
  /// of them made from what went past on screen. So the next request is the
  /// last one with more on the end, which is all a prompt cache asks for.
  fn stopped(&mut self, messages: Vec<Message>) {
    if messages.is_empty() {
      return;
    }
    // How those tool calls went goes with them, or the transcript they
    // leave behind would forget which failed and what each changed.
    let outcomes = std::mem::take(&mut self.outcomes);
    let result = self.session.append_with(messages, &outcomes);
    self.report(result);
  }

  // ------------------------------------------------------------ agent events

  fn handle_agent(&mut self, ev: AgentEvent) {
    // Nothing to do with a run: the list was asked for by the UI, and comes
    // back whether or not the model is busy.
    if let AgentEvent::Models(models) = ev {
      self.take_models(models);
      return;
    }
    if self.run.is_none() {
      return; // stale event from an aborted run
    }
    match ev {
      AgentEvent::Text(delta) => match self.entries.last_mut() {
        Some(Entry::Assistant(text)) => text.push_str(&delta),
        _ => self.entries.push(Entry::Assistant(delta)),
      },
      // The run has read one of the messages waiting behind it, so it stops
      // being something on its way and becomes a prompt like any other —
      // drawn where the run reached it, which is where it was sent from.
      AgentEvent::Steered { text, images } => self.entries.push(Entry::User { text, images }),
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
        let summary = summarize_args(&name, &args);
        self.entries.push(Entry::ToolCall {
          wrote: wrote_content(&name, &args),
          edited: edited_path(&name, &args),
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
        self.question = Some(Question {
          dialog: Dialog::new(questions),
          reply,
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
        self.stopped(messages);
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
      AgentEvent::Ended { messages } => {
        self.run = None;
        self.writing.clear();
        self.close_question();
        let full = self.overflowed();
        if self.compacting {
          // A compaction that did not finish is not worth starting again on
          // the next turn: the room it was going to make is not coming.
          self.compacting = false;
          self.resuming = false;
          self.outcomes.clear();
          self.next_queued();
          return;
        }
        self.stopped(messages);
        // Esc asked for this. The run has stopped where it stood and said
        // what it got through, so the rest of what Esc means happens here.
        if std::mem::take(&mut self.aborting) {
          finish_running(&mut self.entries);
          self.entries.push(Entry::Info("Aborted.".into()));
          self.strand_queued();
          return;
        }
        if full {
          // The run stopped at a turn boundary to let this happen, and
          // goes on once there is room again.
          self.resuming = true;
          self.compact();
          return;
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
              "Compacted {} into a summary; kept the last {}.",
              messages(compacted.summarized),
              messages(compacted.kept)
            )));
            self.entries.push(Entry::Summary(compacted.summary));
            // What was cut short to make this room carries on where it
            // stopped — unless the user has said something since, which is
            // what it would have read next anyway.
            if resuming && !self.agents.control.steering() {
              self.continue_run();
              return;
            }
          }
          // Nothing left to summarize but the turn the context is full of.
          // Carrying on regardless would only fill it again and ask for the
          // same summary, so this is where it stops and the user decides.
          None if resuming => self.entries.push(Entry::Info(
            "The context is full and there is nothing left to compact — /continue to carry on anyway.".into(),
          )),
          None => self.notify("Nothing to compact."),
        }
        self.next_queued();
      }
      // Answered above, before a run was looked for: the list is the UI's
      // own errand and arrives whether or not one is going.
      AgentEvent::Models(_) => {}
    }
  }

  // ------------------------------------------------------------ drawing

  fn draw(&mut self, f: &mut Frame) {
    // A query is one line, whatever the prompt the box is holding for the
    // moment came to.
    // A questionnaire takes the keyboard whole, so a list under it is not
    // being narrowed while it is up.
    let filtering = self.overlay.is_some() && self.question.is_none();
    let lines = match filtering {
      true => 1,
      false => self.input.lines().len(),
    };
    let input_height = lines.clamp(1, MAX_INPUT_LINES) as u16 + 2;
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

    self.thumb = None;
    if self.question.is_some() {
      self.draw_question(f, transcript_area);
    } else if self.overlay.is_some() {
      self.draw_overlay(f, transcript_area);
    } else {
      self.draw_transcript(f, transcript_area);
    }

    // Input box. A list that is being narrowed borrows it for the query,
    // which is why the query is not drawn in the list's own title: it is
    // typed into a box, so it is shown in one, and the box is already there.
    // What was half-written in it is not lost — it is drawn again the moment
    // the list is gone.
    let border_color = if self.run.is_some() {
      Color::DarkGray
    } else {
      Color::Gray
    };
    let mut block = Block::default()
      .borders(Borders::ALL)
      .border_type(BorderType::Rounded)
      .border_style(Style::default().fg(border_color));
    if let Some(filter) = self.overlay.as_mut().filter(|_| filtering).map(|o| &mut o.filter) {
      filter.field.set_block(block.title(" filter "));
      f.render_widget(&filter.field, input_area);
    } else {
      // What the `@tokens` in the box found, said along the bottom border
      // rather than on a line of its own: it costs the transcript nothing,
      // and it is gone again the moment the tokens are.
      if let Some(strip) = attachment_strip(&self.attachments, self.cfg.vision, input_area.width) {
        block = block.title_bottom(strip);
      }
      self.input.set_block(block);
      f.render_widget(&self.input, input_area);
    }
    self.draw_footer(f, footer_area);
    self.draw_toast(f, transcript_area);
  }

  /// Drawn last, over whatever the transcript area holds, so it shows above a
  /// list or a question just as it does above the conversation.
  fn draw_toast(&mut self, f: &mut Frame, area: Rect) {
    if self.toast.as_ref().is_some_and(|(_, until)| Instant::now() >= *until) {
      self.toast = None;
    }
    let Some((text, _)) = &self.toast else { return };
    // A list or a question is framed, and its top border says which keys do
    // what, so the toast goes inside the frame rather than over the hints.
    let area = match self.overlay.is_some() || self.question.is_some() {
      true => area.inner(Margin::new(1, 1)),
      false => area,
    };
    // Wrapped to half the screen, so a sentence stays in the corner rather
    // than turning into a banner across the conversation.
    let most = (area.width / 2).max(24).min(area.width.saturating_sub(4));
    let lines = crate::markdown::wrap_text(text, most, Style::default());
    let widest = lines.iter().map(Line::width).max().unwrap_or(0) as u16;
    let width = (widest + 4).min(area.width);
    let toast = Rect {
      x: area.right().saturating_sub(width),
      y: area.y,
      width,
      height: (lines.len() as u16 + 2).min(area.height),
    };
    let block = Block::default()
      .borders(Borders::ALL)
      .border_type(BorderType::Rounded)
      .border_style(Style::default().fg(Color::Gray))
      .padding(Padding::horizontal(1));
    f.render_widget(Clear, toast);
    f.render_widget(Paragraph::new(lines).block(block), toast);
  }

  fn draw_transcript(&mut self, f: &mut Frame, transcript_area: Rect) {
    // Pinned to the bottom unless the user scrolled up. In `always` mode
    // the scrollbar gets its own column; in `auto` mode it is
    // overlaid on the transcript's last column while visible.
    let mut content_area = transcript_area;
    if self.scrollbar == ScrollbarMode::Always && content_area.width > 1 {
      content_area.width -= 1;
    }
    // Wrapped here rather than by the `Paragraph`, which wraps as it draws
    // and keeps where each line landed to itself: a row on screen is a line
    // of this list, which is what lets the mouse be told what it points at.
    let lines = self.transcript_lines(content_area.width);
    let total = lines.len();
    let viewport = content_area.height as usize;
    let max_scroll = total.saturating_sub(viewport);
    if self.anchor.is_some_and(|a| a >= max_scroll) {
      self.anchor = None;
    }
    let offset = self.anchor.unwrap_or(max_scroll);
    self.view = (offset, max_scroll);
    let visible: Vec<Line<'static>> = lines
      .iter()
      .enumerate()
      .skip(offset)
      .take(viewport)
      .map(|(at, line)| match self.selection.and_then(|s| s.columns(at)) {
        Some((from, to)) => highlight(line, from, to),
        None => line.clone(),
      })
      .collect();
    f.render_widget(Paragraph::new(visible), content_area);
    self.rendered = lines;
    self.content = content_area;
    let overflows = total > viewport;
    let show_scrollbar = match self.scrollbar {
      ScrollbarMode::Always => viewport > 0,
      ScrollbarMode::Auto => overflows && self.scrollbar_fading(),
      ScrollbarMode::Hidden => false,
    };
    // A scrollbar with nothing to scroll draws neither track nor thumb, and
    // an `auto` one that has faded draws nothing at all, so in either case
    // there is nothing for the mouse to take hold of — the column it was on
    // is transcript again, to be read and selected like the rest of it: the
    // thumb `draw` cleared stays cleared.
    if show_scrollbar {
      if max_scroll > 0 && transcript_area.width > 0 {
        let track = Rect {
          x: transcript_area.right() - 1,
          width: 1,
          ..transcript_area
        };
        let (start, len) = thumb_bounds(track.height, max_scroll, viewport, offset);
        self.thumb = Some(Thumb { track, start, len });
      }
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
    let command_width = COMMANDS.iter().map(|(n, ..)| n.len() + 1).max().unwrap_or(0);
    // A file list is as wide as its longest name, so the sizes line up in a
    // column of their own.
    let path_width = c
      .items
      .iter()
      .filter_map(|m| match m {
        Match::Path { name, dir, .. } => Some(name.chars().count() + usize::from(*dir) + 1),
        Match::Command { .. } => None,
      })
      .max()
      .unwrap_or(0);
    let dim = Style::default().add_modifier(Modifier::DIM);
    let lines: Vec<Line> = c
      .items
      .iter()
      .enumerate()
      .skip(first)
      .take(rows)
      .map(|(i, m)| {
        let selected = i == c.selected;
        let base = if selected {
          Style::default().bold()
        } else {
          Style::default()
        };
        let mut spans = vec![pointer(selected)];
        let (sigil, name, highlights, trailing, width, right) = match m {
          Match::Command { index, highlights } => {
            let (name, description, _) = COMMANDS[*index];
            (
              "/",
              name.to_string(),
              highlights,
              String::new(),
              command_width,
              description.to_string(),
            )
          }
          Match::Path {
            name,
            highlights,
            meta,
            dir,
            ..
          } => (
            "",
            name.clone(),
            highlights,
            if *dir { "/".to_string() } else { String::new() },
            path_width,
            meta.clone(),
          ),
        };
        spans.push(Span::styled(sigil, base));
        spans.extend(picked_out(&name, highlights, base));
        let shown = name.chars().count() + trailing.chars().count();
        spans.push(Span::styled(trailing, base.fg(Color::Cyan)));
        spans.push(Span::raw(" ".repeat(width.saturating_sub(shown))));
        spans.push(Span::styled(right, dim));
        Line::from(spans)
      })
      .collect();
    f.render_widget(Paragraph::new(lines), area);
  }

  fn draw_question(&self, f: &mut Frame, area: Rect) {
    let Some(question) = &self.question else { return };
    // The dialog says which keys do what along its own bottom, where the
    // answer to that changes with the question.
    let block = Block::default()
      .borders(Borders::ALL)
      .border_type(BorderType::Rounded)
      .border_style(Style::default().fg(Color::Gray))
      .title(" The model is asking ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    let height = inner.height as usize;
    let (lines, focus) = question.dialog.lines(inner.width);
    // More dialog than screen: scroll it just far enough to keep the row the
    // cursor is on in view, which is the row being answered.
    let first = focus.saturating_sub(height.saturating_sub(1)).min(focus);
    f.render_widget(
      Paragraph::new(lines.into_iter().skip(first).take(height).collect::<Vec<_>>()),
      inner,
    );
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
    // Only the rows the filter left, which is all of them until something is
    // typed.
    let shown = overlay.matched();
    // A filter that has narrowed the list to nothing says so, rather than
    // leaving an empty box to be read as a provider with no models, or as a
    // directory with nothing saved in it.
    let emptied = shown.is_empty().then_some(match &overlay.list {
      OverlayList::Models(_) => "  No model matches.",
      OverlayList::Sessions(_) => "  No session matches.",
      OverlayList::Goto(_) => "  No prompt matches.",
      _ => "  No point matches.",
    });
    if let Some(text) = emptied {
      let line = Line::from(Span::styled(text, Style::default().add_modifier(Modifier::DIM)));
      f.render_widget(Paragraph::new(line), inner);
      return;
    }
    // The window ends at the selection, so moving down walks off the bottom
    // rather than jumping the list around.
    let first = overlay.selected.saturating_sub(height.saturating_sub(1));
    // Each row is a title and a dim note about it, the title clipped so the
    // note always fits.
    let rows: Vec<(String, String)> = match &overlay.list {
      OverlayList::Sessions(sessions) => shown
        .iter()
        .filter_map(|(at, _)| sessions.get(*at))
        .enumerate()
        .map(|(i, s)| {
          // The row being asked about says what it is being asked, where it
          // otherwise says what it is.
          let note = match overlay.confirming && i == overlay.selected {
            true => "delete? Ctrl+D to confirm".to_string(),
            false => format!(
              "{}  {} msgs  {}",
              shorten_home(Path::new(&s.cwd)),
              s.message_count,
              s.age()
            ),
          };
          (s.title().to_string(), note)
        })
        .collect(),
      // The tree is indented at its branch points, so a conversation that
      // went two ways reads as two ways. Both lists say how long the
      // conversation would be once you got there.
      OverlayList::Tree(points) => shown
        .iter()
        .filter_map(|(at, _)| points.get(*at))
        .enumerate()
        .map(|(i, p)| {
          let note = match (overlay.confirming && i == overlay.selected, p.here) {
            (true, _) => "delete? Ctrl+D to confirm".to_string(),
            (false, true) => "here".to_string(),
            (false, false) => messages(p.len),
          };
          (point_label(p), note)
        })
        .collect(),
      // Prompts alone, drawn flat: a fork is started from a question rather
      // than from a place in a shape, and there is no branch to read here.
      OverlayList::Fork(points) => shown
        .iter()
        .filter_map(|(at, _)| points.get(*at))
        .map(|p| (p.label.clone(), format!("keeps {}", messages(p.len))))
        .collect(),
      OverlayList::Goto(marks) => shown
        .iter()
        .filter_map(|(at, _)| marks.get(*at))
        .map(|m| (m.label.clone(), String::new()))
        .collect(),
      // The window on the right where a session says its size: it is the
      // one thing about a model worth choosing between, and most providers
      // do not report it at all.
      OverlayList::Models(models) => shown
        .iter()
        .filter_map(|(at, _)| models.get(*at))
        .map(|model| {
          let note = match (&model.name, model.context_length) {
            (_, Some(window)) => format!("{} ctx", window_label(window)),
            (Some(name), None) => name.clone(),
            (None, None) => String::new(),
          };
          let here = self.cfg.model == model.id;
          (format!("{}{}", model.id, if here { "  (in use)" } else { "" }), note)
        })
        .collect(),
    };
    let dim = Style::default().add_modifier(Modifier::DIM);
    let mut lines = Vec::new();
    for (i, (title, note)) in rows.into_iter().enumerate().skip(first).take(height) {
      let selected = i == overlay.selected;
      let avail = width.saturating_sub(2 + note.chars().count() + 2);
      let title: String = title.chars().take(avail).collect();
      let pad = width.saturating_sub(2 + title.chars().count() + note.chars().count());
      let base = match selected {
        true => Style::default().bold(),
        false => Style::default(),
      };
      // The letters the filter matched, picked out the way the `/` popup
      // picks out its own. Nothing typed is nothing picked out.
      let highlights = shown.get(i).map(|(_, hits)| hits.as_slice()).unwrap_or(&[]);
      let mut spans = vec![pointer(selected)];
      spans.extend(picked_out(&title, highlights, base));
      spans.push(Span::raw(" ".repeat(pad)));
      spans.push(Span::styled(note, dim));
      lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), inner);
  }

  /// What the context comes to.
  ///
  /// The figure the last call was weighed at wherever there is one. Where
  /// there is not — before the first answer, and after a compaction, when the
  /// last figure is about a conversation that has been replaced — the same
  /// chars/4 estimate the cut point is chosen with, which is better than the
  /// footer going blank at the moment the room made is the thing to see.
  ///
  /// Nothing here is marked as a guess, because a figure that is nothing but
  /// the provider's own count is the exception rather than the rule: it is
  /// one turn behind whatever has happened since, it is a chars/4 estimate
  /// wherever the estimate ran higher, and it is an estimate throughout for a
  /// provider that reports no usage at all. A mark that honest would be on
  /// almost every figure, which is the same as being on none of them.
  fn context(&self) -> u64 {
    match self.context_tokens {
      Some(tokens) => tokens,
      None => estimate_tokens(&self.session.history),
    }
  }

  fn draw_footer(&self, f: &mut Frame, footer_area: Rect) {
    let cwd = shorten_home(&self.cwd);
    let mut left = vec![
      Span::raw(cwd).dim(),
      Span::raw("  "),
      Span::raw(model_label(&self.cfg)).dim(),
    ];
    if let Some(label) = mcp_label(self.mcp.0, self.mcp.1) {
      left.push(Span::raw("  "));
      left.push(Span::raw(label).dim());
    }
    if self.run.is_some() {
      left.push(Span::raw("  "));
      let verb = if self.compacting { "compacting" } else { "working" };
      left.push(Span::raw(format!("{} {verb} — esc to abort", SPINNER[self.tick % SPINNER.len()])).fg(Color::Yellow));
      let queued = self.agents.control.waiting().len();
      if queued > 0 {
        left.push(Span::raw(format!("  ({queued} queued)")).dim());
      }
    } else if let Some(anchor) = self.anchor {
      let behind = self.view.1 - anchor;
      left.push(Span::raw("  "));
      left.push(Span::raw(format!("↑ {behind} lines — pgdn to follow")).dim());
    }
    let context = (self.context() * 100)
      .checked_div(self.cfg.compaction.context_window)
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
    let mut starts = Vec::with_capacity(self.entries.len());
    for (at, entry) in self.entries.iter().enumerate() {
      starts.push(lines.len());
      match entry {
        Entry::User { text, images } => {
          lines.push(Line::default());
          for (i, l) in text.lines().enumerate() {
            let prefix = if i == 0 { "❯ " } else { "  " };
            lines.push(Line::from(vec![
              Span::styled(prefix, Style::default().fg(Color::Cyan).bold()),
              Span::styled(l.to_string(), Style::default().bold()),
            ]));
          }
          // What was attached, under the prompt that attached it and drawn
          // the same way a tool's picture is: at the width the transcript
          // has, folded, and cached by its bytes and that width.
          for image in images {
            let gutter = Span::styled("│", Style::default().fg(Color::Cyan));
            draw_image(
              image,
              gutter,
              width,
              self.tools_fold,
              (&mut cached, &mut live),
              &mut lines,
            );
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
          let hidden = self.thinking_fold.hidden(wrapped.len(), REASONING_LINES);
          // Collapsed, the header stands for the whole block and says nothing
          // about its size.
          let header = match (self.thinking_fold, hidden) {
            (Fold::Collapsed, _) | (_, 0) => "· thinking…".to_string(),
            (_, 1) => "· thinking… (1 earlier line hidden)".to_string(),
            (_, n) => format!("· thinking… ({n} earlier lines hidden)"),
          };
          lines.push(Line::styled(header, thinking));
          for line in &wrapped[hidden..] {
            lines.push(prefix(REASONING_INDENT, line.clone()));
          }
          live.insert(key, wrapped);
        }
        Entry::ToolCall { name, summary, .. } => {
          lines.push(Line::default());
          lines.extend(call_lines(name, self.tools_fold.summary(summary), dim));
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
            (None, Some(diff)) => (
              diff_lines(edited_by(&self.entries[..at], call), diff),
              DIFF_LINES,
              false,
            ),
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
              fold: self.tools_fold,
              width: width as usize,
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
            draw_image(
              image,
              stripe.clone(),
              width,
              self.tools_fold,
              (&mut cached, &mut live),
              &mut lines,
            );
          }
          if name == "bash" && (*running || took.is_some()) && self.tools_fold != Fold::Collapsed {
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
          // Counted in lines as drawn, as a reasoning block is, so a long
          // paragraph cannot slip past the fold as one line.
          let body = width.saturating_sub(2);
          let key = (hash(SUMMARY_KIND, text), body, false);
          let wrapped = match cached.remove(&key) {
            Some(wrapped) => wrapped,
            None => crate::markdown::wrap_text(text, body, dim),
          };
          // The summary opens with the goal, so a preview keeps its start.
          let hidden = self.summary_fold.hidden(wrapped.len(), SUMMARY_LINES);
          for line in &wrapped[..wrapped.len() - hidden] {
            lines.push(prefix("  ", line.clone()));
          }
          if hidden > 0 && self.summary_fold == Fold::Preview {
            let note = match hidden {
              1 => "  … 1 more line (ctrl+s)".to_string(),
              n => format!("  … {n} more lines (ctrl+s)"),
            };
            lines.push(Line::styled(note, mark_style(None)));
          }
          live.insert(key, wrapped);
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
        //
        // An edit's halves are lines of the file it names, so they are
        // coloured like it — the same colours the diff they become is drawn
        // in, which is what keeps the call from recolouring as it lands.
        _ => {
          let path = (writing.name == "edit")
            .then(|| writing_path(&writing.args))
            .flatten()
            .unwrap_or_default();
          blocks
            .iter()
            .flat_map(|(mark, text)| {
              let prefix = match *mark {
                "│" => String::new(),
                mark => format!("{mark} "),
              };
              marked_code(&prefix, mark_style(mark.chars().next()), language_of(&path), text)
            })
            .collect()
        }
      };
      // Drawn and highlighted as it is typed, so nothing moves or recolours
      // under the reader when the call is finally made.
      let mut header = call_lines(&writing.name, self.tools_fold.summary(&summary), dim);
      // The cursor follows the model: at the end of the arguments until there
      // is a body to write into, then at the end of what has arrived — or
      // stays on the arguments when the body is folded away entirely.
      if (body.is_empty() || self.tools_fold == Fold::Collapsed)
        && let Some(last) = header.last_mut()
      {
        last.spans.push(Span::styled("▌", Style::default().fg(Color::Yellow)));
      }
      lines.extend(header);
      Preview {
        body,
        // Nothing has gone right or wrong yet, so the plain gutter.
        gutter: gutter(true, false),
        cap: TOOL_OUTPUT_LINES,
        fold: self.tools_fold,
        width: width as usize,
        // The tail is what is being written; what came before is already said.
        from_end: true,
        cursor: true,
      }
      .draw(&mut lines);
    }
    // What the user typed while the run was going, at the end because that is
    // where it will be sent from — under everything the run is still saying,
    // not above it.
    let waiting = self.agents.control.waiting();
    if !waiting.is_empty() {
      lines.push(Line::default());
      for text in &waiting {
        lines.push(Line::styled(format!("Queued: {}", first_line(text)), dim.italic()));
      }
    }
    self.starts = starts;
    self.markdown = live;
    // Every entry that renders itself has already wrapped to the width; this
    // is for the ones shown as they were written — a prompt, an error, a
    // block unfolded — and it is what makes the list a list of rows.
    lines
      .into_iter()
      .flat_map(|line| crate::markdown::wrap_line(line, width))
      .collect()
  }
}

/// The mark a picker puts on the row the cursor is on.
fn pointer(selected: bool) -> Span<'static> {
  Span::styled(
    if selected { "› " } else { "  " },
    Style::default().fg(Color::Cyan).bold(),
  )
}

/// `text` in `base`, with the letters at `highlights` picked out in cyan and
/// underlined — the letters of a picker's row a query matched.
fn picked_out(text: &str, highlights: &[u32], base: Style) -> Vec<Span<'static>> {
  // One span a letter only where there are letters to pick out.
  if highlights.is_empty() {
    return vec![Span::styled(text.to_string(), base)];
  }
  let hit = base.fg(Color::Cyan).underlined();
  text
    .chars()
    .enumerate()
    .map(|(at, ch)| {
      Span::styled(
        ch.to_string(),
        if highlights.contains(&(at as u32)) { hit } else { base },
      )
    })
    .collect()
}

/// Rendered text and pictures, by what they were drawn from and at what width.
type RenderCache = HashMap<(u64, u16, bool), Vec<Line<'static>>>;

/// An image in the transcript, under `gutter`, folded like any other block.
/// Its drawing is taken from last frame's cache when it is there, and kept for
/// the next.
fn draw_image(
  image: &[u8],
  gutter: Span<'static>,
  width: u16,
  fold: Fold,
  (cached, live): (&mut RenderCache, &mut RenderCache),
  lines: &mut Vec<Line<'static>>,
) {
  // Two columns of indent and the gutter, as every other block of a tool's
  // output is drawn.
  let cols = width.saturating_sub(3);
  let key = (hash(IMAGE_KIND, image), cols, false);
  let drawn = cached.remove(&key).unwrap_or_else(|| {
    crate::images::blocks(image, cols, IMAGE_MAX_LINES).unwrap_or_else(|| {
      vec![Line::styled(
        "[image could not be drawn]",
        Style::default().add_modifier(Modifier::DIM),
      )]
    })
  });
  Preview {
    body: drawn.iter().map(|line| line.spans.clone()).collect(),
    gutter,
    cap: IMAGE_LINES,
    fold,
    width: width as usize,
    from_end: false,
    cursor: false,
  }
  .draw(lines);
  live.insert(key, drawn);
}

/// Identifies a message or an image by content, for the rendered-text cache.
/// `kind` keeps an assistant message and a reasoning block apart when they
/// read the same.
fn hash<T: Hash + ?Sized>(kind: u8, content: &T) -> u64 {
  let mut hasher = DefaultHasher::new();
  kind.hash(&mut hasher);
  content.hash(&mut hasher);
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
/// What the input box's `@tokens` resolved to, as the bottom border reads
/// it: each attachment named with the size it will be sent at, and each
/// token that found nothing marked as the mistake it probably is.
///
/// `None` when there is nothing to say, which leaves the border plain.
fn attachment_strip(tokens: &[Token], vision: bool, width: u16) -> Option<Line<'static>> {
  if tokens.is_empty() {
    return None;
  }
  let mut spans = vec![Span::raw("─ ")];
  // Two for the corners, two for the lead-in, one for the trailing space.
  let room = usize::from(width).saturating_sub(5);
  let mut used = 0;
  let mut dropped = 0;
  for token in tokens {
    let (text, style) = match token.state {
      attach::State::Image { width, height } => (
        format!("▣ {} {width}×{height}", token.name()),
        match vision {
          true => Style::default().fg(Color::Cyan),
          // It resolved, but this model will not be sent it.
          false => Style::default().fg(Color::DarkGray),
        },
      ),
      // Not a mistake: a path still being completed reads as a directory
      // right up until it names a file, and the popup below is already
      // listing what is in it.
      attach::State::Directory => (
        format!("▤ {}/", token.name()),
        Style::default().add_modifier(Modifier::DIM),
      ),
      attach::State::Missing => (
        format!("⚠ {} not found", token.name()),
        Style::default().fg(Color::Yellow),
      ),
      attach::State::NotAnImage => (
        format!("⚠ {} not an image", token.name()),
        Style::default().fg(Color::Yellow),
      ),
    };
    let sep = if used == 0 { 0 } else { 2 };
    let wide = text.chars().count() + sep;
    // What will not fit is counted rather than cut in half.
    if used + wide > room.saturating_sub(if dropped > 0 { 3 } else { 0 }) {
      dropped += 1;
      continue;
    }
    if sep > 0 {
      spans.push(Span::raw("  "));
    }
    spans.push(Span::styled(text, style));
    used += wide;
  }
  if dropped > 0 {
    spans.push(Span::styled(
      format!(" +{dropped}"),
      Style::default().fg(Color::DarkGray),
    ));
  }
  spans.push(Span::raw(" "));
  Some(Line::from(spans))
}

/// Read the images `text`'s `@tokens` name. The notes are for the
/// transcript: what could not be attached, and why.
///
/// A slash command is only ever its text, and text with no `@` in it never
/// touches the disk.
fn attach_images(text: &str, cwd: &Path, vision: bool) -> (Prompt, Vec<String>) {
  if text.starts_with('/') || !text.contains('@') {
    return (Prompt::text(text.to_string()), Vec::new());
  }
  let tokens = attach::tokens(text, cwd);
  let mut images = Vec::new();
  let mut notes = Vec::new();
  for token in &tokens {
    match &token.state {
      attach::State::Image { .. } if !vision => notes.push(format!(
        "Not attached — {}: this model does not take images (--no-vision).",
        token.text
      )),
      attach::State::Image { .. } => match attach::load(token) {
        Ok(image) => images.push(image),
        Err(reason) => notes.push(format!("Not attached — {reason}")),
      },
      // A token that resolved to nothing is worth saying out loud: it reads
      // like an attachment in the prompt, and nothing else would say that
      // the model was never given one.
      attach::State::Directory => notes.push(format!("Not attached — {}: that is a directory.", token.text)),
      attach::State::Missing => notes.push(format!("Not attached — {}: no such file.", token.text)),
      attach::State::NotAnImage => notes.push(format!("Not attached — {}: not an image fa can send.", token.text)),
    }
  }
  (
    Prompt {
      text: text.to_string(),
      images,
    },
    notes,
  )
}

/// A pasted path to an image, as the token that attaches it — relative to
/// the working directory when it is under it, and quoted when it has a space
/// in it. `None` for a paste that is anything else, which is nearly every
/// paste.
fn as_token(text: &str, cwd: &Path) -> Option<String> {
  let trimmed = text.trim();
  // One path and nothing else. A pasted paragraph that mentions a file is
  // not a request to attach it.
  if trimmed.is_empty() || trimmed.contains('\n') || trimmed.starts_with('@') {
    return None;
  }
  let path = crate::tools::resolve(cwd, trimmed);
  attach::dimensions(&path)?;
  let shown = path.strip_prefix(cwd).unwrap_or(&path).to_string_lossy().into_owned();
  Some(match shown.contains(' ') {
    true => format!("@\"{shown}\" "),
    false => format!("@{shown} "),
  })
}

/// What a popup row is, for keeping the selection on the same row as the
/// list is filtered down.
fn key(item: &Match) -> String {
  match item {
    Match::Command { index, .. } => COMMANDS[*index].0.to_string(),
    Match::Path { insert, .. } => insert.clone(),
  }
}

/// Which of `rows` `query` leaves, best match first: each as its index into
/// `rows` and the letters of it the query matched.
///
/// The same fuzzy match the `/` popup is filtered by, over the one string a
/// row is drawn as — so every letter picked out is a letter on screen.
fn filter_rows<'a>(
  matcher: &mut Matcher,
  rows: impl IntoIterator<Item = &'a str>,
  query: &str,
) -> Vec<(usize, Vec<u32>)> {
  let rows = rows.into_iter().enumerate();
  if query.is_empty() {
    return rows.map(|(at, _)| (at, Vec::new())).collect();
  }
  let pattern = Pattern::parse(query, CaseMatching::Ignore, Normalization::Smart);
  let mut buf = Vec::new();
  let mut scored: Vec<(u32, usize, Vec<u32>)> = rows
    .filter_map(|(at, row)| {
      let mut highlights = Vec::new();
      let score = pattern.indices(Utf32Str::new(row, &mut buf), matcher, &mut highlights)?;
      highlights.sort_unstable();
      highlights.dedup();
      Some((score, at, highlights))
    })
    .collect();
  // Ties keep the order the list was in, which is the order it was sorted
  // into: a query matching a whole family of models lists them as a family,
  // and a query matching several sessions lists the newest of them first.
  scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
  scored.into_iter().map(|(_, at, hits)| (at, hits)).collect()
}

/// Which models `query` leaves, matched over ids alone: a provider's display
/// name is another spelling of the same thing, and matching both would rank a
/// model twice for looking like itself.
fn filter_models(matcher: &mut Matcher, models: &[ModelInfo], query: &str) -> Vec<(usize, Vec<u32>)> {
  filter_rows(matcher, models.iter().map(|model| model.id.as_str()), query)
}

/// Which sessions `query` leaves, matched over the titles: what a session is
/// remembered by is its name, or the question it opened with.
fn filter_sessions(matcher: &mut Matcher, sessions: &[SessionInfo], query: &str) -> Vec<(usize, Vec<u32>)> {
  filter_rows(matcher, sessions.iter().map(SessionInfo::title), query)
}

/// Which points `query` leaves, matched over the rows they are drawn as —
/// indent and all, so the letters picked out sit under the letters typed.
///
/// The rows are built to be matched and built again to be drawn, which is two
/// short strings a keystroke and a list that cannot say one thing and match
/// another.
fn filter_points(matcher: &mut Matcher, points: &[Point], query: &str) -> Vec<(usize, Vec<u32>)> {
  let labels: Vec<String> = points.iter().map(point_label).collect();
  filter_rows(matcher, labels.iter().map(String::as_str), query)
}

/// A point as its row reads: indented by the branch points crossed to reach
/// it.
fn point_label(point: &Point) -> String {
  format!("{}{}", "  ".repeat(point.depth), point.label)
}

fn filter_commands(matcher: &mut Matcher, query: &str) -> Vec<Match> {
  filter_rows(matcher, COMMANDS.iter().map(|(name, ..)| *name), query)
    .into_iter()
    .map(|(index, highlights)| Match::Command { index, highlights })
    .collect()
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

/// The file an `edit` was asked to change, from the asking.
fn edited_path(name: &str, args: &serde_json::Value) -> Option<String> {
  (name == "edit")
    .then(|| args.get("path")?.as_str().map(str::to_string))
    .flatten()
}

/// The file an edit changed, from the call that asked for it.
fn edited_by<'a>(entries: &'a [Entry], call: &str) -> Option<&'a str> {
  entries.iter().rev().find_map(|entry| match entry {
    Entry::ToolCall {
      call: at,
      edited: Some(path),
      ..
    } if at == call => Some(path.as_str()),
    _ => None,
  })
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
  marked_code("", mark_style(None), language_of(path), content)
}

/// What a file's name says it is written in — its extension, which is all a
/// transcript ever has to go on, and the empty string when it has not even
/// that.
fn language_of(path: &str) -> &str {
  Path::new(path)
    .extension()
    .and_then(|extension| extension.to_str())
    .unwrap_or_default()
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
  Some(marked_code("", mark_style(None), "json", &pretty))
}

/// A diff as the transcript shows it: `+12` still says what became of the
/// line, and the code after it is highlighted as the file it was changed in.
///
/// The mark and the number keep the colour they always had, because once the
/// code carries the grammar's colours they are the only thing left saying what
/// changed. Context is dimmed over its highlighting, so what the edit did still
/// comes forward. With no path, no grammar for it, or a diff this cannot take
/// apart, it falls back to the marked-up text it has always been.
fn diff_lines(path: Option<&str>, diff: &str) -> Vec<Vec<Span<'static>>> {
  let Some((language, column)) = path.map(language_of).zip(code_column(diff)) else {
    return marked_lines(diff, true);
  };
  // A line too short to reach that column is all numbering and no code.
  let split: Vec<(&str, &str)> = diff
    .lines()
    .map(|line| line.split_at_checked(column).unwrap_or((line, "")))
    .collect();

  // Each side of the edit, put back together from the lines that belong to
  // it: a `-` line is only in the old file and a `+` line only in the new,
  // and everything around them is in both. Highlighting the two as files is
  // what a string or a comment running over several lines needs — one line on
  // its own says nothing about where it started.
  let (mut old, mut new) = (String::new(), String::new());
  let mut placed: Vec<Option<(bool, usize)>> = Vec::with_capacity(split.len());
  let (mut olds, mut news) = (0, 0);
  for (head, code) in &split {
    let push = |side: &mut String| {
      side.push_str(code);
      side.push('\n');
    };
    match head.as_bytes().first() {
      Some(b'+') => {
        push(&mut new);
        placed.push(Some((true, news)));
        news += 1;
      }
      Some(b'-') => {
        push(&mut old);
        placed.push(Some((false, olds)));
        olds += 1;
      }
      // The line where context was skipped is numbered by nothing and is part
      // of neither file.
      _ if !head.bytes().any(|b| b.is_ascii_digit()) => placed.push(None),
      _ => {
        push(&mut old);
        push(&mut new);
        olds += 1;
        placed.push(Some((true, news)));
        news += 1;
      }
    }
  }

  let (old, new) = (
    crate::highlight::highlight(language, &old),
    crate::highlight::highlight(language, &new),
  );
  if old.is_none() && new.is_none() {
    return marked_lines(diff, true);
  }
  split
    .iter()
    .zip(placed)
    .map(|((head, code), place)| {
      let mark = head.chars().next();
      let style = mark_style(mark);
      let spans = place
        .and_then(|(is_new, i)| if is_new { new.as_ref() } else { old.as_ref() }?.get(i))
        .filter(|spans| !spans.is_empty());
      let mut row = vec![Span::styled(format!(" {head}"), style)];
      match spans {
        // Context is the file as it was and as it stays; dimming it over its
        // own colours is what keeps the changed lines the ones that read.
        Some(spans) if mark == Some(' ') => row.extend(
          spans
            .iter()
            .map(|span| Span::styled(span.content.clone(), span.style.add_modifier(Modifier::DIM))),
        ),
        Some(spans) => row.extend(spans.iter().cloned()),
        None => row.push(Span::styled((*code).to_string(), style)),
      }
      row
    })
    .collect()
}

/// Lines of `text` drawn under `prefix`, highlighted as `language` where
/// there is a grammar for it and left in `style` where there is not.
///
/// The prefix keeps `style` either way: it is the transcript talking about the
/// line — the half of an edit it belongs to — not part of the line itself.
fn marked_code(prefix: &str, style: Style, language: &str, text: &str) -> Vec<Vec<Span<'static>>> {
  let highlighted = crate::highlight::highlight(language, text);
  // Split rather than `lines`, so a body ending in a newline keeps the empty
  // line the model is about to write into.
  text
    .split('\n')
    .enumerate()
    .map(|(i, line)| {
      let mut row = vec![Span::styled(format!(" {prefix}"), style)];
      match highlighted
        .as_ref()
        .and_then(|lines| lines.get(i))
        .filter(|spans| !spans.is_empty())
      {
        Some(spans) => row.extend(spans.iter().cloned()),
        None => row.push(Span::styled(line.to_string(), style)),
      }
      row
    })
    .collect()
}

/// The column a diff's own numbering gives way to the code.
///
/// Every line is written as a mark, a line number right-aligned to the width
/// of the longest, and a space — so the column is the same on every line, and
/// the narrowest number is the one that finds it: code that is itself a number
/// can only push the guess further right, and the skipped-context line has no
/// number to go by at all.
fn code_column(diff: &str) -> Option<usize> {
  diff
    .lines()
    .filter_map(|line| {
      let number = line.bytes().enumerate().skip(1);
      let digits = number.take_while(|(_, b)| b.is_ascii_digit() || *b == b' ');
      let last = digits.filter(|(_, b)| b.is_ascii_digit()).map(|(i, _)| i).last()?;
      Some(last + 2)
    })
    .min()
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

/// How much of a folding block is shown: nothing, the first or last few lines
/// of it, or all of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fold {
  Collapsed,
  Preview,
  Full,
}

impl Fold {
  /// The next one the key steps to. From the preview a press shows the rest,
  /// as it always has, and a second one folds the block away.
  fn next(self) -> Fold {
    match self {
      Fold::Preview => Fold::Full,
      Fold::Full => Fold::Collapsed,
      Fold::Collapsed => Fold::Preview,
    }
  }

  fn label(self) -> &'static str {
    match self {
      Fold::Collapsed => "collapsed",
      Fold::Preview => "previewed",
      Fold::Full => "expanded",
    }
  }

  /// What of a call's arguments is shown on its line: none of them once its
  /// block is collapsed, which leaves only the tool's name.
  fn summary(self, summary: &str) -> &str {
    match self {
      Fold::Collapsed => "",
      _ => summary,
    }
  }

  /// How many of `len` lines are folded away, when a preview keeps `cap`.
  fn hidden(self, len: usize, cap: usize) -> usize {
    match self {
      Fold::Collapsed => len,
      Fold::Preview => len.saturating_sub(cap),
      Fold::Full => 0,
    }
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
  fold: Fold,
  /// The columns the transcript has, which is what a folded line is cut to.
  width: usize,
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
      fold,
      width,
      from_end,
      cursor,
    } = self;
    // Collapsed, the block is not drawn at all, not even to say it is there.
    if fold == Fold::Collapsed {
      return;
    }
    let hidden = fold.hidden(body.len(), cap);
    let row = |line: Vec<Span<'static>>, tip: bool| {
      let mut spans = vec![Span::raw("  "), gutter.clone()];
      // Folded, a line is a row: one that wraps spends rows the fold was
      // counting, so a block held to ten lines could still fill the screen.
      // Unfolding shows the rest — of a long line as much as of a long block.
      spans.extend(match fold == Fold::Full {
        true => line,
        // The two columns of indent, the gutter, and the cursor when there is
        // one, are the room the text does not have.
        false => clip(line, width.saturating_sub(3 + usize::from(tip))),
      });
      if tip {
        spans.push(Span::styled("▌", Style::default().fg(Color::Yellow)));
      }
      Line::from(spans)
    };
    let note = vec![Span::styled(fold_note(hidden, from_end), mark_style(None))];
    if hidden > 0 && from_end {
      out.push(row(note.clone(), false));
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
      out.push(row(note, false));
    }
  }
}

/// A line split at columns `from` and `to`: what is drawn before them, what is
/// drawn between them, and what is drawn after. The span a cut falls inside is
/// itself cut, and each piece keeps the styling of the span it came out of.
///
/// Columns as the terminal counts them, so a wide character is the two cells
/// it is drawn on, and one straddling a cut goes to the side its first cell is
/// on.
fn cut(line: &Line<'static>, from: usize, to: usize) -> (Vec<Span<'static>>, Vec<Span<'static>>, Vec<Span<'static>>) {
  use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
  let mut parts: [Vec<Span<'static>>; 3] = Default::default();
  let mut at = 0;
  for span in &line.spans {
    let width = span.content.width();
    // A span the cuts miss is one of the three pieces as it stands.
    let whole = match (at + width <= from, at >= to) {
      (true, _) => Some(0),
      (_, true) => Some(2),
      _ if at >= from && at + width <= to => Some(1),
      _ => None,
    };
    if let Some(part) = whole {
      parts[part].push(span.clone());
      at += width;
      continue;
    }
    let mut pieces = [String::new(), String::new(), String::new()];
    for ch in span.content.chars() {
      let part = match (at < from, at < to) {
        (true, _) => 0,
        (_, true) => 1,
        _ => 2,
      };
      pieces[part].push(ch);
      at += ch.width().unwrap_or(0);
    }
    for (part, text) in pieces.into_iter().enumerate() {
      if !text.is_empty() {
        parts[part].push(Span::styled(text, span.style));
      }
    }
  }
  let [before, inside, after] = parts;
  (before, inside, after)
}

/// The text of `line` between columns `from` and `to`, without the blanks a
/// line ends in — which are the gap to the right margin rather than anything
/// written on it.
fn selected(line: &Line<'static>, from: usize, to: usize) -> String {
  let text: String = cut(line, from, to).1.iter().map(|span| span.content.as_ref()).collect();
  text.trim_end().to_string()
}

/// `line` with its columns between `from` and `to` drawn as selected.
///
/// Reversed rather than given a colour of its own, so the selection reads as
/// one against any of the colours the transcript is drawn in, and takes the
/// terminal's own idea of what a selection looks like with it.
fn highlight(line: &Line<'static>, from: usize, to: usize) -> Line<'static> {
  let (before, inside, after) = cut(line, from, to);
  if inside.is_empty() {
    return line.clone();
  }
  let mut spans = before;
  spans.extend(
    inside
      .into_iter()
      .map(|span| Span::styled(span.content, span.style.add_modifier(Modifier::REVERSED))),
  );
  spans.extend(after);
  Line {
    spans,
    style: line.style,
    alignment: line.alignment,
  }
}

/// A line's spans cut to `width` columns with an ellipsis where the rest of it
/// was, or as they are when they fit.
///
/// Columns rather than characters, since what is being fitted is a terminal,
/// and the styling of what is kept is kept with it: a clipped line is the same
/// line, shorter.
fn clip(line: Vec<Span<'static>>, width: usize) -> Vec<Span<'static>> {
  use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
  if line.iter().map(|span| span.content.width()).sum::<usize>() <= width {
    return line;
  }
  // Nowhere to put even the ellipsis: the transcript is narrower than its own
  // gutter, and there is nothing useful to say in what is left.
  if width == 0 {
    return Vec::new();
  }
  let mut out: Vec<Span<'static>> = Vec::with_capacity(line.len() + 1);
  let mut used = 0;
  for span in line {
    let span_width = span.content.width();
    // A column short of the width, since the ellipsis is going to want one.
    if used + span_width < width {
      used += span_width;
      out.push(span);
      continue;
    }
    // The span the line runs out in, kept as far as it goes.
    let mut kept = String::new();
    for c in span.content.chars() {
      let w = c.width().unwrap_or(0);
      if used + w + 1 > width {
        break;
      }
      kept.push(c);
      used += w;
    }
    if !kept.is_empty() {
      out.push(Span::styled(kept, span.style));
    }
    break;
  }
  out.push(Span::styled("…", mark_style(None)));
  out
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

/// Code as the lines it is written on, so a command reads as the command it is
/// and a call's arguments as the JSON they are.
///
/// Falls back to the plain line wherever the grammar was not built in or had
/// nothing to say about it, which is the same text either way.
fn code_spans(language: &str, code: &str, style: Style) -> Vec<Vec<Span<'static>>> {
  let highlighted = crate::highlight::highlight(language, code);
  code
    .split('\n')
    .enumerate()
    .map(|(i, line)| {
      match highlighted
        .as_ref()
        .and_then(|lines| lines.get(i))
        .filter(|spans| !spans.is_empty())
      {
        Some(spans) => spans
          .iter()
          .map(|span| Span::styled(span.content.clone(), style.patch(span.style)))
          .collect(),
        None => vec![Span::styled(line.to_string(), style)],
      }
    })
    .collect()
}

/// A call as the transcript announces it: the tool's name, then the arguments
/// `summary` made of them, over as many lines as they were written on.
///
/// A command is code, and reads as code. So is what a tool the agent did not
/// bring is being asked for: nothing here knows what its arguments mean, so
/// they are shown as the JSON they arrived as. A script written over several
/// lines is shown over all of them, indented to where its first line starts so
/// it reads as the one block it is — and so a line arriving only ever adds to
/// what is on screen, rather than moving it.
fn call_lines(name: &str, summary: &str, style: Style) -> Vec<Line<'static>> {
  const MARK: &str = "⚙ ";
  let body = match name {
    // Nothing but the name, for a call whose arguments are folded away.
    _ if summary.is_empty() => vec![Vec::new()],
    "bash" => code_spans("bash", summary, style),
    name if !crate::tools::BUILT_IN.contains(&name) => code_spans("json", summary, style),
    _ => summary
      .split('\n')
      .map(|line| vec![Span::styled(line.to_string(), style)])
      .collect(),
  };
  let indent = " ".repeat(MARK.chars().count() + name.chars().count() + 1);
  body
    .into_iter()
    .enumerate()
    .map(|(i, code)| {
      let mut spans = match i {
        0 => vec![
          Span::styled(MARK, Style::default().fg(Color::Yellow)),
          Span::styled(name.to_string(), Style::default().fg(Color::Yellow).bold()),
          Span::raw(" "),
        ],
        _ => vec![Span::raw(indent.clone())],
      };
      spans.extend(code);
      Line::from(spans)
    })
    .collect()
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
///
/// What is drawn is the session's transcript rather than the history the model
/// is sent: a compaction leaves the turns it summarized on screen, and
/// reopening the session is no reason to lose them.
fn entries_from_history(session: &Session) -> Vec<Entry> {
  let history = session.transcript(session.leaf());
  let now = Instant::now();
  let mut entries = Vec::new();
  let results = Results::collect(&history);
  let mut answered: HashSet<usize> = HashSet::new();
  for message in &history {
    match message {
      Message::System { .. } => {}
      Message::User { content } => {
        // A prompt that attached images is one text part, then a note and an
        // image for each: the images belong to the prompt, and the notes are
        // what was said to the model about them rather than anything the
        // transcript has to repeat.
        let mut attached: Vec<Vec<u8>> = content
          .iter()
          .filter_map(|c| match c {
            UserContent::Image(image) => crate::images::source_bytes(&image.data),
            _ => None,
          })
          .collect();
        // Asked before the images are handed to the prompt, which empties
        // the list: what decides whether a later text part is a note is
        // that there were images, not that there still are.
        let attachments = !attached.is_empty();
        let mut prompt_seen = false;
        for c in content {
          match c {
            UserContent::Text(t) => {
              let summary = t
                .text
                .strip_prefix(SUMMARY_PREFIX)
                .and_then(|r| r.strip_suffix(SUMMARY_SUFFIX));
              if let Some(s) = summary {
                entries.push(Entry::Summary(s.to_string()));
                continue;
              }
              if prompt_seen && attachments {
                continue;
              }
              prompt_seen = true;
              entries.push(Entry::User {
                text: t.text.clone(),
                images: std::mem::take(&mut attached),
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
                edited: edited_path(&call.function.name, &call.function.arguments),
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
  // The whole path, checkpoints included: a compaction does not stop the
  // entries above it being the way the conversation came.
  let here: HashSet<&str> = session.lineage(session.leaf(), true).into_iter().collect();
  // The rows under `parent`, each with the indent it is drawn at. One child is
  // the conversation carrying on, and reads at the same level. More than one
  // is somewhere it went two ways, which is what the indent is for — and where
  // the path still in use is listed first.
  let kids = |parent: Option<&str>, depth: usize| {
    let mut kids = children.get(&parent).cloned().unwrap_or_default();
    let branching = kids.len() > 1;
    if branching {
      kids.sort_by_key(|node| !here.contains(node.id.as_str()));
    }
    let depth = depth + usize::from(branching);
    kids.into_iter().map(move |node| (node, depth))
  };
  // Depth first, each entry before what grew from it. Walked with a stack of
  // its own rather than by recursion: a session is one entry per message, and
  // a long one is deeper than the thread's stack.
  let mut stack: Vec<(&Node, usize)> = kids(None, 0).rev().collect();
  let mut out = Vec::new();
  while let Some((node, depth)) = stack.pop() {
    if let Some(point) = point(session, node, depth) {
      out.push(point);
    }
    // A step that is no place to stop still has children that are.
    stack.extend(kids(Some(&node.id), depth).rev());
  }
  out
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
    id: node.id.clone(),
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

/// A context window as a picker row says it: round, because the figures are
/// round and the column they are drawn in is narrow.
fn window_label(tokens: u64) -> String {
  match tokens {
    0..1_000 => tokens.to_string(),
    1_000..1_000_000 => format!("{}K", tokens / 1_000),
    _ => format!("{:.1}M", tokens as f64 / 1_000_000.0),
  }
}

/// What a session runs on, as the footer and the session file spell it: the
/// provider, and the model it is pointed at now.
fn model_label(cfg: &agent::Config) -> String {
  format!("{}/{}", cfg.provider.label(), cfg.model)
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
    // Whole, however many lines it runs to: what a call is about to do is the
    // part worth reading in full. The trailing newline a heredoc ends on is
    // not a line of it.
    "bash" => get("command").map(|c| c.trim_end().to_string()),
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
  partial_str(args, key).unwrap_or_default()
}

/// The file a call is writing into its arguments, as far as it has arrived —
/// which is what says how to colour the text arriving with it.
fn writing_path(args: &str) -> Option<String> {
  match serde_json::from_str::<serde_json::Value>(args) {
    Ok(value) => value.get("path").and_then(|p| p.as_str()).map(str::to_string),
    Err(_) => partial_str(args, "path"),
  }
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

  /// The rows ratatui itself paints the thumb on, for [`thumb_bounds`] to be
  /// held against.
  fn drawn_thumb(track: u16, max_scroll: usize, viewport: usize, offset: usize) -> (u16, u16) {
    use ratatui::buffer::Buffer;
    use ratatui::widgets::StatefulWidget;

    let area = Rect::new(0, 0, 1, track);
    let mut buf = Buffer::empty(area);
    let mut state = ScrollbarState::new(max_scroll)
      .position(offset)
      .viewport_content_length(viewport);
    Scrollbar::new(ScrollbarOrientation::VerticalRight)
      .begin_symbol(None)
      .end_symbol(None)
      .track_symbol(Some("│"))
      .thumb_symbol("┃")
      .render(area, &mut buf, &mut state);
    let rows: Vec<u16> = (0..track).filter(|&y| buf[(0, y)].symbol() == "┃").collect();
    (rows[0], rows.len() as u16)
  }

  #[test]
  fn the_thumb_is_looked_for_where_ratatui_draws_it() {
    for track in [3_u16, 10, 40] {
      for max_scroll in [1_usize, 5, 200, 5000] {
        for viewport in [track as usize, track as usize * 2] {
          for offset in [0, 1, max_scroll / 3, max_scroll - 1, max_scroll] {
            assert_eq!(
              thumb_bounds(track, max_scroll, viewport, offset),
              drawn_thumb(track, max_scroll, viewport, offset),
              "track {track}, max_scroll {max_scroll}, viewport {viewport}, offset {offset}"
            );
          }
        }
      }
    }
  }

  #[test]
  fn dragging_the_thumb_to_either_end_of_its_track_reaches_either_end_of_the_transcript() {
    let (max_scroll, viewport, track) = (500, 30, 30_u16);
    let (start, len) = thumb_bounds(track, max_scroll, viewport, 0);
    let thumb = Thumb {
      track: Rect::new(79, 0, 1, track),
      start,
      len,
    };
    assert_eq!(thumb_offset(-4, thumb, max_scroll), 0);
    assert_eq!(thumb_offset(i16::MAX.into(), thumb, max_scroll), max_scroll);
    // Halfway down the track is halfway down the transcript, and the thumb
    // ends up back under the cursor that put it there.
    let travel = (track - len) as isize;
    let middle = thumb_offset(travel / 2, thumb, max_scroll);
    assert_eq!(thumb_bounds(track, max_scroll, viewport, middle).0, travel as u16 / 2);
  }

  fn names(matches: &[Match]) -> Vec<&'static str> {
    matches
      .iter()
      .map(|m| match m {
        Match::Command { index, .. } => COMMANDS[*index].0,
        Match::Path { .. } => "path",
      })
      .collect()
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
  fn the_model_filter_is_fuzzy_and_keeps_a_family_together() {
    let mut matcher = Matcher::new(Config::DEFAULT);
    let models: Vec<ModelInfo> = ["gpt-5.2", "claude-sonnet-5", "claude-haiku-4-5"]
      .iter()
      .map(|id| ModelInfo {
        id: (*id).to_string(),
        name: None,
        context_length: None,
      })
      .collect();
    let mut ids = |query| {
      filter_models(&mut matcher, &models, query)
        .into_iter()
        .map(|(at, _)| models[at].id.as_str())
        .collect::<Vec<_>>()
    };
    // Nothing typed leaves the list as the provider gave it.
    assert_eq!(ids(""), ["gpt-5.2", "claude-sonnet-5", "claude-haiku-4-5"]);
    // Fuzzy: letters in order, not contiguous.
    assert_eq!(ids("snt"), ["claude-sonnet-5"]);
    // Everything one prefix matches, still in the order it was listed in.
    assert_eq!(ids("claude"), ["claude-sonnet-5", "claude-haiku-4-5"]);
    assert!(ids("zzz").is_empty());
    // And the letters it matched are picked out where they are: s-o-n-net.
    let matched = filter_models(&mut matcher, &models, "son");
    assert_eq!(matched[0].1, [7, 8, 9]);
    // Nothing typed, nothing picked out.
    assert!(filter_models(&mut matcher, &models, "")[0].1.is_empty());
  }

  #[test]
  fn the_session_filter_matches_the_titles_the_rows_are_drawn_as() {
    let mut matcher = Matcher::new(Config::DEFAULT);
    let session = |name: Option<&str>, first: &str| SessionInfo {
      path: PathBuf::from("/tmp/s.jsonl"),
      id: "s".into(),
      name: name.map(str::to_string),
      cwd: "/home/u/work".into(),
      modified: chrono::Local::now(),
      message_count: 2,
      first_message: first.into(),
    };
    let sessions = vec![
      session(None, "fix the scrollbar"),
      session(Some("release notes"), "unrelated first message"),
      session(None, "add a filter to the picker"),
    ];
    let mut titles = |query| {
      filter_sessions(&mut matcher, &sessions, query)
        .into_iter()
        .map(|(at, _)| sessions[at].title())
        .collect::<Vec<_>>()
    };
    // Nothing typed leaves the list as the store listed it, newest first.
    assert_eq!(
      titles(""),
      ["fix the scrollbar", "release notes", "add a filter to the picker"]
    );
    // A name stands in for the first message, and is what gets matched.
    assert_eq!(titles("release"), ["release notes"]);
    // Fuzzy, over the first message where there is no name.
    assert_eq!(titles("scrlbr"), ["fix the scrollbar"]);
    assert!(titles("zzz").is_empty());
    // And the letters it matched are picked out where they are: f-i-lter.
    let matched = filter_sessions(&mut matcher, &sessions, "fil");
    assert_eq!(matched[0].1, [6, 7, 8]);
  }

  #[test]
  fn the_point_filter_matches_the_rows_the_tree_is_drawn_as() {
    let mut matcher = Matcher::new(Config::DEFAULT);
    let point = |label: &str, depth: usize| Point {
      id: "n".into(),
      leaf: Some("n".into()),
      text: None,
      label: label.into(),
      depth,
      len: 2,
      here: false,
    };
    let points = vec![
      point("❯ what does main.rs do?", 0),
      point("⚙ read", 0),
      point("It prints hi.", 1),
      point("Actually it prints hi and exits 0.", 1),
    ];
    let mut labels = |query| {
      filter_points(&mut matcher, &points, query)
        .into_iter()
        .map(|(at, _)| points[at].label.as_str())
        .collect::<Vec<_>>()
    };
    // Nothing typed leaves the list as the walk built it, so a branch still
    // hangs under the point it parts at.
    assert_eq!(
      labels(""),
      [
        "❯ what does main.rs do?",
        "⚙ read",
        "It prints hi.",
        "Actually it prints hi and exits 0."
      ]
    );
    // A tool row is found by the tool it ran, and a turn by what it said.
    assert_eq!(labels("read"), ["⚙ read"]);
    assert_eq!(labels("exits"), ["Actually it prints hi and exits 0."]);
    assert!(labels("zzz").is_empty());
    // The indent counts as part of the row, so the letters picked out land on
    // the letters the row draws: two spaces, then I-t, and p a space later.
    let matched = filter_points(&mut matcher, &points, "itp");
    assert_eq!(matched[0].1, [2, 3, 5]);
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
    let Match::Command { highlights, .. } = &filter_commands(&mut matcher, "nm")[0] else {
      panic!("a command")
    };
    assert_eq!(highlights, &[0, 2]);
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
    let prompt = session.lineage(session.leaf(), false)[0].to_string();
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
    // Every line of it, exactly as the finished summary shows the same
    // command.
    assert_eq!(quote(r#"{"command":"one\ntwo"#), "one\ntwo");
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
    // A command is drawn on the call's own lines; the block under them is for
    // the tools that carry a file in their arguments.
    assert_eq!(writing_body("bash", r#"{"command":"ls -la"#), []);
  }

  /// A run of spans as the text it draws.
  fn text_of(spans: &[Span<'static>]) -> String {
    spans.iter().map(|span| span.content.as_ref()).collect()
  }

  /// Drawn lines as their text, with the styling dropped.
  fn drawn(lines: &[Line<'static>]) -> Vec<String> {
    lines
      .iter()
      .map(|line| line.spans.iter().map(|span| span.content.as_ref()).collect())
      .collect()
  }

  #[test]
  fn a_script_is_announced_over_every_line_it_was_written_on() {
    // A multi-line command is what the call will do, so all of it is on
    // screen — later lines indented to where the first one starts.
    let summary = summarize_args(
      "bash",
      &serde_json::json!({ "command": "for f in *; do\n  wc -l $f\ndone\n" }),
    );
    assert_eq!(summary, "for f in *; do\n  wc -l $f\ndone");
    assert_eq!(
      drawn(&call_lines("bash", &summary, Style::default())),
      ["⚙ bash for f in *; do", "         wc -l $f", "       done"]
    );
    // And a command of one line is still the one line it was.
    assert_eq!(
      drawn(&call_lines("bash", "ls -la", Style::default())),
      ["⚙ bash ls -la"]
    );
  }

  #[test]
  fn a_call_with_no_arguments_to_show_is_still_a_line() {
    assert_eq!(drawn(&call_lines("mystery", "", Style::default())), ["⚙ mystery "]);
  }

  #[test]
  fn a_line_too_long_for_the_width_is_cut_to_it() {
    let plain = |line: &str| vec![Span::raw(line.to_string())];
    // What fits is left alone, ellipsis and all.
    assert_eq!(text(&[clip(plain("short"), 10)]), ["short"]);
    // What does not is cut to the width, with the ellipsis inside it rather
    // than one column past it.
    assert_eq!(text(&[clip(plain("0123456789abc"), 10)]), ["012345678…"]);
    // Columns, not characters: a wide one takes two of them.
    assert_eq!(text(&[clip(plain("ありがとう"), 5)]), ["あり…"]);
    // Narrower than the ellipsis itself, there is nothing to say.
    assert_eq!(clip(plain("anything"), 0), []);
    // A clipped line is the same line, shorter: what is kept keeps its colour.
    let clipped = clip(
      vec![
        Span::styled("keep", Style::default().fg(Color::Green)),
        Span::styled("cut", Style::default().fg(Color::Red)),
      ],
      6,
    );
    assert_eq!(text(std::slice::from_ref(&clipped)), ["keepc…"]);
    assert_eq!(clipped[0].style.fg, Some(Color::Green));
    assert_eq!(clipped[1].style.fg, Some(Color::Red));
  }

  #[test]
  fn only_a_folded_block_cuts_its_lines() {
    let long = "x".repeat(40);
    let block = |fold| {
      let mut out = Vec::new();
      Preview {
        body: vec![vec![Span::raw(long.clone())]],
        gutter: gutter(false, false),
        cap: TOOL_OUTPUT_LINES,
        fold,
        width: 20,
        from_end: false,
        cursor: false,
      }
      .draw(&mut out);
      out
    };
    // Folded, the line is a row: cut to the width the transcript has, indent
    // and gutter included.
    let folded = drawn(&block(Fold::Preview));
    assert_eq!(folded, ["   ".to_string() + &"x".repeat(16) + "…"]);
    // Unfolded, it is whole, over as many rows as the wrap takes.
    assert_eq!(folded[0].chars().count(), 20);
    assert_eq!(drawn(&block(Fold::Full)), [format!("   {long}")]);
    // Collapsed, nothing at all.
    assert!(block(Fold::Collapsed).is_empty());
  }

  #[test]
  fn a_selection_covers_the_cells_the_drag_went_over() {
    let selection = Selection {
      anchor: (2, 4),
      head: (4, 1),
    };
    // Nothing on the lines it does not reach.
    assert_eq!(selection.columns(1), None);
    assert_eq!(selection.columns(5), None);
    // From where it started on the first line, all of the lines between, and
    // up to and including the cell it ended on.
    assert_eq!(selection.columns(2), Some((4, usize::MAX)));
    assert_eq!(selection.columns(3), Some((0, usize::MAX)));
    assert_eq!(selection.columns(4), Some((0, 2)));
    // Dragged the other way it covers the same cells.
    let backwards = Selection {
      anchor: selection.head,
      head: selection.anchor,
    };
    for at in 1..=5 {
      assert_eq!(backwards.columns(at), selection.columns(at));
    }
    // A press that went nowhere is a click, and a click selects one cell —
    // which is what makes it worth telling apart from a selection.
    let click = Selection {
      anchor: (2, 4),
      head: (2, 4),
    };
    assert!(click.is_empty());
    assert_eq!(click.columns(2), Some((4, 5)));
  }

  #[test]
  fn a_line_is_cut_where_the_selection_starts_and_ends() {
    let line = Line::from(vec![
      Span::styled("❯ ", Style::default().fg(Color::Cyan)),
      Span::styled("hello world", Style::default().bold()),
    ]);
    // Columns, not characters or spans: the cut falls inside the span it
    // falls inside, and each piece keeps the styling it had.
    let (before, inside, after) = cut(&line, 2, 7);
    assert_eq!(text_of(&before), "❯ ");
    assert_eq!(text_of(&inside), "hello");
    assert_eq!(text_of(&after), " world");
    assert_eq!(inside[0].style, Style::default().bold());
    // A selection that starts past the end of the line takes nothing from it.
    assert_eq!(cut(&line, 40, usize::MAX).1, []);
    // The blanks a line ends in are the margin, not text that was selected.
    let padded = Line::raw("word     ");
    assert_eq!(selected(&padded, 0, usize::MAX), "word");
  }

  #[test]
  fn a_wide_character_is_selected_by_either_of_its_columns() {
    let line = Line::raw("日本語");
    // The cell a wide character starts on is the one it belongs to, so a
    // selection ending on its second column still holds all of it.
    assert_eq!(selected(&line, 0, 2), "日");
    assert_eq!(selected(&line, 0, 3), "日本");
    assert_eq!(selected(&line, 2, 4), "本");
  }

  #[test]
  fn what_is_selected_is_drawn_reversed_and_nothing_else_is() {
    let line = Line::from(vec![Span::raw("ab"), Span::styled("cd", Style::default().bold())]);
    let shown = highlight(&line, 1, 3);
    // The same line, cell for cell — only how three of its cells are drawn
    // has changed.
    assert_eq!(text_of(&shown.spans), "abcd");
    let reversed: String = shown
      .spans
      .iter()
      .filter(|span| span.style.add_modifier.contains(Modifier::REVERSED))
      .map(|span| span.content.as_ref())
      .collect();
    assert_eq!(reversed, "bc");
    // And a cell that was bold before is bold and selected now, not one or
    // the other.
    let c = shown.spans.iter().find(|span| span.content == "c").unwrap();
    assert!(c.style.add_modifier.contains(Modifier::BOLD | Modifier::REVERSED));
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
        Entry::User { text, images } => match images.len() {
          0 => format!("user {text}"),
          n => format!("user {text} +{n} image"),
        },
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
      edited: None,
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
        edited: None,
        started: Instant::now(),
      },
      announced("ls", "c2"),
    ];
    assert_eq!(wrote_by(&entries, "c1"), Some(("a.rs", "fn main() {}")));
    assert_eq!(wrote_by(&entries, "c2"), None, "a command wrote no file");
  }

  #[test]
  fn only_an_edit_carries_the_file_it_changed() {
    let args = serde_json::json!({ "path": "a.rs", "edits": [] });
    assert_eq!(edited_path("edit", &args).as_deref(), Some("a.rs"));
    assert!(edited_path("write", &args).is_none(), "a write is shown as the file");

    let entries = vec![
      Entry::ToolCall {
        name: "edit".into(),
        summary: "a.rs (1 edit)".into(),
        call: "c1".into(),
        wrote: None,
        edited: Some("a.rs".into()),
        started: Instant::now(),
      },
      announced("ls", "c2"),
    ];
    assert_eq!(edited_by(&entries, "c1"), Some("a.rs"));
    assert_eq!(edited_by(&entries, "c2"), None, "a command changed no file");
  }

  #[test]
  fn a_diffs_numbering_ends_where_its_code_begins() {
    // The width comes from the file's length, and every line is written to
    // it: mark, number, space.
    assert_eq!(code_column("    ...\n  4 l4\n- 6 l6"), Some(4));
    // Code that is a number of its own would put the column further right,
    // which is why the narrowest line is the one believed.
    assert_eq!(code_column("+1 42\n 2 x"), Some(3));
    assert_eq!(code_column("nothing numbered here"), None);
  }

  #[test]
  #[cfg(feature = "lang-rust")]
  fn a_diff_is_highlighted_as_the_file_it_changed() {
    let diff = " 1 fn main() {\n-2     let x = 1;\n+2     let y = \"hi\";\n 3 }";
    let lines = diff_lines(Some("src/main.rs"), diff);
    let spans = |i: usize| -> Vec<(String, Option<Color>, bool)> {
      lines[i]
        .iter()
        .map(|span| {
          (
            span.content.to_string(),
            span.style.fg,
            span.style.add_modifier.contains(Modifier::DIM),
          )
        })
        .collect()
    };

    // The mark and its number still say what became of the line.
    assert_eq!(spans(2)[0], (" +2 ".to_string(), Some(Color::Green), false));
    assert_eq!(spans(1)[0], (" -2 ".to_string(), Some(Color::Red), false));
    // The code after them is the file's, in the file's colours.
    assert!(
      spans(2).contains(&("let".to_string(), Some(Color::Magenta), false)),
      "{:?}",
      spans(2)
    );
    assert!(
      spans(2).contains(&("\"hi\"".to_string(), Some(Color::Green), false)),
      "a string on the line that was added: {:?}",
      spans(2)
    );
    // The line that was taken out is highlighted from the file as it was,
    // which is the only place that line still exists.
    assert!(
      spans(1).contains(&("1".to_string(), Some(Color::Cyan), false)),
      "{:?}",
      spans(1)
    );
    // Context keeps its colours but stays out of the way.
    assert!(
      spans(0).contains(&("fn".to_string(), Some(Color::Magenta), true)),
      "{:?}",
      spans(0)
    );
    // Nothing is lost on the way: every line reads as it was written.
    let text: Vec<String> = lines
      .iter()
      .map(|line| line.iter().map(|span| span.content.as_ref()).collect())
      .collect();
    assert_eq!(text, diff.lines().map(|line| format!(" {line}")).collect::<Vec<_>>());
  }

  #[test]
  #[cfg(feature = "lang-rust")]
  fn an_edit_is_coloured_the_same_while_it_is_still_being_written() {
    // The half arriving is drawn in the file's colours, so nothing recolours
    // under the reader when the call lands and becomes a diff.
    let lines = marked_code("+ ", mark_style(Some('+')), "rs", "    let x = 1;\n}");
    let spans: Vec<(String, Option<Color>)> = lines[0]
      .iter()
      .map(|span| (span.content.to_string(), span.style.fg))
      .collect();
    assert_eq!(spans[0], (" + ".to_string(), Some(Color::Green)));
    assert!(spans.contains(&("let".to_string(), Some(Color::Magenta))), "{spans:?}");
    // A file the call has not named yet, or one with no grammar: the text as
    // it is, under its mark.
    let plain = painted(&marked_code("- ", mark_style(Some('-')), "", "let x = 1;"));
    assert_eq!(plain, [(" - let x = 1;".to_string(), vec![Some(Color::Red); 2])]);
  }

  #[test]
  fn a_path_is_read_from_arguments_that_are_still_arriving() {
    assert_eq!(
      writing_path(r#"{"path":"src/main.rs","edits":[]}"#).as_deref(),
      Some("src/main.rs")
    );
    assert_eq!(
      writing_path(r#"{"path":"src/main.rs","edits":[{"oldText":"a"#).as_deref(),
      Some("src/main.rs"),
      "half an edit still says which file it is in"
    );
    assert_eq!(writing_path(r#"{"pat"#), None);
  }

  #[test]
  fn a_diff_of_a_file_we_cannot_read_is_marked_up_as_before() {
    let diff = " 1 hello\n-2 there\n+2 world";
    let plain = painted(&marked_lines(diff, true));
    assert_eq!(painted(&diff_lines(Some("notes.txt"), diff)), plain);
    assert_eq!(painted(&diff_lines(None, diff)), plain, "a call that named no file");
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
    let answered = session.lineage(session.leaf(), false)[1].to_string();
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

  // ------------------------------------------------- attachments

  /// A directory with one small PNG in it, and its bytes.
  fn with_png(name: &str) -> (PathBuf, Vec<u8>) {
    let dir = std::env::temp_dir().join(format!("fa-ui-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let image = image::ImageBuffer::from_pixel(6, 4, image::Rgb([9u8, 200, 60]));
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
      .write_to(&mut bytes, image::ImageFormat::Png)
      .unwrap();
    let bytes = bytes.into_inner();
    std::fs::write(dir.join("shot.png"), &bytes).unwrap();
    (dir, bytes)
  }

  /// The bottom border's summary as it reads, spans joined.
  fn strip(text: &str, dir: &Path, vision: bool, width: u16) -> Option<String> {
    attachment_strip(&attach::tokens(text, dir), vision, width)
      .map(|line| line.spans.iter().map(|s| s.content.to_string()).collect())
  }

  #[test]
  fn the_border_says_what_a_token_found() {
    let (dir, _) = with_png("strip");
    assert_eq!(
      strip("why does @shot.png do that?", &dir, true, 60).as_deref(),
      Some("─ ▣ shot.png 6×4 ")
    );
  }

  #[test]
  fn the_border_says_when_a_token_found_nothing() {
    let (dir, _) = with_png("missing");
    assert_eq!(
      strip("look at @gone.png", &dir, true, 60).as_deref(),
      Some("─ ⚠ gone.png not found ")
    );
    std::fs::write(dir.join("notes.txt"), "words").unwrap();
    assert_eq!(
      strip("read @notes.txt", &dir, true, 60).as_deref(),
      Some("─ ⚠ notes.txt not an image ")
    );
    // And nothing at all to say when there is no token.
    assert!(strip("look at nothing", &dir, true, 60).is_none());
  }

  #[test]
  fn the_border_drops_what_will_not_fit_and_counts_it() {
    let (dir, _) = with_png("narrow");
    let narrow = strip("@shot.png @shot.png @shot.png", &dir, true, 30).expect("a strip");
    assert!(narrow.contains("+2"), "the rest are counted: {narrow:?}");
    assert!(narrow.chars().count() <= 30, "it fits the border: {narrow:?}");
  }

  #[test]
  fn a_prompt_carries_its_image_and_the_transcript_draws_the_same_bytes() {
    let (dir, bytes) = with_png("prompt");
    let (prompt, notes) = attach_images("what is @shot.png", &dir, true);
    assert_eq!(prompt.images.len(), 1);
    assert_eq!(prompt.preview(), vec![bytes]);
    assert!(notes.is_empty(), "it resolved, so there is nothing to report");
  }

  #[test]
  fn a_token_that_resolved_to_nothing_is_reported_rather_than_passed_over() {
    let (dir, _) = with_png("report");
    let (prompt, notes) = attach_images("look at @gone.png", &dir, true);
    assert!(prompt.images.is_empty());
    // The text still goes as typed; only the attachment is missing.
    assert_eq!(prompt.text, "look at @gone.png");
    assert_eq!(notes, ["Not attached — @gone.png: no such file."]);
  }

  #[test]
  fn a_model_without_vision_is_told_rather_than_sent_the_image() {
    let (dir, _) = with_png("novision");
    let (prompt, notes) = attach_images("what is @shot.png", &dir, false);
    assert!(prompt.images.is_empty());
    assert_eq!(notes.len(), 1);
    assert!(notes[0].contains("--no-vision"), "{notes:?}");
  }

  #[test]
  fn a_slash_command_never_attaches_anything() {
    let (dir, _) = with_png("command");
    let (prompt, notes) = attach_images("/name @shot.png", &dir, true);
    assert!(prompt.images.is_empty());
    assert!(notes.is_empty());
  }

  #[test]
  fn a_pasted_image_path_is_written_down_as_a_token() {
    let (dir, _) = with_png("paste");
    // A full path under the working directory comes back relative to it.
    let dropped = dir.join("shot.png").display().to_string();
    assert_eq!(as_token(&dropped, &dir).as_deref(), Some("@shot.png "));
    // And stays a full path when it is not under the working directory.
    let elsewhere = dir.parent().unwrap().join("fa-ui-paste-elsewhere");
    assert_eq!(
      as_token(&dropped, &elsewhere).as_deref(),
      Some(format!("@{dropped} ").as_str())
    );
    // Anything that is not a lone path to an image is left alone.
    assert!(as_token("shot.png is the one", &dir).is_none());
    assert!(as_token(&dir.join("gone.png").display().to_string(), &dir).is_none());
  }

  #[test]
  fn a_resumed_session_draws_the_image_its_prompt_attached() {
    let (dir, bytes) = with_png("resume");
    let (prompt, _) = attach_images("what is @shot.png", &dir, true);
    let session = session_of(vec![prompt.message()]);
    let entries = entries_from_history(&session);
    // One prompt, carrying its image — and not a second entry for the note
    // that labels it, which was written for the model rather than the screen.
    assert_eq!(shapes(&entries), ["user what is @shot.png +1 image"]);
    let Entry::User { images, .. } = &entries[0] else {
      panic!("a prompt")
    };
    assert_eq!(images, &[bytes]);
  }
}
