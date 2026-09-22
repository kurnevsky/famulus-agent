//! The questionnaire behind the `ask` tool: what the model may put to the
//! user, the dialog that puts it, and the answer that goes back.
//!
//! The tool itself lives in `tools`, because that is where tools live. What is
//! here is everything about a question that does not need a model — its shape,
//! the state a half-answered questionnaire is in, and how it draws — so all of
//! it can be tested without one.
//!
//! The behaviour is `rpiv-ask-user-question`'s, the pi extension of the same
//! name: up to four questions in one dialog, two to four written-out options
//! each, a free-text row appended to every one of them, and a submit tab that
//! reviews the answers before they go back. What the model reads at the end is
//! that extension's envelope, word for word, so a prompt written for one reads
//! the same coming out of the other.

use std::collections::{BTreeMap, HashMap, HashSet};

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui_textarea::{CursorMove, DataCursor, TextArea};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::markdown::{wrap_line, wrap_text};

/// Questions one call may ask. Four is an interruption; more is an interview.
pub const MAX_QUESTIONS: usize = 4;
/// A question with one option is not a question.
pub const MIN_OPTIONS: usize = 2;
const MAX_OPTIONS: usize = 4;
const MAX_HEADER: u32 = 16;
const MAX_LABEL: u32 = 60;

/// The free-text row appended under every question's options.
const OTHER_LABEL: &str = "Type something.";
/// The row a multi-select question is committed from, since `Enter` on the
/// options themselves ticks boxes.
const NEXT_LABEL: &str = "Next";

/// Labels the model may not write, because the dialog writes them itself.
///
/// `Other` is not one of the dialog's own rows and is refused anyway: models
/// are trained on interfaces that offer it, and one that reaches for it would
/// be authoring a second free-text row beside the real one.
pub const RESERVED_LABELS: [&str; 3] = ["Other", OTHER_LABEL, NEXT_LABEL];

/// What an answer reads as when there is nothing in it — a confirmed empty
/// free-text row, or a multi-select question committed with no box ticked.
const NO_INPUT: &str = "(no input)";
const DECLINED: &str = "User declined to answer questions";
const ENVELOPE_PREFIX: &str = "User has answered your questions:";
const ENVELOPE_SUFFIX: &str = "You can now continue with the user's answers in mind.";

const POINTER: &str = "› ";
const NO_POINTER: &str = "  ";
const CHECKED: &str = "[✔]";
const UNCHECKED: &str = "[ ]";
/// The cursor of the free-text row when there is no character under it to
/// reverse — at the end of a line, or on a space.
const CURSOR: &str = "█";

// ---------------------------------------------------------------- questions

/// One option of one question, as the model wrote it.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
pub struct Choice {
  /// MAX 60 CHARACTERS. The display text for this option that the user will
  /// see and select. Should be concise (1-5 words) and clearly describe the
  /// choice.
  #[schemars(length(max = MAX_LABEL))]
  pub label: String,
  /// Explanation of what this option means or what will happen if chosen.
  /// Useful for providing context about trade-offs or implications.
  pub description: String,
}

/// One question of a questionnaire.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
pub struct Question {
  /// The complete question to ask the user. Should be clear, specific, and
  /// end with a question mark. Example: "Which library should we use for date
  /// formatting?" If multiSelect is true, phrase it accordingly, e.g. "Which
  /// features do you want to enable?"
  pub question: String,
  /// MAX 16 CHARACTERS. Very short chip/tag shown next to the question.
  /// Examples: "Auth method", "Library", "Approach".
  #[schemars(length(max = MAX_HEADER))]
  pub header: String,
  /// The available choices for this question. Must have 2-4 options. Each
  /// option should be a distinct, mutually exclusive choice (unless
  /// multiSelect is enabled). The "Type something." row is appended
  /// automatically — do NOT write it.
  #[schemars(length(min = MIN_OPTIONS as u32, max = MAX_OPTIONS as u32))]
  pub options: Vec<Choice>,
  /// Set to true to allow the user to select multiple options instead of just
  /// one. Use when choices are not mutually exclusive.
  #[serde(default, rename = "multiSelect")]
  pub multi_select: bool,
}

impl Question {
  /// How the question is named where there is no room for it — the tab strip,
  /// and the review list. Numbered when the model left the header out.
  fn chip(&self, index: usize) -> String {
    match self.header.trim().is_empty() {
      true => format!("Q{}", index + 1),
      false => self.header.clone(),
    }
  }
}

/// Fold the line terminators a model sometimes leaves inside a string it is
/// streaming.
///
/// A bare `\r` is a cursor-control byte rather than text: drawn as written it
/// returns the terminal to column zero and the rest of the row overwrites what
/// was already there. `\r\n` is a genuine line break and stays one; a lone
/// `\r` is dropped rather than turned into a space, which would leave a gap in
/// the middle of a word.
fn normalize(text: &str) -> String {
  text.replace("\r\n", "\n").replace('\r', "")
}

/// The questions as they will be shown, with the line terminators folded.
///
/// Before validation rather than after it, so the reserved-label and
/// duplicate-label checks compare what the user will actually read: `Other\r`
/// is `Other`, and would otherwise walk straight past a check on `Other`.
pub fn prepare(questions: Vec<Question>) -> Vec<Question> {
  questions
    .into_iter()
    .map(|q| Question {
      question: normalize(&q.question),
      header: normalize(&q.header),
      options: q
        .options
        .into_iter()
        .map(|o| Choice {
          label: normalize(&o.label),
          description: normalize(&o.description),
        })
        .collect(),
      multi_select: q.multi_select,
    })
    .collect()
}

/// Whether the questionnaire can be put to the user at all, with the sentence
/// the model gets back when it cannot.
///
/// Written for the model to read and act on, since it is the only one who can
/// fix any of it by asking again.
pub fn validate(questions: &[Question]) -> Result<(), String> {
  if questions.is_empty() {
    return Err("Error: At least one question is required".into());
  }
  if questions.len() > MAX_QUESTIONS {
    return Err(format!(
      "Error: At most {MAX_QUESTIONS} questions are allowed per invocation"
    ));
  }
  let mut asked: HashSet<&str> = HashSet::new();
  for q in questions {
    if !asked.insert(q.question.as_str()) {
      return Err("Error: Question text must be unique within an invocation".into());
    }
  }
  for q in questions {
    if q.options.len() < MIN_OPTIONS {
      return Err(format!("Error: Each question requires at least {MIN_OPTIONS} options"));
    }
    let mut labels: HashSet<&str> = HashSet::new();
    for o in &q.options {
      // Before the duplicate check: a question whose options are two copies of
      // `Next` is wrong for the reason that names the rule it broke.
      if RESERVED_LABELS.contains(&o.label.as_str()) {
        return Err(format!(
          "Error: Option label is reserved ({})",
          RESERVED_LABELS.join(", ")
        ));
      }
      if !labels.insert(o.label.as_str()) {
        return Err("Error: Option labels must be unique within a question".into());
      }
    }
  }
  Ok(())
}

// ---------------------------------------------------------------- answers

/// What the user said to one question.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answer {
  /// One of the options, by its label.
  Chose(String),
  /// What they typed instead; empty when they confirmed an empty row.
  Typed(String),
  /// The boxes ticked on a multi-select question, in the order asked.
  Ticked(Vec<String>),
}

impl Answer {
  /// The answer as one line, which is how both the review list and the model
  /// read it.
  fn scalar(&self) -> String {
    match self {
      Answer::Chose(label) => label.clone(),
      Answer::Typed(text) if text.is_empty() => NO_INPUT.to_string(),
      Answer::Typed(text) => text.clone(),
      Answer::Ticked(labels) if labels.is_empty() => NO_INPUT.to_string(),
      Answer::Ticked(labels) => labels.join(", "),
    }
  }
}

/// What the dialog came back with: the questions that were answered, in the
/// order they were asked. Empty when the user walked away — the questionnaire
/// was dismissed, or there was no terminal to show it in — however much of it
/// they had answered by then.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
  pub answers: Vec<(usize, Answer)>,
}

impl Outcome {
  /// What the model reads.
  ///
  /// A cancelled questionnaire and an empty one say the same sentence: the
  /// model has one signal for "they did not answer" rather than two to tell
  /// apart. Answering some of the questions and not the rest is allowed, and
  /// the ones left blank simply say nothing here.
  pub fn response(&self, questions: &[Question]) -> String {
    if self.answers.is_empty() {
      return DECLINED.to_string();
    }
    let segments: Vec<String> = self
      .answers
      .iter()
      .filter_map(|(index, answer)| {
        let question = questions.get(*index)?;
        Some(format!("\"{}\"=\"{}\".", question.question, answer.scalar()))
      })
      .collect();
    format!("{ENVELOPE_PREFIX} {} {ENVELOPE_SUFFIX}", segments.join(" "))
  }
}

// ---------------------------------------------------------------- the draft

/// What the user is typing on a question's free-text row.
///
/// The row is in the middle of a list the arrow keys are already walking, so
/// the keys are routed by hand rather than through the text area's own
/// bindings: `Up` at the top of the draft is not a cursor move, it is the
/// option above.
type Draft = TextArea<'static>;

fn draft_text(draft: &Draft) -> String {
  draft.lines().join("\n")
}

/// The draft's lines with the cursor drawn over the character it is on, so
/// moving it shifts nothing, before they are wrapped to `width`.
fn draft_shown(draft: &Draft, width: u16, style: Style) -> Vec<Line<'static>> {
  let DataCursor(row, column) = draft.cursor();
  let lines = draft.lines().iter().enumerate().map(|(index, line)| {
    if index != row {
      return Line::styled(line.clone(), style);
    }
    let at = line.char_indices().nth(column).map_or(line.len(), |(at, _)| at);
    let (before, rest) = line.split_at(at);
    let mut after = rest.chars();
    let cursor = match after.next() {
      Some(c) if !c.is_whitespace() => Span::styled(c.to_string(), style.add_modifier(Modifier::REVERSED)),
      _ => Span::styled(CURSOR, style),
    };
    Line::from(vec![
      Span::styled(before.to_string(), style),
      cursor,
      Span::styled(after.as_str().to_string(), style),
    ])
  });
  lines.flat_map(|line| wrap_line(line, width.max(1))).collect()
}

// ---------------------------------------------------------------- the dialog

/// What a row of the list says.
enum Label<'a> {
  Text(String),
  /// The free-text row while it has the keyboard, drawn with its cursor.
  Draft(&'a Draft),
}

/// A row of the list under a question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Row {
  Choice(usize),
  /// The free-text row.
  Other,
  /// The row that commits a multi-select question.
  Next,
}

/// A questionnaire being answered.
///
/// One tab per question, plus a submit tab when there is more than one — a
/// single question is answered by answering it, with nothing to review.
pub struct Dialog {
  questions: Vec<Question>,
  /// The tab shown. `questions.len()` is the submit tab.
  tab: usize,
  /// The row of that tab the cursor is on.
  row: usize,
  /// The free-text row has the keyboard, so printable keys are text rather
  /// than commands.
  typing: bool,
  answers: BTreeMap<usize, Answer>,
  /// What was typed on each question's free-text row, kept while the user
  /// looks at the other options and at the other questions.
  drafts: HashMap<usize, Draft>,
  /// Which of submit and cancel the submit tab is on.
  submit: usize,
}

impl Dialog {
  pub fn new(questions: Vec<Question>) -> Self {
    Self {
      questions,
      tab: 0,
      row: 0,
      typing: false,
      answers: BTreeMap::new(),
      drafts: HashMap::new(),
      submit: 0,
    }
  }

  /// More than one question, which is what brings the tab strip and the submit
  /// tab with it.
  fn tabbed(&self) -> bool {
    self.questions.len() > 1
  }

  fn on_submit_tab(&self) -> bool {
    self.tab >= self.questions.len()
  }

  fn rows(&self) -> Vec<Row> {
    let Some(question) = self.questions.get(self.tab) else {
      return Vec::new();
    };
    let mut rows: Vec<Row> = (0..question.options.len()).map(Row::Choice).collect();
    rows.push(Row::Other);
    if question.multi_select {
      rows.push(Row::Next);
    }
    rows
  }

  fn row_at(&self, index: usize) -> Option<Row> {
    self.rows().get(index).copied()
  }

  fn draft(&mut self) -> &mut Draft {
    self.drafts.entry(self.tab).or_default()
  }

  /// The tab a confirmed answer moves on to: the next question, then the
  /// submit tab. `None` finishes the questionnaire, which is what a single
  /// question does, having nothing to move on to.
  fn next_tab(&self) -> Option<usize> {
    match self.tabbed() {
      false => None,
      true => Some((self.tab + 1).min(self.questions.len())),
    }
  }

  fn go_to(&mut self, tab: usize) {
    self.tab = tab;
    self.row = 0;
    self.typing = false;
    self.submit = 0;
  }

  /// Whether the box on option `index` of the tab shown is ticked, which is
  /// what its answer says: ticking a box is answering the question.
  fn is_ticked(&self, index: usize) -> bool {
    let Some(Answer::Ticked(labels)) = self.answers.get(&self.tab) else {
      return false;
    };
    let option = &self.questions[self.tab].options[index];
    labels.contains(&option.label)
  }

  /// The labels of the ticked boxes, in the order the question asked them.
  fn ticked_labels(&self) -> Vec<String> {
    match self.answers.get(&self.tab) {
      Some(Answer::Ticked(labels)) => labels.clone(),
      _ => Vec::new(),
    }
  }

  fn answered(&self) -> Outcome {
    Outcome {
      answers: self.answers.clone().into_iter().collect(),
    }
  }

  /// Record an answer and move on — to the next question, or out of the
  /// dialog when this was the last thing left to answer.
  fn confirm(&mut self, answer: Answer) -> Option<Outcome> {
    // A typed answer and a set of ticked boxes are two answers to one
    // question, so the boxes go out when the typing comes in: the one answer
    // replaces the other.
    self.answers.insert(self.tab, answer);
    match self.next_tab() {
      Some(tab) => {
        self.go_to(tab);
        None
      }
      None => Some(self.answered()),
    }
  }

  /// Tick or untick the box on the focused row, and keep the answer in step
  /// with it, so the tab strip says the question is answered as soon as one
  /// box is.
  fn toggle(&mut self, index: usize) {
    let labels: Vec<String> = (0..self.questions[self.tab].options.len())
      .filter(|&at| self.is_ticked(at) != (at == index))
      .map(|at| self.questions[self.tab].options[at].label.clone())
      .collect();
    match labels.is_empty() {
      true => {
        self.answers.remove(&self.tab);
      }
      false => {
        self.answers.insert(self.tab, Answer::Ticked(labels));
      }
    }
  }

  /// Text the user pasted. It is only ever text, so it goes to the free-text
  /// row and nowhere else — on a list of choices there is nothing for it to
  /// mean, and the Enter inside it never picks one.
  pub fn paste(&mut self, text: &str) {
    if self.typing {
      _ = self.draft().insert_str(text);
    }
  }

  /// A key the user pressed. `Some` is the questionnaire's answer, and the end
  /// of the dialog.
  pub fn key(&mut self, key: KeyEvent) -> Option<Outcome> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    // Esc leaves from everywhere, including mid-word in the free-text row:
    // the row is a way of answering the question, not a place to be stuck in.
    if key.code == KeyCode::Esc {
      return Some(Outcome::default());
    }
    if self.typing {
      return self.typing_key(key, ctrl);
    }
    if self.on_submit_tab() {
      return self.submit_key(key);
    }
    if let Some(action) = self.tab_key(key) {
      self.go_to(action);
      return None;
    }
    match key.code {
      KeyCode::Up => self.move_by(-1),
      KeyCode::Down => self.move_by(1),
      KeyCode::Char(' ') if self.multi() => {
        // Not on `Next`, which is a command and not a choice, and not on the
        // free-text row, where a space is the space the user typed.
        if let Some(Row::Choice(index)) = self.row_at(self.row) {
          self.toggle(index);
        }
      }
      KeyCode::Enter => return self.enter(),
      _ => {}
    }
    None
  }

  fn multi(&self) -> bool {
    self.questions.get(self.tab).is_some_and(|q| q.multi_select)
  }

  /// Which tab a key moves to, when it moves to one at all. Only a
  /// questionnaire with a strip of tabs has anywhere to go.
  fn tab_key(&self, key: KeyEvent) -> Option<usize> {
    if !self.tabbed() {
      return None;
    }
    let total = self.questions.len() + 1;
    let step = match key.code {
      KeyCode::Tab | KeyCode::Right => 1,
      KeyCode::BackTab | KeyCode::Left => total - 1,
      _ => return None,
    };
    Some((self.tab + step) % total)
  }

  fn move_by(&mut self, delta: isize) {
    let rows = self.rows().len();
    if rows == 0 {
      return;
    }
    self.row = (self.row as isize + delta).rem_euclid(rows as isize) as usize;
    // The free-text row takes the keyboard by being reached, so a question is
    // answered in one's own words by walking down to the row and typing.
    self.typing = self.row_at(self.row) == Some(Row::Other);
  }

  /// `Enter` on a row that is not the free-text one, which `typing_key` has
  /// already taken.
  fn enter(&mut self) -> Option<Outcome> {
    match self.row_at(self.row)? {
      // On a multi-select question `Enter` ticks the box under the cursor, the
      // way `Space` does — committing is what the `Next` row is for, so
      // neither hand has to leave the home row to tick a list of boxes.
      Row::Choice(index) if self.multi() => {
        self.toggle(index);
        None
      }
      Row::Choice(index) => {
        let label = self.questions[self.tab].options[index].label.clone();
        self.confirm(Answer::Chose(label))
      }
      Row::Next => {
        let labels = self.ticked_labels();
        self.confirm(Answer::Ticked(labels))
      }
      Row::Other => None,
    }
  }

  /// Keys while the free-text row has them.
  fn typing_key(&mut self, key: KeyEvent, ctrl: bool) -> Option<Outcome> {
    match key.code {
      KeyCode::Enter if crate::ui::is_newline(&key) => self.draft().insert_newline(),
      KeyCode::Char('j') if ctrl => self.draft().insert_newline(),
      KeyCode::Enter => {
        let text = draft_text(self.draft());
        return self.confirm(Answer::Typed(text));
      }
      // Pi's line-kill, taken as the whole draft rather than the line: the row
      // is one answer, and clearing it is what the key is reached for.
      KeyCode::Char('u') if ctrl => *self.draft() = Draft::default(),
      KeyCode::Backspace => _ = self.draft().delete_char(),
      KeyCode::Left => self.draft().move_cursor(CursorMove::Back),
      KeyCode::Right => self.draft().move_cursor(CursorMove::Forward),
      KeyCode::Home => self.draft().move_cursor(CursorMove::Head),
      KeyCode::End => self.draft().move_cursor(CursorMove::End),
      // Inside a draft of more than one line the arrows are the draft's; at
      // its ends they belong to the list again.
      KeyCode::Up if self.draft().cursor().0 == 0 => self.move_by(-1),
      KeyCode::Up => self.draft().move_cursor(CursorMove::Up),
      KeyCode::Down if self.draft().cursor().0 + 1 == self.draft().lines().len() => self.move_by(1),
      KeyCode::Down => self.draft().move_cursor(CursorMove::Down),
      KeyCode::Char(c) if !ctrl => self.draft().insert_char(c),
      _ => {}
    }
    None
  }

  fn submit_key(&mut self, key: KeyEvent) -> Option<Outcome> {
    if let Some(tab) = self.tab_key(key) {
      self.go_to(tab);
      return None;
    }
    match key.code {
      KeyCode::Up | KeyCode::Down => {
        self.submit = 1 - self.submit;
        None
      }
      // Submitting is allowed with questions left blank: the warning above the
      // picker says which, and a partial answer beats a dismissed dialog.
      KeyCode::Enter => match self.submit {
        1 => Some(Outcome::default()),
        _ => Some(self.answered()),
      },
      _ => None,
    }
  }

  // -------------------------------------------------------------- drawing

  /// The dialog at `width` columns, and which of its lines the cursor is on —
  /// which is the line that has to stay on screen when there are more of them
  /// than there is room for.
  pub fn lines(&self, width: u16) -> (Vec<Line<'static>>, usize) {
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut focus = 0;
    if self.tabbed() {
      out.push(self.tab_bar());
      out.push(Line::raw(""));
    }
    match self.questions.get(self.tab) {
      Some(question) => self.draw_question(question, width, &mut out, &mut focus),
      None => self.draw_submit(width, &mut out, &mut focus),
    }
    out.push(Line::raw(""));
    out.push(Line::styled(
      clip(&self.hint(), width),
      Style::default().add_modifier(Modifier::DIM),
    ));
    (out, focus)
  }

  /// The strip of tabs: a box per question, filled once it has an answer, and
  /// the submit tab at the end.
  fn tab_bar(&self) -> Line<'static> {
    let mut spans = vec![Span::styled(" ← ", Style::default().fg(Color::DarkGray))];
    for (index, question) in self.questions.iter().enumerate() {
      let answered = self.answers.contains_key(&index);
      let box_ = if answered { "■" } else { "□" };
      let style = match (index == self.tab, answered) {
        (true, _) => Style::default().fg(Color::Black).bg(Color::Cyan),
        (false, true) => Style::default().fg(Color::Green),
        (false, false) => Style::default().fg(Color::DarkGray),
      };
      spans.push(Span::styled(format!(" {box_} {} ", question.chip(index)), style));
      spans.push(Span::raw(" "));
    }
    let all = self.answers.len() == self.questions.len();
    let style = match (self.on_submit_tab(), all) {
      (true, _) => Style::default().fg(Color::Black).bg(Color::Cyan),
      (false, true) => Style::default().fg(Color::Green),
      (false, false) => Style::default().fg(Color::DarkGray),
    };
    spans.push(Span::styled(" ✓ Submit ", style));
    spans.push(Span::styled(" →", Style::default().fg(Color::DarkGray)));
    Line::from(spans)
  }

  fn draw_question(&self, question: &Question, width: u16, out: &mut Vec<Line<'static>>, focus: &mut usize) {
    // With a strip of tabs the header is already up there; without one it is
    // the only place the question's subject is said.
    if !self.tabbed() && !question.header.trim().is_empty() {
      out.push(Line::styled(
        format!(" {} ", question.header),
        Style::default().fg(Color::Black).bg(Color::Cyan),
      ));
      out.push(Line::raw(""));
    }
    out.extend(wrap_text(&question.question, width, Style::default().bold()));
    out.push(Line::raw(""));

    // The number column is as wide as the last row's number, which is the
    // free-text row's — one past the options.
    let digits = (question.options.len() + 1).to_string().len();
    let draft = self.drafts.get(&self.tab);
    for (index, row) in self.rows().into_iter().enumerate() {
      let active = index == self.row;
      if active {
        *focus = out.len();
      }
      match row {
        Row::Choice(at) => {
          let option = &question.options[at];
          let ticked = question.multi_select.then(|| self.is_ticked(at));
          let label = match self.confirmed_mark(Some(&option.label), active) {
            true => format!("{} ✔", option.label),
            false => option.label.clone(),
          };
          self.draw_row(Label::Text(label), Some(at + 1), digits, ticked, active, width, out);
          let indent = " ".repeat(prefix_width(digits, ticked.is_some()));
          for line in wrap_text(
            &option.description,
            width.saturating_sub(indent.len() as u16),
            Style::default().add_modifier(Modifier::DIM),
          ) {
            out.push(lead(indent.clone(), line));
          }
        }
        Row::Other => {
          let typed = draft.map(draft_text).unwrap_or_default();
          let empty = Draft::default();
          let label = match (active && self.typing, typed.is_empty()) {
            // While the row has the keyboard it shows the cursor, so an empty
            // draft still reads as somewhere to type.
            (true, _) => Label::Draft(draft.unwrap_or(&empty)),
            // What was typed stays in the row while the other options are
            // looked at, rather than reverting to the invitation.
            (false, false) => Label::Text(match self.confirmed_mark(None, active) {
              true => format!("{typed} ✔"),
              false => typed,
            }),
            (false, true) => Label::Text(OTHER_LABEL.to_string()),
          };
          let box_ = question.multi_select.then_some(false);
          self.draw_row(
            label,
            Some(question.options.len() + 1),
            digits,
            box_,
            active,
            width,
            out,
          );
        }
        // No number and no box: it is the question's full stop, not one of
        // its answers.
        Row::Next => self.draw_row(
          Label::Text(NEXT_LABEL.to_string()),
          None,
          digits,
          None,
          active,
          width,
          out,
        ),
      }
    }
  }

  /// Whether this row is the answer the question already has. The cursor's own
  /// row is never marked — it is already wearing the pointer, and two marks on
  /// one row read as two different things.
  fn confirmed_mark(&self, label: Option<&str>, active: bool) -> bool {
    if active || self.multi() {
      return false;
    }
    match (self.answers.get(&self.tab), label) {
      (Some(Answer::Chose(chosen)), Some(label)) => chosen == label,
      (Some(Answer::Typed(_)), None) => true,
      _ => false,
    }
  }

  /// One row: the pointer, its number, its box if the question has boxes, and
  /// as much of the label as the width holds, wrapped under itself.
  #[allow(clippy::too_many_arguments)]
  fn draw_row(
    &self,
    label: Label,
    number: Option<usize>,
    digits: usize,
    ticked: Option<bool>,
    active: bool,
    width: u16,
    out: &mut Vec<Line<'static>>,
  ) {
    let mut prefix = match active {
      true => POINTER.to_string(),
      false => NO_POINTER.to_string(),
    };
    if let Some(number) = number {
      prefix.push_str(&format!("{number:>digits$}. "));
    }
    if let Some(ticked) = ticked {
      prefix.push_str(if ticked { CHECKED } else { UNCHECKED });
      prefix.push(' ');
    }
    let indent = " ".repeat(prefix.chars().count());
    let style = match active {
      true => Style::default().fg(Color::Cyan).bold(),
      false => Style::default(),
    };
    let width = width.saturating_sub(prefix.chars().count() as u16);
    let wrapped = match label {
      Label::Text(text) => wrap_text(&text, width, style),
      Label::Draft(draft) => draft_shown(draft, width, style),
    };
    for (at, line) in wrapped.into_iter().enumerate() {
      out.push(lead(
        match at {
          0 => prefix.clone(),
          _ => indent.clone(),
        },
        line,
      ));
    }
  }

  fn draw_submit(&self, width: u16, out: &mut Vec<Line<'static>>, focus: &mut usize) {
    out.push(Line::styled(
      "Review your answers",
      Style::default().fg(Color::Cyan).bold(),
    ));
    out.push(Line::raw(""));
    let dim = Style::default().add_modifier(Modifier::DIM);
    for (index, question) in self.questions.iter().enumerate() {
      let Some(answer) = self.answers.get(&index) else {
        continue;
      };
      out.push(Line::styled(format!(" ● {}", question.chip(index)), dim));
      for line in wrap_text(&answer.scalar(), width.saturating_sub(5), Style::default()) {
        out.push(lead("   → ".to_string(), line));
      }
    }
    out.push(Line::raw(""));
    let missing: Vec<String> = self
      .questions
      .iter()
      .enumerate()
      .filter(|(index, _)| !self.answers.contains_key(index))
      .map(|(index, question)| question.chip(index))
      .collect();
    out.push(match missing.is_empty() {
      true => Line::styled("Ready to submit your answers?", dim),
      false => Line::styled(
        format!("⚠ Answer remaining questions before submitting: {}", missing.join(", ")),
        Style::default().fg(Color::Yellow),
      ),
    });
    for (index, label) in ["Submit answers", "Cancel"].into_iter().enumerate() {
      let active = index == self.submit;
      if active {
        *focus = out.len();
      }
      self.draw_row(
        Label::Text(label.to_string()),
        Some(index + 1),
        1,
        None,
        active,
        width,
        out,
      );
    }
  }

  /// The line along the bottom saying which keys do something here. It is the
  /// only teacher this dialog has, so it changes with what the keys currently
  /// do rather than listing all of them all of the time.
  fn hint(&self) -> String {
    let mut parts = vec!["Enter to select", "↑/↓ to navigate"];
    if self.multi() {
      parts.push("Space to toggle");
    }
    if self.tabbed() {
      parts.push("Tab to switch questions");
    }
    parts.push("Esc to cancel");
    if self.typing {
      // The one the input box's own placeholder names, and the one every
      // terminal reports: Shift+Enter breaks a line here too, but only where
      // the kitty keyboard protocol says it was pressed.
      parts.push("Alt+Enter for newline");
      parts.push("Ctrl+U to clear");
    }
    parts.join(" · ")
  }
}

/// One line's worth of `text`, with an ellipsis where the rest of it was.
///
/// The hint is the one line that must not wrap — it is written last and the
/// dialog's height is counted, so a second line of it would push the answer
/// the user is looking at off the bottom. What falls off the end is the keys
/// that matter least, which is why the line is written in that order.
fn clip(text: &str, width: u16) -> String {
  use unicode_width::UnicodeWidthChar;
  let width = width as usize;
  if text.chars().map(|c| c.width().unwrap_or(0)).sum::<usize>() <= width {
    return text.to_string();
  }
  let mut out = String::new();
  let mut used = 0;
  for c in text.chars() {
    let w = c.width().unwrap_or(0);
    if used + w + 1 > width {
      break;
    }
    out.push(c);
    used += w;
  }
  out.push('…');
  out
}

/// How far a row's text is indented: the pointer, the number and its dot, and
/// the box when the question has boxes.
fn prefix_width(digits: usize, boxed: bool) -> usize {
  NO_POINTER.chars().count() + digits + 2 + if boxed { CHECKED.chars().count() + 1 } else { 0 }
}

/// Puts `lead` in front of a rendered line, keeping the rest of its spans.
fn lead(lead: String, line: Line<'static>) -> Line<'static> {
  let mut spans = Vec::with_capacity(line.spans.len() + 1);
  spans.push(Span::raw(lead));
  spans.extend(line.spans);
  Line::from(spans)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn choice(label: &str) -> Choice {
    Choice {
      label: label.into(),
      description: format!("what {label} means"),
    }
  }

  fn question(text: &str, labels: &[&str], multi: bool) -> Question {
    Question {
      question: text.into(),
      header: text.split(' ').next().unwrap_or_default().into(),
      options: labels.iter().map(|l| choice(l)).collect(),
      multi_select: multi,
    }
  }

  fn press(dialog: &mut Dialog, code: KeyCode) -> Option<Outcome> {
    dialog.key(KeyEvent::new(code, KeyModifiers::NONE))
  }

  fn type_in(dialog: &mut Dialog, text: &str) {
    for c in text.chars() {
      press(dialog, KeyCode::Char(c));
    }
  }

  #[test]
  fn an_option_is_chosen_and_the_model_reads_it_back() {
    let questions = vec![question("Which cache?", &["Memory", "Disk"], false)];
    let mut dialog = Dialog::new(questions.clone());
    press(&mut dialog, KeyCode::Down);
    let outcome = press(&mut dialog, KeyCode::Enter).expect("one question answers the whole dialog");
    assert_eq!(outcome.answers, vec![(0, Answer::Chose("Disk".into()))]);
    assert_eq!(
      outcome.response(&questions),
      "User has answered your questions: \"Which cache?\"=\"Disk\". \
       You can now continue with the user's answers in mind."
    );
  }

  #[test]
  fn walking_onto_the_free_text_row_takes_the_keyboard() {
    let questions = vec![question("Which cache?", &["Memory", "Disk"], false)];
    let mut dialog = Dialog::new(questions.clone());
    // Two options, then the row that is typed into.
    press(&mut dialog, KeyCode::Up);
    assert!(dialog.typing, "the last row is the free-text one");
    type_in(&mut dialog, "redis");
    let outcome = press(&mut dialog, KeyCode::Enter).expect("an answer");
    assert_eq!(outcome.answers, vec![(0, Answer::Typed("redis".into()))]);
    assert!(outcome.response(&questions).contains("\"Which cache?\"=\"redis\""));
  }

  #[test]
  fn an_empty_free_text_answer_says_so_rather_than_saying_nothing() {
    let questions = vec![question("Which cache?", &["Memory", "Disk"], false)];
    let mut dialog = Dialog::new(questions.clone());
    press(&mut dialog, KeyCode::Up);
    let outcome = press(&mut dialog, KeyCode::Enter).expect("an answer");
    assert!(outcome.response(&questions).contains("\"Which cache?\"=\"(no input)\""));
  }

  #[test]
  fn a_draft_survives_a_walk_around_the_list() {
    let mut dialog = Dialog::new(vec![question("Which cache?", &["Memory", "Disk"], false)]);
    press(&mut dialog, KeyCode::Up);
    type_in(&mut dialog, "redis");
    press(&mut dialog, KeyCode::Down);
    assert!(!dialog.typing, "the top of the list is not the free-text row");
    press(&mut dialog, KeyCode::Up);
    let outcome = press(&mut dialog, KeyCode::Enter).expect("an answer");
    assert_eq!(outcome.answers, vec![(0, Answer::Typed("redis".into()))]);
  }

  #[test]
  fn the_draft_is_edited_where_the_cursor_is() {
    let mut dialog = Dialog::new(vec![question("Which cache?", &["Memory", "Disk"], false)]);
    press(&mut dialog, KeyCode::Up);
    type_in(&mut dialog, "redis");
    press(&mut dialog, KeyCode::Left);
    press(&mut dialog, KeyCode::Backspace);
    type_in(&mut dialog, "y");
    assert_eq!(draft_text(&dialog.drafts[&0]), "redys");
    dialog.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
    assert_eq!(draft_text(&dialog.drafts[&0]), "");
  }

  #[test]
  fn moving_the_cursor_moves_no_letters() {
    let mut dialog = Dialog::new(vec![question("Which cache?", &["Memory", "Disk"], false)]);
    press(&mut dialog, KeyCode::Up);
    type_in(&mut dialog, "redis");
    let row = |dialog: &Dialog| {
      let (lines, focus) = dialog.lines(40);
      lines[focus].to_string()
    };
    assert!(row(&dialog).ends_with("redis█"), "at the end the cursor is a block");
    press(&mut dialog, KeyCode::Left);
    let (lines, focus) = dialog.lines(40);
    assert!(
      lines[focus].to_string().ends_with("redis"),
      "the cursor is drawn over the s"
    );
    assert!(
      lines[focus]
        .spans
        .iter()
        .any(|span| span.content == "s" && span.style.add_modifier.contains(Modifier::REVERSED))
    );
  }

  #[test]
  fn arrows_walk_a_multi_line_draft_before_they_walk_the_list() {
    let mut dialog = Dialog::new(vec![question("Which cache?", &["Memory", "Disk"], false)]);
    press(&mut dialog, KeyCode::Up);
    type_in(&mut dialog, "one");
    dialog.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
    type_in(&mut dialog, "two");
    assert_eq!(draft_text(&dialog.drafts[&0]), "one\ntwo");
    // Up moves between the draft's own lines, and only then leaves the row.
    press(&mut dialog, KeyCode::Up);
    assert!(dialog.typing, "still in the draft");
    press(&mut dialog, KeyCode::Up);
    assert!(!dialog.typing, "out of the draft and up the list");
  }

  #[test]
  fn a_pasted_answer_keeps_its_newlines_instead_of_confirming_at_them() {
    let mut dialog = Dialog::new(vec![question("Which cache?", &["Memory", "Disk"], false)]);
    press(&mut dialog, KeyCode::Up);
    dialog.paste("one\ntwo");
    assert_eq!(draft_text(&dialog.drafts[&0]), "one\ntwo");
    assert!(
      dialog.typing,
      "the paste answered nothing and the row still has the keys"
    );
    // On a row of choices there is nothing a paste could mean.
    press(&mut dialog, KeyCode::Down);
    dialog.paste("three");
    assert_eq!(draft_text(&dialog.drafts[&0]), "one\ntwo");
  }

  #[test]
  fn boxes_are_ticked_and_committed_from_the_next_row() {
    let questions = vec![question("Which features?", &["Logs", "Metrics", "Traces"], true)];
    let mut dialog = Dialog::new(questions.clone());
    press(&mut dialog, KeyCode::Char(' '));
    press(&mut dialog, KeyCode::Down);
    press(&mut dialog, KeyCode::Down);
    // Enter ticks rather than commits, which is what the Next row is for.
    assert!(press(&mut dialog, KeyCode::Enter).is_none());
    assert_eq!(dialog.ticked_labels(), vec!["Logs".to_string(), "Traces".into()]);
    press(&mut dialog, KeyCode::Up);
    press(&mut dialog, KeyCode::Up);
    press(&mut dialog, KeyCode::Up);
    let outcome = press(&mut dialog, KeyCode::Enter).expect("the Next row commits");
    assert_eq!(
      outcome.answers,
      vec![(0, Answer::Ticked(vec!["Logs".into(), "Traces".into()]))]
    );
    assert!(
      outcome
        .response(&questions)
        .contains("\"Which features?\"=\"Logs, Traces\""),
      "{}",
      outcome.response(&questions)
    );
  }

  #[test]
  fn typing_over_ticked_boxes_replaces_them() {
    let mut dialog = Dialog::new(vec![question("Which features?", &["Logs", "Metrics"], true)]);
    press(&mut dialog, KeyCode::Char(' '));
    press(&mut dialog, KeyCode::Down);
    press(&mut dialog, KeyCode::Down);
    type_in(&mut dialog, "all of them");
    let outcome = press(&mut dialog, KeyCode::Enter).expect("an answer");
    assert_eq!(outcome.answers, vec![(0, Answer::Typed("all of them".into()))]);
  }

  #[test]
  fn several_questions_walk_their_tabs_and_end_on_the_submit_tab() {
    let questions = vec![
      question("Which cache?", &["Memory", "Disk"], false),
      question("Which tests?", &["Unit", "Integration"], false),
    ];
    let mut dialog = Dialog::new(questions.clone());
    // Answering the first moves to the second, and the second to the review.
    assert!(press(&mut dialog, KeyCode::Enter).is_none());
    assert_eq!(dialog.tab, 1);
    assert!(press(&mut dialog, KeyCode::Enter).is_none());
    assert!(dialog.on_submit_tab());
    let outcome = press(&mut dialog, KeyCode::Enter).expect("the submit row answers");
    assert_eq!(
      outcome.answers,
      vec![(0, Answer::Chose("Memory".into())), (1, Answer::Chose("Unit".into()))]
    );
    let response = outcome.response(&questions);
    assert!(
      response.contains("\"Which cache?\"=\"Memory\". \"Which tests?\"=\"Unit\"."),
      "{response}"
    );
  }

  #[test]
  fn a_tab_gone_back_to_says_what_it_was_answered_with() {
    let mut dialog = Dialog::new(vec![
      question("Which cache?", &["Memory", "Disk"], false),
      question("Which tests?", &["Unit", "Integration"], false),
    ]);
    press(&mut dialog, KeyCode::Down);
    press(&mut dialog, KeyCode::Enter);
    press(&mut dialog, KeyCode::BackTab);
    let text: Vec<String> = dialog.lines(60).0.iter().map(ToString::to_string).collect();
    assert!(text.iter().any(|l| l.contains("2. Disk ✔")), "{text:?}");
    // The row the cursor is on is never marked twice — it is already wearing
    // the pointer.
    assert!(text.iter().any(|l| l.contains("› 1. Memory")), "{text:?}");
    // The same for an answer in the user's own words.
    press(&mut dialog, KeyCode::Up);
    type_in(&mut dialog, "redis");
    press(&mut dialog, KeyCode::Enter);
    press(&mut dialog, KeyCode::BackTab);
    let text: Vec<String> = dialog.lines(60).0.iter().map(ToString::to_string).collect();
    assert!(text.iter().any(|l| l.contains("3. redis ✔")), "{text:?}");
  }

  #[test]
  fn the_hint_is_clipped_rather_than_wrapped() {
    // It is the last line and the dialog's height is counted from it, so a
    // second line of hint would push the row being answered off the bottom.
    let dialog = Dialog::new(vec![question("Which cache?", &["Memory", "Disk"], false)]);
    let (lines, _) = dialog.lines(20);
    let hint = lines.last().expect("a hint").to_string();
    assert!(hint.ends_with('…'), "{hint:?}");
    assert!(hint.chars().count() <= 20, "{hint:?}");
  }

  #[test]
  fn a_questionnaire_can_be_submitted_with_questions_left_blank() {
    let questions = vec![
      question("Which cache?", &["Memory", "Disk"], false),
      question("Which tests?", &["Unit", "Integration"], false),
    ];
    let mut dialog = Dialog::new(questions.clone());
    assert!(press(&mut dialog, KeyCode::Enter).is_none());
    // Tab past the second question, leaving it alone.
    assert!(press(&mut dialog, KeyCode::Tab).is_none());
    assert!(dialog.on_submit_tab());
    let outcome = press(&mut dialog, KeyCode::Enter).expect("an answer");
    assert_eq!(outcome.answers, vec![(0, Answer::Chose("Memory".into()))]);
  }

  #[test]
  fn cancelling_reads_as_a_decline_however_much_was_answered() {
    let questions = vec![
      question("Which cache?", &["Memory", "Disk"], false),
      question("Which tests?", &["Unit", "Integration"], false),
    ];
    let mut dialog = Dialog::new(questions.clone());
    press(&mut dialog, KeyCode::Enter);
    let outcome = press(&mut dialog, KeyCode::Esc).expect("Esc ends it");
    assert!(outcome.answers.is_empty());
    assert_eq!(outcome.response(&questions), DECLINED);
    // The cancel row of the submit tab says the same thing.
    let mut dialog = Dialog::new(questions.clone());
    press(&mut dialog, KeyCode::Enter);
    press(&mut dialog, KeyCode::Enter);
    press(&mut dialog, KeyCode::Down);
    let outcome = press(&mut dialog, KeyCode::Enter).expect("cancel ends it");
    assert_eq!(outcome.response(&questions), DECLINED);
  }

  #[test]
  fn an_answer_is_read_back_out_of_the_tab_it_was_left_on() {
    let questions = vec![
      question("Which features?", &["Logs", "Metrics"], true),
      question("Which tests?", &["Unit", "Integration"], false),
    ];
    let mut dialog = Dialog::new(questions);
    press(&mut dialog, KeyCode::Char(' '));
    press(&mut dialog, KeyCode::Tab);
    assert!(
      dialog.ticked_labels().is_empty(),
      "the boxes belong to the tab that has them"
    );
    press(&mut dialog, KeyCode::BackTab);
    assert_eq!(dialog.ticked_labels(), vec!["Logs".to_string()]);
  }

  #[test]
  fn what_the_model_may_not_ask() {
    let ok = question("Which cache?", &["Memory", "Disk"], false);
    assert!(validate(std::slice::from_ref(&ok)).is_ok());
    assert_eq!(validate(&[]).unwrap_err(), "Error: At least one question is required");
    assert_eq!(
      validate(&vec![ok.clone(); 5]).unwrap_err(),
      "Error: At most 4 questions are allowed per invocation"
    );
    assert_eq!(
      validate(&[ok.clone(), ok.clone()]).unwrap_err(),
      "Error: Question text must be unique within an invocation"
    );
    assert_eq!(
      validate(&[question("Which cache?", &["Memory"], false)]).unwrap_err(),
      "Error: Each question requires at least 2 options"
    );
    assert_eq!(
      validate(&[question("Which cache?", &["Memory", "Other"], false)]).unwrap_err(),
      "Error: Option label is reserved (Other, Type something., Next)"
    );
    assert_eq!(
      validate(&[question("Which cache?", &["Memory", "Memory"], false)]).unwrap_err(),
      "Error: Option labels must be unique within a question"
    );
  }

  #[test]
  fn a_carriage_return_inside_a_label_never_reaches_the_terminal() {
    let prepared = prepare(vec![Question {
      question: "Which\r\ncache?".into(),
      header: "Cache\r".into(),
      options: vec![
        Choice {
          label: "Other\r".into(),
          description: "in\rline".into(),
        },
        choice("Disk"),
      ],
      multi_select: false,
    }]);
    assert_eq!(prepared[0].question, "Which\ncache?");
    assert_eq!(prepared[0].header, "Cache");
    assert_eq!(prepared[0].options[0].description, "inline");
    // And the folded label is the reserved one it was hiding behind the CR.
    assert_eq!(
      validate(&prepared).unwrap_err(),
      "Error: Option label is reserved (Other, Type something., Next)"
    );
  }

  #[test]
  fn the_dialog_draws_its_question_its_options_and_where_the_cursor_is() {
    let mut dialog = Dialog::new(vec![
      question("Which cache?", &["Memory", "Disk"], false),
      question("Which tests?", &["Unit", "Integration"], true),
    ]);
    let (lines, focus) = dialog.lines(60);
    let text: Vec<String> = lines.iter().map(ToString::to_string).collect();
    assert!(text.iter().any(|l| l.contains("□ Which")), "a tab strip: {text:?}");
    assert!(text.iter().any(|l| l.contains("Which cache?")), "{text:?}");
    assert!(text[focus].contains("› 1. Memory"), "the cursor's row: {text:?}");
    assert!(text.iter().any(|l| l.contains("3. Type something.")), "{text:?}");
    assert!(
      text.last().is_some_and(|l| l.contains("Tab to switch questions")),
      "the hint: {text:?}"
    );

    // The multi-select question has boxes, a Next row, and says so.
    press(&mut dialog, KeyCode::Tab);
    let (lines, _) = dialog.lines(60);
    let text: Vec<String> = lines.iter().map(ToString::to_string).collect();
    assert!(text.iter().any(|l| l.contains("› 1. [ ] Unit")), "{text:?}");
    assert!(text.iter().any(|l| l.contains("Next")), "{text:?}");
    assert!(
      text.last().is_some_and(|l| l.contains("Space to toggle")),
      "the hint: {text:?}"
    );

    // And the submit tab reviews what has been answered so far.
    press(&mut dialog, KeyCode::Char(' '));
    press(&mut dialog, KeyCode::Tab);
    let (lines, focus) = dialog.lines(60);
    let text: Vec<String> = lines.iter().map(ToString::to_string).collect();
    assert!(text.iter().any(|l| l.contains("Review your answers")), "{text:?}");
    assert!(text.iter().any(|l| l.contains("→ Unit")), "{text:?}");
    assert!(
      text.iter().any(|l| l.contains("⚠ Answer remaining questions")),
      "{text:?}"
    );
    assert!(text[focus].contains("Submit answers"), "{text:?}");
  }
}
