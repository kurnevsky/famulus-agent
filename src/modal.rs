//! Putting something to the user while a run waits on the answer.
//!
//! A [`Component`] handed to a [`Host`] is drawn over the screen and takes the
//! keyboard until it finishes, and whatever it finishes with is what the side
//! that showed it gets back. The `ask` tool's questionnaire and an MCP
//! server's form both come up this way, so the UI has one kind of thing to
//! draw and route keys to, whoever is asking.

use std::fmt;

use ratatui::crossterm::event::KeyEvent;
use ratatui::text::Line;
use tokio::sync::{mpsc, oneshot};

use crate::agent::AgentEvent;

// ---------------------------------------------------------------- components

/// Something drawn over the screen that takes the keyboard until it is done.
///
/// It is driven one key at a time and drawn from scratch every frame: the
/// state is its own, and the screen only ever asks what it looks like now.
pub trait Component: Send + 'static {
  /// What it finishes with.
  type Output: Send + 'static;

  /// What the frame around it says it is.
  fn title(&self) -> String;

  /// A key the user pressed. `Some` finishes it with that answer.
  fn key(&mut self, key: KeyEvent) -> Option<Self::Output>;

  /// A bracketed paste, which is text rather than the keys it is made of.
  fn paste(&mut self, _text: &str) {}

  /// It laid out at `width` columns, and which of the lines the cursor is on
  /// — the one to keep on screen when there are more than there is room for.
  fn lines(&self, width: u16) -> (Vec<Line<'static>>, usize);
}

/// A component as the screen holds it: whatever its answer is, it has already
/// been told where to send it.
///
/// Dropping one unfinished is how it is put away without an answer — the run
/// that showed it was stopped, or the terminal went — and the side waiting on
/// it reads that as nobody having answered.
pub trait Modal: Send {
  fn title(&self) -> String;
  /// A key the user pressed; `true` when that finished it, and it can go.
  fn key(&mut self, key: KeyEvent) -> bool;
  fn paste(&mut self, text: &str);
  fn lines(&self, width: u16) -> (Vec<Line<'static>>, usize);
}

impl fmt::Debug for dyn Modal {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_tuple("Modal").field(&self.title()).finish()
  }
}

/// A component and the channel its answer goes down.
struct Shown<C: Component> {
  component: C,
  reply: Option<oneshot::Sender<C::Output>>,
}

impl<C: Component> Modal for Shown<C> {
  fn title(&self) -> String {
    self.component.title()
  }

  fn key(&mut self, key: KeyEvent) -> bool {
    let Some(output) = self.component.key(key) else {
      return false;
    };
    if let Some(reply) = self.reply.take() {
      let _ = reply.send(output);
    }
    true
  }

  fn paste(&mut self, text: &str) {
    self.component.paste(text);
  }

  fn lines(&self, width: u16) -> (Vec<Line<'static>>, usize) {
    self.component.lines(width)
  }
}

// ---------------------------------------------------------------- the host

/// What a tool or a server reaches the user through: a modal goes to the UI
/// with the rest of what the run has to say.
#[derive(Clone)]
pub struct Host {
  tx: mpsc::UnboundedSender<AgentEvent>,
}

impl Host {
  pub fn new(tx: mpsc::UnboundedSender<AgentEvent>) -> Self {
    Self { tx }
  }

  /// Show `component`, and wait for however long the user takes to finish
  /// it. `None` is no answer at all: the component was put away before it
  /// finished.
  pub async fn show<C: Component>(&self, component: C) -> Option<C::Output> {
    let (tx, rx) = oneshot::channel();
    let _ = self.tx.send(AgentEvent::Show(Box::new(Shown {
      component,
      reply: Some(tx),
    })));
    rx.await.ok()
  }

  /// Tell the UI something that is not a question: a server whose tools
  /// changed, say. Nothing waits on it.
  #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
  pub fn tell(&self, event: AgentEvent) {
    let _ = self.tx.send(event);
  }
}

/// A host whose first modal is handed to `answer`, standing in for the UI.
#[cfg(test)]
pub fn answered_by(answer: impl FnOnce(Box<dyn Modal>) + Send + 'static) -> Host {
  let (tx, mut rx) = mpsc::unbounded_channel();
  tokio::spawn(async move {
    if let Some(AgentEvent::Show(modal)) = rx.recv().await {
      answer(modal);
    }
  });
  Host::new(tx)
}

#[cfg(test)]
mod tests {
  use ratatui::crossterm::event::KeyCode;

  use super::*;

  /// Counts keys, and finishes on the third.
  struct Three(usize);

  impl Component for Three {
    type Output = usize;

    fn title(&self) -> String {
      "three".into()
    }

    fn key(&mut self, _key: KeyEvent) -> Option<usize> {
      self.0 += 1;
      (self.0 == 3).then_some(self.0)
    }

    fn lines(&self, _width: u16) -> (Vec<Line<'static>>, usize) {
      (vec![Line::raw(self.0.to_string())], 0)
    }
  }

  #[tokio::test]
  async fn a_component_answers_with_what_it_finished_with() {
    let host = answered_by(|mut modal| {
      assert_eq!(modal.title(), "three");
      assert!(!modal.key(KeyCode::Enter.into()));
      assert!(!modal.key(KeyCode::Enter.into()));
      assert_eq!(modal.lines(10).0, vec![Line::raw("2")]);
      assert!(modal.key(KeyCode::Enter.into()), "the third key finishes it");
    });
    assert_eq!(host.show(Three(0)).await, Some(3));
  }

  #[tokio::test]
  async fn one_put_away_unfinished_has_no_answer() {
    assert_eq!(answered_by(drop).show(Three(0)).await, None);
  }
}
