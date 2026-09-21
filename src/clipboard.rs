//! Copying, as an escape handed to the terminal rather than as a call on the
//! machine this is running on.
//!
//! OSC 52 asks the terminal to put text on its own clipboard, which is the
//! same thing when it is drawing on the same machine and the right thing when
//! it is not: over ssh the text lands where the user is actually looking,
//! without an X11 or Wayland connection to reach for. Terminals that do not
//! speak it, or that ship it turned off — tmux wants `set -g set-clipboard
//! on`, xterm `allowWindowOps` — ignore the escape, which is why nothing here
//! can report that the text arrived.

use std::io::{self, Write};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;

/// What is handed over at once. Terminals cap the escape they will read and
/// drop the rest of it — tmux at around 74 KB of base64 — so text that would
/// not survive the trip is refused here, where it can be said out loud,
/// rather than arriving on the clipboard with its end missing. Counted before
/// encoding, which is where the user's own figure is.
const MAX_BYTES: usize = 48 * 1024;

/// Both selections at once: the primary one, which is where a selection made
/// with the mouse belongs and what a middle click pastes, and the clipboard
/// proper, which is what `Ctrl+V` pastes and the only one some terminals have.
const SELECTIONS: &str = "pc";

/// Put `text` on the terminal's clipboard. The error is for the user to read:
/// it says what stopped rather than what failed.
pub fn copy(text: &str) -> Result<(), String> {
  if text.len() > MAX_BYTES {
    return Err(format!(
      "Too much to copy: {} KB, where a clipboard escape carries {} KB.",
      text.len() / 1024,
      MAX_BYTES / 1024
    ));
  }
  let mut out = io::stdout().lock();
  out
    .write_all(sequence(text).as_bytes())
    .and_then(|()| out.flush())
    .map_err(|err| format!("Could not copy: {err}"))
}

/// The escape carrying `text`, ended with `BEL` — which every terminal that
/// reads OSC 52 accepts, where the string terminator some prefer is the one
/// screen and tmux are picky about.
fn sequence(text: &str) -> String {
  format!("\x1b]52;{SELECTIONS};{}\x07", STANDARD.encode(text))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn the_escape_carries_the_text_as_base64_for_both_selections() {
    assert_eq!(sequence("hi"), "\x1b]52;pc;aGk=\x07");
    // Newlines and non-ascii go through the encoding rather than through the
    // terminal's parser, which is the point of it.
    assert_eq!(sequence("a\nб"), "\x1b]52;pc;YQrQsQ==\x07");
  }

  #[test]
  fn more_than_the_escape_carries_is_refused_rather_than_cut() {
    let long = "x".repeat(MAX_BYTES + 1);
    assert!(copy(&long).is_err());
  }
}
