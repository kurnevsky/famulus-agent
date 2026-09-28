//! The last say on what reaches the terminal.
//!
//! Ratatui sends a frame as the cells that changed since the last one, and
//! where it moves the cursor between them is worked out from how wide it
//! believes each symbol is. A terminal that draws one of them a column wider
//! or narrower shifts the rest of that row, and since ratatui still believes
//! those cells hold what it sent, it never draws them again: the row stays
//! wrong until something happens to change it. Width is not something
//! terminals agree on — combining marks, emoji, scripts newer than their
//! tables, text laid out right to left — and the text on screen comes from
//! models, tools and files, which can hold any of it.
//!
//! So before a frame goes out, every cell is held to a list of characters
//! terminals, the Linux console among them, draw at the width ratatui gives
//! them, one to a cell, and anything else is drawn as `�`. The list is short
//! on purpose: a character it leaves out costs a `�`, one it wrongly lets in
//! costs a broken screen.

use ratatui::buffer::{Buffer, CellDiffOption};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// What a cell that could not be trusted shows instead.
const REPLACEMENT: &str = "\u{fffd}";

/// Replace every symbol in `buffer` whose width the terminal might see
/// differently.
///
/// A replaced symbol is one column where it may have been two; the cell after
/// a wide symbol is already blank, so the row keeps its layout, and the blank
/// is drawn like any other cell.
pub fn guard(buffer: &mut Buffer) {
  for cell in &mut buffer.content {
    // A cell that asked to be left alone or drawn at a width of its own is
    // covering something ratatui does not draw, and knows what it is doing.
    if cell.diff_option != CellDiffOption::None {
      continue;
    }
    let symbol = cell.symbol();
    // A symbol of nothing is drawn as nothing, whatever it means.
    if symbol.is_empty() || safe(symbol) {
      continue;
    }
    match plain(symbol) {
      Some(plain) => cell.set_symbol(&plain),
      None => cell.set_symbol(REPLACEMENT),
    };
  }
}

/// `symbol` without what only says how an emoji is to look — drawn as text
/// or as a picture, and in which skin tone — if what is left can be trusted
/// and is no wider than the columns laid out for `symbol`: any it does not
/// fill are blank, as after a replacement.
fn plain(symbol: &str) -> Option<String> {
  let plain: String = symbol
    .chars()
    .filter(|c| !matches!(c, '\u{fe0e}' | '\u{fe0f}' | '\u{1f3fb}'..='\u{1f3ff}'))
    .collect();
  (safe(&plain) && plain.width() <= symbol.width()).then_some(plain)
}

/// Whether `symbol` is drawn the same by every terminal: one character — a
/// cluster is as wide as each terminal decides — narrow or wide as the lists
/// below say, and as ratatui measures it.
fn safe(symbol: &str) -> bool {
  let mut chars = symbol.chars();
  let (Some(c), None) = (chars.next(), chars.next()) else {
    return false;
  };
  match c.width() {
    Some(1) => narrow(c),
    Some(2) => wide(c),
    _ => false,
  }
}

/// One column in every terminal: Latin, Greek and Cyrillic, punctuation, and
/// the symbols an interface is drawn with — arrows, box drawing, blocks,
/// shapes, dingbats, braille, and the letters of mathematics. The soft hyphen
/// is drawn by some and not others. A few symbols an emoji skin tone can
/// follow are taken for emoji, and so two columns, by some terminals, and the
/// Linux console draws the gender signs, mostly written joined to another
/// emoji, as nothing.
fn narrow(c: char) -> bool {
  matches!(
    c,
    '\u{20}'..='\u{7e}'
      | '\u{a0}'..='\u{ac}'
      | '\u{ae}'..='\u{2ff}'
      | '\u{370}'..='\u{482}'
      | '\u{48a}'..='\u{52f}'
      | '\u{1e00}'..='\u{1fff}'
      | '\u{2010}'..='\u{2027}'
      | '\u{2030}'..='\u{205e}'
      | '\u{2070}'..='\u{20cf}'
      | '\u{2100}'..='\u{2bff}'
      | '\u{fffd}'
      | '\u{1d400}'..='\u{1d7ff}'
  ) && !matches!(
    c,
    '\u{261d}' | '\u{26f9}' | '\u{270c}' | '\u{270d}' | '\u{2640}' | '\u{2642}' | '\u{26a7}'
  )
}

/// Two columns in every terminal: the ideographs, kana and Hangul of Chinese,
/// Japanese and Korean, and their full-width punctuation and forms — only the
/// blocks that have been full for long enough that no terminal's tables are
/// missing any of them — and emoji, which every terminal of the last few years
/// and the Linux console since 6.16 draw at two columns. The console draws
/// the skin tones and hair styles, only ever meant to follow another emoji,
/// as nothing, and so they are not here.
fn wide(c: char) -> bool {
  matches!(
    c,
    '\u{3000}'..='\u{303e}'
      | '\u{3041}'..='\u{3096}'
      | '\u{309b}'..='\u{30ff}'
      | '\u{3131}'..='\u{318e}'
      | '\u{3400}'..='\u{4dbf}'
      | '\u{4e00}'..='\u{9fff}'
      | '\u{ac00}'..='\u{d7a3}'
      | '\u{ff01}'..='\u{ff60}'
      | '\u{ffe0}'..='\u{ffe6}'
      | '\u{2300}'..='\u{2bff}'
      | '\u{1f000}'..='\u{1f3fa}'
      | '\u{1f400}'..='\u{1f64f}'
      | '\u{1f680}'..='\u{1f9af}'
      | '\u{1f9b4}'..='\u{1faff}'
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use ratatui::layout::Rect;
  use ratatui::style::Style;

  fn drawn(text: &str) -> Vec<String> {
    let mut buffer = Buffer::empty(Rect::new(0, 0, 12, 1));
    buffer.set_string(0, 0, text, Style::default());
    guard(&mut buffer);
    buffer.content.iter().map(|cell| cell.symbol().to_string()).collect()
  }

  #[test]
  fn what_every_terminal_agrees_on_is_left_alone() {
    assert_eq!(drawn("a─╭▄⠋ё€日本").concat(), "a─╭▄⠋ё€日 本  ");
  }

  #[test]
  fn anything_else_is_one_column_of_replacement_and_the_row_keeps_its_layout() {
    // A cluster, emoji joined into one two columns wide, and a syllable whose
    // vowel sign ratatui counts as a column and some terminals as none. What
    // was two columns wide is still two, the second blank.
    let cells = drawn("e\u{301}👨\u{200d}💻x\u{915}\u{93e}y");
    let replaced = "\u{fffd}";
    assert_eq!(cells[..7], [replaced, replaced, " ", "x", replaced, " ", "y"]);
  }

  #[test]
  fn emoji_are_two_columns() {
    let cells = drawn("✅❌🚀♀");
    assert_eq!(cells[..7], ["✅", " ", "❌", " ", "🚀", " ", "\u{fffd}"]);
  }

  #[test]
  fn how_an_emoji_is_to_look_goes_where_what_is_left_fits() {
    // Asked for as a picture, which takes two columns, and in a skin tone.
    // An emoji asked for as text is laid out in one column, too few for it.
    let cells = drawn("⚠\u{fe0f}👍\u{1f3fd}✅\u{fe0e}x");
    assert_eq!(cells[..6], ["⚠", " ", "👍", " ", "\u{fffd}", "x"]);
  }

  #[test]
  fn a_cell_drawn_over_by_something_else_is_not_touched() {
    let mut buffer = Buffer::empty(Rect::new(0, 0, 1, 1));
    buffer.content[0].set_symbol("🚀").set_diff_option(CellDiffOption::Skip);
    guard(&mut buffer);
    assert_eq!(buffer.content[0].symbol(), "🚀");
  }
}
