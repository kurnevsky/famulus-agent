//! Colours kept in a file rather than in the code that draws with them.
//!
//! `theme.toml` is found where `config.toml` is, on the XDG search path and
//! nowhere else, and merged the same way: key by key, the nearer file winning.
//! `--theme`, or `theme` in `config.toml`, names one file to read instead.
//! A file only says what it changes; everything it leaves out keeps its
//! default.
//!
//! ```toml
//! [ui]
//! accent = "light-cyan"
//!
//! [markdown]
//! link = "#5f87ff"
//!
//! [syntax]
//! comment = "242"
//! operator = "yellow"
//! ```
//!
//! A colour is a name from the terminal's palette (`red`, `dark-gray`,
//! `light-blue`, …), an index into its 256 written as a string (`"242"`), a
//! `#rrggbb`, or `reset` for the terminal's own colour. The defaults are
//! palette names, so they follow whatever theme the terminal is wearing, light
//! or dark — all but a changed line's background, which no palette colour can
//! be without drowning the text on it in its own hue. Those are dark tints,
//! 24-bit where the terminal draws 24-bit colour and the nearest the 256 have
//! where it does not. Bold, italics and underlines are not colours and stay
//! where they are.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use ratatui::style::Color;
use serde::{Deserialize, Deserializer};

/// The name colours are kept under, in each configuration directory.
pub const FILE: &str = "theme.toml";

static THEME: OnceLock<Theme> = OnceLock::new();

/// The colours this run draws with: what [`set`] was given, or the defaults
/// if nothing was — which is what a test sees.
pub fn theme() -> &'static Theme {
  THEME.get_or_init(Theme::default)
}

/// Fix the colours for the rest of the run. Only the first call counts:
/// whatever has been drawn, highlighted and cached by then was drawn in it.
pub fn set(theme: Theme) {
  let _ = THEME.set(theme);
}

/// A section a key fa does not know is an error in, as it is in `config.toml`:
/// a misspelled `acent` that went quietly would be a colour that never took.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Theme {
  pub ui: Ui,
  pub markdown: Markdown,
  /// The one section open to new keys: a capture name not among the defaults
  /// is one more thing highlighted.
  #[serde(deserialize_with = "syntax")]
  pub syntax: Syntax,
}

/// Everything around the conversation, and the marks fa puts in it.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Ui {
  /// What points at something: the prompt's `❯`, the row a picker is on, the
  /// letters a filter matched, a question's header and the option it is on.
  pub accent: Color,
  /// Text drawn on an `accent` background.
  pub on_accent: Color,
  /// What is there without asking to be read: placeholders, the input box's
  /// border while a turn runs, the scrollbar's track, a question not yet
  /// answered.
  pub muted: Color,
  /// The rounded box every panel is drawn in, and the scrollbar's thumb.
  pub border: Color,
  /// A model's reasoning.
  pub thinking: Color,
  /// The spinner while a turn runs.
  pub working: Color,
  /// Something worth a second look: an attachment that will not be sent, an
  /// answer that is missing, a context window filling up.
  pub warning: Color,
  /// An error, a tool call that failed, a context window all but full.
  pub error: Color,
  /// A tool call that finished, a question that has its answer.
  pub success: Color,
  /// A tool call's mark and name, and the cursor while its arguments arrive.
  pub tool: Color,
  /// The header a compacted context is summarized under.
  pub summary: Color,
  /// The `+` side of a diff.
  pub added: Color,
  /// The `-` side of a diff.
  pub removed: Color,
  /// What a `+` line is drawn on, the width of the transcript.
  pub added_background: Color,
  /// What a `-` line is drawn on, the width of the transcript.
  pub removed_background: Color,
}

/// Which colours the terminal can draw, as far as the defaults are concerned:
/// a colour a file gives is drawn as it is written either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Colors {
  /// 24-bit if COLORTERM says the terminal has it, the 256 if not
  Auto,
  /// Any #rrggbb, as it is
  Truecolor,
  /// Only the 256-colour palette
  #[value(name = "256")]
  Palette,
}

impl Colors {
  /// Whether `#rrggbb` reaches the screen as itself.
  ///
  /// `COLORTERM` is the one thing terminals agree to say it with, and it says
  /// only yes: ssh and tmux both drop it, so a terminal that has 24-bit colour
  /// can still look as though it does not — which is what naming it is for.
  pub fn truecolor(self) -> bool {
    match self {
      Colors::Auto => says_truecolor(std::env::var("COLORTERM").ok().as_deref()),
      Colors::Truecolor => true,
      Colors::Palette => false,
    }
  }

  /// What a changed line is drawn on unless a file says otherwise, added
  /// then removed.
  ///
  /// Dark enough for the line's own colours to read on, which is a dark
  /// terminal's background; a light one wants lighter ones. Without 24-bit
  /// colour the tints are not left for the terminal to round, since the
  /// nearest of the 256 to that green is a grey: the 256 have a darkest green
  /// and red of their own, brighter than the tints, and those are used
  /// instead.
  fn diff_backgrounds(truecolor: bool) -> (Color, Color) {
    match truecolor {
      true => (Color::Rgb(0x00, 0x28, 0x00), Color::Rgb(0x3f, 0x00, 0x01)),
      false => (Color::Indexed(22), Color::Indexed(52)),
    }
  }
}

/// What `COLORTERM` says when the terminal draws 24-bit colour.
fn says_truecolor(colorterm: Option<&str>) -> bool {
  colorterm.is_some_and(|value| value.eq_ignore_ascii_case("truecolor") || value.eq_ignore_ascii_case("24bit"))
}

impl Default for Ui {
  fn default() -> Self {
    let (added_background, removed_background) = Colors::diff_backgrounds(true);
    Self {
      accent: Color::Cyan,
      on_accent: Color::Black,
      // Bright black rather than a dimmed foreground, which some terminals
      // ignore and others render as the text colour proper.
      muted: Color::DarkGray,
      border: Color::Gray,
      thinking: Color::DarkGray,
      working: Color::Yellow,
      warning: Color::Yellow,
      error: Color::Red,
      success: Color::Green,
      tool: Color::Yellow,
      summary: Color::Magenta,
      added: Color::Green,
      removed: Color::Red,
      added_background,
      removed_background,
    }
  }
}

/// What the model's Markdown is drawn in.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Markdown {
  pub heading: Color,
  /// A code block with no grammar to highlight it, and the text a grammar
  /// leaves alone.
  pub code_block: Color,
  pub inline_code: Color,
  pub link: Color,
  /// Everything drawn around the text rather than in it: a code block's
  /// fences, a quote's bar, a rule, a table's lines.
  pub border: Color,
  pub quote: Color,
  pub bullet: Color,
  pub footnote: Color,
}

impl Default for Markdown {
  fn default() -> Self {
    Self {
      heading: Color::Yellow,
      code_block: Color::Green,
      inline_code: Color::Cyan,
      link: Color::Blue,
      border: Color::DarkGray,
      quote: Color::DarkGray,
      bullet: Color::Green,
      footnote: Color::Blue,
    }
  }
}

/// Capture names highlighted, each with the colour it draws in.
///
/// A capture matches the entry whose dotted parts it all contains, the longest
/// such entry winning and ties going to whichever is listed first. So
/// `@function.method` lands on `function`, and `@keyword.function` — which
/// matches both `keyword` and `function` by one part each — needs an entry of
/// its own to stop the tie deciding it. Listing a name is also what makes it
/// highlight at all: a capture with no entry here produces no event, and its
/// text stays as the code's default foreground.
#[derive(Debug, PartialEq)]
pub struct Syntax(pub Vec<(String, Color)>);

impl Default for Syntax {
  fn default() -> Self {
    let names = [
      ("attribute", Color::Magenta),
      ("boolean", Color::Cyan),
      ("character", Color::Green),
      ("comment", Color::DarkGray),
      ("constant", Color::Cyan),
      ("constant.builtin", Color::Cyan),
      ("constructor", Color::Yellow),
      ("escape", Color::Magenta),
      ("function", Color::Blue),
      ("function.builtin", Color::Blue),
      ("function.macro", Color::Magenta),
      ("keyword", Color::Magenta),
      // `def`, `fun`, `fn` and friends: a keyword, not the function it
      // introduces.
      ("keyword.function", Color::Magenta),
      ("label", Color::Magenta),
      ("module", Color::Yellow),
      ("number", Color::Cyan),
      ("property", Color::Cyan),
      ("punctuation.special", Color::Magenta),
      ("string", Color::Green),
      ("string.special", Color::Magenta),
      ("tag", Color::Blue),
      ("type", Color::Yellow),
      ("type.builtin", Color::Yellow),
      ("variable.builtin", Color::Red),
      // A record field, as the neovim-flavoured queries name it.
      ("variable.member", Color::Cyan),
    ];
    Self(
      names
        .into_iter()
        .map(|(name, colour)| (name.to_string(), colour))
        .collect(),
    )
  }
}

/// The defaults, with what the file says laid over them: a name already there
/// keeps its place in the order ties are settled by, and a new one goes after
/// all of them, where it can only win a tie against another new one.
fn syntax<'de, D: Deserializer<'de>>(de: D) -> Result<Syntax, D::Error> {
  let given = BTreeMap::<String, Color>::deserialize(de)?;
  let mut syntax = Syntax::default();
  for (name, colour) in given {
    match syntax.0.iter_mut().find(|(known, _)| *known == name) {
      Some((_, known)) => *known = colour,
      None => syntax.0.push((name, colour)),
    }
  }
  Ok(syntax)
}

/// Read every file that is there, the nearer file winning key by key — within
/// a section, so a nearer file changing one colour of `[ui]` leaves the rest
/// of the further file's `[ui]` alone.
///
/// A file that makes no sense is a failure, as `config.toml` is: it is read
/// before the terminal is taken, where saying so costs nothing. `strict` is
/// for a file asked for by name, where not being there is worth saying too.
///
/// The defaults that depend on what the terminal can draw go in underneath
/// every file, as the furthest of them, so any file saying otherwise wins.
pub fn load(paths: &[PathBuf], strict: bool, colors: Colors) -> Result<Theme> {
  let (added, removed) = Colors::diff_backgrounds(colors.truecolor());
  let mut merged = toml::Table::new();
  overlay(
    &mut merged,
    toml::Table::from_iter([(
      "ui".to_string(),
      toml::Value::Table(toml::Table::from_iter([
        ("added-background".to_string(), toml::Value::String(added.to_string())),
        (
          "removed-background".to_string(),
          toml::Value::String(removed.to_string()),
        ),
      ])),
    )]),
  );
  for path in paths {
    let text = match std::fs::read_to_string(path) {
      Ok(text) => text,
      Err(err) if err.kind() == std::io::ErrorKind::NotFound && !strict => continue,
      Err(err) => return Err(err).with_context(|| format!("could not read {}", path.display())),
    };
    // Each file on its own first, so a mistake is placed by the line it is on
    // in the file it is in.
    toml::from_str::<Theme>(&text).with_context(|| format!("could not read {}", path.display()))?;
    overlay(&mut merged, toml::from_str(&text)?);
  }
  Ok(toml::Value::Table(merged).try_into()?)
}

/// `table` laid over `merged`, section by section.
fn overlay(merged: &mut toml::Table, table: toml::Table) {
  for (section, keys) in table {
    match (merged.get_mut(&section), keys) {
      (Some(toml::Value::Table(into)), toml::Value::Table(keys)) => into.extend(keys),
      (_, keys) => {
        merged.insert(section, keys);
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a_file_says_only_what_it_changes_in_any_way_ratatui_spells_a_colour() {
    let read: Theme = toml::from_str(
      r##"
        [ui]
        accent = "light-cyan"
        muted = "242"

        [markdown]
        link = "#5f87ff"

        [syntax]
        comment = "reset"
        operator = "bright yellow"
      "##,
    )
    .expect("a readable file");
    assert_eq!(read.ui.accent, Color::LightCyan);
    assert_eq!(read.ui.muted, Color::Indexed(242));
    assert_eq!(read.ui.error, Color::Red, "what it left out keeps its default");
    assert_eq!(read.markdown.link, Color::Rgb(0x5f, 0x87, 0xff));
    assert_eq!(read.markdown.heading, Color::Yellow);

    // A known name keeps its place, so ties are settled as they were; a new
    // one is added after them all.
    let defaults = Syntax::default().0;
    let at = |syntax: &Syntax, name: &str| syntax.0.iter().position(|(known, _)| known == name);
    assert_eq!(at(&read.syntax, "comment"), at(&Syntax::default(), "comment"));
    assert_eq!(read.syntax.0[at(&read.syntax, "comment").unwrap()].1, Color::Reset);
    assert_eq!(read.syntax.0.len(), defaults.len() + 1);
    assert_eq!(
      read.syntax.0.last(),
      Some(&("operator".to_string(), Color::LightYellow))
    );

    assert_eq!(toml::from_str::<Theme>("").expect("empty"), Theme::default());
  }

  #[test]
  fn a_key_or_a_colour_it_does_not_know_is_refused() {
    let err = toml::from_str::<Theme>("[ui]\nacent = \"red\"")
      .expect_err("an unknown key is refused")
      .to_string();
    assert!(err.contains("acent"), "which key it was: {err}");
    assert!(toml::from_str::<Theme>("[colours]\nred = \"red\"").is_err());
    assert!(toml::from_str::<Theme>("[ui]\naccent = \"chartreuse\"").is_err());
  }

  #[test]
  fn the_nearest_file_has_the_last_word_key_by_key_within_a_section() {
    let dir = std::env::temp_dir().join(format!("fa-theme-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory");
    let system = dir.join("system.toml");
    let user = dir.join("user.toml");
    std::fs::write(&system, "[ui]\naccent = \"blue\"\nerror = \"magenta\"").expect("a file");
    std::fs::write(&user, "[ui]\naccent = \"green\"").expect("a file");

    let missing = dir.join("nowhere.toml");
    let read = load(&[system, user.clone(), missing.clone()], false, Colors::Truecolor).expect("both files");
    assert_eq!(read.ui.accent, Color::Green, "read last, so it wins");
    assert_eq!(read.ui.error, Color::Magenta, "the rest of the section stays");
    // A file that is not there is only worth saying when it was asked for.
    assert!(load(&[missing], true, Colors::Truecolor).is_err());

    let broken = dir.join("broken.toml");
    std::fs::write(&broken, "[ui]\naccent = \"green\"\nerror = \"nope\"").expect("a file");
    let err = format!(
      "{:#}",
      load(&[user, broken], false, Colors::Truecolor).expect_err("a broken file")
    );
    assert!(
      err.contains("broken.toml") && err.contains("line 3"),
      "where it went wrong: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn the_diff_backgrounds_default_to_what_the_terminal_can_draw_and_a_file_still_wins() {
    let backgrounds = |theme: &Theme| (theme.ui.added_background, theme.ui.removed_background);
    let deep = load(&[], false, Colors::Truecolor).expect("no files");
    assert_eq!(
      backgrounds(&deep),
      (Color::Rgb(0x00, 0x28, 0x00), Color::Rgb(0x3f, 0x00, 0x01))
    );
    assert_eq!(deep, Theme::default(), "which is what a test draws with");
    let shallow = load(&[], false, Colors::Palette).expect("no files");
    assert_eq!(backgrounds(&shallow), (Color::Indexed(22), Color::Indexed(52)));
    assert_eq!(shallow.ui.accent, Color::Cyan, "and nothing else changes with it");

    // A colour a file gives is drawn as written, whatever the terminal is
    // thought to manage.
    let dir = std::env::temp_dir().join(format!("fa-theme-colors-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory");
    let file = dir.join("theme.toml");
    std::fs::write(&file, "[ui]\nadded-background = \"#102010\"").expect("a file");
    let read = load(&[file], false, Colors::Palette).expect("a file");
    assert_eq!(backgrounds(&read), (Color::Rgb(0x10, 0x20, 0x10), Color::Indexed(52)));
    let _ = std::fs::remove_dir_all(&dir);

    // COLORTERM only ever says yes.
    assert!(says_truecolor(Some("truecolor")) && says_truecolor(Some("24bit")));
    assert!(says_truecolor(Some("TrueColor")));
    assert!(!says_truecolor(Some("yes")) && !says_truecolor(Some("")) && !says_truecolor(None));
  }
}
