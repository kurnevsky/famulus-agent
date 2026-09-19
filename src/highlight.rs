//! Syntax highlighting for fenced code blocks, via tree-sitter.
//!
//! Every grammar is a C parser compiled into the binary, so each language is
//! its own cargo feature and the whole module folds away when none is on —
//! `highlight` then returns `None` and a code block keeps the single colour it
//! had before. Nothing here fails loudly: an unknown fence, a grammar whose
//! queries will not compile, or a parse that errors all fall back the same way.
//!
//! Colours come from the terminal's own palette, like the rest of the renderer,
//! so they keep working against a light background. Tokens no rule matched are
//! left unstyled rather than given a colour of their own — the code's default
//! foreground is the one colour guaranteed to be readable.

use ratatui::text::Span;

#[cfg(feature = "syntax")]
use std::sync::OnceLock;

#[cfg(feature = "syntax")]
use ratatui::style::{Color, Modifier, Style};
#[cfg(feature = "syntax")]
use tree_sitter::Language;
#[cfg(feature = "syntax")]
use tree_sitter_highlight::{HighlightConfiguration, HighlightEvent, Highlighter};

/// Capture names we recognise, each with the style it draws in.
///
/// A capture matches the entry whose dotted parts it all contains, the longest
/// such entry winning and ties going to whichever is listed first. So
/// `@function.method` lands on `function`, and `@keyword.function` — which
/// matches both `keyword` and `function` by one part each — needs an entry of
/// its own to stop the tie deciding it. Listing a name is also what makes it
/// highlight at all: a capture with no entry here produces no event, and its
/// text stays as the code's default foreground.
#[cfg(feature = "syntax")]
const THEME: &[(&str, Style)] = &[
  ("attribute", Style::new().fg(Color::Magenta)),
  ("boolean", Style::new().fg(Color::Cyan)),
  ("character", Style::new().fg(Color::Green)),
  (
    "comment",
    Style::new().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
  ),
  ("constant", Style::new().fg(Color::Cyan)),
  (
    "constant.builtin",
    Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
  ),
  ("constructor", Style::new().fg(Color::Yellow)),
  ("escape", Style::new().fg(Color::Magenta)),
  ("function", Style::new().fg(Color::Blue)),
  (
    "function.builtin",
    Style::new().fg(Color::Blue).add_modifier(Modifier::BOLD),
  ),
  ("function.macro", Style::new().fg(Color::Magenta)),
  ("keyword", Style::new().fg(Color::Magenta)),
  // `def`, `fun`, `fn` and friends: a keyword, not the function it introduces.
  ("keyword.function", Style::new().fg(Color::Magenta)),
  ("label", Style::new().fg(Color::Magenta)),
  ("module", Style::new().fg(Color::Yellow)),
  ("number", Style::new().fg(Color::Cyan)),
  ("property", Style::new().fg(Color::Cyan)),
  ("punctuation.special", Style::new().fg(Color::Magenta)),
  ("string", Style::new().fg(Color::Green)),
  ("string.special", Style::new().fg(Color::Magenta)),
  ("tag", Style::new().fg(Color::Blue)),
  ("type", Style::new().fg(Color::Yellow)),
  ("type.builtin", Style::new().fg(Color::Yellow)),
  ("variable.builtin", Style::new().fg(Color::Red)),
  // A record field, as the neovim-flavoured queries name it.
  ("variable.member", Style::new().fg(Color::Cyan)),
];

/// Above this many bytes a block is left plain. A transcript's code comes from
/// the model and is small; a pasted file is not worth parsing on every frame.
#[cfg(feature = "syntax")]
const MAX_SOURCE: usize = 256 * 1024;

/// Haskell highlights, written here instead of taken from the grammar.
///
/// `tree-sitter-haskell` ships a neovim query, where a pattern's priority is
/// its specificity. Here the last pattern to match a node wins, and that query
/// ends with a `(variable) @type` meant for type variables — which under this
/// rule paints every identifier in the block yellow. Ordering is the whole
/// point of what follows: general first, the names that introduce a binding
/// last.
#[cfg(feature = "lang-haskell")]
const HASKELL_HIGHLIGHTS: &str = r#"
(comment) @comment
(haddock) @comment
(pragma) @attribute

[
  "module" "where" "import" "qualified" "as" "hiding"
  "data" "newtype" "type" "class" "instance" "deriving" "family" "role"
  "let" "in" "if" "then" "else" "case" "of" "do" "mdo" "rec"
  "forall" "foreign" "export" "pattern" "default"
  "infix" "infixl" "infixr" "via" "stock" "anyclass"
] @keyword

(integer) @number
(float) @number
(char) @character
(string) @string
(quasiquote (quoter) @function.macro)

(name) @type
(constructor) @constructor
(module) @module
(field_name) @variable.member

; The name a declaration binds, and the head of an application: everything
; else that is a bare `(variable)` is left unstyled, which is what keeps the
; block from turning into one colour.
(signature name: (variable) @function)
(function name: (variable) @function)
(bind name: (variable) @function)
(apply function: (variable) @function)
"#;

/// Spans for each line of `source`, or `None` when `lang` is not one we can
/// parse — in which case the caller draws the block in its fallback colour.
///
/// Lines are indexed as `source.split('\n')` yields them, so line `i` of the
/// input is entry `i` of the result. Highlighting is best-effort: unfinished
/// code mid-stream parses into error nodes and still highlights whatever it
/// could make sense of.
#[cfg(feature = "syntax")]
pub fn highlight(lang: &str, source: &str) -> Option<Vec<Vec<Span<'static>>>> {
  if source.len() > MAX_SOURCE {
    return None;
  }
  let config = grammar(&lang.trim().to_lowercase())?;
  let mut highlighter = Highlighter::new();
  // The same lookup answers the injection callback, so a nested grammar is
  // found exactly when its own feature is on.
  let events = highlighter
    .highlight(config, source.as_bytes(), None, None, |name: &str| grammar(name))
    .ok()?;

  let mut lines: Vec<Vec<Span<'static>>> = vec![Vec::new()];
  // Innermost capture wins, which is how nesting is meant to read: the `"` of
  // a string inside a macro is a string, not a macro.
  let mut open: Vec<Style> = Vec::new();
  for event in events {
    match event.ok()? {
      HighlightEvent::HighlightStart(highlight) => {
        open.push(THEME.get(highlight.0).map(|(_, style)| *style).unwrap_or_default());
      }
      HighlightEvent::HighlightEnd => {
        open.pop();
      }
      HighlightEvent::Source { start, end } => {
        let style = open.last().copied().unwrap_or_default();
        // Byte offsets from the parser, which does not split a code point.
        let text = source.get(start..end)?;
        for (i, chunk) in text.split('\n').enumerate() {
          if i > 0 {
            lines.push(Vec::new());
          }
          if !chunk.is_empty() {
            lines.last_mut()?.push(Span::styled(chunk.to_string(), style));
          }
        }
      }
    }
  }
  Some(lines)
}

#[cfg(not(feature = "syntax"))]
pub fn highlight(_lang: &str, _source: &str) -> Option<Vec<Vec<Span<'static>>>> {
  None
}

/// Builds a grammar's configuration once and hands out the shared reference.
///
/// The queries are compiled here, which is far too slow to repeat per frame —
/// hence the `static` per language. `tree_sitter_highlight` also wants a
/// `&'static` back from the injection callback, which a cache of this shape is
/// the simplest way to give it.
#[cfg(feature = "syntax")]
macro_rules! config {
  ($name:literal, $language:expr, $highlights:expr) => {
    config!($name, $language, $highlights, "", "")
  };
  ($name:literal, $language:expr, $highlights:expr, $injections:expr, $locals:expr) => {{
    static CONFIG: OnceLock<Option<HighlightConfiguration>> = OnceLock::new();
    CONFIG
      .get_or_init(|| build($name, $language.into(), $highlights, $injections, $locals))
      .as_ref()
  }};
}

/// The grammar a fence's info string asks for, by name or by the extension
/// people write instead.
///
/// The same lookup answers tree-sitter's injection callback, so a `<script>` in
/// HTML or a `sql!` in Rust picks up the nested grammar when its feature is on
/// and is left plain when it is not.
#[cfg(feature = "syntax")]
fn grammar(lang: &str) -> Option<&'static HighlightConfiguration> {
  match lang {
    #[cfg(feature = "lang-bash")]
    "bash" | "sh" | "shell" | "zsh" | "console" => {
      config!("bash", tree_sitter_bash::LANGUAGE, tree_sitter_bash::HIGHLIGHT_QUERY)
    }
    #[cfg(feature = "lang-c")]
    "c" | "h" => config!("c", tree_sitter_c::LANGUAGE, tree_sitter_c::HIGHLIGHT_QUERY),
    #[cfg(feature = "lang-cpp")]
    "cpp" | "c++" | "cc" | "cxx" | "hpp" => {
      config!("cpp", tree_sitter_cpp::LANGUAGE, tree_sitter_cpp::HIGHLIGHT_QUERY)
    }
    #[cfg(feature = "lang-css")]
    "css" => config!("css", tree_sitter_css::LANGUAGE, tree_sitter_css::HIGHLIGHTS_QUERY),
    #[cfg(feature = "lang-go")]
    "go" | "golang" => config!("go", tree_sitter_go::LANGUAGE, tree_sitter_go::HIGHLIGHTS_QUERY),
    #[cfg(feature = "lang-haskell")]
    "haskell" | "hs" => config!(
      "haskell",
      tree_sitter_haskell::LANGUAGE,
      HASKELL_HIGHLIGHTS,
      tree_sitter_haskell::INJECTIONS_QUERY,
      ""
    ),
    #[cfg(feature = "lang-html")]
    "html" | "htm" => config!(
      "html",
      tree_sitter_html::LANGUAGE,
      tree_sitter_html::HIGHLIGHTS_QUERY,
      tree_sitter_html::INJECTIONS_QUERY,
      ""
    ),
    #[cfg(feature = "lang-java")]
    "java" => config!("java", tree_sitter_java::LANGUAGE, tree_sitter_java::HIGHLIGHTS_QUERY),
    #[cfg(feature = "lang-javascript")]
    "javascript" | "js" | "jsx" | "mjs" | "cjs" => config!(
      "javascript",
      tree_sitter_javascript::LANGUAGE,
      // The JSX rules live in a query of their own, and the grammar parses JSX
      // whether or not the fence said so.
      [
        tree_sitter_javascript::HIGHLIGHT_QUERY,
        tree_sitter_javascript::JSX_HIGHLIGHT_QUERY
      ]
      .concat(),
      tree_sitter_javascript::INJECTIONS_QUERY,
      tree_sitter_javascript::LOCALS_QUERY
    ),
    #[cfg(feature = "lang-json")]
    "json" | "jsonc" => config!("json", tree_sitter_json::LANGUAGE, tree_sitter_json::HIGHLIGHTS_QUERY),
    #[cfg(feature = "lang-python")]
    "python" | "py" => config!(
      "python",
      tree_sitter_python::LANGUAGE,
      tree_sitter_python::HIGHLIGHTS_QUERY
    ),
    #[cfg(feature = "lang-rust")]
    "rust" | "rs" => config!(
      "rust",
      tree_sitter_rust::LANGUAGE,
      tree_sitter_rust::HIGHLIGHTS_QUERY,
      tree_sitter_rust::INJECTIONS_QUERY,
      ""
    ),
    #[cfg(feature = "lang-scala")]
    "scala" | "sc" | "sbt" => config!(
      "scala",
      tree_sitter_scala::LANGUAGE,
      tree_sitter_scala::HIGHLIGHTS_QUERY,
      "",
      tree_sitter_scala::LOCALS_QUERY
    ),
    #[cfg(feature = "lang-toml")]
    "toml" => config!(
      "toml",
      tree_sitter_toml_ng::LANGUAGE,
      tree_sitter_toml_ng::HIGHLIGHTS_QUERY
    ),
    // TypeScript's own query only adds the type syntax; the rest of the
    // language is JavaScript's, and TSX needs the JSX rules on top. The two
    // grammars are separate because TSX reads `<T>` as an element.
    #[cfg(feature = "lang-typescript")]
    "typescript" | "ts" | "mts" | "cts" => config!(
      "typescript",
      tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
      [
        tree_sitter_javascript::HIGHLIGHT_QUERY,
        tree_sitter_typescript::HIGHLIGHTS_QUERY
      ]
      .concat(),
      tree_sitter_javascript::INJECTIONS_QUERY,
      tree_sitter_typescript::LOCALS_QUERY
    ),
    #[cfg(feature = "lang-typescript")]
    "tsx" => config!(
      "tsx",
      tree_sitter_typescript::LANGUAGE_TSX,
      [
        tree_sitter_javascript::HIGHLIGHT_QUERY,
        tree_sitter_javascript::JSX_HIGHLIGHT_QUERY,
        tree_sitter_typescript::HIGHLIGHTS_QUERY
      ]
      .concat(),
      tree_sitter_javascript::INJECTIONS_QUERY,
      tree_sitter_typescript::LOCALS_QUERY
    ),
    #[cfg(feature = "lang-yaml")]
    "yaml" | "yml" => config!("yaml", tree_sitter_yaml::LANGUAGE, tree_sitter_yaml::HIGHLIGHTS_QUERY),
    _ => None,
  }
}

/// Compiles a grammar's queries, or gives up on it for good.
///
/// A query that will not compile against its grammar — a version skew between
/// two crates, say — is a build-time mistake we cannot fix at runtime, so the
/// language is simply dropped rather than taking the transcript down with it.
#[cfg(feature = "syntax")]
fn build(
  name: &str,
  language: Language,
  highlights: impl AsRef<str>,
  injections: impl AsRef<str>,
  locals: impl AsRef<str>,
) -> Option<HighlightConfiguration> {
  let mut config = HighlightConfiguration::new(
    language,
    name,
    highlights.as_ref(),
    injections.as_ref(),
    locals.as_ref(),
  )
  .ok()?;
  let names: Vec<&str> = THEME.iter().map(|(name, _)| *name).collect();
  config.configure(&names);
  Some(config)
}

#[cfg(all(test, feature = "syntax"))]
mod tests {
  use super::*;

  /// The text of a line, for asserting the source survived round-tripping.
  fn plain(line: &[Span<'static>]) -> String {
    line.iter().map(|s| s.content.as_ref()).collect()
  }

  /// The style the first span whose text is `needle` was given.
  fn style_of(lines: &[Vec<Span<'static>>], needle: &str) -> Option<Style> {
    lines
      .iter()
      .flatten()
      .find(|span| span.content == needle)
      .map(|span| span.style)
  }

  #[test]
  fn an_unknown_language_is_not_highlighted() {
    assert!(highlight("", "fn main() {}").is_none());
    assert!(highlight("brainfuck", "+++").is_none());
  }

  #[test]
  #[cfg(feature = "lang-rust")]
  fn keywords_strings_and_comments_get_their_colours() {
    let lines = highlight("rust", "// note\nfn main() { let s = \"hi\"; }").unwrap();
    assert_eq!(style_of(&lines, "// note").unwrap().fg, Some(Color::DarkGray));
    assert_eq!(style_of(&lines, "fn").unwrap().fg, Some(Color::Magenta));
    assert_eq!(style_of(&lines, "main").unwrap().fg, Some(Color::Blue));
    assert_eq!(style_of(&lines, "\"hi\"").unwrap().fg, Some(Color::Green));
  }

  #[test]
  #[cfg(feature = "lang-haskell")]
  fn haskell_variables_are_not_all_types() {
    // What the grammar's own query would do here, and the reason we do not
    // use it: its trailing `(variable) @type` wins under this precedence and
    // every identifier comes out yellow.
    let lines = highlight("haskell", "norm x y = sqrt (x * x + y * y)").unwrap();
    assert_eq!(style_of(&lines, "norm").unwrap().fg, Some(Color::Blue));
    assert_eq!(style_of(&lines, "sqrt").unwrap().fg, Some(Color::Blue));
    // The arguments are bare variables, and nothing in this line is a type.
    assert!(
      !lines.iter().flatten().any(|s| s.style.fg == Some(Color::Yellow)),
      "{lines:?}"
    );
  }

  #[test]
  #[cfg(feature = "lang-scala")]
  fn scala_def_is_a_keyword_not_a_function() {
    // `@keyword.function` matches both `keyword` and `function` by one part,
    // and without its own entry the tie would colour `def` like a call.
    let lines = highlight("scala", "object D { def f: Int = 1 }").unwrap();
    assert_eq!(style_of(&lines, "def").unwrap().fg, Some(Color::Magenta));
  }

  #[test]
  #[cfg(feature = "lang-rust")]
  fn lines_keep_their_text_and_their_count() {
    // Nothing may be dropped or reordered: the renderer indexes these by the
    // source's own line numbers.
    let source = "fn f() {\n    let x = 1;\n\n    x\n}";
    let lines = highlight("rust", source).unwrap();
    let text: Vec<String> = lines.iter().map(|l| plain(l)).collect();
    assert_eq!(text, source.split('\n').collect::<Vec<_>>());
  }

  #[test]
  #[cfg(feature = "lang-rust")]
  fn an_alias_finds_the_same_grammar() {
    assert_eq!(
      highlight("rs", "fn f() {}").map(|l| l.len()),
      highlight("RUST", "fn f() {}").map(|l| l.len())
    );
  }

  #[test]
  #[cfg(feature = "lang-rust")]
  fn unfinished_code_still_highlights() {
    // Mid-stream a block is cut wherever the token happened to stop.
    let lines = highlight("rust", "fn main() { let s = \"unter").unwrap();
    assert_eq!(style_of(&lines, "fn").unwrap().fg, Some(Color::Magenta));
  }

  #[test]
  fn every_enabled_grammar_compiles_its_queries() {
    // A query that will not compile against its grammar — two crates out of
    // step, most likely — is silent at runtime: the language simply stops
    // highlighting. This is the only place that says so.
    let enabled: &[&str] = &[
      #[cfg(feature = "lang-bash")]
      "bash",
      #[cfg(feature = "lang-c")]
      "c",
      #[cfg(feature = "lang-cpp")]
      "cpp",
      #[cfg(feature = "lang-css")]
      "css",
      #[cfg(feature = "lang-go")]
      "go",
      #[cfg(feature = "lang-haskell")]
      "haskell",
      #[cfg(feature = "lang-html")]
      "html",
      #[cfg(feature = "lang-java")]
      "java",
      #[cfg(feature = "lang-javascript")]
      "javascript",
      #[cfg(feature = "lang-json")]
      "json",
      #[cfg(feature = "lang-python")]
      "python",
      #[cfg(feature = "lang-rust")]
      "rust",
      #[cfg(feature = "lang-scala")]
      "scala",
      #[cfg(feature = "lang-toml")]
      "toml",
      #[cfg(feature = "lang-typescript")]
      "typescript",
      #[cfg(feature = "lang-typescript")]
      "tsx",
      #[cfg(feature = "lang-yaml")]
      "yaml",
    ];
    for lang in enabled {
      assert!(grammar(lang).is_some(), "{lang} queries did not compile");
    }
  }

  #[test]
  #[cfg(all(feature = "lang-html", feature = "lang-javascript"))]
  fn an_injected_language_is_highlighted_too() {
    // The `<script>` body is JavaScript, reached through the injection query.
    let lines = highlight("html", "<script>let x = 1;</script>").unwrap();
    assert_eq!(style_of(&lines, "let").unwrap().fg, Some(Color::Magenta));
  }
}
