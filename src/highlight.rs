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

/// Gleam's function names, said again after its own query takes them back.
///
/// That query ends on a bare `(identifier) @variable`, which under the
/// last-match-wins rule wins over the `@function` captured for a name earlier
/// in the file — and `variable` is not a name we style, so every function in
/// the block came out the code's default colour. Appended rather than replacing
/// the query, so the rest of it stays whatever upstream makes it.
#[cfg(feature = "lang-gleam")]
const GLEAM_FUNCTIONS: &str = r#"
(function name: (identifier) @function)
(external_function name: (identifier) @function)
(function_call function: (identifier) @function)
"#;

/// The rule that tells SQL's numbers from its strings, in this engine's dialect.
///
/// `tree-sitter-sequel` calls every literal a string and then narrows that with
/// `(#match? @number "^[-+]?%d+$")` — a Lua pattern, where `%d` is a digit.
/// `#match?` here is a regex, in which `%d` is a literal per cent, so the
/// pattern never fires and numbers stay string-green.
#[cfg(feature = "lang-sql")]
const SQL_NUMBERS: &str = r#"
((literal) @number (#match? @number "^[-+]?[0-9]+$"))
((literal) @number (#match? @number "^[-+]?[0-9]*\\.[0-9]*$"))
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
    #[cfg(feature = "lang-c-sharp")]
    "csharp" | "c#" | "cs" => config!(
      "c-sharp",
      tree_sitter_c_sharp::LANGUAGE,
      tree_sitter_c_sharp::HIGHLIGHTS_QUERY
    ),
    #[cfg(feature = "lang-cmake")]
    "cmake" => config!(
      "cmake",
      tree_sitter_cmake::LANGUAGE,
      tree_sitter_cmake::HIGHLIGHTS_QUERY,
      tree_sitter_cmake::INJECTIONS_QUERY,
      ""
    ),
    #[cfg(feature = "lang-cpp")]
    "cpp" | "c++" | "cc" | "cxx" | "hpp" => {
      config!("cpp", tree_sitter_cpp::LANGUAGE, tree_sitter_cpp::HIGHLIGHT_QUERY)
    }
    #[cfg(feature = "lang-css")]
    "css" => config!("css", tree_sitter_css::LANGUAGE, tree_sitter_css::HIGHLIGHTS_QUERY),
    #[cfg(feature = "lang-dart")]
    "dart" => config!(
      "dart",
      tree_sitter_dart::LANGUAGE,
      tree_sitter_dart::HIGHLIGHTS_QUERY,
      "",
      tree_sitter_dart::LOCALS_QUERY
    ),
    #[cfg(feature = "lang-diff")]
    "diff" | "patch" => config!("diff", tree_sitter_diff::LANGUAGE, tree_sitter_diff::HIGHLIGHTS_QUERY),
    #[cfg(feature = "lang-elixir")]
    "elixir" | "ex" | "exs" => config!(
      "elixir",
      tree_sitter_elixir::LANGUAGE,
      tree_sitter_elixir::HIGHLIGHTS_QUERY,
      tree_sitter_elixir::INJECTIONS_QUERY,
      ""
    ),
    #[cfg(feature = "lang-erlang")]
    "erlang" | "erl" => config!(
      "erlang",
      tree_sitter_erlang::LANGUAGE,
      tree_sitter_erlang::HIGHLIGHTS_QUERY
    ),
    #[cfg(feature = "lang-fortran")]
    "fortran" | "f90" | "f95" => config!(
      "fortran",
      tree_sitter_fortran::LANGUAGE,
      tree_sitter_fortran::HIGHLIGHTS_QUERY
    ),
    #[cfg(feature = "lang-gleam")]
    "gleam" => config!(
      "gleam",
      tree_sitter_gleam::LANGUAGE,
      [tree_sitter_gleam::HIGHLIGHT_QUERY, GLEAM_FUNCTIONS].concat(),
      "",
      tree_sitter_gleam::LOCALS_QUERY
    ),
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
    #[cfg(feature = "lang-ini")]
    "ini" | "cfg" => config!("ini", tree_sitter_ini::LANGUAGE, tree_sitter_ini::HIGHLIGHTS_QUERY),
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
    // Reached through JavaScript's injections, which hand a comment to this
    // grammar; a `jsdoc` fence is nobody's habit but costs nothing to accept.
    #[cfg(feature = "lang-jsdoc")]
    "jsdoc" => config!(
      "jsdoc",
      tree_sitter_jsdoc::LANGUAGE,
      tree_sitter_jsdoc::HIGHLIGHTS_QUERY
    ),
    #[cfg(feature = "lang-json")]
    "json" | "jsonc" => config!("json", tree_sitter_json::LANGUAGE, tree_sitter_json::HIGHLIGHTS_QUERY),
    #[cfg(feature = "lang-kotlin")]
    "kotlin" | "kt" | "kts" => config!(
      "kotlin",
      tree_sitter_kotlin_sg::LANGUAGE,
      tree_sitter_kotlin_sg::HIGHLIGHTS_QUERY
    ),
    #[cfg(feature = "lang-lua")]
    "lua" => config!(
      "lua",
      tree_sitter_lua::LANGUAGE,
      tree_sitter_lua::HIGHLIGHTS_QUERY,
      tree_sitter_lua::INJECTIONS_QUERY,
      tree_sitter_lua::LOCALS_QUERY
    ),
    #[cfg(feature = "lang-make")]
    "make" | "makefile" | "mk" => config!("make", tree_sitter_make::LANGUAGE, tree_sitter_make::HIGHLIGHTS_QUERY),
    #[cfg(feature = "lang-nix")]
    "nix" => config!(
      "nix",
      tree_sitter_nix::LANGUAGE,
      tree_sitter_nix::HIGHLIGHTS_QUERY,
      tree_sitter_nix::INJECTIONS_QUERY,
      ""
    ),
    #[cfg(feature = "lang-ocaml")]
    "ocaml" | "ml" => config!(
      "ocaml",
      tree_sitter_ocaml::LANGUAGE_OCAML,
      tree_sitter_ocaml::HIGHLIGHTS_QUERY,
      "",
      tree_sitter_ocaml::LOCALS_QUERY
    ),
    // An `.mli` is a different grammar in the same crate, sharing the query.
    #[cfg(feature = "lang-ocaml")]
    "ocaml_interface" | "mli" => config!(
      "ocaml_interface",
      tree_sitter_ocaml::LANGUAGE_OCAML_INTERFACE,
      tree_sitter_ocaml::HIGHLIGHTS_QUERY,
      "",
      tree_sitter_ocaml::LOCALS_QUERY
    ),
    #[cfg(feature = "lang-php")]
    "php" => config!(
      "php",
      tree_sitter_php::LANGUAGE_PHP,
      tree_sitter_php::HIGHLIGHTS_QUERY,
      tree_sitter_php::INJECTIONS_QUERY,
      ""
    ),
    #[cfg(feature = "lang-powershell")]
    "powershell" | "pwsh" | "ps1" => config!(
      "powershell",
      tree_sitter_powershell::LANGUAGE,
      tree_sitter_powershell::HIGHLIGHTS_QUERY
    ),
    #[cfg(feature = "lang-python")]
    "python" | "py" => config!(
      "python",
      tree_sitter_python::LANGUAGE,
      tree_sitter_python::HIGHLIGHTS_QUERY
    ),
    #[cfg(feature = "lang-r")]
    "r" | "rscript" => config!(
      "r",
      tree_sitter_r::LANGUAGE,
      tree_sitter_r::HIGHLIGHTS_QUERY,
      "",
      tree_sitter_r::LOCALS_QUERY
    ),
    // JavaScript injects this one into every regex literal.
    #[cfg(feature = "lang-regex")]
    "regex" | "regexp" => config!(
      "regex",
      tree_sitter_regex::LANGUAGE,
      tree_sitter_regex::HIGHLIGHTS_QUERY
    ),
    #[cfg(feature = "lang-ruby")]
    "ruby" | "rb" => config!(
      "ruby",
      tree_sitter_ruby::LANGUAGE,
      tree_sitter_ruby::HIGHLIGHTS_QUERY,
      "",
      tree_sitter_ruby::LOCALS_QUERY
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
    // Dialect-agnostic: the grammar takes MySQL, Postgres and SQLite alike.
    #[cfg(feature = "lang-sql")]
    "sql" | "mysql" | "postgresql" | "sqlite" => config!(
      "sql",
      tree_sitter_sequel::LANGUAGE,
      [tree_sitter_sequel::HIGHLIGHTS_QUERY, SQL_NUMBERS].concat()
    ),
    #[cfg(feature = "lang-swift")]
    "swift" => config!(
      "swift",
      tree_sitter_swift::LANGUAGE,
      tree_sitter_swift::HIGHLIGHTS_QUERY,
      tree_sitter_swift::INJECTIONS_QUERY,
      tree_sitter_swift::LOCALS_QUERY
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
    // The crate carries a DTD grammar as well, which no fence asks for.
    #[cfg(feature = "lang-xml")]
    "xml" | "svg" | "xsd" => config!(
      "xml",
      tree_sitter_xml::LANGUAGE_XML,
      tree_sitter_xml::XML_HIGHLIGHT_QUERY
    ),
    #[cfg(feature = "lang-yaml")]
    "yaml" | "yml" => config!("yaml", tree_sitter_yaml::LANGUAGE, tree_sitter_yaml::HIGHLIGHTS_QUERY),
    #[cfg(feature = "lang-zig")]
    "zig" => config!(
      "zig",
      tree_sitter_zig::LANGUAGE,
      tree_sitter_zig::HIGHLIGHTS_QUERY,
      tree_sitter_zig::INJECTIONS_QUERY,
      ""
    ),
    _ => None,
  }
}

/// A query with neovim's spell-checking captures taken out.
///
/// Queries written for neovim mark where prose lives by capturing a node twice
/// — `(comment) @comment @spell`. Of several captures on one node the last one
/// wins here, and `@spell` is not a name we style, so left in it silences the
/// `@comment` in front of it and comments come out plain. Removing the name
/// leaves the pattern itself, and its other captures, exactly as they were.
#[cfg(feature = "syntax")]
fn without_spell(query: &str) -> String {
  // `@nospell` first: it ends in the shorter name.
  query.replace("@nospell", "").replace("@spell", "")
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
    &without_spell(highlights.as_ref()),
    injections.as_ref(),
    locals.as_ref(),
  )
  .ok()?;
  // A pattern can be guarded by a predicate this query engine has never heard
  // of — neovim's `#lua-match?`, nearly always — and an unknown predicate is
  // ignored rather than refused, so the pattern fires everywhere it was meant
  // to be held back. Zig's query is the clearest case: its rule for capitalised
  // identifiers, unguarded, makes a type of every identifier in the block.
  // Dropping those patterns loses a little colour and keeps the rest honest.
  for pattern in 0..config.query.pattern_count() {
    if !config.query.general_predicates(pattern).is_empty() {
      config.query.disable_pattern(pattern);
    }
  }
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
      #[cfg(feature = "lang-c-sharp")]
      "csharp",
      #[cfg(feature = "lang-cmake")]
      "cmake",
      #[cfg(feature = "lang-cpp")]
      "cpp",
      #[cfg(feature = "lang-css")]
      "css",
      #[cfg(feature = "lang-dart")]
      "dart",
      #[cfg(feature = "lang-diff")]
      "diff",
      #[cfg(feature = "lang-elixir")]
      "elixir",
      #[cfg(feature = "lang-erlang")]
      "erlang",
      #[cfg(feature = "lang-fortran")]
      "fortran",
      #[cfg(feature = "lang-gleam")]
      "gleam",
      #[cfg(feature = "lang-go")]
      "go",
      #[cfg(feature = "lang-haskell")]
      "haskell",
      #[cfg(feature = "lang-html")]
      "html",
      #[cfg(feature = "lang-ini")]
      "ini",
      #[cfg(feature = "lang-java")]
      "java",
      #[cfg(feature = "lang-javascript")]
      "javascript",
      #[cfg(feature = "lang-jsdoc")]
      "jsdoc",
      #[cfg(feature = "lang-json")]
      "json",
      #[cfg(feature = "lang-kotlin")]
      "kotlin",
      #[cfg(feature = "lang-lua")]
      "lua",
      #[cfg(feature = "lang-make")]
      "make",
      #[cfg(feature = "lang-nix")]
      "nix",
      #[cfg(feature = "lang-ocaml")]
      "ocaml",
      #[cfg(feature = "lang-ocaml")]
      "mli",
      #[cfg(feature = "lang-php")]
      "php",
      #[cfg(feature = "lang-powershell")]
      "powershell",
      #[cfg(feature = "lang-python")]
      "python",
      #[cfg(feature = "lang-r")]
      "r",
      #[cfg(feature = "lang-regex")]
      "regex",
      #[cfg(feature = "lang-ruby")]
      "ruby",
      #[cfg(feature = "lang-rust")]
      "rust",
      #[cfg(feature = "lang-scala")]
      "scala",
      #[cfg(feature = "lang-sql")]
      "sql",
      #[cfg(feature = "lang-swift")]
      "swift",
      #[cfg(feature = "lang-toml")]
      "toml",
      #[cfg(feature = "lang-typescript")]
      "typescript",
      #[cfg(feature = "lang-typescript")]
      "tsx",
      #[cfg(feature = "lang-xml")]
      "xml",
      #[cfg(feature = "lang-yaml")]
      "yaml",
      #[cfg(feature = "lang-zig")]
      "zig",
    ];
    for lang in enabled {
      assert!(grammar(lang).is_some(), "{lang} queries did not compile");
    }
  }

  #[test]
  #[cfg(feature = "lang-swift")]
  fn a_spell_capture_does_not_silence_the_comment_in_front_of_it() {
    // Swift's query says `(comment) @comment @spell`, and of two captures on
    // one node the last one wins: with `@spell` left in, comments went plain.
    let lines = highlight("swift", "// note\nfunc f() {}").unwrap();
    assert_eq!(style_of(&lines, "// note").unwrap().fg, Some(Color::DarkGray));
  }

  #[test]
  #[cfg(feature = "lang-zig")]
  fn a_pattern_we_cannot_check_does_not_fire() {
    // Zig's `(identifier) @type` is guarded by a `#lua-match?` on a leading
    // capital. The guard means nothing to this engine, so the pattern is
    // dropped instead — otherwise every identifier in the block is a type.
    let lines = highlight("zig", "pub fn main() void { const x = other; }").unwrap();
    assert_eq!(style_of(&lines, "void").unwrap().fg, Some(Color::Yellow));
    // Nothing styled `other`, so it comes back inside a plain run rather than
    // a span of its own: what matters is that no span calling it a type does.
    assert!(
      !lines
        .iter()
        .flatten()
        .any(|s| s.content.contains("other") && s.style.fg == Some(Color::Yellow)),
      "{lines:?}"
    );
  }

  #[test]
  #[cfg(feature = "lang-gleam")]
  fn gleam_function_names_outlive_the_catch_all() {
    // The grammar's query ends on `(identifier) @variable`, which would take
    // back the `@function` it gave the name two dozen patterns earlier.
    let lines = highlight("gleam", "pub fn add(a: Int) -> Int { a + 1 }").unwrap();
    assert_eq!(style_of(&lines, "add").unwrap().fg, Some(Color::Blue));
  }

  #[test]
  #[cfg(feature = "lang-sql")]
  fn sql_numbers_are_not_strings() {
    // Every literal is a string to the grammar's query until a `#match?` says
    // otherwise, and the one it ships is a Lua pattern that never matches.
    let lines = highlight("sql", "SELECT name FROM t WHERE id = 1;").unwrap();
    assert_eq!(style_of(&lines, "1").unwrap().fg, Some(Color::Cyan));
  }

  #[test]
  #[cfg(all(feature = "lang-html", feature = "lang-javascript"))]
  fn an_injected_language_is_highlighted_too() {
    // The `<script>` body is JavaScript, reached through the injection query.
    let lines = highlight("html", "<script>let x = 1;</script>").unwrap();
    assert_eq!(style_of(&lines, "let").unwrap().fg, Some(Color::Magenta));
  }
}
