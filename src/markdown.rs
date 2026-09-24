//! Markdown to ratatui lines.
//!
//! Every block is wrapped to the width it is handed, and nested structures
//! recurse with the width their prefix leaves over: a list item gets
//! `width - marker`, a blockquote `width - 2`. That is what keeps a wrapped
//! bullet hanging under its text rather than under its marker, and what keeps
//! `│ ` on every line of a quote — neither survives if wrapping is left to
//! `Paragraph::wrap`, which knows nothing about the structure above it.

use std::borrow::Cow;
use std::collections::HashMap;

// `::markdown` is the crate; this module shares its name.
use ::markdown::ParseOptions;
use ::markdown::mdast::{AlignKind, FootnoteDefinition, List, Node};
use mdstitch::{StitchOptions, stitch};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Indent for the body of a fenced code block.
const CODE_INDENT: &str = "  ";
/// Prefix on every line of a blockquote.
const QUOTE_PREFIX: &str = "│ ";
/// A rule is capped rather than left to span a wide terminal.
const HR_MAX: usize = 80;
/// Nested lists step in by this much per level.
const LIST_INDENT: usize = 4;

/// Colours come from the terminal's own palette, so they keep working
/// against a light background.
mod style {
  use super::*;

  pub fn heading(depth: u8) -> Style {
    let style = Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD);
    if depth == 1 {
      style.add_modifier(Modifier::UNDERLINED)
    } else {
      style
    }
  }

  pub const CODE_BLOCK: Style = Style::new().fg(Color::Green);
  /// Everything drawn around the text rather than in it: a code block's
  /// fences, a quote's bar, a rule, a table's lines.
  pub const BORDER: Style = Style::new().fg(Color::DarkGray);
  pub const INLINE_CODE: Style = Style::new().fg(Color::Cyan);
  pub const LINK: Style = Style::new().fg(Color::Blue).add_modifier(Modifier::UNDERLINED);
  pub const DIM: Style = Style::new().add_modifier(Modifier::DIM);
  pub const QUOTE: Style = Style::new().fg(Color::DarkGray).add_modifier(Modifier::ITALIC);
  pub const BULLET: Style = Style::new().fg(Color::Green);
  pub const FOOTNOTE: Style = Style::new().fg(Color::Blue);
}

/// Renders `md` into lines that each fit `width` columns.
///
/// `streaming` marks text that is still arriving, whose last token may be cut
/// in half. Only then is the markdown completed before parsing — a finished
/// message is rendered exactly as the GFM spec reads it, so an unpaired `~~`
/// or `**` stays literal instead of striking or bolding the rest of the line.
///
/// Never fails: markdown that will not parse falls back to its own source
/// text, which is what the caller would have shown anyway.
pub fn render(md: &str, width: u16, streaming: bool) -> Vec<Line<'static>> {
  let width = (width as usize).max(1);
  // Mid-stream, `**bold` would render its asterisks until the closing pair
  // arrives. mdstitch closes emphasis, code, links and math, and borrows
  // rather than allocates when there is nothing open.
  let stitched = if streaming {
    stitch(md, &StitchOptions::default())
  } else {
    Cow::Borrowed(md)
  };
  // Tabs reach the parser untouched: CommonMark counts one as advancing to
  // the next four-column stop, which is what makes `\tcode` an indented code
  // block. They are expanded only on the way out, for display.
  let Ok(root) = ::markdown::to_mdast(&stitched, &ParseOptions::gfm()) else {
    return stitched.lines().map(|l| Line::raw(expand_tabs(l))).collect();
  };
  let refs = Refs::collect(&root);
  let mut out = Vec::new();
  blocks(children(&root), width, &refs, &mut out);
  refs.render_footnotes(width, &mut out);
  // A trailing blank line would push the transcript's own spacing around.
  trim_blank(&mut out);
  out
}

/// Word-wraps a line that is already drawn the way it will be shown, for the
/// transcript to fold its rows itself instead of leaving them to a
/// `Paragraph` — which wraps as it draws and keeps to itself where each line
/// ended up, where the mouse needs to be told what row it is pointing at.
///
/// A line that fits comes back as it is, which is nearly all of them: every
/// renderer here wraps to the width as it goes, and only text shown as it was
/// written — a prompt, an error, an unfolded block — can still be too long.
///
/// The indent a line starts with is part of it and is kept; the space a break
/// falls on belongs to neither side and is dropped, as `wrap` drops it.
pub fn wrap_line(line: Line<'static>, width: u16) -> Vec<Line<'static>> {
  fold(line, width.into())
}

/// `wrap_line` to any width, a width of none leaving the line as it is.
fn fold(line: Line<'static>, width: usize) -> Vec<Line<'static>> {
  // A tab is measured as no columns and drawn as however many the terminal
  // feels like, and a carriage return draws the rest of the line over the
  // start of it, so a line carrying either is rebuilt even when it fits.
  let literal = line.spans.iter().all(|span| !span.content.contains(['\n', '\t', '\r']));
  if width == 0 || (literal && line.width() <= width) {
    return vec![line];
  }
  let (style, alignment) = (line.style, line.alignment);
  let mut lines: Vec<Line<'static>> = Vec::new();
  let mut current: Vec<Span<'static>> = Vec::new();
  let mut used = 0;
  // Until the first word of the line, its spaces are the indent it was
  // written with rather than the debris of a break.
  let mut indent = true;

  let mut push_line = |current: &mut Vec<Span<'static>>, used: &mut usize| {
    // The space a break falls on is drawn on neither line. A line of nothing
    // but spaces is an indent that never got its word, and keeps them.
    if current.iter().any(|span| !span.content.trim().is_empty()) {
      while current.last().is_some_and(|span| span.content.trim().is_empty()) {
        current.pop();
      }
    }
    lines.push(Line::from(std::mem::take(current)).style(style));
    *used = 0;
  };

  for span in line.spans {
    // A carriage return is not a column and not a break; what it meant was
    // for the terminal to draw over what it had already drawn, which a
    // transcript that scrolls has no way of honouring.
    let content = expand_tabs(&span.content).replace('\r', "");
    for (i, chunk) in content.split('\n').enumerate() {
      if i > 0 {
        // A break the text asked for starts a line of its own, indent and
        // all — it is a line as written, not the remains of one.
        push_line(&mut current, &mut used);
        indent = true;
      }
      for word in words(chunk) {
        let blank = word.trim().is_empty();
        let w = word.width();
        if blank && used == 0 && !indent {
          continue;
        }
        // A word too long for a line of its own gains nothing by being moved
        // to one, so it is broken where the line it is on runs out.
        if w > width && !blank {
          let mut piece = String::new();
          for ch in word.chars() {
            if used + ch.width().unwrap_or(0) > width {
              if !piece.is_empty() {
                current.push(Span::styled(std::mem::take(&mut piece), span.style));
              }
              push_line(&mut current, &mut used);
            }
            used += ch.width().unwrap_or(0);
            piece.push(ch);
          }
          if !piece.is_empty() {
            current.push(Span::styled(piece, span.style));
          }
          indent = false;
          continue;
        }
        if used + w > width && used > 0 {
          push_line(&mut current, &mut used);
          indent = false;
          if blank {
            continue;
          }
        }
        current.push(Span::styled(word.to_string(), span.style));
        used += w;
        indent &= blank;
      }
    }
  }
  if !current.is_empty() || lines.is_empty() {
    lines.push(Line::from(current).style(style));
  }
  for line in &mut lines {
    line.alignment = alignment;
  }
  lines
}

/// Word-wraps plain text to `width` columns, for prose that is shown as
/// written rather than parsed — the reasoning block, which is the model
/// thinking aloud and not markdown it meant for us to render.
pub fn wrap_text(text: &str, width: u16, style: Style) -> Vec<Line<'static>> {
  plain(text, width.into(), style)
}

/// Each line of `text` wrapped on its own, all in the one style.
fn plain(text: &str, width: usize, style: Style) -> Vec<Line<'static>> {
  text
    .lines()
    .flat_map(|line| wrap(vec![Span::styled(line.to_string(), style)], width))
    .collect()
}

fn children(node: &Node) -> &[Node] {
  node.children().map(Vec::as_slice).unwrap_or_default()
}

/// A terminal cell grid has no tab stops, so a tab becomes the four columns
/// CommonMark counts it as.
fn expand_tabs(text: &str) -> String {
  text.replace('\t', "    ")
}

/// Text a program wrote for a terminal, as the terminal would have left it.
///
/// A control character is never drawn, but `unicode-width` still counts it
/// as a column, so a line carrying one is laid out wider than it is drawn.
/// An escape sequence is one unit, a colour or a cursor move, and goes as a
/// whole: dropping only its escape byte would leave `[31m` in the text. A
/// carriage return starts its line again, so a progress bar keeps its last
/// state rather than every state it passed through, and a backspace takes
/// back the character before it. Every other control character goes, except
/// the newlines and tabs that are laid out later, and so do the invisible
/// characters terminals do not agree on the width of.
pub fn printable(text: &str) -> Cow<'_, str> {
  if !text
    .chars()
    .any(|c| (c.is_control() && c != '\n' && c != '\t') || invisible(c))
  {
    return Cow::Borrowed(text);
  }
  let mut out = String::with_capacity(text.len());
  for (i, line) in text.split('\n').enumerate() {
    if i > 0 {
      out.push('\n');
    }
    settle(line, &mut out);
  }
  Cow::Owned(out)
}

/// Invisible characters `unicode-width` and terminals measure differently: a
/// soft hyphen it counts as nothing and they draw, marks that prefix a number
/// it counts as part of the digit after them, separators it counts as a column,
/// and the directional controls a terminal that lays out right-to-left text
/// would reorder the rest of the line by. Joiners are not among them, since
/// emoji and whole scripts are spelled with them.
fn invisible(c: char) -> bool {
  matches!(
    c,
    '\u{ad}'
      | '\u{600}'..='\u{605}'
      | '\u{61c}'
      | '\u{6dd}'
      | '\u{70f}'
      | '\u{890}'..='\u{891}'
      | '\u{8e2}'
      | '\u{180e}'
      | '\u{200b}'
      | '\u{200e}'..='\u{200f}'
      | '\u{2028}'..='\u{202e}'
      | '\u{2060}'..='\u{206f}'
      | '\u{feff}'
      | '\u{fff9}'..='\u{fffb}'
      | '\u{110bd}'
      | '\u{110cd}'
      | '\u{13430}'..='\u{1343f}'
      | '\u{1bca0}'..='\u{1bca3}'
      | '\u{1d173}'..='\u{1d17a}'
      | '\u{e0001}'
      | '\u{e0020}'..='\u{e007f}'
  )
}

/// One line of terminal output onto the end of `out`, as it was left: its
/// escapes and controls acted on, or dropped where acting on them means
/// nothing to a transcript.
fn settle(line: &str, out: &mut String) {
  // Where the line starts in `out`, and whether a carriage return has sent
  // the cursor back to it: the next character written clears the line, and a
  // line ending before one arrives keeps what was there.
  let start = out.len();
  let mut returned = false;
  let mut chars = line.chars().peekable();
  while let Some(c) = chars.next() {
    match c {
      '\r' => returned = true,
      '\u{8}' => {
        if !returned && out.len() > start {
          out.pop();
        }
      }
      '\u{1b}' => {
        if chars.next_if_eq(&'[').is_some() {
          skip_csi(&mut chars);
        } else if chars.next_if(|c| matches!(c, ']' | 'P' | 'X' | '^' | '_')).is_some() {
          // Operating system commands, device controls and the rest of the
          // string-carrying sequences, which run to a string terminator — or
          // a bell, which xterm takes for one.
          while let Some(c) = chars.next() {
            if c == '\u{7}' || c == '\u{9c}' || (c == '\u{1b}' && chars.next_if_eq(&'\\').is_some()) {
              break;
            }
          }
        } else {
          // Everything else is its intermediates and a final character. An
          // escape followed by neither is a stray, and whatever it was
          // followed by is text.
          while chars.next_if(|c| ('\u{20}'..='\u{2f}').contains(c)).is_some() {}
          chars.next_if(|c| ('\u{30}'..='\u{7e}').contains(c));
        }
      }
      '\u{9b}' => skip_csi(&mut chars),
      c if (c.is_control() && c != '\t') || invisible(c) => {}
      c => {
        if returned {
          out.truncate(start);
          returned = false;
        }
        out.push(c);
      }
    }
  }
}

/// The rest of a control sequence: parameters and intermediates, up to the
/// final character that ends it. One cut short ends where it stops looking
/// like a sequence, so the text after it is kept.
fn skip_csi(chars: &mut std::iter::Peekable<impl Iterator<Item = char>>) {
  while chars.next_if(|c| ('\u{20}'..='\u{3f}').contains(c)).is_some() {}
  chars.next_if(|c| ('\u{40}'..='\u{7e}').contains(c));
}

/// What a message's body refers to but does not carry inline: footnotes and
/// link definitions. Both are written as their own blocks, so both have to be
/// resolved up front and held back from the flow.
///
/// Footnotes are numbered as GFM numbers them — by the order their references
/// appear, not the order they are defined — and one that nothing refers to is
/// dropped, as on github.com.
#[derive(Default)]
struct Refs<'a> {
  referenced: Vec<String>,
  /// Link and image definitions, by their (already lowercased) label.
  links: HashMap<String, String>,
  /// Every footnote's body, referred to or not, by its label. A label
  /// defined twice keeps its first body, as on github.com.
  footnotes: HashMap<&'a str, &'a FootnoteDefinition>,
}

impl<'a> Refs<'a> {
  fn collect(root: &'a Node) -> Self {
    let mut refs = Refs::default();
    refs.walk(root);
    refs
  }

  fn walk(&mut self, node: &'a Node) {
    match node {
      Node::FootnoteReference(reference) if !self.referenced.contains(&reference.identifier) => {
        self.referenced.push(reference.identifier.clone());
      }
      Node::Definition(definition) => {
        self.links.insert(definition.identifier.clone(), definition.url.clone());
      }
      Node::FootnoteDefinition(definition) => {
        self.footnotes.entry(&definition.identifier).or_insert(definition);
      }
      _ => {}
    }
    for child in children(node) {
      self.walk(child);
    }
  }

  /// The marker a reference carries, 1-based, or `None` when nothing refers
  /// to it — in which case the reference was never seen in the first place.
  fn number(&self, identifier: &str) -> Option<usize> {
    self.referenced.iter().position(|id| id == identifier).map(|i| i + 1)
  }

  /// Appends the footnote list, in reference order, after the body.
  fn render_footnotes(&self, width: usize, out: &mut Vec<Line<'static>>) {
    // Numbered by position among the references, so a reference with no
    // body leaves a gap rather than renumbering the ones after it.
    let definitions: Vec<(usize, &FootnoteDefinition)> = self
      .referenced
      .iter()
      .enumerate()
      .filter_map(|(i, id)| Some((i + 1, *self.footnotes.get(id.as_str())?)))
      .collect();
    if definitions.is_empty() {
      return;
    }
    out.push(Line::default());
    out.push(Line::styled("─".repeat(width.min(HR_MAX)), style::BORDER));
    for (number, definition) in definitions {
      let marker = Span::styled(format!("[{number}] "), style::FOOTNOTE);
      let body = width.saturating_sub(marker.width()).max(1);
      let mut inner = Vec::new();
      blocks(&definition.children, body, self, &mut inner);
      hang(inner, &marker, &mut false, out);
    }
  }
}

/// Renders a run of block nodes, blank-separated.
fn blocks(nodes: &[Node], width: usize, refs: &Refs, out: &mut Vec<Line<'static>>) {
  // Definitions carry no visible text of their own: footnotes are listed at
  // the end and link labels are resolved where they are used. They must not
  // count as siblings either, or one sitting between two paragraphs would
  // leave its blank line behind in the body.
  let nodes: Vec<&Node> = nodes
    .iter()
    .filter(|node| !matches!(node, Node::FootnoteDefinition(_) | Node::Definition(_)))
    .collect();
  for (i, node) in nodes.iter().enumerate() {
    block(node, width, refs, out);
    // A list stays tight against the paragraph that introduces it.
    let next_is_list = matches!(nodes.get(i + 1), Some(Node::List(_)));
    let intro = matches!(node, Node::Paragraph(_)) && next_is_list;
    if i + 1 < nodes.len() && !intro {
      out.push(Line::default());
    }
  }
}

fn block(node: &Node, width: usize, refs: &Refs, out: &mut Vec<Line<'static>>) {
  match node {
    Node::Heading(heading) => {
      let style = style::heading(heading.depth);
      let mut spans = Vec::new();
      // The level is marked with `###` from depth 3 down, where bold alone
      // stops being enough to tell the levels apart.
      if heading.depth >= 3 {
        spans.push(Span::styled("#".repeat(heading.depth as usize) + " ", style));
      }
      inline(&heading.children, style, refs, &mut spans);
      out.extend(wrap(spans, width));
    }
    Node::Paragraph(paragraph) => {
      let mut spans = Vec::new();
      inline(&paragraph.children, Style::default(), refs, &mut spans);
      out.extend(wrap(spans, width));
    }
    Node::Code(code) => {
      let lang = code.lang.clone().unwrap_or_default();
      out.push(Line::styled(format!("```{lang}"), style::BORDER));
      // A language we have a grammar for is coloured token by token; anything
      // else — no info string, a language not built in — keeps the one colour
      // the whole block used to have.
      let highlighted = crate::highlight::highlight(&lang, &code.value);
      // Long code wraps rather than being clipped: a transcript has no
      // horizontal scroll, so a clipped line silently loses the end of a
      // command.
      for (i, line) in code.value.lines().enumerate() {
        // In code the leading whitespace is content, not the wrapping debris
        // `wrap` drops, so it is held aside and re-applied — which also gives
        // a wrapped line a hanging indent at its own nesting level. Tabs are
        // expanded first so the indent is measured in the columns it occupies.
        let line = expand_tabs(line);
        let code = line.trim_start();
        let indent = format!("{CODE_INDENT}{}", &line[..line.len() - code.len()]);
        let body = width.saturating_sub(indent.width()).max(1);
        // The highlighter indexes by the source's own line numbers, so a run
        // it could not finish leaves later lines plain rather than shifted.
        let spans = match highlighted.as_ref().and_then(|lines| lines.get(i)) {
          Some(spans) => trim_indent(spans.clone()),
          None => vec![Span::styled(code.to_string(), style::CODE_BLOCK)],
        };
        for wrapped in wrap(spans, body) {
          out.push(prefix(Span::raw(indent.clone()), wrapped));
        }
      }
      out.push(Line::styled("```", style::BORDER));
    }
    Node::List(list) => list_block(list, 0, width, refs, out),
    Node::Blockquote(_) => {
      let body = width.saturating_sub(QUOTE_PREFIX.width()).max(1);
      let mut inner = Vec::new();
      blocks(children(node), body, refs, &mut inner);
      trim_blank(&mut inner);
      for mut line in inner {
        // The quote style is a floor, not an override: inline code and links
        // inside a quote keep their own colour.
        for span in &mut line.spans {
          span.style = style::QUOTE.patch(span.style);
        }
        out.push(prefix(Span::styled(QUOTE_PREFIX, style::BORDER), line));
      }
    }
    Node::ThematicBreak(_) => {
      out.push(Line::styled("─".repeat(width.min(HR_MAX)), style::BORDER));
    }
    Node::Table(table) => table_block(&table.children, &table.align, width, refs, out),
    Node::Html(html) => out.extend(plain(&html.value, width, style::DIM)),
    Node::Math(math) => out.extend(plain(&math.value, width, Style::default())),
    // Link reference targets and footnote bodies carry no visible text of
    // their own; anything else block-shaped is rendered as its inline run.
    Node::Definition(_) | Node::FootnoteDefinition(_) => {}
    other => {
      let mut spans = Vec::new();
      inline(std::slice::from_ref(other), Style::default(), refs, &mut spans);
      if !spans.is_empty() {
        out.extend(wrap(spans, width));
      }
    }
  }
}

/// A list, rendered at `depth` levels of nesting.
fn list_block(list: &List, depth: usize, width: usize, refs: &Refs, out: &mut Vec<Line<'static>>) {
  let indent = " ".repeat(LIST_INDENT * depth);
  let start = list.start.unwrap_or(1);
  for (i, item) in list.children.iter().enumerate() {
    let Node::ListItem(item) = item else { continue };
    let bullet = if list.ordered {
      format!("{}. ", start as usize + i)
    } else {
      "- ".to_string()
    };
    let task = match item.checked {
      Some(true) => "[x] ",
      Some(false) => "[ ] ",
      None => "",
    };
    let marker = Span::styled(format!("{indent}{bullet}{task}"), style::BULLET);
    let body = width.saturating_sub(marker.width()).max(1);

    let mut inner: Vec<Line<'static>> = Vec::new();
    // Only the item's first line carries the marker; everything after it —
    // later paragraphs included — hangs under the text.
    let mut marked = false;
    for child in &item.children {
      // A nested list indents from the outer width rather than the item's, so
      // its markers line up in a column of their own.
      if let Node::List(nested) = child {
        hang(std::mem::take(&mut inner), &marker, &mut marked, out);
        list_block(nested, depth + 1, width, refs, out);
        marked = true;
        continue;
      }
      block(child, body, refs, &mut inner);
      if item.spread {
        inner.push(Line::default());
      }
    }
    hang(inner, &marker, &mut marked, out);
    if !marked {
      out.push(Line::from(marker));
    }
  }
}

/// A GFM table, with columns shrunk to fit and cells wrapped inside them.
fn table_block(rows: &[Node], align: &[AlignKind], width: usize, refs: &Refs, out: &mut Vec<Line<'static>>) {
  let cells: Vec<Vec<Vec<Span<'static>>>> = rows
    .iter()
    .enumerate()
    .map(|(r, row)| {
      // The header row is bold throughout.
      let base = match r {
        0 => Style::new().add_modifier(Modifier::BOLD),
        _ => Style::default(),
      };
      children(row)
        .iter()
        .map(|cell| {
          let mut spans = Vec::new();
          inline(children(cell), base, refs, &mut spans);
          spans
        })
        .collect()
    })
    .collect();
  let columns = cells.iter().map(Vec::len).max().unwrap_or(0);
  if columns == 0 {
    return;
  }
  // Borders cost `│ ` before every column and a trailing `│`.
  let overhead = columns * 3 + 1;
  let natural: Vec<usize> = (0..columns)
    .map(|c| {
      cells
        .iter()
        .filter_map(|row| row.get(c))
        .map(|spans| spans.iter().map(Span::width).sum::<usize>())
        .max()
        .unwrap_or(0)
        .max(1)
    })
    .collect();
  let widths = fit(&natural, width.saturating_sub(overhead).max(columns));

  let rule = |left: &str, mid: &str, right: &str| {
    let mut s = String::from(left);
    for (i, w) in widths.iter().enumerate() {
      s.push_str(&"─".repeat(w + 2));
      s.push_str(if i + 1 == widths.len() { right } else { mid });
    }
    Line::styled(s, style::BORDER)
  };

  out.push(rule("┌", "┬", "┐"));
  // A header-only table has nothing to separate from.
  let body_rows = cells.len() > 1;
  for (r, row) in cells.iter().enumerate() {
    // Every cell wraps to its column, then the row is as tall as the tallest.
    let wrapped: Vec<Vec<Line<'static>>> = (0..columns)
      .map(|c| wrap(row.get(c).cloned().unwrap_or_default(), widths[c]))
      .collect();
    let height = wrapped.iter().map(Vec::len).max().unwrap_or(1).max(1);
    for line in 0..height {
      let mut spans = vec![Span::styled("│", style::BORDER)];
      for (c, column) in wrapped.iter().enumerate() {
        let content = column.get(line).cloned().unwrap_or_default();
        let pad = widths[c].saturating_sub(content.width());
        let (before, after) = match align.get(c) {
          Some(AlignKind::Right) => (pad, 0),
          Some(AlignKind::Center) => (pad / 2, pad - pad / 2),
          _ => (0, pad),
        };
        spans.push(Span::raw(" ".repeat(before + 1)));
        spans.extend(content.spans);
        spans.push(Span::raw(" ".repeat(after + 1)));
        spans.push(Span::styled("│", style::BORDER));
      }
      out.push(Line::from(spans));
    }
    if r == 0 && body_rows {
      out.push(rule("├", "┼", "┤"));
    }
  }
  out.push(rule("└", "┴", "┘"));
}

/// Shrinks columns to `budget`, taking from the widest first so narrow columns
/// survive intact.
fn fit(natural: &[usize], budget: usize) -> Vec<usize> {
  let mut widths = natural.to_vec();
  let mut total: usize = widths.iter().sum();
  while total > budget {
    let Some(widest) = widths
      .iter()
      .enumerate()
      .filter(|(_, w)| **w > 1)
      .max_by_key(|(_, w)| **w)
      .map(|(i, _)| i)
    else {
      break;
    };
    widths[widest] -= 1;
    total -= 1;
  }
  widths
}

/// Link text, followed by its target when the text does not already say it.
///
/// An autolink shows its own URL, so repeating it would only add noise.
fn link_spans(children: &[Node], url: &str, base: Style, refs: &Refs, out: &mut Vec<Span<'static>>) {
  let start = out.len();
  inline(children, base.patch(style::LINK), refs, out);
  let text: String = out[start..].iter().map(|s| s.content.as_ref()).collect();
  let bare = url.strip_prefix("mailto:").unwrap_or(url);
  if text != url && text != bare {
    out.push(Span::styled(format!(" ({url})"), base.patch(style::DIM)));
  }
}

/// An image is a placeholder here: the transcript cannot show one inline.
fn image_span(alt: &str, base: Style) -> Span<'static> {
  let alt = if alt.is_empty() { "image" } else { alt };
  Span::styled(format!("[{alt}]"), base.patch(style::DIM))
}

/// Flattens inline nodes into styled spans, `base` being the style inherited
/// from the block around them.
fn inline(nodes: &[Node], base: Style, refs: &Refs, out: &mut Vec<Span<'static>>) {
  for node in nodes {
    match node {
      Node::Text(text) => out.push(Span::styled(text.value.clone(), base)),
      Node::Strong(strong) => inline(&strong.children, base.add_modifier(Modifier::BOLD), refs, out),
      Node::Emphasis(emphasis) => inline(&emphasis.children, base.add_modifier(Modifier::ITALIC), refs, out),
      Node::Delete(delete) => inline(&delete.children, base.add_modifier(Modifier::CROSSED_OUT), refs, out),
      Node::InlineCode(code) => out.push(Span::styled(code.value.clone(), base.patch(style::INLINE_CODE))),
      Node::InlineMath(math) => out.push(Span::styled(math.value.clone(), base)),
      Node::Link(link) => link_spans(&link.children, &link.url, base, refs, out),
      // `[text][label]`, whose target lives in a definition elsewhere. Without
      // resolving it the text would render as unmarked prose, losing both the
      // fact that it is a link and where it points.
      Node::LinkReference(reference) => match refs.links.get(&reference.identifier) {
        Some(url) => link_spans(&reference.children, url, base, refs, out),
        None => inline(&reference.children, base, refs, out),
      },
      Node::Image(image) => out.push(image_span(&image.alt, base)),
      Node::ImageReference(reference) => out.push(image_span(&reference.alt, base)),
      Node::FootnoteReference(reference) => {
        // Numbered rather than named, so a `[^implementation-note]` does not
        // swallow the line it sits in. The list at the end uses the same
        // numbers, which is the only thing tying the two together here.
        if let Some(number) = refs.number(&reference.identifier) {
          out.push(Span::styled(format!("[{number}]"), base.patch(style::FOOTNOTE)));
        }
      }
      Node::Break(_) => out.push(Span::styled("\n", base)),
      Node::Html(html) => out.push(Span::styled(html.value.clone(), base)),
      other => inline(children(other), base, refs, out),
    }
  }
}

/// Drops the leading whitespace of a highlighted code line, which the caller
/// re-applies as a literal indent.
///
/// `wrap` would drop it anyway — it treats leading space as wrapping debris —
/// but it has to go before the first word is measured, or a deeply indented
/// line would be wrapped as if the indent cost nothing.
fn trim_indent(spans: Vec<Span<'static>>) -> Vec<Span<'static>> {
  let mut out = Vec::with_capacity(spans.len());
  for span in spans {
    // Once any content is through, the rest of the line is kept verbatim:
    // whitespace inside it is the code's own spacing.
    if !out.is_empty() {
      out.push(span);
      continue;
    }
    let trimmed = span.content.trim_start();
    if !trimmed.is_empty() {
      out.push(Span::styled(trimmed.to_string(), span.style));
    }
  }
  out
}

/// Drops the blank lines a run of blocks ends with.
fn trim_blank(lines: &mut Vec<Line<'static>>) {
  while lines.last().is_some_and(|l| l.width() == 0) {
    lines.pop();
  }
}

/// Hangs `lines` off `marker`: the first line carries it, unless `marked` says
/// it has been put down already, and the rest are indented under the text.
fn hang(mut lines: Vec<Line<'static>>, marker: &Span<'static>, marked: &mut bool, out: &mut Vec<Line<'static>>) {
  trim_blank(&mut lines);
  let continuation = " ".repeat(marker.width());
  for line in lines {
    // A blank line between paragraphs needs no indent; padding it would only
    // leave trailing whitespace behind.
    if *marked && line.width() == 0 {
      out.push(line);
      continue;
    }
    let lead = match *marked {
      true => Span::raw(continuation.clone()),
      false => marker.clone(),
    };
    out.push(prefix(lead, line));
    *marked = true;
  }
}

/// Puts `lead` in front of `line`, keeping the rest of its spans.
fn prefix(lead: Span<'static>, mut line: Line<'static>) -> Line<'static> {
  line.spans.insert(0, lead);
  line
}

/// Word-wraps a run of spans to `width`, carrying each span's style onto its
/// fragments.
fn wrap(spans: Vec<Span<'static>>, width: usize) -> Vec<Line<'static>> {
  fold(Line::from(spans), width.max(1))
}

/// Splits on whitespace, keeping the separators so spacing survives wrapping.
fn words(text: &str) -> Vec<&str> {
  let mut out = Vec::new();
  let mut start = 0;
  let mut in_space = None;
  for (i, ch) in text.char_indices() {
    let space = ch.is_whitespace();
    match in_space {
      Some(was) if was != space => {
        out.push(&text[start..i]);
        start = i;
        in_space = Some(space);
      }
      None => in_space = Some(space),
      _ => {}
    }
  }
  if start < text.len() {
    out.push(&text[start..]);
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  fn plain(lines: &[Line<'static>]) -> Vec<String> {
    lines
      .iter()
      .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
      .collect()
  }

  #[test]
  fn wraps_paragraphs_to_width() {
    let lines = render("one two three four five", 9, false);
    assert_eq!(plain(&lines), ["one two", "three", "four five"]);
  }

  #[test]
  fn list_continuations_hang_under_the_text() {
    let lines = render("- alpha beta gamma", 11, false);
    assert_eq!(plain(&lines), ["- alpha", "  beta", "  gamma"]);
  }

  #[test]
  fn nested_lists_indent() {
    let lines = render("- outer\n    - inner", 20, false);
    assert_eq!(plain(&lines), ["- outer", "    - inner"]);
  }

  #[test]
  fn blockquotes_keep_their_border() {
    let lines = render("> alpha beta gamma", 11, false);
    assert_eq!(plain(&lines), ["│ alpha", "│ beta", "│ gamma"]);
  }

  #[test]
  fn headings_are_styled_and_marked_from_depth_three() {
    let lines = render("# Title", 20, false);
    assert_eq!(plain(&lines), ["Title"]);
    assert!(lines[0].spans[0].style.add_modifier.contains(Modifier::BOLD));
    assert_eq!(plain(&render("### Sub", 20, false)), ["### Sub"]);
  }

  #[test]
  fn code_blocks_keep_their_fence_and_indent() {
    let lines = render("```rust\nfn main() {}\n```", 20, false);
    assert_eq!(plain(&lines), ["```rust", "  fn main() {}", "```"]);
  }

  #[test]
  #[cfg(feature = "lang-rust")]
  fn a_known_language_is_highlighted_token_by_token() {
    let lines = render("```rust\n    let x = 1; // note\n```", 40, false);
    // The indent is still the block's own, carried around the highlighting.
    assert_eq!(plain(&lines), ["```rust", "      let x = 1; // note", "```"]);
    let span = |needle: &str| lines[1].spans.iter().find(|s| s.content == needle).unwrap();
    assert_eq!(span("let").style.fg, Some(Color::Magenta));
    assert_eq!(span("1").style.fg, Some(Color::Cyan));
    assert_eq!(span("// note").style.fg, Some(Color::DarkGray));
  }

  #[test]
  fn a_language_we_cannot_parse_keeps_the_one_colour() {
    // No info string, or one no grammar answers to: the block is green, as
    // every block was before there were grammars.
    for md in ["```\nx = 1\n```", "```brainfuck\n+++\n```"] {
      let lines = render(md, 40, false);
      let body: Vec<&Span<'static>> = lines[1].spans.iter().skip(1).collect();
      assert!(body.iter().all(|s| s.style.fg == Some(Color::Green)), "{md}: {body:?}");
    }
  }

  #[test]
  fn code_keeps_its_own_indentation() {
    // Leading whitespace is content here, not the separator `wrap` discards.
    let lines = render("```\nfn f() {\n    body();\n}\n```", 30, false);
    assert_eq!(plain(&lines), ["```", "  fn f() {", "      body();", "  }", "```"]);
  }

  #[test]
  fn inline_styles_become_spans() {
    let lines = render("a **b** `c`", 20, false);
    assert_eq!(plain(&lines), ["a b c"]);
    let bold = lines[0].spans.iter().find(|s| s.content == "b").unwrap();
    assert!(bold.style.add_modifier.contains(Modifier::BOLD));
    let code = lines[0].spans.iter().find(|s| s.content == "c").unwrap();
    assert_eq!(code.style.fg, Some(Color::Cyan));
  }

  #[test]
  fn links_print_their_url_only_when_it_differs() {
    assert_eq!(
      plain(&render("[text](https://e.com)", 40, false)),
      ["text (https://e.com)"]
    );
    assert_eq!(plain(&render("<https://e.com>", 40, false)), ["https://e.com"]);
  }

  #[test]
  fn strikethrough_follows_the_parser_default() {
    let struck = |md: &str| {
      render(md, 40, false)[0]
        .spans
        .iter()
        .any(|s| s.style.add_modifier.contains(Modifier::CROSSED_OUT))
    };
    // GFM spec example: `~~Hi~~ Hello, world!`
    assert!(struck("~~Hi~~ Hello, world!"));
    // The spec says "two tildes", but github.com strikes a single one and
    // markdown-rs follows github.com by default. We take the default.
    assert!(struck("~one~"));
    // Matching counts are not enough: a run of three or more never opens
    // strikethrough, so the tildes stay literal text.
    assert!(!struck("a ~~~~four~~~~ b"));
    assert_eq!(plain(&render("a ~~~~four~~~~ b", 40, false)), ["a ~~~~four~~~~ b"]);
    // cmark-gfm refuses to pair openers and closers of unequal length.
    assert!(!struck("a ~~mismatched~ b"));
    assert_eq!(plain(&render("a ~~mismatched~ b", 40, false)), ["a ~~mismatched~ b"]);
  }

  #[test]
  fn strikethrough_stops_at_a_paragraph_break() {
    // GFM spec example: "As with regular emphasis delimiters, a new paragraph
    // will cause strikethrough parsing to cease."
    let lines = render("This ~~has a\n\nnew paragraph~~.", 40, false);
    assert_eq!(plain(&lines), ["This ~~has a", "", "new paragraph~~."]);
    assert!(
      !lines
        .iter()
        .flat_map(|l| &l.spans)
        .any(|s| s.style.add_modifier.contains(Modifier::CROSSED_OUT))
    );
  }

  #[test]
  fn unpaired_markers_stay_literal_once_the_message_is_final() {
    // Completing them is only right while the text is still arriving; a
    // finished message is read exactly as the spec reads it.
    assert_eq!(
      plain(&render("a ~~open and never closed", 40, false)),
      ["a ~~open and never closed"]
    );
    assert_eq!(
      plain(&render("a **open and never closed", 40, false)),
      ["a **open and never closed"]
    );
  }

  #[test]
  fn a_tilde_run_at_the_start_of_a_line_opens_a_code_fence() {
    // `~~~` is a CommonMark fence, so this is a code block whose info string
    // is `three~~~` — not strikethrough, and not literal text either.
    assert_eq!(plain(&render("~~~three~~~", 40, false)), ["```three~~~", "```"]);
  }

  #[test]
  fn unterminated_emphasis_is_closed_mid_stream() {
    // The half-streamed `**bold` must not show its asterisks.
    assert_eq!(plain(&render("a **bol", 20, true)), ["a bol"]);
  }

  #[test]
  fn long_words_are_broken_rather_than_clipped() {
    let lines = render("aaaaaaaaaa", 4, false);
    assert_eq!(plain(&lines), ["aaaa", "aaaa", "aa"]);
  }

  #[test]
  fn wide_characters_count_two_columns() {
    let lines = render("日本語テスト", 6, false);
    assert_eq!(plain(&lines), ["日本語", "テスト"]);
  }

  #[test]
  fn plain_text_wraps_without_being_parsed() {
    // The reasoning block is prose, not markdown: `# ` and `**` stay as the
    // model wrote them, and the count of lines is the count as drawn.
    let lines = wrap_text("# not a heading **not bold**", 12, Style::default());
    assert_eq!(plain(&lines), ["# not a", "heading", "**not bold**"]);
    // A paragraph with no newlines still counts as the rows it occupies,
    // which is what makes a collapsed block consistent across providers.
    let paragraph = "one two three four five six";
    assert_eq!(paragraph.lines().count(), 1);
    let wide = wrap_text(paragraph, 20, Style::default()).len();
    let narrow = wrap_text(paragraph, 9, Style::default()).len();
    assert!(wide > 1 && narrow > wide, "wide {wide}, narrow {narrow}");
  }

  #[test]
  fn tables_fit_the_width() {
    let lines = render("| a | b |\n| - | - |\n| 1 | 2 |", 20, false);
    let rendered = plain(&lines);
    assert!(rendered.iter().all(|l| l.width() <= 20), "{rendered:?}");
    assert!(rendered[0].starts_with('┌'));
  }

  #[test]
  fn malformed_markdown_falls_back_to_source() {
    assert!(!render("# \u{0}\n[", 20, false).is_empty());
  }

  #[test]
  fn footnotes_are_numbered_and_listed_at_the_end() {
    let lines = render("Note[^a] and[^b].\n\n[^b]: Second.\n\n[^a]: First.", 40, false);
    assert_eq!(
      plain(&lines),
      [
        "Note[1] and[2].",
        "",
        "────────────────────────────────────────",
        "[1] First.",
        "[2] Second.",
      ]
    );
  }

  #[test]
  fn footnote_definitions_leave_no_gap_in_the_body() {
    // The definition sits between two paragraphs in the source but renders
    // at the end, so it must not leave its blank line behind.
    let lines = render("One[^n].\n\n[^n]: Note.\n\nTwo.", 40, false);
    assert_eq!(plain(&lines)[..3], ["One[1].", "", "Two."]);
  }

  #[test]
  fn footnote_bodies_wrap_with_a_hanging_indent() {
    let lines = render("x[^n]\n\n[^n]: alpha beta gamma delta", 15, false);
    assert_eq!(
      plain(&lines),
      ["x[1]", "", "───────────────", "[1] alpha beta", "    gamma delta",]
    );
  }

  #[test]
  fn an_unreferenced_definition_is_dropped() {
    // github.com lists only the notes something points at.
    assert_eq!(
      plain(&render("Body.\n\n[^loose]: Nothing points here.", 40, false)),
      ["Body."]
    );
  }

  #[test]
  fn a_reference_with_no_definition_stays_literal() {
    assert_eq!(plain(&render("see [^gone] here", 40, false)), ["see [^gone] here"]);
  }

  #[test]
  fn a_footnote_body_may_hold_several_blocks() {
    let lines = render("x[^n]\n\n[^n]: First paragraph.\n\n    Second paragraph.", 40, false);
    assert_eq!(
      plain(&lines)[3..],
      ["[1] First paragraph.", "", "    Second paragraph."]
    );
  }

  #[test]
  fn reference_links_resolve_to_their_definition() {
    // The label is defined in a block of its own, so an unresolved reference
    // would render as unmarked prose — losing both the link and its target.
    let lines = render("see [text][ref]\n\n[ref]: https://e.com", 40, false);
    assert_eq!(plain(&lines), ["see text (https://e.com)"]);
    let text = lines[0].spans.iter().find(|s| s.content == "text").unwrap();
    assert_eq!(text.style.fg, Some(Color::Blue));
    // Collapsed and shortcut forms, and labels matched case-insensitively.
    assert_eq!(
      plain(&render("[ref][]\n\n[ref]: https://e.com", 40, false)),
      ["ref (https://e.com)"]
    );
    assert_eq!(
      plain(&render("[CASE]\n\n[case]: https://e.com", 40, false)),
      ["CASE (https://e.com)"]
    );
  }

  #[test]
  fn reference_images_show_their_alt_text() {
    assert_eq!(plain(&render("![alt][img]\n\n[img]: /a.png", 40, false)), ["[alt]"]);
  }

  #[test]
  fn link_definitions_leave_no_gap_in_the_body() {
    let lines = render("One.\n\n[ref]: https://e.com\n\nTwo.", 40, false);
    assert_eq!(plain(&lines), ["One.", "", "Two."]);
  }

  #[test]
  fn a_tab_indents_code_by_four_columns() {
    // CommonMark counts a tab as advancing to the next four-column stop, so
    // a tab-indented line is a code block, not a paragraph.
    assert_eq!(
      plain(&render("para\n\n\tcode line", 40, false)),
      ["para", "", "```", "  code line", "```"]
    );
    // And inside a fence a tab keeps the width it stands for.
    assert_eq!(
      plain(&render("```\nif x:\n\tpass\n```", 40, false)),
      ["```", "  if x:", "      pass", "```"]
    );
  }

  #[test]
  fn a_header_only_table_has_no_dangling_separator() {
    let lines = render("| a | b |\n| - | - |", 20, false);
    let rendered = plain(&lines);
    assert_eq!(rendered.len(), 3, "{rendered:?}");
    assert!(rendered[2].starts_with('└'));
  }

  #[test]
  fn a_line_that_fits_is_the_line_it_was() {
    let line = Line::from(vec![
      Span::raw("  "),
      Span::styled("gutter", Style::default().fg(Color::Green)),
    ])
    .style(Style::default().add_modifier(Modifier::DIM));
    let wrapped = wrap_line(line.clone(), 20);
    // Untouched, down to the indent it starts with, the colour of each span
    // and the style of the line itself — which is where a dimmed line keeps
    // its dimming.
    assert_eq!(wrapped, [line]);
  }

  #[test]
  fn a_line_too_long_for_the_width_is_wrapped_to_it() {
    let line = Line::from(vec![
      Span::styled("alpha beta ", Style::default().fg(Color::Red)),
      Span::styled("gamma delta", Style::default().fg(Color::Blue)),
    ]);
    let wrapped = wrap_line(line, 12);
    assert_eq!(plain(&wrapped), ["alpha beta", "gamma delta"]);
    // Each piece is still the colour of the span it came out of.
    assert_eq!(wrapped[0].spans[0].style.fg, Some(Color::Red));
    assert_eq!(wrapped[1].spans[0].style.fg, Some(Color::Blue));
  }

  #[test]
  fn spaces_wider_than_the_line_are_a_break_not_a_row() {
    assert_eq!(plain(&wrap_line(Line::raw("a   ccc"), 2)), ["a", "cc", "c"]);
  }

  #[test]
  fn the_indent_a_line_starts_with_survives_the_wrap() {
    // The break drops the space it falls on, but the indent is not debris:
    // it is where the line was written to start.
    let line = Line::from(vec![Span::raw("  "), Span::raw("alpha beta gamma")]);
    assert_eq!(plain(&wrap_line(line, 10)), ["  alpha", "beta gamma"]);
  }

  #[test]
  fn a_word_longer_than_the_line_is_broken_into_lines() {
    let line = Line::raw("0123456789abcde");
    assert_eq!(plain(&wrap_line(line, 6)), ["012345", "6789ab", "cde"]);
    // And it is broken where the line it is on runs out rather than moved to
    // a line of its own, which it would not fit either: a block's indent and
    // gutter would otherwise be a row with nothing on it.
    let indented = Line::from(vec![Span::raw("  "), Span::raw("0123456789abcde")]);
    assert_eq!(plain(&wrap_line(indented, 6)), ["  0123", "456789", "abcde"]);
  }

  #[test]
  fn what_the_terminal_would_draw_for_itself_is_drawn_here_instead() {
    // A tab is no columns to us and four to the terminal, and a carriage
    // return would draw the rest of the line over the start of it: both are
    // settled here, where a row is still a row.
    assert_eq!(plain(&wrap_line(Line::raw("a\tb"), 20)), ["a    b"]);
    assert_eq!(plain(&wrap_line(Line::raw("done\r"), 20)), ["done"]);
    // A span carrying a newline is two lines, since it would be drawn as two,
    // and the second keeps the indent it was written with.
    let line = Line::from(Span::raw("one\n  two"));
    assert_eq!(plain(&wrap_line(line, 20)), ["one", "  two"]);
  }

  #[test]
  fn every_wrapped_line_fits_the_width_it_was_given() {
    let text = "Reticulating splines — 日本語のテキスト, a-very-long-unbroken-token, and\ttabs.";
    // From two columns, which is the narrowest a wide character fits in.
    for width in [2_u16, 4, 7, 20, 33] {
      for line in wrap_line(Line::raw(text), width) {
        assert!(line.width() <= width as usize, "{width}: {line:?}");
      }
    }
  }

  #[test]
  fn text_with_nothing_to_settle_is_not_copied() {
    assert!(matches!(printable("plain\n\tindented"), Cow::Borrowed(_)));
  }

  #[test]
  fn escape_sequences_go_whole() {
    assert_eq!(
      printable("\x1b[1;31mred\x1b[0m and \x1b[2K\x1b[Gplain"),
      "red and plain"
    );
    assert_eq!(
      printable("\x1b]0;title\x07\x1b]8;;https://a.b\x1b\\link\x1b]8;;\x1b\\"),
      "link"
    );
    assert_eq!(printable("\x1b(Bcharset \x1b=keypad \u{9b}31mC1"), "charset keypad C1");
    // Cut short, a sequence gives back the text it was about to swallow.
    assert_eq!(printable("stray\x1b\nnext \x1b[12\nline"), "stray\nnext \nline");
  }

  #[test]
  fn a_carriage_return_leaves_what_was_written_last() {
    assert_eq!(printable(" 10%\r 55%\r100%\ndone"), "100%\ndone");
    assert_eq!(printable("kept\r\nnext\r"), "kept\nnext");
  }

  #[test]
  fn a_backspace_takes_back_a_character() {
    assert_eq!(printable("_\x08Bo\x08old"), "Bold");
    assert_eq!(printable("a\n\x08b"), "a\nb");
  }

  #[test]
  fn other_controls_are_not_counted_as_columns() {
    let text = printable("bell\x07 nul\0 del\x7f nel\u{85} bad\u{fffd}");
    assert_eq!(text, "bell nul del nel bad\u{fffd}");
    assert_eq!(text.width(), text.chars().count());
  }

  #[test]
  fn invisibles_terminals_measure_their_own_way_go() {
    assert_eq!(
      printable("soft\u{ad}hyphen \u{202e}rtl\u{202c} \u{600}12 a\u{2028}b"),
      "softhyphen rtl 12 ab"
    );
    // A joiner is part of how an emoji is spelled, and stays.
    assert_eq!(
      printable("\u{1f468}\u{200d}\u{1f4bb}\u{ad}"),
      "\u{1f468}\u{200d}\u{1f4bb}"
    );
  }
}
