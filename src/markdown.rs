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
use ::markdown::mdast::{AlignKind, List, Node};
use mdstitch::{StitchOptions, stitch};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

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
    let style = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);
    if depth == 1 {
      style.add_modifier(Modifier::UNDERLINED)
    } else {
      style
    }
  }

  pub fn code_block() -> Style {
    Style::default().fg(Color::Green)
  }

  pub fn code_border() -> Style {
    Style::default().fg(Color::DarkGray)
  }

  pub fn inline_code() -> Style {
    Style::default().fg(Color::Cyan)
  }

  pub fn link() -> Style {
    Style::default().fg(Color::Blue).add_modifier(Modifier::UNDERLINED)
  }

  pub fn link_url() -> Style {
    Style::default().add_modifier(Modifier::DIM)
  }

  pub fn quote() -> Style {
    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)
  }

  pub fn quote_border() -> Style {
    Style::default().fg(Color::DarkGray)
  }

  pub fn rule() -> Style {
    Style::default().fg(Color::DarkGray)
  }

  pub fn bullet() -> Style {
    Style::default().fg(Color::Green)
  }

  pub fn footnote() -> Style {
    Style::default().fg(Color::Blue)
  }

  pub fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
  }
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
  refs.render_footnotes(&root, width, &mut out);
  // A trailing blank line would push the transcript's own spacing around.
  while out.last().is_some_and(|l| l.width() == 0) {
    out.pop();
  }
  out
}

/// Word-wraps plain text to `width` columns, for prose that is shown as
/// written rather than parsed — the reasoning block, which is the model
/// thinking aloud and not markdown it meant for us to render.
pub fn wrap_text(text: &str, width: u16, style: Style) -> Vec<Line<'static>> {
  let width = (width as usize).max(1);
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

/// What a message's body refers to but does not carry inline: footnotes and
/// link definitions. Both are written as their own blocks, so both have to be
/// resolved up front and held back from the flow.
///
/// Footnotes are numbered as GFM numbers them — by the order their references
/// appear, not the order they are defined — and one that nothing refers to is
/// dropped, as on github.com.
#[derive(Default)]
struct Refs {
  referenced: Vec<String>,
  /// Link and image definitions, by their (already lowercased) label.
  links: HashMap<String, String>,
}

impl Refs {
  fn collect(root: &Node) -> Self {
    let mut refs = Refs::default();
    refs.walk(root);
    refs
  }

  fn walk(&mut self, node: &Node) {
    match node {
      Node::FootnoteReference(reference) if !self.referenced.contains(&reference.identifier) => {
        self.referenced.push(reference.identifier.clone());
      }
      Node::Definition(definition) => {
        self.links.insert(definition.identifier.clone(), definition.url.clone());
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
  fn render_footnotes(&self, root: &Node, width: usize, out: &mut Vec<Line<'static>>) {
    let mut definitions: Vec<(usize, &Node)> = Vec::new();
    collect_footnote_definitions(root, self, &mut definitions);
    if definitions.is_empty() {
      return;
    }
    definitions.sort_by_key(|(number, _)| *number);
    out.push(Line::default());
    out.push(Line::styled("─".repeat(width.min(HR_MAX)), style::rule()));
    for (number, definition) in definitions {
      let marker = format!("[{number}] ");
      let continuation = " ".repeat(marker.width());
      let body = width.saturating_sub(marker.width()).max(1);
      let mut inner = Vec::new();
      blocks(children(definition), body, self, &mut inner);
      while inner.last().is_some_and(|l| l.width() == 0) {
        inner.pop();
      }
      for (i, line) in inner.into_iter().enumerate() {
        // A blank line between the note's paragraphs needs no indent; padding
        // it would only leave trailing whitespace behind.
        if i > 0 && line.width() == 0 {
          out.push(line);
          continue;
        }
        let lead = if i == 0 { &marker } else { &continuation };
        let style = if i == 0 { style::footnote() } else { Style::default() };
        out.push(prefix(Span::styled(lead.clone(), style), line));
      }
    }
  }
}

/// Pairs each referenced definition with its number, wherever it was written.
fn collect_footnote_definitions<'a>(node: &'a Node, refs: &Refs, out: &mut Vec<(usize, &'a Node)>) {
  if let Node::FootnoteDefinition(definition) = node {
    if let Some(number) = refs.number(&definition.identifier) {
      out.push((number, node));
    }
    return;
  }
  for child in children(node) {
    collect_footnote_definitions(child, refs, out);
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
      out.push(Line::styled(format!("```{lang}"), style::code_border()));
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
          None => vec![Span::styled(code.to_string(), style::code_block())],
        };
        for wrapped in wrap(spans, body) {
          out.push(prefix(Span::raw(indent.clone()), wrapped));
        }
      }
      out.push(Line::styled("```", style::code_border()));
    }
    Node::List(list) => list_block(list, 0, width, refs, out),
    Node::Blockquote(_) => {
      let body = width.saturating_sub(QUOTE_PREFIX.width()).max(1);
      let mut inner = Vec::new();
      blocks(children(node), body, refs, &mut inner);
      while inner.last().is_some_and(|l| l.width() == 0) {
        inner.pop();
      }
      for line in inner {
        // The quote style is a floor, not an override: inline code and links
        // inside a quote keep their own colour.
        let line = Line::from(
          line
            .spans
            .into_iter()
            .map(|span| {
              let style = style::quote().patch(span.style);
              Span::styled(span.content, style)
            })
            .collect::<Vec<_>>(),
        );
        out.push(prefix(Span::styled(QUOTE_PREFIX, style::quote_border()), line));
      }
    }
    Node::ThematicBreak(_) => {
      out.push(Line::styled("─".repeat(width.min(HR_MAX)), style::rule()));
    }
    Node::Table(table) => table_block(&table.children, &table.align, width, refs, out),
    Node::Html(html) => {
      for line in html.value.lines() {
        out.extend(wrap(vec![Span::styled(line.to_string(), style::dim())], width));
      }
    }
    Node::Math(math) => {
      for line in math.value.lines() {
        out.extend(wrap(vec![Span::raw(line.to_string())], width));
      }
    }
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
    let marker = format!("{indent}{bullet}{task}");
    let continuation = " ".repeat(marker.width());
    let body = width.saturating_sub(marker.width()).max(1);

    let mut inner: Vec<Line<'static>> = Vec::new();
    // Only the item's first line carries the marker; everything after it —
    // later paragraphs included — hangs under the text.
    let mut marked = false;
    let flush = |inner: &mut Vec<Line<'static>>, marked: &mut bool, out: &mut Vec<Line<'static>>| {
      while inner.last().is_some_and(|l| l.width() == 0) {
        inner.pop();
      }
      for line in inner.drain(..) {
        // A blank line between the item's paragraphs needs no indent; padding
        // it would only leave trailing whitespace behind.
        if *marked && line.width() == 0 {
          out.push(line);
          continue;
        }
        let lead = if *marked { &continuation } else { &marker };
        let style = if *marked { Style::default() } else { style::bullet() };
        out.push(prefix(Span::styled(lead.clone(), style), line));
        *marked = true;
      }
    };
    for child in &item.children {
      // A nested list indents from the outer width rather than the item's, so
      // its markers line up in a column of their own.
      if let Node::List(nested) = child {
        flush(&mut inner, &mut marked, out);
        list_block(nested, depth + 1, width, refs, out);
        marked = true;
        continue;
      }
      block(child, body, refs, &mut inner);
      if item.spread {
        inner.push(Line::default());
      }
    }
    flush(&mut inner, &mut marked, out);
    if !marked {
      out.push(Line::from(Span::styled(marker.clone(), style::bullet())));
    }
  }
}

/// A GFM table, with columns shrunk to fit and cells wrapped inside them.
fn table_block(rows: &[Node], align: &[AlignKind], width: usize, refs: &Refs, out: &mut Vec<Line<'static>>) {
  let cells: Vec<Vec<Vec<Span<'static>>>> = rows
    .iter()
    .map(|row| {
      children(row)
        .iter()
        .map(|cell| {
          let mut spans = Vec::new();
          inline(children(cell), Style::default(), refs, &mut spans);
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
        .map(|spans| spans.iter().map(|s| s.content.width()).sum::<usize>())
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
    Line::styled(s, style::rule())
  };

  out.push(rule("┌", "┬", "┐"));
  // A header-only table has nothing to separate from.
  let body_rows = cells.len() > 1;
  for (r, row) in cells.iter().enumerate() {
    // Every cell wraps to its column, then the row is as tall as the tallest.
    let wrapped: Vec<Vec<Line<'static>>> = (0..columns)
      .map(|c| {
        let spans = row.get(c).cloned().unwrap_or_default();
        let spans = if r == 0 {
          spans
            .into_iter()
            .map(|s| Span::styled(s.content, s.style.add_modifier(Modifier::BOLD)))
            .collect()
        } else {
          spans
        };
        wrap(spans, widths[c])
      })
      .collect();
    let height = wrapped.iter().map(Vec::len).max().unwrap_or(1).max(1);
    for line in 0..height {
      let mut spans = vec![Span::styled("│", style::rule())];
      for (c, column) in wrapped.iter().enumerate() {
        let content = column.get(line).cloned().unwrap_or_default();
        let used: usize = content.spans.iter().map(|s| s.content.width()).sum();
        let pad = widths[c].saturating_sub(used);
        let (before, after) = match align.get(c) {
          Some(AlignKind::Right) => (pad, 0),
          Some(AlignKind::Center) => (pad / 2, pad - pad / 2),
          _ => (0, pad),
        };
        spans.push(Span::raw(" ".repeat(before + 1)));
        spans.extend(content.spans);
        spans.push(Span::raw(" ".repeat(after + 1)));
        spans.push(Span::styled("│", style::rule()));
        let _ = c;
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
  inline(children, base.patch(style::link()), refs, out);
  let text: String = out[start..].iter().map(|s| s.content.as_ref()).collect();
  let bare = url.strip_prefix("mailto:").unwrap_or(url);
  if text != url && text != bare {
    out.push(Span::styled(format!(" ({url})"), base.patch(style::link_url())));
  }
}

/// An image is a placeholder here: the transcript cannot show one inline.
fn image_span(alt: &str, base: Style) -> Span<'static> {
  let alt = if alt.is_empty() { "image" } else { alt };
  Span::styled(format!("[{alt}]"), base.patch(style::link_url()))
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
      Node::InlineCode(code) => out.push(Span::styled(code.value.clone(), base.patch(style::inline_code()))),
      Node::InlineMath(math) => out.push(Span::styled(math.value.clone(), base)),
      Node::Link(link) => link_spans(&link.children, &link.url, base, refs, out),
      // `[text][label]`, whose target lives in a definition elsewhere. Without
      // resolving it the text would render as unmarked prose, losing both the
      // fact that it is a link and where it points.
      Node::LinkReference(reference) => match refs.links.get(&reference.identifier) {
        Some(url) => link_spans(&reference.children, &url.clone(), base, refs, out),
        None => inline(&reference.children, base, refs, out),
      },
      Node::Image(image) => out.push(image_span(&image.alt, base)),
      Node::ImageReference(reference) => out.push(image_span(&reference.alt, base)),
      Node::FootnoteReference(reference) => {
        // Numbered rather than named, so a `[^implementation-note]` does not
        // swallow the line it sits in. The list at the end uses the same
        // numbers, which is the only thing tying the two together here.
        if let Some(number) = refs.number(&reference.identifier) {
          out.push(Span::styled(format!("[{number}]"), base.patch(style::footnote())));
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

/// Puts `lead` in front of `line`, keeping the rest of its spans.
fn prefix(lead: Span<'static>, line: Line<'static>) -> Line<'static> {
  let mut spans = Vec::with_capacity(line.spans.len() + 1);
  spans.push(lead);
  spans.extend(line.spans);
  Line::from(spans)
}

/// Word-wraps a run of spans to `width`, splitting spans where needed and
/// carrying each span's style onto its fragments.
///
/// Measures in terminal columns, so CJK and emoji occupy the two cells they
/// actually take. A `\n` span (a hard break) forces a new line.
fn wrap(spans: Vec<Span<'static>>, width: usize) -> Vec<Line<'static>> {
  let width = width.max(1);
  let mut lines = Vec::new();
  let mut current: Vec<Span<'static>> = Vec::new();
  let mut used = 0;

  let mut push_line = |current: &mut Vec<Span<'static>>, used: &mut usize| {
    // The separator that fell at the break belongs to neither line.
    while current.last().is_some_and(|s| s.content.trim().is_empty()) {
      current.pop();
    }
    lines.push(Line::from(std::mem::take(current)));
    *used = 0;
  };

  for span in spans {
    let style = span.style;
    // A tab would otherwise be measured as one column and drawn as none.
    let content = expand_tabs(&span.content);
    for (i, chunk) in content.split('\n').enumerate() {
      if i > 0 {
        push_line(&mut current, &mut used);
      }
      for word in words(chunk) {
        let w = word.width();
        // Leading space on a fresh line is wrapping debris, not content.
        if word.trim().is_empty() && used == 0 {
          continue;
        }
        if used + w > width && used > 0 {
          push_line(&mut current, &mut used);
          if word.trim().is_empty() {
            continue;
          }
        }
        // A word longer than the line has to be broken somewhere.
        if w > width {
          let pieces = split(word, width);
          let last = pieces.len() - 1;
          for (i, piece) in pieces.into_iter().enumerate() {
            let piece_width = piece.width();
            current.push(Span::styled(piece, style));
            if i < last {
              push_line(&mut current, &mut used);
            } else {
              used += piece_width;
            }
          }
          continue;
        }
        current.push(Span::styled(word.to_string(), style));
        used += w;
      }
    }
  }
  while current.last().is_some_and(|s| s.content.trim().is_empty()) {
    current.pop();
  }
  if !current.is_empty() {
    lines.push(Line::from(current));
  }
  if lines.is_empty() {
    lines.push(Line::default());
  }
  lines
}

/// Cuts a word too long for the line into `width`-wide pieces.
fn split(word: &str, width: usize) -> Vec<String> {
  let mut pieces = Vec::new();
  let mut piece = String::new();
  let mut used = 0;
  for ch in word.chars() {
    let w = ch.to_string().width();
    if used + w > width && !piece.is_empty() {
      pieces.push(std::mem::take(&mut piece));
      used = 0;
    }
    piece.push(ch);
    used += w;
  }
  if !piece.is_empty() {
    pieces.push(piece);
  }
  pieces
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
    // Wrapping splits a span at each space, so the comment arrives in pieces.
    assert_eq!(span("//").style.fg, Some(Color::DarkGray));
    assert_eq!(span("note").style.fg, Some(Color::DarkGray));
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
}
