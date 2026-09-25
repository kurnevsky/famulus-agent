//! Built-in tools: read, write, edit, bash, ask.

use std::collections::HashMap;
use std::io::Write;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use rig_agent::tool::{Tool, ToolContext, ToolExecutionError};
use rig_core::message::{MimeType, ToolResultContent};
use rustix::process::{Pid, Signal, kill_process_group};

use crate::ask::{self, Dialog, Question};
use crate::modal::Host;
use crate::{edit, images, pdf};
use schemars::generate::SchemaSettings;
use schemars::transform::RecursiveTransform;
use schemars::{JsonSchema, Schema};
use serde::Deserialize;
use tokio::io::AsyncReadExt;

const MAX_LINES: usize = 2000;
const MAX_BYTES: usize = 50 * 1024;

/// The names the agent's own tools answer to.
pub const BUILT_IN: [&str; 5] = [
  ReadTool::NAME,
  BashTool::NAME,
  EditTool::NAME,
  WriteTool::NAME,
  AskTool::NAME,
];

/// Whether `name` is one of fa's own tools — the five built in, and the two
/// that read what MCP servers hold, since those are no one server's to
/// narrow. These are what `--tools` and `--no-tools` speak of; a server's
/// tools are narrowed in its own table instead.
pub fn own(name: &str) -> bool {
  #[cfg(feature = "mcp")]
  if crate::resources::NAMES.contains(&name) {
    return true;
  }
  BUILT_IN.contains(&name)
}

/// Which of fa's own tools a session keeps from the model: what `--tools` and
/// `--no-tools` came to as it started, which is when all of them are there.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rules {
  pub refused: Vec<String>,
}

impl Rules {
  pub fn permits(&self, name: &str) -> bool {
    !self.refused.iter().any(|n| n == name)
  }
}

/// Which of `available` to offer the model, given what was allowed and what
/// was refused, with the names asked for that nothing answers to. Nothing
/// asked for is everything offered.
///
/// An allow-list is the first word and a deny-list the last, so naming a tool
/// in both refuses it — the narrower intent wins, which is the safe way round
/// for a list whose point is usually to keep something away from the model.
pub fn choose(available: &[String], allow: &[String], deny: &[String]) -> (Vec<String>, Vec<String>) {
  let unknown: Vec<String> = allow
    .iter()
    .chain(deny)
    .filter(|name| !available.contains(name))
    .cloned()
    .collect();
  let chosen = available
    .iter()
    .filter(|name| (allow.is_empty() || allow.contains(name)) && !deny.contains(name))
    .cloned()
    .collect();
  (chosen, unknown)
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ToolError(String);

impl From<std::io::Error> for ToolError {
  fn from(e: std::io::Error) -> Self {
    ToolError(e.to_string())
  }
}

impl ToolError {
  #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
  pub fn new(message: impl Into<String>) -> Self {
    ToolError(message.into())
  }

  /// Our error messages are written for the model (e.g. "oldText not found"),
  /// so forward them instead of rig's redacted default feedback.
  pub fn into_execution_error(self) -> ToolExecutionError {
    ToolExecutionError::other(self.0.clone()).with_model_feedback(self.0)
  }
}

/// The parts of a tool that are the same for every one of them.
///
/// Declaring the argument type here is what keeps the schema from drifting
/// from it: `parameters` is generated from the same type rather than naming
/// it a second time. `map_error` forwards what the tool actually said — rig's
/// default redacts it, and fa's messages are written for the model to read.
macro_rules! tool_args {
  ($args:ty) => {
    type Args = $args;
    type Error = $crate::tools::ToolError;

    fn parameters(&self) -> serde_json::Value {
      $crate::tools::schema::<$args>()
    }

    fn map_error(&self, error: $crate::tools::ToolError) -> rig_agent::tool::ToolExecutionError {
      error.into_execution_error()
    }
  };
}
#[cfg_attr(not(feature = "mcp"), allow(unused_imports))]
pub(crate) use tool_args;

/// JSON schema for a tool's arguments, stripped of metadata that some
/// OpenAI-compatible servers reject (`$schema`, `title`, integer `format`).
///
/// Taken off every schema and subschema, and only those: an argument that
/// happens to be called `title` is a property rather than metadata, and stays.
pub fn schema<T: JsonSchema>() -> serde_json::Value {
  SchemaSettings::draft2020_12()
    .with(|settings| settings.meta_schema = None)
    .with_transform(RecursiveTransform(|schema: &mut Schema| {
      schema.remove("title");
      schema.remove("format");
    }))
    .into_generator()
    .into_root_schema_for::<T>()
    .to_value()
}

/// Path normalization: Unicode spaces folded, a leading `@` (chat file
/// reference) stripped, `~` expanded, `file://` URLs accepted, then resolved
/// against the working directory.
pub fn resolve(cwd: &Path, path: &str) -> PathBuf {
  resolve_literal(cwd, path.strip_prefix('@').unwrap_or(path))
}

/// `resolve`, for a path whose leading `@` is part of its name rather than a
/// reference to it.
pub fn resolve_literal(cwd: &Path, path: &str) -> PathBuf {
  let mut normalized: String = path
    .chars()
    .map(|c| match c {
      '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
      c => c,
    })
    .collect();
  if let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
    if normalized == "~" {
      return PathBuf::from(home);
    }
    if let Some(rest) = normalized.strip_prefix("~/") {
      return Path::new(&home).join(rest);
    }
  }
  if let Some(rest) = normalized.strip_prefix("file://") {
    normalized = rest.to_string();
  }
  let p = Path::new(&normalized);
  if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) }
}

/// For reads, name variants are also tried when the file is missing: NFD
/// normalization and curly apostrophes, as produced by macOS.
async fn resolve_read_path(cwd: &Path, path: &str) -> PathBuf {
  use unicode_normalization::UnicodeNormalization;
  let resolved = resolve(cwd, path);
  if tokio::fs::metadata(&resolved).await.is_ok() {
    return resolved;
  }
  let text = resolved.to_string_lossy().into_owned();
  let nfd: String = text.nfd().collect();
  let curly = text.replace('\'', "\u{2019}");
  let nfd_curly = nfd.replace('\'', "\u{2019}");
  for variant in [nfd, curly, nfd_curly] {
    if variant != text && tokio::fs::metadata(&variant).await.is_ok() {
      return PathBuf::from(variant);
    }
  }
  resolved
}

// ---------------------------------------------------------------- read

#[derive(Clone)]
pub struct ReadTool {
  pub cwd: PathBuf,
  /// Whether the model accepts images. When false, `read` describes the
  /// image but omits its data, with a note in its place.
  pub vision: bool,
}

#[derive(Deserialize, JsonSchema)]
pub struct ReadArgs {
  /// Path to the file to read (relative or absolute)
  path: String,
  /// Line number to start reading from (1-indexed)
  offset: Option<usize>,
  /// Maximum number of lines to read
  limit: Option<usize>,
}

impl Tool for ReadTool {
  const NAME: &'static str = "read";
  type Output = Vec<ToolResultContent>;
  tool_args!(ReadArgs);

  fn description(&self) -> String {
    format!(
      "Read the contents of a file. Supports text files, images (jpg, png, gif, webp, bmp) and PDFs. \
             Images are sent as attachments, PDFs as the Markdown of their text; other binary files are refused. \
             For text files and PDFs, output is truncated to {MAX_LINES} lines \
             or {}KB (whichever is hit first). Use offset/limit for large files. When you need the \
             full file, continue with offset until complete.",
      MAX_BYTES / 1024
    )
  }

  async fn call(&self, _ctx: &mut ToolContext, args: ReadArgs) -> Result<Vec<ToolResultContent>, ToolError> {
    let path = resolve_read_path(&self.cwd, &args.path).await;
    let bytes = tokio::fs::read(&path)
      .await
      .map_err(|e| ToolError(format!("{}: {e}", path.display())))?;

    if let Some(format) = images::detect(&bytes) {
      return Ok(read_image(&bytes, format, self.vision));
    }
    if pdf::detect(&bytes) {
      return read_pdf(bytes, &path, &args).await;
    }
    if is_binary(&bytes) {
      return Err(ToolError(format!(
        "{}: binary file ({}), not text. Use bash to inspect it (e.g. file, xxd, strings).",
        path.display(),
        format_size(bytes.len())
      )));
    }

    // Decode leniently, so a stray byte in some other encoding is shown as
    // a replacement character rather than refusing the file.
    let text = String::from_utf8_lossy(&bytes);
    let output = window(&text, args.offset, args.limit, |line| {
      format!("Use bash: sed -n '{line}p' {} | head -c {MAX_BYTES}", args.path)
    })?;
    Ok(vec![ToolResultContent::text(output)])
  }
}

/// The lines of `text` from `offset` (1-indexed) on, at most `limit` of them
/// and within `MAX_LINES` and `MAX_BYTES`, with a note saying where to go on
/// from when that is not the end. Lines are counted with a plain split, so a
/// trailing newline yields one final empty line. `wide` says how else to get
/// at a first line too long to show any of.
fn window(
  text: &str,
  offset: Option<usize>,
  limit: Option<usize>,
  wide: impl FnOnce(usize) -> String,
) -> Result<String, ToolError> {
  let all_lines: Vec<&str> = text.split('\n').collect();
  let total_lines = all_lines.len();
  let start = offset.map_or(0, |o| o.saturating_sub(1));
  let start_display = start + 1;
  // Only an offset can start past the end: there is always a first line.
  if start >= total_lines {
    return Err(ToolError(format!(
      "Offset {start_display} is beyond end of file ({total_lines} lines total)"
    )));
  }
  let end = limit.map_or(total_lines, |limit| (start + limit).min(total_lines));
  let selected = all_lines[start..end].join("\n");

  let cut = keep(&selected, false);
  Ok(if cut.partial() {
    format!(
      "[Line {start_display} is {}, exceeds {} limit. {}]",
      format_size(all_lines[start].len()),
      format_size(MAX_BYTES),
      wide(start_display)
    )
  } else if let Some(by) = cut.by {
    let end_display = start_display + cut.kept.len() - 1;
    let next = end_display + 1;
    format!(
      "{}\n\n[Showing lines {start_display}-{end_display} of {total_lines}{}. Use offset={next} to continue.]",
      cut.text(),
      by.limit()
    )
  } else if end < total_lines {
    let remaining = total_lines - end;
    let next = end + 1;
    format!("{selected}\n\n[{remaining} more lines in file. Use offset={next} to continue.]")
  } else {
    selected
  })
}

/// A PDF comes back as the Markdown of its text layer, paged through by line
/// like any text file, with a note for the pages that did not come through
/// whole.
async fn read_pdf(bytes: Vec<u8>, path: &Path, args: &ReadArgs) -> Result<Vec<ToolResultContent>, ToolError> {
  let pdf = pdf::extract(bytes)
    .await
    .map_err(|e| ToolError(format!("{}: {e}", path.display())))?;
  let note = pdf.note();
  if pdf.markdown.trim().is_empty() {
    return Ok(vec![ToolResultContent::text(note.unwrap_or_default())]);
  }
  let output = window(&pdf.markdown, args.offset, args.limit, |_| {
    "Use bash to extract the text (e.g. pdftotext).".to_string()
  })?;
  Ok(vec![ToolResultContent::text(match note {
    Some(note) => format!("{output}\n\n{note}"),
    None => output,
  })])
}

/// How much of the start of a file is looked at to tell whether it is text.
const SNIFF_BYTES: usize = 8 * 1024;

/// Whether a file's bytes look like something other than text: a NUL in its
/// first few kilobytes, or there more than one character in ten that no text
/// would hold — a control other than whitespace and escape, or a byte that is
/// not UTF-8. A stray byte in a file in some other encoding is let through,
/// to be decoded leniently.
fn is_binary(bytes: &[u8]) -> bool {
  let sample = &bytes[..bytes.len().min(SNIFF_BYTES)];
  if sample.contains(&0) {
    return true;
  }
  let (mut chars, mut odd) = (0, 0);
  for chunk in sample.utf8_chunks() {
    for c in chunk.valid().chars() {
      chars += 1;
      if c.is_control() && !matches!(c, '\t' | '\n' | '\r' | '\x0c' | '\x1b') {
        odd += 1;
      }
    }
    if !chunk.invalid().is_empty() {
      chars += 1;
      odd += 1;
    }
  }
  odd * 10 > chars
}

/// What of some output fits in `MAX_LINES` lines and `MAX_BYTES` bytes.
struct Cut<'a> {
  /// The output as it was given.
  content: &'a str,
  /// Every line, without the empty one after a final newline.
  lines: Vec<&'a str>,
  /// The lines kept: from the start, or up to the end, depending on which end
  /// was kept.
  kept: Vec<&'a str>,
  /// Which limit made the cut; `None` when everything fits.
  by: Option<TruncatedBy>,
  from_end: bool,
}

impl Cut<'_> {
  /// Whether the cut kept no line at all: one line on its own is longer than
  /// the byte budget.
  fn partial(&self) -> bool {
    self.by.is_some() && self.kept.is_empty()
  }

  /// What is left of the output. A line longer than the whole budget is cut
  /// inside itself, keeping the end that was being kept, and between
  /// characters rather than inside one.
  fn text(&self) -> String {
    if self.by.is_none() {
      return self.content.to_string();
    }
    if !self.partial() {
      return self.kept.join("\n");
    }
    match self.from_end {
      false => {
        let line = self.lines[0];
        line[..line.floor_char_boundary(MAX_BYTES)].to_string()
      }
      true => {
        let line = self.lines[self.lines.len() - 1];
        line[line.ceil_char_boundary(line.len() - MAX_BYTES)..].to_string()
      }
    }
  }
}

/// Keep whole lines within the limits, from the start of `content` or from
/// its end.
fn keep(content: &str, from_end: bool) -> Cut<'_> {
  let mut lines: Vec<&str> = content.split('\n').collect();
  if content.ends_with('\n') {
    lines.pop();
  }
  let (mut kept, by) = match from_end {
    true => within(lines.iter().rev().copied()),
    false => within(lines.iter().copied()),
  };
  if from_end {
    kept.reverse();
  }
  Cut {
    content,
    lines,
    kept,
    by,
    from_end,
  }
}

/// The lines, in the order given, until one of the limits is reached, and
/// which one it was.
fn within<'a>(lines: impl Iterator<Item = &'a str>) -> (Vec<&'a str>, Option<TruncatedBy>) {
  let mut kept = Vec::new();
  let mut bytes = 0;
  for line in lines {
    if kept.len() >= MAX_LINES {
      return (kept, Some(TruncatedBy::Lines));
    }
    let line_bytes = line.len() + usize::from(!kept.is_empty());
    if bytes + line_bytes > MAX_BYTES {
      return (kept, Some(TruncatedBy::Bytes));
    }
    kept.push(line);
    bytes += line_bytes;
  }
  (kept, None)
}

/// A tool result cut to the size the built-in tools keep to, or `None` when
/// it is already within it.
///
/// The built-in tools bound their output as they make it: `bash` never holds
/// more than a rolling tail, and `read` stops at the line it stops at. A
/// server's reply arrives whole and unasked, so the cut is made here, on what
/// came back — the head of it, since the front of an answer is the part that
/// answers, and the rest to a file the model can read if the head was not
/// enough.
///
/// What is kept is text: several blocks become one, because a JSON block cut
/// in half is no longer JSON, and a note saying where the rest went is worth
/// more to the model than a shape it cannot parse. Images are left alone and
/// in the order they came — they are bounded where they are decoded, and it
/// is the text that runs away with a context window.
pub fn cap_reply(content: &[ToolResultContent]) -> Option<Vec<ToolResultContent>> {
  let mut images = Vec::new();
  let mut said = Vec::new();
  for block in content {
    match block {
      ToolResultContent::Text(text) => said.push(text.text.clone()),
      ToolResultContent::Json { value } => said.push(value.to_string()),
      ToolResultContent::Image(_) => images.push(block.clone()),
    }
  }
  let text = said.join("\n");
  let cut = keep(&text, false);
  let by = cut.by?;
  let full = spill("fa-mcp", &text).map_or_else(|| "(unavailable)".to_string(), |p| p.display().to_string());
  let shown = cut.text();
  let kept = match cut.partial() {
    // One line longer than the whole budget — a JSON document written flat,
    // usually. There is no line to stop at, so it is cut where the budget
    // runs out.
    true => format!(
      "{shown}\n\n[Showing first {} of line 1 (line is {}). Full output: {full}]",
      format_size(shown.len()),
      format_size(cut.lines[0].len())
    ),
    false => format!(
      "{shown}\n\n[Showing lines 1-{} of {}{}. Full output: {full}]",
      cut.kept.len(),
      cut.lines.len(),
      by.limit()
    ),
  };
  let mut capped = vec![ToolResultContent::text(kept)];
  capped.extend(images);
  Some(capped)
}

/// Where output too long to hand over whole is put, so the note that cuts it
/// can say where the rest is.
fn temp_path(prefix: &str) -> PathBuf {
  let nanos = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .map_or(0, |d| d.as_nanos());
  std::env::temp_dir().join(format!("{prefix}-{:x}{:x}.log", nanos, std::process::id()))
}

/// The whole of a reply, written down once. A file that cannot be written is
/// a note that says so rather than a call that fails: the model still has the
/// head, which is the part it was going to read.
fn spill(prefix: &str, text: &str) -> Option<PathBuf> {
  let path = temp_path(prefix);
  std::fs::write(&path, text).ok()?;
  Some(path)
}

#[cfg(test)]
mod capping_tests {
  use super::*;

  fn image() -> ToolResultContent {
    ToolResultContent::image_base64("AAAA".to_string(), Some(rig_core::message::ImageMediaType::PNG), None)
  }

  /// The one text block a cut reply comes back as, and the note it ends with.
  fn cut(content: &[ToolResultContent]) -> (String, String) {
    let capped = cap_reply(content).expect("a reply over the limits is cut");
    assert!(
      matches!(capped.last(), Some(ToolResultContent::Image(_))),
      "the images it came with are kept, and last"
    );
    let ToolResultContent::Text(text) = &capped[0] else {
      panic!("what was said comes back as one text block")
    };
    let (body, note) = text.text.rsplit_once("\n\n").expect("a note saying what was cut");
    (body.to_string(), note.to_string())
  }

  fn full_output(note: &str) -> String {
    let path = note.rsplit("Full output: ").next().unwrap().trim_end_matches(']');
    let text = std::fs::read_to_string(path).expect("the whole of it, written down");
    std::fs::remove_file(path).expect("a file to remove");
    text
  }

  #[test]
  fn a_reply_within_the_limits_is_left_exactly_as_it_came() {
    let json = serde_json::json!({ "ok": true });
    let content = vec![ToolResultContent::text("short"), ToolResultContent::json(json), image()];
    assert!(cap_reply(&content).is_none(), "nothing to cut, so nothing is rewritten");
    // Every line and byte of the budget still fits.
    let edge = "x".repeat(MAX_BYTES);
    assert!(cap_reply(&[ToolResultContent::text(edge)]).is_none());
    // The newline after the last line is not a byte of any line.
    let edge = "x".repeat(MAX_BYTES) + "\n";
    assert!(cap_reply(&[ToolResultContent::text(edge)]).is_none());
    let edge = std::iter::repeat_n("y", MAX_LINES).collect::<Vec<_>>().join("\n");
    assert!(cap_reply(&[ToolResultContent::text(edge)]).is_none());
  }

  #[test]
  fn a_long_reply_keeps_its_head_and_says_where_the_rest_went() {
    let lines: Vec<String> = (1..=3000).map(|i| i.to_string()).collect();
    let content = vec![ToolResultContent::text(lines.join("\n")), image()];
    let (body, note) = cut(&content);
    assert!(body.starts_with("1\n2\n"), "the front of the answer, not the back");
    assert!(body.ends_with("\n2000"));
    assert_eq!(
      note.split(". Full output: ").next().unwrap(),
      "[Showing lines 1-2000 of 3000"
    );
    assert_eq!(full_output(&note).lines().count(), 3000);
  }

  #[test]
  fn the_byte_budget_is_the_other_way_to_run_out() {
    let lines: Vec<String> = (0..10).map(|_| "x".repeat(10 * 1024)).collect();
    let content = vec![ToolResultContent::text(lines.join("\n")), image()];
    let (body, note) = cut(&content);
    assert_eq!(body.lines().count(), 4);
    assert!(
      note.starts_with("[Showing lines 1-4 of 10 (50.0KB limit). Full output: "),
      "{note}"
    );
    assert_eq!(full_output(&note).len(), 10 * 10 * 1024 + 9);
  }

  #[test]
  fn one_line_longer_than_the_budget_is_cut_inside_it_and_between_characters() {
    // A JSON document written flat: there is no line to stop at, and the
    // character at the cut is three bytes wide, so the cut moves off it.
    let flat = "€".repeat(20_000);
    let content = vec![ToolResultContent::text(flat.clone()), image()];
    let (body, note) = cut(&content);
    assert!(body.len() <= MAX_BYTES && MAX_BYTES - body.len() < 3, "{}", body.len());
    assert!(flat.starts_with(&body), "the head of the line, verbatim");
    assert!(
      note.starts_with("[Showing first 50.0KB of line 1 (line is 58.6KB). Full output: "),
      "{note}"
    );
    assert_eq!(full_output(&note), flat);
  }
}

/// Image files come back as a note plus the image itself. Images the
/// pipeline cannot deliver are replaced by the reason.
pub const NON_VISION_NOTE: &str =
  "[Current model does not support images. The image will be omitted from this request.]";

pub fn read_image(bytes: &[u8], format: image::ImageFormat, vision: bool) -> Vec<ToolResultContent> {
  let processed = images::process(bytes, format);
  let mut note = match &processed {
    Ok(image) => image.note(format!("Read image file [{}]", image.media_type.to_mime_type())),
    Err(reason) => format!("Read image file [{}]\n{reason}", format.to_mime_type()),
  };
  if !vision {
    note.push('\n');
    note.push_str(NON_VISION_NOTE);
  }
  let mut content = vec![ToolResultContent::text(note)];
  if let Ok(image) = processed
    && vision
  {
    content.push(ToolResultContent::image_base64(
      image.base64(),
      Some(image.media_type),
      None,
    ));
  }
  content
}

#[cfg(test)]
mod read_tests {
  use super::*;

  async fn read(dir: &Path, path: &str, offset: Option<usize>, limit: Option<usize>) -> Result<String, ToolError> {
    let tool = ReadTool {
      cwd: dir.to_path_buf(),
      vision: true,
    };
    let out = tool
      .call(
        &mut ToolContext::new(),
        ReadArgs {
          path: path.into(),
          offset,
          limit,
        },
      )
      .await?;
    Ok(out.iter().filter_map(|c| c.as_text()).collect::<Vec<_>>().join("\n"))
  }

  fn dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fa-read-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
  }

  #[tokio::test]
  async fn notes_match_pi() {
    let dir = dir("notes");
    let big: String = (1..=3000).map(|i| format!("line {i}\n")).collect();
    std::fs::write(dir.join("big.txt"), &big).unwrap();
    let out = read(&dir, "big.txt", None, None).await.unwrap();
    assert!(out.starts_with("line 1\n"));
    // 3000 lines plus the empty line after the trailing newline.
    assert!(
      out.ends_with("line 2000\n\n[Showing lines 1-2000 of 3001. Use offset=2001 to continue.]"),
      "{}",
      &out[out.len() - 90..]
    );
    let out = read(&dir, "big.txt", Some(2990), None).await.unwrap();
    assert!(out.ends_with("line 3000\n"), "{out}");
    let out = read(&dir, "big.txt", Some(10), Some(2)).await.unwrap();
    assert_eq!(
      out,
      "line 10\nline 11\n\n[2990 more lines in file. Use offset=12 to continue.]"
    );
    let err = read(&dir, "big.txt", Some(5000), None).await.unwrap_err();
    assert_eq!(err.0, "Offset 5000 is beyond end of file (3001 lines total)");

    let wide = format!("{}\nshort\n", "x".repeat(60 * 1024));
    std::fs::write(dir.join("wide.txt"), wide).unwrap();
    let out = read(&dir, "wide.txt", None, None).await.unwrap();
    assert_eq!(
      out,
      "[Line 1 is 60.0KB, exceeds 50.0KB limit. Use bash: sed -n '1p' wide.txt | head -c 51200]"
    );
    let out = read(&dir, "wide.txt", Some(2), None).await.unwrap();
    assert_eq!(out, "short\n");
    std::fs::remove_dir_all(dir).unwrap();
  }

  #[tokio::test]
  async fn non_vision_model_gets_a_note_instead_of_the_image() {
    let dir = dir("novision");
    let img = image::ImageBuffer::from_fn(4, 4, |_, _| image::Rgb([1u8, 2, 3]));
    img.save(dir.join("pic.png")).unwrap();
    let tool = ReadTool {
      cwd: dir.clone(),
      vision: false,
    };
    let out = tool
      .call(
        &mut ToolContext::new(),
        ReadArgs {
          path: "pic.png".into(),
          offset: None,
          limit: None,
        },
      )
      .await
      .unwrap();
    assert_eq!(out.len(), 1, "image data omitted");
    assert_eq!(
      out[0].as_text().unwrap(),
      "Read image file [image/png]\n[Current model does not support images. The image will be omitted from this request.]"
    );
    std::fs::remove_dir_all(dir).unwrap();
  }

  #[tokio::test]
  async fn binary_is_refused() {
    let dir = dir("binary");
    std::fs::write(dir.join("nul.dat"), b"text\0more text").unwrap();
    let err = read(&dir, "nul.dat", None, None).await.unwrap_err();
    assert_eq!(
      err.0,
      format!(
        "{}: binary file (14B), not text. Use bash to inspect it (e.g. file, xxd, strings).",
        dir.join("nul.dat").display()
      )
    );
    // No NUL, but mostly bytes that are not UTF-8 and controls.
    std::fs::write(dir.join("noise.dat"), [0xff, 0xfe, 0x01, 0x02, b'o', b'k']).unwrap();
    assert!(read(&dir, "noise.dat", None, None).await.is_err());
    std::fs::write(dir.join("empty.txt"), "").unwrap();
    assert_eq!(read(&dir, "empty.txt", None, None).await.unwrap(), "");
    // Escapes and form feeds are what text written for a terminal holds.
    std::fs::write(dir.join("colour.txt"), "\x1b[1mbold\x1b[0m\x0c\r\n").unwrap();
    assert_eq!(
      read(&dir, "colour.txt", None, None).await.unwrap(),
      "\x1b[1mbold\x1b[0m\x0c\r\n"
    );
    std::fs::remove_dir_all(dir).unwrap();
  }

  /// A PDF of one page per string in `pages`, each drawn in Helvetica.
  fn pdf(pages: &[&str]) -> Vec<u8> {
    let n = pages.len();
    let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 4 + 2 * i)).collect();
    let mut objects = vec![
      "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
      format!("<< /Type /Pages /Kids [{}] /Count {n} >>", kids.join(" ")),
      "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
    ];
    for (i, text) in pages.iter().enumerate() {
      objects.push(format!(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {} 0 R >>",
        5 + 2 * i
      ));
      let stream = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET");
      objects.push(format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len()));
    }
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, object) in objects.iter().enumerate() {
      offsets.push(out.len());
      out.extend(format!("{} 0 obj\n{object}\nendobj\n", i + 1).bytes());
    }
    let xref = out.len();
    out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
    for offset in offsets {
      out.extend(format!("{offset:010} 00000 n \n").bytes());
    }
    out.extend(
      format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
        objects.len() + 1
      )
      .bytes(),
    );
    out
  }

  #[tokio::test]
  async fn a_pdf_is_read_as_markdown_and_paged_like_text() {
    let dir = dir("pdf");
    std::fs::write(dir.join("doc.pdf"), pdf(&["Hello from page one", "And page two"])).unwrap();
    let out = read(&dir, "doc.pdf", None, None).await.unwrap();
    assert_eq!(
      out,
      "<!-- Page 1 -->\n\n## Hello from page one\n\n<!-- Page 2 -->\n\n## And page two\n"
    );
    assert_eq!(
      read(&dir, "doc.pdf", Some(3), Some(1)).await.unwrap(),
      "## Hello from page one\n\n[5 more lines in file. Use offset=4 to continue.]"
    );

    // Nothing but the header is not a document.
    std::fs::write(dir.join("broken.pdf"), "%PDF-1.7\nnot really\n").unwrap();
    let err = read(&dir, "broken.pdf", None, None).await.unwrap_err();
    assert!(
      err.0.starts_with(&format!("{}: ", dir.join("broken.pdf").display())),
      "{}",
      err.0
    );
    std::fs::remove_dir_all(dir).unwrap();
  }

  #[tokio::test]
  async fn stray_bytes_are_decoded_leniently_and_paths_are_normalized() {
    let dir = dir("paths");
    std::fs::write(dir.join("latin1.txt"), b"caf\xe9 au lait").unwrap();
    assert_eq!(
      read(&dir, "@latin1.txt", None, None).await.unwrap(),
      "caf\u{FFFD} au lait"
    );
    // Curly apostrophe fallback, for macOS-named files.
    std::fs::write(dir.join("it\u{2019}s.txt"), "curly").unwrap();
    assert_eq!(read(&dir, "it's.txt", None, None).await.unwrap(), "curly");
    assert_eq!(resolve(Path::new("/w"), &format!("file://{}", dir.display())), dir);
    std::fs::remove_dir_all(dir).unwrap();
  }
}

// ---------------------------------------------------------------- write

#[derive(Clone)]
pub struct WriteTool {
  pub cwd: PathBuf,
}

#[derive(Deserialize, JsonSchema)]
pub struct WriteArgs {
  /// Path to the file to write (relative or absolute)
  path: String,
  /// Content to write to the file
  content: String,
}

impl Tool for WriteTool {
  const NAME: &'static str = "write";
  type Output = String;
  tool_args!(WriteArgs);

  fn description(&self) -> String {
    "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. \
         Automatically creates parent directories."
      .into()
  }

  async fn call(&self, _ctx: &mut ToolContext, args: WriteArgs) -> Result<String, ToolError> {
    let path = resolve(&self.cwd, &args.path);
    let _guard = lock_file(&path).await;
    if let Some(parent) = path.parent() {
      tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(&path, &args.content).await?;
    // Nothing is left for the transcript here: what a write put in the file
    // is what it was asked to, and the asking is already in the call.
    Ok(format!("Successfully wrote to {}", args.path))
  }
}

// ---------------------------------------------------------------- edit

/// Host-only detail attached to a file a tool changed: the numbered diff for
/// the UI. It is never sent to the model.
#[derive(Clone, Debug)]
pub struct EditDiff {
  pub diff: String,
}

/// Leave the change behind for the UI to draw, so a file's own before and
/// after is what the transcript shows rather than a sentence saying that
/// something happened.
///
/// A change of nothing leaves nothing: there is no diff to draw, and the
/// tool's own words are the whole story.
fn attach_diff(ctx: &mut ToolContext, before: &str, after: &str) {
  let diff = edit::generate_diff_string(before, after, 4);
  if !diff.trim().is_empty() {
    ctx.insert_result(EditDiff { diff });
  }
}

/// Per-file locks so concurrent edits/writes to one path are serialized.
static FILE_LOCKS: LazyLock<Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>> = LazyLock::new(Default::default);

async fn lock_file(path: &Path) -> tokio::sync::OwnedMutexGuard<()> {
  let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
  let lock = FILE_LOCKS
    .lock()
    .unwrap_or_else(|e| e.into_inner())
    .entry(key)
    .or_default()
    .clone();
  lock.lock_owned().await
}

fn io_error_code(error: &std::io::Error) -> String {
  use std::io::ErrorKind;
  let code = match error.kind() {
    ErrorKind::NotFound => "ENOENT",
    ErrorKind::PermissionDenied => "EACCES",
    ErrorKind::IsADirectory => "EISDIR",
    ErrorKind::NotADirectory => "ENOTDIR",
    _ => return error.to_string(),
  };
  format!("Error code: {code}")
}

#[derive(Clone)]
pub struct EditTool {
  pub cwd: PathBuf,
}

#[derive(Deserialize, JsonSchema)]
pub struct EditArgs {
  /// Path to the file to edit (relative or absolute)
  path: String,
  /// One or more targeted replacements. Each edit is matched against the original file, not
  /// incrementally. Do not include overlapping or nested edits. If two changes touch the same
  /// block or nearby lines, merge them into one edit instead.
  edits: Vec<edit::Edit>,
}

impl Tool for EditTool {
  const NAME: &'static str = "edit";
  type Output = String;
  tool_args!(EditArgs);

  fn description(&self) -> String {
    "Edit a single file using exact text replacement. Every edits[].oldText must match a \
         unique, non-overlapping region of the original file. If two changes affect the same \
         block or nearby lines, merge them into one edit instead of emitting overlapping edits. \
         Do not include large unchanged regions just to connect distant changes."
      .into()
  }

  async fn call(&self, ctx: &mut ToolContext, args: EditArgs) -> Result<String, ToolError> {
    if args.edits.is_empty() {
      return Err(ToolError(
        "Edit tool input is invalid. edits must contain at least one replacement.".into(),
      ));
    }
    let path = resolve(&self.cwd, &args.path);
    let _guard = lock_file(&path).await;

    // A directory, a missing file and one that may not be written to are all
    // found out by trying, and all reported the same way.
    let could_not_edit =
      |e: std::io::Error| ToolError(format!("Could not edit file: {}. {}.", args.path, io_error_code(&e)));
    let raw = tokio::fs::read_to_string(&path).await.map_err(could_not_edit)?;

    let (bom, content) = edit::split_bom(&raw);
    let ending = edit::detect_line_ending(content);
    let normalized = edit::normalize_to_lf(content);
    let applied = edit::apply_edits(&normalized, &args.edits, &args.path).map_err(ToolError)?;

    let final_content = format!("{bom}{}", edit::restore_line_endings(&applied.new, ending));
    tokio::fs::write(&path, final_content).await.map_err(could_not_edit)?;

    attach_diff(ctx, &applied.base, &applied.new);
    Ok(format!(
      "Successfully replaced {} block(s) in {}.",
      args.edits.len(),
      args.path
    ))
  }
}

#[cfg(test)]
mod choosing_tests {
  use super::*;

  fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| name.to_string()).collect()
  }

  #[test]
  fn an_allow_list_is_the_first_word_and_a_deny_list_the_last() {
    let there = names(&["read", "write", "edit", "bash", "weather"]);
    let none: Vec<String> = Vec::new();

    // Nothing asked for is everything offered.
    assert_eq!(choose(&there, &none, &none), (there.clone(), none.clone()));

    // Only these.
    let (chosen, unknown) = choose(&there, &names(&["read", "weather"]), &none);
    assert_eq!(chosen, names(&["read", "weather"]));
    assert!(unknown.is_empty());

    // Everything but these — which is how a session goes read-only.
    let (chosen, _) = choose(&there, &none, &names(&["write", "edit", "bash"]));
    assert_eq!(chosen, names(&["read", "weather"]));

    // Named in both, a tool is refused: the narrower intent wins, which is
    // the safe way round for a list meant to keep something away.
    let (chosen, _) = choose(&there, &names(&["read", "bash"]), &names(&["bash"]));
    assert_eq!(chosen, names(&["read"]));

    // A name nothing answers to is handed back, since a typo in a list like
    // this is a tool quietly left in or out.
    let (chosen, unknown) = choose(&there, &names(&["reed"]), &names(&["bahs"]));
    assert_eq!(unknown, names(&["reed", "bahs"]));
    assert!(chosen.is_empty(), "a misspelled allow-list allows nothing");
  }
}

#[cfg(test)]
mod write_tests {
  use super::*;

  async fn write(dir: &Path, content: &str) -> (Result<String, ToolError>, ToolContext) {
    let mut ctx = ToolContext::new();
    let result = WriteTool { cwd: dir.to_path_buf() }
      .call(
        &mut ctx,
        WriteArgs {
          path: "f.txt".into(),
          content: content.into(),
        },
      )
      .await;
    (result, ctx)
  }

  #[tokio::test]
  async fn a_write_puts_the_file_down_and_leaves_nothing_behind() {
    let dir = std::env::temp_dir().join(format!("fa-write-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let (result, ctx) = write(&dir, "one\ntwo\n").await;
    assert_eq!(result.unwrap(), "Successfully wrote to f.txt");
    assert_eq!(std::fs::read_to_string(dir.join("f.txt")).unwrap(), "one\ntwo\n");
    // Nothing of its own: what a write put in the file is what it was asked
    // to put there, and the asking is already in the transcript, so that is
    // where the transcript reads it from.
    assert!(ctx.result::<EditDiff>().is_none(), "a write is not a change to mark up");

    // The same on a rewrite, which is still just the file as it now reads.
    let (result, ctx) = write(&dir, "one\ntwo!\n").await;
    assert!(result.is_ok());
    assert!(ctx.result::<EditDiff>().is_none());
    std::fs::remove_dir_all(&dir).unwrap();
  }
}

#[cfg(test)]
mod edit_tests {
  use super::*;

  fn tool(dir: &Path) -> EditTool {
    EditTool { cwd: dir.to_path_buf() }
  }

  fn temp_file(name: &str, content: &[u8]) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("fa-edit-{}-{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("f.txt");
    std::fs::write(&file, content).unwrap();
    (dir, file)
  }

  async fn run(dir: &Path, edits: Vec<(&str, &str)>) -> (Result<String, ToolError>, ToolContext) {
    let mut ctx = ToolContext::new();
    let result = tool(dir)
      .call(
        &mut ctx,
        EditArgs {
          path: "f.txt".into(),
          edits: edits
            .into_iter()
            .map(|(o, n)| edit::Edit {
              old_text: o.into(),
              new_text: n.into(),
            })
            .collect(),
        },
      )
      .await;
    (result, ctx)
  }

  #[tokio::test]
  async fn edits_preserve_bom_and_crlf_and_report_diff() {
    let (dir, file) = temp_file("crlf", "\u{FEFF}one\r\ntwo\r\nthree\r\n".as_bytes());
    let (result, ctx) = run(&dir, vec![("two", "2")]).await;
    assert_eq!(result.unwrap(), "Successfully replaced 1 block(s) in f.txt.");
    assert_eq!(
      std::fs::read(&file).unwrap(),
      "\u{FEFF}one\r\n2\r\nthree\r\n".as_bytes()
    );
    let diff = ctx.result::<EditDiff>().expect("diff attached for the UI");
    assert_eq!(diff.text_lines(), [" 1 one", "-2 two", "+2 2", " 3 three"]);
    std::fs::remove_dir_all(dir).unwrap();
  }

  impl EditDiff {
    fn text_lines(&self) -> Vec<&str> {
      self.diff.lines().collect()
    }
  }

  #[tokio::test]
  async fn fuzzy_whitespace_match_and_pi_errors() {
    // Trailing spaces on line 1 defeat the exact match; the fuzzy match
    // rewrites lines 1-2 from normalized text and leaves line 3 untouched.
    let (dir, file) = temp_file("fuzzy", b"fn a() {}   \nfn b() {}\nfn c() {}   \n");
    let (result, _) = run(&dir, vec![("fn a() {}\nfn b() {}", "fn a() { 1 }\nfn b() { 2 }")]).await;
    result.unwrap();
    assert_eq!(
      std::fs::read_to_string(&file).unwrap(),
      "fn a() { 1 }\nfn b() { 2 }\nfn c() {}   \n"
    );

    let (result, _) = run(&dir, vec![("nope", "x")]).await;
    assert_eq!(
      result.unwrap_err().0,
      "Could not find the exact text in f.txt. The old text must match exactly including all whitespace and newlines."
    );
    let (result, _) = run(&dir, vec![]).await;
    assert_eq!(
      result.unwrap_err().0,
      "Edit tool input is invalid. edits must contain at least one replacement."
    );
    std::fs::remove_dir_all(&dir).unwrap();
    let (result, _) = run(&dir, vec![("a", "b")]).await;
    assert_eq!(result.unwrap_err().0, "Could not edit file: f.txt. Error code: ENOENT.");
    // Found out by reading it, like the rest.
    std::fs::create_dir_all(dir.join("f.txt")).unwrap();
    let (result, _) = run(&dir, vec![("a", "b")]).await;
    assert_eq!(result.unwrap_err().0, "Could not edit file: f.txt. Error code: EISDIR.");
    std::fs::remove_dir_all(&dir).unwrap();
  }
}

// ---------------------------------------------------------------- bash

/// Where a running command reports throttled snapshots of its output, put
/// into its context by the loop that dispatched it.
///
/// Bound to the call it answers for before the tool ever sees it, so the
/// transcript can put the output under the right line when several commands
/// run at once — and the command itself never has to know which one it is.
/// Nothing about the call travels through the arguments to tell it, and
/// nothing about it is offered to the model.
#[derive(Clone)]
pub struct Output(pub Arc<dyn Fn(String) + Send + Sync>);

/// Minimum interval between live output snapshots.
const UPDATE_THROTTLE: Duration = Duration::from_millis(100);
/// After the process exits, how long to keep draining pipes held open by
/// lingering background children before giving up on them.
const DRAIN_GRACE: Duration = Duration::from_millis(500);

#[derive(Clone)]
pub struct BashTool {
  pub cwd: PathBuf,
}

#[derive(Deserialize, JsonSchema)]
pub struct BashArgs {
  /// Shell command to execute
  command: String,
  /// Timeout in seconds (optional, no default timeout)
  timeout: Option<f64>,
}

impl Tool for BashTool {
  const NAME: &'static str = "bash";
  type Output = String;
  tool_args!(BashArgs);

  fn description(&self) -> String {
    format!(
      "Execute a bash command in the current working directory. Returns stdout and stderr. \
             Output is truncated to last {MAX_LINES} lines or {}KB (whichever is hit first). \
             If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
      MAX_BYTES / 1024
    )
  }

  async fn call(&self, ctx: &mut ToolContext, args: BashArgs) -> Result<String, ToolError> {
    let timeout = match args.timeout {
      Some(t) if !(t.is_finite() && t > 0.0) => {
        return Err(ToolError("Invalid timeout: must be a finite number of seconds".into()));
      }
      other => other.map(Duration::from_secs_f64),
    };
    if !self.cwd.is_dir() {
      return Err(ToolError(format!(
        "Working directory does not exist: {}\nCannot execute bash commands.",
        self.cwd.display()
      )));
    }

    // Own process group so a timeout or abort can kill the whole tree,
    // not just the shell.
    // One pipe for both streams, not one each. Two pipes carry no record of
    // which was written first, so a command that says something on each ends
    // up reported in whichever order they happened to be read — while down a
    // single pipe the kernel keeps the order the command wrote in, which is
    // the order a terminal would have shown and the order the model should
    // read. It is what `2>&1` does, done here so the command need not.
    let (reads, writes) = std::io::pipe()?;
    let mut child = tokio::process::Command::new("bash")
      .arg("-c")
      .arg(&args.command)
      .current_dir(&self.cwd)
      .stdin(Stdio::null())
      .stdout(Stdio::from(writes.try_clone()?))
      .stderr(Stdio::from(writes))
      .process_group(0)
      .spawn()?;
    let mut guard = ProcessGroupGuard::new(child.id());
    // Both write ends belong to the child now; ours are gone with the
    // builder, which is what lets this see the end of the output at all.
    let mut merged = tokio::net::unix::pipe::Receiver::from_owned_fd(reads.into())?;

    let mut output = OutputAccumulator::new();
    let mut throttle = UpdateThrottle::new(ctx.get::<Output>().cloned());
    // The timeout while the command runs; once it has exited, how long what
    // it left behind may keep the pipe open.
    let mut deadline = timeout.map(|t| tokio::time::Instant::now() + t);
    let mut exit = None;
    let mut timed_out = false;
    let mut open = true;
    let mut buf = [0u8; 8192];

    while open || exit.is_none() {
      tokio::select! {
          r = merged.read(&mut buf), if open => match r {
              Ok(n) if n > 0 => {
                  output.append(&buf[..n]);
                  throttle.maybe_emit(&output);
              }
              _ => open = false,
          },
          status = child.wait(), if exit.is_none() => {
              exit = Some(status?);
              deadline = Some(tokio::time::Instant::now() + DRAIN_GRACE);
          }
          _ = tokio::time::sleep_until(deadline.unwrap_or_else(tokio::time::Instant::now)), if deadline.is_some() => {
              if exit.is_none() {
                  timed_out = true;
                  guard.kill();
                  deadline = None;
              } else {
                  // Process is gone but something still holds the pipes.
                  break;
              }
          }
      }
    }
    // The loop only ends once the command has exited.
    let status = exit.expect("the command has exited");
    if !timed_out {
      guard.disarm();
    }

    let text = output.render(if timed_out { "" } else { "(no output)" });
    if timed_out {
      let secs = args.timeout.unwrap_or_default();
      return Err(ToolError(append_status(
        &text,
        &format!("Command timed out after {secs} seconds"),
      )));
    }
    let code = status
      .code()
      .unwrap_or_else(|| status.signal().map_or(1, |sig| 128 + sig));
    if code != 0 {
      return Err(ToolError(append_status(
        &text,
        &format!("Command exited with code {code}"),
      )));
    }
    Ok(text)
  }
}

fn append_status(text: &str, status: &str) -> String {
  if text.is_empty() {
    status.to_string()
  } else {
    format!("{text}\n\n{status}")
  }
}

/// Kills the command's process group when dropped, unless disarmed after a
/// normal exit. Dropping happens on timeout and when the run is aborted.
struct ProcessGroupGuard {
  pgid: Option<Pid>,
}

impl ProcessGroupGuard {
  fn new(pid: Option<u32>) -> Self {
    Self {
      pgid: pid.and_then(|p| i32::try_from(p).ok()).and_then(Pid::from_raw),
    }
  }

  /// The command was started as the leader of its own group, so the group is
  /// all of it, and a failure means nothing is left in it to kill.
  fn kill(&self) {
    if let Some(pgid) = self.pgid {
      let _ = kill_process_group(pgid, Signal::KILL);
    }
  }

  fn disarm(&mut self) {
    self.pgid = None;
  }
}

impl Drop for ProcessGroupGuard {
  fn drop(&mut self) {
    self.kill();
  }
}

struct UpdateThrottle {
  sink: Option<Output>,
  last: Option<Instant>,
}

impl UpdateThrottle {
  fn new(sink: Option<Output>) -> Self {
    Self { sink, last: None }
  }

  fn maybe_emit(&mut self, output: &OutputAccumulator) {
    let Some(Output(sink)) = &self.sink else { return };
    if self.last.is_some_and(|t| t.elapsed() < UPDATE_THROTTLE) {
      return;
    }
    self.last = Some(Instant::now());
    sink(output.tail());
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TruncatedBy {
  Lines,
  Bytes,
}

impl TruncatedBy {
  /// What a note says about the limit that was reached: the byte budget by
  /// name, since the line count is already in the lines the note gives.
  fn limit(self) -> String {
    match self {
      TruncatedBy::Lines => String::new(),
      TruncatedBy::Bytes => format!(" ({} limit)", format_size(MAX_BYTES)),
    }
  }
}

/// Streams command output, keeping only a rolling tail in memory and spilling
/// the complete output to a temp file once it exceeds the model-facing limits.
struct OutputAccumulator {
  tail: Vec<u8>,
  tail_starts_at_line_boundary: bool,
  total_bytes: usize,
  completed_lines: usize,
  open_line: bool,
  current_line_bytes: usize,
  pending: Vec<u8>,
  temp: Option<(PathBuf, std::fs::File)>,
}

impl OutputAccumulator {
  const MAX_ROLLING_BYTES: usize = MAX_BYTES * 2;

  fn new() -> Self {
    Self {
      tail: Vec::new(),
      tail_starts_at_line_boundary: true,
      total_bytes: 0,
      completed_lines: 0,
      open_line: false,
      current_line_bytes: 0,
      pending: Vec::new(),
      temp: None,
    }
  }

  fn total_lines(&self) -> usize {
    self.completed_lines + usize::from(self.open_line)
  }

  fn append(&mut self, data: &[u8]) {
    self.total_bytes += data.len();
    match data.iter().rposition(|&b| b == b'\n') {
      Some(last) => {
        self.completed_lines += data.iter().filter(|&&b| b == b'\n').count();
        self.current_line_bytes = data.len() - last - 1;
        self.open_line = self.current_line_bytes > 0;
      }
      None => {
        self.current_line_bytes += data.len();
        self.open_line = true;
      }
    }

    self.tail.extend_from_slice(data);
    if self.tail.len() > Self::MAX_ROLLING_BYTES * 2 {
      let mut start = self.tail.len() - Self::MAX_ROLLING_BYTES;
      while start < self.tail.len() && (self.tail[start] & 0xc0) == 0x80 {
        start += 1;
      }
      self.tail_starts_at_line_boundary = self.tail[start - 1] == b'\n';
      self.tail.drain(..start);
    }

    if self.temp.is_some() || self.exceeds_limits() {
      self.ensure_temp_file();
      if let Some((_, file)) = &mut self.temp {
        let _ = file.write_all(data);
      }
    } else {
      self.pending.extend_from_slice(data);
    }
  }

  fn exceeds_limits(&self) -> bool {
    self.total_bytes > MAX_BYTES || self.total_lines() > MAX_LINES
  }

  fn ensure_temp_file(&mut self) {
    if self.temp.is_some() {
      return;
    }
    let path = temp_path("fa-bash");
    if let Ok(mut file) = std::fs::File::create(&path) {
      let _ = file.write_all(&self.pending);
      self.pending = Vec::new();
      self.temp = Some((path, file));
    }
  }

  fn visible_tail(&self) -> String {
    let text = String::from_utf8_lossy(&self.tail);
    if self.tail_starts_at_line_boundary {
      return text.into_owned();
    }
    match text.find('\n') {
      Some(i) => text[i + 1..].to_string(),
      None => text.into_owned(),
    }
  }

  /// The end of the output as it stands, for showing while it runs.
  fn tail(&self) -> String {
    keep(&self.visible_tail(), true).text()
  }

  /// Model-facing rendering: the tail, then a note on what was cut and where
  /// the full output lives.
  fn render(&self, empty_text: &str) -> String {
    let tail = self.visible_tail();
    let cut = keep(&tail, true);
    let shown = cut.text();
    let mut text = match shown.is_empty() {
      true => empty_text.to_string(),
      false => shown.clone(),
    };
    if !self.exceeds_limits() {
      return text;
    }
    let path = self
      .temp
      .as_ref()
      .map_or_else(|| "(unavailable)".to_string(), |(p, _)| p.display().to_string());
    let total_lines = self.total_lines();
    if cut.partial() {
      text.push_str(&format!(
        "\n\n[Showing last {} of line {total_lines} (line is {}). Full output: {path}]",
        format_size(shown.len()),
        format_size(self.current_line_bytes)
      ));
      return text;
    }
    let start_line = total_lines - cut.kept.len() + 1;
    // A tail that fits whole was cut already, by the rolling window: it is the
    // output in front of it that ran over.
    let by = cut.by.unwrap_or(match self.total_bytes > MAX_BYTES {
      true => TruncatedBy::Bytes,
      false => TruncatedBy::Lines,
    });
    text.push_str(&format!(
      "\n\n[Showing lines {start_line}-{total_lines} of {total_lines}{}. Full output: {path}]",
      by.limit()
    ));
    text
  }
}

fn format_size(bytes: usize) -> String {
  if bytes < 1024 {
    format!("{bytes}B")
  } else if bytes < 1024 * 1024 {
    format!("{:.1}KB", bytes as f64 / 1024.0)
  } else {
    format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
  }
}

#[cfg(test)]
mod bash_tests {
  #[test]
  fn a_tool_offers_what_it_acts_on_first() {
    // A model writes arguments in the order it is offered them, and the
    // transcript can only show what has arrived — so a call whose path came
    // last would be a nameless block of text until it finished. Sorting the
    // properties (serde_json's default) put `content` before `path`.
    let first = |schema: serde_json::Value| schema["properties"].as_object().unwrap().keys().next().unwrap().clone();
    assert_eq!(first(schema::<WriteArgs>()), "path");
    assert_eq!(first(schema::<EditArgs>()), "path");
    assert_eq!(first(schema::<ReadArgs>()), "path");
    assert_eq!(first(schema::<BashArgs>()), "command");
  }

  #[test]
  fn an_argument_named_like_metadata_is_still_an_argument() {
    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct Titled {
      title: String,
      format: Option<u32>,
    }
    let schema = schema::<Titled>();
    let properties = schema["properties"].as_object().unwrap();
    assert!(
      properties.contains_key("title") && properties.contains_key("format"),
      "{schema}"
    );
    assert_eq!(schema["required"], serde_json::json!(["title"]));
    // The metadata itself still goes, at the top and inside.
    assert!(
      schema.get("title").is_none() && schema.get("$schema").is_none(),
      "{schema}"
    );
    assert!(properties["format"].get("format").is_none(), "{schema}");
  }

  #[tokio::test]
  async fn a_command_is_told_which_call_it_is_without_being_asked_for_it() {
    // A command reports its output down whatever it was handed, and is
    // told nothing else — so the model is never offered a field that is
    // not its own, and the arguments it sends are only ever its own too.
    let schema = schema::<BashArgs>();
    let properties = schema["properties"].as_object().unwrap();
    assert_eq!(properties.len(), 2, "{schema}");
    assert!(properties.contains_key("command") && properties.contains_key("timeout"));

    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = seen.clone();
    let mut ctx = ToolContext::new();
    ctx.insert(Output(Arc::new(move |text| sink.lock().unwrap().push(text))));
    let args = BashArgs {
      command: "echo first; sleep 0.3; echo second".into(),
      timeout: None,
    };
    BashTool {
      cwd: std::env::temp_dir(),
    }
    .call(&mut ctx, args)
    .await
    .unwrap();
    assert!(
      seen.lock().unwrap().iter().any(|text| text == "first\n"),
      "output goes where the dispatcher said, as it arrives"
    );
  }

  use super::*;
  use std::sync::Mutex;

  fn tool() -> BashTool {
    BashTool {
      cwd: std::env::temp_dir(),
    }
  }

  async fn run(tool: &BashTool, command: &str, timeout: Option<f64>) -> Result<String, ToolError> {
    tool
      .call(
        &mut ToolContext::new(),
        BashArgs {
          command: command.into(),
          timeout,
        },
      )
      .await
  }

  #[tokio::test]
  async fn merges_streams_and_reports_exit_code() {
    let err = run(&tool(), "echo out; echo err 1>&2; exit 3", None).await.unwrap_err();
    assert_eq!(err.0, "out\nerr\n\n\nCommand exited with code 3");
    assert_eq!(run(&tool(), "true", None).await.unwrap(), "(no output)");
    assert_eq!(run(&tool(), "printf hi", None).await.unwrap(), "hi");
  }

  #[tokio::test]
  async fn the_two_streams_are_read_in_the_order_they_were_written() {
    // Both go down one pipe, so the kernel keeps the order the command wrote
    // in. Read from a pipe each, this would be whichever happened to be
    // polled first — and it was, about one run in eight.
    let interleaved = run(&tool(), "echo a; echo b 1>&2; echo c; echo d 1>&2", None)
      .await
      .unwrap();
    assert_eq!(interleaved, "a\nb\nc\nd\n");
    // Starting on the error stream, which a rule as simple as "take stdout
    // first" would put the wrong way round.
    let error_first = run(&tool(), "echo first 1>&2; echo second", None).await.unwrap();
    assert_eq!(error_first, "first\nsecond\n");
  }

  #[tokio::test]
  async fn timeout_kills_the_process_group() {
    let started = Instant::now();
    let err = run(&tool(), "echo before; sleep 30 | cat; echo after", Some(0.3))
      .await
      .unwrap_err();
    assert!(started.elapsed() < Duration::from_secs(5), "{:?}", started.elapsed());
    assert_eq!(err.0, "before\n\n\nCommand timed out after 0.3 seconds");
    assert!(
      run(&tool(), "x", Some(-1.0))
        .await
        .unwrap_err()
        .0
        .contains("Invalid timeout")
    );
  }

  #[tokio::test]
  async fn long_output_is_tail_truncated_and_saved() {
    let text = run(&tool(), "seq 1 3000", None).await.unwrap();
    let (body, note) = text.rsplit_once("\n\n").unwrap();
    assert!(body.starts_with("1001\n"), "{}", &body[..20]);
    assert!(body.ends_with("\n3000"));
    let prefix = "[Showing lines 1001-3000 of 3000. Full output: ";
    assert!(note.starts_with(prefix), "{note}");
    let path = note[prefix.len()..].trim_end_matches(']');
    let full = std::fs::read_to_string(path).unwrap();
    assert_eq!(full.lines().count(), 3000);
    std::fs::remove_file(path).unwrap();
  }

  #[tokio::test]
  async fn byte_limit_note_and_partial_line() {
    // Ten lines of 10KB exceed 50KB by bytes, not lines.
    let text = run(
      &tool(),
      "for i in $(seq 1 10); do head -c 10240 /dev/zero | tr '\\0' x; echo; done",
      None,
    )
    .await
    .unwrap();
    let note = text.rsplit("\n\n").next().unwrap();
    assert!(
      note.starts_with("[Showing lines 7-10 of 10 (50.0KB limit). Full output: "),
      "{note}"
    );
    // A single 60KB line keeps only its end.
    let text = run(&tool(), "head -c 61440 /dev/zero | tr '\\0' y", None)
      .await
      .unwrap();
    let note = text.rsplit("\n\n").next().unwrap();
    assert!(
      note.starts_with("[Showing last 50.0KB of line 1 (line is 60.0KB). Full output: "),
      "{note}"
    );
    let path = text.rsplit("Full output: ").next().unwrap().trim_end_matches(']');
    let _ = std::fs::remove_file(path);
  }

  #[tokio::test]
  async fn missing_cwd_is_reported() {
    let tool = BashTool {
      cwd: PathBuf::from("/nonexistent/dir"),
    };
    let err = run(&tool, "true", None).await.unwrap_err();
    assert!(err.0.starts_with("Working directory does not exist: /nonexistent/dir"));
  }
}

// ---------------------------------------------------------------- ask

/// Puts a questionnaire to the user through `host`.
#[derive(Clone)]
pub struct AskTool {
  pub host: Host,
}

#[derive(Deserialize, JsonSchema)]
pub struct AskArgs {
  /// Questions to ask the user (1-4 questions)
  #[schemars(length(min = 1, max = ask::MAX_QUESTIONS as u32))]
  questions: Vec<Question>,
}

impl Tool for AskTool {
  const NAME: &'static str = "ask";
  type Output = String;
  tool_args!(AskArgs);

  /// `rpiv-ask-user-question`'s own description, less what it says about
  /// `preview` — an option here carries no artifact to put beside the list,
  /// and a description advertising a field the schema does not have would only
  /// teach the model to send one.
  fn description(&self) -> String {
    "Ask the user one or more structured questions during execution. Use when you need to:\n\
         1. Gather user preferences or requirements\n\
         2. Clarify ambiguous instructions\n\
         3. Get decisions on implementation choices as you work\n\
         4. Offer choices to the user about what direction to take\n\n\
         Usage notes:\n\
         - Users can type a custom answer via the automatically appended \"Type something.\" row on \
         every question or press Esc to abandon the questionnaire. Do NOT author \"Other\" or \
         \"Type something.\" labels yourself — reserved labels are rejected at runtime.\n\
         - Use multiSelect: true when multiple answers are valid.\n\
         - If you recommend a specific option, make that the first option in the list and add \
         \"(Recommended)\" at the end of the label.\n\
         - Group all clarifying questions into one call rather than asking again straight after."
      .into()
  }

  /// Ask, and wait for however long the user takes.
  ///
  /// A dialog that goes away without answering — the run was aborted, the
  /// terminal is gone — reads the same as a decline: the model is told nobody
  /// answered rather than left waiting.
  async fn call(&self, _ctx: &mut ToolContext, args: AskArgs) -> Result<String, ToolError> {
    let questions = ask::prepare(args.questions);
    ask::validate(&questions).map_err(ToolError)?;
    let outcome = self.host.show(Dialog::new(questions.clone())).await.unwrap_or_default();
    Ok(outcome.response(&questions))
  }
}

#[cfg(test)]
mod ask_tests {
  use ratatui::crossterm::event::KeyCode;

  use super::*;
  use crate::modal::answered_by;

  fn questions() -> Vec<Question> {
    vec![
      serde_json::from_value(serde_json::json!({
        "question": "Which cache?",
        "header": "Cache",
        "options": [
          { "label": "Memory", "description": "fast, lost on restart" },
          { "label": "Disk", "description": "survives a restart" },
        ],
      }))
      .unwrap(),
    ]
  }

  async fn ask(tool: &AskTool) -> Result<String, ToolError> {
    tool
      .call(&mut ToolContext::new(), AskArgs { questions: questions() })
      .await
  }

  #[tokio::test]
  async fn the_answer_the_dialog_gave_is_what_the_model_reads() {
    let tool = AskTool {
      host: answered_by(|mut modal| {
        assert_eq!(modal.title(), "The model is asking");
        assert!(!modal.key(KeyCode::Down.into()));
        assert!(modal.key(KeyCode::Enter.into()), "one question, answered");
      }),
    };
    assert_eq!(
      ask(&tool).await.unwrap(),
      "User has answered your questions: \"Which cache?\"=\"Disk\". \
       You can now continue with the user's answers in mind."
    );
  }

  #[tokio::test]
  async fn a_dialog_that_never_answers_reads_as_a_decline() {
    // Dropping it unanswered is what an aborted run leaves behind.
    let tool = AskTool {
      host: answered_by(drop),
    };
    assert_eq!(ask(&tool).await.unwrap(), "User declined to answer questions");
  }

  #[test]
  fn the_model_is_told_what_a_question_may_be() {
    // The bounds are the schema's to advertise; a model that ignores them is
    // caught again by `ask::validate` before anything is drawn.
    let schema = schema::<AskArgs>();
    let question = &schema["$defs"]["Question"]["properties"];
    assert_eq!(schema["properties"]["questions"]["maxItems"], 4);
    assert_eq!(question["options"]["minItems"], 2);
    assert_eq!(question["options"]["maxItems"], 4);
    assert_eq!(question["header"]["maxLength"], 16);
    assert_eq!(schema["$defs"]["Choice"]["properties"]["label"]["maxLength"], 60);
    // The name the extension's models have been writing all along.
    assert!(question.get("multiSelect").is_some(), "{schema}");
  }

  #[tokio::test]
  async fn a_questionnaire_the_user_cannot_be_shown_is_refused_before_it_is() {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let tool = AskTool { host: Host::new(tx) };
    let err = tool
      .call(&mut ToolContext::new(), AskArgs { questions: Vec::new() })
      .await
      .unwrap_err();
    assert_eq!(err.0, "Error: At least one question is required");
    assert!(rx.try_recv().is_err(), "nothing was shown");
  }
}
