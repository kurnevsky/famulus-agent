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

use crate::ask::{self, Outcome, Question};
use crate::{edit, images};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::AsyncReadExt;
use tokio::sync::oneshot;

const MAX_LINES: usize = 2000;
const MAX_BYTES: usize = 50 * 1024;

/// The names the agent's own tools answer to, in the order the system prompt
/// introduces them.
pub const BUILT_IN: [&str; 5] = [
  ReadTool::NAME,
  BashTool::NAME,
  EditTool::NAME,
  WriteTool::NAME,
  AskTool::NAME,
];

/// Which of `available` to offer the model, given what was allowed and what
/// was refused, with the names asked for that nothing answers to.
///
/// `None` is everything: a session that said nothing about tools is offered
/// all of them, which is not the same as one that allowed all of them by name
/// and would lose a tool that arrived later.
///
/// An allow-list is the first word and a deny-list the last, so naming a tool
/// in both refuses it — the narrower intent wins, which is the safe way round
/// for a list whose point is usually to keep something away from the model.
pub fn choose(available: &[String], allow: &[String], deny: &[String]) -> (Option<Vec<String>>, Vec<String>) {
  let unknown: Vec<String> = allow
    .iter()
    .chain(deny)
    .filter(|name| !available.contains(name))
    .cloned()
    .collect();
  if allow.is_empty() && deny.is_empty() {
    return (None, unknown);
  }
  let chosen = available
    .iter()
    .filter(|name| allow.is_empty() || allow.contains(name))
    .filter(|name| !deny.contains(name))
    .cloned()
    .collect();
  (Some(chosen), unknown)
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
  /// Our error messages are written for the model (e.g. "oldText not found"),
  /// so forward them instead of rig's redacted default feedback.
  fn into_execution_error(self) -> ToolExecutionError {
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
    type Error = ToolError;

    fn parameters(&self) -> serde_json::Value {
      schema::<$args>()
    }

    fn map_error(&self, error: ToolError) -> ToolExecutionError {
      error.into_execution_error()
    }
  };
}

/// JSON schema for a tool's arguments, stripped of metadata that some
/// OpenAI-compatible servers reject (`$schema`, `title`, integer `format`).
fn schema<T: JsonSchema>() -> serde_json::Value {
  let mut value = serde_json::to_value(schemars::schema_for!(T)).expect("schema serializes");
  fn strip(v: &mut serde_json::Value) {
    match v {
      serde_json::Value::Object(map) => {
        map.remove("$schema");
        map.remove("title");
        map.remove("format");
        map.values_mut().for_each(strip);
      }
      serde_json::Value::Array(items) => items.iter_mut().for_each(strip),
      _ => {}
    }
  }
  strip(&mut value);
  // The call id a hook injects is ours, not the model's: it is never asked
  // for, so it is never offered.
  if let Some(properties) = value.get_mut("properties").and_then(|p| p.as_object_mut()) {
    properties.remove(CALL_ARG);
  }
  value
}

/// Path normalization: Unicode spaces folded, a leading `@` (chat file
/// reference) stripped, `~` expanded, `file://` URLs accepted, then resolved
/// against the working directory.
fn resolve(cwd: &Path, path: &str) -> PathBuf {
  let mut normalized: String = path
    .chars()
    .map(|c| match c {
      '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
      c => c,
    })
    .collect();
  if let Some(rest) = normalized.strip_prefix('@') {
    normalized = rest.to_string();
  }
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
      "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). \
             Images are sent as attachments. For text files, output is truncated to {MAX_LINES} lines \
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

    // Decode leniently and count lines with a plain split, so a trailing
    // newline yields one final empty line.
    let text = String::from_utf8_lossy(&bytes);
    let all_lines: Vec<&str> = text.split('\n').collect();
    let total_lines = all_lines.len();
    let start = args.offset.map_or(0, |o| o.saturating_sub(1));
    let start_display = start + 1;
    if start >= total_lines {
      return Err(ToolError(format!(
        "Offset {} is beyond end of file ({total_lines} lines total)",
        args.offset.unwrap_or(start_display)
      )));
    }
    let (selected, user_limited) = match args.limit {
      Some(limit) => {
        let end = (start + limit).min(total_lines);
        (all_lines[start..end].join("\n"), Some(end - start))
      }
      None => (all_lines[start..].join("\n"), None),
    };

    let head = truncate_head(&selected);
    let output = if head.first_line_exceeds_limit {
      format!(
        "[Line {start_display} is {}, exceeds {} limit. Use bash: sed -n '{start_display}p' {} | head -c {MAX_BYTES}]",
        format_size(all_lines[start].len()),
        format_size(MAX_BYTES),
        args.path
      )
    } else if head.truncated {
      let end_display = start_display + head.output_lines - 1;
      let next = end_display + 1;
      if head.truncated_by == Some(TruncatedBy::Lines) {
        format!(
          "{}\n\n[Showing lines {start_display}-{end_display} of {total_lines}. Use offset={next} to continue.]",
          head.content
        )
      } else {
        format!(
          "{}\n\n[Showing lines {start_display}-{end_display} of {total_lines} ({} limit). Use offset={next} to continue.]",
          head.content,
          format_size(MAX_BYTES)
        )
      }
    } else if let Some(shown) = user_limited
      && start + shown < total_lines
    {
      let remaining = total_lines - (start + shown);
      let next = start + shown + 1;
      format!(
        "{}\n\n[{remaining} more lines in file. Use offset={next} to continue.]",
        head.content
      )
    } else {
      head.content
    };
    Ok(vec![ToolResultContent::text(output)])
  }
}

struct HeadTruncation {
  content: String,
  truncated: bool,
  truncated_by: Option<TruncatedBy>,
  output_lines: usize,
  first_line_exceeds_limit: bool,
}

/// Keep the first `MAX_LINES` lines within `MAX_BYTES`.
fn truncate_head(content: &str) -> HeadTruncation {
  let mut lines: Vec<&str> = content.split('\n').collect();
  if content.ends_with('\n') {
    lines.pop();
  }
  if lines.len() <= MAX_LINES && content.len() <= MAX_BYTES {
    return HeadTruncation {
      content: content.to_string(),
      truncated: false,
      truncated_by: None,
      output_lines: lines.len(),
      first_line_exceeds_limit: false,
    };
  }
  if lines.first().is_some_and(|l| l.len() > MAX_BYTES) {
    return HeadTruncation {
      content: String::new(),
      truncated: true,
      truncated_by: Some(TruncatedBy::Bytes),
      output_lines: 0,
      first_line_exceeds_limit: true,
    };
  }
  let mut kept: Vec<&str> = Vec::new();
  let mut bytes = 0;
  let mut by = TruncatedBy::Lines;
  for (i, line) in lines.iter().enumerate().take(MAX_LINES) {
    let line_bytes = line.len() + usize::from(i > 0);
    if bytes + line_bytes > MAX_BYTES {
      by = TruncatedBy::Bytes;
      break;
    }
    kept.push(line);
    bytes += line_bytes;
  }
  if kept.len() >= MAX_LINES && bytes <= MAX_BYTES {
    by = TruncatedBy::Lines;
  }
  HeadTruncation {
    content: kept.join("\n"),
    truncated: true,
    truncated_by: Some(by),
    output_lines: kept.len(),
    first_line_exceeds_limit: false,
  }
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
  let head = truncate_head(&text);
  if !head.truncated {
    return None;
  }
  let full = spill("fa-mcp", &text).map_or_else(|| "(unavailable)".to_string(), |p| p.display().to_string());
  let kept = match head.first_line_exceeds_limit {
    // One line longer than the whole budget — a JSON document written flat,
    // usually. There is no line to stop at, so it is cut where the budget
    // runs out, at a character boundary rather than inside one.
    true => {
      let mut end = MAX_BYTES;
      while !text.is_char_boundary(end) {
        end -= 1;
      }
      format!(
        "{}\n\n[Showing first {} of line 1 (line is {}). Full output: {full}]",
        &text[..end],
        format_size(end),
        format_size(text.split('\n').next().unwrap_or_default().len())
      )
    }
    false => {
      let mut lines = text.split('\n').count();
      if text.ends_with('\n') {
        lines -= 1;
      }
      let limit = match head.truncated_by == Some(TruncatedBy::Bytes) {
        true => format!(" ({} limit)", format_size(MAX_BYTES)),
        false => String::new(),
      };
      format!(
        "{}\n\n[Showing lines 1-{} of {lines}{limit}. Full output: {full}]",
        head.content, head.output_lines
      )
    }
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
  let mut file = std::fs::File::create(&path).ok()?;
  file.write_all(text.as_bytes()).ok()?;
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
const NON_VISION_NOTE: &str = "[Current model does not support images. The image will be omitted from this request.]";

fn read_image(bytes: &[u8], format: image::ImageFormat, vision: bool) -> Vec<ToolResultContent> {
  let mime = images::mime_type(format);
  match images::process(bytes, format) {
    Ok(image) => {
      let mut note = format!("Read image file [{}]", image.media_type.to_mime_type());
      for hint in &image.hints {
        note.push('\n');
        note.push_str(hint);
      }
      if !vision {
        note.push('\n');
        note.push_str(NON_VISION_NOTE);
        return vec![ToolResultContent::text(note)];
      }
      vec![
        ToolResultContent::text(note),
        ToolResultContent::image_base64(image.base64, Some(image.media_type), None),
      ]
    }
    Err(reason) => {
      let mut note = format!("Read image file [{mime}]\n{reason}");
      if !vision {
        note.push('\n');
        note.push_str(NON_VISION_NOTE);
      }
      vec![ToolResultContent::text(note)]
    }
  }
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
  async fn binary_is_decoded_leniently_and_paths_are_normalized() {
    let dir = dir("paths");
    std::fs::write(dir.join("bin.dat"), [0xff, 0xfe, b'o', b'k']).unwrap();
    assert_eq!(read(&dir, "@bin.dat", None, None).await.unwrap(), "\u{FFFD}\u{FFFD}ok");
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
  if !diff.text.trim().is_empty() {
    ctx.insert_result(EditDiff { diff: diff.text });
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
pub struct Replacement {
  /// Exact text for one targeted replacement. It must be unique in the original file and
  /// must not overlap with any other edits[].oldText in the same call.
  #[serde(rename = "oldText")]
  old_text: String,
  /// Replacement text for this targeted edit.
  #[serde(rename = "newText")]
  new_text: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct EditArgs {
  /// Path to the file to edit (relative or absolute)
  path: String,
  /// One or more targeted replacements. Each edit is matched against the original file, not
  /// incrementally. Do not include overlapping or nested edits. If two changes touch the same
  /// block or nearby lines, merge them into one edit instead.
  edits: Vec<Replacement>,
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

    let could_not_edit =
      |e: &std::io::Error| ToolError(format!("Could not edit file: {}. {}.", args.path, io_error_code(e)));
    let meta = tokio::fs::metadata(&path).await.map_err(|e| could_not_edit(&e))?;
    if meta.is_dir() {
      return Err(ToolError(format!(
        "Could not edit file: {}. Error code: EISDIR.",
        args.path
      )));
    }
    if meta.permissions().readonly() {
      return Err(ToolError(format!(
        "Could not edit file: {}. Error code: EACCES.",
        args.path
      )));
    }
    let raw = tokio::fs::read_to_string(&path).await.map_err(|e| could_not_edit(&e))?;

    let (bom, content) = edit::split_bom(&raw);
    let ending = edit::detect_line_ending(content);
    let normalized = edit::normalize_to_lf(content);
    let edits: Vec<edit::Edit> = args
      .edits
      .iter()
      .map(|e| edit::Edit {
        old_text: e.old_text.clone(),
        new_text: e.new_text.clone(),
      })
      .collect();
    let applied = edit::apply_edits(&normalized, &edits, &args.path).map_err(ToolError)?;

    let final_content = format!("{bom}{}", edit::restore_line_endings(&applied.new, ending));
    tokio::fs::write(&path, final_content).await?;

    attach_diff(ctx, &applied.base, &applied.new);
    Ok(format!(
      "Successfully replaced {} block(s) in {}.",
      edits.len(),
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

    // Nothing asked for is everything offered, and says so as `None` rather
    // than by listing what there happens to be today.
    assert_eq!(choose(&there, &none, &none), (None, none.clone()));

    // Only these.
    let (chosen, unknown) = choose(&there, &names(&["read", "weather"]), &none);
    assert_eq!(chosen.as_deref(), Some(&names(&["read", "weather"])[..]));
    assert!(unknown.is_empty());

    // Everything but these — which is how a session goes read-only.
    let (chosen, _) = choose(&there, &none, &names(&["write", "edit", "bash"]));
    assert_eq!(chosen.as_deref(), Some(&names(&["read", "weather"])[..]));

    // Named in both, a tool is refused: the narrower intent wins, which is
    // the safe way round for a list meant to keep something away.
    let (chosen, _) = choose(&there, &names(&["read", "bash"]), &names(&["bash"]));
    assert_eq!(chosen.as_deref(), Some(&names(&["read"])[..]));

    // Refusing everything is a session that can only answer, not a session
    // with no list at all.
    let (chosen, _) = choose(&there, &none, &there);
    assert_eq!(chosen, Some(Vec::new()));

    // A name nothing answers to is handed back, since a typo in a list like
    // this is a tool quietly left in or out.
    let (chosen, unknown) = choose(&there, &names(&["reed"]), &names(&["bahs"]));
    assert_eq!(unknown, names(&["reed", "bahs"]));
    assert_eq!(chosen, Some(Vec::new()), "a misspelled allow-list allows nothing");
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
            .map(|(o, n)| Replacement {
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
  }
}

// ---------------------------------------------------------------- bash

/// Receives throttled snapshots of a running command's output for live display.
/// Live output of a running command, with the tool call it belongs to — so a
/// transcript can put it under the right one when several run at once.
pub type OutputSink = Arc<dyn Fn(String, String) + Send + Sync>;

/// The argument `UiHook` injects to tell a call which call it is.
///
/// Nothing in rig hands a tool its own call id, and the hook that knows it
/// runs before the body rather than around it. Rewriting the arguments is the
/// one channel between them, so the id travels as one — stripped from the
/// schema, so it is never something the model is asked for or sends.
pub const CALL_ARG: &str = "__call";

/// Minimum interval between live output snapshots.
const UPDATE_THROTTLE: Duration = Duration::from_millis(100);
/// After the process exits, how long to keep draining pipes held open by
/// lingering background children before giving up on them.
const DRAIN_GRACE: Duration = Duration::from_millis(500);

#[derive(Clone)]
pub struct BashTool {
  pub cwd: PathBuf,
  pub on_output: Option<OutputSink>,
}

#[derive(Deserialize, JsonSchema)]
pub struct BashArgs {
  /// Shell command to execute
  command: String,
  /// Timeout in seconds (optional, no default timeout)
  timeout: Option<f64>,
  /// Not the model's: see `CALL_ARG`.
  #[serde(default, rename = "__call")]
  call: String,
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

  async fn call(&self, _ctx: &mut ToolContext, args: BashArgs) -> Result<String, ToolError> {
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

    let mut output = OutputAccumulator::new("fa-bash");
    let mut throttle = UpdateThrottle::new(self.on_output.clone(), args.call.clone());
    let far_future = tokio::time::Instant::now() + Duration::from_secs(365 * 24 * 3600);
    let mut deadline = timeout.map(|t| tokio::time::Instant::now() + t);
    let mut drain_deadline: Option<tokio::time::Instant> = None;
    let mut exit = None;
    let mut timed_out = false;
    let mut open = true;
    let mut buf = [0u8; 8192];

    while open || exit.is_none() {
      let wake = match (deadline, drain_deadline) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => far_future,
      };
      tokio::select! {
          r = merged.read(&mut buf), if open => match r {
              Ok(n) if n > 0 => {
                  output.append(&buf[..n]);
                  throttle.maybe_emit(&mut output);
              }
              _ => open = false,
          },
          status = child.wait(), if exit.is_none() => {
              exit = Some(status?);
              deadline = None;
              drain_deadline = Some(tokio::time::Instant::now() + DRAIN_GRACE);
          }
          _ = tokio::time::sleep_until(wake), if deadline.is_some() || drain_deadline.is_some() => {
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
    let status = match exit {
      Some(status) => status,
      None => child.wait().await?,
    };
    if !timed_out {
      guard.disarm();
    }

    let snapshot = output.finish();
    let text = format_output(&snapshot, &output, if timed_out { "" } else { "(no output)" });
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
  pgid: Option<i32>,
}

impl ProcessGroupGuard {
  fn new(pid: Option<u32>) -> Self {
    Self {
      pgid: pid.and_then(|p| i32::try_from(p).ok()),
    }
  }

  fn kill(&self) {
    if let Some(pgid) = self.pgid {
      // SAFETY: plain libc calls with a pid we spawned; failures are ignored.
      unsafe {
        if libc::kill(-pgid, libc::SIGKILL) != 0 {
          libc::kill(pgid, libc::SIGKILL);
        }
      }
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
  sink: Option<OutputSink>,
  /// The call whose output this is.
  call: String,
  last: Option<Instant>,
}

impl UpdateThrottle {
  fn new(sink: Option<OutputSink>, call: String) -> Self {
    Self { sink, call, last: None }
  }

  fn maybe_emit(&mut self, output: &mut OutputAccumulator) {
    let Some(sink) = &self.sink else { return };
    if self.last.is_some_and(|t| t.elapsed() < UPDATE_THROTTLE) {
      return;
    }
    self.last = Some(Instant::now());
    sink(self.call.clone(), output.snapshot(false).content);
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TruncatedBy {
  Lines,
  Bytes,
}

struct Truncation {
  truncated: bool,
  truncated_by: Option<TruncatedBy>,
  total_lines: usize,
  output_lines: usize,
  output_bytes: usize,
  last_line_partial: bool,
}

struct Snapshot {
  content: String,
  truncation: Truncation,
  full_output_path: Option<PathBuf>,
}

/// Streams command output, keeping only a rolling tail in memory and spilling
/// the complete output to a temp file once it exceeds the model-facing limits.
struct OutputAccumulator {
  prefix: &'static str,
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

  fn new(prefix: &'static str) -> Self {
    Self {
      prefix,
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
    let path = temp_path(self.prefix);
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

  fn snapshot(&mut self, persist_if_truncated: bool) -> Snapshot {
    let (content, tail_by, output_lines, output_bytes, last_line_partial) = truncate_tail(&self.visible_tail());
    let truncated = self.exceeds_limits();
    let truncated_by = truncated.then(|| {
      tail_by.unwrap_or(if self.total_bytes > MAX_BYTES {
        TruncatedBy::Bytes
      } else {
        TruncatedBy::Lines
      })
    });
    if persist_if_truncated && truncated {
      self.ensure_temp_file();
    }
    Snapshot {
      content,
      truncation: Truncation {
        truncated,
        truncated_by,
        total_lines: self.total_lines(),
        output_lines,
        output_bytes,
        last_line_partial,
      },
      full_output_path: self.temp.as_ref().map(|(p, _)| p.clone()),
    }
  }

  fn finish(&mut self) -> Snapshot {
    let snapshot = self.snapshot(true);
    if let Some((_, file)) = &mut self.temp {
      let _ = file.flush();
    }
    snapshot
  }
}

/// Keep the last `MAX_LINES` lines within `MAX_BYTES`.
/// Returns (content, truncated_by, output_lines, output_bytes, last_line_partial).
fn truncate_tail(content: &str) -> (String, Option<TruncatedBy>, usize, usize, bool) {
  let mut lines: Vec<&str> = content.split('\n').collect();
  if content.ends_with('\n') {
    lines.pop();
  }
  if lines.len() <= MAX_LINES && content.len() <= MAX_BYTES {
    return (content.to_string(), None, lines.len(), content.len(), false);
  }
  let mut kept: Vec<&str> = Vec::new();
  let mut bytes = 0;
  let mut by = TruncatedBy::Lines;
  let mut partial = false;
  let mut partial_line = String::new();
  for line in lines.iter().rev() {
    if kept.len() >= MAX_LINES {
      break;
    }
    let line_bytes = line.len() + usize::from(!kept.is_empty());
    if bytes + line_bytes > MAX_BYTES {
      by = TruncatedBy::Bytes;
      if kept.is_empty() {
        // A single line larger than the budget: keep its end.
        let mut start = line.len() - MAX_BYTES;
        while !line.is_char_boundary(start) {
          start += 1;
        }
        partial_line = line[start..].to_string();
        bytes = partial_line.len();
        partial = true;
      }
      break;
    }
    kept.push(line);
    bytes += line_bytes;
  }
  if partial {
    return (partial_line, Some(by), 1, bytes, true);
  }
  kept.reverse();
  if kept.len() >= MAX_LINES && bytes <= MAX_BYTES {
    by = TruncatedBy::Lines;
  }
  let out = kept.join("\n");
  let bytes = out.len();
  (out, Some(by), kept.len(), bytes, false)
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

/// Model-facing rendering: the tail, then a note on what was cut and where
/// the full output lives.
fn format_output(snapshot: &Snapshot, output: &OutputAccumulator, empty_text: &str) -> String {
  let t = &snapshot.truncation;
  let mut text = if snapshot.content.is_empty() {
    empty_text.to_string()
  } else {
    snapshot.content.clone()
  };
  if t.truncated {
    let path = snapshot
      .full_output_path
      .as_ref()
      .map_or_else(|| "(unavailable)".to_string(), |p| p.display().to_string());
    let start_line = t.total_lines - t.output_lines + 1;
    let end_line = t.total_lines;
    if t.last_line_partial {
      text.push_str(&format!(
        "\n\n[Showing last {} of line {end_line} (line is {}). Full output: {path}]",
        format_size(t.output_bytes),
        format_size(output.current_line_bytes)
      ));
    } else if t.truncated_by == Some(TruncatedBy::Lines) {
      text.push_str(&format!(
        "\n\n[Showing lines {start_line}-{end_line} of {}. Full output: {path}]",
        t.total_lines
      ));
    } else {
      text.push_str(&format!(
        "\n\n[Showing lines {start_line}-{end_line} of {} ({} limit). Full output: {path}]",
        t.total_lines,
        format_size(MAX_BYTES)
      ));
    }
  }
  text
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
  fn the_injected_call_id_is_never_offered_to_the_model() {
    // It is how a hook tells a command which call it is; the model neither
    // sends it nor should know it exists.
    let schema = schema::<BashArgs>();
    let properties = schema["properties"].as_object().unwrap();
    assert!(properties.contains_key("command"));
    assert!(!properties.contains_key(CALL_ARG), "{schema}");
    assert!(!schema.to_string().contains(CALL_ARG), "{schema}");
  }

  #[test]
  fn a_command_is_told_which_call_it_is() {
    // The hook rewrites the arguments to carry it, so it has to survive the
    // trip back through deserialization.
    let args: BashArgs = serde_json::from_value(serde_json::json!({
      "command": "echo hi",
      CALL_ARG: "call_7",
    }))
    .unwrap();
    assert_eq!(args.command, "echo hi");
    assert_eq!(args.call, "call_7");
    // The tool still runs standalone, without a hook to tell it anything.
    let plain: BashArgs = serde_json::from_value(serde_json::json!({ "command": "echo hi" })).unwrap();
    assert!(plain.call.is_empty());
  }

  use super::*;
  use std::sync::Mutex;

  fn tool() -> BashTool {
    BashTool {
      cwd: std::env::temp_dir(),
      on_output: None,
    }
  }

  async fn run(tool: &BashTool, command: &str, timeout: Option<f64>) -> Result<String, ToolError> {
    tool
      .call(
        &mut ToolContext::new(),
        BashArgs {
          call: String::new(),
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
  async fn streams_partial_output() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = seen.clone();
    let tool = BashTool {
      cwd: std::env::temp_dir(),
      on_output: Some(Arc::new(move |_call, text| sink.lock().unwrap().push(text))),
    };
    let out = run(&tool, "echo first; sleep 0.3; echo second", None).await.unwrap();
    assert_eq!(out, "first\nsecond\n");
    let seen = seen.lock().unwrap();
    assert!(seen.iter().any(|s| s == "first\n"), "{seen:?}");
  }

  #[tokio::test]
  async fn missing_cwd_is_reported() {
    let tool = BashTool {
      cwd: PathBuf::from("/nonexistent/dir"),
      on_output: None,
    };
    let err = run(&tool, "true", None).await.unwrap_err();
    assert!(err.0.starts_with("Working directory does not exist: /nonexistent/dir"));
  }
}

// ---------------------------------------------------------------- ask

/// Puts a questionnaire to the user and hands back the channel its answer
/// comes down. `None` is a session with no terminal to ask in.
pub type QuestionSink = Arc<dyn Fn(Vec<Question>, oneshot::Sender<Outcome>) + Send + Sync>;

#[derive(Clone)]
pub struct AskTool {
  pub ask: Option<QuestionSink>,
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
  /// terminal is gone — drops the channel, which reads the same as a decline:
  /// the model is told nobody answered rather than left waiting.
  async fn call(&self, _ctx: &mut ToolContext, args: AskArgs) -> Result<String, ToolError> {
    let Some(sink) = &self.ask else {
      return Err(ToolError(
        "Error: UI not available (running in non-interactive mode)".into(),
      ));
    };
    let questions = ask::prepare(args.questions);
    ask::validate(&questions).map_err(ToolError)?;
    let (tx, rx) = oneshot::channel();
    sink(questions.clone(), tx);
    let outcome = rx.await.unwrap_or_else(|_| Outcome::declined());
    Ok(outcome.response(&questions))
  }
}

#[cfg(test)]
mod ask_tests {
  use super::*;

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
      ask: Some(Arc::new(|questions: Vec<Question>, reply| {
        assert_eq!(questions.len(), 1);
        let _ = reply.send(Outcome {
          answers: vec![(0, crate::ask::Answer::Chose("Disk".into()))],
          cancelled: false,
        });
      })),
    };
    assert_eq!(
      ask(&tool).await.unwrap(),
      "User has answered your questions: \"Which cache?\"=\"Disk\". \
       You can now continue with the user's answers in mind."
    );
  }

  #[tokio::test]
  async fn a_dialog_that_never_answers_reads_as_a_decline() {
    // Dropping the channel is what an aborted run leaves behind.
    let tool = AskTool {
      ask: Some(Arc::new(|_questions, _reply| {})),
    };
    assert_eq!(ask(&tool).await.unwrap(), "User declined to answer questions");
  }

  #[tokio::test]
  async fn a_session_with_no_terminal_says_so_instead_of_waiting() {
    let tool = AskTool { ask: None };
    assert_eq!(
      ask(&tool).await.unwrap_err().0,
      "Error: UI not available (running in non-interactive mode)"
    );
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
    let tool = AskTool {
      ask: Some(Arc::new(|_questions, _reply| panic!("nothing to show"))),
    };
    let err = tool
      .call(&mut ToolContext::new(), AskArgs { questions: Vec::new() })
      .await
      .unwrap_err();
    assert_eq!(err.0, "Error: At least one question is required");
  }
}
