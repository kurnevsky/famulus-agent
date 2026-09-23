//! Text-replacement engine for the `edit` tool: BOM and line-ending
//! handling, exact-then-fuzzy matching, overlap checks, and the numbered
//! diff rendering for the UI.

use schemars::JsonSchema;
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

// One replacement, as the `edit` tool is given it. No doc comment: it would
// become part of the schema the model reads.
#[derive(Deserialize, JsonSchema)]
#[schemars(rename = "Replacement")]
pub struct Edit {
  /// Exact text for one targeted replacement. It must be unique in the original file and
  /// must not overlap with any other edits[].oldText in the same call.
  #[serde(rename = "oldText")]
  pub old_text: String,
  /// Replacement text for this targeted edit.
  #[serde(rename = "newText")]
  pub new_text: String,
}

#[derive(Debug)]
pub struct Applied {
  /// The file content the edits were matched against (LF-normalized, no BOM).
  pub base: String,
  pub new: String,
}

pub fn split_bom(content: &str) -> (&'static str, &str) {
  match content.strip_prefix('\u{FEFF}') {
    Some(rest) => ("\u{FEFF}", rest),
    None => ("", content),
  }
}

pub fn detect_line_ending(content: &str) -> &'static str {
  match (content.find("\r\n"), content.find('\n')) {
    (Some(crlf), Some(lf)) if crlf < lf => "\r\n",
    _ => "\n",
  }
}

pub fn normalize_to_lf(text: &str) -> String {
  text.replace("\r\n", "\n").replace('\r', "\n")
}

pub fn restore_line_endings(text: &str, ending: &str) -> String {
  if ending == "\r\n" {
    text.replace('\n', "\r\n")
  } else {
    text.to_string()
  }
}

/// Progressive normalization: NFKC, trailing whitespace stripped per line,
/// typographic quotes/dashes/spaces folded to ASCII.
pub fn normalize_for_fuzzy_match(text: &str) -> String {
  let nfkc: String = text.nfkc().collect();
  let trimmed = nfkc.split('\n').map(str::trim_end).collect::<Vec<_>>().join("\n");
  trimmed
    .chars()
    .map(|c| match c {
      '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
      '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
      '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
      '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
      c => c,
    })
    .collect()
}

#[derive(Clone)]
struct Replacement {
  edit_index: usize,
  index: usize,
  len: usize,
  new_text: String,
}

/// Apply `edits` to LF-normalized content. Returns the base and new content
/// for diffing.
pub fn apply_edits(normalized: &str, edits: &[Edit], path: &str) -> Result<Applied, String> {
  let total = edits.len();
  // A call with one edit is told about "the text"; one with several, about
  // the edit by its index.
  let say = |one: String, several: String| if total == 1 { one } else { several };
  let edits: Vec<Edit> = edits
    .iter()
    .map(|e| Edit {
      old_text: normalize_to_lf(&e.old_text),
      new_text: normalize_to_lf(&e.new_text),
    })
    .collect();
  for (i, edit) in edits.iter().enumerate() {
    if edit.old_text.is_empty() {
      return Err(say(
        format!("oldText must not be empty in {path}."),
        format!("edits[{i}].oldText must not be empty in {path}."),
      ));
    }
  }

  // An edit the file does not hold word for word is looked for again with
  // quotes, dashes, odd spaces and trailing whitespace evened out, on both
  // sides — and then every edit is, so they are all matched against one text.
  let used_fuzzy = edits.iter().any(|e| !normalized.contains(&e.old_text));
  let base_for_replacement = if used_fuzzy {
    normalize_for_fuzzy_match(normalized)
  } else {
    normalized.to_string()
  };

  let mut matched: Vec<Replacement> = Vec::with_capacity(edits.len());
  for (i, edit) in edits.iter().enumerate() {
    let needle = match used_fuzzy {
      true => normalize_for_fuzzy_match(&edit.old_text),
      false => edit.old_text.clone(),
    };
    // Folding can leave nothing of an edit that was all whitespace, and an
    // empty needle is found everywhere.
    let mut found = base_for_replacement
      .match_indices(needle.as_str())
      .map(|(at, _)| at)
      .filter(|_| !needle.is_empty());
    let Some(index) = found.next() else {
      return Err(say(
        format!(
          "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
        ),
        format!(
          "Could not find edits[{i}] in {path}. The oldText must match exactly including all whitespace and newlines."
        ),
      ));
    };
    let occurrences = 1 + found.count();
    if occurrences > 1 {
      return Err(say(
        format!(
          "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
        ),
        format!(
          "Found {occurrences} occurrences of edits[{i}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
        ),
      ));
    }
    matched.push(Replacement {
      edit_index: i,
      index,
      len: needle.len(),
      new_text: edit.new_text.clone(),
    });
  }

  matched.sort_by_key(|r| r.index);
  for pair in matched.windows(2) {
    let (prev, cur) = (&pair[0], &pair[1]);
    if prev.index + prev.len > cur.index {
      return Err(format!(
        "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
        prev.edit_index, cur.edit_index
      ));
    }
  }

  let new = if used_fuzzy {
    apply_preserving_unchanged_lines(normalized, &base_for_replacement, &matched)?
  } else {
    apply_replacements(&base_for_replacement, &matched, 0)
  };
  if new == normalized {
    return Err(say(
      format!(
        "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
      ),
      format!("No changes made to {path}. The replacements produced identical content."),
    ));
  }
  Ok(Applied {
    base: normalized.to_string(),
    new,
  })
}

/// `replacements` must be sorted by index and non-overlapping.
fn apply_replacements(content: &str, replacements: &[Replacement], offset: usize) -> String {
  let mut result = content.to_string();
  for r in replacements.iter().rev() {
    let start = r.index - offset;
    result.replace_range(start..start + r.len, &r.new_text);
  }
  result
}

/// When matching used fuzzy normalization, replaced regions come from the
/// normalized text but every untouched line is copied verbatim from the
/// original, so unrelated whitespace is never rewritten.
fn apply_preserving_unchanged_lines(
  original: &str,
  base: &str,
  replacements: &[Replacement],
) -> Result<String, String> {
  let original_lines = original.split_inclusive('\n').collect::<Vec<_>>();
  let mut spans = Vec::new();
  let mut offset = 0;
  for line in base.split_inclusive('\n') {
    spans.push((offset, offset + line.len()));
    offset += line.len();
  }
  if original_lines.len() != spans.len() {
    return Err("Cannot preserve unchanged lines because the base content has a different line count.".into());
  }

  // Group replacements by the (possibly shared) line ranges they touch.
  let mut groups: Vec<(usize, usize, Vec<Replacement>)> = Vec::new();
  for r in replacements {
    let (start_line, end_line) = line_range(base, r);
    if let Some(last) = groups.last_mut()
      && start_line < last.1
    {
      last.1 = last.1.max(end_line);
      last.2.push(r.clone());
      continue;
    }
    groups.push((start_line, end_line, vec![r.clone()]));
  }

  let mut result = String::with_capacity(original.len());
  let mut next_line = 0;
  for (start_line, end_line, group) in groups {
    result.extend(original_lines[next_line..start_line].iter().copied());
    let group_start = spans[start_line].0;
    let group_end = spans[end_line - 1].1;
    result.push_str(&apply_replacements(&base[group_start..group_end], &group, group_start));
    next_line = end_line;
  }
  result.extend(original_lines[next_line..].iter().copied());
  Ok(result)
}

/// The lines of `base` a replacement touches, as a range of line indices.
fn line_range(base: &str, r: &Replacement) -> (usize, usize) {
  // Counted in bytes: a newline is one, so no character is cut in half by
  // looking at the last byte of the replacement.
  let line_of = |at: usize| base.as_bytes()[..at].iter().filter(|&&b| b == b'\n').count();
  (line_of(r.index), line_of(r.index + r.len - 1) + 1)
}

/// Compact diff: `+N line` / `-N line` for changes, ` N line` for up to
/// `context` lines around them, and `...` where context was skipped.
pub fn generate_diff_string(old: &str, new: &str, context: usize) -> String {
  let diff = similar::TextDiff::from_lines(old, new);
  let width = old.split('\n').count().max(new.split('\n').count()).to_string().len();
  let gap = format!(" {} ...", " ".repeat(width));
  let groups = diff.grouped_ops(context);
  let mut out: Vec<String> = Vec::new();
  for (i, group) in groups.iter().enumerate() {
    // Lines were skipped between any two groups, and before the first one
    // unless it starts at the top.
    if i > 0 || group[0].old_range().start > 0 {
      out.push(gap.clone());
    }
    for change in group.iter().flat_map(|op| diff.iter_changes(op)) {
      // An added line is numbered where it lands, everything else where it
      // was.
      let at = change.old_index().or(change.new_index()).unwrap_or_default();
      let line = change.value().trim_end_matches('\n');
      out.push(format!("{}{:>width$} {line}", change.tag(), at + 1));
    }
  }
  let skipped_after = groups
    .last()
    .and_then(|group| group.last())
    .is_some_and(|op| op.old_range().end < diff.old_len());
  if skipped_after {
    out.push(gap);
  }
  out.join("\n")
}

#[cfg(test)]
mod tests {
  use super::*;

  fn edit(old: &str, new: &str) -> Edit {
    Edit {
      old_text: old.into(),
      new_text: new.into(),
    }
  }

  #[test]
  fn line_endings_and_bom() {
    assert_eq!(detect_line_ending("a\r\nb\n"), "\r\n");
    assert_eq!(detect_line_ending("a\nb\r\n"), "\n");
    assert_eq!(normalize_to_lf("a\r\nb\rc"), "a\nb\nc");
    assert_eq!(restore_line_endings("a\nb", "\r\n"), "a\r\nb");
    assert_eq!(split_bom("\u{FEFF}x"), ("\u{FEFF}", "x"));
  }

  #[test]
  fn exact_and_multiple_edits() {
    let applied = apply_edits("one\ntwo\nthree\n", &[edit("one", "1"), edit("three", "3")], "f").unwrap();
    assert_eq!(applied.new, "1\ntwo\n3\n");
  }

  #[test]
  fn fuzzy_match_preserves_untouched_lines() {
    // Trailing whitespace and smart quotes differ from the model's text;
    // the unrelated line keeps its trailing spaces.
    let content = "keep   \nsay \u{201C}hi\u{201D}   \nend\n";
    let applied = apply_edits(content, &[edit("say \"hi\"", "say \"bye\"")], "f").unwrap();
    assert_eq!(applied.new, "keep   \nsay \"bye\"\nend\n");
  }

  #[test]
  fn error_messages_match_pi() {
    let e = apply_edits("a\n", &[edit("zzz", "y")], "f.txt").unwrap_err();
    assert_eq!(
      e,
      "Could not find the exact text in f.txt. The old text must match exactly including all whitespace and newlines."
    );
    let e = apply_edits("a\n", &[edit("zzz", "y"), edit("a", "b")], "f.txt").unwrap_err();
    assert_eq!(
      e,
      "Could not find edits[0] in f.txt. The oldText must match exactly including all whitespace and newlines."
    );
    let e = apply_edits("a a\n", &[edit("a", "b")], "f.txt").unwrap_err();
    assert_eq!(
      e,
      "Found 2 occurrences of the text in f.txt. The text must be unique. Please provide more context to make it unique."
    );
    let e = apply_edits("abc\n", &[edit("ab", "x"), edit("bc", "y")], "f.txt").unwrap_err();
    assert_eq!(
      e,
      "edits[0] and edits[1] overlap in f.txt. Merge them into one edit or target disjoint regions."
    );
    let e = apply_edits("a\n", &[edit("a", "a")], "f.txt").unwrap_err();
    assert!(e.starts_with("No changes made to f.txt. The replacement produced identical content."));
    let e = apply_edits("a\n", &[edit("", "a")], "f.txt").unwrap_err();
    assert_eq!(e, "oldText must not be empty in f.txt.");
  }

  #[test]
  fn diff_string_has_pi_layout() {
    let old = (1..=12).map(|i| format!("l{i}")).collect::<Vec<_>>().join("\n") + "\n";
    let new = old.replace("l6", "L6").replace("l7", "L7");
    assert_eq!(
      generate_diff_string(&old, &new, 2),
      "    ...\n  4 l4\n  5 l5\n- 6 l6\n- 7 l7\n+ 6 L6\n+ 7 L7\n  8 l8\n  9 l9\n    ..."
    );
    // A change on the first line has nothing skipped above it.
    assert_eq!(
      generate_diff_string("a\nb\nc\n", "A\nb\nc\n", 1),
      "-1 a\n+1 A\n 2 b\n   ..."
    );
    // Two changes far apart are two stretches, with the gap marked once
    // between them; a context that reaches the ends leaves no mark there.
    let new = old.replace("l2\n", "L2\n").replace("l11", "L11");
    assert_eq!(
      generate_diff_string(&old, &new, 1),
      "  1 l1\n- 2 l2\n+ 2 L2\n  3 l3\n    ...\n 10 l10\n-11 l11\n+11 L11\n 12 l12"
    );
    assert_eq!(generate_diff_string(&old, &old, 2), "", "no change, nothing to draw");
  }
}
