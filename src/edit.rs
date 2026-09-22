//! Text-replacement engine for the `edit` tool: BOM and line-ending
//! handling, exact-then-fuzzy matching, overlap checks, and the numbered
//! diff rendering for the UI.

use unicode_normalization::UnicodeNormalization;

pub struct Edit {
  pub old_text: String,
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

struct FuzzyMatch {
  index: usize,
  len: usize,
  fuzzy: bool,
}

fn fuzzy_find(content: &str, old_text: &str) -> Option<FuzzyMatch> {
  if let Some(index) = content.find(old_text) {
    return Some(FuzzyMatch {
      index,
      len: old_text.len(),
      fuzzy: false,
    });
  }
  let fuzzy_content = normalize_for_fuzzy_match(content);
  let fuzzy_old = normalize_for_fuzzy_match(old_text);
  fuzzy_content.find(&fuzzy_old).map(|index| FuzzyMatch {
    index,
    len: fuzzy_old.len(),
    fuzzy: true,
  })
}

fn count_occurrences(content: &str, old_text: &str) -> usize {
  let fuzzy_old = normalize_for_fuzzy_match(old_text);
  if fuzzy_old.is_empty() {
    return 0;
  }
  normalize_for_fuzzy_match(content).matches(&fuzzy_old).count()
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
  let edits: Vec<Edit> = edits
    .iter()
    .map(|e| Edit {
      old_text: normalize_to_lf(&e.old_text),
      new_text: normalize_to_lf(&e.new_text),
    })
    .collect();
  for (i, edit) in edits.iter().enumerate() {
    if edit.old_text.is_empty() {
      return Err(if total == 1 {
        format!("oldText must not be empty in {path}.")
      } else {
        format!("edits[{i}].oldText must not be empty in {path}.")
      });
    }
  }

  let used_fuzzy = edits
    .iter()
    .any(|e| fuzzy_find(normalized, &e.old_text).is_some_and(|m| m.fuzzy));
  let base_for_replacement = if used_fuzzy {
    normalize_for_fuzzy_match(normalized)
  } else {
    normalized.to_string()
  };

  let mut matched: Vec<Replacement> = Vec::with_capacity(edits.len());
  for (i, edit) in edits.iter().enumerate() {
    let Some(found) = fuzzy_find(&base_for_replacement, &edit.old_text) else {
      return Err(if total == 1 {
        format!(
          "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
        )
      } else {
        format!(
          "Could not find edits[{i}] in {path}. The oldText must match exactly including all whitespace and newlines."
        )
      });
    };
    let occurrences = count_occurrences(&base_for_replacement, &edit.old_text);
    if occurrences > 1 {
      return Err(if total == 1 {
        format!(
          "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
        )
      } else {
        format!(
          "Found {occurrences} occurrences of edits[{i}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
        )
      });
    }
    matched.push(Replacement {
      edit_index: i,
      index: found.index,
      len: found.len,
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
    return Err(if total == 1 {
      format!(
        "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
      )
    } else {
      format!("No changes made to {path}. The replacements produced identical content.")
    });
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

fn split_lines_with_endings(content: &str) -> Vec<&str> {
  let mut lines = Vec::new();
  let mut rest = content;
  while let Some(i) = rest.find('\n') {
    lines.push(&rest[..=i]);
    rest = &rest[i + 1..];
  }
  if !rest.is_empty() {
    lines.push(rest);
  }
  lines
}

/// When matching used fuzzy normalization, replaced regions come from the
/// normalized text but every untouched line is copied verbatim from the
/// original, so unrelated whitespace is never rewritten.
fn apply_preserving_unchanged_lines(
  original: &str,
  base: &str,
  replacements: &[Replacement],
) -> Result<String, String> {
  let original_lines = split_lines_with_endings(original);
  let mut spans = Vec::new();
  let mut offset = 0;
  for line in split_lines_with_endings(base) {
    spans.push((offset, offset + line.len()));
    offset += line.len();
  }
  if original_lines.len() != spans.len() {
    return Err("Cannot preserve unchanged lines because the base content has a different line count.".into());
  }

  // Group replacements by the (possibly shared) line ranges they touch.
  let mut groups: Vec<(usize, usize, Vec<Replacement>)> = Vec::new();
  for r in replacements {
    let (start_line, end_line) = line_range(&spans, r)?;
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

fn line_range(spans: &[(usize, usize)], r: &Replacement) -> Result<(usize, usize), String> {
  let end = r.index + r.len;
  let start_line = spans
    .iter()
    .position(|&(s, e)| r.index >= s && r.index < e)
    .ok_or("Replacement range is outside the base content.")?;
  let mut end_line = start_line;
  while end_line < spans.len() && spans[end_line].1 < end {
    end_line += 1;
  }
  if end_line >= spans.len() {
    return Err("Replacement range is outside the base content.".into());
  }
  Ok((start_line, end_line + 1))
}

/// Compact diff: `+N line` / `-N line` for changes, ` N line` for up to
/// `context` lines around them, and `...` where context was skipped.
pub fn generate_diff_string(old: &str, new: &str, context: usize) -> String {
  use similar::ChangeTag;
  let diff = similar::TextDiff::from_lines(old, new);
  // Collapse the change stream into runs of equal/removed/added lines.
  let mut parts: Vec<(ChangeTag, Vec<&str>)> = Vec::new();
  for change in diff.iter_all_changes() {
    let line = change.value().trim_end_matches('\n');
    match parts.last_mut() {
      Some((tag, lines)) if *tag == change.tag() => lines.push(line),
      _ => parts.push((change.tag(), vec![line])),
    }
  }

  let width = old.split('\n').count().max(new.split('\n').count()).to_string().len();
  let num = |n: usize| format!("{n:>width$}");
  let mut out: Vec<String> = Vec::new();
  let (mut old_num, mut new_num) = (1usize, 1usize);

  for (i, (tag, lines)) in parts.iter().enumerate() {
    match tag {
      ChangeTag::Insert => {
        for line in lines {
          out.push(format!("+{} {line}", num(new_num)));
          new_num += 1;
        }
      }
      ChangeTag::Delete => {
        for line in lines {
          out.push(format!("-{} {line}", num(old_num)));
          old_num += 1;
        }
      }
      ChangeTag::Equal => {
        // Context is kept on the side of a run that touches a change: the end
        // of the change before it, the start of the change after it.
        let after_change = i > 0;
        let before_change = i + 1 < parts.len();
        let head = if after_change { context } else { 0 };
        let tail = if before_change { context } else { 0 };
        let hidden = head..lines.len().saturating_sub(tail);
        for (k, line) in lines.iter().enumerate() {
          if !hidden.contains(&k) {
            out.push(format!(" {} {line}", num(old_num + k)));
          } else if k == hidden.start && (after_change || before_change) {
            out.push(format!(" {} ...", " ".repeat(width)));
          }
        }
        old_num += lines.len();
        new_num += lines.len();
      }
    }
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
  }
}
