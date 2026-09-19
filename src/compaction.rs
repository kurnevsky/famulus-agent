//! Context compaction: when the context grows past
//! `context_window - reserve_tokens`, the older part of the conversation is
//! summarized by the model and replaced with a structured checkpoint, while the
//! most recent ~`keep_recent_tokens` stay verbatim.

use rig_agent::agent::Agent;
use rig_agent::completion::{Prompt, PromptError};
use rig_core::completion::Message;
use rig_core::message::{AssistantContent, ToolResultContent, UserContent};

#[derive(Clone, Copy, Debug)]
pub struct Settings {
  pub enabled: bool,
  pub context_window: u64,
  pub reserve_tokens: u64,
  pub keep_recent_tokens: u64,
}

#[derive(Debug)]
pub struct Compacted {
  pub history: Vec<Message>,
  pub summary: String,
  pub summarized: usize,
  pub kept: usize,
}

pub const SUMMARY_PREFIX: &str =
  "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const SUMMARY_SUFFIX: &str = "\n</summary>";

pub const SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or \"(none)\" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or \"(none)\" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.";

const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed
- UPDATE \"Next Steps\" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.";

const TOOL_RESULT_MAX_CHARS: usize = 2000;
const ESTIMATED_IMAGE_CHARS: usize = 4800;

pub fn should_compact(context_tokens: u64, settings: &Settings) -> bool {
  settings.enabled && context_tokens > settings.context_window.saturating_sub(settings.reserve_tokens)
}

/// Conservative chars/4 estimate, used when the provider reports no usage and
/// for choosing the cut point.
pub fn estimate_tokens(messages: &[Message]) -> u64 {
  messages.iter().map(|m| message_chars(m) as u64 / 4).sum()
}

fn message_chars(message: &Message) -> usize {
  match message {
    Message::System { content } => content.len(),
    Message::User { content } => content
      .iter()
      .map(|c| match c {
        UserContent::Text(t) => t.text.len(),
        UserContent::ToolResult(r) => r.content.iter().map(tool_result_chars).sum(),
        _ => ESTIMATED_IMAGE_CHARS,
      })
      .sum(),
    Message::Assistant { content, .. } => content
      .iter()
      .map(|c| match c {
        AssistantContent::Text(t) => t.text.len(),
        AssistantContent::ToolCall(call) => call.function.name.len() + call.function.arguments.to_string().len(),
        AssistantContent::Reasoning(r) => r.display_text().len(),
        AssistantContent::Image(_) => ESTIMATED_IMAGE_CHARS,
      })
      .sum(),
  }
}

fn tool_result_chars(content: &ToolResultContent) -> usize {
  match content {
    ToolResultContent::Text(t) => t.text.len(),
    ToolResultContent::Json { value } => value.to_string().len(),
    ToolResultContent::Image(_) => ESTIMATED_IMAGE_CHARS,
  }
}

/// A message may start a kept tail only if it begins a user turn, so tool
/// call/result pairs are never split.
fn starts_turn(message: &Message) -> bool {
  matches!(message, Message::User { content } if content.iter().any(|c| matches!(c, UserContent::Text(_))))
}

fn previous_summary(message: &Message) -> Option<&str> {
  let Message::User { content } = message else {
    return None;
  };
  let [UserContent::Text(text)] = content.as_slice() else {
    return None;
  };
  text
    .text
    .strip_prefix(SUMMARY_PREFIX)
    .and_then(|rest| rest.strip_suffix(SUMMARY_SUFFIX))
}

/// Index of the first message to keep verbatim. Walks back from the end until
/// roughly `keep_recent_tokens` are accumulated, then snaps forward to the next
/// turn boundary. Returns `None` when there is nothing worth summarizing.
pub fn cut_point(history: &[Message], keep_recent_tokens: u64) -> Option<usize> {
  let first = usize::from(history.first().is_some_and(previous_summary_present));
  let turn_starts: Vec<usize> = (first..history.len()).filter(|&i| starts_turn(&history[i])).collect();

  let mut accumulated = 0u64;
  let mut exceeded_at = None;
  for i in (first..history.len()).rev() {
    accumulated += message_chars(&history[i]) as u64 / 4;
    if accumulated >= keep_recent_tokens {
      exceeded_at = Some(i);
      break;
    }
  }
  let last_turn = turn_starts.last().copied();
  let cut = match exceeded_at {
    // Everything fits in the budget: keep the most recent turn verbatim
    // and summarize whatever came before it.
    None => last_turn,
    // Keep from the first turn boundary at or after the overflow. If the
    // overflow sits inside the last turn, keep that whole turn; if it sits
    // inside the only turn, the turn itself is too big: summarize it all.
    Some(i) => match turn_starts.iter().copied().find(|&c| c >= i).or(last_turn) {
      Some(c) if c > first => Some(c),
      _ => Some(history.len()),
    },
  };
  cut.filter(|&c| c > first)
}

fn previous_summary_present(message: &Message) -> bool {
  previous_summary(message).is_some()
}

/// Render messages in the summarization transcript format.
pub fn serialize(messages: &[Message]) -> String {
  let mut parts: Vec<String> = Vec::new();
  for message in messages {
    match message {
      Message::System { .. } => {}
      Message::User { content } => {
        let text: Vec<&str> = content
          .iter()
          .filter_map(|c| {
            if let UserContent::Text(t) = c {
              Some(t.text.as_str())
            } else {
              None
            }
          })
          .collect();
        if !text.is_empty() {
          parts.push(format!("[User]: {}", text.join("\n")));
        }
        for c in content {
          if let UserContent::ToolResult(result) = c {
            let text: String = result
              .content
              .iter()
              .filter_map(|c| c.as_text())
              .collect::<Vec<_>>()
              .join("\n");
            if !text.is_empty() {
              parts.push(format!("[Tool result]: {}", truncate_for_summary(&text)));
            }
          }
        }
      }
      Message::Assistant { content, .. } => {
        let mut thinking = Vec::new();
        let mut text = Vec::new();
        let mut calls = Vec::new();
        for c in content {
          match c {
            AssistantContent::Reasoning(r) => thinking.push(r.display_text()),
            AssistantContent::Text(t) => text.push(t.text.as_str()),
            AssistantContent::ToolCall(call) => {
              let args = match &call.function.arguments {
                serde_json::Value::Object(map) => map
                  .iter()
                  .map(|(k, v)| format!("{k}={v}"))
                  .collect::<Vec<_>>()
                  .join(", "),
                other => other.to_string(),
              };
              calls.push(format!("{}({args})", call.function.name));
            }
            AssistantContent::Image(_) => {}
          }
        }
        if !thinking.is_empty() {
          parts.push(format!("[Assistant thinking]: {}", thinking.join("\n")));
        }
        if !text.is_empty() {
          parts.push(format!("[Assistant]: {}", text.join("\n")));
        }
        if !calls.is_empty() {
          parts.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
        }
      }
    }
  }
  parts.join("\n\n")
}

fn truncate_for_summary(text: &str) -> String {
  if text.len() <= TOOL_RESULT_MAX_CHARS {
    return text.to_string();
  }
  let mut end = TOOL_RESULT_MAX_CHARS;
  while !text.is_char_boundary(end) {
    end -= 1;
  }
  format!("{}\n[... truncated {} chars]", &text[..end], text.len() - end)
}

pub fn summary_message(summary: &str) -> Message {
  Message::user(format!("{SUMMARY_PREFIX}{}{SUMMARY_SUFFIX}", summary.trim()))
}

/// Summarize everything before the cut point and rebuild the history as
/// `[summary, kept tail...]`. Returns `None` when there is nothing to compact.
pub async fn compact(
  summarizer: &Agent,
  history: Vec<Message>,
  settings: &Settings,
) -> Result<Option<Compacted>, PromptError> {
  let Some(cut) = cut_point(&history, settings.keep_recent_tokens) else {
    return Ok(None);
  };
  let previous = history.first().and_then(previous_summary).map(str::to_string);
  let start = usize::from(previous.is_some());
  let conversation = serialize(&history[start..cut]);

  // Block order: conversation, previous summary, instructions.
  let mut prompt = format!("<conversation>\n{conversation}\n</conversation>\n\n");
  if let Some(previous) = &previous {
    prompt.push_str(&format!("<previous-summary>\n{previous}\n</previous-summary>\n\n"));
  }
  prompt.push_str(if previous.is_some() {
    UPDATE_SUMMARIZATION_PROMPT
  } else {
    SUMMARIZATION_PROMPT
  });
  let summary = summarizer.prompt(prompt).await?;
  let summary = summary.trim().to_string();
  if summary.is_empty() {
    return Err(PromptError::CompletionError(
      rig_core::completion::CompletionError::ResponseError("summarizer returned an empty summary".into()),
    ));
  }

  let kept = history.len() - cut;
  let mut new_history = Vec::with_capacity(kept + 1);
  new_history.push(summary_message(&summary));
  new_history.extend(history.into_iter().skip(cut));
  Ok(Some(Compacted {
    history: new_history,
    summary,
    summarized: cut - start,
    kept,
  }))
}

#[cfg(test)]
mod tests {
  use super::*;
  use rig_core::message::{ToolCall, ToolCallId, ToolFunction, ToolResult, ToolResultContent};

  fn user(text: &str) -> Message {
    Message::user(text)
  }
  fn assistant(text: &str) -> Message {
    Message::assistant(text)
  }
  fn tool_turn() -> (Message, Message) {
    let call = ToolCall::new(
      ToolCallId::mint(),
      ToolFunction::new("bash".into(), serde_json::json!({"command": "ls"})),
    );
    let result = ToolResult {
      call: call.id.clone(),
      provider: None,
      name: "bash".into(),
      content: vec![ToolResultContent::text("a\nb")],
    };
    (
      Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(call)],
      },
      Message::User {
        content: vec![UserContent::ToolResult(result)],
      },
    )
  }

  #[test]
  fn cut_lands_on_turn_start_and_never_splits_tool_pairs() {
    let (call, result) = tool_turn();
    let history = vec![
      user(&"x".repeat(400)),
      call,
      result,
      assistant(&"y".repeat(400)),
      user("second"),
      assistant("done"),
    ];
    // Budget of 10 tokens is exceeded by the 400-char assistant reply at index 3,
    // so the tail must start at the next turn start: index 4.
    assert_eq!(cut_point(&history, 10), Some(4));
    // Large budget: nothing exceeds it, fall back to keeping the last turn.
    assert_eq!(cut_point(&history, 1_000_000), Some(4));
    // Overflow inside the first turn with a single turn only: summarize everything.
    let single = vec![user(&"x".repeat(400)), assistant(&"y".repeat(400))];
    assert_eq!(cut_point(&single, 10), Some(2));
    // Nothing to summarize.
    assert_eq!(cut_point(&[], 10), None);
    assert_eq!(cut_point(&[summary_message("s"), user("q")], 10), None);
  }

  #[test]
  fn previous_summary_is_recognized_and_excluded() {
    let history = vec![
      summary_message("old summary"),
      user("a"),
      assistant("b"),
      user("c"),
      assistant("d"),
    ];
    assert_eq!(previous_summary(&history[0]), Some("old summary"));
    assert_eq!(cut_point(&history, 1), Some(3));
  }

  #[test]
  fn serializes_like_pi() {
    let (call, result) = tool_turn();
    let text = serialize(&[user("hello"), call, result, assistant("bye")]);
    assert_eq!(
      text,
      "[User]: hello\n\n[Assistant tool calls]: bash(command=\"ls\")\n\n[Tool result]: a\nb\n\n[Assistant]: bye"
    );
  }

  #[test]
  fn should_compact_uses_reserve() {
    let s = Settings {
      enabled: true,
      context_window: 100,
      reserve_tokens: 20,
      keep_recent_tokens: 10,
    };
    assert!(!should_compact(80, &s));
    assert!(should_compact(81, &s));
    assert!(!should_compact(81, &Settings { enabled: false, ..s }));
  }
}
