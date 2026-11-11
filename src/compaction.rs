//! Context compaction: when the context grows past
//! `context_window - reserve_tokens`, the older part of the conversation is
//! summarized by the model and replaced with a structured checkpoint, while the
//! most recent ~`keep_recent_tokens` stay verbatim.
//!
//! The summary is asked for at the end of the conversation itself — the same
//! preamble, the same tools, the same messages a run sends — so a server that
//! keeps the last prompt it read has most of the request already, and reads
//! only the instructions. Rendered as a transcript of its own instead, the
//! conversation would be read from its first token again, which on a local
//! model is minutes rather than seconds. The transcript is what is left for
//! when that cannot work.

use crate::agent::Runtime;
use rig_core::completion::Message;
use rig_core::message::{AssistantContent, AssistantMessage, ToolResultContent, UserContent};

#[derive(Clone, Copy, Debug)]
pub struct Settings {
  pub enabled: bool,
  pub context_window: u64,
  pub reserve_tokens: u64,
  pub keep_recent_tokens: u64,
  /// Whether the beginning of a turn the cut falls inside gets a summary of
  /// its own, written to serve the rest of that turn — which is still there,
  /// verbatim, underneath it.
  pub turn_summary: bool,
}

/// The window a model is held to when nothing says otherwise: neither
/// `--context-window` nor the provider, which mostly does not report one.
pub const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;

/// What a compaction came to: the summary standing for everything before the
/// cut, and the tail after it, verbatim. The history it leaves is the one
/// followed by the other.
#[derive(Debug)]
pub struct Compacted {
  pub summary: String,
  pub kept: Vec<Message>,
  /// How many messages the summary stands for, not counting the summary a
  /// previous compaction left, which it carries on from.
  pub summarized: usize,
}

pub const SUMMARY_PREFIX: &str =
  "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const SUMMARY_SUFFIX: &str = "\n</summary>";

/// The preamble of a summary asked for over a transcript.
pub const SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

/// What a summary asked for at the end of the conversation is told first: the
/// preamble above it is the agent's own, and the tools are still on offer.
const IN_PLACE_LEAD: &str = "The conversation stops here to be compacted, and you are its context summarization assistant now. Do NOT continue the conversation. Do NOT respond to any questions in it. Do NOT call any tools. ONLY output the structured summary asked for below.\n\n";

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

const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above open with a summary of the conversation before them, in <summary> tags. The messages after it are NEW conversation messages to incorporate into that existing summary.

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

/// What the beginning of a split turn is summarized into, however it was
/// shown.
macro_rules! turn_prefix_format {
  () => {
    "Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."
  };
}

/// The beginning of a split turn, asked about at the end of the conversation:
/// everything before the turn is there too, and is summarized apart.
const TURN_PREFIX_IN_PLACE_PROMPT: &str = concat!(
  "The conversation above ends partway through a turn that was too large to keep: the one the user's latest request began. Only the PREFIX of that turn is shown, from that request on. The SUFFIX (recent work) is retained, and everything before the request is summarized separately.\n\n",
  turn_prefix_format!()
);

/// The beginning of a split turn, rendered as a transcript of its own.
const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = concat!(
  "This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.\n\n",
  turn_prefix_format!()
);

/// What stands in for the checkpoint when the split turn is all there was to
/// summarize.
const NO_PRIOR_HISTORY: &str = "No prior history.";

/// How the two are joined into the one message the conversation keeps.
const TURN_SUMMARY_HEADING: &str = "**Turn Context (split turn):**";

const TOOL_RESULT_MAX_CHARS: usize = 2000;
const ESTIMATED_IMAGE_CHARS: usize = 4800;

pub fn should_compact(context_tokens: u64, settings: &Settings) -> bool {
  settings.enabled && context_tokens > settings.context_window.saturating_sub(settings.reserve_tokens)
}

/// Conservative chars/4 estimate, used when the provider reports no usage and
/// for choosing the cut point.
pub fn estimate_tokens(messages: &[Message]) -> u64 {
  messages.iter().map(message_tokens).sum()
}

/// One message at chars/4, rounded up. Rounded down, anything shorter than
/// four characters is free, and a tail of small messages — a long run of
/// little calls — is walked further back than the budget it is held to.
fn message_tokens(message: &Message) -> u64 {
  (message_chars(message) as u64).div_ceil(4)
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
    Message::Assistant(AssistantMessage { content, .. }) => content
      .iter()
      .map(|c| match c {
        AssistantContent::Text(t) => t.text.len(),
        AssistantContent::ToolCall(call) => {
          call.function.name.len() + call.function.arguments_value().to_string().len()
        }
        AssistantContent::Reasoning(r) => r.text.len(),
        AssistantContent::Image(_) => ESTIMATED_IMAGE_CHARS,
        _ => 0,
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

/// Whether a message begins a user turn, which is where the tail starts when
/// the whole conversation fits the budget.
fn starts_turn(message: &Message) -> bool {
  matches!(message, Message::User { content } if content.iter().any(|c| matches!(c, UserContent::Text(_))))
}

/// Whether a kept tail may start here: somewhere the model can be asked to
/// carry on from, which is a user turn or a turn of its own. Never a tool
/// result, which belongs to the call above it — so a call and its answer are
/// never parted, whichever of these the cut lands on.
///
/// A turn may be: the model's own messages are cut points too, so a run long
/// enough to fill the window on its own — which is the run compaction is for
/// — keeps the budget it was promised rather than the little that happens to
/// come after its last user message. What is cut off is what the summary is
/// for.
fn can_follow(message: &Message) -> bool {
  match message {
    Message::Assistant { .. } => true,
    Message::User { .. } => starts_turn(message),
    Message::System { .. } => false,
  }
}

/// The turn the cut falls inside, when it falls inside one rather than
/// between two: the index of the user message that began it.
///
/// What is kept from that turn is the work the model was in the middle of,
/// and what is cut off is the request that set it going — which the one
/// checkpoint would describe as finished business among everything else.
/// Summarized on its own instead, it can be written for the half still there.
fn split_turn(history: &[Message], cut: usize, first: usize) -> Option<usize> {
  if starts_turn(history.get(cut)?) {
    return None;
  }
  (first..cut).rfind(|&i| starts_turn(&history[i]))
}

/// The summary a compaction checkpoint carries, when `message` is one.
pub fn previous_summary(message: &Message) -> Option<&str> {
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

/// Whether `message` is a compaction checkpoint, as it travels in a history.
pub fn is_summary(message: &Message) -> bool {
  previous_summary(message).is_some()
}

/// Index of the first message to keep verbatim, of those from `first` on —
/// which is past the summary a previous compaction left, when there is one.
/// Walks back from the end until roughly `keep_recent_tokens` are
/// accumulated, then forward to the nearest place the model can carry on
/// from. Returns `None` when there is nothing worth summarizing.
fn cut_point(history: &[Message], first: usize, keep_recent_tokens: u64) -> Option<usize> {
  let cut_at = |i: &usize| can_follow(&history[*i]);

  let mut accumulated = 0u64;
  let mut exceeded_at = None;
  for i in (first..history.len()).rev() {
    accumulated += message_tokens(&history[i]);
    if accumulated >= keep_recent_tokens {
      exceeded_at = Some(i);
      break;
    }
  }
  let cut = match exceeded_at {
    // Everything fits in the budget: keep the most recent turn verbatim
    // and summarize whatever came before it.
    None => (first..history.len()).rfind(|&i| starts_turn(&history[i])),
    // Keep from the nearest cut at or after the overflow, which is as much
    // of the budget as can be kept without parting a call from its answer.
    Some(i) => (i..history.len())
      .find(cut_at)
      .filter(|&c| c > first)
      // Nothing after the overflow to keep from — the tail is one tool
      // result worth more than the whole budget, or the overflow is the
      // oldest message there is. Keep from the last cut instead: the call
      // the model made and what came back. Keeping nothing at all would
      // leave it a summary and no work to carry on with, which is the one
      // thing a compaction must not do.
      .or_else(|| (first..history.len()).rfind(cut_at)),
  };
  cut.filter(|&c| c > first)
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
      Message::Assistant(AssistantMessage { content, .. }) => {
        let mut thinking = Vec::new();
        let mut text = Vec::new();
        let mut calls = Vec::new();
        for c in content {
          match c {
            AssistantContent::Reasoning(r) => thinking.push(r.text.clone()),
            AssistantContent::Text(t) => text.push(t.text.as_str()),
            AssistantContent::ToolCall(call) => {
              let args = call
                .function
                .arguments
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ");
              calls.push(format!("{}({args})", call.function.name));
            }
            _ => {}
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
  let end = text.floor_char_boundary(TOOL_RESULT_MAX_CHARS);
  format!("{}\n[... truncated {} chars]", &text[..end], text.len() - end)
}

pub fn summary_message(summary: &str) -> Message {
  Message::user(format!("{SUMMARY_PREFIX}{}{SUMMARY_SUFFIX}", summary.trim()))
}

/// The messages a summary is asked for with at the end of `prefix`: the
/// prefix itself, untouched, and `instructions` after it.
///
/// After a message of the user's own they are appended to it rather than sent
/// as one more: some chat templates refuse two user messages in a row, and the
/// text before the instructions is the same either way. After anything else —
/// a tool's answer, which travels as a message of its own, or the model's —
/// they are a new message.
pub fn continued(prefix: &[Message], instructions: &str) -> Vec<Message> {
  let instructions = format!("{IN_PLACE_LEAD}{instructions}");
  let mut messages = prefix.to_vec();
  match messages.last_mut() {
    Some(Message::User { content }) if !content.iter().any(|c| matches!(c, UserContent::ToolResult(_))) => {
      match content.last_mut() {
        Some(UserContent::Text(text)) => text.text = format!("{}\n\n{instructions}", text.text),
        _ => content.push(UserContent::text(instructions)),
      }
    }
    _ => messages.push(Message::user(instructions)),
  }
  messages
}

/// A summary written to `instructions` at the end of `prefix`, the
/// conversation as far as the summary is to cover.
///
/// Asked as the next message of the conversation, so all of `prefix` is what a
/// server that kept the last request has already read. Failing that — too big
/// a request for the window, or an answer that is no summary — over a
/// transcript of `prefix[from..]`, written to `transcript` instead.
async fn summarize(
  summarizer: &Runtime,
  prefix: &[Message],
  from: usize,
  instructions: &str,
  transcript: &str,
) -> anyhow::Result<String> {
  if let Some(summary) = summarizer.ask_after(continued(prefix, instructions)).await? {
    return Ok(summary);
  }
  let prompt = format!(
    "<conversation>\n{}\n</conversation>\n\n{transcript}",
    serialize(&prefix[from..])
  );
  answer(summarizer, prompt).await
}

/// What the summarizer said, which may not be nothing: an empty checkpoint
/// stands for the conversation every bit as much as a full one does, and
/// there would be no telling afterwards what it was standing for.
async fn answer(summarizer: &Runtime, prompt: String) -> anyhow::Result<String> {
  let summary = summarizer.ask(prompt).await?.trim().to_string();
  if summary.is_empty() {
    anyhow::bail!("summarizer returned an empty summary");
  }
  Ok(summary)
}

/// Summarize everything before the cut point, keeping the tail after it.
/// Returns `None` when there is nothing to compact.
///
/// One summarizer call, or two when the cut falls inside a turn and
/// `turn_summary` is on: the conversation before that turn, and the beginning
/// of the turn itself — asked for in terms of the rest of that turn, which is
/// still there underneath it — joined into the one message the history keeps.
///
/// The turn is asked about first. Both are asked at the end of the
/// conversation, the turn's beginning at the cut and the checkpoint where the
/// turn begins, so the one is a prefix of the other: asked longest first, the
/// shorter finds all of itself in what the server kept of the longer. The
/// other way about, the server would have dropped the turn by the time it was
/// asked to read it again.
pub async fn compact(
  summarizer: &Runtime,
  mut history: Vec<Message>,
  settings: &Settings,
) -> anyhow::Result<Option<Compacted>> {
  let previous = history.first().and_then(previous_summary);
  let start = usize::from(previous.is_some());
  let Some(cut) = cut_point(&history, start, settings.keep_recent_tokens) else {
    return Ok(None);
  };
  let split = settings
    .turn_summary
    .then(|| split_turn(&history, cut, start))
    .flatten();

  let turn = match split {
    Some(turn) => Some(
      summarize(
        summarizer,
        &history[..cut],
        turn,
        TURN_PREFIX_IN_PLACE_PROMPT,
        TURN_PREFIX_SUMMARIZATION_PROMPT,
      )
      .await?,
    ),
    None => None,
  };

  // The checkpoint stops where a split turn begins. When the turn is all
  // there was to summarize it has nothing new to say, and the last
  // compaction's summary — or the words for having none — stands.
  //
  // Its transcript starts from the first message, the previous summary
  // included: the instructions find it where the conversation opens, either
  // way it is shown.
  let end = split.unwrap_or(cut);
  let checkpoint = match end > start {
    true => {
      let instructions = match previous {
        Some(_) => UPDATE_SUMMARIZATION_PROMPT,
        None => SUMMARIZATION_PROMPT,
      };
      summarize(summarizer, &history[..end], 0, instructions, instructions).await?
    }
    false => previous.unwrap_or(NO_PRIOR_HISTORY).to_string(),
  };
  let summary = match turn {
    Some(prefix) => format!("{checkpoint}\n\n---\n\n{TURN_SUMMARY_HEADING}\n\n{prefix}"),
    None => checkpoint,
  };

  Ok(Some(Compacted {
    summary,
    kept: history.split_off(cut),
    summarized: cut - start,
  }))
}

#[cfg(test)]
mod tests {
  use super::*;
  use rig_core::message::{CallId, ToolCall, ToolFunction, ToolName, ToolResultContent};

  /// Where the tail starts, past the summary in front when there is one.
  fn cut(history: &[Message], keep_recent_tokens: u64) -> Option<usize> {
    let first = usize::from(history.first().is_some_and(is_summary));
    cut_point(history, first, keep_recent_tokens)
  }

  fn user(text: &str) -> Message {
    Message::user(text)
  }
  fn assistant(text: &str) -> Message {
    Message::assistant(text)
  }
  fn tool_turn() -> (Message, Message) {
    let call = ToolCall::from_wire(
      "",
      ToolFunction::new(ToolName::new("bash").unwrap(), serde_json::json!({"command": "ls"})),
    );
    let result = call.result(vec![ToolResultContent::text("a\nb")]);
    (
      Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(call)])),
      Message::User {
        content: vec![UserContent::ToolResult(result)],
      },
    )
  }

  #[test]
  fn cut_keeps_what_the_budget_allows_and_never_splits_tool_pairs() {
    let (call, result) = tool_turn();
    let history = vec![
      user(&"x".repeat(400)),
      call,
      result,
      assistant(&"y".repeat(400)),
      user("second"),
      assistant("done"),
    ];
    // Budget of 10 tokens is exceeded by the 400-char assistant reply at
    // index 3, and the tail starts there: an answer of the model's own is
    // somewhere it can carry on from, so the budget is kept rather than
    // given up as far as the next thing the user said.
    assert_eq!(cut(&history, 10), Some(3));
    // Large budget: nothing exceeds it, fall back to keeping the last turn.
    assert_eq!(cut(&history, 1_000_000), Some(4));
    // Nothing to summarize.
    assert_eq!(cut(&[], 10), None);
    assert_eq!(cut(&[summary_message("s"), user("q")], 10), None);

    // A budget that runs out inside a tool result keeps neither it nor the
    // call it answers: the cut moves on past the pair rather than landing
    // between them, where the result would answer a call nobody made.
    let (call, _) = tool_turn();
    let answered = Message::User {
      content: vec![UserContent::tool_result(
        CallId::from_wire("c"),
        ToolName::new("bash").unwrap(),
        vec![ToolResultContent::text("z".repeat(2000))],
      )],
    };
    let big = vec![user("go"), call, answered, assistant("done")];
    assert_eq!(cut(&big, 100), Some(3));
  }

  /// A turn kept whole is not summarized twice: only the turn the cut falls
  /// inside has a beginning left over to say anything about.
  #[test]
  fn only_a_turn_the_cut_falls_inside_has_a_prefix_to_summarize() {
    let (call, result) = tool_turn();
    let history = vec![
      user("first"),
      call.clone(),
      result.clone(),
      assistant("done"),
      user("second"),
      assistant("answered"),
    ];
    // The cut is where the second turn begins, so that turn is kept whole
    // and there is nothing of it to summarize apart from the rest.
    assert_eq!(split_turn(&history, 4, 0), None);
    // Inside the first turn, wherever: the turn began at 0.
    assert_eq!(split_turn(&history, 1, 0), Some(0));
    assert_eq!(split_turn(&history, 3, 0), Some(0));
    // A turn that began before the last compaction: its own start is inside
    // that summary already, and there is nothing this side of it to ask
    // about. What the cut drops of it goes into the summary it came from.
    let carried = vec![summary_message("old"), call.clone(), result.clone(), assistant("x")];
    assert_eq!(split_turn(&carried, 3, 1), None);
    // The summary is not itself the start of a turn to be summarized again,
    // where a turn of this side of it would have been.
    let asked_again = vec![summary_message("old"), user("carry on"), call, result, assistant("x")];
    assert_eq!(split_turn(&asked_again, 4, 1), Some(1));
  }

  /// One turn too big for the budget has no boundary to keep from. Summarizing
  /// it whole would hand the model a summary and nothing to carry on with —
  /// which is the state a compaction is supposed to rescue it from, not put it
  /// in. So the tail starts at the last thing the model can follow.
  #[test]
  fn a_turn_too_big_to_keep_still_leaves_the_work_in_progress_behind() {
    let (call, result) = tool_turn();
    let asked = user(&"x".repeat(4000));
    // The user's request is summarized; the call it led to and what came
    // back stay, so the run picking this up has its own work in front of it.
    let history = vec![asked.clone(), call.clone(), result.clone()];
    assert_eq!(cut(&history, 10), Some(1));

    // The same with a summary already in front: it is not summarized twice,
    // and the request that followed it goes into the updated one.
    let history = vec![summary_message("old"), asked.clone(), call.clone(), result.clone()];
    assert_eq!(cut(&history, 10), Some(2));

    // But when the summary is all that is in front of the turn, there is
    // nothing left to summarize and nothing to be gained by trying.
    assert_eq!(cut(&[summary_message("old"), call, result], 10), None);

    // A turn of two messages splits the same way: what was asked is
    // summarized, the answer to it stays.
    let single = vec![asked, assistant(&"y".repeat(400))];
    assert_eq!(cut(&single, 10), Some(1));
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
    // A budget the last message fills on its own: the tail is that message.
    assert_eq!(cut(&history, 1), Some(4));
    // And one nothing here comes near keeps the last turn. Either way the
    // cut is past the summary, which is not summarized a second time.
    assert_eq!(cut(&history, 1_000), Some(3));
  }

  /// The instructions go after the conversation without touching any of
  /// it, and after the user's own words they join them rather than make two
  /// user messages in a row.
  #[test]
  fn instructions_follow_the_conversation_as_it_was() {
    let instructed = |messages: &[Message]| match messages.last() {
      Some(Message::User { content }) => match content.last() {
        Some(UserContent::Text(t)) => t.text.clone(),
        other => panic!("ends in text, not {other:?}"),
      },
      other => panic!("ends with the user, not {other:?}"),
    };

    // After the model's words, a message of their own.
    let prefix = vec![user("q"), assistant("a")];
    let asked = continued(&prefix, "Summarize.");
    assert_eq!(asked[..2], prefix[..]);
    assert_eq!(asked.len(), 3);
    assert_eq!(instructed(&asked), format!("{IN_PLACE_LEAD}Summarize."));

    // After a tool's answer too, which travels as a message apart.
    let (call, result) = tool_turn();
    let prefix = vec![user("q"), call, result];
    let asked = continued(&prefix, "Summarize.");
    assert_eq!(asked[..3], prefix[..]);
    assert_eq!(asked.len(), 4);

    // After the user's own, joined onto them.
    let prefix = vec![user("q"), assistant("a"), user("next")];
    let asked = continued(&prefix, "Summarize.");
    assert_eq!(asked[..2], prefix[..2]);
    assert_eq!(asked.len(), 3);
    assert_eq!(instructed(&asked), format!("next\n\n{IN_PLACE_LEAD}Summarize."));
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
      turn_summary: true,
    };
    assert!(!should_compact(80, &s));
    assert!(should_compact(81, &s));
    assert!(!should_compact(81, &Settings { enabled: false, ..s }));
  }
}
