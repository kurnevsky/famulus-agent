//! What MCP servers have written out for the user to send: prompts, each a
//! message or a few that a server makes from the arguments it is given.
//!
//! One is sent as `/server:name`, offered by the `/` popup beside the
//! commands. What follows the name is its arguments, a word each in the order
//! the server gives them — quoted, for one with spaces — and the last one
//! taking the rest of the line, so a prompt that wants one thing said takes
//! it as it is typed. A required one left out is asked for in a form, with
//! the ones given filled in.
//!
//! What the server makes of them goes to the model in place of what was
//! typed, as the messages the server wrote: text as text, an image as an
//! image, a resource it carries as its text under a line naming it, and one
//! it only points at as the `&server:uri` the model reads it by.

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rig_core::completion::Message;
use rig_core::message::{AssistantContent, UserContent};
use rmcp::model::{ContentBlock, GetPromptRequestParams, Prompt, PromptMessage, ResourceContents, Role};
use rmcp::service::ServerSink;
use serde_json::{Map, Value};
use std::time::Duration;

use crate::elicit::Form;
use crate::modal::Host;
use crate::resources::bounded;

/// Ask a server for its prompts, every page of them, given up on after
/// `timeout`.
pub async fn inventory(peer: &ServerSink, timeout: Option<Duration>) -> Result<Vec<Prompt>, rmcp::ServiceError> {
  bounded(timeout, peer.list_all_prompts()).await
}

/// What the popup says beside a prompt: the arguments it takes, `<…>` for a
/// required one and `[…]` for the rest, then what the server says it does.
pub fn described(prompt: &Prompt) -> String {
  let arguments =
    prompt
      .arguments
      .as_deref()
      .unwrap_or_default()
      .iter()
      .map(|argument| match argument.required == Some(true) {
        true => format!("<{}>", argument.name),
        false => format!("[{}]", argument.name),
      });
  let about = prompt.description.as_deref().or(prompt.title.as_deref());
  // One line, however the server wrote it.
  let about = about.map(|d| d.split_whitespace().collect::<Vec<_>>().join(" "));
  arguments
    .chain(about.filter(|d| !d.is_empty()))
    .collect::<Vec<_>>()
    .join(" ")
}

/// A prompt one server offers, and what asking it takes.
pub struct Asking {
  pub server: String,
  pub prompt: Prompt,
  pub peer: ServerSink,
  pub timeout: Option<Duration>,
}

impl Asking {
  /// Have the server write the prompt out from `rest`, what was typed after
  /// its name — asking the user through `host` for what `rest` leaves out.
  /// `None` is the user putting the form away: nothing to send.
  pub async fn expand(self, rest: &str, host: &Host, vision: bool) -> Result<Option<Vec<Message>>, String> {
    let command = format!("/{}:{}", self.server, self.prompt.name);
    let declared = self.prompt.arguments.as_deref().unwrap_or_default();
    let mut given = arguments(declared, rest).map_err(|err| format!("{command}: {err}"))?;
    let missing = declared
      .iter()
      .any(|argument| argument.required == Some(true) && !given.contains_key(&argument.name));
    if missing {
      let message = self
        .prompt
        .description
        .clone()
        .unwrap_or_else(|| format!("What {command} takes."));
      let form = Form::arguments(&command, message, declared, &given);
      let answered = host.show(form).await;
      match answered {
        Some(result) if result.action == rmcp::model::ElicitationAction::Accept => {
          given = match result.content {
            Some(Value::Object(content)) => content,
            _ => Map::new(),
          };
        }
        _ => return Ok(None),
      }
    }
    let mut asked = GetPromptRequestParams::new(&self.prompt.name);
    if !given.is_empty() {
      asked = asked.with_arguments(given);
    }
    let written = bounded(self.timeout, self.peer.get_prompt(asked))
      .await
      .map_err(|err| format!("{command}: {err}"))?;
    let messages = messages(written.messages, &self.server, vision);
    match messages.is_empty() {
      true => Err(format!("{command}: the server wrote nothing out")),
      false => Ok(Some(messages)),
    }
  }
}

// ---------------------------------------------------------------- arguments

/// What was typed after a prompt's name, as the arguments it `declared`: a
/// word each, in order, and the last one given the rest of the line. A word
/// in quotes may have spaces in it, and so may the rest, which is taken as
/// it is — quotes kept, unless they are around the whole of it.
fn arguments(declared: &[rmcp::model::PromptArgument], rest: &str) -> Result<Map<String, Value>, String> {
  let mut given = Map::new();
  let mut rest = rest.trim();
  if declared.is_empty() {
    return match rest.is_empty() {
      true => Ok(given),
      false => Err("takes no arguments".to_string()),
    };
  }
  for (at, argument) in declared.iter().enumerate() {
    if rest.is_empty() {
      break;
    }
    let (value, after) = match at + 1 == declared.len() {
      true => (unquoted(rest), ""),
      false => word(rest),
    };
    given.insert(argument.name.clone(), Value::String(value.to_string()));
    rest = after.trim_start();
  }
  Ok(given)
}

/// The first word of `text`, quotes off, and what comes after it.
fn word(text: &str) -> (&str, &str) {
  if let Some(inner) = text.strip_prefix('"') {
    return match inner.find('"') {
      Some(end) => (&inner[..end], &inner[end + 1..]),
      // An unclosed quote runs to the end.
      None => (inner, ""),
    };
  }
  let end = text.find(char::is_whitespace).unwrap_or(text.len());
  (&text[..end], &text[end..])
}

/// `text` without the quotes around it, when they are around the whole of
/// it and nothing inside is quoted.
fn unquoted(text: &str) -> &str {
  match text.strip_prefix('"').and_then(|inner| inner.strip_suffix('"')) {
    Some(inner) if !inner.contains('"') => inner,
    _ => text,
  }
}

// ---------------------------------------------------------------- messages

/// What the server wrote, as the conversation takes it: the messages in
/// order, those of one side in a row made one.
fn messages(written: Vec<PromptMessage>, server: &str, vision: bool) -> Vec<Message> {
  let mut out: Vec<Message> = Vec::new();
  for PromptMessage { role, content, .. } in written {
    match (role, out.last_mut()) {
      (Role::User, Some(Message::User { content: parts })) => parts.extend(user_parts(content, server, vision)),
      (Role::User, _) => out.push(Message::User {
        content: user_parts(content, server, vision),
      }),
      (Role::Assistant, Some(Message::Assistant { content: parts, .. })) => {
        parts.push(AssistantContent::text(said(content, server)))
      }
      (Role::Assistant, _) => out.push(Message::Assistant {
        id: None,
        content: vec![AssistantContent::text(said(content, server))],
      }),
    }
  }
  out
}

/// `&server:uri`, the way a resource is named everywhere else.
fn reference(server: &str, uri: &str) -> String {
  crate::attach::written('&', &format!("{server}:{uri}"))
}

/// One block of what the user is to say: text as text, an image as an image
/// behind a note naming it, and the rest in words.
fn user_parts(block: ContentBlock, server: &str, vision: bool) -> Vec<UserContent> {
  let picture = match &block {
    ContentBlock::Image(image) => Some((
      format!("[Image from {server} — {}]", image.mime_type),
      image.data.clone(),
    )),
    ContentBlock::Resource(embedded) => match &embedded.resource {
      ResourceContents::BlobResourceContents {
        uri, mime_type, blob, ..
      } => {
        let mime = mime_type.as_deref().unwrap_or("unknown type");
        Some((format!("[Resource {} — {mime}]", reference(server, uri)), blob.clone()))
      }
      _ => None,
    },
    _ => None,
  };
  let Some((header, data)) = picture else {
    return vec![UserContent::text(said(block, server))];
  };
  let bytes = STANDARD.decode(data.trim()).ok();
  let (Some(bytes), Some(format)) = (bytes.as_deref(), bytes.as_deref().and_then(crate::images::detect)) else {
    return vec![UserContent::text(format!("{header}\nNot an image fa can send."))];
  };
  if !vision {
    return vec![UserContent::text(format!(
      "{header}\n{}",
      crate::tools::NON_VISION_NOTE
    ))];
  }
  match crate::images::process(bytes, format) {
    Ok(image) => vec![
      UserContent::text(image.note(header)),
      UserContent::image_base64(image.base64(), Some(image.media_type.clone()), None),
    ],
    Err(reason) => vec![UserContent::text(format!("{header}\n{reason}"))],
  }
}

/// One block as words: text as it is, a resource carried whole as its text
/// under a line naming it, one only pointed at as that line alone, and
/// anything with no words in it said to be what it is. What an assistant
/// message is made of, since what a model once said is only ever text.
fn said(block: ContentBlock, server: &str) -> String {
  match block {
    ContentBlock::Text(text) => text.text,
    ContentBlock::Image(image) => format!("[Image from {server} — {}]", image.mime_type),
    ContentBlock::Audio(audio) => format!("[Audio from {server} — {}, not sent]", audio.mime_type),
    ContentBlock::Resource(embedded) => match embedded.resource {
      ResourceContents::TextResourceContents {
        uri, mime_type, text, ..
      } => {
        let mime = mime_type.map(|mime| format!(" — {mime}")).unwrap_or_default();
        format!("[Resource {}{mime}]\n{text}", reference(server, &uri))
      }
      ResourceContents::BlobResourceContents { uri, mime_type, .. } => {
        let mime = mime_type.as_deref().unwrap_or("unknown type");
        format!("[Resource {} — {mime}, not text]", reference(server, &uri))
      }
      _ => format!("[A resource from {server} of a kind fa cannot read]"),
    },
    ContentBlock::ResourceLink(link) => {
      let mut line = format!("[Resource {} — {}", reference(server, &link.uri), link.name);
      if let Some(mime) = &link.mime_type {
        line.push_str(&format!(" ({mime})"));
      }
      line.push(']');
      if let Some(description) = link.description.as_deref().filter(|d| !d.trim().is_empty()) {
        line.push_str(": ");
        line.push_str(&description.split_whitespace().collect::<Vec<_>>().join(" "));
      }
      line
    }
    // A kind of block newer than this client.
    _ => format!("[Something from {server} fa cannot read]"),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rmcp::model::PromptArgument;

  fn declared(names: &[(&str, bool)]) -> Vec<PromptArgument> {
    names
      .iter()
      .map(|(name, required)| PromptArgument::new(*name).with_required(*required))
      .collect()
  }

  /// The arguments `rest` gives, by name.
  fn given(declared: &[PromptArgument], rest: &str) -> Vec<(String, String)> {
    let mut given: Vec<(String, String)> = arguments(declared, rest)
      .expect("arguments")
      .into_iter()
      .map(|(key, value)| (key, value.as_str().expect("a string").to_string()))
      .collect();
    given.sort();
    given
  }

  fn pairs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
  }

  #[test]
  fn arguments_are_a_word_each_and_the_last_takes_the_rest() {
    let two = declared(&[("pr", true), ("focus", false)]);
    assert_eq!(given(&two, ""), pairs(&[]));
    assert_eq!(given(&two, " 12 "), pairs(&[("pr", "12")]));
    assert_eq!(
      given(&two, "12 the error  handling\nand tests"),
      pairs(&[("focus", "the error  handling\nand tests"), ("pr", "12")])
    );
    assert_eq!(
      given(&two, "\"a b\" \"all of it\""),
      pairs(&[("focus", "all of it"), ("pr", "a b")])
    );
    // Quotes inside the rest are the rest's.
    assert_eq!(
      given(&two, "1 say \"hi\" twice"),
      pairs(&[("focus", "say \"hi\" twice"), ("pr", "1")])
    );
    let one = declared(&[("text", true)]);
    assert_eq!(given(&one, "one two"), pairs(&[("text", "one two")]));
    let none = declared(&[]);
    assert_eq!(given(&none, "  "), pairs(&[]));
    assert!(arguments(&none, "stray").is_err());
  }

  #[test]
  fn a_prompt_is_described_by_what_it_takes_and_does() {
    let prompt = Prompt::new(
      "review",
      Some("Review a\n  change."),
      Some(declared(&[("pr", true), ("focus", false)])),
    );
    assert_eq!(described(&prompt), "<pr> [focus] Review a change.");
    assert_eq!(described(&Prompt::new("x", None::<String>, None)), "");
  }

  #[test]
  fn what_a_server_writes_is_the_conversation_it_says() {
    let written = vec![
      PromptMessage::new_text(Role::User, "Look at this."),
      PromptMessage::new(Role::User, ContentBlock::embedded_text("file:///a.rs", "fn main() {}")),
      PromptMessage::new_text(Role::Assistant, "Looking."),
      PromptMessage::new(
        Role::User,
        ContentBlock::ResourceLink(rmcp::model::Resource::new("note://today", "today")),
      ),
    ];
    let messages = messages(written, "srv", true);
    assert_eq!(messages.len(), 3, "{messages:?}");
    let Message::User { content } = &messages[0] else {
      panic!("the user first: {messages:?}")
    };
    let texts: Vec<&str> = content
      .iter()
      .filter_map(|c| match c {
        UserContent::Text(t) => Some(t.text.as_str()),
        _ => None,
      })
      .collect();
    assert_eq!(
      texts,
      [
        "Look at this.",
        "[Resource &srv:file:///a.rs — text/plain]\nfn main() {}"
      ]
    );
    assert!(matches!(&messages[1], Message::Assistant { .. }));
    let Message::User { content } = &messages[2] else {
      panic!("the user again: {messages:?}")
    };
    assert!(
      matches!(content.first(), Some(UserContent::Text(t)) if t.text == "[Resource &srv:note://today — today]"),
      "{content:?}"
    );
  }

  #[test]
  fn an_image_is_sent_to_a_model_that_takes_them_and_named_to_one_that_does_not() {
    let mut png = Vec::new();
    image::RgbImage::new(2, 2)
      .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
      .expect("a png");
    let block = || ContentBlock::image(STANDARD.encode(&png), "image/png");
    let shown = user_parts(block(), "srv", true);
    assert!(shown.iter().any(|c| matches!(c, UserContent::Image(_))), "{shown:?}");
    let told = user_parts(block(), "srv", false);
    assert!(!told.iter().any(|c| matches!(c, UserContent::Image(_))), "{told:?}");
    let broken = user_parts(ContentBlock::image("AAAA", "image/png"), "srv", true);
    assert!(
      matches!(broken.as_slice(), [UserContent::Text(t)] if t.text.contains("Not an image")),
      "{broken:?}"
    );
  }
}
