//! Tool calls as the model writes them.
//!
//! Rig holds a call back until it is whole: a call takes its place in the
//! answer only once it has a name, an id and arguments that parse, since one
//! that never gets them is dropped, and a place given out early would be a
//! hole. So the stream says nothing of a call until it ends — and a call can
//! take a minute to write, when it is a file.
//!
//! What the provider sent is still there to read, though. [`Drafting`] wraps
//! a model's wire and reads each frame for the pieces of a call before rig's
//! own decoder gets it, and hands them on in the same stream as payloads rig
//! does not model: in order with everything else, for the run reading that
//! stream and nobody else. Rig's decoder sees the frame untouched, so the
//! call it finishes is the one the conversation keeps; what is read here is
//! only ever drawn.

use rig_core::DynModel;
use rig_core::driver::Model;
use rig_core::error::EncodeError;
use rig_core::operation::Completion;
use rig_core::streaming::UnknownPayload;
use rig_core::wire::{Decoder, Descriptor, Encoded, Flow, Mode, Out, Request, Wire, WireEvent, WireFrame};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The key a piece of a call is filed under, among the payloads rig does not
/// model.
const KEY: &str = "fa_draft";

/// A piece of a call being written: its name or id when this is where they
/// were said, and whatever of its arguments came with them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Fragment {
  /// Which of the turn's calls this is, as the provider numbers them.
  pub index: u64,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub id: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub name: Option<String>,
  #[serde(default, skip_serializing_if = "String::is_empty")]
  pub args: String,
}

impl Fragment {
  /// The piece of a call a payload carries, when it is one of these.
  pub fn from_payload(payload: &UnknownPayload) -> Option<Self> {
    serde_json::from_value(payload.value().get(KEY)?.clone()).ok()
  }

  fn into_payload(self) -> UnknownPayload {
    let mut wrapped = serde_json::Map::new();
    wrapped.insert(KEY.to_string(), serde_json::to_value(self).unwrap_or_default());
    UnknownPayload::new(Value::Object(wrapped))
  }
}

/// `model`, with the calls it writes told of as they are written.
pub fn drafting<W>(model: Model<W>) -> DynModel<Completion>
where
  W: Wire<Op = Completion, Payload = Encoded, Frame = WireFrame>,
{
  Model::new(Drafting(model.wire), model.transport).erase()
}

/// A wire that is `W` in every way, but whose replies carry the pieces of
/// each call as it is written.
#[derive(Clone, Debug)]
pub struct Drafting<W>(W);

impl<W> Wire for Drafting<W>
where
  W: Wire<Op = Completion, Frame = WireFrame>,
{
  type Op = Completion;
  type Payload = W::Payload;
  type Frame = WireFrame;
  type Decoder<'id> = Reading<W::Decoder<'id>>;

  fn describe(&self) -> Descriptor<'_> {
    self.0.describe()
  }

  fn encode(&self, request: Request<Self>, mode: Mode) -> Result<Self::Payload, EncodeError> {
    self.0.encode(request, mode)
  }

  fn decoder<'id>(&self) -> Self::Decoder<'id> {
    Reading(self.0.decoder())
  }
}

/// Rig's decoder, with each frame read for calls on its way in.
pub struct Reading<D>(D);

impl<'id, D> Decoder<'id, Completion, WireFrame> for Reading<D>
where
  D: Decoder<'id, Completion, WireFrame>,
{
  type Event = (D::Event, Vec<Fragment>);

  fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
    let fragments = fragments(&frame.as_str());
    match self.0.classify(frame) {
      WireEvent::Known(event) => WireEvent::Known((event, fragments)),
      WireEvent::Unknown { event_type, value } => WireEvent::Unknown { event_type, value },
      WireEvent::Corrupt(error) => WireEvent::Corrupt(error),
    }
  }

  fn decode(
    &mut self,
    (event, fragments): Self::Event,
    mut out: Out<'id, Completion>,
  ) -> Result<Flow, rig_core::ProviderError> {
    for fragment in fragments {
      out.unknown(fragment.into_payload());
    }
    self.0.decode(event, out)
  }

  fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, rig_core::ProviderError> {
    self.0.eof(out)
  }
}

/// The pieces of calls one frame carries, in whichever of the shapes a
/// provider streams them in: OpenAI's chat chunks and its Responses events,
/// Anthropic's content blocks, and Cohere's tool-call events. A frame of none
/// of these — or a provider that only ever sends a call whole — has none.
fn fragments(text: &str) -> Vec<Fragment> {
  // Most frames are text, and most of those cannot be any of these: no need
  // to parse them a second time to find that out.
  const MARKS: [&str; 5] = [
    "tool_calls",
    "tool_use",
    "input_json_delta",
    "function_call",
    "tool-call-",
  ];
  if !MARKS.iter().any(|mark| text.contains(mark)) {
    return Vec::new();
  }
  let Ok(frame) = serde_json::from_str::<Value>(text) else {
    return Vec::new();
  };
  let string = |value: &Value| value.as_str().filter(|s| !s.is_empty()).map(str::to_string);
  let index = |value: &Value| value.as_u64().unwrap_or(0);
  match frame.get("type").and_then(Value::as_str) {
    // Responses: the call is named when its item is added, and its
    // arguments follow under the item's place in the output.
    Some("response.output_item.added") => {
      let item = &frame["item"];
      match item["type"].as_str() {
        Some("function_call") => vec![Fragment {
          index: index(&frame["output_index"]),
          id: string(&item["call_id"]),
          name: string(&item["name"]),
          args: item["arguments"].as_str().unwrap_or_default().to_string(),
        }],
        _ => Vec::new(),
      }
    }
    Some("response.function_call_arguments.delta") => vec![Fragment {
      index: index(&frame["output_index"]),
      args: frame["delta"].as_str().unwrap_or_default().to_string(),
      ..Fragment::default()
    }],
    // Anthropic: a `tool_use` block opens with the call's name, and its
    // input arrives as JSON in pieces under the block's index.
    Some("content_block_start") => {
      let block = &frame["content_block"];
      match block["type"].as_str() {
        Some("tool_use") => vec![Fragment {
          index: index(&frame["index"]),
          id: string(&block["id"]),
          name: string(&block["name"]),
          ..Fragment::default()
        }],
        _ => Vec::new(),
      }
    }
    Some("content_block_delta") => {
      let delta = &frame["delta"];
      match delta["type"].as_str() {
        Some("input_json_delta") => vec![Fragment {
          index: index(&frame["index"]),
          args: delta["partial_json"].as_str().unwrap_or_default().to_string(),
          ..Fragment::default()
        }],
        _ => Vec::new(),
      }
    }
    // Cohere: one call at a time, opened with its name and continued with
    // its arguments.
    Some("tool-call-start" | "tool-call-delta") => {
      let call = &frame["delta"]["message"]["tool_calls"];
      vec![Fragment {
        index: index(&frame["index"]),
        id: string(&call["id"]),
        name: string(&call["function"]["name"]),
        args: call["function"]["arguments"].as_str().unwrap_or_default().to_string(),
      }]
    }
    // OpenAI chat: each chunk's first choice carries pieces of any number of
    // calls, each under its own index. A choice of another candidate is not
    // the answer being written.
    _ => frame["choices"]
      .as_array()
      .into_iter()
      .flatten()
      .filter(|choice| choice["index"].as_u64().unwrap_or(0) == 0)
      .flat_map(|choice| choice["delta"]["tool_calls"].as_array().into_iter().flatten())
      .map(|call| Fragment {
        index: index(&call["index"]),
        id: string(&call["id"]),
        name: string(&call["function"]["name"]),
        args: call["function"]["arguments"].as_str().unwrap_or_default().to_string(),
      })
      .collect(),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn fragment(index: u64, id: Option<&str>, name: Option<&str>, args: &str) -> Fragment {
    Fragment {
      index,
      id: id.map(str::to_string),
      name: name.map(str::to_string),
      args: args.to_string(),
    }
  }

  #[test]
  fn a_chat_chunk_carries_pieces_of_each_of_its_calls() {
    let opening = r#"{"choices":[{"index":0,"delta":{"tool_calls":[
      {"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}},
      {"index":1,"id":"call_2","type":"function","function":{"name":"read","arguments":"{\"pa"}}
    ]}}]}"#;
    assert_eq!(
      fragments(opening),
      [
        fragment(0, Some("call_1"), Some("bash"), ""),
        fragment(1, Some("call_2"), Some("read"), "{\"pa"),
      ]
    );
    let more = r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"th\""}}]}}]}"#;
    assert_eq!(fragments(more), [fragment(1, None, None, "th\"")]);
    // Another candidate's calls are not the answer's.
    let other = r#"{"choices":[{"index":1,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"x"}}]}}]}"#;
    assert_eq!(fragments(other), []);
  }

  #[test]
  fn a_responses_call_is_named_when_added_and_written_in_deltas() {
    let added = r#"{"type":"response.output_item.added","output_index":2,
      "item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"write","arguments":""}}"#;
    assert_eq!(fragments(added), [fragment(2, Some("call_1"), Some("write"), "")]);
    let delta =
      r#"{"type":"response.function_call_arguments.delta","output_index":2,"item_id":"fc_1","delta":"{\"path\""}"#;
    assert_eq!(fragments(delta), [fragment(2, None, None, "{\"path\"")]);
    let message = r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}"#;
    assert_eq!(fragments(message), []);
  }

  #[test]
  fn an_anthropic_tool_block_is_named_then_filled() {
    let start = r#"{"type":"content_block_start","index":1,
      "content_block":{"type":"tool_use","id":"toolu_1","name":"edit","input":{}}}"#;
    assert_eq!(fragments(start), [fragment(1, Some("toolu_1"), Some("edit"), "")]);
    let delta =
      r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"pa"}}"#;
    assert_eq!(fragments(delta), [fragment(1, None, None, "{\"pa")]);
    let text = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"tool_use"}}"#;
    assert_eq!(fragments(text), []);
  }

  #[test]
  fn a_cohere_call_is_opened_then_continued() {
    let start = r#"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":
      {"id":"c1","type":"function","function":{"name":"bash","arguments":""}}}}}"#;
    assert_eq!(fragments(start), [fragment(0, Some("c1"), Some("bash"), "")]);
    let delta =
      r#"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{"function":{"arguments":"{}"}}}}}"#;
    assert_eq!(fragments(delta), [fragment(0, None, None, "{}")]);
  }

  #[test]
  fn text_is_not_a_call() {
    assert_eq!(
      fragments(r#"{"choices":[{"index":0,"delta":{"content":"hello"}}]}"#),
      []
    );
    assert_eq!(fragments("not json, but it mentions tool_calls"), []);
  }

  #[test]
  fn a_fragment_survives_the_trip_through_the_stream() {
    let piece = fragment(3, Some("call_9"), Some("bash"), "{\"command\"");
    assert_eq!(Fragment::from_payload(&piece.clone().into_payload()), Some(piece));
    assert_eq!(
      Fragment::from_payload(&UnknownPayload::new(serde_json::json!({"other": 1}))),
      None
    );
  }
}
