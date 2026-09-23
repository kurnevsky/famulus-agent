//! What an MCP server asks the user, and the answer it gets back.
//!
//! A server in the middle of a tool call may ask for something only the user
//! can give — a choice, a name, a confirmation — by sending a form: a message,
//! and a flat object schema of strings, numbers, booleans and enums. It is put
//! to the user as the same dialog the `ask` tool draws, one tab per field, and
//! what comes back is checked against the schema before it goes: a number that
//! is not one, or a required field left blank, is said above the form rather
//! than sent for the server to refuse.
//!
//! Only forms are taken. A server that wants the user sent to a URL is told
//! no, since a terminal cannot be relied on to open one.

use std::borrow::Cow;

use ratatui::crossterm::event::KeyEvent;
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use rmcp::model::{
  ClientCapabilities, ClientInfo, ConstTitle, ElicitRequestParams, ElicitResult, ElicitationAction,
  ElicitationCapability, ElicitationSchema, EnumSchema, ErrorData, FormElicitationCapability, MultiSelectEnumSchema,
  PrimitiveSchemaDefinition, SingleSelectEnumSchema,
};
use rmcp::service::{NotificationContext, RequestContext, RoleClient};
use serde_json::{Map, Value};

use crate::ask::{Answer, Choice, Dialog, Question, Refusal};
use crate::markdown::wrap_text;
use crate::modal::{Component, Host};

/// The client side of one server's connection: what it says it can do, and
/// what it does when asked.
#[derive(Clone)]
pub struct Client {
  /// The server's name in the file, which is who the form says is asking.
  pub server: String,
  pub host: Host,
  /// Where what it offers is kept, for when it says that has changed.
  pub watch: crate::mcp::Watch,
}

impl rmcp::ClientHandler for Client {
  async fn create_elicitation(
    &self,
    request: ElicitRequestParams,
    _context: RequestContext<RoleClient>,
  ) -> Result<ElicitResult, ErrorData> {
    let ElicitRequestParams::FormElicitationParams {
      message,
      requested_schema,
      ..
    } = request
    else {
      return Ok(ElicitResult::new(ElicitationAction::Decline));
    };
    let form = Form::new(&self.server, message, &requested_schema);
    // Put away unanswered — the run was stopped — is the user walking away,
    // which the protocol has its own word for.
    Ok(
      self
        .host
        .show(form)
        .await
        .unwrap_or_else(|| ElicitResult::new(ElicitationAction::Cancel)),
    )
  }

  async fn on_tool_list_changed(&self, context: NotificationContext<RoleClient>) {
    self.watch.changed(&context.peer).await;
  }

  fn get_info(&self) -> ClientInfo {
    let mut capabilities = ClientCapabilities::default();
    capabilities.elicitation = Some(ElicitationCapability::new().with_form(FormElicitationCapability::new()));
    let mut info = ClientInfo::default();
    info.capabilities = capabilities;
    info
  }
}

// ---------------------------------------------------------------- the form

/// The values a field may take, each with the label it is offered under.
type Options = Vec<(String, String)>;

/// What one field takes.
#[derive(Clone, Debug, PartialEq)]
enum Kind {
  Text {
    min: Option<u32>,
    max: Option<u32>,
  },
  Number {
    min: Option<f64>,
    max: Option<f64>,
  },
  Integer {
    min: Option<i64>,
    max: Option<i64>,
  },
  Boolean,
  /// One of these values.
  One(Options),
  /// Any of these, with bounds on how many.
  Many {
    options: Options,
    min: Option<u64>,
    max: Option<u64>,
  },
}

impl Kind {
  /// What the dialog offers to choose from, in order.
  fn labels(&self) -> Vec<String> {
    match self {
      Kind::Boolean => vec![YES.to_string(), NO.to_string()],
      Kind::One(options) | Kind::Many { options, .. } => options.iter().map(|(label, _)| label.clone()).collect(),
      _ => Vec::new(),
    }
  }
}

/// One property of the schema, as the dialog asks it.
#[derive(Clone, Debug)]
struct Field {
  key: String,
  /// What the user knows it by.
  name: String,
  kind: Kind,
  required: bool,
}

/// What a field of a kind this client does not know takes.
const ANYTHING: Kind = Kind::Text { min: None, max: None };

const YES: &str = "Yes";
const NO: &str = "No";

/// A form from a server, being filled in.
pub struct Form {
  server: String,
  message: String,
  fields: Vec<Field>,
  dialog: Dialog,
  /// Why the last attempt to submit was not sent, until the next key.
  error: Option<String>,
}

impl Form {
  pub fn new(server: &str, message: String, schema: &ElicitationSchema) -> Self {
    let required = schema.required.as_deref().unwrap_or_default();
    let (fields, questions): (Vec<Field>, Vec<Question>) = schema
      .properties
      .iter()
      .map(|(key, property)| field(key, property, required.contains(key)))
      .unzip();
    Self {
      server: server.to_string(),
      message,
      fields,
      dialog: Dialog::new(questions),
      error: None,
    }
  }

  /// What the answers come to as the object the schema describes, or what is
  /// wrong with them, for the user to put right.
  fn content(&self, answers: &[(usize, Answer)]) -> Result<Map<String, Value>, String> {
    let mut content = Map::new();
    for (index, answer) in answers {
      let Some(field) = self.fields.get(*index) else {
        continue;
      };
      if let Some(value) = value(field, answer)? {
        content.insert(field.key.clone(), value);
      }
    }
    let missing: Vec<&str> = self
      .fields
      .iter()
      .filter(|field| field.required && !content.contains_key(&field.key))
      .map(|field| field.name.as_str())
      .collect();
    match missing.is_empty() {
      true => Ok(content),
      false => Err(format!("Required: {}", missing.join(", "))),
    }
  }
}

impl Component for Form {
  type Output = ElicitResult;

  fn title(&self) -> String {
    format!("{} is asking", self.server)
  }

  fn key(&mut self, key: KeyEvent) -> Option<ElicitResult> {
    self.error = None;
    let outcome = self.dialog.key(key)?;
    match outcome.refused {
      Some(Refusal::Cancelled) => return Some(ElicitResult::new(ElicitationAction::Cancel)),
      Some(Refusal::Declined) => return Some(ElicitResult::new(ElicitationAction::Decline)),
      None => {}
    }
    match self.content(&outcome.answers) {
      Ok(content) => Some(ElicitResult::new(ElicitationAction::Accept).with_content(Value::Object(content))),
      // The dialog is still where it was, answers and all, so what is wrong
      // can be put right in place.
      Err(error) => {
        self.error = Some(error);
        None
      }
    }
  }

  fn paste(&mut self, text: &str) {
    self.dialog.paste(text);
  }

  fn lines(&self, width: u16) -> (Vec<Line<'static>>, usize) {
    let mut lines = wrap_text(&self.message, width, Style::default());
    lines.push(Line::raw(""));
    if let Some(error) = &self.error {
      lines.extend(wrap_text(
        &format!("⚠ {error}"),
        width,
        Style::default().fg(Color::Yellow),
      ));
      lines.push(Line::raw(""));
    }
    let (form, focus) = self.dialog.lines(width);
    let focus = focus + lines.len();
    lines.extend(form);
    (lines, focus)
  }
}

type Text = Option<Cow<'static, str>>;

/// One property as a field, and as the question the dialog asks for it.
fn field(key: &str, property: &PrimitiveSchemaDefinition, required: bool) -> (Field, Question) {
  let (title, description, kind): (&Text, &Text, Kind) = match property {
    PrimitiveSchemaDefinition::String(s) => (
      &s.title,
      &s.description,
      Kind::Text {
        min: s.min_length,
        max: s.max_length,
      },
    ),
    PrimitiveSchemaDefinition::Number(n) => (
      &n.title,
      &n.description,
      Kind::Number {
        min: n.minimum,
        max: n.maximum,
      },
    ),
    PrimitiveSchemaDefinition::Integer(n) => (
      &n.title,
      &n.description,
      Kind::Integer {
        min: n.minimum,
        max: n.maximum,
      },
    ),
    PrimitiveSchemaDefinition::Boolean(b) => (&b.title, &b.description, Kind::Boolean),
    PrimitiveSchemaDefinition::Enum(e) => choices(e).unwrap_or((&None, &None, ANYTHING)),
    // A kind of property newer than this client is still something to type:
    // the server says what is wrong with it, if anything is.
    _ => (&None, &None, ANYTHING),
  };
  let name = title.as_deref().unwrap_or(key).to_string();
  let mut question = description.as_deref().unwrap_or(&name).to_string();
  if required {
    question.push_str(" (required)");
  }
  let asked = Question {
    question,
    header: name.clone(),
    options: kind
      .labels()
      .into_iter()
      .map(|label| Choice {
        label,
        description: String::new(),
      })
      .collect(),
    multi_select: matches!(kind, Kind::Many { .. }),
    options_only: matches!(kind, Kind::Boolean | Kind::One(_) | Kind::Many { .. }),
  };
  let field = Field {
    key: key.to_string(),
    name,
    kind,
    required,
  };
  (field, asked)
}

/// An enum's title and description, and what it takes — offered under the
/// titles of its values, where it gives them. `None` is a kind of enum newer
/// than this client.
fn choices(schema: &EnumSchema) -> Option<(&Text, &Text, Kind)> {
  let same = |values: &[String]| values.iter().map(|v| (v.clone(), v.clone())).collect();
  let titled = |values: &[ConstTitle]| values.iter().map(|c| (c.title.clone(), c.const_.clone())).collect();
  Some(match schema {
    EnumSchema::Single(SingleSelectEnumSchema::Untitled(s)) => (&s.title, &s.description, Kind::One(same(&s.enum_))),
    EnumSchema::Single(SingleSelectEnumSchema::Titled(s)) => (&s.title, &s.description, Kind::One(titled(&s.one_of))),
    EnumSchema::Legacy(s) => {
      let labels = s.enum_names.as_ref().unwrap_or(&s.enum_);
      let options = labels.iter().cloned().zip(s.enum_.iter().cloned()).collect();
      (&s.title, &s.description, Kind::One(options))
    }
    EnumSchema::Multi(MultiSelectEnumSchema::Untitled(s)) => (
      &s.title,
      &s.description,
      Kind::Many {
        options: same(&s.items.enum_),
        min: s.min_items,
        max: s.max_items,
      },
    ),
    EnumSchema::Multi(MultiSelectEnumSchema::Titled(s)) => (
      &s.title,
      &s.description,
      Kind::Many {
        options: titled(&s.items.any_of),
        min: s.min_items,
        max: s.max_items,
      },
    ),
    _ => return None,
  })
}

/// What one answer is as the value the field takes. `None` is no answer —
/// an empty text row — which leaves the field out rather than sending it
/// blank.
fn value(field: &Field, answer: &Answer) -> Result<Option<Value>, String> {
  let name = &field.name;
  match (&field.kind, answer) {
    (_, Answer::Typed(text)) if text.trim().is_empty() => Ok(None),
    (Kind::Text { min, max }, Answer::Typed(text)) => {
      within(text.chars().count() as u32, *min, *max)
        .map_err(|range| format!("{name} must be {range} characters long"))?;
      Ok(Some(Value::String(text.clone())))
    }
    (Kind::Number { min, max }, Answer::Typed(text)) => {
      let number: f64 = text.trim().parse().map_err(|_| format!("{name} must be a number"))?;
      within(number, *min, *max).map_err(|range| format!("{name} must be {range}"))?;
      Ok(serde_json::Number::from_f64(number).map(Value::Number))
    }
    (Kind::Integer { min, max }, Answer::Typed(text)) => {
      let number: i64 = text
        .trim()
        .parse()
        .map_err(|_| format!("{name} must be a whole number"))?;
      within(number, *min, *max).map_err(|range| format!("{name} must be {range}"))?;
      Ok(Some(Value::from(number)))
    }
    (Kind::Boolean, Answer::Chose(label)) => Ok(Some(Value::Bool(label == YES))),
    (Kind::One(options), Answer::Chose(label)) => Ok(pick(options, label).map(Value::String)),
    (Kind::Many { options, min, max }, Answer::Ticked(labels)) => {
      within(labels.len() as u64, *min, *max).map_err(|range| format!("{name} takes {range} choices"))?;
      let picked = labels.iter().filter_map(|label| pick(options, label));
      Ok(Some(Value::Array(picked.map(Value::String).collect())))
    }
    // An answer of a shape the question was not asked in: nothing the dialog
    // gives, since the rows it offers are the ones the kind allows.
    _ => Ok(None),
  }
}

/// The value behind the label the user picked.
fn pick(options: &Options, label: &str) -> Option<String> {
  options
    .iter()
    .find(|(shown, _)| shown == label)
    .map(|(_, value)| value.clone())
}

/// Whether `value` keeps to the bounds, and when it does not, the bounds in
/// words for a sentence saying so.
fn within<T: PartialOrd + std::fmt::Display>(value: T, min: Option<T>, max: Option<T>) -> Result<(), String> {
  let low = min.as_ref().is_some_and(|min| value < *min);
  let high = max.as_ref().is_some_and(|max| value > *max);
  if !low && !high {
    return Ok(());
  }
  Err(match (min, max) {
    (Some(min), Some(max)) => format!("between {min} and {max}"),
    (Some(min), None) => format!("at least {min}"),
    (None, Some(max)) => format!("at most {max}"),
    (None, None) => unreachable!("out of no bounds"),
  })
}

#[cfg(test)]
mod tests {
  use ratatui::crossterm::event::{KeyCode, KeyModifiers};

  use super::*;

  fn form(schema: serde_json::Value) -> Form {
    let schema: ElicitationSchema = serde_json::from_value(schema).expect("a schema");
    Form::new("srv", "Tell me.".into(), &schema)
  }

  fn press(form: &mut Form, code: KeyCode) -> Option<ElicitResult> {
    form.key(KeyEvent::new(code, KeyModifiers::NONE))
  }

  fn type_in(form: &mut Form, text: &str) {
    for c in text.chars() {
      assert!(press(form, KeyCode::Char(c)).is_none());
    }
  }

  fn text(form: &Form) -> String {
    let (lines, _) = form.lines(60);
    lines.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n")
  }

  #[test]
  fn each_kind_of_field_is_asked_the_way_it_is_answered() {
    let form = form(serde_json::json!({
      "type": "object",
      "properties": {
        "a_name": { "type": "string", "title": "Name" },
        "b_size": { "type": "string", "oneOf": [
          { "const": "s", "title": "Small" },
          { "const": "l", "title": "Large" },
        ] },
        "c_tags": { "type": "array", "items": { "type": "string", "enum": ["x", "y"] } },
        "d_ok": { "type": "boolean" },
      },
      "required": ["a_name"],
    }));
    let kinds: Vec<&Kind> = form.fields.iter().map(|field| &field.kind).collect();
    assert!(matches!(kinds[0], Kind::Text { .. }));
    assert_eq!(
      kinds[1],
      &Kind::One(vec![("Small".into(), "s".into()), ("Large".into(), "l".into())])
    );
    assert!(matches!(kinds[2], Kind::Many { .. }));
    assert_eq!(kinds[3], &Kind::Boolean);
    let shown = text(&form);
    assert!(shown.starts_with("Tell me."), "the server's message first: {shown}");
    assert!(shown.contains("Name (required)"), "{shown}");
    assert_eq!(form.title(), "srv is asking");
  }

  #[test]
  fn answers_go_back_as_the_values_the_schema_names() {
    let mut form = form(serde_json::json!({
      "type": "object",
      "properties": {
        "count": { "type": "integer", "maximum": 5 },
        "size": { "type": "string", "oneOf": [
          { "const": "s", "title": "Small" },
          { "const": "l", "title": "Large" },
        ] },
      },
    }));
    // A number over the bound is said, and the form stays up.
    type_in(&mut form, "9");
    assert!(press(&mut form, KeyCode::Enter).is_none());
    press(&mut form, KeyCode::Down);
    assert!(press(&mut form, KeyCode::Enter).is_none(), "Large, then the review");
    assert!(press(&mut form, KeyCode::Enter).is_none(), "refused");
    assert!(text(&form).contains("count must be at most 5"), "{}", text(&form));

    press(&mut form, KeyCode::Tab);
    press(&mut form, KeyCode::Backspace);
    type_in(&mut form, "4");
    press(&mut form, KeyCode::Enter);
    press(&mut form, KeyCode::Tab);
    let result = press(&mut form, KeyCode::Enter).expect("submitted");
    assert_eq!(result.action, ElicitationAction::Accept);
    assert_eq!(result.content, Some(serde_json::json!({ "count": 4, "size": "l" })));
  }

  #[test]
  fn walking_away_and_saying_no_are_told_apart() {
    let schema = serde_json::json!({
      "type": "object",
      "properties": { "a": { "type": "boolean" }, "b": { "type": "boolean" } },
    });
    let mut walked = form(schema.clone());
    let result = press(&mut walked, KeyCode::Esc).expect("gone");
    assert_eq!(result.action, ElicitationAction::Cancel);

    let mut refused = form(schema);
    press(&mut refused, KeyCode::BackTab);
    press(&mut refused, KeyCode::Down);
    let result = press(&mut refused, KeyCode::Enter).expect("Cancel on the submit tab");
    assert_eq!(result.action, ElicitationAction::Decline);
    assert_eq!(result.content, None);
  }

  #[test]
  fn a_required_field_left_blank_is_named() {
    let mut form = form(serde_json::json!({
      "type": "object",
      "properties": { "who": { "type": "string", "title": "Who" } },
      "required": ["who"],
    }));
    assert!(press(&mut form, KeyCode::Enter).is_none());
    assert!(text(&form).contains("Required: Who"), "{}", text(&form));
    type_in(&mut form, "me");
    let result = press(&mut form, KeyCode::Enter).expect("answered");
    assert_eq!(result.content, Some(serde_json::json!({ "who": "me" })));
  }
}
