//! What MCP servers hold for reading: resources, each a URI a server can be
//! asked for, and templates for the URIs it can make up.
//!
//! The model gets them as two tools of fa's own, beside the five: one to see
//! what there is, and one to read it. They are only offered when a server
//! says it has resources at all. The listing is the one kept for completing
//! `&server:uri` in the input box; a read asks the server, so a URI no list
//! names — one a template makes, or one a tool's answer points at instead of
//! carrying it — is still one `read_resource` away.

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rig_agent::tool::{Tool, ToolContext};
use rig_core::message::ToolResultContent;
use std::collections::BTreeMap;
use std::time::Duration;

use rmcp::ServiceError;
use rmcp::model::{ReadResourceRequestParams, Resource, ResourceContents, ResourceTemplate};
use rmcp::service::ServerSink;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::mcp::Catalog;
use crate::tools::{ToolError, read_image, tool_args};

/// What one server has to read, as it last said: fetched when it comes up
/// and again whenever it says its list has changed. What an `&` in the input
/// box is completed from, and what `list_resources` answers with.
#[derive(Clone, Debug, Default)]
pub struct ServerResources {
  pub resources: Vec<Resource>,
  /// URI templates, with `{…}` for the parts to fill in.
  pub templates: Vec<ResourceTemplate>,
  /// Why the last asking for the lists failed, if it did: what is above is
  /// then what an earlier one said, or nothing.
  pub failed: Option<String>,
}

// ---------------------------------------------------------------- asking

/// `ask`, given up on after `timeout` when there is one.
pub async fn bounded<T>(
  timeout: Option<Duration>,
  ask: impl Future<Output = Result<T, ServiceError>>,
) -> Result<T, ServiceError> {
  match timeout {
    Some(limit) => tokio::time::timeout(limit, ask)
      .await
      .unwrap_or(Err(ServiceError::Timeout { timeout: limit })),
    None => ask.await,
  }
}

/// Ask a server what it has to read, given up on after `timeout`: both
/// lists, every page of them, the templates being ones a server may not keep
/// at all, and may say so as an error rather than as an empty list.
pub async fn inventory(peer: &ServerSink, timeout: Option<Duration>) -> Result<ServerResources, ServiceError> {
  bounded(timeout, async {
    Ok(ServerResources {
      resources: peer.list_all_resources().await?,
      templates: peer.list_all_resource_templates().await.unwrap_or_default(),
      failed: None,
    })
  })
  .await
}

/// The names the two answer to, which no server's tool may take.
pub const NAMES: [&str; 2] = [ListResources::NAME, ReadResource::NAME];

/// Why `named` is not a server to ask, with the ones there are.
fn unknown<'a>(named: &str, servers: impl Iterator<Item = &'a String>) -> ToolError {
  let names: Vec<&str> = servers.map(String::as_str).collect();
  ToolError::new(format!(
    "No server {named} with resources. Servers with resources: {}",
    names.join(", ")
  ))
}

// ---------------------------------------------------------------- list

pub struct ListResources {
  pub catalog: Catalog,
}

#[derive(Deserialize, JsonSchema)]
pub struct ListArgs {
  /// Only this server's resources (all servers when omitted)
  server: Option<String>,
}

impl Tool for ListResources {
  const NAME: &'static str = "list_resources";
  type Output = String;
  tool_args!(ListArgs);

  fn description(&self) -> String {
    "List the resources MCP servers offer for reading — documents, records, files — grouped by \
     server, each with the URI to read it by, and the URI templates a server accepts, whose {…} \
     parts you fill in to make a URI. Read one with read_resource, giving the server and the URI."
      .to_string()
  }

  async fn call(&self, _ctx: &mut ToolContext, args: ListArgs) -> Result<String, ToolError> {
    listing(&self.catalog.resources(), args.server.as_deref())
  }
}

/// What `held` lists, for every server or the one `named`.
fn listing(held: &BTreeMap<String, ServerResources>, named: Option<&str>) -> Result<String, ToolError> {
  let servers: Vec<(&String, &ServerResources)> = match named {
    None => held.iter().collect(),
    Some(named) => vec![held.get_key_value(named).ok_or_else(|| unknown(named, held.keys()))?],
  };
  let mut out = Vec::new();
  for (server, listed) in servers {
    out.push(format!("{server}:"));
    // A listing that failed is not an empty one: what is there may still be
    // read by a URI known some other way.
    if let Some(err) = &listed.failed {
      out.push(format!("  (could not list: {err})"));
    }
    match (listed.resources.is_empty(), listed.failed.is_some()) {
      (true, true) => {}
      (true, false) => out.push("  (no resources)".to_string()),
      (false, failed) => {
        if failed {
          out.push("  as listed before:".to_string());
        }
        out.extend(listed.resources.iter().map(|r| {
          line(
            &r.uri,
            r.title.as_deref().unwrap_or(&r.name),
            r.mime_type.as_deref(),
            r.description.as_deref(),
          )
        }))
      }
    }
    if !listed.templates.is_empty() {
      out.push("  templates (fill in the {…} parts to make a URI):".to_string());
      out.extend(listed.templates.iter().map(|t| {
        line(
          &t.uri_template,
          t.title.as_deref().unwrap_or(&t.name),
          t.mime_type.as_deref(),
          t.description.as_deref(),
        )
      }));
    }
  }
  Ok(out.join("\n"))
}

/// One resource or template as a line of the listing.
fn line(uri: &str, name: &str, mime: Option<&str>, description: Option<&str>) -> String {
  let mut line = format!("  {uri} — {name}");
  if let Some(mime) = mime {
    line.push_str(&format!(" ({mime})"));
  }
  // One line, however the server wrote it.
  let description = description.map(|d| d.split_whitespace().collect::<Vec<_>>().join(" "));
  if let Some(description) = description.filter(|d| !d.is_empty()) {
    line.push_str(": ");
    line.push_str(&description);
  }
  line
}

// ---------------------------------------------------------------- read

pub struct ReadResource {
  pub catalog: Catalog,
  /// Whether the model takes images, as for `read`.
  pub vision: bool,
}

#[derive(Deserialize, JsonSchema)]
pub struct ReadArgs {
  /// The server that has the resource, as list_resources names it
  server: String,
  /// The resource's URI, as list_resources gives it or a template makes it
  uri: String,
}

impl Tool for ReadResource {
  const NAME: &'static str = "read_resource";
  type Output = Vec<ToolResultContent>;
  tool_args!(ReadArgs);

  fn description(&self) -> String {
    "Read a resource from an MCP server by its URI: one from list_resources, one a template was \
     filled in to make, one a tool's answer pointed at, or one the user named in a message as \
     &server:uri — the server's name, a colon, then the URI, maybe all in quotes: \
     &notes:note://today is server notes, URI note://today. Text comes back as text and images as \
     images; other binary contents are only described, by type and size."
      .to_string()
  }

  async fn call(&self, _ctx: &mut ToolContext, args: ReadArgs) -> Result<Vec<ToolResultContent>, ToolError> {
    let (peer, timeout) = self
      .catalog
      .resource_peer(&args.server)
      .ok_or_else(|| unknown(&args.server, self.catalog.resources().keys()))?;
    let read = bounded(timeout, peer.read_resource(ReadResourceRequestParams::new(&args.uri)))
      .await
      .map_err(|err| ToolError::new(format!("{}: {err}", args.uri)))?;
    if read.contents.is_empty() {
      return Ok(vec![ToolResultContent::text(format!("{}: empty", args.uri))]);
    }
    // One resource may come in several parts; each is headed by its own URI
    // when there is more than one, so they can be told apart.
    let several = read.contents.len() > 1;
    Ok(
      read
        .contents
        .into_iter()
        .flat_map(|contents| told(contents, &args.uri, several, self.vision))
        .collect(),
    )
  }
}

/// One part of what was read under `asked`, as the model is given it: text
/// as text, an image looked at the same way `read` looks at one, and
/// anything else binary said to be what it is, since its bytes mean nothing
/// as text. `headed` puts the part's own URI in front, for telling several
/// apart.
fn told(contents: ResourceContents, asked: &str, headed: bool, vision: bool) -> Vec<ToolResultContent> {
  match contents {
    ResourceContents::TextResourceContents { uri, text, .. } => vec![ToolResultContent::text(match headed {
      true => format!("{uri}:\n{text}"),
      false => text,
    })],
    ResourceContents::BlobResourceContents {
      uri, mime_type, blob, ..
    } => {
      let mime = mime_type.as_deref().unwrap_or("unknown type");
      let Ok(bytes) = STANDARD.decode(blob.trim()) else {
        return vec![ToolResultContent::text(format!(
          "{uri}: binary ({mime}), not valid base64"
        ))];
      };
      let Some(format) = crate::images::detect(&bytes) else {
        return vec![ToolResultContent::text(format!(
          "{uri}: binary ({mime}, {} bytes), not text",
          bytes.len()
        ))];
      };
      let image = read_image(&bytes, format, vision);
      match headed {
        true => std::iter::once(ToolResultContent::text(format!("{uri}:")))
          .chain(image)
          .collect(),
        false => image,
      }
    }
    // A kind of contents newer than this client.
    _ => vec![ToolResultContent::text(format!(
      "{asked}: contents of a kind fa cannot read"
    ))],
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn text(content: &[ToolResultContent]) -> String {
    content
      .iter()
      .filter_map(|c| match c {
        ToolResultContent::Text(t) => Some(t.text.clone()),
        _ => None,
      })
      .collect::<Vec<_>>()
      .join("|")
  }

  #[test]
  fn a_listing_is_a_line_a_resource() {
    assert_eq!(
      line("file:///a.md", "A", Some("text/markdown"), Some(" The\n  first\tone ")),
      "  file:///a.md — A (text/markdown): The first one"
    );
    assert_eq!(line("x://y", "Y", None, Some("   ")), "  x://y — Y");
  }

  #[test]
  fn a_listing_is_what_the_servers_hold_for_the_ones_asked_about() {
    let mut held = BTreeMap::new();
    held.insert(
      "notes".to_string(),
      ServerResources {
        resources: vec![Resource::new("note://today", "today")],
        templates: vec![ResourceTemplate::new("note://{day}", "day")],
        failed: None,
      },
    );
    held.insert("empty".to_string(), ServerResources::default());
    assert_eq!(
      listing(&held, Some("notes")).expect("a server it has"),
      "notes:\n  note://today — today\n  templates (fill in the {…} parts to make a URI):\n  note://{day} — day"
    );
    let all = listing(&held, None).expect("every server");
    assert!(
      all.contains("empty:\n  (no resources)") && all.contains("notes:"),
      "{all}"
    );
    // A listing that failed says so, and keeps what an earlier one said.
    held.insert(
      "down".to_string(),
      ServerResources {
        failed: Some("timed out".to_string()),
        ..ServerResources::default()
      },
    );
    assert_eq!(
      listing(&held, Some("down")).expect("a server it has"),
      "down:\n  (could not list: timed out)"
    );
    held.get_mut("notes").expect("notes").failed = Some("timed out".to_string());
    assert_eq!(
      listing(&held, Some("notes")).expect("a server it has"),
      "notes:\n  (could not list: timed out)\n  as listed before:\n  note://today — today\n  templates (fill in the {…} parts to make a URI):\n  note://{day} — day"
    );
    held.remove("down");
    let err = listing(&held, Some("other")).expect_err("no such server").to_string();
    assert!(err.contains("Servers with resources: empty, notes"), "{err}");
  }

  #[test]
  fn what_is_read_is_text_an_image_or_said_to_be_neither() {
    let contents = |contents: ResourceContents, headed: bool, vision: bool| told(contents, "file:///b", headed, vision);
    let hello = ResourceContents::text("hello", "file:///a");
    assert_eq!(text(&contents(hello.clone(), false, true)), "hello");
    assert_eq!(text(&contents(hello, true, true)), "file:///a:\nhello");

    let blob = |bytes: &[u8]| ResourceContents::BlobResourceContents {
      uri: "file:///b".into(),
      mime_type: Some("application/octet-stream".into()),
      blob: STANDARD.encode(bytes),
      meta: None,
    };
    let said = text(&contents(blob(&[0, 1, 2]), false, true));
    assert!(said.contains("binary (application/octet-stream, 3 bytes)"), "{said}");

    // A PNG is an image the model is shown, or told about when it cannot be.
    let mut png = Vec::new();
    image::RgbImage::new(2, 2)
      .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
      .expect("a png");
    let shown = contents(blob(&png), false, true);
    assert!(
      shown.iter().any(|c| matches!(c, ToolResultContent::Image(_))),
      "{shown:?}"
    );
    let told = contents(blob(&png), false, false);
    assert!(
      !told.iter().any(|c| matches!(c, ToolResultContent::Image(_))),
      "{told:?}"
    );
  }
}
