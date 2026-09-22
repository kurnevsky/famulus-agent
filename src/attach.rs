//! Images attached to a prompt, written as `@path` in the input box.
//!
//! The token stays in the text that is sent. It is not the image — that
//! travels as a content part of its own — but the sentence has to read as
//! one, the token says which image is which when there is more than one,
//! and the session keeps only what was sent: a prompt handed back by `/tree`
//! or walked back to with `Up` brings its attachments with it because they
//! are written in it.

use std::path::{Path, PathBuf};

use image::ImageFormat;
use rig_core::completion::Message;
use rig_core::message::{MimeType, UserContent};

use crate::images::{self, ProcessedImage};
use crate::tools::resolve;

/// Trailing characters a sentence leaves stuck to a path: `@shot.png,` is a
/// token and a comma. Only tried when the whole run is not a file, so a name
/// that really does end in one still resolves.
const TRAILING: &[char] = &['.', ',', ';', ':', '!', '?', ')', ']', '}', '"', '\''];

/// An `@path` as it stands in the input box, and what is behind it.
pub struct Token {
  /// Byte range of the token in the input text, `@` and quotes included.
  pub range: (usize, usize),
  /// The token as written, `@` included: what labels the image in the
  /// message that carries it.
  pub text: String,
  pub path: PathBuf,
  pub state: State,
}

impl Token {
  /// What the input box's own summary calls it: the file's name, which is
  /// the part that tells two attachments apart.
  pub fn name(&self) -> String {
    self
      .path
      .file_name()
      .map(|n| n.to_string_lossy().into_owned())
      .unwrap_or_else(|| self.text.clone())
  }

  pub fn is_image(&self) -> bool {
    matches!(self.state, State::Image { .. })
  }
}

/// What resolving a token found.
pub enum State {
  /// An image that can be sent, at the size it is on disk.
  Image { width: u32, height: u32 },
  /// A directory: not an attachment, but the way to one — this is what a
  /// token reads as while its path is still being completed.
  Directory,
  /// Nothing at that path.
  Missing,
  /// Something is there, but not an image fa can send.
  NotAnImage,
}

/// Every `@path` in `text`, in the order they are written.
///
/// A token starts a word: an email address is not an attachment. A path with
/// spaces goes in quotes, which is also how a dropped file is written down.
pub fn tokens(text: &str, cwd: &Path) -> Vec<Token> {
  let mut out = Vec::new();
  let mut at = 0;
  while let Some(found) = text[at..].find('@') {
    let start = at + found;
    at = start + '@'.len_utf8();
    if start > 0 && !text[..start].ends_with(char::is_whitespace) {
      continue;
    }
    let rest = &text[at..];
    let (raw, len, quoted) = match rest.strip_prefix('"') {
      Some(inner) => match inner.find('"') {
        Some(end) => (&inner[..end], end + 2, true),
        // An unclosed quote is still being typed.
        None => (inner, inner.len() + 1, true),
      },
      None => {
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        (&rest[..end], end, false)
      }
    };
    if raw.is_empty() {
      continue;
    }
    at += len;
    out.push(probe(text, start, at, raw, quoted, cwd));
  }
  out
}

/// Resolve one token, trying it without a sentence's punctuation when the
/// path as written is not there.
fn probe(text: &str, start: usize, end: usize, raw: &str, quoted: bool, cwd: &Path) -> Token {
  let trimmed = match quoted {
    true => raw,
    false => raw.trim_end_matches(TRAILING),
  };
  let token = |raw: &str, end: usize| {
    // `tools::resolve` strips a leading `@`, which is a chat reference when
    // the model writes one — but our own sigil has already been taken off,
    // so a second one is part of the name. `./` keeps it from being eaten,
    // which is what otherwise turned `@@` into the working directory.
    let path = match raw.starts_with('@') {
      true => resolve(cwd, &format!("./{raw}")),
      false => resolve(cwd, raw),
    };
    let state = describe(&path);
    Token {
      range: (start, end),
      text: text[start..end].to_string(),
      path,
      state,
    }
  };
  let whole = token(raw, end);
  // The punctuation is the sentence's only when what is left is a file: a
  // name that ends in a bracket is still that name.
  if whole.is_image() || trimmed.is_empty() || trimmed == raw {
    return whole;
  }
  let shorter = token(trimmed, end - (raw.len() - trimmed.len()));
  match shorter.is_image() {
    true => shorter,
    // Neither is there; the shorter one is the likelier thing to have meant,
    // and the friendlier thing to name in the report.
    false => match whole.state {
      State::NotAnImage => whole,
      _ => shorter,
    },
  }
}

/// What is at `path`, read from the file's header rather than by decoding it:
/// this runs as the user types.
fn describe(path: &Path) -> State {
  if path.is_dir() {
    return State::Directory;
  }
  if !path.is_file() {
    return State::Missing;
  }
  let Ok(reader) = image::ImageReader::open(path).and_then(image::ImageReader::with_guessed_format) else {
    return State::NotAnImage;
  };
  if !reader.format().is_some_and(images::supported) {
    return State::NotAnImage;
  }
  match reader.into_dimensions() {
    Ok((width, height)) => State::Image { width, height },
    Err(_) => State::NotAnImage,
  }
}

/// The size of the image at `path`, read from its header. `None` for
/// anything that is not one.
pub fn dimensions(path: &Path) -> Option<(u32, u32)> {
  match describe(path) {
    State::Image { width, height } => Some((width, height)),
    _ => None,
  }
}

/// Whether `path` is an image worth offering for an attachment, by its name
/// alone — for listing a directory, where opening every file would be a
/// syscall a keystroke.
pub fn looks_like_image(path: &Path) -> bool {
  ImageFormat::from_path(path).is_ok_and(images::supported)
}

/// An image read and prepared for sending: the bytes as the model will be
/// given them, which is also what the transcript draws.
pub struct Attached {
  /// The token that named it, for the note that labels it.
  pub token: String,
  pub image: ProcessedImage,
}

/// Read and prepare the image behind a token. `Err` is the reason, in the
/// words the transcript says it in.
pub fn load(token: &Token) -> Result<Attached, String> {
  let bytes = std::fs::read(&token.path).map_err(|err| format!("{}: {err}", token.text))?;
  let format = images::detect(&bytes).ok_or_else(|| format!("{}: not an image fa can send.", token.text))?;
  let image = images::process(&bytes, format).map_err(|reason| format!("{}: {reason}", token.text))?;
  Ok(Attached {
    token: token.text.clone(),
    image,
  })
}

/// A prompt on its way to the model: what was typed, and the images its
/// tokens resolved to.
pub struct Prompt {
  pub text: String,
  pub images: Vec<Attached>,
}

impl Prompt {
  /// A prompt carrying nothing but its text, which is every prompt without
  /// an `@` in it and every slash command.
  pub fn text(text: String) -> Self {
    Self {
      text,
      images: Vec::new(),
    }
  }

  /// The message as the provider is given it: the prose, then each image
  /// behind a note naming it — the shape `read` already answers in, and
  /// where the hints about a converted or resized image belong.
  pub fn message(&self) -> Message {
    let mut content = Vec::with_capacity(1 + self.images.len() * 2);
    content.push(UserContent::text(self.text.clone()));
    for Attached { token, image } in &self.images {
      let header = format!("[Attached {token} — {}]", image.media_type.to_mime_type());
      content.push(UserContent::text(image.note(header)));
      content.push(UserContent::image_base64(
        image.base64(),
        Some(image.media_type.clone()),
        None,
      ));
    }
    Message::User { content }
  }

  /// The bytes the transcript draws under the prompt.
  pub fn preview(&self) -> Vec<Vec<u8>> {
    self
      .images
      .iter()
      .map(|attached| attached.image.bytes.clone())
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use image::{ImageBuffer, Rgba};
  use std::io::Cursor;

  fn dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fa-attach-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
  }

  fn png(dir: &Path, name: &str, width: u32, height: u32) -> PathBuf {
    let image = ImageBuffer::from_fn(width, height, |x, y| Rgba([(x % 256) as u8, (y % 256) as u8, 7, 255]));
    let mut out = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
      .write_to(&mut out, ImageFormat::Png)
      .unwrap();
    let path = dir.join(name);
    std::fs::write(&path, out.into_inner()).unwrap();
    path
  }

  #[test]
  fn a_token_resolves_against_the_working_directory() {
    let dir = dir("cwd");
    png(&dir, "shot.png", 8, 4);
    let found = tokens("why does @shot.png do that?", &dir);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].text, "@shot.png");
    assert!(matches!(found[0].state, State::Image { width: 8, height: 4 }));
  }

  #[test]
  fn a_full_path_is_taken_as_it_is() {
    let dir = dir("absolute");
    let path = png(&dir, "shot.png", 6, 6);
    // Resolved from somewhere else entirely, to be sure the cwd is not used.
    let found = tokens(&format!("look at @{}", path.display()), Path::new("/"));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].path, path);
    assert!(found[0].is_image());
  }

  #[test]
  fn a_path_with_spaces_goes_in_quotes() {
    let dir = dir("spaces");
    let path = png(&dir, "a shot.png", 4, 4);
    let found = tokens(&format!("see @\"{}\" please", path.display()), Path::new("/"));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].path, path);
    assert!(found[0].is_image());
    // Unquoted, the run stops at the space and finds nothing.
    let bare = tokens(&format!("see @{} please", path.display()), Path::new("/"));
    assert!(!bare[0].is_image());
  }

  #[test]
  fn the_sentences_punctuation_is_not_part_of_the_path() {
    let dir = dir("punctuation");
    png(&dir, "shot.png", 4, 4);
    let found = tokens("what is in @shot.png?", &dir);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].text, "@shot.png");
    assert!(found[0].is_image());
  }

  #[test]
  fn a_name_that_ends_in_punctuation_is_still_that_name() {
    let dir = dir("bracket");
    png(&dir, "shot(1).png", 4, 4);
    // Trimming would leave `shot(1`, which is not the file; the whole run is.
    let found = tokens("look at @shot(1).png", &dir);
    assert!(found[0].is_image());
    assert_eq!(found[0].name(), "shot(1).png");
  }

  #[test]
  fn a_second_sigil_is_part_of_the_name_rather_than_the_working_directory() {
    let dir = dir("doubled");
    // `@@` used to hand a bare `@` to a resolver that strips one, leaving an
    // empty path that joined to the working directory — so the box reported
    // the directory's own name as a missing file.
    let found = tokens("@@", &dir);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].path, dir.join("@"));
    assert_eq!(found[0].name(), "@");
    assert!(matches!(found[0].state, State::Missing));
    // And a file whose name really does start with one is reachable.
    png(&dir, "@odd.png", 4, 4);
    let found = tokens("look at @\"@odd.png\"", &dir);
    assert!(found[0].is_image(), "a name may start with the sigil");
    assert_eq!(found[0].name(), "@odd.png");
  }

  #[test]
  fn a_directory_is_told_apart_from_a_missing_file() {
    let dir = dir("directory");
    std::fs::create_dir_all(dir.join("shots")).unwrap();
    // What a path reads as while it is still being completed. Calling it
    // missing would be wrong, and it is the common case: every completed
    // path passes through it.
    let found = tokens("@shots/", &dir);
    assert!(matches!(found[0].state, State::Directory));
    // `~` is a directory too, not a missing file named after the home one.
    if std::env::var_os("HOME").is_some() {
      let found = tokens("@~", &dir);
      assert!(matches!(found[0].state, State::Directory));
    }
  }

  #[test]
  fn an_email_address_is_not_an_attachment() {
    let dir = dir("email");
    assert!(tokens("mail someone@example.com about it", &dir).is_empty());
  }

  #[test]
  fn what_is_not_there_and_what_is_not_an_image_are_told_apart() {
    let dir = dir("states");
    std::fs::write(dir.join("notes.txt"), "not an image").unwrap();
    let found = tokens("@gone.png and @notes.txt", &dir);
    assert_eq!(found.len(), 2);
    assert!(matches!(found[0].state, State::Missing));
    assert!(matches!(found[1].state, State::NotAnImage));
  }

  #[test]
  fn several_tokens_keep_the_order_they_are_written_in() {
    let dir = dir("several");
    png(&dir, "a.png", 2, 2);
    png(&dir, "b.png", 3, 3);
    let found = tokens("compare @a.png and @b.png", &dir);
    let names: Vec<String> = found.iter().map(Token::name).collect();
    assert_eq!(names, ["a.png", "b.png"]);
    // The ranges are where they are written, for the popup to replace one.
    assert_eq!(
      &"compare @a.png and @b.png"[found[0].range.0..found[0].range.1],
      "@a.png"
    );
    assert_eq!(
      &"compare @a.png and @b.png"[found[1].range.0..found[1].range.1],
      "@b.png"
    );
  }

  #[test]
  fn the_message_carries_the_prose_then_a_note_and_the_image() {
    let dir = dir("message");
    png(&dir, "shot.png", 4, 4);
    let found = tokens("what is @shot.png", &dir);
    let prompt = Prompt {
      text: "what is @shot.png".into(),
      images: vec![load(&found[0]).unwrap()],
    };
    let Message::User { content } = prompt.message() else {
      panic!("a user message")
    };
    assert_eq!(content.len(), 3);
    // The token is left in the prose: the sentence reads, and the note below
    // binds it to the image that follows.
    assert!(matches!(&content[0], UserContent::Text(t) if t.text == "what is @shot.png"));
    assert!(matches!(&content[1], UserContent::Text(t) if t.text == "[Attached @shot.png — image/png]"));
    assert!(matches!(&content[2], UserContent::Image(_)));
  }

  #[test]
  fn a_prompt_with_no_images_is_one_text_part() {
    let Message::User { content } = Prompt::text("plain".into()).message() else {
      panic!("a user message")
    };
    assert_eq!(content.len(), 1);
  }
}
