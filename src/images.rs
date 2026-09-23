//! Image preparation for the `read` tool: detect by magic bytes, convert anything but PNG/JPEG to PNG, and shrink until the
//! image fits within 2000x2000 pixels and 4.5 MB of base64.
//!
//! And the other direction: an image a tool answered with, drawn in the
//! transcript as half-blocks.

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageFormat, Rgba};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use rig_core::message::{DocumentSourceKind, ImageMediaType, ToolResultContent};

const MAX_WIDTH: u32 = 2000;
const MAX_HEIGHT: u32 = 2000;
/// Limit on the base64-encoded size.
const MAX_BASE64_BYTES: usize = 9 * 512 * 1024;
const JPEG_QUALITIES: [u8; 4] = [80, 70, 55, 40];
const UNCONVERTIBLE: &str = "[Image omitted: could not be converted to a supported inline image format.]";

pub struct ProcessedImage {
  /// The image as the model will be given it: PNG or JPEG, within the size
  /// limits. Kept as bytes rather than as base64 because the transcript
  /// draws the same image it sends, and drawing wants the bytes.
  pub bytes: Vec<u8>,
  pub media_type: ImageMediaType,
  /// Notes for the model about conversion or resizing.
  pub hints: Vec<String>,
}

impl ProcessedImage {
  /// `header`, then what converting or resizing the image did, a line each.
  pub fn note(&self, header: String) -> String {
    std::iter::once(header)
      .chain(self.hints.iter().cloned())
      .collect::<Vec<_>>()
      .join("\n")
  }

  /// The image as a provider takes it inline.
  pub fn base64(&self) -> String {
    STANDARD.encode(&self.bytes)
  }
}

/// Whether an image of this format can be sent inline.
pub fn supported(format: ImageFormat) -> bool {
  matches!(
    format,
    ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::Gif | ImageFormat::WebP | ImageFormat::Bmp
  )
}

/// The image formats fa accepts: jpg, png, gif, webp, bmp.
pub fn detect(bytes: &[u8]) -> Option<ImageFormat> {
  image::guess_format(bytes).ok().filter(|f| supported(*f))
}

/// Whether `bytes` of image stay under the base64 limit once encoded.
fn fits(bytes: &[u8]) -> bool {
  base64::encoded_len(bytes.len(), true).is_some_and(|len| len < MAX_BASE64_BYTES)
}

/// Prepare an image for the model. `Err` carries the note shown in place of
/// the image when it cannot be delivered.
pub fn process(bytes: &[u8], format: ImageFormat) -> Result<ProcessedImage, String> {
  let decoded = image::load_from_memory_with_format(bytes, format).map_err(|_| UNCONVERTIBLE.to_string())?;
  let (width, height) = decoded.dimensions();
  let mut hints = Vec::new();

  // Normalize to PNG or JPEG; providers accept both inline.
  let (bytes, media_type) = match format {
    ImageFormat::Png => (bytes.to_vec(), ImageMediaType::PNG),
    ImageFormat::Jpeg => (bytes.to_vec(), ImageMediaType::JPEG),
    other => {
      let png = encode_png(&decoded).ok_or_else(|| UNCONVERTIBLE.to_string())?;
      hints.push(format!("[Image converted from {} to image/png.]", other.to_mime_type()));
      (png, ImageMediaType::PNG)
    }
  };

  if width <= MAX_WIDTH && height <= MAX_HEIGHT && fits(&bytes) {
    return Ok(ProcessedImage {
      bytes,
      media_type,
      hints,
    });
  }

  let (encoded, media_type, w, h) = shrink(&decoded)
    .ok_or_else(|| "[Image omitted: could not be resized below the inline image size limit.]".to_string())?;
  let scale = width as f64 / w as f64;
  hints.push(format!(
        "[Image: original {width}x{height}, displayed at {w}x{h}. Multiply coordinates by {scale:.2} to map to original image.]"
    ));
  Ok(ProcessedImage {
    bytes: encoded,
    media_type,
    hints,
  })
}

/// Fit within the max dimensions, try PNG then JPEG at decreasing quality,
/// and keep shrinking by 25% until something fits.
fn shrink(image: &DynamicImage) -> Option<(Vec<u8>, ImageMediaType, u32, u32)> {
  let (ow, oh) = image.dimensions();
  // The box the image is fitted into, aspect kept; it is what shrinks.
  let (mut bw, mut bh) = (ow.min(MAX_WIDTH), oh.min(MAX_HEIGHT));
  loop {
    let resized = if (bw, bh) == (ow, oh) {
      image.clone()
    } else {
      image.resize(bw, bh, FilterType::Lanczos3)
    };
    let (w, h) = resized.dimensions();
    // Encoded one at a time, stopping at the first that fits: an encode of a
    // large image is the expensive part.
    let png = std::iter::once_with(|| encode_png(&resized).map(|b| (b, ImageMediaType::PNG)));
    let jpegs = JPEG_QUALITIES
      .iter()
      .map(|&quality| encode_jpeg(&resized, quality).map(|b| (b, ImageMediaType::JPEG)));
    if let Some((bytes, media_type)) = png.chain(jpegs).flatten().find(|(b, _)| fits(b)) {
      return Some((bytes, media_type, w, h));
    }
    if w == 1 && h == 1 {
      return None;
    }
    bw = (bw * 3 / 4).max(1);
    bh = (bh * 3 / 4).max(1);
  }
}

fn encode_png(image: &DynamicImage) -> Option<Vec<u8>> {
  let mut out = Vec::new();
  image.write_with_encoder(PngEncoder::new(&mut out)).ok()?;
  Some(out)
}

/// JPEG has no alpha channel; the encoder is handed the image without it.
fn encode_jpeg(image: &DynamicImage, quality: u8) -> Option<Vec<u8>> {
  let mut out = Vec::new();
  image
    .write_with_encoder(JpegEncoder::new_with_quality(&mut out, quality))
    .ok()?;
  Some(out)
}

/// A pixel this transparent is left to the terminal's own background.
const OPAQUE_ALPHA: u8 = 128;

/// What a tool result shows in the transcript: the text it said, and the
/// images it carried, which are drawn rather than described.
pub fn split(content: &[ToolResultContent]) -> (String, Vec<Vec<u8>>) {
  let mut images = Vec::new();
  let text = content
    .iter()
    .filter_map(|c| match c {
      ToolResultContent::Text(t) => Some(t.text.clone()),
      ToolResultContent::Json { value } => Some(value.to_string()),
      // An image the transcript can draw needs no line saying it was one.
      // An image it cannot reach — a URL, a file the provider holds — is
      // still worth the placeholder, since nothing else would say it came.
      ToolResultContent::Image(image) => match source_bytes(&image.data) {
        Some(bytes) => {
          images.push(bytes);
          None
        }
        None => Some("[image]".to_string()),
      },
    })
    .collect::<Vec<_>>()
    .join("\n");
  (text, images)
}

/// The bytes behind an image, for the sources that carry them.
pub fn source_bytes(source: &DocumentSourceKind) -> Option<Vec<u8>> {
  match source {
    DocumentSourceKind::Base64(data) => STANDARD.decode(data).ok(),
    DocumentSourceKind::Raw(bytes) => Some(bytes.clone()),
    _ => None,
  }
}

/// Draw an image as half-blocks, within `cols` columns and `rows` lines.
///
/// A cell is `▄`: its foreground is the lower pixel and its background the
/// upper one, so one cell carries one pixel across and two down — which is
/// the shape of a terminal cell, and keeps the image's own proportions. The
/// image is never enlarged: a 16x16 icon is drawn as the 8 lines it is
/// rather than blown up to the width of the transcript.
pub fn blocks(bytes: &[u8], cols: u16, rows: u16) -> Option<Vec<Line<'static>>> {
  if cols == 0 || rows == 0 {
    return None;
  }
  let image = image::load_from_memory(bytes).ok()?;
  let (width, height) = image.dimensions();
  if width == 0 || height == 0 {
    return None;
  }
  let (bound_w, bound_h) = (u32::from(cols).min(width), (u32::from(rows) * 2).min(height));
  let pixels = match (bound_w, bound_h) == (width, height) {
    true => image.to_rgba8(),
    // Triangle rather than Lanczos3: this runs on every width change, and at
    // preview size the difference is not one the blocks can show.
    false => image.resize(bound_w, bound_h, FilterType::Triangle).to_rgba8(),
  };
  let (width, height) = pixels.dimensions();
  let mut lines = Vec::with_capacity(height.div_ceil(2) as usize);
  for y in (0..height).step_by(2) {
    let mut spans: Vec<Span<'static>> = Vec::new();
    for x in 0..width {
      let upper = pixels.get_pixel(x, y);
      let lower = (y + 1 < height).then(|| pixels.get_pixel(x, y + 1));
      let (ch, style) = cell(upper, lower);
      // A run of one colour is one span: a screenshot is mostly flat
      // background, and a span per pixel would be a span per cell of it.
      match spans.last_mut() {
        Some(last) if last.style == style => last.content.to_mut().push(ch),
        _ => spans.push(Span::styled(ch.to_string(), style)),
      }
    }
    lines.push(Line::from(spans));
  }
  Some(lines)
}

/// The character and colours one cell is drawn with, from the two pixels it
/// stands for. Transparent halves keep the terminal's background instead of a
/// colour of their own, so an icon on no background reads as one.
fn cell(upper: &Rgba<u8>, lower: Option<&Rgba<u8>>) -> (char, Style) {
  let colour = |p: &Rgba<u8>| (p.0[3] >= OPAQUE_ALPHA).then(|| Color::Rgb(p.0[0], p.0[1], p.0[2]));
  match (colour(upper), lower.and_then(colour)) {
    (Some(upper), Some(lower)) => ('▄', Style::default().fg(lower).bg(upper)),
    (Some(upper), None) => ('▀', Style::default().fg(upper)),
    (None, Some(lower)) => ('▄', Style::default().fg(lower)),
    (None, None) => (' ', Style::default()),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use image::ImageBuffer;
  use std::io::Cursor;

  fn png(width: u32, height: u32) -> Vec<u8> {
    let img = ImageBuffer::from_fn(width, height, |x, y| Rgba([(x % 256) as u8, (y % 256) as u8, 128, 255]));
    let mut out = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(img)
      .write_to(&mut out, ImageFormat::Png)
      .unwrap();
    out.into_inner()
  }

  /// A transparent image, to see what the terminal's own background is left
  /// showing through.
  fn clear_png(width: u32, height: u32) -> Vec<u8> {
    let img = ImageBuffer::from_fn(width, height, |_, _| Rgba([9u8, 9, 9, 0]));
    let mut out = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(img)
      .write_to(&mut out, ImageFormat::Png)
      .unwrap();
    out.into_inner()
  }

  /// Cells in a drawn line, counting the runs that were merged into one span.
  fn cells(line: &Line<'static>) -> String {
    line.spans.iter().map(|s| s.content.as_ref()).collect()
  }

  #[test]
  fn detects_supported_formats_only() {
    assert_eq!(detect(&png(4, 4)), Some(ImageFormat::Png));
    assert_eq!(detect(b"just some text"), None);
    let mut bmp = Cursor::new(Vec::new());
    image::load_from_memory(&png(4, 4))
      .unwrap()
      .write_to(&mut bmp, ImageFormat::Bmp)
      .unwrap();
    assert_eq!(detect(&bmp.into_inner()), Some(ImageFormat::Bmp));
  }

  #[test]
  fn small_png_passes_through_untouched() {
    let bytes = png(10, 10);
    let out = process(&bytes, ImageFormat::Png).unwrap();
    assert_eq!(out.media_type, ImageMediaType::PNG);
    assert!(out.hints.is_empty());
    assert_eq!(out.bytes, bytes);
  }

  #[test]
  fn bmp_is_converted_to_png() {
    let mut bmp = Cursor::new(Vec::new());
    image::load_from_memory(&png(4, 4))
      .unwrap()
      .write_to(&mut bmp, ImageFormat::Bmp)
      .unwrap();
    let out = process(&bmp.into_inner(), ImageFormat::Bmp).unwrap();
    assert_eq!(out.media_type, ImageMediaType::PNG);
    assert_eq!(out.hints, ["[Image converted from image/bmp to image/png.]"]);
    assert_eq!(image::guess_format(&out.bytes).unwrap(), ImageFormat::Png);
  }

  #[test]
  fn a_cell_carries_the_two_pixels_it_stands_for() {
    // Four pixels across and four down is four cells across and two lines,
    // each cell the lower pixel on the upper one.
    let lines = blocks(&png(4, 4), 80, 40).unwrap();
    assert_eq!(lines.len(), 2);
    assert_eq!(cells(&lines[0]), "▄▄▄▄");
    let first = &lines[0].spans[0];
    assert_eq!(first.style.bg, Some(Color::Rgb(0, 0, 128)));
    assert_eq!(first.style.fg, Some(Color::Rgb(0, 1, 128)));
  }

  #[test]
  fn an_odd_row_is_drawn_as_the_upper_half_alone() {
    let lines = blocks(&png(2, 3), 80, 40).unwrap();
    assert_eq!(lines.len(), 2);
    assert_eq!(cells(&lines[1]), "▀▀");
    assert_eq!(lines[1].spans[0].style.fg, Some(Color::Rgb(0, 2, 128)));
    assert_eq!(lines[1].spans[0].style.bg, None);
  }

  #[test]
  fn transparent_pixels_are_left_to_the_terminal() {
    let lines = blocks(&clear_png(3, 2), 80, 40).unwrap();
    assert_eq!(cells(&lines[0]), "   ");
    // One run of one style, rather than a span a cell.
    assert_eq!(lines[0].spans.len(), 1);
    assert_eq!(lines[0].spans[0].style.fg, None);
  }

  #[test]
  fn a_small_image_is_drawn_at_its_own_size() {
    // Not blown up to the width it was offered: an icon is an icon.
    let lines = blocks(&png(8, 8), 200, 100).unwrap();
    assert_eq!(lines.len(), 4);
    assert_eq!(cells(&lines[0]).chars().count(), 8);
  }

  #[test]
  fn a_large_image_is_scaled_to_fit_both_bounds() {
    // 400x200 in 40 columns is 40 pixels across, 20 down — ten lines, and
    // the proportions it arrived with.
    let lines = blocks(&png(400, 200), 40, 40).unwrap();
    assert_eq!(lines.len(), 10);
    assert_eq!(cells(&lines[0]).chars().count(), 40);
    // The line cap binds when it is the tighter of the two.
    let lines = blocks(&png(400, 200), 400, 6).unwrap();
    assert_eq!(lines.len(), 6);
    assert_eq!(cells(&lines[0]).chars().count(), 24);
  }

  #[test]
  fn a_wide_image_fills_the_width_it_is_given() {
    // The line bound is there for the very tall and narrow; what a picture
    // is usually shaped like is bound by the width, which is the detail the
    // terminal can hold.
    let lines = blocks(&png(1200, 800), 77, 80).unwrap();
    assert_eq!(cells(&lines[0]).chars().count(), 77);
    assert_eq!(lines.len(), 26);
  }

  #[test]
  fn nothing_is_drawn_for_bytes_that_are_not_an_image() {
    assert!(blocks(b"just some text", 40, 40).is_none());
    assert!(blocks(&png(4, 4), 0, 40).is_none());
  }

  #[test]
  fn a_result_splits_into_what_it_said_and_what_it_showed() {
    let bytes = png(4, 4);
    let content = vec![
      ToolResultContent::text("Read image file [image/png]"),
      ToolResultContent::image_base64(STANDARD.encode(&bytes), Some(ImageMediaType::PNG), None),
    ];
    let (text, images) = split(&content);
    assert_eq!(text, "Read image file [image/png]");
    assert_eq!(images, [bytes]);
  }

  #[test]
  fn an_image_the_transcript_cannot_reach_keeps_its_placeholder() {
    let content = vec![ToolResultContent::Image(rig_core::message::Image {
      data: DocumentSourceKind::url("https://example.com/cat.png"),
      ..Default::default()
    })];
    let (text, images) = split(&content);
    assert_eq!(text, "[image]");
    assert!(images.is_empty());
  }

  #[test]
  fn oversized_image_is_resized_with_a_note() {
    let out = process(&png(3000, 300), ImageFormat::Png).unwrap();
    let decoded = image::load_from_memory(&out.bytes).unwrap();
    assert_eq!(decoded.dimensions(), (2000, 200));
    assert_eq!(
      out.hints,
      ["[Image: original 3000x300, displayed at 2000x200. Multiply coordinates by 1.50 to map to original image.]"]
    );
  }
}
