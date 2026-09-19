//! Image preparation for the `read` tool, following pi's rules: detect by
//! magic bytes, convert anything but PNG/JPEG to PNG, and shrink until the
//! image fits within 2000x2000 pixels and 4.5 MB of base64.

use std::io::Cursor;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageFormat};
use rig_core::message::ImageMediaType;

const MAX_WIDTH: u32 = 2000;
const MAX_HEIGHT: u32 = 2000;
/// Limit on the base64-encoded size, as pi measures it.
const MAX_BASE64_BYTES: usize = (4.5 * 1024.0 * 1024.0) as usize;
const JPEG_QUALITIES: [u8; 5] = [80, 85, 70, 55, 40];

pub struct ProcessedImage {
  pub base64: String,
  pub media_type: ImageMediaType,
  /// Notes for the model about conversion or resizing.
  pub hints: Vec<String>,
}

/// The image formats the `read` tool accepts (pi's list: jpg, png, gif, webp, bmp).
pub fn detect(bytes: &[u8]) -> Option<ImageFormat> {
  match image::guess_format(bytes).ok()? {
    f @ (ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::Gif | ImageFormat::WebP | ImageFormat::Bmp) => Some(f),
    _ => None,
  }
}

pub fn mime_type(format: ImageFormat) -> &'static str {
  match format {
    ImageFormat::Png => "image/png",
    ImageFormat::Jpeg => "image/jpeg",
    ImageFormat::Gif => "image/gif",
    ImageFormat::WebP => "image/webp",
    ImageFormat::Bmp => "image/bmp",
    _ => "application/octet-stream",
  }
}

fn base64_len(bytes: usize) -> usize {
  bytes.div_ceil(3) * 4
}

/// Prepare an image for the model. `Err` carries the note pi shows in place of
/// the image when it cannot be delivered.
pub fn process(bytes: &[u8], format: ImageFormat) -> Result<ProcessedImage, String> {
  let decoded = image::load_from_memory_with_format(bytes, format)
    .map_err(|_| "[Image omitted: could not be converted to a supported inline image format.]".to_string())?;
  let (width, height) = decoded.dimensions();
  let mut hints = Vec::new();

  // Normalize to PNG or JPEG; providers accept both inline.
  let (bytes, media_type) = match format {
    ImageFormat::Png => (bytes.to_vec(), ImageMediaType::PNG),
    ImageFormat::Jpeg => (bytes.to_vec(), ImageMediaType::JPEG),
    other => {
      let png = encode_png(&decoded)
        .ok_or_else(|| "[Image omitted: could not be converted to a supported inline image format.]".to_string())?;
      hints.push(format!("[Image converted from {} to image/png.]", mime_type(other)));
      (png, ImageMediaType::PNG)
    }
  };

  if width <= MAX_WIDTH && height <= MAX_HEIGHT && base64_len(bytes.len()) < MAX_BASE64_BYTES {
    return Ok(ProcessedImage {
      base64: STANDARD.encode(&bytes),
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
    base64: STANDARD.encode(&encoded),
    media_type,
    hints,
  })
}

/// pi's strategy: fit within the max dimensions, try PNG then JPEG at
/// decreasing quality, and keep shrinking by 25% until something fits.
fn shrink(image: &DynamicImage) -> Option<(Vec<u8>, ImageMediaType, u32, u32)> {
  let (ow, oh) = image.dimensions();
  let ratio = f64::min(MAX_WIDTH as f64 / ow as f64, MAX_HEIGHT as f64 / oh as f64).min(1.0);
  let mut w = ((ow as f64 * ratio) as u32).max(1);
  let mut h = ((oh as f64 * ratio) as u32).max(1);
  loop {
    let resized = if (w, h) == (ow, oh) {
      image.clone()
    } else {
      image.resize_exact(w, h, FilterType::Lanczos3)
    };
    let mut candidates: Vec<(Vec<u8>, ImageMediaType)> = Vec::new();
    if let Some(png) = encode_png(&resized) {
      candidates.push((png, ImageMediaType::PNG));
    }
    for quality in JPEG_QUALITIES {
      if let Some(jpeg) = encode_jpeg(&resized, quality) {
        candidates.push((jpeg, ImageMediaType::JPEG));
      }
    }
    if let Some((bytes, media_type)) = candidates
      .into_iter()
      .find(|(b, _)| base64_len(b.len()) < MAX_BASE64_BYTES)
    {
      return Some((bytes, media_type, w, h));
    }
    if w == 1 && h == 1 {
      return None;
    }
    w = ((w as f64 * 0.75) as u32).max(1);
    h = ((h as f64 * 0.75) as u32).max(1);
  }
}

fn encode_png(image: &DynamicImage) -> Option<Vec<u8>> {
  let mut out = Cursor::new(Vec::new());
  image.write_to(&mut out, ImageFormat::Png).ok()?;
  Some(out.into_inner())
}

fn encode_jpeg(image: &DynamicImage, quality: u8) -> Option<Vec<u8>> {
  let mut out = Cursor::new(Vec::new());
  // JPEG has no alpha channel.
  let rgb = image.to_rgb8();
  let encoder = JpegEncoder::new_with_quality(&mut out, quality);
  rgb.write_with_encoder(encoder).ok()?;
  Some(out.into_inner())
}

#[cfg(test)]
mod tests {
  use super::*;
  use image::{ImageBuffer, Rgba};

  fn png(width: u32, height: u32) -> Vec<u8> {
    let img = ImageBuffer::from_fn(width, height, |x, y| Rgba([(x % 256) as u8, (y % 256) as u8, 128, 255]));
    let mut out = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(img)
      .write_to(&mut out, ImageFormat::Png)
      .unwrap();
    out.into_inner()
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
    assert_eq!(STANDARD.decode(out.base64).unwrap(), bytes);
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
    assert_eq!(
      image::guess_format(&STANDARD.decode(out.base64).unwrap()).unwrap(),
      ImageFormat::Png
    );
  }

  #[test]
  fn oversized_image_is_resized_with_a_note() {
    let out = process(&png(3000, 300), ImageFormat::Png).unwrap();
    let decoded = image::load_from_memory(&STANDARD.decode(out.base64).unwrap()).unwrap();
    assert_eq!(decoded.dimensions(), (2000, 200));
    assert_eq!(
      out.hints,
      ["[Image: original 3000x300, displayed at 2000x200. Multiply coordinates by 1.50 to map to original image.]"]
    );
  }
}
