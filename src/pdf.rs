//! PDFs for the `read` tool: detect by magic bytes and turn the text layer
//! into Markdown, a `<!-- Page N -->` marker before each page so the model
//! can tell where it is in the document.

use pdf_inspector::{MarkdownOptions, OCR_REASON_SUSPECTED_GARBLED_TEXT, PdfError, PdfOptions};

/// Whether `bytes` are a PDF: they open with its header.
pub fn detect(bytes: &[u8]) -> bool {
  bytes.starts_with(b"%PDF-")
}

pub struct Extracted {
  pub markdown: String,
  pub pages: u32,
  /// 1-indexed pages with no text to take: a scan, text drawn as outlines,
  /// or nothing at all. What they show is not in `markdown`.
  pub missing: Vec<u32>,
  /// 1-indexed pages whose text is in `markdown` but came through a font
  /// that did not decode cleanly, so some of it may be wrong.
  pub garbled: Vec<u32>,
}

impl Extracted {
  /// What the model is told about the pages that did not come through
  /// whole: a line for those left out and one for those that may be
  /// garbled. `None` when every page did.
  pub fn note(&self) -> Option<String> {
    if self.markdown.trim().is_empty() {
      return Some(format!(
        "[PDF has {} but no text to extract: scanned, or text drawn as shapes.]",
        if self.pages == 1 {
          "1 page".to_string()
        } else {
          format!("{} pages", self.pages)
        }
      ));
    }
    let mut notes = Vec::new();
    if !self.missing.is_empty() {
      let (noun, verb, they) = if self.missing.len() == 1 {
        ("Page", "has", "it shows")
      } else {
        ("Pages", "have", "they show")
      };
      notes.push(format!(
        "[{noun} {} {verb} no text to extract (scanned, or text drawn as shapes); what {they} is not included.]",
        ranges(&self.missing)
      ));
    }
    if !self.garbled.is_empty() {
      let noun = if self.garbled.len() == 1 { "Page" } else { "Pages" };
      notes.push(format!(
        "[{noun} {}: text in fonts that did not decode cleanly, some of it may be garbled.]",
        ranges(&self.garbled)
      ));
    }
    (!notes.is_empty()).then(|| notes.join("\n"))
  }
}

/// Sorted page numbers with each run written as its ends: `1, 3-5, 9`.
fn ranges(pages: &[u32]) -> String {
  let mut runs: Vec<(u32, u32)> = Vec::new();
  for &page in pages {
    match runs.last_mut() {
      Some((_, last)) if *last + 1 == page => *last = page,
      _ => runs.push((page, page)),
    }
  }
  runs
    .iter()
    .map(|&(first, last)| {
      if first == last {
        first.to_string()
      } else {
        format!("{first}-{last}")
      }
    })
    .collect::<Vec<_>>()
    .join(", ")
}

/// Turn a PDF into Markdown. Parsing is CPU-bound and can take a while on a
/// large document, so it is done off the async runtime; a document malformed
/// enough to make the parser panic is reported like any other failure.
pub async fn extract(bytes: Vec<u8>) -> Result<Extracted, String> {
  let result = tokio::task::spawn_blocking(move || {
    let options = PdfOptions::new().markdown(MarkdownOptions {
      include_page_numbers: true,
      ..MarkdownOptions::default()
    });
    pdf_inspector::process_pdf_mem_with_options(&bytes, options)
  })
  .await
  .map_err(|_| "could not parse PDF".to_string())?
  .map_err(|e: PdfError| e.to_string())?;

  // Only a page with a reason given is known to have lost something: the
  // crate also flags every page of a document with little text on it, text
  // and all, in case that little is all that could be read.
  let (mut missing, mut garbled) = (Vec::new(), Vec::new());
  for page in &result.ocr_reasons_by_page {
    if page.reasons.is_empty() {
      continue;
    }
    let only_garbled = page
      .reasons
      .iter()
      .all(|reason| reason == OCR_REASON_SUSPECTED_GARBLED_TEXT);
    if only_garbled { &mut garbled } else { &mut missing }.push(page.page);
  }
  missing.sort_unstable();
  garbled.sort_unstable();
  Ok(Extracted {
    markdown: result.markdown.unwrap_or_default(),
    pages: result.page_count,
    missing,
    garbled,
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn runs_of_pages_are_written_as_ranges() {
    assert_eq!(ranges(&[1]), "1");
    assert_eq!(ranges(&[1, 3, 4, 5, 9, 10]), "1, 3-5, 9-10");
  }

  #[test]
  fn notes_name_the_pages_that_did_not_come_through() {
    let pdf = |markdown: &str, missing: Vec<u32>, garbled: Vec<u32>| Extracted {
      markdown: markdown.to_string(),
      pages: 30,
      missing,
      garbled,
    };
    assert_eq!(pdf("text", vec![], vec![]).note(), None);
    assert_eq!(
      pdf("text", vec![14, 15, 16], vec![1]).note().unwrap(),
      "[Pages 14-16 have no text to extract (scanned, or text drawn as shapes); what they show is not included.]\n\
       [Page 1: text in fonts that did not decode cleanly, some of it may be garbled.]"
    );
    assert_eq!(
      pdf("\n\n", vec![], vec![]).note().unwrap(),
      "[PDF has 30 pages but no text to extract: scanned, or text drawn as shapes.]"
    );
  }
}
