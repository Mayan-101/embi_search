use std::path::Path;

/// A single chunk of text ready to be embedded.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub chunk_index: usize,
}

/// Read a file and split it into embeddable chunks based on its type.
///
/// - `.pdf` → extract text, split on page breaks
/// - `.md`  → split on heading boundaries
/// - `.html` → strip tags, split on heading/paragraph boundaries
/// - anything else → split on paragraph boundaries
///
/// Images inside PDFs are skipped — only text content is extracted.
pub fn chunk_file(
    path: &Path,
    max_chunk_chars: usize,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let raw_text = match ext.as_str() {
        "pdf" => {
            let bytes = std::fs::read(path)?;
            pdf_extract::extract_text_from_mem(&bytes)
                .map_err(|e| format!("PDF extraction failed for {:?}: {}", path, e))?
        }
        "html" | "htm" => {
            let html = std::fs::read_to_string(path)?;
            strip_html_tags(&html)
        }
        _ => std::fs::read_to_string(path)?,
    };

    let raw_sections = match ext.as_str() {
        "pdf" => split_pdf_pages(&raw_text),
        "md" => split_markdown_headings(&raw_text),
        "html" | "htm" => split_paragraphs(&raw_text),
        _ => split_paragraphs(&raw_text),
    };

    // Apply merge/split rules and build final chunks.
    let final_sections = merge_and_split(raw_sections, max_chunk_chars);

    let chunks: Vec<Chunk> = final_sections
        .into_iter()
        .enumerate()
        .filter(|(_, text)| text.len() >= 50)
        .map(|(i, text)| Chunk {
            text,
            chunk_index: i,
        })
        .collect();

    Ok(chunks)
}

// ---------------------------------------------------------------------------
// Format-specific splitters
// ---------------------------------------------------------------------------

/// Split PDF text on form-feed characters (page breaks).
fn split_pdf_pages(text: &str) -> Vec<String> {
    text.split('\x0c')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Split Markdown on heading boundaries (`# `, `## `, `### `, etc.).
/// Each heading and everything until the next heading becomes one section.
fn split_markdown_headings(text: &str) -> Vec<String> {
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        if line.starts_with('#') && line.contains(' ') {
            // Found a heading — flush the previous section.
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sections.push(trimmed);
            }
            current = String::new();
        }
        current.push_str(line);
        current.push('\n');
    }

    // Flush the last section.
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sections.push(trimmed);
    }

    sections
}

/// Strip all HTML tags, leaving only the text content.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut inside_tag = false;

    for ch in html.chars() {
        match ch {
            '<' => inside_tag = true,
            '>' => {
                inside_tag = false;
                // Insert a space after closing tags to prevent words from merging.
                result.push(' ');
            }
            _ if !inside_tag => result.push(ch),
            _ => {}
        }
    }

    // Collapse multiple whitespace into single spaces and trim.
    result
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split plain text on paragraph boundaries (double newlines).
fn split_paragraphs(text: &str) -> Vec<String> {
    text.split("\n\n")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Merge / split logic
// ---------------------------------------------------------------------------

/// Merge small consecutive sections and split oversized ones.
///
/// 1. Consecutive sections smaller than `max / 4` are merged.
/// 2. Sections exceeding `max` are split on `\n\n`, then `. `, then hard-split.
fn merge_and_split(sections: Vec<String>, max_chunk_chars: usize) -> Vec<String> {
    let merge_threshold = max_chunk_chars / 4;
    let mut merged: Vec<String> = Vec::new();
    let mut buffer = String::new();

    for section in sections {
        if buffer.len() + section.len() + 1 <= merge_threshold {
            // Accumulate small sections.
            if !buffer.is_empty() {
                buffer.push_str("\n\n");
            }
            buffer.push_str(&section);
        } else {
            // Flush the buffer, then handle the current section.
            if !buffer.is_empty() {
                merged.push(std::mem::take(&mut buffer));
            }

            if section.len() <= max_chunk_chars {
                merged.push(section);
            } else {
                // Oversized — need to split further.
                merged.extend(force_split(&section, max_chunk_chars));
            }
        }
    }

    if !buffer.is_empty() {
        merged.push(buffer);
    }

    merged
}

/// Force-split an oversized section: first by `\n\n`, then by `. `, then hard-split.
fn force_split(text: &str, max: usize) -> Vec<String> {
    let mut result = Vec::new();

    // Try splitting on paragraph boundaries first.
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut current = String::new();

    for para in paragraphs {
        if current.len() + para.len() + 2 <= max {
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
        } else {
            if !current.is_empty() {
                result.push(std::mem::take(&mut current));
            }
            if para.len() <= max {
                current = para.to_string();
            } else {
                // Still too big — hard split by character boundary.
                let mut start = 0;
                while start < para.len() {
                    let end = (start + max).min(para.len());
                    // Try to find a space near the boundary to avoid splitting mid-word.
                    let split_at = if end < para.len() {
                        para[start..end]
                            .rfind(' ')
                            .map(|pos| start + pos + 1)
                            .unwrap_or(end)
                    } else {
                        end
                    };
                    result.push(para[start..split_at].trim().to_string());
                    start = split_at;
                }
            }
        }
    }

    if !current.is_empty() {
        result.push(current);
    }

    result
}
