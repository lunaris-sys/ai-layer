//! Document-parsing isolation for the Lunaris AI layer (Foundation §8.4).
//!
//! Untrusted documents — a PDF, a web page, a file the user asked the AI
//! to summarise — are parsed in a **separate, sandboxed subprocess**
//! before any of their text reaches a prompt. The subprocess has no
//! network access and no filesystem access; it reads the document bytes
//! from stdin and writes only the extracted, stripped plain text to
//! stdout. A parser exploited by a crafted document therefore cannot
//! reach the network or the graph, and only inert text crosses the
//! sandbox boundary.
//!
//! This crate is both:
//! - the **library** the AI layer calls: [`parse_document`] spawns the
//!   sandbox worker, feeds it bytes, and returns the extracted text (or
//!   an error — callers fail closed and pass no text on);
//! - the **worker binary** (`lunaris-doc-sandbox`): it calls
//!   [`apply_sandbox`] to lock itself down, then reads stdin, runs
//!   [`extract_text`], and writes stdout.
//!
//! The sandbox uses no_new_privs + a Landlock ruleset that grants no
//! filesystem access + a seccomp filter that blocks socket creation, so
//! it needs no privileges (unprivileged Landlock and seccomp, Linux
//! ≥5.13). Already-open fds (stdin/stdout) keep working; opening any new
//! path or socket is denied.

#![warn(missing_docs)]

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use thiserror::Error;

#[cfg(target_os = "linux")]
mod sandbox;
#[cfg(target_os = "linux")]
pub use sandbox::apply_sandbox;

/// The largest document the worker will accept, and the largest text it
/// will return. Bounds memory against a hostile or pathological input.
pub const MAX_BYTES: usize = 16 * 1024 * 1024;

/// How long the worker is allowed to run before the parent kills it.
const PARSE_TIMEOUT: Duration = Duration::from_secs(20);

/// A document-isolation failure. Every variant means no trustworthy
/// text was produced, so callers must treat it as fail-closed and pass
/// nothing on to the model.
#[derive(Debug, Error)]
pub enum SandboxError {
    /// The sandbox could not be installed (Landlock/seccomp/prctl
    /// failed). The worker exits rather than parse unsandboxed.
    #[error("sandbox setup failed: {0}")]
    Setup(String),
    /// The worker could not be spawned, or I/O to it failed.
    #[error("worker process error: {0}")]
    Process(String),
    /// The worker exited non-zero (it failed to sandbox or to parse).
    #[error("worker failed: {0}")]
    WorkerFailed(String),
    /// The worker exceeded the time budget and was killed.
    #[error("worker timed out")]
    Timeout,
    /// The input or the produced text exceeded [`MAX_BYTES`].
    #[error("document too large")]
    TooLarge,
}

/// Extract inert plain text from raw document bytes.
///
/// This is the transformation that runs **inside** the sandbox. The
/// first version handles UTF-8 / plain text: it decodes lossily and
/// strips control characters that could carry hidden instructions
/// (ANSI escape sequences, C0/C1 controls), keeping only ordinary
/// printable text plus newlines and tabs. Richer extractors (PDF, HTML,
/// office formats) plug in here behind the same boundary, so the risky
/// parse always runs sandboxed.
pub fn extract_text(bytes: &[u8]) -> Result<String, SandboxError> {
    if bytes.len() > MAX_BYTES {
        return Err(SandboxError::TooLarge);
    }
    let decoded = String::from_utf8_lossy(bytes);
    // Normalise CR / CRLF to LF up front so the loop only has to keep
    // LF and tab; this avoids turning a CRLF into a double newline.
    let normalized = decoded.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = String::with_capacity(normalized.len());
    for ch in normalized.chars() {
        if ch == '\n' || ch == '\t' {
            out.push(ch);
        } else if ch.is_control() {
            // Drop every other control char (C0/C1, ANSI escape
            // introducer): no readable content, can hide instructions
            // or terminal escapes.
            continue;
        } else if is_invisible_or_format(ch) {
            // Invisible, format, and bidirectional-override characters:
            // text that is hidden, reordered, or smuggled past a reader.
            continue;
        } else {
            out.push(ch);
        }
    }
    Ok(out)
}

/// Whether `ch` is an invisible, format, or bidirectional-control
/// character that should never survive into the extracted text. Covers
/// the practically-dangerous slice of Unicode's Default_Ignorable set:
/// soft hyphen, combining grapheme joiner, zero-width and bidi marks,
/// the word joiner and invisible math operators, deprecated format
/// controls, variation selectors, and the tag characters.
fn is_invisible_or_format(ch: char) -> bool {
    matches!(ch,
        '\u{00AD}'                  // soft hyphen
        | '\u{034F}'                // combining grapheme joiner
        | '\u{061C}'                // arabic letter mark
        | '\u{115F}'..='\u{1160}'   // hangul choseong/jungseong fillers
        | '\u{17B4}'..='\u{17B5}'   // khmer inherent vowels
        | '\u{180B}'..='\u{180F}'   // mongolian variation/separator
        | '\u{200B}'..='\u{200F}'   // zero-width + directional marks
        | '\u{202A}'..='\u{202E}'   // bidi embedding/override
        | '\u{2060}'..='\u{2064}'   // word joiner + invisible operators
        | '\u{2066}'..='\u{206F}'   // bidi isolates + deprecated format
        | '\u{3164}'                // hangul filler
        | '\u{FE00}'..='\u{FE0F}'   // variation selectors
        | '\u{FEFF}'                // zero-width no-break space / BOM
        | '\u{FFA0}'                // halfwidth hangul filler
        | '\u{1BCA0}'..='\u{1BCA3}' // shorthand format controls
        | '\u{1D173}'..='\u{1D17A}' // musical beam/slur format controls
        | '\u{E0000}'..='\u{E0FFF}' // tags + supplementary variation selectors
    )
}

/// Parse a document by running the sandbox worker as a subprocess.
///
/// `sandbox_bin` is the path to the `lunaris-doc-sandbox` binary. The
/// `document` bytes are written to the worker's stdin; its stdout (the
/// extracted text) is returned. The worker is killed if it runs past
/// the time budget, and both input and output are bounded by
/// [`MAX_BYTES`]. Any failure is a [`SandboxError`]; the caller passes
/// no text to the model on error.
pub fn parse_document(sandbox_bin: &Path, document: &[u8]) -> Result<String, SandboxError> {
    if document.len() > MAX_BYTES {
        return Err(SandboxError::TooLarge);
    }

    let mut child = Command::new(sandbox_bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| SandboxError::Process(format!("spawn: {e}")))?;

    // Feed stdin from a thread so a large document cannot deadlock
    // against a full stdout pipe.
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| SandboxError::Process("no stdin".to_string()))?;
    let input = document.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
        // Drop closes stdin so the worker sees EOF.
    });

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| SandboxError::Process("no stdout".to_string()))?;

    // Read stdout (capped) on a thread, so the wait can time out even if
    // the worker wedges mid-write.
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout
            .by_ref()
            .take((MAX_BYTES as u64) + 1)
            .read_to_end(&mut buf);
        buf
    });

    // Poll for exit up to the timeout, then kill.
    let deadline = std::time::Instant::now() + PARSE_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = writer.join();
                    let _ = reader.join();
                    return Err(SandboxError::Timeout);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => return Err(SandboxError::Process(format!("wait: {e}"))),
        }
    };

    let _ = writer.join();
    let output = reader
        .join()
        .map_err(|_| SandboxError::Process("stdout reader panicked".to_string()))?;

    if !status.success() {
        return Err(SandboxError::WorkerFailed(format!(
            "exit status {status}"
        )));
    }
    if output.len() > MAX_BYTES {
        return Err(SandboxError::TooLarge);
    }
    // The worker already emitted valid UTF-8 from extract_text.
    String::from_utf8(output)
        .map_err(|e| SandboxError::Process(format!("non-utf8 output: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_keeps_plain_text_and_newlines() {
        let t = extract_text(b"Hello world.\nSecond line.\tTabbed.").unwrap();
        assert_eq!(t, "Hello world.\nSecond line.\tTabbed.");
    }

    #[test]
    fn extract_strips_ansi_and_control_chars() {
        // ESC[31m ... ESC[0m and a bell should not survive.
        let raw = b"normal \x1b[31mred\x1b[0m text\x07end";
        let t = extract_text(raw).unwrap();
        assert!(!t.contains('\x1b'));
        assert!(!t.contains('\x07'));
        assert!(t.contains("normal"));
        assert!(t.contains("text"));
        assert!(t.contains("end"));
    }

    #[test]
    fn extract_strips_zero_width_and_bidi() {
        let raw = "vis\u{200B}ible\u{202E}reversed\u{FEFF}".as_bytes();
        let t = extract_text(raw).unwrap();
        assert!(!t.contains('\u{200B}'));
        assert!(!t.contains('\u{202E}'));
        assert!(!t.contains('\u{FEFF}'));
        assert!(t.contains("visible"));
    }

    #[test]
    fn extract_strips_the_wider_default_ignorable_set() {
        // soft hyphen, word joiner, variation selector, a tag char, CGJ.
        let raw =
            "a\u{00AD}b\u{2060}c\u{FE0F}d\u{E0041}e\u{034F}f".as_bytes();
        let t = extract_text(raw).unwrap();
        for c in ['\u{00AD}', '\u{2060}', '\u{FE0F}', '\u{E0041}', '\u{034F}'] {
            assert!(!t.contains(c), "must strip U+{:04X}", c as u32);
        }
        assert_eq!(t, "abcdef");
    }

    #[test]
    fn extract_normalises_crlf() {
        let t = extract_text(b"a\r\nb\rc").unwrap();
        assert_eq!(t, "a\nb\nc");
    }

    #[test]
    fn extract_rejects_oversize_input() {
        let big = vec![b'x'; MAX_BYTES + 1];
        assert!(matches!(extract_text(&big), Err(SandboxError::TooLarge)));
    }
}
