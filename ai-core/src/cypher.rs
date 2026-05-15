//! Post-build Cypher self-check.
//!
//! The structured DSL in [`crate::graph_query`] is the primary
//! safety boundary: the AI never emits Cypher, the daemon builds it
//! from a typed, validated [`crate::graph_query::GraphQuery`]. This
//! module is defence-in-depth on top of that: after the builder runs,
//! [`verify_built_cypher`] re-scans the *generated* string and
//! confirms it carries no write keyword, has a `LIMIT`, and
//! references only labels the validated query declared.
//!
//! A failure here is a *builder bug*, not a caller attack, because
//! the builder input was already validated. The daemon treats it as
//! an internal error rather than feeding it back to the model.
//!
//! The scan honours Cypher string literals (`"..."` / `'...'` with
//! `\` escapes) and comments (`// ...` and `/* ... */`) so a quoted
//! filter value such as `'CREATE'` does not trip the keyword check.

use std::collections::BTreeSet;

/// Write / side-effecting keywords that must never appear in a
/// daemon-built read query.
const FORBIDDEN: &[&str] = &[
    "CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP", "CALL", "LOAD", "FOREACH",
];

/// Errors raised by the self-check.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SelfCheckError {
    /// Built Cypher contains a forbidden keyword.
    #[error("built cypher contains forbidden keyword '{keyword}'")]
    ForbiddenKeyword {
        /// The offending keyword (uppercase).
        keyword: String,
    },
    /// Built Cypher has no `LIMIT` clause.
    #[error("built cypher has no LIMIT clause")]
    MissingLimit,
    /// Built Cypher references a label the validated query did not
    /// declare.
    #[error("built cypher references unexpected label '{label}'")]
    UnexpectedLabel {
        /// The unexpected label.
        label: String,
    },
}

/// Verify a daemon-built Cypher string.
///
/// `expected_labels` is the set returned by
/// [`crate::graph_query::GraphQuery::referenced_labels`].
pub fn verify_built_cypher(
    cypher: &str,
    expected_labels: &BTreeSet<String>,
) -> Result<(), SelfCheckError> {
    let stripped = strip_strings_and_comments(cypher);
    if let Some(keyword) = forbidden_keyword(&stripped) {
        return Err(SelfCheckError::ForbiddenKeyword { keyword });
    }
    if !has_limit(&stripped) {
        return Err(SelfCheckError::MissingLimit);
    }
    for label in extract_namespaces(cypher) {
        if !expected_labels.contains(&label) {
            return Err(SelfCheckError::UnexpectedLabel { label });
        }
    }
    Ok(())
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn forbidden_keyword(stripped: &str) -> Option<String> {
    let upper = stripped.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    for kw in FORBIDDEN {
        let kbytes = kw.as_bytes();
        let mut i = 0;
        while i + kbytes.len() <= bytes.len() {
            if bytes[i..i + kbytes.len()] == *kbytes {
                let before_ok = i == 0 || !is_word_byte(bytes[i - 1]);
                let after_idx = i + kbytes.len();
                let after_ok = after_idx == bytes.len() || !is_word_byte(bytes[after_idx]);
                if before_ok && after_ok {
                    return Some(kw.to_string());
                }
            }
            i += 1;
        }
    }
    None
}

fn has_limit(stripped: &str) -> bool {
    let upper = stripped.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    let needle = b"LIMIT";
    let mut i = 0;
    while i + needle.len() <= bytes.len() {
        if bytes[i..i + needle.len()] == *needle {
            let before_ok = i == 0 || !is_word_byte(bytes[i - 1]);
            let after_idx = i + needle.len();
            let after_ok = after_idx == bytes.len() || !is_word_byte(bytes[after_idx]);
            if before_ok && after_ok {
                // Confirm a digit follows (after optional whitespace).
                let mut j = after_idx;
                while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                    j += 1;
                }
                if j < bytes.len() && bytes[j].is_ascii_digit() {
                    return true;
                }
            }
        }
        i += 1;
    }
    false
}

fn is_label_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'.'
}

/// Extract every `:label` namespace referenced by the Cypher,
/// skipping string literals and comments.
fn extract_namespaces(cypher: &str) -> Vec<String> {
    let bytes = cypher.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'"' | b'\'' => {
                let quote = b;
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == quote {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                i += 2;
                while i + 1 < bytes.len() {
                    if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
            }
            b':' => {
                let mut start = i + 1;
                if start < bytes.len() && bytes[start] == b':' {
                    start += 1;
                }
                let mut end = start;
                while end < bytes.len() && is_label_char(bytes[end]) {
                    end += 1;
                }
                if end > start {
                    if let Ok(label) = std::str::from_utf8(&bytes[start..end]) {
                        if !label.is_empty() && !out.iter().any(|n| n == label) {
                            out.push(label.to_string());
                        }
                    }
                }
                i = end;
            }
            _ => i += 1,
        }
    }
    out
}

/// Replace string-literal and comment content with spaces, preserving
/// byte length, so keyword / limit scans don't trip inside them.
fn strip_strings_and_comments(cypher: &str) -> String {
    let bytes = cypher.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'"' | b'\'' => {
                let quote = b;
                out.push(b' ');
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        out.push(b' ');
                        out.push(b' ');
                        i += 2;
                        continue;
                    }
                    if bytes[i] == quote {
                        out.push(b' ');
                        i += 1;
                        break;
                    }
                    out.push(b' ');
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    out.push(b' ');
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                out.push(b' ');
                out.push(b' ');
                i += 2;
                while i + 1 < bytes.len() {
                    if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        out.push(b' ');
                        out.push(b' ');
                        i += 2;
                        break;
                    }
                    out.push(b' ');
                    i += 1;
                }
            }
            _ => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn well_formed_built_cypher_passes() {
        let cypher = "MATCH (f:File)-[:ACCESSED_BY]->(a:App)\n\
                      RETURN f.path, a.name\nLIMIT 50";
        verify_built_cypher(cypher, &labels(&["File", "ACCESSED_BY", "App"]))
            .expect("must pass");
    }

    #[test]
    fn missing_limit_is_caught() {
        let cypher = "MATCH (f:File) RETURN f.path";
        assert_eq!(
            verify_built_cypher(cypher, &labels(&["File"])),
            Err(SelfCheckError::MissingLimit)
        );
    }

    #[test]
    fn write_keyword_is_caught() {
        let cypher = "MATCH (f:File) CREATE (x:File) RETURN f LIMIT 1";
        match verify_built_cypher(cypher, &labels(&["File"])) {
            Err(SelfCheckError::ForbiddenKeyword { keyword }) => {
                assert_eq!(keyword, "CREATE");
            }
            other => panic!("expected ForbiddenKeyword, got {other:?}"),
        }
    }

    #[test]
    fn unexpected_label_is_caught() {
        // Builder somehow emitted a label the DSL never declared.
        let cypher = "MATCH (f:File)-[:ACCESSED_BY]->(s:Session) RETURN f LIMIT 1";
        match verify_built_cypher(cypher, &labels(&["File", "ACCESSED_BY"])) {
            Err(SelfCheckError::UnexpectedLabel { label }) => {
                assert_eq!(label, "Session");
            }
            other => panic!("expected UnexpectedLabel, got {other:?}"),
        }
    }

    #[test]
    fn keyword_inside_string_literal_is_ignored() {
        let cypher =
            "MATCH (f:File) WHERE f.path CONTAINS 'CREATE' RETURN f.path LIMIT 10";
        verify_built_cypher(cypher, &labels(&["File"])).expect("must pass");
    }

    #[test]
    fn label_inside_string_literal_is_ignored() {
        // A quoted value that looks like `:Session` must not count
        // as a referenced label.
        let cypher =
            "MATCH (f:File) WHERE f.path = 'a:Session b' RETURN f.path LIMIT 10";
        verify_built_cypher(cypher, &labels(&["File"])).expect("must pass");
    }

    #[test]
    fn keyword_must_be_whole_word() {
        // "created_at" must not trip the CREATE check.
        let cypher = "MATCH (p:Project) RETURN p.created_at LIMIT 5";
        verify_built_cypher(cypher, &labels(&["Project"])).expect("must pass");
    }
}
