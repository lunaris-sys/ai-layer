//! Endpoint allowlist enforcement.
//!
//! Decisions land in [`AllowlistDecision`]. The allowlist matches on
//! the parsed URL's *host*, never on a string prefix, because prefix
//! matching is a known pivot for SSRF (Foundation §8.4.6).
//!
//! Schemes are also restricted: HTTPS for any non-loopback host, and
//! HTTP only for `localhost` / `127.0.0.1` / `::1` so a local Ollama
//! or llama.cpp can be reached without TLS.

use std::collections::BTreeSet;
use url::Url;

/// Allowlist policy.
///
/// `hosts` is the canonical set of permitted hostnames. The default
/// shipped by Lunaris covers the canonical providers in
/// Foundation §5.3.
#[derive(Debug, Clone)]
pub struct Allowlist {
    hosts: BTreeSet<String>,
}

impl Allowlist {
    /// Build an allowlist from an arbitrary iterator of hostnames.
    pub fn new<I, S>(hosts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            hosts: hosts.into_iter().map(|h| h.into().to_ascii_lowercase()).collect(),
        }
    }

    /// The default Lunaris allowlist. Covers Anthropic, OpenAI, and
    /// the loopback origins for local providers.
    pub fn default_lunaris() -> Self {
        Self::new([
            "api.anthropic.com",
            "api.openai.com",
            "localhost",
            "127.0.0.1",
            "::1",
        ])
    }

    /// Borrow the underlying host set (for `list_allowed_endpoints`).
    pub fn hosts(&self) -> impl Iterator<Item = &str> {
        self.hosts.iter().map(String::as_str)
    }

    /// Evaluate whether `endpoint_url` may be reached.
    pub fn check(&self, endpoint_url: &str) -> AllowlistDecision {
        let parsed = match Url::parse(endpoint_url) {
            Ok(url) => url,
            Err(_) => return AllowlistDecision::Rejected(RejectReason::InvalidUrl),
        };
        let scheme = parsed.scheme();
        let host_opt = parsed
            .host_str()
            .map(|h| h.to_ascii_lowercase())
            .filter(|h| !h.is_empty());

        // Scheme is checked first so `file:///etc/passwd` and similar
        // surface as DisallowedScheme rather than MissingHost.
        let is_loopback = host_opt
            .as_deref()
            .map(|h| h == "localhost" || h == "127.0.0.1" || h == "::1")
            .unwrap_or(false);
        match scheme {
            "https" => {}
            "http" if is_loopback => {}
            other => {
                return AllowlistDecision::Rejected(RejectReason::DisallowedScheme {
                    scheme: other.to_string(),
                })
            }
        }

        let host = match host_opt {
            Some(h) => h,
            None => return AllowlistDecision::Rejected(RejectReason::MissingHost),
        };
        if !self.hosts.contains(&host) {
            return AllowlistDecision::Rejected(RejectReason::HostNotAllowed { host });
        }
        AllowlistDecision::Allowed { host }
    }
}

/// Outcome of an allowlist check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllowlistDecision {
    /// The endpoint is permitted; the normalised host is included so
    /// the audit log records the canonical form rather than the
    /// caller-supplied URL.
    Allowed {
        /// Canonical lowercase hostname.
        host: String,
    },
    /// The endpoint is rejected. The reason is structured so the
    /// daemon can report a stable error code to callers.
    Rejected(RejectReason),
}

/// Reasons a URL fails the allowlist check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// URL did not parse.
    InvalidUrl,
    /// Parsed URL had no host component.
    MissingHost,
    /// Scheme is not `https`, or `http` on a non-loopback host.
    DisallowedScheme {
        /// The offending scheme.
        scheme: String,
    },
    /// Host is not in the allowlist.
    HostNotAllowed {
        /// Canonical lowercase hostname.
        host: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_to_allowlisted_host_passes() {
        let al = Allowlist::default_lunaris();
        assert_eq!(
            al.check("https://api.anthropic.com/v1/messages"),
            AllowlistDecision::Allowed {
                host: "api.anthropic.com".to_string()
            }
        );
    }

    #[test]
    fn http_to_localhost_passes() {
        let al = Allowlist::default_lunaris();
        assert_eq!(
            al.check("http://localhost:11434/v1/chat/completions"),
            AllowlistDecision::Allowed {
                host: "localhost".to_string()
            }
        );
        assert_eq!(
            al.check("http://127.0.0.1:11434/v1/chat/completions"),
            AllowlistDecision::Allowed {
                host: "127.0.0.1".to_string()
            }
        );
    }

    #[test]
    fn http_to_remote_host_is_rejected() {
        let al = Allowlist::default_lunaris();
        match al.check("http://api.anthropic.com/v1/messages") {
            AllowlistDecision::Rejected(RejectReason::DisallowedScheme { scheme }) => {
                assert_eq!(scheme, "http");
            }
            other => panic!("unexpected decision: {other:?}"),
        }
    }

    #[test]
    fn file_scheme_is_rejected() {
        let al = Allowlist::default_lunaris();
        match al.check("file:///etc/passwd") {
            AllowlistDecision::Rejected(RejectReason::DisallowedScheme { scheme }) => {
                assert_eq!(scheme, "file");
            }
            other => panic!("unexpected decision: {other:?}"),
        }
    }

    #[test]
    fn host_outside_allowlist_is_rejected() {
        let al = Allowlist::default_lunaris();
        match al.check("https://evil.example.com/x") {
            AllowlistDecision::Rejected(RejectReason::HostNotAllowed { host }) => {
                assert_eq!(host, "evil.example.com");
            }
            other => panic!("unexpected decision: {other:?}"),
        }
    }

    #[test]
    fn prefix_attack_on_allowlisted_host_is_rejected() {
        // The classic SSRF payload: attacker hopes string-prefix matching
        // will accept "api.anthropic.com.attacker.example".
        let al = Allowlist::default_lunaris();
        match al.check("https://api.anthropic.com.attacker.example/x") {
            AllowlistDecision::Rejected(RejectReason::HostNotAllowed { host }) => {
                assert_eq!(host, "api.anthropic.com.attacker.example");
            }
            other => panic!("unexpected decision: {other:?}"),
        }
    }

    #[test]
    fn host_match_is_case_insensitive() {
        let al = Allowlist::default_lunaris();
        match al.check("https://API.ANTHROPIC.COM/v1/messages") {
            AllowlistDecision::Allowed { host } => assert_eq!(host, "api.anthropic.com"),
            other => panic!("unexpected decision: {other:?}"),
        }
    }

    #[test]
    fn invalid_url_is_rejected() {
        let al = Allowlist::default_lunaris();
        assert_eq!(
            al.check("not a url"),
            AllowlistDecision::Rejected(RejectReason::InvalidUrl)
        );
    }

    #[test]
    fn empty_host_url_is_rejected_as_invalid() {
        // `https://` parses to a WHATWG URL with the empty-host error
        // surfaced as a parse failure, so we land on `InvalidUrl`.
        // The `MissingHost` variant stays in the enum as defensive
        // coverage for any future url crate change that returns an
        // empty host string instead of an error.
        let al = Allowlist::default_lunaris();
        assert_eq!(
            al.check("https://"),
            AllowlistDecision::Rejected(RejectReason::InvalidUrl)
        );
    }

    #[test]
    fn hosts_iterator_returns_canonical_set() {
        let al = Allowlist::new(["A.example", "b.example"]);
        let hosts: Vec<_> = al.hosts().collect();
        // BTreeSet sorts ascii-lowercase: a.example, b.example
        assert_eq!(hosts, vec!["a.example", "b.example"]);
    }
}
