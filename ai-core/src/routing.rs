//! Routing engine for the Lunaris AI layer.
//!
//! Loads `~/.config/lunaris/ai-routing.toml` and resolves each request
//! to a provider name using first-match-wins semantics
//! (Foundation §5.3). The Settings UI is the authoritative editor for
//! this file; the engine stays a pure consumer and never writes back.
//!
//! ## Schema
//!
//! ```toml
//! [[rule]]
//! name = "code requests stay local"
//! match.content_type = "code"
//! provider = "ollama-llama3-8b"
//!
//! [[rule]]
//! name = "default fallback"
//! provider = "ollama-llama3-8b"
//! fallback = "anthropic-claude-sonnet-4-7"
//!
//! [providers.ollama-llama3-8b]
//! backend  = "ollama"
//! endpoint = "http://localhost:11434"
//! model    = "llama3:8b"
//!
//! [providers.anthropic-claude-sonnet-4-7]
//! backend         = "anthropic"
//! model           = "claude-sonnet-4-7"
//! api_key_keyring = "anthropic-api-key"
//! ```
//!
//! ## Semantics
//!
//! * Rules are evaluated top-down. The first rule whose every set
//!   match dimension agrees with the request wins (AND within a rule,
//!   OR across rules).
//! * A rule with no `match` block matches any request, so it makes a
//!   natural default-tail entry.
//! * `fallback` is reported alongside the primary provider name; the
//!   caller decides when to switch (typically on
//!   [`crate::provider::ProviderError::Unavailable`]).
//! * Unknown provider names are rejected at config load, not at
//!   resolve time, so misconfigurations surface in Settings before
//!   they reach a query.

use serde::Deserialize;
use std::collections::HashMap;

/// Top-level routing configuration parsed from `ai-routing.toml`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RoutingConfig {
    /// Rules evaluated top-down (first match wins).
    #[serde(default, rename = "rule")]
    pub rules: Vec<RuleSpec>,

    /// Provider definitions referenced by [`RuleSpec::provider`] and
    /// [`RuleSpec::fallback`].
    #[serde(default)]
    pub providers: HashMap<String, ProviderSpec>,
}

/// A single routing rule.
#[derive(Debug, Clone, Deserialize)]
pub struct RuleSpec {
    /// Human-readable label, surfaced in Settings and the audit log.
    pub name: String,

    /// Conditions that must all be met for this rule to apply
    /// (AND semantics within a rule).
    #[serde(default, rename = "match")]
    pub matcher: MatchSpec,

    /// Provider key. Must exist in [`RoutingConfig::providers`].
    pub provider: String,

    /// Optional provider key used when the primary is unavailable.
    #[serde(default)]
    pub fallback: Option<String>,
}

/// Conditions on a routing rule. All set fields must match.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct MatchSpec {
    /// Classifier-detected content type.
    #[serde(default)]
    pub content_type: Option<ContentType>,

    /// PII classifier hit.
    #[serde(default)]
    pub contains_personal_data: Option<bool>,

    /// Caller's app identity (per peer-auth).
    #[serde(default)]
    pub caller_app_id: Option<String>,

    /// Inclusive lower bound on
    /// [`RoutingContext::query_size_tokens`]. When set, the rule
    /// applies only if the request size is at least this many tokens.
    #[serde(default)]
    pub query_size_tokens_min: Option<u32>,

    /// Higher-level intent tag.
    #[serde(default)]
    pub intent: Option<Intent>,
}

/// Content classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ContentType {
    /// Source code in any language.
    Code,
    /// Free-form prose.
    Prose,
    /// Mathematical expressions or proofs.
    Math,
    /// Structured data (JSON, TOML, CSV, ...).
    StructuredData,
}

/// Higher-level intent of the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Intent {
    /// Question-and-answer interaction.
    Qa,
    /// Tool invocation.
    ToolUse,
    /// Summarisation of an input document.
    Summarize,
    /// Translation between languages.
    Translate,
}

/// Provider definition referenced by routing rules.
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderSpec {
    /// Backend kind. Canonical values: `ollama`, `llamacpp`,
    /// `anthropic`, `openai`.
    pub backend: String,
    /// Endpoint base URL. Required for HTTP backends, ignored for
    /// in-process ones.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Model identifier.
    pub model: String,
    /// Keyring entry name where the API key is stored. Used by cloud
    /// backends; ignored by local ones.
    #[serde(default)]
    pub api_key_keyring: Option<String>,
}

/// Per-request match dimensions handed to the engine at resolve time.
#[derive(Debug, Clone, Default)]
pub struct RoutingContext {
    /// Detected content type for the prompt.
    pub content_type: Option<ContentType>,
    /// Whether the PII classifier matched.
    pub contains_personal_data: bool,
    /// Identity of the calling app.
    pub caller_app_id: Option<String>,
    /// Estimated number of input tokens.
    pub query_size_tokens: u32,
    /// Detected intent.
    pub intent: Option<Intent>,
}

/// Resolution of a request to a provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRoute {
    /// Name of the matched rule (mirrors [`RuleSpec::name`]).
    pub rule_name: String,
    /// Name of the primary provider chosen.
    pub provider: String,
    /// Optional fallback provider name.
    pub fallback: Option<String>,
}

/// Errors raised by the routing engine.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RouteError {
    /// No rule matched the request.
    #[error("no matching route")]
    NoMatchingRoute,
    /// A rule referenced a provider not declared in `[providers.*]`.
    #[error("rule '{rule}' references unknown provider '{provider}'")]
    UnknownProvider {
        /// Rule name.
        rule: String,
        /// Referenced provider name.
        provider: String,
    },
}

/// Errors raised while loading a routing config from TOML.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// TOML deserialisation failed.
    #[error("toml parse: {0}")]
    Toml(#[from] toml::de::Error),
    /// The config parsed but was internally inconsistent.
    #[error(transparent)]
    Validation(RouteError),
}

/// The routing engine.
///
/// Constructed once at daemon startup or whenever
/// `ai-routing.toml` changes. The engine is immutable after
/// construction; reloads create a fresh [`RoutingEngine`] which
/// atomically replaces the old one.
#[derive(Debug, Clone)]
pub struct RoutingEngine {
    config: RoutingConfig,
}

impl RoutingEngine {
    /// Wrap an already-parsed [`RoutingConfig`].
    ///
    /// Returns an error if any rule references a provider that is
    /// not declared in `[providers.*]`.
    pub fn new(config: RoutingConfig) -> Result<Self, RouteError> {
        for rule in &config.rules {
            if !config.providers.contains_key(&rule.provider) {
                return Err(RouteError::UnknownProvider {
                    rule: rule.name.clone(),
                    provider: rule.provider.clone(),
                });
            }
            if let Some(fallback) = &rule.fallback {
                if !config.providers.contains_key(fallback) {
                    return Err(RouteError::UnknownProvider {
                        rule: rule.name.clone(),
                        provider: fallback.clone(),
                    });
                }
            }
        }
        Ok(Self { config })
    }

    /// Parse a TOML document into a fully validated engine.
    pub fn from_toml(toml_str: &str) -> Result<Self, ParseError> {
        let config: RoutingConfig = toml::from_str(toml_str)?;
        Self::new(config).map_err(ParseError::Validation)
    }

    /// Resolve a request to a provider using first-match-wins.
    pub fn resolve(&self, ctx: &RoutingContext) -> Result<ResolvedRoute, RouteError> {
        for rule in &self.config.rules {
            if rule_matches(&rule.matcher, ctx) {
                return Ok(ResolvedRoute {
                    rule_name: rule.name.clone(),
                    provider: rule.provider.clone(),
                    fallback: rule.fallback.clone(),
                });
            }
        }
        Err(RouteError::NoMatchingRoute)
    }

    /// Borrow the raw provider catalog. Used by the proxy and the
    /// adapter layer to materialise concrete provider instances.
    pub fn providers(&self) -> &HashMap<String, ProviderSpec> {
        &self.config.providers
    }

    /// Number of rules in the config (mainly for diagnostics).
    pub fn rule_count(&self) -> usize {
        self.config.rules.len()
    }
}

fn rule_matches(matcher: &MatchSpec, ctx: &RoutingContext) -> bool {
    if let Some(want) = matcher.content_type {
        if ctx.content_type != Some(want) {
            return false;
        }
    }
    if let Some(want) = matcher.contains_personal_data {
        if ctx.contains_personal_data != want {
            return false;
        }
    }
    if let Some(want) = matcher.caller_app_id.as_deref() {
        if ctx.caller_app_id.as_deref() != Some(want) {
            return false;
        }
    }
    if let Some(min) = matcher.query_size_tokens_min {
        if ctx.query_size_tokens < min {
            return false;
        }
    }
    if let Some(want) = matcher.intent {
        if ctx.intent != Some(want) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(name: &str) -> (String, ProviderSpec) {
        (
            name.to_string(),
            ProviderSpec {
                backend: "ollama".to_string(),
                endpoint: Some("http://localhost:11434".to_string()),
                model: "llama3:8b".to_string(),
                api_key_keyring: None,
            },
        )
    }

    fn engine_with_rules(rules: Vec<RuleSpec>, provider_names: &[&str]) -> RoutingEngine {
        let providers = provider_names.iter().map(|n| provider(n)).collect();
        RoutingEngine::new(RoutingConfig { rules, providers }).expect("engine builds")
    }

    #[test]
    fn empty_config_yields_no_matching_route() {
        let engine = engine_with_rules(vec![], &[]);
        let err = engine
            .resolve(&RoutingContext::default())
            .expect_err("must fail");
        assert_eq!(err, RouteError::NoMatchingRoute);
    }

    #[test]
    fn default_rule_with_empty_match_always_wins() {
        let engine = engine_with_rules(
            vec![RuleSpec {
                name: "default".to_string(),
                matcher: MatchSpec::default(),
                provider: "local".to_string(),
                fallback: None,
            }],
            &["local"],
        );
        let route = engine.resolve(&RoutingContext::default()).unwrap();
        assert_eq!(route.rule_name, "default");
        assert_eq!(route.provider, "local");
        assert!(route.fallback.is_none());
    }

    #[test]
    fn specific_rule_beats_default_when_listed_first() {
        let engine = engine_with_rules(
            vec![
                RuleSpec {
                    name: "code-local".to_string(),
                    matcher: MatchSpec {
                        content_type: Some(ContentType::Code),
                        ..MatchSpec::default()
                    },
                    provider: "local".to_string(),
                    fallback: None,
                },
                RuleSpec {
                    name: "default".to_string(),
                    matcher: MatchSpec::default(),
                    provider: "cloud".to_string(),
                    fallback: None,
                },
            ],
            &["local", "cloud"],
        );
        let route = engine
            .resolve(&RoutingContext {
                content_type: Some(ContentType::Code),
                ..RoutingContext::default()
            })
            .unwrap();
        assert_eq!(route.provider, "local");
        let route = engine
            .resolve(&RoutingContext {
                content_type: Some(ContentType::Prose),
                ..RoutingContext::default()
            })
            .unwrap();
        assert_eq!(route.provider, "cloud");
    }

    #[test]
    fn pii_rule_keeps_personal_data_local() {
        let engine = engine_with_rules(
            vec![
                RuleSpec {
                    name: "pii-local".to_string(),
                    matcher: MatchSpec {
                        contains_personal_data: Some(true),
                        ..MatchSpec::default()
                    },
                    provider: "local".to_string(),
                    fallback: None,
                },
                RuleSpec {
                    name: "default".to_string(),
                    matcher: MatchSpec::default(),
                    provider: "cloud".to_string(),
                    fallback: None,
                },
            ],
            &["local", "cloud"],
        );
        let route = engine
            .resolve(&RoutingContext {
                contains_personal_data: true,
                ..RoutingContext::default()
            })
            .unwrap();
        assert_eq!(route.provider, "local");
        let route = engine.resolve(&RoutingContext::default()).unwrap();
        assert_eq!(route.provider, "cloud");
    }

    #[test]
    fn and_semantics_within_a_rule() {
        let engine = engine_with_rules(
            vec![RuleSpec {
                name: "code-and-large".to_string(),
                matcher: MatchSpec {
                    content_type: Some(ContentType::Code),
                    query_size_tokens_min: Some(1000),
                    ..MatchSpec::default()
                },
                provider: "local".to_string(),
                fallback: None,
            }],
            &["local"],
        );

        // Both conditions met → match
        let route = engine
            .resolve(&RoutingContext {
                content_type: Some(ContentType::Code),
                query_size_tokens: 1500,
                ..RoutingContext::default()
            })
            .unwrap();
        assert_eq!(route.provider, "local");

        // Wrong content type → no match
        assert_eq!(
            engine.resolve(&RoutingContext {
                content_type: Some(ContentType::Prose),
                query_size_tokens: 1500,
                ..RoutingContext::default()
            }),
            Err(RouteError::NoMatchingRoute)
        );

        // Right content type, too small → no match
        assert_eq!(
            engine.resolve(&RoutingContext {
                content_type: Some(ContentType::Code),
                query_size_tokens: 100,
                ..RoutingContext::default()
            }),
            Err(RouteError::NoMatchingRoute)
        );
    }

    #[test]
    fn fallback_field_is_propagated() {
        let engine = engine_with_rules(
            vec![RuleSpec {
                name: "default".to_string(),
                matcher: MatchSpec::default(),
                provider: "local".to_string(),
                fallback: Some("cloud".to_string()),
            }],
            &["local", "cloud"],
        );
        let route = engine.resolve(&RoutingContext::default()).unwrap();
        assert_eq!(route.provider, "local");
        assert_eq!(route.fallback.as_deref(), Some("cloud"));
    }

    #[test]
    fn unknown_provider_in_rule_is_rejected_at_build_time() {
        let cfg = RoutingConfig {
            rules: vec![RuleSpec {
                name: "default".to_string(),
                matcher: MatchSpec::default(),
                provider: "missing".to_string(),
                fallback: None,
            }],
            providers: HashMap::new(),
        };
        let err = RoutingEngine::new(cfg).expect_err("must reject");
        assert!(matches!(
            err,
            RouteError::UnknownProvider { ref provider, .. } if provider == "missing"
        ));
    }

    #[test]
    fn unknown_fallback_provider_is_also_rejected() {
        let cfg = RoutingConfig {
            rules: vec![RuleSpec {
                name: "default".to_string(),
                matcher: MatchSpec::default(),
                provider: "local".to_string(),
                fallback: Some("missing".to_string()),
            }],
            providers: HashMap::from([provider("local")]),
        };
        let err = RoutingEngine::new(cfg).expect_err("must reject");
        assert!(matches!(
            err,
            RouteError::UnknownProvider { ref provider, .. } if provider == "missing"
        ));
    }

    #[test]
    fn caller_app_id_matches_exactly() {
        let engine = engine_with_rules(
            vec![
                RuleSpec {
                    name: "settings-local".to_string(),
                    matcher: MatchSpec {
                        caller_app_id: Some("lunaris-app-settings".to_string()),
                        ..MatchSpec::default()
                    },
                    provider: "local".to_string(),
                    fallback: None,
                },
                RuleSpec {
                    name: "default".to_string(),
                    matcher: MatchSpec::default(),
                    provider: "cloud".to_string(),
                    fallback: None,
                },
            ],
            &["local", "cloud"],
        );
        let route = engine
            .resolve(&RoutingContext {
                caller_app_id: Some("lunaris-app-settings".to_string()),
                ..RoutingContext::default()
            })
            .unwrap();
        assert_eq!(route.provider, "local");
        let route = engine
            .resolve(&RoutingContext {
                caller_app_id: Some("com.example.other".to_string()),
                ..RoutingContext::default()
            })
            .unwrap();
        assert_eq!(route.provider, "cloud");
    }

    #[test]
    fn full_toml_round_trip() {
        let toml_str = r#"
[[rule]]
name = "code-local"
match.content_type = "code"
provider = "local"

[[rule]]
name = "pii-local"
match.contains_personal_data = true
provider = "local"

[[rule]]
name = "default"
provider = "local"
fallback = "cloud"

[providers.local]
backend = "ollama"
endpoint = "http://localhost:11434"
model = "llama3:8b"

[providers.cloud]
backend = "anthropic"
model = "claude-sonnet-4-7"
api_key_keyring = "anthropic-api-key"
"#;
        let engine = RoutingEngine::from_toml(toml_str).expect("parses");
        assert_eq!(engine.rule_count(), 3);
        let route = engine
            .resolve(&RoutingContext {
                content_type: Some(ContentType::Code),
                ..RoutingContext::default()
            })
            .unwrap();
        assert_eq!(route.rule_name, "code-local");
        let route = engine.resolve(&RoutingContext::default()).unwrap();
        assert_eq!(route.rule_name, "default");
        assert_eq!(route.fallback.as_deref(), Some("cloud"));
    }
}
