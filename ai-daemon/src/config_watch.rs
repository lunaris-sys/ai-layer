//! `ai.toml` config loading + live watch.
//!
//! The AI layer is opt-in (Foundation §5.1-5.2): the daemon starts
//! fail-closed and only begins serving queries once Settings writes
//! `enabled = true` into `~/.config/lunaris/ai.toml`. This module is
//! the watcher that makes that toggle live (Phase 9-α S7).
//!
//! Scope of the live reload:
//!
//! * `enabled` — applied live. Toggling it in Settings switches the
//!   AI layer on/off without a daemon restart.
//! * `provider`: read once at startup. The provider name (`ai.provider`)
//!   and the optional `[provider]` section (model, context window, audit
//!   token) are applied at startup only; a provider change needs a daemon
//!   restart, the same convention `graph.toml` uses (Settings surfaces that
//!   hint). Live provider switching waits for multi-provider routing; with a
//!   single catalogued provider there is nothing to switch between.

use std::sync::Arc;

use os_sdk::config::Config;

use lunaris_ai_core::capability::access_tier_from_level;
use lunaris_ai_core::graph_query::QueryScope;
use lunaris_ai_core::graph_schema::GraphSchema;

use crate::service::AiDaemonService;

/// The catalogued LLM provider the daemon forwards completions through,
/// resolved from `ai.toml`: the name from the shared `ai.provider` key (also
/// read by ai-agent, written by Settings), and the model, context window, and
/// audit token from an optional `[provider]` section. Mirrors ai-agent's
/// `ProviderSettings` (`ai-agent/src/config.rs`) so both halves of the AI layer
/// read one provider config from one file; a shared `ai-core` type is a clean
/// follow-up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderSettings {
    /// Catalogued provider name the proxy forwards to (`ai.provider`).
    pub name: String,
    /// Model identifier (`[provider] model`).
    pub model: String,
    /// The model's usable input context window, in tokens
    /// (`[provider] context_window`).
    pub context_window: u32,
    /// Capability token presented to the proxy (`[provider] audit_token`).
    pub audit_token: String,
}

impl Default for ProviderSettings {
    fn default() -> Self {
        Self {
            name: DEFAULT_PROVIDER.to_string(),
            model: DEFAULT_MODEL.to_string(),
            context_window: DEFAULT_CONTEXT_WINDOW,
            audit_token: DEFAULT_AUDIT_TOKEN.to_string(),
        }
    }
}

/// Settings parsed from `ai.toml`. The [`Default`] is the fail-closed posture
/// used when the file cannot be read: disabled, the default local provider
/// (unused while disabled), and Minimal access (level 0, no graph reads until
/// the user raises it).
#[derive(Debug, Clone, Default)]
pub struct AiSettings {
    /// Whether the AI layer accepts queries.
    pub enabled: bool,
    /// The provider the daemon forwards completions through.
    pub provider: ProviderSettings,
    /// Global read access level 0..=4 (Foundation §8.4 table). Decides
    /// how much of the graph the AI can see; mapped to an
    /// `AccessTier` by `lunaris_ai_core::capability::access_tier_from_level`.
    pub access_level: u8,
}

/// Catalogued provider when `ai.provider` is absent: the local Ollama backend.
/// Used only for a missing key (the unconfigured query daemon needs a backend),
/// not for a present-but-invalid value, which fails closed.
const DEFAULT_PROVIDER: &str = "ollama-default";
/// Model when `[provider] model` is omitted. Matches the `ollama-default`
/// backend and ai-agent's default, so a named provider works out of the box.
const DEFAULT_MODEL: &str = "llama3:8b";
/// Conservative input context window when `[provider] context_window` is
/// omitted: llama3:8b ships 8192. A real deployment sets its model's window.
const DEFAULT_CONTEXT_WINDOW: u32 = 8_192;
/// Token presented to the proxy when `[provider] audit_token` is omitted. The
/// proxy only records it until S15 validates it against the caller identity.
const DEFAULT_AUDIT_TOKEN: &str = "ai-daemon-default-token";

/// Read `ai.access_level` as a clamped `u8`. The config loader only
/// decodes TOML integers as `i64`, so this narrows it: a missing,
/// negative, or out-of-byte-range value yields 0 (Minimal), and
/// `access_tier_from_level` clamps anything above 4 back to Minimal as
/// well. A malformed level therefore never widens access.
fn read_access_level(cfg: &Config) -> u8 {
    u8::try_from(cfg.get::<i64>("ai.access_level").unwrap_or(0)).unwrap_or(0)
}

/// Resolve the provider config from `ai.toml`: the name from the shared
/// `ai.provider` key, the model, context window, and audit token from an
/// optional `[provider]` section.
///
/// The provider name decides whether and where LLM traffic leaves, so an
/// *absent* key (an unconfigured daemon) defaults to the local backend, the
/// value Settings writes, while a *present but invalid* value (blank or
/// wrong-typed) fails closed: it yields an empty name the proxy rejects, so a
/// cleared or corrupted field never silently re-enables forwarding. Absent is
/// distinguished from invalid deliberately, because the provider is read once
/// at startup (a change needs a restart): defaulting an absent key keeps a
/// daemon that started before `ai.toml` existed from latching an empty provider
/// for its lifetime, whereas an explicitly invalid value is a misconfiguration
/// that should not be papered over with a backend. The within-provider fields
/// (model, window, token) default on a blank value, since they select within
/// an already-named backend.
fn read_provider(cfg: &Config) -> ProviderSettings {
    let non_empty = |key: &str| cfg.get::<String>(key).filter(|s| !s.is_empty());
    let name = match cfg.get_raw("ai.provider") {
        // Absent: default to the local backend (an unconfigured query daemon
        // needs one, and this matches the value Settings writes).
        None => DEFAULT_PROVIDER.to_string(),
        // Present: only a non-empty string forwards; a blank or wrong-typed
        // value fails closed rather than guessing a backend.
        Some(_) => non_empty("ai.provider").unwrap_or_else(|| {
            tracing::warn!(
                "ai.provider is set but blank or not a string; the daemon will not \
                 forward to any backend (queries fail at the provider step)"
            );
            String::new()
        }),
    };
    // The loader decodes TOML integers as i64; narrow to a positive u32,
    // falling back to the default on a missing, non-positive, or oversized
    // value so a malformed window never reaches the provider.
    let context_window = cfg
        .get::<i64>("provider.context_window")
        .and_then(|v| u32::try_from(v).ok())
        .filter(|&w| w > 0)
        .unwrap_or(DEFAULT_CONTEXT_WINDOW);
    ProviderSettings {
        name,
        model: non_empty("provider.model").unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        context_window,
        audit_token: non_empty("provider.audit_token")
            .unwrap_or_else(|| DEFAULT_AUDIT_TOKEN.to_string()),
    }
}

/// Drive the service into the fail-closed security state: no queries
/// accepted, no graph access. Applied on every path where `ai.toml`
/// cannot be trusted — it could not be loaded, watched, or reparsed —
/// so a malformed, truncated, or unreadable file never leaves a broad
/// scope or an enabled daemon in place, regardless of what an earlier
/// valid load had applied.
fn fail_closed(service: &AiDaemonService) {
    // Publish disabled + Minimal as one atomic admission state, so a
    // concurrent query cannot catch a half-applied transition.
    service.set_admission(
        false,
        QueryScope::for_tier(access_tier_from_level(0), &GraphSchema::knowledge_graph()),
    );
}

/// Read `ai.toml` once. A missing or unreadable file yields the
/// fail-closed default (`enabled = false`).
pub fn load_ai_settings() -> AiSettings {
    match Config::load("ai") {
        Ok(cfg) => AiSettings {
            enabled: cfg.get::<bool>("ai.enabled").unwrap_or(false),
            provider: read_provider(&cfg),
            access_level: read_access_level(&cfg),
        },
        Err(err) => {
            tracing::warn!(error = %err, "ai.toml unreadable, defaulting to disabled");
            AiSettings::default()
        }
    }
}

/// Apply a freshly-loaded `ai.toml` to the service as one atomic
/// admission state.
///
/// A disabled daemon is given an effective Minimal scope, so the
/// configured tier is never installed while the AI layer is off; either
/// way `enabled` and the scope are published together via
/// [`AiDaemonService::set_admission`], so the query path samples a
/// consistent pair and never a torn one.
fn apply_config(service: &AiDaemonService, cfg: &Config) {
    let enabled = cfg.get::<bool>("ai.enabled").unwrap_or(false);
    let level = read_access_level(cfg);
    let effective_level = if enabled { level } else { 0 };
    let scope = QueryScope::for_tier(
        access_tier_from_level(effective_level),
        &GraphSchema::knowledge_graph(),
    );
    service.set_admission(enabled, scope);
    tracing::info!(enabled, access_level = level, "ai.toml applied");
}

/// Spawn the `ai.toml` watch thread.
///
/// The watcher is the sole owner of the service's admission state: the
/// daemon is constructed fail-closed (disabled, no graph access) and
/// this thread publishes the configured admission only after the file
/// watch is armed, then keeps it live on every change. Because the
/// pre-publish state is fail-closed, there is no window where a stale
/// startup snapshot serves broad access before the watcher is live, and
/// a write that lands before the watch is armed is picked up by the
/// initial publish rather than missed.
///
/// Runs on a dedicated OS thread because [`os_sdk::config::ConfigWatcher`]
/// exposes a blocking `recv()`. The thread exits when the watcher is
/// dropped (process shutdown).
pub fn spawn_config_watch(service: Arc<AiDaemonService>) {
    std::thread::Builder::new()
        .name("ai-config-watch".to_string())
        .spawn(move || {
            // On any load/watch setup failure the service stays in its
            // fail-closed startup state; recovery needs a daemon restart.
            // The live reload path below recovers on its own once the
            // watcher is running.
            let mut cfg = match Config::load("ai") {
                Ok(c) => c,
                Err(err) => {
                    tracing::warn!(error = %err, "ai.toml watch: load failed, failing closed");
                    fail_closed(&service);
                    return;
                }
            };
            let watcher = match cfg.watch() {
                Ok(w) => w,
                Err(err) => {
                    tracing::warn!(error = %err, "ai.toml watch: cannot watch, failing closed");
                    fail_closed(&service);
                    return;
                }
            };
            // Initial publish, now that the watch is armed. Reload first
            // so it reflects the on-disk file as of after registration,
            // closing the gap between the daemon's fail-closed startup
            // state and the first change event (a write before the watch
            // armed fires no event).
            if let Err(err) = cfg.reload() {
                tracing::warn!(error = %err, "ai.toml initial reload failed, failing closed");
                fail_closed(&service);
            } else {
                apply_config(&service, &cfg);
            }
            tracing::info!("ai.toml watch active");
            while watcher.recv().is_ok() {
                if let Err(err) = cfg.reload() {
                    // A malformed or partially-written ai.toml must not
                    // leave a previously broad admission in place: we
                    // cannot trust the prior in-memory values. Fail closed
                    // and keep the watcher alive so a later valid rewrite
                    // recovers.
                    tracing::warn!(error = %err, "ai.toml reload failed, failing closed");
                    fail_closed(&service);
                    continue;
                }
                apply_config(&service, &cfg);
            }
            tracing::info!("ai.toml watch stopped");
        })
        .expect("spawn ai-config-watch thread");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Parse a provider config from an `ai.toml` body, exercising the same
    /// dot-notation reads `load_ai_settings` uses (via a temp file, since the
    /// loader reads from disk).
    fn provider_from(toml: &str) -> ProviderSettings {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{toml}").unwrap();
        let cfg = Config::load_path(file.path()).unwrap();
        read_provider(&cfg)
    }

    #[test]
    fn a_name_only_config_uses_the_default_model_window_and_token() {
        // The shape Settings writes: [ai] enabled + provider, no [provider].
        let p = provider_from("[ai]\nenabled = true\nprovider = \"ollama-default\"\n");
        assert_eq!(p.name, "ollama-default");
        assert_eq!(p.model, DEFAULT_MODEL);
        assert_eq!(p.context_window, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(p.audit_token, DEFAULT_AUDIT_TOKEN);
    }

    #[test]
    fn a_provider_section_overrides_the_model_window_and_token() {
        let p = provider_from(
            "[ai]\nenabled = true\nprovider = \"my-cloud\"\n\n\
             [provider]\nmodel = \"claude-opus-4-8\"\ncontext_window = 200000\naudit_token = \"tok-123\"\n",
        );
        assert_eq!(p.name, "my-cloud");
        assert_eq!(p.model, "claude-opus-4-8");
        assert_eq!(p.context_window, 200_000);
        assert_eq!(p.audit_token, "tok-123");
    }

    #[test]
    fn an_absent_provider_falls_back_to_the_default_local_backend() {
        // An unconfigured query daemon needs a backend, so an *absent*
        // ai.provider defaults to the local backend (the value Settings
        // writes). This keeps a daemon that started before the config existed
        // from latching an empty provider, since the provider is read once at
        // startup.
        for toml in ["", "[ai]\nenabled = true\n"] {
            assert_eq!(provider_from(toml).name, DEFAULT_PROVIDER, "toml: {toml:?}");
        }
    }

    #[test]
    fn a_present_but_blank_provider_fails_closed_instead_of_defaulting() {
        // A *present* but blank provider is a misconfiguration: it fails closed
        // (the proxy rejects an empty name), rather than silently routing to a
        // backend. Defaulting it would let a cleared Settings field re-enable
        // forwarding. This is the case absent-key handling deliberately differs
        // from.
        let p = provider_from("[ai]\nenabled = true\nprovider = \"\"\n");
        assert_eq!(p.name, "");
    }

    #[test]
    fn a_present_wrong_typed_provider_fails_closed_and_fields_default() {
        // A present but wrong-typed provider name fails closed (no forwarding);
        // wrong-typed within-provider fields fall back to safe defaults rather
        // than reaching the backend malformed.
        let p = provider_from(
            "[ai]\nprovider = 123\n\n[provider]\nmodel = 5\ncontext_window = \"big\"\naudit_token = true\n",
        );
        assert_eq!(p.name, "");
        assert_eq!(p.model, DEFAULT_MODEL);
        assert_eq!(p.context_window, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(p.audit_token, DEFAULT_AUDIT_TOKEN);
    }

    #[test]
    fn a_non_positive_or_oversized_context_window_falls_back_to_the_default() {
        for toml in [
            "[provider]\ncontext_window = 0\n",
            "[provider]\ncontext_window = -5\n",
            "[provider]\ncontext_window = 9999999999\n",
        ] {
            assert_eq!(
                provider_from(toml).context_window,
                DEFAULT_CONTEXT_WINDOW,
                "toml: {toml:?}"
            );
        }
    }

    #[test]
    fn a_blank_model_or_token_does_not_blank_what_reaches_the_proxy() {
        // An empty string in the file is treated as absent, not a literal
        // empty value that would reach the proxy.
        let p = provider_from("[provider]\nmodel = \"\"\naudit_token = \"\"\n");
        assert_eq!(p.model, DEFAULT_MODEL);
        assert_eq!(p.audit_token, DEFAULT_AUDIT_TOKEN);
    }
}
