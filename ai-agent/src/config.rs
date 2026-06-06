//! The agent daemon's runtime configuration, read from `ai.toml`.
//!
//! Deliberately minimal and **fail-safe**: anything missing or malformed
//! yields the safe defaults (nothing enabled, no graph read, suggest-only),
//! so a broken config never leaves the agent enabled or over-granted.

use std::collections::BTreeMap;

use lunaris_ai_core::capability::{access_tier_from_level, AccessTier, ActionPermissions, BaselineMode};
use serde::Deserialize;

use crate::loader::Provenance;

/// The LLM provider the agent loop drives, resolved from `ai.toml`. A
/// `kind: agent` behaviour cannot run without one, so `None` keeps agent
/// behaviours skipped (the same fail-closed posture as a disabled daemon).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderSettings {
    /// Catalogued provider name the proxy forwards to (the shared
    /// `ai.provider` key).
    pub name: String,
    /// Model identifier (`[provider] model`).
    pub model: String,
    /// The model's usable input context window, in tokens
    /// (`[provider] context_window`). Defaults to a conservative low value
    /// when omitted, so an under-specified provider compacts early and fails
    /// closed rather than overflowing.
    pub context_window: u32,
    /// Capability token presented to the proxy (`[provider] audit_token`).
    /// Defaults to a fixed agent token; the proxy only records it until
    /// Phase 9-γ S15 validates it against the caller identity.
    pub audit_token: String,
}

/// The agent's resolved runtime configuration.
pub struct AgentConfig {
    /// Behaviour name to the provenance it was approved for (the loader
    /// only enables a behaviour matching this). Built-in only for now.
    pub enabled: BTreeMap<String, Provenance>,
    /// The global Knowledge-Graph read tier.
    pub read_tier: AccessTier,
    /// The per-application action permissions (baseline + autonomous apps).
    pub actions: ActionPermissions,
    /// The LLM provider for `kind: agent` behaviours, if one is configured
    /// and AI is enabled. `None` means agent behaviours cannot run.
    pub provider: Option<ProviderSettings>,
    /// Whether the agent may **execute** proven workflow decisions (write to
    /// the Knowledge Graph), not just surface them. Default `false`:
    /// suggest-mode, where a decision is gated, audited, and reported but never
    /// acted on. Opt in with `[agent] executor_live = true`; the write still
    /// passes the full predict -> gate -> re-validate -> audit chain.
    ///
    /// Status of the deployment prerequisites (it defaults off and nothing flips
    /// it yet): (1) the execution semantics are decided, (2) the cancellation
    /// behaviour is bounded and accepted, and only (3) full proof atomicity
    /// remains as a hard blocker. Detail:
    ///
    /// 1. **Execution semantics (decided).** The executor fires on a proven
    ///    `PreviewThenExecute`, which in the capability model is the *Supervised*
    ///    lift ("preview with a cancellation window, then execute"). For a safe,
    ///    reversible, invisible curation action via a *deterministic workflow*
    ///    (auto-tag's `FILE_PART_OF`), it executes **silently and immediately**,
    ///    with no per-action prompt: per-file confirmation is annoying, and these
    ///    workflows make no LLM call so they cost no tokens. The user inspects
    ///    what was curated after the fact via the read-only activity view (the
    ///    `silent curator + pull` interaction model), not a pre-action window.
    ///    This deliberately overrides the literal Supervised window for safe
    ///    workflow curation; it does NOT extend to `kind: agent` LLM behaviours
    ///    (which are not wired to execute) or to high-impact / external-triggered
    ///    actions (which always confirm regardless).
    /// 2. **Cancellation (bounded, accepted).** The dispatch loop stays
    ///    cancellable (a reload/shutdown can drop an in-flight dispatch), kept on
    ///    purpose: it aborts a long `kind: agent` loop promptly, and for the
    ///    workflow write a *drop is the correct revocation behaviour* (a config
    ///    change removing the grant means the write should not be forced through;
    ///    a dropped write is not re-authorised on the next run). The write is
    ///    pre-audited and idempotent, so if its request was already sent it is
    ///    durably recorded and reconcilable, never lost. The write also has its
    ///    own timeout, so a stalled knowledge socket cannot park the dispatch
    ///    (and the daemon) waiting on it. Residual: an already-sent write can
    ///    still commit under a just-revoked grant (the bounded D-2 class, at most
    ///    the one in-flight event). A narrower per-write completion shield is
    ///    possible but not clearly more correct, since forcing the write through
    ///    a revocation is the opposite of what a revocation wants.
    /// 3. **Proof atomicity.** The executor re-validates the full proof, then
    ///    performs a separate write; the daemon enforces only endpoint existence
    ///    and edge absence atomically, not the gate's `PathUnderField`. A fact
    ///    outside the write predicate can change in between (gap A2, needs a
    ///    graph snapshot/version).
    pub executor_live: bool,
}

#[derive(Deserialize, Default)]
struct RawAi {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    access_level: u8,
    #[serde(default)]
    action_mode: Option<String>,
    #[serde(default)]
    autonomous_apps: Vec<String>,
    /// The catalogued provider name, shared with the rest of the product
    /// (`ai.provider`, written by Settings, read by `ai-daemon`).
    #[serde(default)]
    provider: Option<String>,
}

#[derive(Deserialize, Default)]
struct RawAgent {
    #[serde(default)]
    enabled: Vec<String>,
    #[serde(default)]
    executor_live: bool,
}

#[derive(Deserialize, Default)]
struct RawProvider {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    context_window: Option<u32>,
    #[serde(default)]
    audit_token: Option<String>,
}

#[derive(Deserialize, Default)]
struct RawConfig {
    #[serde(default)]
    ai: RawAi,
    #[serde(default)]
    agent: RawAgent,
    #[serde(default)]
    provider: RawProvider,
}

/// Default model when `[provider] model` is omitted. Matches the catalogued
/// `ollama-default` backend (and `ai-daemon`'s hardcoded model), so a config
/// that only names the provider works out of the box.
const DEFAULT_MODEL: &str = "llama3:8b";
/// Conservative fallback window when `[provider] context_window` is omitted:
/// low enough that an under-specified provider compacts early and never
/// overflows. A deployment sets the model's real window.
const DEFAULT_CONTEXT_WINDOW: u32 = 8_192;
/// Fixed token presented to the proxy when `[provider] audit_token` is omitted.
const DEFAULT_AUDIT_TOKEN: &str = "ai-agent-default-token";

impl AgentConfig {
    /// The safe default: disabled, no graph read, suggest-only actions, no
    /// provider (so agent behaviours cannot run).
    pub fn fail_closed() -> Self {
        Self {
            enabled: BTreeMap::new(),
            read_tier: AccessTier::Minimal,
            actions: ActionPermissions::suggest_only(),
            provider: None,
            executor_live: false,
        }
    }

    /// Parse from `ai.toml` text. A malformed document falls back to the
    /// safe defaults rather than erroring (fail-closed). The read level is
    /// clamped by `access_tier_from_level`, and `action_mode` can never be
    /// autonomous (a [`BaselineMode`]); autonomy is per-app only.
    pub fn parse(toml_text: &str) -> Self {
        let raw: RawConfig = toml::from_str(toml_text).unwrap_or_default();
        // `[ai] enabled` (default off) is the global AI master switch, the
        // same flag the ai-daemon gates on. With AI disabled the agent runs
        // nothing, whatever the per-behaviour `[agent] enabled` list says.
        if !raw.ai.enabled {
            return Self::fail_closed();
        }
        // Only built-in behaviours exist for now, so an enabled name is
        // approved for the built-in provenance.
        let enabled = raw
            .agent
            .enabled
            .into_iter()
            .map(|name| (name, Provenance::BuiltIn))
            .collect();
        let baseline = raw
            .ai
            .action_mode
            .as_deref()
            .map(BaselineMode::parse)
            .unwrap_or(BaselineMode::Suggest);
        // A provider is wired only when one is named via the shared
        // `ai.provider` key (so the standard Settings-authored config wires
        // it, and a bare `[ai] enabled` without a provider stays workflow-only
        // rather than guessing a backend). The model, window, and token fall
        // back to safe defaults matching the catalogued backend when an
        // optional `[provider]` section does not override them.
        let provider = raw
            .ai
            .provider
            .filter(|name| !name.is_empty())
            .map(|name| ProviderSettings {
                name,
                model: raw.provider.model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
                context_window: raw.provider.context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
                audit_token: raw
                    .provider
                    .audit_token
                    .unwrap_or_else(|| DEFAULT_AUDIT_TOKEN.to_string()),
            });
        Self {
            enabled,
            read_tier: access_tier_from_level(raw.ai.access_level),
            actions: ActionPermissions::new(baseline, raw.ai.autonomous_apps),
            provider,
            executor_live: raw.agent.executor_live,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_enabled_read_tier_and_actions() {
        let cfg = AgentConfig::parse(
            r#"
[ai]
enabled = true
access_level = 2
action_mode = "supervised"
autonomous_apps = ["org.lunaris.files"]

[agent]
enabled = ["auto-tag-by-project"]
"#,
        );
        assert_eq!(cfg.enabled.get("auto-tag-by-project"), Some(&Provenance::BuiltIn));
        assert_eq!(cfg.read_tier, AccessTier::ProjectScoped);
        assert!(cfg.actions.is_autonomous("org.lunaris.files"));
        // The executor opt-in defaults off (suggest-mode) when unspecified.
        assert!(!cfg.executor_live);
    }

    #[test]
    fn executor_live_is_opt_in() {
        let cfg = AgentConfig::parse(
            "[ai]\nenabled = true\n[agent]\nexecutor_live = true\n",
        );
        assert!(cfg.executor_live, "[agent] executor_live = true opts into executing");
        // Fail-closed config never executes.
        assert!(!AgentConfig::fail_closed().executor_live);
    }

    #[test]
    fn malformed_or_empty_config_fails_closed() {
        for text in ["", "not valid toml = =", "[ai]\naccess_level = 99"] {
            let cfg = AgentConfig::parse(text);
            assert!(cfg.enabled.is_empty());
            // 99 is out of range -> clamped to Minimal; empty -> Minimal.
            assert_eq!(cfg.read_tier, AccessTier::Minimal);
        }
    }

    #[test]
    fn global_ai_disable_overrides_enabled_behaviours() {
        // AI off globally must run nothing, even with behaviours listed and a
        // read level requested, so the master switch genuinely stops the agent.
        for text in [
            "[agent]\nenabled = [\"auto-tag-by-project\"]\n",
            "[ai]\nenabled = false\naccess_level = 4\n\n[agent]\nenabled = [\"auto-tag-by-project\"]\n",
        ] {
            let cfg = AgentConfig::parse(text);
            assert!(cfg.enabled.is_empty(), "AI off must enable no behaviours");
            assert_eq!(cfg.read_tier, AccessTier::Minimal);
            assert!(!cfg.actions.is_autonomous("org.lunaris.files"));
        }
    }

    #[test]
    fn action_mode_can_never_be_autonomous_globally() {
        let cfg = AgentConfig::parse("[ai]\nenabled = true\naction_mode = \"autonomous\"\n");
        // A global autonomous request collapses to the safe baseline.
        assert!(!cfg.actions.is_autonomous("any.app"));
    }

    #[test]
    fn the_standard_settings_config_wires_a_provider() {
        // The shape the Settings UI writes: [ai] enabled + provider. The agent
        // must wire a provider from it, with safe default model/window/token.
        let cfg = AgentConfig::parse("[ai]\nenabled = true\nprovider = \"ollama-default\"\n");
        let p = cfg.provider.expect("ai.provider wires a provider");
        assert_eq!(p.name, "ollama-default");
        assert_eq!(p.model, DEFAULT_MODEL);
        assert_eq!(p.context_window, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(p.audit_token, DEFAULT_AUDIT_TOKEN);
    }

    #[test]
    fn a_provider_section_overrides_the_model_window_and_token() {
        let cfg = AgentConfig::parse(
            r#"
[ai]
enabled = true
provider = "my-cloud"

[provider]
model = "claude-opus-4-8"
context_window = 200000
audit_token = "tok-123"
"#,
        );
        let p = cfg.provider.expect("a provider is configured");
        assert_eq!(p.name, "my-cloud");
        assert_eq!(p.model, "claude-opus-4-8");
        assert_eq!(p.context_window, 200000);
        assert_eq!(p.audit_token, "tok-123");
    }

    #[test]
    fn no_provider_without_a_named_provider() {
        // A bare enabled config (no ai.provider), and an empty name, stay
        // workflow-only rather than guessing a backend.
        for text in [
            "[ai]\nenabled = true\n",
            "[ai]\nenabled = true\nprovider = \"\"\n",
        ] {
            assert!(AgentConfig::parse(text).provider.is_none(), "config: {text:?}");
        }
    }

    #[test]
    fn ai_disabled_yields_no_provider_even_when_named() {
        // The master switch off must leave nothing runnable, including a named
        // provider.
        let cfg = AgentConfig::parse("[ai]\nenabled = false\nprovider = \"ollama-default\"\n");
        assert!(cfg.provider.is_none());
    }
}
