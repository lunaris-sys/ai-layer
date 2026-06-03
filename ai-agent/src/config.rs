//! The agent daemon's runtime configuration, read from `ai.toml`.
//!
//! Deliberately minimal and **fail-safe**: anything missing or malformed
//! yields the safe defaults (nothing enabled, no graph read, suggest-only),
//! so a broken config never leaves the agent enabled or over-granted.

use std::collections::BTreeMap;

use lunaris_ai_core::capability::{access_tier_from_level, AccessTier, ActionPermissions, BaselineMode};
use serde::Deserialize;

use crate::loader::Provenance;

/// The agent's resolved runtime configuration.
pub struct AgentConfig {
    /// Behaviour name to the provenance it was approved for (the loader
    /// only enables a behaviour matching this). Built-in only for now.
    pub enabled: BTreeMap<String, Provenance>,
    /// The global Knowledge-Graph read tier.
    pub read_tier: AccessTier,
    /// The per-application action permissions (baseline + autonomous apps).
    pub actions: ActionPermissions,
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
}

#[derive(Deserialize, Default)]
struct RawAgent {
    #[serde(default)]
    enabled: Vec<String>,
}

#[derive(Deserialize, Default)]
struct RawConfig {
    #[serde(default)]
    ai: RawAi,
    #[serde(default)]
    agent: RawAgent,
}

impl AgentConfig {
    /// The safe default: disabled, no graph read, suggest-only actions.
    pub fn fail_closed() -> Self {
        Self {
            enabled: BTreeMap::new(),
            read_tier: AccessTier::Minimal,
            actions: ActionPermissions::suggest_only(),
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
        Self {
            enabled,
            read_tier: access_tier_from_level(raw.ai.access_level),
            actions: ActionPermissions::new(baseline, raw.ai.autonomous_apps),
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
}
