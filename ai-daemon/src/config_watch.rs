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
//! * `provider` — read once at startup. A provider change needs a
//!   daemon restart, the same convention `graph.toml` uses (Settings
//!   surfaces that hint). Live provider switching waits for the
//!   multi-provider routing of Phase 9-β/γ; with a single catalogued
//!   provider in Phase 9-α there is nothing to switch between.

use std::sync::Arc;

use os_sdk::config::Config;

use lunaris_ai_core::capability::access_tier_from_level;
use lunaris_ai_core::graph_query::QueryScope;
use lunaris_ai_core::graph_schema::GraphSchema;

use crate::service::AiDaemonService;

/// Settings parsed from `ai.toml`.
#[derive(Debug, Clone)]
pub struct AiSettings {
    /// Whether the AI layer accepts queries.
    pub enabled: bool,
    /// Catalogued provider name the daemon dispatches through.
    pub provider: String,
    /// Global read access level 0..=4 (Foundation §8.4 table). Decides
    /// how much of the graph the AI can see; mapped to an
    /// `AccessTier` by `lunaris_ai_core::capability::access_tier_from_level`.
    pub access_level: u8,
}

impl Default for AiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: "ollama-default".to_string(),
            // Fail closed: no graph access until the user raises it.
            access_level: 0,
        }
    }
}

/// Read `ai.access_level` as a clamped `u8`. The config loader only
/// decodes TOML integers as `i64`, so this narrows it: a missing,
/// negative, or out-of-byte-range value yields 0 (Minimal), and
/// `access_tier_from_level` clamps anything above 4 back to Minimal as
/// well. A malformed level therefore never widens access.
fn read_access_level(cfg: &Config) -> u8 {
    u8::try_from(cfg.get::<i64>("ai.access_level").unwrap_or(0)).unwrap_or(0)
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
            provider: cfg
                .get::<String>("ai.provider")
                .unwrap_or_else(|| "ollama-default".to_string()),
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
