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

use crate::service::AiDaemonService;

/// Settings parsed from `ai.toml`.
#[derive(Debug, Clone)]
pub struct AiSettings {
    /// Whether the AI layer accepts queries.
    pub enabled: bool,
    /// Catalogued provider name the daemon dispatches through.
    pub provider: String,
}

impl Default for AiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: "ollama-default".to_string(),
        }
    }
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
        },
        Err(err) => {
            tracing::warn!(error = %err, "ai.toml unreadable, defaulting to disabled");
            AiSettings::default()
        }
    }
}

/// Spawn the `ai.toml` watch thread.
///
/// Runs on a dedicated OS thread because [`os_sdk::config::ConfigWatcher`]
/// exposes a blocking `recv()`. On every change it reloads the file
/// and applies `enabled` to the service. The thread exits when the
/// watcher is dropped (process shutdown).
pub fn spawn_config_watch(service: Arc<AiDaemonService>) {
    std::thread::Builder::new()
        .name("ai-config-watch".to_string())
        .spawn(move || {
            let mut cfg = match Config::load("ai") {
                Ok(c) => c,
                Err(err) => {
                    tracing::warn!(error = %err, "ai.toml watch: load failed");
                    return;
                }
            };
            let watcher = match cfg.watch() {
                Ok(w) => w,
                Err(err) => {
                    tracing::warn!(error = %err, "ai.toml watch: cannot watch");
                    return;
                }
            };
            tracing::info!("ai.toml watch active");
            while watcher.recv().is_ok() {
                if let Err(err) = cfg.reload() {
                    tracing::warn!(error = %err, "ai.toml reload failed");
                    continue;
                }
                let enabled = cfg.get::<bool>("ai.enabled").unwrap_or(false);
                service.set_enabled(enabled);
                tracing::info!(enabled, "ai.toml changed, applied enabled state");
            }
            tracing::info!("ai.toml watch stopped");
        })
        .expect("spawn ai-config-watch thread");
}
