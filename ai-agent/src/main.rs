//! `lunaris-ai-agent` daemon entry point.
//!
//! Wires the library into a running daemon. The daemon does nothing, and
//! exits, unless at least one behaviour is enabled (Foundation §5.5).
//!
//! Two lifecycle properties matter for a security-relevant daemon:
//!
//! - **Settings changes apply live.** The daemon watches `ai.toml` and, on
//!   any change, tears the whole pipeline down and rebuilds it from the
//!   fresh config. Disabling a behaviour or lowering the read tier therefore
//!   takes effect without a restart, rather than leaving already-loaded
//!   behaviours running under stale grants. A malformed or removed config
//!   reloads to the safe defaults (nothing enabled) and the daemon exits.
//! - **Boot order is forgiving.** A late or briefly-unavailable Event Bus at
//!   session start does not kill the daemon: the initial subscription retries
//!   with backoff until the bus appears (or a shutdown signal arrives).

use std::path::{Path, PathBuf};
use std::sync::mpsc::TryRecvError;
use std::time::Duration;

use tokio::sync::watch;

use lunaris_ai_agent::behaviour::BehaviourKind;
use lunaris_ai_agent::config::AgentConfig;
use lunaris_ai_agent::engine::{DispatchOutcome, Dispatcher};
use lunaris_ai_agent::gate::Gate;
use lunaris_ai_agent::graph::{UnixGraph, DEFAULT_GRAPH_SOCKET};
use lunaris_ai_agent::handlers::builtin_handlers;
use lunaris_ai_agent::loader::{load, BehaviourSource};
use lunaris_ai_agent::seams::{NullObserver, SystemClock, TriggerSource};
use lunaris_ai_agent::source::{subscription_types, EventBusSource, DEFAULT_CONSUMER_SOCKET};
use lunaris_ai_core::audit::LedgerAuditSink;
use lunaris_ai_core::capability::Capability;
use lunaris_ai_core::provider::AIProvider;
use os_sdk::config::{Config, ConfigWatcher};

/// Backoff bounds for the initial Event Bus subscription retry.
const SUBSCRIBE_BACKOFF_INITIAL: Duration = Duration::from_millis(500);
const SUBSCRIBE_BACKOFF_MAX: Duration = Duration::from_secs(30);

/// How often the config watcher is polled. A settings change is picked up
/// within this interval; it is well below any human-perceptible delay and
/// keeps the watcher in the async scope rather than parking a thread.
const CONFIG_POLL_INTERVAL: Duration = Duration::from_millis(300);

/// Why an epoch ended.
enum EpochEnd {
    /// Config changed (or the watch was lost): rebuild from fresh settings.
    Reload,
    /// A shutdown signal arrived: stop the daemon.
    Shutdown,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // A single OS-signal listener fans a shutdown request out to every part
    // of the loop through a watch channel (idempotent: once true it stays
    // true, so any later wait resolves immediately).
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });

    // Collaborators that never change at runtime, built once.
    let handlers = builtin_handlers();
    let audit = LedgerAuditSink::at_default_socket();
    let observer = NullObserver;
    let graph = UnixGraph::new(graph_socket());
    let ai_path = ai_config_path();

    run(&handlers, &audit, &observer, &graph, &ai_path, shutdown_rx).await
}

/// The epoch loop. Each iteration is one config epoch: load settings and
/// behaviours, run the dispatcher, and rebuild on the next config change.
async fn run(
    handlers: &lunaris_ai_agent::engine::HandlerRegistry,
    audit: &LedgerAuditSink,
    observer: &NullObserver,
    graph: &UnixGraph,
    ai_path: &Path,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        // Arm the config watch *before* reading the config, so no settings
        // change can slip through the gap between resolving the config and
        // registering the watch (S16 startup-gap closure). A change that
        // lands between arming and reading is also safe: the read below sees
        // the latest content, and the queued event triggers one harmless
        // extra reload. A malformed config cannot be watched (and reads to
        // the empty defaults), so the daemon exits per §5.5 just below.
        let watcher = match Config::load_path(ai_path).and_then(|c| c.watch()) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!(error = %e, "ai.toml is not readable or watchable; exiting");
                return Ok(());
            }
        };

        let config = load_config(ai_path);
        let outcome = load(&behaviour_sources(), &config.enabled);
        for err in &outcome.errors {
            tracing::warn!(error = %err, "behaviour failed to load");
        }

        // No LLM provider is wired yet: a `kind: agent` behaviour needs one
        // (routed through ai-proxy) and lands with the first agent behaviour.
        // A `kind: agent` behaviour therefore cannot run, so it is excluded
        // from the runnable set below (and logged), not silently kept alive.
        let provider: Option<&dyn AIProvider> = None;

        // Foundation §5.5: with nothing *runnable* the daemon has no reason to
        // run. A behaviour is runnable when enabled and either a workflow or
        // (for an agent) backed by a configured provider. Exit cleanly
        // otherwise (the supervisor restarts it when a runnable behaviour is
        // enabled); this also covers a removed config.
        let mut runnable = 0usize;
        for b in &outcome.loaded {
            if !b.status.is_enabled() {
                continue;
            }
            if provider.is_none() && b.behaviour.manifest.kind == BehaviourKind::Agent {
                tracing::warn!(
                    behaviour = %b.behaviour.manifest.name,
                    "agent behaviour is enabled but no AI provider is configured; it will not run"
                );
                continue;
            }
            runnable += 1;
        }
        if runnable == 0 {
            tracing::info!("no runnable behaviours; the agent has nothing to do, exiting");
            return Ok(());
        }

        tracing::info!(runnable, "starting agent");

        let read_tier = config.read_tier;
        let capability = Capability::new(read_tier, config.actions);
        let gate = Gate::new(&capability, audit, observer);
        let clock = SystemClock;
        // `read_tier` gates which behaviours may read at all: the dispatcher
        // denies the graph to any behaviour whose declared `reads` exceeds it.
        // It does NOT yet constrain the *content* of an allowed behaviour's
        // queries to the tier (mandatory Cypher anchor injection on the
        // current session / active project / lookback window). That finer,
        // value-level enforcement has to live in the knowledge daemon, which
        // does not yet carry a per-query tier on the wire; it is the same
        // documented S16 follow-up the ai-daemon shares, not an agent-local
        // concern. A process-local scope wrapper here would not bind a
        // compromised handler (it could reach the knowledge socket directly),
        // and B1 behaviours are trusted first-party built-ins, so the coarse
        // gate is the boundary today.
        let dispatcher =
            Dispatcher::new(&outcome.loaded, handlers, graph, read_tier, gate, provider, &clock);

        // Subscribe to exactly the event types the enabled behaviours need.
        let types = subscription_types(&outcome.loaded);
        let mut source =
            match subscribe_with_retry(consumer_socket(), types, &watcher, &mut shutdown_rx).await {
                Ok(s) => s,
                Err(EpochEnd::Shutdown) => return Ok(()),
                Err(EpochEnd::Reload) => continue,
            };

        match dispatch_until_change(&dispatcher, &mut source, &watcher, &mut shutdown_rx).await {
            EpochEnd::Shutdown => {
                tracing::info!("shutdown signal received, stopping");
                return Ok(());
            }
            EpochEnd::Reload => {
                tracing::info!("reloading agent settings");
                // Loop: rebuild the pipeline from the fresh config.
            }
        }
    }
}

/// Dispatch events until the config changes or a shutdown signal arrives.
///
/// `biased` checks the config watcher before pulling the next event, and a
/// revocation that lands between subscribing and acting is honored before the
/// event is dispatched. So a settings change always wins over processing a
/// further event under the old grants (at most the one event already in
/// flight finishes under the previous settings).
async fn dispatch_until_change(
    dispatcher: &Dispatcher<'_>,
    source: &mut EventBusSource,
    watcher: &ConfigWatcher,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> EpochEnd {
    loop {
        let control = tokio::select! {
            biased;
            end = wait_config_change(watcher, shutdown_rx) => Some(end),
            maybe_event = source.recv() => match maybe_event {
                // The SDK consumer reconnects internally, so a closed source
                // means it is permanently gone; rebuild to recover.
                None => {
                    tracing::warn!("event source closed; rebuilding");
                    Some(EpochEnd::Reload)
                }
                Some(event) => match watcher.try_recv() {
                    Ok(()) | Err(TryRecvError::Disconnected) => Some(EpochEnd::Reload),
                    Err(TryRecvError::Empty) => {
                        for outcome in dispatcher.dispatch(&event).await {
                            log_dispatch_outcome(&outcome);
                        }
                        None
                    }
                },
            },
        };
        if let Some(end) = control {
            return end;
        }
    }
}

/// Surface a dispatch outcome. Decisions are also recorded in the audit
/// ledger by the gate; this is the operational view, so a refused, failed,
/// or skipped behaviour is visible rather than silent. (Proposed actions do
/// not yet have a downstream consumer: the suggestion surface and action
/// executor are later phases; suggest-mode decisions are audited regardless.)
fn log_dispatch_outcome(outcome: &DispatchOutcome) {
    match outcome {
        DispatchOutcome::Decided {
            behaviour,
            action,
            decision,
            audit_index,
        } => tracing::info!(
            behaviour = %behaviour,
            summary = %action.summary,
            ?decision,
            audit_index = *audit_index,
            "behaviour decision gated and audited"
        ),
        DispatchOutcome::Refused { behaviour, reason } => {
            tracing::warn!(behaviour = %behaviour, reason = %reason, "behaviour action refused")
        }
        DispatchOutcome::Failed { behaviour, reason } => {
            tracing::warn!(behaviour = %behaviour, reason = %reason, "behaviour handler failed")
        }
        DispatchOutcome::Terminal { behaviour, outcome } => {
            tracing::debug!(behaviour = %behaviour, outcome = %outcome, "behaviour reached a terminal condition")
        }
        DispatchOutcome::Skipped { behaviour, reason } => {
            tracing::debug!(behaviour = %behaviour, reason = %reason, "behaviour skipped")
        }
    }
}

/// Subscribe to the Event Bus, retrying with backoff until it is reachable.
/// A config change during the wait aborts to a rebuild (so the daemon does
/// not subscribe with the old, possibly-revoked settings), and a shutdown
/// signal stops it. Either is returned as `Err(EpochEnd)`.
async fn subscribe_with_retry(
    socket: String,
    types: Vec<String>,
    watcher: &ConfigWatcher,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> Result<EventBusSource, EpochEnd> {
    let mut backoff = SUBSCRIBE_BACKOFF_INITIAL;
    loop {
        tokio::select! {
            biased;
            end = wait_config_change(watcher, shutdown_rx) => return Err(end),
            res = EventBusSource::subscribe(socket.clone(), types.clone()) => match res {
                Ok(source) => return Ok(source),
                Err(e) => tracing::warn!(error = %e, "event bus unavailable, retrying"),
            },
        }
        tokio::select! {
            _ = tokio::time::sleep(backoff) => {}
            end = wait_config_change(watcher, shutdown_rx) => return Err(end),
        }
        backoff = (backoff * 2).min(SUBSCRIBE_BACKOFF_MAX);
    }
}

/// Resolve when `ai.toml` changes (or the watch is lost), or when a shutdown
/// signal arrives. Polls the watcher so it stays in the async scope.
async fn wait_config_change(
    watcher: &ConfigWatcher,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> EpochEnd {
    loop {
        match watcher.try_recv() {
            Ok(()) => return EpochEnd::Reload,
            Err(TryRecvError::Empty) => {}
            // A lost watch means we can no longer observe revocations, so
            // rebuild (which re-establishes the watch, fail-closed).
            Err(TryRecvError::Disconnected) => return EpochEnd::Reload,
        }
        if sleep_or_shutdown(CONFIG_POLL_INTERVAL, shutdown_rx).await {
            return EpochEnd::Shutdown;
        }
    }
}

/// Sleep for `dur`, returning `true` if a shutdown signal arrived first.
async fn sleep_or_shutdown(dur: Duration, shutdown_rx: &mut watch::Receiver<bool>) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(dur) => false,
        _ = shutdown_requested(shutdown_rx) => true,
    }
}

/// Resolve once a shutdown has been signalled. Resolves immediately if it
/// already has, and also if the signal plumbing is gone (fail toward stop).
async fn shutdown_requested(shutdown_rx: &mut watch::Receiver<bool>) {
    let _ = shutdown_rx.wait_for(|&stop| stop).await;
}

/// Load the agent config, fail-closed if it is absent or unreadable (the
/// agent stays disabled rather than guessing).
fn load_config(path: &Path) -> AgentConfig {
    match std::fs::read_to_string(path) {
        Ok(text) => AgentConfig::parse(&text),
        Err(_) => {
            tracing::info!("no ai.toml found; using safe defaults (agent disabled)");
            AgentConfig::fail_closed()
        }
    }
}

/// The path to `ai.toml` (`LUNARIS_AI_CONFIG` overrides the default).
fn ai_config_path() -> PathBuf {
    if let Ok(path) = std::env::var("LUNARIS_AI_CONFIG") {
        return PathBuf::from(path);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".config/lunaris/ai.toml")
}

/// The behaviour source directories: the system (built-in) directory and the
/// user directory.
fn behaviour_sources() -> Vec<BehaviourSource> {
    let mut sources = vec![BehaviourSource::builtin("/usr/share/lunaris/agent/behaviours")];
    if let Ok(home) = std::env::var("HOME") {
        sources.push(BehaviourSource::user(format!(
            "{home}/.local/share/lunaris/agent/behaviours"
        )));
    }
    // Dev-only: stand in for the not-yet-installed system directory when
    // running from a checkout. Compiled out of release builds so an
    // environment variable can never inject built-in-provenance behaviours
    // into a deployed system (it would otherwise satisfy a built-in-only
    // config approval from an attacker-controllable path).
    #[cfg(debug_assertions)]
    if let Ok(dir) = std::env::var("LUNARIS_AGENT_BEHAVIOURS") {
        sources.push(BehaviourSource::builtin(dir));
    }
    sources
}

fn consumer_socket() -> String {
    std::env::var("LUNARIS_CONSUMER_SOCKET").unwrap_or_else(|_| DEFAULT_CONSUMER_SOCKET.to_string())
}

fn graph_socket() -> String {
    std::env::var("LUNARIS_KNOWLEDGE_SOCKET").unwrap_or_else(|_| DEFAULT_GRAPH_SOCKET.to_string())
}

/// Resolve when Ctrl-C or SIGTERM arrives, so the daemon stops cleanly.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}
