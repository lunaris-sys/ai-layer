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

use lunaris_ai_agent::behaviour::{BehaviourKind, ReadScope};
use lunaris_ai_agent::config::{AgentConfig, ProviderSettings};
use lunaris_ai_agent::engine::{reads_satisfied, DispatchOutcome, Dispatcher};
use lunaris_ai_agent::gate::Gate;
use lunaris_ai_agent::slice::{FsPathResolver, ProcMountsPolicy};
use lunaris_ai_agent::graph::{UnixGraph, DEFAULT_GRAPH_SOCKET};
use lunaris_ai_agent::handlers::builtin_handlers;
use lunaris_ai_agent::loader::{load, BehaviourSource};
use lunaris_ai_agent::seams::{NullObserver, SystemClock, TriggerSource};
use lunaris_ai_agent::source::{subscription_types, EventBusSource, DEFAULT_CONSUMER_SOCKET};
use lunaris_ai_core::audit::LedgerAuditSink;
use lunaris_ai_core::capability::{AccessTier, Capability};
use lunaris_ai_core::provider::AIProvider;
use lunaris_ai_providers::proxied::{ProxiedConfig, ProxiedProvider};
use os_sdk::config::{Config, ConfigWatcher};
use zbus::Connection;

/// The well-known D-Bus name the agent owns so `ai-proxy` peer-authorises its
/// completion forwards (Foundation §8.4.6: outbound LLM traffic transits the
/// proxy, which checks the caller owns this name).
const AGENT_BUS_NAME: &str = "org.lunaris.AIAgent1";

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

    // The session-bus connection a configured provider forwards on, owned
    // once for the process and reused across epochs (owning the name per
    // epoch would thrash). Established lazily, only when a provider is
    // configured, and retried on a later epoch if the bus was not yet
    // reachable, so a late session bus or a transient failure is recovered on
    // the next config change rather than disabling agent behaviours forever.
    let mut connection: Option<Connection> = None;

    run(
        &handlers,
        &audit,
        &observer,
        &graph,
        &ai_path,
        &mut connection,
        shutdown_rx,
    )
    .await
}

/// Open a session-bus connection and own [`AGENT_BUS_NAME`] as the sole,
/// non-replaceable owner. Returns `None` (with a log line) when no session bus
/// is reachable or the name is already owned, so a provider simply cannot be
/// built and agent behaviours stay skipped rather than the daemon failing.
///
/// The name is requested with `DoNotQueue` and *without* `AllowReplacement`,
/// so this owner cannot later be displaced and the daemon never queues behind
/// another owner. Only primary ownership counts; anything else (the name is
/// already taken) means a second instance, and that instance must not run a
/// provider whose forwards the proxy would attribute to the real owner.
async fn establish_agent_connection() -> Option<Connection> {
    use zbus::fdo::{RequestNameFlags, RequestNameReply};

    let connection = match Connection::session().await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "no session bus; agent (LLM) behaviours will not run");
            return None;
        }
    };
    match connection
        .request_name_with_flags(AGENT_BUS_NAME, RequestNameFlags::DoNotQueue.into())
        .await
    {
        Ok(RequestNameReply::PrimaryOwner) | Ok(RequestNameReply::AlreadyOwner) => Some(connection),
        Ok(other) => {
            tracing::warn!(name = AGENT_BUS_NAME, ?other, "agent bus name is already owned; agent (LLM) behaviours will not run");
            None
        }
        Err(e) => {
            tracing::warn!(error = %e, name = AGENT_BUS_NAME, "could not own the agent bus name; agent (LLM) behaviours will not run");
            None
        }
    }
}

/// Build the proxied LLM provider for this epoch, **best-effort and
/// non-blocking**: own the bus name (lazily, once) and build the provider, but
/// never wait on the bus. If the session bus is unavailable or the build
/// fails, return `None` so agent behaviours skip while unrelated workflow
/// behaviours still subscribe and run; a build error clears the stored
/// connection so the next epoch re-establishes rather than reusing a dead one.
/// Recovery is by reload (a config change rebuilds the epoch and retries) or,
/// for an agent-only config with nothing else runnable, by the supervisor
/// restart §5.5 already prescribes. Blocking the epoch to retry the provider
/// would starve workflow behaviours on an LLM/bus outage, so it is avoided; a
/// concurrent background retry that re-arms agent behaviours mid-epoch without
/// a reload (recovering a transient bus/proxy outage faster) is a deliberate
/// daemon-hardening follow-up, not done here.
async fn build_provider(
    settings: &ProviderSettings,
    connection: &mut Option<Connection>,
) -> Option<ProxiedProvider> {
    if connection.is_none() {
        *connection = establish_agent_connection().await;
    }
    // Build off the borrow, then act on the result so a failure can clear the
    // connection without overlapping the borrow.
    let built = match connection.as_ref() {
        Some(conn) => Some(ProxiedProvider::with_connection(provider_config(settings), conn).await),
        None => None,
    };
    match built {
        Some(Ok(provider)) => Some(provider),
        Some(Err(e)) => {
            tracing::warn!(error = %e, "could not build the LLM provider; re-establishing the connection next epoch; agent behaviours will not run this epoch");
            *connection = None;
            None
        }
        None => {
            tracing::warn!("a provider is configured but the session bus is unavailable; agent behaviours will not run this epoch (retried on reload)");
            None
        }
    }
}

/// Map the resolved provider settings onto the proxy adapter's config.
fn provider_config(settings: &ProviderSettings) -> ProxiedConfig {
    ProxiedConfig {
        name: settings.name.clone(),
        model: settings.model.clone(),
        audit_token: settings.audit_token.clone(),
        context_window: settings.context_window,
    }
}

/// Whether a behaviour can actually run this epoch, mirroring the dispatcher's
/// own eligibility: it must be enabled and its declared read scope satisfied
/// by the configured tier (the dispatcher skips it otherwise), and a
/// `kind: agent` behaviour additionally needs an LLM provider wired (a
/// workflow never does).
fn behaviour_is_runnable(
    enabled: bool,
    kind: BehaviourKind,
    reads: ReadScope,
    read_tier: AccessTier,
    has_provider: bool,
) -> bool {
    enabled
        && reads_satisfied(reads, read_tier)
        && (kind != BehaviourKind::Agent || has_provider)
}

/// Whether an enabled agent behaviour that the configured tier actually allows
/// to run needs an LLM provider this epoch. Over-scoped agents (skipped by the
/// dispatcher anyway) do not count, so a workflow-only epoch is never blocked
/// retrying a provider for a behaviour that could not run regardless.
fn agent_needs_provider(
    enabled: bool,
    kind: BehaviourKind,
    reads: ReadScope,
    read_tier: AccessTier,
) -> bool {
    enabled && kind == BehaviourKind::Agent && reads_satisfied(reads, read_tier)
}

/// The epoch loop. Each iteration is one config epoch: load settings and
/// behaviours, run the dispatcher, and rebuild on the next config change.
async fn run(
    handlers: &lunaris_ai_agent::engine::HandlerRegistry,
    audit: &LedgerAuditSink,
    observer: &NullObserver,
    graph: &UnixGraph,
    ai_path: &Path,
    connection: &mut Option<Connection>,
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

        // Build this epoch's LLM provider from the fresh config. When a
        // provider is configured and an enabled agent behaviour needs it, the
        // session bus is treated as a dependency to retry with backoff (like
        // the Event Bus path), so a late bus does not leave agent behaviours
        // offline; a config change or shutdown during the wait ends the epoch.
        // The provider is rebuilt per epoch (it owns a cheap clone of the
        // connection) so a settings change repoints it.
        let needs_provider = outcome.loaded.iter().any(|b| {
            agent_needs_provider(
                b.status.is_enabled(),
                b.behaviour.manifest.kind,
                b.behaviour.manifest.reads,
                config.read_tier,
            )
        });
        // Build the provider only when one is configured and an eligible
        // agent behaviour needs it. Best-effort and non-blocking: an
        // unavailable bus leaves agents skipped but never blocks workflow
        // behaviours from subscribing and running.
        let provider_holder: Option<ProxiedProvider> = match (needs_provider, &config.provider) {
            (true, Some(settings)) => build_provider(settings, connection).await,
            _ => None,
        };
        let provider: Option<&dyn AIProvider> =
            provider_holder.as_ref().map(|p| p as &dyn AIProvider);

        // Foundation §5.5: with nothing *runnable* the daemon has no reason to
        // run. A behaviour is runnable when enabled and either a workflow or
        // (for an agent) backed by a configured provider. Exit cleanly
        // otherwise (the supervisor restarts it when a runnable behaviour is
        // enabled); this also covers a removed config.
        let mut runnable = 0usize;
        for b in &outcome.loaded {
            let enabled = b.status.is_enabled();
            let kind = b.behaviour.manifest.kind;
            let reads = b.behaviour.manifest.reads;
            if behaviour_is_runnable(enabled, kind, reads, config.read_tier, provider.is_some()) {
                runnable += 1;
            } else if agent_needs_provider(enabled, kind, reads, config.read_tier) {
                // An eligible agent behaviour kept off only by a missing
                // provider (an over-scoped one is skipped by the dispatcher
                // with its own log, not here).
                tracing::warn!(
                    behaviour = %b.behaviour.manifest.name,
                    "agent behaviour is enabled but no AI provider is available; it will not run"
                );
            }
        }
        if runnable == 0 {
            tracing::info!("no runnable behaviours; the agent has nothing to do, exiting");
            return Ok(());
        }

        tracing::info!(runnable, "starting agent");

        let read_tier = config.read_tier;
        let capability = Capability::new(read_tier, config.actions);
        // The world-model seams the gate's predict-before-act step reads
        // through: the same graph the handlers use, plus the production path
        // and read-only mount resolvers.
        let paths = FsPathResolver;
        let mounts = ProcMountsPolicy;
        let gate = Gate::new(&capability, audit, observer, &paths, &mounts);
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
        // Wait for the next event, ending the epoch on a config change or
        // shutdown before any further dispatch under the old grants.
        let event = tokio::select! {
            biased;
            end = wait_config_change(watcher, shutdown_rx) => return end,
            maybe_event = source.recv() => match maybe_event {
                // The SDK consumer reconnects internally, so a closed source
                // means it is permanently gone; rebuild to recover.
                None => {
                    tracing::warn!("event source closed; rebuilding");
                    return EpochEnd::Reload;
                }
                Some(event) => event,
            },
        };
        // A change that landed between subscribing and now is honored before
        // the event is dispatched.
        if matches!(watcher.try_recv(), Ok(()) | Err(TryRecvError::Disconnected)) {
            return EpochEnd::Reload;
        }
        // Race the dispatch (which, for a `kind: agent` behaviour, may run a
        // whole bounded loop) against a config change or shutdown, so a
        // revocation aborts an in-flight agent loop at its next await rather
        // than letting it run to its budget under stale grants. Dropping the
        // dispatch future cancels it cleanly: suggest-mode executes nothing,
        // so no partial action is left behind, and the gate audits before it
        // decides, so a dropped step leaves a record but no surfaced action.
        if let Some(end) = dispatch_or_reload(
            dispatcher.dispatch(&event),
            wait_config_change(watcher, shutdown_rx),
        )
        .await
        {
            return end;
        }
    }
}

/// Run `dispatch` to completion, logging its outcomes, unless `abort` (a
/// config change or shutdown) resolves first. Returns `Some(end)` when aborted
/// (the dispatch future is dropped, cancelling it), `None` when the dispatch
/// completed. `biased` so a pending revocation wins over finishing the event.
///
/// Revocation contract: dropping the dispatch future stops the in-flight
/// agent loop at its next await, so no further provider call or gate decision
/// is made under the old grants, and (suggest-mode) nothing is executed. One
/// caveat is inherent to cancelling a future, not specific to this code: a
/// provider call already inside `complete` may have sent its proxy forward, so
/// that single LLM egress can still complete upstream under the old prompt and
/// grants; its response is then discarded with the dropped future. Aborting
/// that already-sent egress needs proxy-side, correlation-id-keyed
/// cancellation (an `ai-proxy` feature), a deliberate follow-up.
async fn dispatch_or_reload(
    dispatch: impl std::future::Future<Output = Vec<DispatchOutcome>>,
    abort: impl std::future::Future<Output = EpochEnd>,
) -> Option<EpochEnd> {
    tokio::select! {
        biased;
        end = abort => Some(end),
        outcomes = dispatch => {
            for outcome in &outcomes {
                log_dispatch_outcome(outcome);
            }
            None
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_provider_makes_an_eligible_agent_behaviour_runnable() {
        use BehaviourKind::{Agent, Workflow};
        let ok = AccessTier::Full; // satisfies any read scope
        // An enabled agent behaviour runs only with a provider; a workflow runs
        // either way; a disabled behaviour never runs.
        assert!(!behaviour_is_runnable(true, Agent, ReadScope::Minimal, ok, false));
        assert!(behaviour_is_runnable(true, Agent, ReadScope::Minimal, ok, true));
        assert!(behaviour_is_runnable(true, Workflow, ReadScope::Minimal, ok, false));
        assert!(!behaviour_is_runnable(false, Agent, ReadScope::Minimal, ok, true));
        // An over-scoped behaviour (read scope exceeds the tier) never runs,
        // even with a provider, matching the dispatcher's own skip.
        assert!(!behaviour_is_runnable(true, Agent, ReadScope::Full, AccessTier::Minimal, true));
        assert!(!behaviour_is_runnable(true, Workflow, ReadScope::Full, AccessTier::Minimal, false));
    }

    #[test]
    fn only_an_eligible_agent_behaviour_needs_a_provider() {
        use BehaviourKind::{Agent, Workflow};
        let ok = AccessTier::Full;
        assert!(agent_needs_provider(true, Agent, ReadScope::Minimal, ok));
        // Workflow never needs one; disabled never needs one; an over-scoped
        // agent (skipped anyway) must not make the daemon block on the bus.
        assert!(!agent_needs_provider(true, Workflow, ReadScope::Minimal, ok));
        assert!(!agent_needs_provider(false, Agent, ReadScope::Minimal, ok));
        assert!(!agent_needs_provider(true, Agent, ReadScope::Full, AccessTier::Minimal));
    }

    #[test]
    fn provider_config_maps_every_setting_onto_the_proxy_config() {
        let settings = ProviderSettings {
            name: "ollama-default".to_string(),
            model: "llama3:8b".to_string(),
            context_window: 131072,
            audit_token: "tok-xyz".to_string(),
        };
        let cfg = provider_config(&settings);
        assert_eq!(cfg.name, "ollama-default");
        assert_eq!(cfg.model, "llama3:8b");
        assert_eq!(cfg.context_window, 131072);
        assert_eq!(cfg.audit_token, "tok-xyz");
    }

    #[tokio::test]
    async fn a_config_change_aborts_an_in_flight_dispatch() {
        // A never-completing dispatch (stands in for a long agent loop) is
        // abandoned the moment a config change is observed.
        let result = dispatch_or_reload(
            std::future::pending::<Vec<DispatchOutcome>>(),
            std::future::ready(EpochEnd::Reload),
        )
        .await;
        assert!(matches!(result, Some(EpochEnd::Reload)));
    }

    #[tokio::test]
    async fn a_completed_dispatch_continues_the_epoch() {
        // With no config change pending, the dispatch completes and the epoch
        // continues (no abort).
        let result = dispatch_or_reload(
            std::future::ready(Vec::<DispatchOutcome>::new()),
            std::future::pending::<EpochEnd>(),
        )
        .await;
        assert!(result.is_none());
    }
}
