//! Peer-credential resolution for proxy callers.
//!
//! Trusting any session-bus process that happens to currently own
//! `org.lunaris.AI1` or `org.lunaris.AIAgent1` is not enough: a
//! well-known name is claimable. If the real daemon is absent, a
//! same-session attacker can grab the name and POST through the
//! proxy with a catalogued URL.
//!
//! This module hardens the check by resolving the caller's PID via
//! `org.freedesktop.DBus.GetConnectionUnixProcessID` and reading
//! `/proc/{pid}/exe` to obtain the canonical executable path. The
//! proxy then matches that path against a static set of allowed
//! executables (the production installation paths) plus an optional
//! env-overrideable dev allowlist.
//!
//! D-Bus session policy (`<allow own="org.lunaris.AI1"/>` in the
//! shipped XML) remains the first layer of defence. Peer-cred
//! verification is the second: even if name-ownership policy is
//! mis-installed or relaxed, only processes running the actual
//! daemon binary can transit the proxy.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::service::CallerIdentity;

/// Canonical production paths for the two AI daemons.
pub const CANONICAL_AI_DAEMON_BIN: &str = "/usr/lib/lunaris/libexec/lunaris-ai-daemon";
/// Canonical production path for the autonomous-agent daemon.
pub const CANONICAL_AI_AGENT_BIN: &str = "/usr/lib/lunaris/libexec/lunaris-ai-agent";

/// Env var that lets developers extend the executable allowlist for
/// dev installs (colon-separated paths). Production deploys leave it
/// unset.
pub const EXTRA_BINS_ENV: &str = "LUNARIS_AI_PROXY_EXTRA_BINS";

/// Mapping from executable path to the well-known name the proxy
/// will record for the caller.
#[derive(Debug, Clone)]
pub struct PeerAuthMap {
    by_exe: BTreeMap<PathBuf, String>,
}

impl Default for PeerAuthMap {
    fn default() -> Self {
        Self::default_lunaris()
    }
}

impl PeerAuthMap {
    /// Build the default mapping.
    ///
    /// In **debug builds only**, a `LUNARIS_AI_PROXY_EXTRA_BINS` env
    /// var extends the executable map for local iteration against
    /// repo-relative binaries (colon-separated `<exe-path>=<name>`
    /// pairs). The override is compiled out of release builds: an
    /// env-readable override would otherwise let a local process
    /// register its own binary as `org.lunaris.AI1`, turning a dev
    /// convenience into part of the production trust boundary.
    /// Release builds therefore trust only the two canonical
    /// install paths.
    pub fn default_lunaris() -> Self {
        let mut by_exe = BTreeMap::new();
        by_exe.insert(PathBuf::from(CANONICAL_AI_DAEMON_BIN), "org.lunaris.AI1".to_string());
        by_exe.insert(
            PathBuf::from(CANONICAL_AI_AGENT_BIN),
            "org.lunaris.AIAgent1".to_string(),
        );
        #[cfg(debug_assertions)]
        if let Ok(extras) = std::env::var(EXTRA_BINS_ENV) {
            for entry in extras.split(':').filter(|s| !s.is_empty()) {
                if let Some((path, name)) = entry.split_once('=') {
                    by_exe.insert(PathBuf::from(path), name.to_string());
                }
            }
        }
        Self { by_exe }
    }

    /// Resolve a caller's executable path to a well-known bus name.
    /// Returns `None` if the path is not in the allowlist.
    pub fn lookup(&self, exe_path: &Path) -> Option<&str> {
        self.by_exe.get(exe_path).map(String::as_str)
    }

    /// All allowed executable paths. Used by tests and diagnostics.
    pub fn allowed_paths(&self) -> impl Iterator<Item = &Path> {
        self.by_exe.keys().map(PathBuf::as_path)
    }
}

/// Errors raised while resolving peer credentials.
#[derive(Debug, thiserror::Error)]
pub enum PeerAuthError {
    /// D-Bus message had no sender.
    #[error("message has no sender")]
    NoSender,
    /// `GetConnectionUnixProcessID` returned an error.
    #[error("could not resolve caller PID: {0}")]
    PidLookup(String),
    /// `/proc/{pid}/exe` could not be read.
    #[error("could not read /proc/{pid}/exe: {error}")]
    ExeLookup {
        /// Caller PID.
        pid: u32,
        /// Reason for failure.
        error: String,
    },
    /// The caller's executable path is not in the allowlist.
    #[error("caller executable '{path}' is not allowed")]
    ExeNotAllowed {
        /// Caller executable path.
        path: String,
    },
    /// The caller's executable maps to a well-known name it does not
    /// actually own on the bus.
    #[error(
        "caller does not own '{name}': sender is {sender}, owner is {owner}"
    )]
    NameOwnershipMismatch {
        /// The well-known name derived from the caller's executable.
        name: String,
        /// The caller's unique bus name.
        sender: String,
        /// The actual owner of the well-known name (or `<none>`).
        owner: String,
    },
}

/// Resolve the caller identity given a unique bus name and the
/// daemon's D-Bus connection.
///
/// Two independent checks must both pass:
///
/// 1. The caller's executable, found via
///    `GetConnectionUnixProcessID` + `/proc/{pid}/exe`, must be in
///    the [`PeerAuthMap`].
/// 2. The caller must actually *own* the well-known name that its
///    executable maps to: `GetNameOwner(name)` must equal the
///    caller's unique bus name.
///
/// Each check alone is bypassable. Executable-path alone falls to an
/// `LD_PRELOAD` constructor on an allowed-but-non-setuid binary.
/// Name-ownership alone falls to a process that claims the name
/// while the real daemon is absent. Requiring both means an attacker
/// would have to run the genuine daemon binary *and* hold its
/// well-known name. Production deployments additionally restrict
/// name ownership through the shipped D-Bus policy XML; binding to
/// systemd unit / cgroup credentials is noted as future hardening.
pub async fn resolve(
    unique_bus_name: &str,
    connection: &zbus::Connection,
    peer_map: &PeerAuthMap,
) -> Result<CallerIdentity, PeerAuthError> {
    let dbus_proxy = zbus::fdo::DBusProxy::new(connection)
        .await
        .map_err(|e| PeerAuthError::PidLookup(format!("DBusProxy: {e}")))?;
    let bus_name = zbus::names::BusName::try_from(unique_bus_name)
        .map_err(|e| PeerAuthError::PidLookup(format!("parse bus name: {e}")))?;
    let pid = dbus_proxy
        .get_connection_unix_process_id(bus_name)
        .await
        .map_err(|e| PeerAuthError::PidLookup(e.to_string()))?;
    let exe_path = std::fs::read_link(format!("/proc/{pid}/exe"))
        .map_err(|e| PeerAuthError::ExeLookup {
            pid,
            error: e.to_string(),
        })?;

    // Check 1: executable allowlist.
    let well_known = peer_map.lookup(&exe_path).ok_or_else(|| {
        PeerAuthError::ExeNotAllowed {
            path: exe_path.to_string_lossy().to_string(),
        }
    })?;

    // Check 2: the caller must own that well-known name.
    let well_known_name = zbus::names::BusName::try_from(well_known)
        .map_err(|e| PeerAuthError::PidLookup(format!("parse well-known name: {e}")))?;
    let owner = dbus_proxy
        .get_name_owner(well_known_name)
        .await
        .map_err(|_| PeerAuthError::NameOwnershipMismatch {
            name: well_known.to_string(),
            sender: unique_bus_name.to_string(),
            owner: "<none>".to_string(),
        })?;
    if owner.as_str() != unique_bus_name {
        return Err(PeerAuthError::NameOwnershipMismatch {
            name: well_known.to_string(),
            sender: unique_bus_name.to_string(),
            owner: owner.as_str().to_string(),
        });
    }

    Ok(CallerIdentity {
        well_known_bus_name: Some(well_known.to_string()),
        unique_bus_name: unique_bus_name.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_map_contains_canonical_paths() {
        let m = PeerAuthMap::default_lunaris();
        let paths: Vec<&str> = m
            .allowed_paths()
            .map(|p| p.to_str().expect("utf-8"))
            .collect();
        assert!(paths.contains(&CANONICAL_AI_DAEMON_BIN));
        assert!(paths.contains(&CANONICAL_AI_AGENT_BIN));
    }

    #[test]
    fn lookup_canonical_paths_returns_well_known_names() {
        let m = PeerAuthMap::default_lunaris();
        assert_eq!(
            m.lookup(Path::new(CANONICAL_AI_DAEMON_BIN)),
            Some("org.lunaris.AI1")
        );
        assert_eq!(
            m.lookup(Path::new(CANONICAL_AI_AGENT_BIN)),
            Some("org.lunaris.AIAgent1")
        );
    }

    #[test]
    fn lookup_unknown_path_returns_none() {
        let m = PeerAuthMap::default_lunaris();
        assert!(m.lookup(Path::new("/usr/bin/bash")).is_none());
    }

    #[test]
    fn extra_bins_env_extends_the_map() {
        let prev = std::env::var(EXTRA_BINS_ENV).ok();
        std::env::set_var(
            EXTRA_BINS_ENV,
            "/tmp/debug-bin=org.lunaris.AI1:/tmp/agent=org.lunaris.AIAgent1",
        );
        let m = PeerAuthMap::default_lunaris();
        assert_eq!(
            m.lookup(Path::new("/tmp/debug-bin")),
            Some("org.lunaris.AI1")
        );
        assert_eq!(
            m.lookup(Path::new("/tmp/agent")),
            Some("org.lunaris.AIAgent1")
        );
        // Restore env state for subsequent tests.
        match prev {
            Some(v) => std::env::set_var(EXTRA_BINS_ENV, v),
            None => std::env::remove_var(EXTRA_BINS_ENV),
        }
    }
}
