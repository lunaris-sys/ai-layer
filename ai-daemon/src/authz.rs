//! Per-session authorization for MCP action servers.
//!
//! An action MCP server (one that mutates state) may only be called
//! once the user has authorized it. Authorization is granted per
//! session, covers exactly the scope the user approved, and is never
//! persisted: the [`AuthorizationStore`] holds grants in memory and
//! [`clear`](AuthorizationStore::clear) drops them all at session
//! end.
//!
//! The flow is split so the D-Bus layer stays thin:
//!
//! 1. The dispatch layer calls [`open_prompt`](AuthorizationStore::open_prompt)
//!    for a scope it needs, getting back a prompt id and a receiver.
//! 2. The D-Bus layer emits an `AuthorizationPrompt` signal carrying
//!    the prompt id and scope, then awaits the receiver.
//! 3. The desktop shell shows the user a modal and calls
//!    `respond_authorization(prompt_id, granted)`, which lands in
//!    [`resolve`](AuthorizationStore::resolve).
//! 4. `resolve` records the grant (if approved) and wakes the
//!    waiting receiver.
//!
//! Approving one scope never widens to another: each scope is a
//! distinct key, so authorizing file management does not authorize
//! email.

use std::collections::{HashMap, HashSet};

use tokio::sync::{oneshot, Mutex};
use uuid::Uuid;

/// In-memory per-session authorization state.
#[derive(Default)]
pub struct AuthorizationStore {
    /// Scopes the user has approved this session.
    granted: Mutex<HashSet<String>>,
    /// Prompts awaiting a user decision, keyed by prompt id.
    pending: Mutex<HashMap<Uuid, Pending>>,
}

struct Pending {
    scope: String,
    responder: oneshot::Sender<bool>,
}

/// Maximum simultaneously-pending prompts. A bound so a flood of
/// unanswered authorization requests cannot grow the pending map
/// without limit.
const MAX_PENDING_PROMPTS: usize = 16;

impl AuthorizationStore {
    /// Build an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a scope is already authorized this session.
    pub async fn is_granted(&self, scope: &str) -> bool {
        self.granted.lock().await.contains(scope)
    }

    /// Open an authorization prompt for `scope`.
    ///
    /// Returns the prompt id (for the D-Bus signal) and a receiver
    /// the caller awaits. The receiver resolves to the user's
    /// decision once [`resolve`](Self::resolve) is called, or to an
    /// error if the prompt is dropped without an answer.
    ///
    /// Returns `None` when [`MAX_PENDING_PROMPTS`] prompts are already
    /// open, so a flood of unanswered requests cannot grow the
    /// pending map without limit. The caller treats `None` as a
    /// denial.
    pub async fn open_prompt(
        &self,
        scope: &str,
    ) -> Option<(Uuid, oneshot::Receiver<bool>)> {
        let mut pending = self.pending.lock().await;
        if pending.len() >= MAX_PENDING_PROMPTS {
            return None;
        }
        let prompt_id = Uuid::new_v4();
        let (tx, rx) = oneshot::channel();
        pending.insert(
            prompt_id,
            Pending {
                scope: scope.to_string(),
                responder: tx,
            },
        );
        Some((prompt_id, rx))
    }

    /// Resolve a pending prompt with the user's decision.
    ///
    /// On approval the prompt's scope is recorded as granted. Returns
    /// `true` if a matching pending prompt existed; `false` if the
    /// prompt id was unknown (already resolved, or never opened).
    pub async fn resolve(&self, prompt_id: Uuid, granted: bool) -> bool {
        let pending = self.pending.lock().await.remove(&prompt_id);
        let Some(pending) = pending else {
            return false;
        };
        if granted {
            self.granted.lock().await.insert(pending.scope);
        }
        // The receiver may already be gone if the requester timed
        // out; that is not an error here.
        let _ = pending.responder.send(granted);
        true
    }

    /// Drop every grant and abandon every pending prompt. Called at
    /// session end; grants are never carried across sessions.
    pub async fn clear(&self) {
        self.granted.lock().await.clear();
        self.pending.lock().await.clear();
    }

    /// Number of scopes currently granted. Mainly for diagnostics.
    pub async fn granted_count(&self) -> usize {
        self.granted.lock().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn approving_a_prompt_grants_exactly_that_scope() {
        let store = AuthorizationStore::new();
        let (id, rx) = store.open_prompt("file-management").await.expect("open prompt");
        assert!(store.resolve(id, true).await);
        assert_eq!(rx.await.unwrap(), true);

        assert!(store.is_granted("file-management").await);
        // A different scope is not granted by approving this one.
        assert!(!store.is_granted("email").await);
    }

    #[tokio::test]
    async fn denying_a_prompt_grants_nothing() {
        let store = AuthorizationStore::new();
        let (id, rx) = store.open_prompt("email").await.expect("open prompt");
        assert!(store.resolve(id, false).await);
        assert_eq!(rx.await.unwrap(), false);
        assert!(!store.is_granted("email").await);
    }

    #[tokio::test]
    async fn resolve_with_unknown_prompt_id_is_noop() {
        let store = AuthorizationStore::new();
        assert!(!store.resolve(Uuid::new_v4(), true).await);
    }

    #[tokio::test]
    async fn clear_drops_all_grants() {
        let store = AuthorizationStore::new();
        let (a, _) = store.open_prompt("file-management").await.expect("open prompt");
        store.resolve(a, true).await;
        let (b, _) = store.open_prompt("terminal").await.expect("open prompt");
        store.resolve(b, true).await;
        assert_eq!(store.granted_count().await, 2);

        store.clear().await;
        assert_eq!(store.granted_count().await, 0);
        assert!(!store.is_granted("file-management").await);
        assert!(!store.is_granted("terminal").await);
    }

    #[tokio::test]
    async fn a_grant_does_not_survive_a_session_reset() {
        let store = AuthorizationStore::new();
        let (id, _) = store.open_prompt("terminal").await.expect("open prompt");
        store.resolve(id, true).await;
        assert!(store.is_granted("terminal").await);
        // Session end.
        store.clear().await;
        assert!(!store.is_granted("terminal").await);
    }

    #[tokio::test]
    async fn pending_prompts_are_capped() {
        let store = AuthorizationStore::new();
        // Fill the pending map to the cap; the receivers are kept
        // alive so the prompts stay pending.
        let mut held = Vec::new();
        for i in 0..MAX_PENDING_PROMPTS {
            let opened = store
                .open_prompt(&format!("scope-{i}"))
                .await
                .expect("under cap");
            held.push(opened);
        }
        // One more must be refused.
        assert!(
            store.open_prompt("overflow").await.is_none(),
            "prompt past the cap must be refused"
        );
        // Resolving one frees a slot.
        let (id, _) = held.pop().unwrap();
        store.resolve(id, false).await;
        assert!(
            store.open_prompt("now-fits").await.is_some(),
            "a freed slot admits a new prompt"
        );
    }

    #[tokio::test]
    async fn the_full_prompt_to_decision_round_trip() {
        let store = std::sync::Arc::new(AuthorizationStore::new());

        // The dispatch side opens a prompt and awaits the decision.
        let (id, rx) = store.open_prompt("calendar").await.expect("open prompt");

        // The shell side responds out of band.
        let store_for_shell = store.clone();
        tokio::spawn(async move {
            store_for_shell.resolve(id, true).await;
        });

        assert_eq!(rx.await.unwrap(), true);
        assert!(store.is_granted("calendar").await);
    }
}
