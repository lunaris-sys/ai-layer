//! Capability model: what the AI may read and what it may do.
//!
//! Foundation §8.4 (the AI security model) separates two risk
//! categories that are configured independently:
//!
//! * **Read access** — a single global level 0..=4 ([`AccessTier`],
//!   defined in [`crate::graph_query`]) deciding how much of the
//!   Knowledge Graph the model can see.
//! * **Action permission** — an [`ActionMode`] deciding how the model
//!   is allowed to act on applications through MCP.
//!
//! Reading and acting are deliberately decoupled, because they are
//! different risk categories: a user may grant Full read while keeping
//! every action in Suggest mode, or run a single application
//! autonomously while the graph stays Minimal. A [`Capability`] is the
//! pair, held per caller.
//!
//! Two rules sit above the configured mode and are deliberately not
//! configurable (Foundation §8.4): a high-impact [`ActionKind`] always
//! requires explicit confirmation regardless of mode, and any action
//! triggered by external content always requires confirmation
//! (prompt-injection containment). [`Capability::decide`] applies both
//! before the per-application mode is consulted.

use std::collections::BTreeSet;

pub use crate::graph_query::AccessTier;

/// How the AI is permitted to act on an application (Foundation §8.4,
/// "Action permissions").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionMode {
    /// The AI proposes an action and the user executes it manually.
    Suggest,
    /// The AI executes the action, but shows a preview with a
    /// cancellation window before proceeding.
    Supervised,
    /// The AI acts without per-action confirmation. Only ever reached
    /// for an individually enabled application — Foundation forbids a
    /// global Autonomous setting, so this variant is never the
    /// baseline (see [`BaselineMode`]).
    Autonomous,
}

/// The non-autonomous baseline action mode.
///
/// Autonomous is excluded by construction: Foundation §8.4 states that
/// "autonomous mode is never a global setting", so the daemon-wide
/// default can only ever be Suggest or Supervised. Autonomy is granted
/// per application through [`ActionPermissions::autonomous_apps`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineMode {
    /// Propose actions; the user executes them.
    Suggest,
    /// Execute with a preview and a cancellation window.
    Supervised,
}

impl BaselineMode {
    /// Widen to the full [`ActionMode`]. A baseline is never
    /// Autonomous, so this only ever yields Suggest or Supervised.
    pub fn as_action_mode(self) -> ActionMode {
        match self {
            BaselineMode::Suggest => ActionMode::Suggest,
            BaselineMode::Supervised => ActionMode::Supervised,
        }
    }

    /// Parse the `action_mode` config string. Anything unrecognised,
    /// including a literal `"autonomous"` (which is not a valid global
    /// setting), falls back to the safest baseline, Suggest.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "supervised" => BaselineMode::Supervised,
            _ => BaselineMode::Suggest,
        }
    }
}

/// The action side of a [`Capability`].
///
/// `default_mode` is the baseline applied to every application and is
/// a [`BaselineMode`], so it can never be Autonomous. Autonomy is
/// opt-in per application: an application id present in
/// `autonomous_apps` resolves to [`ActionMode::Autonomous`], everything
/// else to the baseline.
#[derive(Debug, Clone)]
pub struct ActionPermissions {
    default_mode: BaselineMode,
    autonomous_apps: BTreeSet<String>,
}

impl ActionPermissions {
    /// Build action permissions from a baseline and a set of
    /// individually-enabled autonomous application ids.
    pub fn new<I, S>(default_mode: BaselineMode, autonomous_apps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            default_mode,
            autonomous_apps: autonomous_apps.into_iter().map(Into::into).collect(),
        }
    }

    /// The safe default: Suggest for every application, nothing
    /// autonomous. This is what a missing or unreadable action config
    /// resolves to.
    pub fn suggest_only() -> Self {
        Self::new(BaselineMode::Suggest, Vec::<String>::new())
    }

    /// The configured baseline mode.
    pub fn default_mode(&self) -> BaselineMode {
        self.default_mode
    }

    /// The effective [`ActionMode`] for a given application: Autonomous
    /// if the application was individually enabled, otherwise the
    /// baseline.
    pub fn mode_for(&self, app_id: &str) -> ActionMode {
        if self.autonomous_apps.contains(app_id) {
            ActionMode::Autonomous
        } else {
            self.default_mode.as_action_mode()
        }
    }

    /// Whether `app_id` is individually enabled for autonomous action.
    pub fn is_autonomous(&self, app_id: &str) -> bool {
        self.autonomous_apps.contains(app_id)
    }
}

/// Classification of an action the AI wants to take.
///
/// The non-`Ordinary` variants are the hardcoded high-impact actions
/// that Foundation §8.4 says "always require explicit confirmation
/// regardless of session permissions, because their consequences are
/// irreversible or high-impact". That list is not configurable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    /// Permanent (non-trash) file deletion.
    PermanentDelete,
    /// Sending any external message: email, chat, or similar.
    SendExternalMessage,
    /// Installing or removing a system package.
    PackageChange,
    /// Modifying system configuration outside `~/.config`.
    SystemConfigChange,
    /// A network request to a host outside the application's declared
    /// permissions.
    UndeclaredNetwork,
    /// Running a command with elevated privileges.
    ElevatedPrivilege,
    /// Any action not on the hardcoded high-impact list. Subject only
    /// to the action mode and the external-content rule.
    Ordinary,
}

impl ActionKind {
    /// Whether this kind is on the hardcoded always-confirm list. True
    /// for every variant except [`ActionKind::Ordinary`].
    pub fn always_requires_confirmation(self) -> bool {
        !matches!(self, ActionKind::Ordinary)
    }
}

/// What the daemon must do before an action proceeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionDecision {
    /// Hand the proposal to the user, who executes it (Suggest mode).
    Propose,
    /// Execute, but show a preview with a cancellation window first
    /// (Supervised mode).
    PreviewThenExecute,
    /// Block on an explicit per-action confirmation dialog, regardless
    /// of mode. Reached for high-impact kinds and externally-triggered
    /// actions.
    RequireConfirmation,
    /// Execute immediately, no confirmation (Autonomous mode for an
    /// individually-enabled application).
    Proceed,
}

/// A caller's capability: its read tier and its action permissions.
///
/// Held per caller. Phase 9-α applied a single daemon-wide
/// [`AccessTier::Minimal`]; S16 makes the tier the configured value and
/// pairs it with the action model.
#[derive(Debug, Clone)]
pub struct Capability {
    /// How much of the graph the caller may read.
    pub read_tier: AccessTier,
    /// How the caller may act on applications.
    pub actions: ActionPermissions,
}

impl Capability {
    /// Build a capability from a read tier and action permissions.
    pub fn new(read_tier: AccessTier, actions: ActionPermissions) -> Self {
        Self { read_tier, actions }
    }

    /// The fail-closed default: no graph access, Suggest-only actions.
    pub fn minimal() -> Self {
        Self {
            read_tier: AccessTier::Minimal,
            actions: ActionPermissions::suggest_only(),
        }
    }

    /// Decide what gate an action must pass.
    ///
    /// Precedence (both overrides are non-configurable, Foundation
    /// §8.4):
    ///
    /// 1. A high-impact [`ActionKind`] always requires confirmation.
    /// 2. An action triggered by external content always requires
    ///    confirmation, regardless of the configured mode — an AI that
    ///    read a malicious document cannot use that as a basis to act
    ///    without the user seeing exactly what it wants to do.
    /// 3. Otherwise the per-application [`ActionMode`] decides.
    pub fn decide(
        &self,
        app_id: &str,
        kind: ActionKind,
        triggered_by_external_content: bool,
    ) -> ActionDecision {
        if kind.always_requires_confirmation() || triggered_by_external_content {
            return ActionDecision::RequireConfirmation;
        }
        match self.actions.mode_for(app_id) {
            ActionMode::Suggest => ActionDecision::Propose,
            ActionMode::Supervised => ActionDecision::PreviewThenExecute,
            ActionMode::Autonomous => ActionDecision::Proceed,
        }
    }
}

/// Map the global read access level (0..=4, Foundation §8.4 table) to
/// an [`AccessTier`]. Any value above 4 is clamped to the strictest
/// tier, Minimal, so a malformed config never widens access.
pub fn access_tier_from_level(level: u8) -> AccessTier {
    match level {
        0 => AccessTier::Minimal,
        1 => AccessTier::SessionScoped,
        2 => AccessTier::ProjectScoped,
        3 => AccessTier::TimeScoped,
        4 => AccessTier::Full,
        _ => AccessTier::Minimal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_never_widens_to_autonomous() {
        assert_eq!(BaselineMode::parse("autonomous"), BaselineMode::Suggest);
        assert_eq!(BaselineMode::parse("Supervised"), BaselineMode::Supervised);
        assert_eq!(BaselineMode::parse("garbage"), BaselineMode::Suggest);
        assert_eq!(
            BaselineMode::Supervised.as_action_mode(),
            ActionMode::Supervised
        );
    }

    #[test]
    fn autonomous_is_per_app_not_global() {
        let perms = ActionPermissions::new(BaselineMode::Supervised, ["org.lunaris.files"]);
        assert_eq!(perms.mode_for("org.lunaris.files"), ActionMode::Autonomous);
        assert_eq!(perms.mode_for("org.lunaris.mail"), ActionMode::Supervised);
        assert!(perms.is_autonomous("org.lunaris.files"));
        assert!(!perms.is_autonomous("org.lunaris.mail"));
    }

    #[test]
    fn high_impact_actions_always_confirm_even_when_autonomous() {
        let cap = Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Supervised, ["org.lunaris.files"]),
        );
        // org.lunaris.files is autonomous, but a permanent delete still
        // requires confirmation.
        assert_eq!(
            cap.decide("org.lunaris.files", ActionKind::PermanentDelete, false),
            ActionDecision::RequireConfirmation
        );
        for kind in [
            ActionKind::PermanentDelete,
            ActionKind::SendExternalMessage,
            ActionKind::PackageChange,
            ActionKind::SystemConfigChange,
            ActionKind::UndeclaredNetwork,
            ActionKind::ElevatedPrivilege,
        ] {
            assert!(kind.always_requires_confirmation());
            assert_eq!(
                cap.decide("org.lunaris.files", kind, false),
                ActionDecision::RequireConfirmation
            );
        }
    }

    #[test]
    fn external_content_forces_confirmation_regardless_of_mode() {
        let cap = Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Supervised, ["org.lunaris.files"]),
        );
        // Ordinary action in an autonomous app would normally Proceed,
        // but external-content provenance forces confirmation.
        assert_eq!(
            cap.decide("org.lunaris.files", ActionKind::Ordinary, true),
            ActionDecision::RequireConfirmation
        );
    }

    #[test]
    fn ordinary_actions_follow_the_configured_mode() {
        let suggest = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        assert_eq!(
            suggest.decide("org.lunaris.files", ActionKind::Ordinary, false),
            ActionDecision::Propose
        );

        let supervised = Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Supervised, Vec::<String>::new()),
        );
        assert_eq!(
            supervised.decide("org.lunaris.files", ActionKind::Ordinary, false),
            ActionDecision::PreviewThenExecute
        );

        let autonomous = Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Suggest, ["org.lunaris.files"]),
        );
        assert_eq!(
            autonomous.decide("org.lunaris.files", ActionKind::Ordinary, false),
            ActionDecision::Proceed
        );
    }

    #[test]
    fn access_level_maps_and_clamps() {
        assert_eq!(access_tier_from_level(0), AccessTier::Minimal);
        assert_eq!(access_tier_from_level(1), AccessTier::SessionScoped);
        assert_eq!(access_tier_from_level(2), AccessTier::ProjectScoped);
        assert_eq!(access_tier_from_level(3), AccessTier::TimeScoped);
        assert_eq!(access_tier_from_level(4), AccessTier::Full);
        // Out-of-range never widens access.
        assert_eq!(access_tier_from_level(5), AccessTier::Minimal);
        assert_eq!(access_tier_from_level(255), AccessTier::Minimal);
    }

    #[test]
    fn minimal_capability_is_fail_closed() {
        let cap = Capability::minimal();
        assert_eq!(cap.read_tier, AccessTier::Minimal);
        assert_eq!(
            cap.decide("any.app", ActionKind::Ordinary, false),
            ActionDecision::Propose
        );
    }
}
