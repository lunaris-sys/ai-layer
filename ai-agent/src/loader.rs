//! Behaviour loader: discovery, provenance stamping, and enablement.
//!
//! This is the layer above the [`crate::behaviour`] parser. It is where the
//! **trust state that must not live in the untrusted `SKILL.md`** is
//! decided:
//!
//! * **Provenance is stamped from the source the file was found in**, never
//!   read from the manifest. A file under the system behaviour directory is
//!   [`Provenance::BuiltIn`]; under the user directory, [`Provenance::User`];
//!   from an installed package, [`Provenance::ThirdParty`]. A downloaded or
//!   agent-authored skill therefore cannot claim to be built-in.
//! * **Enablement comes from trusted Settings**, passed in as the set of
//!   enabled behaviour names — not from the file. A behaviour that is not in
//!   that set is loaded but marked [`Status::Disabled`], so it is listed in
//!   Settings (to be toggled) but never dispatched. A manifest cannot
//!   enable itself (the `enabled` key is not even a manifest field; the
//!   parser rejects it).
//!
//! Discovery is fail-soft per behaviour: a single unparseable `SKILL.md`
//! becomes a [`LoadError`] in the errors list and is *not* loaded, but it
//! does not sink the other behaviours.
//!
//! Not decided here (deliberately): whether a behaviour's declared `reads`
//! tier is *satisfiable* under the user's global read level. Because the
//! access tiers are non-nested label *lenses* (e.g. project-scoped grants
//! `Project` but time-scoped does not), satisfiability is a label-coverage
//! question that belongs to the read/grounding layer where the graph schema
//! is available — not an ordinal comparison here. The loader carries the
//! requirement (`loaded.behaviour.manifest.reads`) for that layer to check.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::behaviour::{parse, Behaviour, BehaviourError};

/// Where a behaviour came from — the basis for its trust tier. Stamped by
/// the loader from the source directory, never self-declared by the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// Shipped with Lunaris (the system behaviour directory).
    BuiltIn,
    /// Authored by the user (the user behaviour directory).
    User,
    /// Installed from a third-party package; the lowest trust tier.
    ThirdParty,
}

/// A directory to discover behaviours under, tagged with the provenance
/// every behaviour found there is stamped with.
#[derive(Debug, Clone)]
pub struct BehaviourSource {
    /// The directory holding one subdirectory per behaviour.
    pub root: PathBuf,
    /// The provenance stamped on every behaviour found under `root`.
    pub provenance: Provenance,
}

impl BehaviourSource {
    /// A built-in (system) source.
    pub fn builtin(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            provenance: Provenance::BuiltIn,
        }
    }

    /// A user source.
    pub fn user(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            provenance: Provenance::User,
        }
    }

    /// A third-party (installed-package) source.
    pub fn third_party(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            provenance: Provenance::ThirdParty,
        }
    }
}

/// Why a loaded behaviour is not active.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisableReason {
    /// Not present in the trusted set of enabled behaviour names. The
    /// default state — behaviours are opt-in (Foundation §5.5).
    NotEnabledInSettings,
    /// Another behaviour was discovered under the same name. The name is
    /// ambiguous, so *every* instance is disabled fail-closed: a user or
    /// third-party `SKILL.md` must never be able to shadow a built-in
    /// behaviour's enabled toggle by reusing its name.
    DuplicateName,
}

/// Whether a loaded behaviour will be dispatched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Enabled in trusted Settings; eligible for dispatch.
    Enabled,
    /// Loaded and listable, but not dispatched.
    Disabled(DisableReason),
}

impl Status {
    /// Whether this behaviour is eligible for dispatch.
    pub fn is_enabled(&self) -> bool {
        matches!(self, Status::Enabled)
    }
}

/// A behaviour as loaded from disk: its parsed contract plus the trust
/// state the loader stamped onto it.
#[derive(Debug, Clone)]
pub struct LoadedBehaviour {
    /// The parsed manifest + body.
    pub behaviour: Behaviour,
    /// Stamped from the source directory.
    pub provenance: Provenance,
    /// The behaviour directory it was loaded from.
    pub dir: PathBuf,
    /// Enabled / disabled, resolved from trusted Settings.
    pub status: Status,
}

/// A behaviour directory that could not be loaded. It is excluded
/// (fail-closed) rather than loaded with guessed defaults.
#[derive(Debug, Error)]
pub enum LoadError {
    /// The `SKILL.md` could not be read.
    #[error("could not read {}: {source}", dir.display())]
    Io {
        /// The behaviour directory.
        dir: PathBuf,
        /// The underlying IO error.
        source: std::io::Error,
    },
    /// The `SKILL.md` failed to parse or validate.
    #[error("invalid behaviour in {}: {source}", dir.display())]
    Parse {
        /// The behaviour directory.
        dir: PathBuf,
        /// The parse/validation failure.
        source: BehaviourError,
    },
    /// The `SKILL.md` is larger than [`MAX_SKILL_BYTES`]; it is rejected
    /// without being read, so an oversized (even disabled) file from an
    /// untrusted source cannot exhaust memory or stall discovery.
    #[error("behaviour in {} is too large ({size} bytes > {limit} limit)", dir.display())]
    TooLarge {
        /// The behaviour directory.
        dir: PathBuf,
        /// The file's size in bytes.
        size: u64,
        /// The enforced limit.
        limit: u64,
    },
}

/// Maximum size of a `SKILL.md`. A behaviour manifest plus its body is
/// small (well under this); anything larger is rejected unread so an
/// untrusted source cannot DoS discovery. Generous so legitimate
/// behaviours with substantial instructions are never affected.
pub const MAX_SKILL_BYTES: u64 = 256 * 1024;

/// The result of a discovery pass: the behaviours that loaded, and the
/// directories that failed (kept separate so a bad behaviour cannot mask
/// the good ones).
#[derive(Debug, Default)]
pub struct LoadOutcome {
    /// Successfully loaded behaviours.
    pub loaded: Vec<LoadedBehaviour>,
    /// Directories that failed to load.
    pub errors: Vec<LoadError>,
}

/// Discover and load every behaviour under each source, stamping
/// provenance from the source and resolving enablement from trusted
/// Settings. A source whose root does not exist contributes nothing (it is
/// not an error — the user directory may simply be absent).
///
/// `enabled` maps a behaviour name to the **provenance it was approved
/// for**: a behaviour is enabled only if it was loaded from that exact
/// source kind. This binds the approval to a trusted identity, so a
/// lower-trust `SKILL.md` reusing a built-in's name cannot inherit its
/// toggle even when the built-in is absent or failed to load.
pub fn load(sources: &[BehaviourSource], enabled: &BTreeMap<String, Provenance>) -> LoadOutcome {
    let mut outcome = LoadOutcome::default();
    for source in sources {
        load_source(source, enabled, &mut outcome);
    }
    // Fail-closed on name collisions across all sources, after enablement
    // is resolved, so a duplicate can never keep an Enabled status.
    disable_duplicate_names(&mut outcome.loaded);
    outcome
}

fn load_source(
    source: &BehaviourSource,
    enabled: &BTreeMap<String, Provenance>,
    outcome: &mut LoadOutcome,
) {
    let entries = match std::fs::read_dir(&source.root) {
        Ok(entries) => entries,
        // A missing source root is normal (e.g. no user behaviours yet).
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(source_err) => {
            outcome.errors.push(LoadError::Io {
                dir: source.root.clone(),
                source: source_err,
            });
            return;
        }
    };

    // Iterate explicitly: a directory-entry or metadata IO failure is
    // recorded as a LoadError, not silently dropped — otherwise a
    // permission error could make an enabled behaviour vanish unnoticed.
    for entry_res in entries {
        let entry = match entry_res {
            Ok(entry) => entry,
            Err(source_err) => {
                outcome.errors.push(LoadError::Io {
                    dir: source.root.clone(),
                    source: source_err,
                });
                continue;
            }
        };
        let dir = entry.path();
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => {}
            Ok(_) => continue, // a stray file under the source root
            Err(source_err) => {
                outcome.errors.push(LoadError::Io { dir, source: source_err });
                continue;
            }
        }
        let skill = dir.join("SKILL.md");
        let size = match std::fs::metadata(&skill) {
            Ok(meta) if meta.is_file() => meta.len(),
            // No SKILL.md (or it is not a file): not a behaviour directory.
            Ok(_) => continue,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            // A real IO failure reaching SKILL.md must be observable.
            Err(source_err) => {
                outcome.errors.push(LoadError::Io { dir, source: source_err });
                continue;
            }
        };
        // Reject oversized files *before* reading them, so an untrusted
        // source cannot exhaust memory or stall discovery.
        if size > MAX_SKILL_BYTES {
            outcome.errors.push(LoadError::TooLarge {
                dir,
                size,
                limit: MAX_SKILL_BYTES,
            });
            continue;
        }
        match load_one(&skill) {
            Ok(behaviour) => {
                let status = resolve_status(&behaviour.manifest.name, source.provenance, enabled);
                outcome.loaded.push(LoadedBehaviour {
                    behaviour,
                    provenance: source.provenance,
                    dir,
                    status,
                });
            }
            Err(err) => outcome.errors.push(err),
        }
    }
}

/// Any name that occurs more than once across all loaded behaviours is
/// ambiguous; disable every instance fail-closed so an enabled toggle can
/// never resolve to the wrong (or a shadowing) implementation.
fn disable_duplicate_names(loaded: &mut [LoadedBehaviour]) {
    let mut counts: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
    for lb in loaded.iter() {
        *counts.entry(lb.behaviour.manifest.name.as_str()).or_insert(0) += 1;
    }
    let duplicates: BTreeSet<String> = counts
        .into_iter()
        .filter(|&(_, n)| n > 1)
        .map(|(name, _)| name.to_string())
        .collect();
    for lb in loaded.iter_mut() {
        if duplicates.contains(&lb.behaviour.manifest.name) {
            lb.status = Status::Disabled(DisableReason::DuplicateName);
        }
    }
}

fn load_one(skill: &Path) -> Result<Behaviour, LoadError> {
    let content = std::fs::read_to_string(skill).map_err(|source| LoadError::Io {
        dir: skill.parent().unwrap_or(skill).to_path_buf(),
        source,
    })?;
    parse(&content).map_err(|source| LoadError::Parse {
        dir: skill.parent().unwrap_or(skill).to_path_buf(),
        source,
    })
}

/// Resolve a behaviour's status from trusted Settings. Enabled only when
/// the name is approved *for this exact provenance* — a lower-trust source
/// cannot satisfy a toggle approved for a built-in. Pure, so the rule is
/// unit-testable without touching the filesystem.
fn resolve_status(
    name: &str,
    provenance: Provenance,
    enabled: &BTreeMap<String, Provenance>,
) -> Status {
    if enabled.get(name) == Some(&provenance) {
        Status::Enabled
    } else {
        Status::Disabled(DisableReason::NotEnabledInSettings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enablement_binds_name_and_provenance() {
        let mut enabled = BTreeMap::new();
        enabled.insert("auto-tag-by-project".to_string(), Provenance::BuiltIn);

        // Approved name + the provenance it was approved for → enabled.
        assert_eq!(
            resolve_status("auto-tag-by-project", Provenance::BuiltIn, &enabled),
            Status::Enabled
        );
        // Same name from a lower-trust source → NOT enabled (no shadowing).
        assert_eq!(
            resolve_status("auto-tag-by-project", Provenance::ThirdParty, &enabled),
            Status::Disabled(DisableReason::NotEnabledInSettings)
        );
        // Unknown name → not enabled.
        assert!(!resolve_status("tidy-downloads", Provenance::BuiltIn, &enabled).is_enabled());
        // Empty settings → nothing is enabled (opt-in default).
        assert!(
            !resolve_status("auto-tag-by-project", Provenance::BuiltIn, &BTreeMap::new())
                .is_enabled()
        );
    }

    #[test]
    fn missing_source_root_is_not_an_error() {
        let sources = [BehaviourSource::user("/nonexistent/lunaris/behaviours")];
        let outcome = load(&sources, &BTreeMap::new());
        assert!(outcome.loaded.is_empty());
        assert!(outcome.errors.is_empty());
    }

    fn loaded(name: &str, provenance: Provenance, status: Status) -> LoadedBehaviour {
        let src = format!(
            "---\nname: {name}\ndescription: d\nkind: workflow\nhandler: h\ntrigger:\n  type: manual\n---\n"
        );
        LoadedBehaviour {
            behaviour: parse(&src).expect("valid"),
            provenance,
            dir: PathBuf::from("/test").join(name),
            status,
        }
    }

    #[test]
    fn duplicate_names_across_sources_are_disabled_fail_closed() {
        // A third-party behaviour reusing a built-in's enabled name must
        // not inherit the toggle: both instances are disabled.
        let mut behaviours = vec![
            loaded("auto-tag-by-project", Provenance::BuiltIn, Status::Enabled),
            loaded("auto-tag-by-project", Provenance::ThirdParty, Status::Enabled),
            loaded("unique-one", Provenance::User, Status::Enabled),
        ];
        disable_duplicate_names(&mut behaviours);

        for lb in &behaviours {
            if lb.behaviour.manifest.name == "auto-tag-by-project" {
                assert_eq!(lb.status, Status::Disabled(DisableReason::DuplicateName));
            } else {
                assert_eq!(lb.status, Status::Enabled); // the unique one is untouched
            }
        }
    }
}
