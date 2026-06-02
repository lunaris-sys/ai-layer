//! Integration tests that run the real sandboxed worker binary and
//! verify the isolation actually holds: no filesystem, no network, and
//! correct text extraction across the subprocess boundary.

#![cfg(target_os = "linux")]

use std::path::Path;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_lunaris-doc-sandbox");

#[test]
fn sandbox_denies_filesystem_access() {
    // The worker self-tests after sandboxing: exit 0 means opening a
    // file was correctly denied by Landlock.
    let status = Command::new(BIN)
        .env("LUNARIS_SANDBOX_SELFTEST", "fs")
        .status()
        .expect("spawn worker");
    assert!(
        status.success(),
        "filesystem access must be denied inside the sandbox (worker exit {status})"
    );
}

#[test]
fn sandbox_denies_network_access() {
    // Exit 0 means socket creation / connect was correctly denied by
    // seccomp.
    let status = Command::new(BIN)
        .env("LUNARIS_SANDBOX_SELFTEST", "net")
        .status()
        .expect("spawn worker");
    assert!(
        status.success(),
        "network access must be denied inside the sandbox (worker exit {status})"
    );
}

#[test]
fn sandbox_denies_process_creation() {
    // Exit 0 means a raw fork was denied by seccomp, so no descendant
    // can outlive the worker holding stdout open.
    let status = Command::new(BIN)
        .env("LUNARIS_SANDBOX_SELFTEST", "fork")
        .status()
        .expect("spawn worker");
    assert!(
        status.success(),
        "process creation must be denied inside the sandbox (worker exit {status})"
    );
}

#[test]
fn sandbox_denies_signalling_other_processes() {
    let status = Command::new(BIN)
        .env("LUNARIS_SANDBOX_SELFTEST", "signal")
        .status()
        .expect("spawn worker");
    assert!(
        status.success(),
        "signalling other processes must be denied (worker exit {status})"
    );
}

#[test]
fn sandbox_denies_path_metadata_probing() {
    let status = Command::new(BIN)
        .env("LUNARIS_SANDBOX_SELFTEST", "stat")
        .status()
        .expect("spawn worker");
    assert!(
        status.success(),
        "path-based stat must be denied (worker exit {status})"
    );
}

#[test]
fn sandbox_denies_path_truncation() {
    // A throwaway file the worker is asked to truncate. If the sandbox
    // works it stays intact; the worst case if it were broken is this
    // disposable file being emptied, never a real one.
    let path = std::env::temp_dir().join(format!(
        "lunaris-trunc-probe-{}",
        std::process::id()
    ));
    std::fs::write(&path, b"keep me intact").expect("write probe file");

    let status = Command::new(BIN)
        .env("LUNARIS_SANDBOX_SELFTEST", "truncate")
        .env("LUNARIS_TRUNCATE_TARGET", &path)
        .status()
        .expect("spawn worker");

    let after = std::fs::read(&path).unwrap_or_default();
    let _ = std::fs::remove_file(&path);
    assert!(
        status.success(),
        "path truncation must be denied inside the sandbox (worker exit {status})"
    );
    assert_eq!(after, b"keep me intact", "the probe file must be untouched");
}

#[test]
fn parse_document_extracts_and_sanitises_through_the_subprocess() {
    let raw = b"Meeting notes:\nbudget approved.\x1b[31m hidden ansi \x1b[0m\x07 done.";
    let text = lunaris_ai_sandbox::parse_document(Path::new(BIN), raw)
        .expect("parse should succeed");
    assert!(!text.contains('\x1b'), "ANSI escapes must be stripped");
    assert!(!text.contains('\x07'), "control chars must be stripped");
    assert!(text.contains("Meeting notes"));
    assert!(text.contains("budget approved"));
    assert!(text.contains("done."));
}

#[test]
fn parse_document_handles_empty_input() {
    let text = lunaris_ai_sandbox::parse_document(Path::new(BIN), b"").expect("ok");
    assert_eq!(text, "");
}
