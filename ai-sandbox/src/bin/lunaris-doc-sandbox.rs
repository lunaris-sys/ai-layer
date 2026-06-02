//! The sandboxed document-parsing worker.
//!
//! Locks itself down with [`lunaris_ai_sandbox::apply_sandbox`] (no
//! new privileges, no filesystem, no network), then reads document
//! bytes from stdin, extracts inert text, and writes it to stdout. The
//! parent ([`lunaris_ai_sandbox::parse_document`]) feeds it the
//! untrusted document and reads back only the stripped text.
//!
//! A non-zero exit means no trustworthy text was produced; the parent
//! treats that as fail-closed.

#[cfg(target_os = "linux")]
fn main() {
    use std::io::{Read, Write};

    // Close every inherited descriptor beyond stdio before anything
    // else. Landlock does not revoke already-open fds, so a leaked
    // parent handle (a graph connection, a socket, a database file)
    // would otherwise survive into the worker and remain usable by
    // exploited parser code. Only stdin/stdout/stderr are kept.
    close_inherited_fds();

    // Lock down before touching any untrusted input. If the sandbox
    // cannot be installed we refuse to parse rather than run exposed.
    if let Err(e) = lunaris_ai_sandbox::apply_sandbox() {
        eprintln!("sandbox setup failed: {e}");
        std::process::exit(3);
    }

    // Self-test hook for the integration tests: after sandboxing, probe
    // a forbidden operation and exit 0 if it was correctly denied, 1 if
    // it unexpectedly succeeded. Never reads stdin in this mode.
    match std::env::var("LUNARIS_SANDBOX_SELFTEST").as_deref() {
        Ok("fs") => std::process::exit(probe_fs_denied()),
        Ok("net") => std::process::exit(probe_net_denied()),
        Ok("truncate") => std::process::exit(probe_truncate_denied()),
        Ok("fork") => std::process::exit(probe_fork_denied()),
        Ok("signal") => std::process::exit(probe_signal_denied()),
        Ok("stat") => std::process::exit(probe_stat_denied()),
        _ => {}
    }

    let mut input = Vec::new();
    if let Err(e) = std::io::stdin()
        .take((lunaris_ai_sandbox::MAX_BYTES as u64) + 1)
        .read_to_end(&mut input)
    {
        eprintln!("read stdin failed: {e}");
        std::process::exit(4);
    }

    match lunaris_ai_sandbox::extract_text(&input) {
        Ok(text) => {
            if std::io::stdout().write_all(text.as_bytes()).is_err() {
                std::process::exit(6);
            }
        }
        Err(e) => {
            eprintln!("extract failed: {e}");
            std::process::exit(5);
        }
    }
}

/// Close every inherited file descriptor above stderr, fail-closed.
///
/// Prefers `close_range`; if that is unavailable (very old kernel) or
/// fails, falls back to closing each descriptor up to the soft
/// `RLIMIT_NOFILE` so no leaked parent fd survives into the worker.
#[cfg(target_os = "linux")]
fn close_inherited_fds() {
    // SAFETY: close_range only closes descriptors in the range; it takes
    // no pointers and cannot corrupt memory.
    let rc = unsafe { libc::close_range(3, libc::c_uint::MAX, 0) };
    if rc == 0 {
        return;
    }
    // Fallback: read the *exact* set of open descriptors from
    // /proc/self/fd and close every one above stderr. A numeric ceiling
    // (RLIMIT_NOFILE) is not a reliable upper bound — a parent can open a
    // high fd and then lower the limit — so we enumerate instead. This
    // runs before the sandbox, so /proc is still readable. Collect first,
    // then close, so closing the directory's own fd mid-iteration cannot
    // truncate the listing.
    let rd = match std::fs::read_dir("/proc/self/fd") {
        Ok(rd) => rd,
        Err(e) => {
            // Cannot guarantee inherited fds are gone: fail closed.
            eprintln!("cannot enumerate file descriptors to close: {e}");
            std::process::exit(8);
        }
    };
    // Iterate strictly: a mid-stream readdir error or an unparsable name
    // means we cannot trust the listing, so fail closed rather than
    // proceed with a partial set of closed descriptors.
    let mut fds: Vec<i32> = Vec::new();
    for entry in rd {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("file descriptor enumeration failed mid-stream: {e}");
                std::process::exit(8);
            }
        };
        let name = entry.file_name();
        let Some(num) = name.to_str().and_then(|s| s.parse::<i32>().ok()) else {
            eprintln!("unparsable file-descriptor entry: {name:?}");
            std::process::exit(8);
        };
        if num > 2 {
            fds.push(num);
        }
    }
    for fd in fds {
        // SAFETY: close on an int fd; closing an already-closed fd just
        // returns EBADF, which is harmless.
        unsafe {
            libc::close(fd);
        }
    }
}

/// Returns 0 if opening a file is denied (sandbox working), 1 if it
/// unexpectedly succeeds. `/etc/passwd` reliably exists, so a failure
/// here is the Landlock denial, not a missing file.
#[cfg(target_os = "linux")]
fn probe_fs_denied() -> i32 {
    match std::fs::File::open("/etc/passwd") {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

/// Returns 0 if creating a socket is denied (sandbox working), 1 if it
/// unexpectedly succeeds. Probes the raw `socket()` syscall rather than
/// a TCP connect, so a pass is the seccomp denial and not an unrelated
/// connection-refused on an unsandboxed host.
#[cfg(target_os = "linux")]
fn probe_net_denied() -> i32 {
    // SAFETY: socket() with constant args creates or fails to create a
    // socket; it takes no pointers.
    let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
    if fd >= 0 {
        unsafe {
            libc::close(fd);
        }
        1
    } else {
        0
    }
}

/// Returns 0 if truncating a path is denied (Landlock v3+ / seccomp
/// working), 1 if it unexpectedly succeeds. The target is a throwaway
/// file supplied by the test via `LUNARIS_TRUNCATE_TARGET`, never a
/// system file: if the sandbox were broken the worst case is that this
/// disposable file is emptied, not real data loss.
#[cfg(target_os = "linux")]
fn probe_truncate_denied() -> i32 {
    let Ok(path) = std::env::var("LUNARIS_TRUNCATE_TARGET") else {
        return 0;
    };
    let Ok(cpath) = std::ffi::CString::new(path) else {
        return 0;
    };
    // SAFETY: truncate() reads the NUL-terminated path and takes no
    // other pointers.
    let rc = unsafe { libc::truncate(cpath.as_ptr(), 0) };
    if rc == 0 {
        1
    } else {
        0
    }
}

/// Returns 0 if creating a child process is denied (sandbox working), 1
/// if a fork unexpectedly succeeds. Probes the raw `fork` syscall; on
/// architectures without it, `clone`/`clone3` are blocked instead, so
/// there is nothing to probe and it reports denied.
#[cfg(target_os = "linux")]
fn probe_fork_denied() -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm"))]
    {
        // SAFETY: a raw fork; if it unexpectedly succeeds the child
        // exits immediately so it cannot run duplicate logic.
        let rc = unsafe { libc::syscall(libc::SYS_fork) };
        if rc < 0 {
            return 0;
        }
        if rc == 0 {
            unsafe { libc::_exit(0) };
        }
        1
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm")))]
    {
        0
    }
}

/// Returns 0 if signalling another process is denied (sandbox working),
/// 1 if it unexpectedly succeeds. Probes `kill(1, 0)` (an existence
/// check against init); under the allowlist `kill`/`tgkill` are not
/// permitted, so it must fail.
#[cfg(target_os = "linux")]
fn probe_signal_denied() -> i32 {
    // SAFETY: kill with signal 0 sends no signal; it only checks
    // permission/existence and takes no pointers.
    let rc = unsafe { libc::kill(1, 0) };
    if rc == 0 {
        1
    } else {
        0
    }
}

/// Returns 0 if path-based stat is denied (sandbox working), 1 if it
/// unexpectedly succeeds. `/etc/passwd` exists, so a failure is the
/// denial: the worker must not be able to probe arbitrary path metadata.
#[cfg(target_os = "linux")]
fn probe_stat_denied() -> i32 {
    match std::fs::metadata("/etc/passwd") {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("the document sandbox is only supported on Linux");
    std::process::exit(2);
}
