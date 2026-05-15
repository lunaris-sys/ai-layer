//! Outbound proxy library for the Lunaris AI layer.
//!
//! Foundation §8.4.6 requires that AI provider traffic exit the host
//! through a dedicated daemon. The proxy is the only component
//! permitted to make outbound HTTPS connections to AI provider
//! endpoints; both `ai-daemon` and `ai-agent` run with
//! `PrivateNetwork=true` in their systemd units and reach the proxy
//! over the session D-Bus.
//!
//! This crate exposes the policy core (allowlist + audit emission +
//! forwarder trait) as a reusable library so it can be unit-tested
//! without spinning up the full daemon. The binary in `main.rs`
//! plumbs the policy into a real reqwest-backed outbound layer plus
//! a zbus service.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod allowlist;
pub mod audit;
pub mod catalog;
pub mod forward;
pub mod peer_auth;
pub mod service;
