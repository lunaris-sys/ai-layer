//! Shared infrastructure for the Lunaris AI layer.
//!
//! Defines the [`provider::AIProvider`] trait and the surface area that
//! both `ai-daemon` and `ai-agent` build on: the routing engine, the MCP
//! client wrapper, the audit-log producer, the capability gate, the
//! content-origin tagging API, and the two-step Cypher pipeline. See
//! `docs/architecture/phase-9-plan.md` for the full mapping.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod audit;
pub mod capability;
pub mod cypher;
pub mod graph_query;
pub mod graph_schema;
pub mod mcp;
pub mod pipeline;
pub mod provider;
pub mod routing;
pub mod tagging;
