//! Agent daemon library for the Lunaris AI layer.
//!
//! Hosts the D-Bus interface (`org.lunaris.AIAgent1`), the Event Bus
//! subscriber, and the per-behaviour trigger dispatcher. Disabling the
//! last enabled behaviour stops the binary entirely so an inactive
//! agent layer has no running process.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
