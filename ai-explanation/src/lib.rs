//! System Explanation Mode for the Lunaris AI layer.
//!
//! Implements Foundation §5.8: queries the Event Bus snapshot plus the
//! Knowledge Graph state and produces a plain-language summary of what
//! the system is doing right now. Used by Settings, by Waypointer's
//! "What is my computer doing?" query, and by the Companion App.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
