//! Provider adapters for the Lunaris AI layer.
//!
//! Each adapter in this crate implements
//! [`lunaris_ai_core::provider::AIProvider`] for a concrete backend.
//! The shipped adapters cover the four canonical providers named in
//! Foundation §5.3 (Ollama, llama.cpp, Anthropic, OpenAI). Additional
//! backends plug in by implementing the same trait.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod ollama;
pub mod proxied;
