//! Prompt-injection classifier for the Lunaris AI layer.
//!
//! Loads Meta's Prompt-Guard-86M as an ONNX export and runs CPU
//! inference per Phase 9-γ S17. The classifier produces a numeric
//! injection score per input; threshold + downstream policy
//! (block, warn, require-confirm) is configurable.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
