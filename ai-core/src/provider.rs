//! Provider-agnostic AI completion interface.
//!
//! Mirrors Foundation §5.3 Listing 5. The Lunaris AI daemon stays
//! unaware of the concrete backend in use; the routing engine maps a
//! request onto a provider, and the provider handles the actual call.

use async_trait::async_trait;

/// A single round of input to a provider.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// The fully constructed prompt with content-origin tags applied.
    pub prompt: String,
    /// Provider-specific extras (model overrides, temperature, max_tokens).
    /// Kept opaque at this layer; adapters interpret these.
    pub extras: serde_json::Value,
}

/// A single round of output from a provider.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The provider's textual response.
    pub text: String,
    /// Audit metadata captured at the provider boundary.
    pub audit: ProviderAudit,
}

/// Audit metadata emitted alongside every completion.
#[derive(Debug, Clone)]
pub struct ProviderAudit {
    /// Name of the provider that served this completion.
    pub provider_name: String,
    /// Model identifier as reported by the provider.
    pub model: String,
    /// Number of input tokens billed (if reported).
    pub input_tokens: Option<u32>,
    /// Number of output tokens billed (if reported).
    pub output_tokens: Option<u32>,
}

/// Errors that can occur inside a provider call.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProviderError {
    /// The provider is currently unreachable.
    #[error("provider unavailable: {0}")]
    Unavailable(String),
    /// The provider exceeded its allotted time budget.
    #[error("provider timed out")]
    Timeout,
    /// The provider rate-limited the request.
    #[error("provider rate-limited")]
    RateLimited,
    /// An internal provider error not classified above.
    #[error("provider internal error: {0}")]
    Internal(String),
}

/// Provider-agnostic completion interface.
///
/// Each adapter in `ai-providers` implements this trait. The routing
/// engine selects a concrete provider at request time.
#[async_trait]
pub trait AIProvider: Send + Sync {
    /// Execute a single completion.
    async fn complete(
        &self,
        req: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError>;

    /// Probe whether the provider is reachable right now.
    async fn available(&self) -> bool;

    /// Human-readable identifier used in routing rules and audit log entries.
    fn name(&self) -> &str;
}
