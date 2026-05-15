//! `lunaris-ai-agent` entry point.
//!
//! Hosts autonomous behaviours that the user has explicitly enabled
//! through Settings. Each behaviour is a separate toggle; the binary
//! is intended to be started only when at least one is enabled
//! (Foundation §5.5). The Phase 9-δ S20 sub-sprint wires the actual
//! Event Bus subscriber and per-behaviour triggers on top of this
//! scaffold.

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    tracing::info!("lunaris-ai-agent: phase 9-α S1 scaffold, no daemon logic yet");
    Ok(())
}
