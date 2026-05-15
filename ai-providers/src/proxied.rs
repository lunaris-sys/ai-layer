//! Proxied provider adapter.
//!
//! Per Foundation §8.4.6 every outbound completion call must transit
//! `ai-proxy`. This adapter is the only [`AIProvider`] implementation
//! `ai-daemon` and `ai-agent` should use in production: it forwards
//! requests to the proxy over D-Bus instead of opening sockets
//! directly. The proxy looks up the catalogued endpoint, enforces
//! the hostname allowlist, and emits an audit record. Reaching a
//! provider any other way would bypass that boundary.
//!
//! ## Wire shape
//!
//! For Phase 9-α we use the OpenAI-compatible chat/completions
//! request and response shapes. All four canonical providers
//! (Ollama, llama.cpp, Anthropic, OpenAI) either expose this shape
//! natively (OpenAI), through a documented compatibility layer
//! (Ollama, llama.cpp, Anthropic via `/v1/messages` compatibility),
//! or via a thin transcoder on the proxy side. Backend-specific
//! shaping is a Phase 9-β concern; the goal here is to satisfy the
//! trust boundary, not to flex every backend's native API.

use async_trait::async_trait;
use lunaris_ai_core::provider::{
    AIProvider, CompletionRequest, CompletionResponse, ProviderAudit, ProviderError,
};
use serde::{Deserialize, Serialize};
use zbus::Connection;

/// Configuration for a [`ProxiedProvider`] instance.
#[derive(Debug, Clone)]
pub struct ProxiedConfig {
    /// Logical name used in routing rules. Must match a catalogued
    /// provider on the proxy side, otherwise the proxy returns
    /// `unknown-provider`.
    pub name: String,
    /// Model identifier reported back in audit records.
    pub model: String,
    /// Capability token presented to the proxy. The proxy records it
    /// in its audit log; Phase 9-γ S15 validates it against the
    /// caller's identity.
    pub audit_token: String,
}

/// D-Bus client for `org.lunaris.AIProxy1`.
///
/// Cheap to clone (`zbus::Proxy` is internally reference-counted).
pub struct ProxyAIClient {
    proxy: zbus::Proxy<'static>,
}

impl ProxyAIClient {
    /// Build the proxy over an existing D-Bus connection.
    ///
    /// The caller must pass the same connection it owns its
    /// well-known bus name on. The proxy authorises a forward by
    /// checking that the calling connection owns `org.lunaris.AI1`
    /// (or `org.lunaris.AIAgent1`); a forward sent from a second,
    /// nameless connection of the same process would be rejected.
    pub async fn with_connection(connection: &Connection) -> Result<Self, ProviderError> {
        let proxy = zbus::Proxy::new(
            connection,
            "org.lunaris.AIProxy1",
            "/org/lunaris/AIProxy1",
            "org.lunaris.AIProxy1",
        )
        .await
        .map_err(|err| ProviderError::Unavailable(format!("ai-proxy proxy: {err}")))?;
        Ok(Self { proxy })
    }

    /// Open a fresh session-bus connection and build the proxy on it.
    ///
    /// Only safe for a caller that does not own a well-known name it
    /// must forward as. A daemon that owns `org.lunaris.AI1` must use
    /// [`with_connection`](Self::with_connection) with that same
    /// connection instead.
    pub async fn connect() -> Result<Self, ProviderError> {
        let connection = Connection::session().await.map_err(|err| {
            ProviderError::Unavailable(format!("ai-proxy session bus: {err}"))
        })?;
        Self::with_connection(&connection).await
    }

    /// Invoke `forward_completion` on the proxy.
    pub async fn forward(
        &self,
        provider_name: &str,
        body_json: &str,
        audit_token: &str,
    ) -> Result<ProxyForwardResponse, ProviderError> {
        let reply: String = self
            .proxy
            .call(
                "forward_completion",
                &(provider_name, body_json, audit_token),
            )
            .await
            .map_err(map_zbus_error)?;
        serde_json::from_str(&reply)
            .map_err(|err| ProviderError::Internal(format!("proxy reply not json: {err}")))
    }
}

/// Decoded response from the proxy.
#[derive(Debug, Clone, Deserialize)]
pub struct ProxyForwardResponse {
    /// HTTP status the proxy observed at the upstream.
    pub upstream_status: u16,
    /// Upstream response body.
    pub body: String,
}

/// [`AIProvider`] that forwards every call through `ai-proxy`.
pub struct ProxiedProvider {
    config: ProxiedConfig,
    client: ProxyAIClient,
}

impl ProxiedProvider {
    /// Build an adapter that forwards over `connection`.
    ///
    /// `connection` must be the connection the daemon owns its
    /// well-known bus name on, so the proxy's peer-auth sees the
    /// forward as coming from that name's owner.
    pub async fn with_connection(
        config: ProxiedConfig,
        connection: &Connection,
    ) -> Result<Self, ProviderError> {
        let client = ProxyAIClient::with_connection(connection).await?;
        Ok(Self { config, client })
    }

    /// Build from an already-connected client. Used in tests.
    pub fn with_client(config: ProxiedConfig, client: ProxyAIClient) -> Self {
        Self { config, client }
    }
}

#[async_trait]
impl AIProvider for ProxiedProvider {
    async fn complete(
        &self,
        req: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let body = ChatRequest {
            model: &self.config.model,
            messages: vec![ChatMessage {
                role: "user",
                content: &req.prompt,
            }],
            stream: false,
        };
        let body_json = serde_json::to_string(&body)
            .map_err(|err| ProviderError::Internal(format!("body serialise: {err}")))?;

        let resp = self
            .client
            .forward(&self.config.name, &body_json, &self.config.audit_token)
            .await?;

        if resp.upstream_status == 429 {
            return Err(ProviderError::RateLimited);
        }
        if (500..600).contains(&resp.upstream_status) {
            return Err(ProviderError::Unavailable(format!(
                "upstream returned HTTP {}",
                resp.upstream_status
            )));
        }
        if !(200..300).contains(&resp.upstream_status) {
            return Err(ProviderError::Internal(format!(
                "upstream returned HTTP {}",
                resp.upstream_status
            )));
        }

        let parsed: ChatResponse = serde_json::from_str(&resp.body).map_err(|err| {
            ProviderError::Internal(format!("upstream body parse: {err}"))
        })?;
        let text = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        Ok(CompletionResponse {
            text,
            audit: ProviderAudit {
                provider_name: self.config.name.clone(),
                model: self.config.model.clone(),
                input_tokens: parsed.usage.as_ref().map(|u| u.prompt_tokens),
                output_tokens: parsed.usage.map(|u| u.completion_tokens),
            },
        })
    }

    async fn available(&self) -> bool {
        // No cheap probe through the proxy; assume the proxy is up if
        // its D-Bus client constructed. Phase 9-γ adds a periodic
        // liveness signal on the proxy side.
        true
    }

    fn name(&self) -> &str {
        &self.config.name
    }
}

fn map_zbus_error(err: zbus::Error) -> ProviderError {
    // `org.lunaris.AIProxy1.<Code>`-style errors are surfaced through
    // zbus::fdo::Error. We map them onto the AIProvider error taxonomy
    // so the caller does not need to know about D-Bus specifics.
    if let zbus::Error::FDO(ref fdo_err) = err {
        match fdo_err.as_ref() {
            zbus::fdo::Error::AccessDenied(detail) => {
                return ProviderError::Unavailable(format!("ai-proxy denied: {detail}"));
            }
            zbus::fdo::Error::InvalidArgs(detail) => {
                return ProviderError::Internal(format!("ai-proxy invalid args: {detail}"));
            }
            _ => {}
        }
    }
    ProviderError::Unavailable(format!("ai-proxy call: {err}"))
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

#[derive(Deserialize)]
struct ChatUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}
