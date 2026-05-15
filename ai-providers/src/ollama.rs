//! Ollama provider adapter.
//!
//! Ollama exposes an OpenAI-compatible REST endpoint at
//! `POST {endpoint}/v1/chat/completions`. Reusing that shape keeps the
//! adapter trivially compatible with llama.cpp (which exposes the same
//! API) and with any local server that mimics OpenAI. See Foundation
//! §5.3 for the local-provider list.
//!
//! The adapter is a thin wrapper around `reqwest` plus tokio
//! `timeout`. It does not implement streaming; the ai-daemon
//! orchestrator owns streaming concerns and dispatches single-shot
//! completions through this layer.

use async_trait::async_trait;
use lunaris_ai_core::provider::{
    AIProvider, CompletionRequest, CompletionResponse, ProviderAudit, ProviderError,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default request timeout. Foundation does not pin a number here;
/// 60 s comfortably accommodates an 8B-parameter local model on
/// modest hardware while still failing closed on a stalled backend.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Configuration for an Ollama adapter instance.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Logical name used in routing rules and the audit log.
    pub name: String,
    /// Endpoint base URL (no trailing slash). Default Ollama install
    /// is `http://localhost:11434`.
    pub endpoint: String,
    /// Model identifier as known to Ollama, for example `llama3:8b`.
    pub model: String,
    /// Per-call timeout. Defaults to [`DEFAULT_TIMEOUT`] if `None`.
    pub timeout: Option<Duration>,
}

/// Ollama adapter implementing [`AIProvider`].
pub struct OllamaProvider {
    config: OllamaConfig,
    http: reqwest::Client,
}

impl OllamaProvider {
    /// Build a new adapter. The HTTP client is constructed up-front so
    /// connections can be pooled across calls.
    pub fn new(config: OllamaConfig) -> Result<Self, ProviderError> {
        let timeout = config.timeout.unwrap_or(DEFAULT_TIMEOUT);
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|err| ProviderError::Internal(err.to_string()))?;
        Ok(Self { config, http })
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.config.endpoint.trim_end_matches('/'))
    }

    fn tags_url(&self) -> String {
        format!("{}/api/tags", self.config.endpoint.trim_end_matches('/'))
    }
}

#[async_trait]
impl AIProvider for OllamaProvider {
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

        let response = self
            .http
            .post(self.chat_completions_url())
            .json(&body)
            .send()
            .await
            .map_err(|err| {
                if err.is_timeout() {
                    ProviderError::Timeout
                } else if err.is_connect() || err.is_request() {
                    ProviderError::Unavailable(err.to_string())
                } else {
                    ProviderError::Internal(err.to_string())
                }
            })?;

        let status = response.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ProviderError::RateLimited);
        }
        if status.is_server_error() {
            return Err(ProviderError::Unavailable(format!(
                "ollama returned HTTP {}",
                status.as_u16()
            )));
        }
        if !status.is_success() {
            return Err(ProviderError::Internal(format!(
                "ollama returned HTTP {}",
                status.as_u16()
            )));
        }

        let parsed: ChatResponse = response
            .json()
            .await
            .map_err(|err| ProviderError::Internal(format!("invalid response body: {err}")))?;

        let text = parsed
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
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
        // `GET /api/tags` is the cheapest authoritative liveness probe
        // Ollama exposes. The OpenAI-compat surface lacks a no-op
        // health endpoint.
        match self.http.get(self.tags_url()).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn name(&self) -> &str {
        &self.config.name
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn provider(server: &MockServer) -> OllamaProvider {
        OllamaProvider::new(OllamaConfig {
            name: "ollama-test".to_string(),
            endpoint: server.uri(),
            model: "llama3:8b".to_string(),
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("provider builds")
    }

    #[tokio::test]
    async fn complete_returns_first_choice_text_and_usage() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "choices": [{"message": {"role": "assistant", "content": "42"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}
            })))
            .mount(&server)
            .await;

        let p = provider(&server);
        let resp = p
            .complete(CompletionRequest {
                prompt: "what is the answer?".to_string(),
                extras: serde_json::json!({}),
            })
            .await
            .expect("complete ok");
        assert_eq!(resp.text, "42");
        assert_eq!(resp.audit.provider_name, "ollama-test");
        assert_eq!(resp.audit.model, "llama3:8b");
        assert_eq!(resp.audit.input_tokens, Some(3));
        assert_eq!(resp.audit.output_tokens, Some(1));
    }

    #[tokio::test]
    async fn http_429_maps_to_rate_limited() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429))
            .mount(&server)
            .await;

        let p = provider(&server);
        let err = p
            .complete(CompletionRequest {
                prompt: "x".to_string(),
                extras: serde_json::json!({}),
            })
            .await
            .expect_err("expected error");
        assert!(matches!(err, ProviderError::RateLimited), "got {err:?}");
    }

    #[tokio::test]
    async fn http_5xx_maps_to_unavailable() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let p = provider(&server);
        let err = p
            .complete(CompletionRequest {
                prompt: "x".to_string(),
                extras: serde_json::json!({}),
            })
            .await
            .expect_err("expected error");
        assert!(matches!(err, ProviderError::Unavailable(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn available_probes_api_tags() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": []
            })))
            .mount(&server)
            .await;

        let p = provider(&server);
        assert!(p.available().await);
    }

    #[tokio::test]
    async fn available_returns_false_when_endpoint_missing() {
        let p = OllamaProvider::new(OllamaConfig {
            name: "ollama-missing".to_string(),
            // Reserved port that nothing should be listening on.
            endpoint: "http://127.0.0.1:1".to_string(),
            model: "llama3:8b".to_string(),
            timeout: Some(Duration::from_millis(200)),
        })
        .expect("provider builds");
        assert!(!p.available().await);
    }

    #[tokio::test]
    async fn name_matches_config() {
        let server = MockServer::start().await;
        let p = provider(&server);
        assert_eq!(p.name(), "ollama-test");
    }
}
