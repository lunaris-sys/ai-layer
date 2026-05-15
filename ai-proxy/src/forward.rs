//! Outbound forwarder abstraction.
//!
//! The proxy service depends on a [`Forwarder`] trait so the policy
//! layer can be unit-tested without TCP. The daemon binary plugs in
//! the real reqwest-backed implementation; tests substitute a stub.

use async_trait::async_trait;

/// Outcome of a successful forward call.
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// Upstream HTTP status code.
    pub status: u16,
    /// Upstream response body as a UTF-8 string. The proxy does not
    /// touch the body content; framing parsing happens at the
    /// AI daemon layer.
    pub body: String,
}

/// Default cap on an upstream response body. LLM completions are
/// large but bounded; 8 MiB leaves generous headroom. A wedged or
/// hostile provider (including the allowlisted localhost endpoint)
/// cannot push the proxy into memory pressure beyond this.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// Errors that a [`Forwarder`] can return.
#[derive(Debug, thiserror::Error)]
pub enum ForwardError {
    /// Transport-level failure (connection refused, DNS error,
    /// TLS handshake, etc.).
    #[error("transport: {0}")]
    Transport(String),
    /// Upstream responded but the response body could not be read.
    #[error("body: {0}")]
    Body(String),
    /// Upstream response exceeded the size cap.
    #[error("upstream response exceeded the {limit}-byte cap")]
    ResponseTooLarge {
        /// The configured cap.
        limit: usize,
    },
}

/// Async outbound HTTP forwarder.
#[async_trait]
pub trait Forwarder: Send + Sync {
    /// POST `body_json` to `endpoint_url` and return the upstream
    /// response.
    async fn post(
        &self,
        endpoint_url: &str,
        body_json: &str,
    ) -> Result<ForwardResult, ForwardError>;
}

/// reqwest-backed forwarder. Built once at daemon startup so
/// connections can be pooled across calls.
pub struct ReqwestForwarder {
    http: reqwest::Client,
    max_response_bytes: usize,
}

impl ReqwestForwarder {
    /// Build the forwarder with the default response cap. Returns an
    /// error if the underlying reqwest client cannot be constructed
    /// (TLS init failure, etc.).
    ///
    /// Redirects are disabled at the transport layer: an allowed
    /// upstream that returns 30x must not be silently followed to a
    /// different host because that would bypass the allowlist and
    /// mis-attribute the audit record. Foundation §8.4.6 lists
    /// redirect-following as a known SSRF pivot.
    pub fn new() -> Result<Self, ForwardError> {
        Self::with_max_response(DEFAULT_MAX_RESPONSE_BYTES)
    }

    /// Build with an explicit response cap. Tests use a small cap to
    /// exercise the oversized-response path cheaply.
    pub fn with_max_response(max_response_bytes: usize) -> Result<Self, ForwardError> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|err| ForwardError::Transport(err.to_string()))?;
        Ok(Self {
            http,
            max_response_bytes,
        })
    }
}

#[async_trait]
impl Forwarder for ReqwestForwarder {
    async fn post(
        &self,
        endpoint_url: &str,
        body_json: &str,
    ) -> Result<ForwardResult, ForwardError> {
        let mut resp = self
            .http
            .post(endpoint_url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body_json.to_string())
            .send()
            .await
            .map_err(|err| ForwardError::Transport(err.to_string()))?;
        let status = resp.status().as_u16();

        // Reject early on a declared length over the cap, so an
        // honest `Content-Length` saves the streaming read entirely.
        if let Some(len) = resp.content_length() {
            if len as usize > self.max_response_bytes {
                return Err(ForwardError::ResponseTooLarge {
                    limit: self.max_response_bytes,
                });
            }
        }

        // Stream the body so a missing or lying `Content-Length`
        // cannot push unbounded data into memory.
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|err| ForwardError::Body(err.to_string()))?
        {
            if buf.len() + chunk.len() > self.max_response_bytes {
                return Err(ForwardError::ResponseTooLarge {
                    limit: self.max_response_bytes,
                });
            }
            buf.extend_from_slice(&chunk);
        }
        let body = String::from_utf8(buf)
            .map_err(|err| ForwardError::Body(format!("non-utf8 response: {err}")))?;
        Ok(ForwardResult { status, body })
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// Forwarder stub that records calls and returns a scripted
    /// response.
    #[derive(Clone)]
    pub struct StubForwarder {
        pub script: Arc<Mutex<Vec<Result<ForwardResult, ForwardError>>>>,
        pub calls: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl StubForwarder {
        pub fn new(script: Vec<Result<ForwardResult, ForwardError>>) -> Self {
            Self {
                script: Arc::new(Mutex::new(script)),
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl Forwarder for StubForwarder {
        async fn post(
            &self,
            endpoint_url: &str,
            body_json: &str,
        ) -> Result<ForwardResult, ForwardError> {
            self.calls
                .lock()
                .await
                .push((endpoint_url.to_string(), body_json.to_string()));
            let mut script = self.script.lock().await;
            if script.is_empty() {
                return Err(ForwardError::Transport("stub exhausted".to_string()));
            }
            script.remove(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn oversized_response_is_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("x".repeat(500)))
            .mount(&server)
            .await;
        let fwd = ReqwestForwarder::with_max_response(100).unwrap();
        let err = fwd
            .post(&format!("{}/x", server.uri()), "{}")
            .await
            .expect_err("must reject oversized body");
        assert!(matches!(
            err,
            ForwardError::ResponseTooLarge { limit: 100 }
        ));
    }

    #[tokio::test]
    async fn response_within_cap_passes_through() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&server)
            .await;
        let fwd = ReqwestForwarder::with_max_response(1024).unwrap();
        let result = fwd
            .post(&format!("{}/x", server.uri()), "{}")
            .await
            .expect("within cap");
        assert_eq!(result.status, 200);
        assert_eq!(result.body, "ok");
    }
}
