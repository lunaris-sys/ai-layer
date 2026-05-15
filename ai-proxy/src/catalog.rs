//! Trusted provider catalog.
//!
//! Foundation §8.4.6 requires that the proxy treat the *endpoint URL*
//! as proxy-owned configuration, never as caller input. If the
//! caller could supply an arbitrary URL and the proxy enforced only a
//! hostname allowlist, the proxy would become a POST gadget for any
//! port/path on an allowed host — Anthropic's `/v1/messages` could
//! turn into `/v1/anything-the-attacker-wants`.
//!
//! The catalog maps a provider *name* (the same key used in
//! `ai-routing.toml`) onto the exact `(endpoint_url, backend)` the
//! proxy will reach. Callers identify their target by name; the URL
//! comes from this catalog.

use std::collections::HashMap;

use serde::Deserialize;

/// Catalogued provider entry.
#[derive(Debug, Clone, Deserialize)]
pub struct CatalogEntry {
    /// Full upstream endpoint URL (scheme + host + path). The proxy
    /// will POST `body_json` to this URL verbatim.
    pub endpoint_url: String,
    /// Backend identifier (`ollama`, `llamacpp`, `anthropic`,
    /// `openai`). Phase 9-α uses this only for logging; Phase 9-β
    /// uses it to dispatch backend-specific request shaping.
    pub backend: String,
}

/// Trusted provider catalog.
#[derive(Debug, Clone, Default)]
pub struct ProviderCatalog {
    entries: HashMap<String, CatalogEntry>,
}

impl ProviderCatalog {
    /// Build a catalog from an explicit map.
    pub fn new(entries: HashMap<String, CatalogEntry>) -> Self {
        Self { entries }
    }

    /// The default Lunaris catalog.
    ///
    /// Phase 9-α ships only the local Ollama provider. The cloud
    /// providers (OpenAI, Anthropic) are deliberately absent: the
    /// proxy does not yet attach API-key authentication or
    /// backend-specific request shaping, so a cloud route would fail
    /// with a provider-side 401/400 rather than work. They are added
    /// in Phase 9-β/γ together with keyring-backed credentials. A
    /// half-working cloud entry would violate the "no stubs, no
    /// for-now" project rule, so it stays out until it functions.
    pub fn default_lunaris() -> Self {
        let mut entries = HashMap::new();
        entries.insert(
            "ollama-default".to_string(),
            CatalogEntry {
                endpoint_url: "http://localhost:11434/v1/chat/completions".to_string(),
                backend: "ollama".to_string(),
            },
        );
        Self::new(entries)
    }

    /// Look up a provider by name.
    pub fn get(&self, provider_name: &str) -> Option<&CatalogEntry> {
        self.entries.get(provider_name)
    }

    /// Iterator over the registered provider names. Used by
    /// `list_allowed_endpoints` on the D-Bus interface.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_catalog_ships_only_the_local_provider() {
        // Phase 9-α: cloud providers are intentionally absent until
        // keyring-backed auth lands.
        let cat = ProviderCatalog::default_lunaris();
        let names: Vec<&str> = cat.names().collect();
        assert_eq!(names, vec!["ollama-default"]);
    }

    #[test]
    fn lookup_returns_full_url() {
        let cat = ProviderCatalog::default_lunaris();
        let entry = cat.get("ollama-default").unwrap();
        assert_eq!(entry.endpoint_url, "http://localhost:11434/v1/chat/completions");
        assert_eq!(entry.backend, "ollama");
    }

    #[test]
    fn unknown_provider_returns_none() {
        let cat = ProviderCatalog::default_lunaris();
        assert!(cat.get("missing-provider").is_none());
    }
}
