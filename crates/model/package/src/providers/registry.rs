//! Architecture to provider resolution, so opening a package never names a model family.
use crate::providers::llama::LlamaProvider;
use crate::providers::qwen::QwenProvider;
use infer_core::{Error, Result};
use infer_spi::{ModelProvider, ProviderMetadata};
use serde::Deserialize;
use std::sync::{Arc, OnceLock};

/// Architecture hints of one Hugging Face configuration, including a nested text backbone. Every
/// hint is optional: exporters emit `null` for fields they do not set.
#[derive(Deserialize)]
struct ArchitectureHints {
    #[serde(default)]
    architectures: Option<Vec<String>>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    text_config: Option<Box<Self>>,
}

impl ArchitectureHints {
    /// Most specific first: explicit `architectures`, then `model_type`, then the text backbone.
    fn collect(&self, output: &mut Vec<String>) {
        if let Some(names) = &self.architectures {
            for name in names {
                push_unique(output, name);
            }
        }
        if let Some(name) = &self.model_type {
            push_unique(output, name);
        }
        if let Some(text) = &self.text_config {
            text.collect(output);
        }
    }
}

fn push_unique(output: &mut Vec<String>, name: &str) {
    if !output.iter().any(|existing| existing == name) {
        output.push(name.to_owned());
    }
}

/// Model providers available to one process, keyed by the architecture names they claim.
///
/// A registry is data: the composition root decides which providers exist. [`ModelRegistry::builtin`]
/// carries the families compiled into this crate, and [`ModelRegistry::register`] adds another one
/// without touching core dispatch.
#[derive(Default)]
pub struct ModelRegistry {
    providers: Vec<Arc<dyn ModelProvider>>,
}

impl ModelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry with the providers compiled into this crate.
    #[must_use]
    pub fn builtin() -> Self {
        let mut registry = Self::new();
        // The built-in list is fixed at compile time, so it cannot conflict with itself.
        registry.providers.push(Arc::new(QwenProvider));
        registry.providers.push(Arc::new(LlamaProvider));
        registry
    }

    /// Add one provider after the built-ins.
    ///
    /// # Errors
    /// Rejects invalid metadata, a provider that claims no architecture, or an architecture that
    /// another registered provider already claims.
    pub fn register(&mut self, provider: Arc<dyn ModelProvider>) -> Result<()> {
        provider.metadata().validate()?;
        if provider.architectures().is_empty() {
            return Err(Error::invalid("model provider claims no architecture"));
        }
        for claimed in provider.architectures() {
            if let Some(existing) = self
                .providers
                .iter()
                .find(|registered| registered.architectures().contains(claimed))
            {
                return Err(Error::invalid(format!(
                    "model architecture {claimed} is already provided by {}",
                    existing.metadata().name
                )));
            }
        }
        self.providers.push(provider);
        Ok(())
    }

    /// Provider that claims one of the configuration's architecture hints.
    ///
    /// # Errors
    /// Rejects malformed configuration, and configurations no registered provider claims.
    pub fn resolve(&self, config: &[u8]) -> Result<Arc<dyn ModelProvider>> {
        let hints: ArchitectureHints =
            serde_json::from_slice(config).map_err(|error| Error::invalid(error.to_string()))?;
        let mut names = Vec::new();
        hints.collect(&mut names);
        self.providers
            .iter()
            .find(|provider| {
                names
                    .iter()
                    .any(|name| provider.architectures().contains(&name.as_str()))
            })
            .cloned()
            .ok_or_else(|| {
                Error::unsupported(format!(
                    "no model provider for architecture {names:?}; implement \
                     infer_spi::ModelProvider and register it"
                ))
            })
    }

    /// Registered providers, for diagnostics.
    #[must_use]
    pub fn providers(&self) -> Vec<ProviderMetadata> {
        self.providers
            .iter()
            .map(|provider| provider.metadata())
            .collect()
    }
}

/// Process-wide registry with the built-in providers.
///
/// Callers that add a model build their own [`ModelRegistry`] and open with
/// `QuantizedPackage::open_with` or `ModelPackage::open_with`.
#[must_use]
pub fn default_registry() -> &'static ModelRegistry {
    static REGISTRY: OnceLock<ModelRegistry> = OnceLock::new();
    REGISTRY.get_or_init(ModelRegistry::builtin)
}

#[cfg(test)]
#[path = "../../tests/unit/providers_registry.rs"]
mod tests;
