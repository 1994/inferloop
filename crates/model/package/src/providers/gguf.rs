//! GGUF adapters, keyed by the name a file declares, so the GGUF platform never names a family.
use crate::providers::dflash2::gguf::DFLASH2;
use crate::providers::qwen::gguf::{QWEN3VL, QWEN3VL_MERGER, QWEN35};
use crate::storage::gguf::{GgufArchitecture, GgufProjector};
use infer_core::{Error, Result};

/// The text architectures (`general.architecture`) and mmproj projectors (`clip.projector_type`)
/// one registry reads.
#[derive(Default)]
pub struct GgufAdapters {
    architectures: Vec<&'static dyn GgufArchitecture>,
    projectors: Vec<&'static dyn GgufProjector>,
}

impl GgufAdapters {
    /// The adapters compiled into this crate.
    #[must_use]
    pub fn builtin() -> Self {
        Self {
            architectures: vec![&QWEN3VL, &QWEN35, &DFLASH2],
            projectors: vec![&QWEN3VL_MERGER],
        }
    }

    /// Add a text architecture.
    /// # Errors
    /// Rejects a name another adapter already reads.
    pub fn register_architecture(&mut self, adapter: &'static dyn GgufArchitecture) -> Result<()> {
        if self
            .architectures
            .iter()
            .any(|known| known.name() == adapter.name())
        {
            return Err(Error::invalid(format!(
                "GGUF architecture {} is already registered",
                adapter.name()
            )));
        }
        self.architectures.push(adapter);
        Ok(())
    }

    /// Add an mmproj projector.
    /// # Errors
    /// Rejects a name another adapter already reads.
    pub fn register_projector(&mut self, adapter: &'static dyn GgufProjector) -> Result<()> {
        if self
            .projectors
            .iter()
            .any(|known| known.name() == adapter.name())
        {
            return Err(Error::invalid(format!(
                "GGUF projector {} is already registered",
                adapter.name()
            )));
        }
        self.projectors.push(adapter);
        Ok(())
    }

    /// The adapter for a `general.architecture` value.
    /// # Errors
    /// Rejects a name no adapter reads, naming the ones that exist.
    pub fn architecture(&self, name: &str) -> Result<&'static dyn GgufArchitecture> {
        self.architectures
            .iter()
            .copied()
            .find(|adapter| adapter.name() == name)
            .ok_or_else(|| {
                unknown(
                    "architecture",
                    name,
                    self.architectures.iter().map(|a| a.name()),
                )
            })
    }

    /// The adapter for a `clip.projector_type` value.
    /// # Errors
    /// Rejects a name no adapter reads, naming the ones that exist.
    pub fn projector(&self, name: &str) -> Result<&'static dyn GgufProjector> {
        self.projectors
            .iter()
            .copied()
            .find(|adapter| adapter.name() == name)
            .ok_or_else(|| unknown("projector", name, self.projectors.iter().map(|a| a.name())))
    }
}

fn unknown<'a>(kind: &str, name: &str, known: impl Iterator<Item = &'a str>) -> Error {
    let known: Vec<_> = known.collect();
    Error::unsupported(format!(
        "GGUF {kind} {name} has no adapter (known: {}); implement the GGUF adapter trait and \
         register it",
        known.join(", ")
    ))
}
