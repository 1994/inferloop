pub mod cuda;
pub mod metal;
use crate::DType;
pub use cuda::{CudaRequirements, NvidiaArchitecture, NvidiaCapabilities};
use infer_core::{DeviceId, Error, Result};
pub use metal::{MetalCapabilities, MetalRequirements};
use serde::{Deserialize, Serialize};

/// Execution API is part of compilation compatibility, independently of dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cuda,
    Metal,
}
/// The variant is the backend discriminator. Backend-specific attributes cannot
/// coexist with a different execution API or be omitted from a CUDA device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "capabilities",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum DeviceBackend {
    Cuda(NvidiaCapabilities),
    Metal(MetalCapabilities),
}
impl DeviceBackend {
    #[must_use]
    pub const fn kind(&self) -> BackendKind {
        match self {
            Self::Cuda(_) => BackendKind::Cuda,
            Self::Metal(_) => BackendKind::Metal,
        }
    }
}
/// Checkpoint container a backend can load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointFormat {
    /// Hugging Face Safetensors.
    Safetensors,
    /// GGUF container with GGML block layouts.
    Gguf,
    /// MLX package.
    Mlx,
}

impl CheckpointFormat {
    /// Stable label used in errors and diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
            Self::Gguf => "gguf",
            Self::Mlx => "mlx",
        }
    }
}

/// Weight storage/compute encoding a backend can execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightEncoding {
    /// F32/BF16/F16 dense weights.
    Dense,
    /// FP8 channel or block scales.
    Fp8,
    /// NVFP4 block scales.
    Nvfp4,
    /// GGML `Q4_K` grouped integers.
    GroupedInt4,
    /// GGML `Q6_K` grouped integers.
    GroupedInt6,
    /// MLX affine quantization.
    MlxAffine,
}

impl WeightEncoding {
    /// Stable label used in errors and diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
            Self::GroupedInt4 => "grouped-int4",
            Self::GroupedInt6 => "grouped-int6",
            Self::MlxAffine => "mlx-affine",
        }
    }
}

/// Checkpoint containers and weight encodings one backend advertises.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendFormats {
    /// Containers the host loader may present to this backend.
    pub checkpoints: &'static [CheckpointFormat],
    /// Encodings the kernels can execute.
    pub encodings: &'static [WeightEncoding],
}

impl BackendFormats {
    /// Whether this backend can load `format`.
    #[must_use]
    pub fn supports_format(self, format: CheckpointFormat) -> bool {
        self.checkpoints.contains(&format)
    }

    /// Whether this backend can execute `encoding`.
    #[must_use]
    pub fn supports_encoding(self, encoding: WeightEncoding) -> bool {
        self.encodings.contains(&encoding)
    }

    /// Require a checkpoint format and every listed encoding.
    /// # Errors
    /// Rejects a container or encoding this backend does not declare.
    pub fn require(self, format: CheckpointFormat, encodings: &[WeightEncoding]) -> Result<()> {
        if !self.supports_format(format) {
            return Err(Error::unsupported(format!(
                "backend does not load {} checkpoints",
                format.label()
            )));
        }
        if let Some(encoding) = encodings
            .iter()
            .find(|encoding| !self.supports_encoding(**encoding))
        {
            return Err(Error::unsupported(format!(
                "backend does not execute {} weights",
                encoding.label()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceCapabilities {
    pub backend: DeviceBackend,
    pub device: DeviceId,
    pub compute_dtypes: Vec<DType>,
    pub memory_bytes: u64,
    pub unified_memory: bool,
    pub profiling: bool,
    /// Speculative decode support the backend can actually execute, so the engine never
    /// discovers a limitation by falling back silently per step.
    #[serde(default)]
    pub speculation: SpeculationCapability,
}
/// Speculative decode support a backend reports to the engine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculationCapability {
    /// Draft candidates the backend verifies per decode step; zero disables speculation.
    pub draft_depth: usize,
    /// Whether only greedy sampling is supported, because rejection needs residual sampling.
    pub greedy_only: bool,
    /// Whether a media prompt can be prefilled while draft proposals are enabled.
    ///
    /// Drafts are verified against target rows the media executor cannot produce, so a backend
    /// that binds both declares the combination here instead of failing a submitted request.
    #[serde(default)]
    pub media_prefill: bool,
}
impl SpeculationCapability {
    /// Whether one request may combine media input with speculative draft proposals.
    ///
    /// Without a loaded draft no proposal is verified, so media prefill runs on its own and the
    /// combination is trivially supported.
    #[must_use]
    pub const fn allows_media_prefill(&self) -> bool {
        self.draft_depth == 0 || self.media_prefill
    }
}
impl DeviceCapabilities {
    #[must_use]
    pub const fn backend_kind(&self) -> BackendKind {
        self.backend.kind()
    }

    /// # Errors
    /// Returns an unsupported error if backend, architecture, features, precision, or memory requirements are unavailable.
    pub fn require(&self, required: &CapabilityRequirements) -> Result<()> {
        let backend_matches = match (&self.backend, &required.backend) {
            (_, None) => true,
            (DeviceBackend::Cuda(c), Some(BackendRequirements::Cuda(r))) => c.satisfies(r),
            (DeviceBackend::Metal(c), Some(BackendRequirements::Metal(r))) => c.satisfies(r),
            _ => false,
        };
        if !backend_matches
            || required
                .compute_dtypes
                .iter()
                .any(|d| !self.compute_dtypes.contains(d))
        {
            return Err(Error::unsupported(
                "device does not meet capability requirements",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "requirements",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum BackendRequirements {
    Cuda(CudaRequirements),
    Metal(MetalRequirements),
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityRequirements {
    pub compute_dtypes: Vec<DType>,
    pub backend: Option<BackendRequirements>,
}

impl BackendKind {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
    #[must_use]
    pub const fn is_device(self) -> bool {
        match self {
            Self::Cuda | Self::Metal => true,
        }
    }
}
