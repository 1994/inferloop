#[cfg(cuda_backend)]
mod cuda;
#[cfg(all(native_backend, not(cuda_backend)))]
#[path = "cuda_unavailable.rs"]
mod cuda;
#[cfg(metal_backend)]
mod metal;
#[cfg(native_backend)]
mod selected;
#[cfg(any(test, cuda_backend))]
mod speculation;
#[cfg(cuda_backend)]
use infer_backend_cuda::device::CudaDevice;
#[cfg(any(test, native_backend))]
use infer_core::{Error, Result};
use std::path::PathBuf;
use std::result::Result as ParseResult;
use std::str::FromStr;

#[cfg(native_backend)]
pub use selected::{SelectedBackend, load};
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendChoice {
    #[default]
    Auto,
    Metal,
    Cuda,
}
/// Explicit CLI draft-depth override; absence is resolved with the attention profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeTokens {
    Auto,
    Depth(usize),
}
impl FromStr for SpeculativeTokens {
    type Err = String;

    fn from_str(value: &str) -> ParseResult<Self, Self::Err> {
        if value == "auto" {
            Ok(Self::Auto)
        } else {
            value
                .parse::<usize>()
                .map(Self::Depth)
                .map_err(|_| "expected 'auto' or a non-negative draft token count".into())
        }
    }
}
/// Explicit target attention numerical profile.
///
/// `Canonical` retains F32 attention. The omitted-option resolver pairs `Bf16` with MTP
/// on supported native-FP4/FP8-KV packages; its initial context remains canonical.
/// `Fp16` remains an explicit experimental override.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum CudaAttentionProfile {
    #[default]
    Canonical,
    Bf16,
    Fp16,
}
impl CudaAttentionProfile {
    /// Stable startup-log name.
    #[cfg(cuda_backend)]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Bf16 => "bf16",
            Self::Fp16 => "fp16",
        }
    }
}
#[cfg(cuda_backend)]
impl From<CudaAttentionProfile> for infer_backend_cuda::AttentionProfile {
    fn from(value: CudaAttentionProfile) -> Self {
        match value {
            CudaAttentionProfile::Canonical => Self::Canonical,
            CudaAttentionProfile::Bf16 => Self::Bf16,
            CudaAttentionProfile::Fp16 => Self::Fp16,
        }
    }
}
#[cfg(metal_backend)]
pub fn metal_available() -> bool {
    metal::available()
}
#[cfg(not(metal_backend))]
pub const fn metal_available() -> bool {
    false
}
#[cfg(cuda_backend)]
pub fn cuda_available() -> bool {
    CudaDevice::new(0).is_ok()
}
#[cfg(not(cuda_backend))]
pub const fn cuda_available() -> bool {
    false
}
pub fn catalog() -> serde_json::Value {
    serde_json::json!({"primary_target":"cuda","supported":["cuda","metal"],"auto_priority":["cuda","metal"],
        "cuda":{"implemented":cfg!(cuda_backend),"available":cuda_available(),"execution":"resident CUDA graphs, synchronous provider", "build_feature":"cuda"},
        "metal":{"implemented":cfg!(metal_backend),"available":metal_available()},
        "testing_backends":{"enabled":false,"supported_for_deployment":false}})
}
/// When a backend captures its shared serving pool.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(
    not(native_backend),
    allow(
        dead_code,
        reason = "The portable CLI records the capture point even when no native backend reads it"
    )
)]
pub enum PoolCapture {
    /// At the first reservation that needs it, priced for that request's length.
    #[default]
    OnDemand,
    /// While loading, so the first concurrent requests batch at once.
    AtLoad,
}
#[derive(Debug, Clone)]
#[cfg_attr(
    not(native_backend),
    allow(
        dead_code,
        reason = "The portable CLI parses GPU allocation options even when no native execution backend is compiled"
    )
)]
pub struct Selection {
    pub kind: BackendChoice,
    pub num_gpu_blocks_override: Option<usize>,
    pub block_size: Option<usize>,
    pub max_num_batched_tokens: Option<usize>,
    /// Total context per sequence, prompt and output together (`--max-model-len`).
    pub max_model_len: Option<usize>,
    /// Service cap on generated tokens per request (`--max-output-tokens`).
    pub max_output_tokens: Option<usize>,
    /// Cap on sequences the scheduler runs at once (`--max-num-seqs`).
    pub max_num_seqs: Option<usize>,
    /// Name clients address this deployment by (`--served-model-name`).
    pub served_model_name: Option<String>,
    pub upload_staging_mib: Option<usize>,
    /// Omitted execution options select a qualified pair or fall back to canonical/depth0.
    pub num_speculative_tokens: Option<SpeculativeTokens>,
    /// Separate `DFlash2` draft package (`--draft-model`); excludes an MTP depth.
    pub draft_model: Option<PathBuf>,
    /// Explicit target attention profile; absence allows paired automatic selection.
    #[cfg(cuda_backend)]
    pub attention_profile: Option<CudaAttentionProfile>,
    pub gpu_memory_utilization: f64,
    /// Measure uncovered geometries; both modes reuse validated cached choices.
    pub autotune: bool,
    /// Serve text only (`--limit-mm-per-prompt` with every limit zero): skip the vision tower.
    pub text_only: bool,
    /// Per-prompt media item limits; all-zero limits are equivalent to `text_only`.
    pub media_limits: infer_models::MediaLimits,
    /// A server captures before it listens; one-shot commands and tuning workers on demand.
    pub pool_capture: PoolCapture,
}
#[cfg(any(test, native_backend))]
fn resolve(
    choice: BackendChoice,
    cuda_available: bool,
    metal_available: bool,
) -> Result<BackendChoice> {
    if choice != BackendChoice::Auto {
        return Ok(choice);
    }
    if cuda_available {
        return Ok(BackendChoice::Cuda);
    }
    if metal_available {
        return Ok(BackendChoice::Metal);
    }
    Err(Error::unsupported(
        "no supported GPU backend available (CUDA or Metal); CPU execution is test-only",
    ))
}
#[cfg(test)]
#[path = "../../tests/unit/backend_mod.rs"]
mod tests;

#[cfg(any(test, cuda_backend))]
mod defaults;
