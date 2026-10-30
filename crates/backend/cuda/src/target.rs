//! Explicit hardware profiles. Architecture identifiers are not feature levels.
use infer_core::{Error, Result};
use infer_ir::DType;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum CudaTarget {
    HopperSm90,
    BlackwellSm120,
    Other(String),
}

impl CudaTarget {
    /// Whether the registered RGB8 image kernel has been qualified on this target.
    #[must_use]
    pub const fn supports_rgb8_preprocessing(&self) -> bool {
        matches!(self, Self::BlackwellSm120)
    }

    /// Compute dtypes this architecture executes natively.
    ///
    /// This is a capability, not a policy: which precision a checkpoint uses is declared by its
    /// model provider ([`infer_spi::PrecisionPolicy`]) and resolved against this answer. Declared
    /// requirements are validated against [`DeviceCapabilities`](infer_ir::DeviceCapabilities).
    #[must_use]
    pub const fn compute_dtypes(&self) -> &'static [DType] {
        match self {
            Self::BlackwellSm120 => &[
                DType::F32,
                DType::Bf16,
                DType::Fp8E4M3,
                DType::Fp8E5M2,
                DType::Fp4E2M1,
            ],
            Self::HopperSm90 => &[DType::F32, DType::Bf16, DType::Fp8E4M3, DType::Fp8E5M2],
            Self::Other(_) => &[DType::F32, DType::Bf16],
        }
    }

    /// Whether this architecture executes `dtype` natively.
    #[must_use]
    pub fn supports_compute(&self, dtype: DType) -> bool {
        self.compute_dtypes().contains(&dtype)
    }

    #[must_use]
    pub fn from_sm_name(name: &str) -> Self {
        match name {
            "sm_90" => Self::HopperSm90,
            "sm_120" => Self::BlackwellSm120,
            other => Self::Other(other.to_owned()),
        }
    }

    /// Gate this implementation's FP4 tile type before allocating launch output.
    /// # Errors
    /// Hopper cannot execute native FP4; other profiles require validation.
    pub fn require_native_nvfp4(&self) -> Result<()> {
        match self {
            Self::BlackwellSm120 => Ok(()),
            Self::HopperSm90 => Err(Error::unsupported(
                "Hopper/H200 has no native FP4 type; use the DecodedBf16 model-loading policy",
            )),
            Self::Other(name) => Err(Error::unsupported(format!(
                "native NVFP4 target {name} has not been validated by this backend"
            ))),
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/target.rs"]
mod tests;
