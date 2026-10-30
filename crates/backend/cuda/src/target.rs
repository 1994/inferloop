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

/// The portable image program targets CUDA compute capability 7.0 or later.
const PORTABLE_IMAGE_MIN_SM: u16 = 70;
/// Compute capability carried by the explicitly identified Hopper target.
const HOPPER_SM: u16 = 90;
/// The image program compiled for the measured Blackwell compute capability.
const SPECIALIZED_IMAGE_SM: u16 = 120;
/// Driver API version needed to JIT the CUDA 11.8 portable PTX.
const PORTABLE_IMAGE_MIN_DRIVER: u32 = 11_080;
/// Driver API version needed to JIT the CUDA 13.4 specialized PTX.
const SPECIALIZED_IMAGE_MIN_DRIVER: u32 = 13_040;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImageKernel {
    Portable,
    Sm120,
}

impl CudaTarget {
    /// Select an image program by compute capability, including future architectures.
    pub(crate) fn image_kernel(&self, driver: u32) -> Result<Option<ImageKernel>> {
        let sm = match self {
            Self::HopperSm90 => HOPPER_SM,
            Self::BlackwellSm120 => SPECIALIZED_IMAGE_SM,
            Self::Other(name) => {
                let Some(sm) = name
                    .strip_prefix("sm_")
                    .and_then(|sm| sm.parse::<u16>().ok())
                else {
                    return Ok(None);
                };
                sm
            }
        };
        if sm < PORTABLE_IMAGE_MIN_SM {
            return Ok(None);
        }
        if driver < PORTABLE_IMAGE_MIN_DRIVER {
            return Err(Error::unsupported(format!(
                "CUDA driver API {driver} cannot load the image processor; CUDA 11.8 or newer is required"
            )));
        }
        Ok(Some(
            if sm == SPECIALIZED_IMAGE_SM && driver >= SPECIALIZED_IMAGE_MIN_DRIVER {
                ImageKernel::Sm120
            } else {
                ImageKernel::Portable
            },
        ))
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
