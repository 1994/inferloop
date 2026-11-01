//! Hardware identity and the device facts that follow from it.
//!
//! The only place the host learns what an architecture can do: a new architecture changes this
//! file and its artifacts, never the host.
use infer_core::{Error, Result};
use infer_ir::DType;

/// Compute capability of the device the kernels run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    Sm70,
    Sm75,
    Sm80,
    Sm86,
    Sm89,
    Sm90,
    Sm100,
    Sm103,
    Sm120,
    Other { major: u32, minor: u32 },
}

/// The portable image program targets CUDA compute capability 7.0 or later.
const PORTABLE_IMAGE_MIN_SM: u32 = 70;
/// The image program compiled for the measured Blackwell compute capability.
const SPECIALIZED_IMAGE_SM: u32 = 120;
/// Driver API version needed to JIT the CUDA 11.8 portable PTX.
const PORTABLE_IMAGE_MIN_DRIVER: u32 = 11_080;
/// Driver API version needed to JIT the CUDA 13.4 specialized PTX.
const SPECIALIZED_IMAGE_MIN_DRIVER: u32 = 13_040;
/// Compute capability digits: `sm_<major><minor>` reads as `major * 10 + minor`.
const SM_MAJOR_SCALE: u32 = 10;
/// Dense SM120 FP4/F32-accumulate operations per SM per clock.
const SM120_FP4: u128 = 4096;
/// Dense SM120 FP8/F32-accumulate operations per SM per clock.
const SM120_FP8: u128 = 1024;
/// SM120 BF16/F32-accumulate rate; the F32-input path records two BF16 products.
const SM120_BF16: u128 = 512;
/// Dense Hopper BF16/F32-accumulate operations per SM per clock.
const SM90_BF16: u128 = 4096;
/// Dense Hopper FP8/F32-accumulate operations per SM per clock.
const SM90_FP8: u128 = 8192;
/// The existing two-term F32 input decomposition uses two BF16 MMA products.
const DENSE_PRODUCTS: u128 = 2;

/// Image preprocessing program an architecture and driver can run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageKernel {
    /// CUDA 11.8 PTX, for compute capability 7.0 or later.
    Portable,
    /// CUDA 13.4 PTX specialized for SM120.
    Sm120,
}

/// Architectural operations per SM per clock, not a table indexed by GPU marketing names.
///
/// Dense/sparse and accumulator precision must not be conflated. Sources: NVIDIA RTX Blackwell
/// whitepaper Table 3 and NVIDIA Hopper Architecture In-Depth, Tensor Core architecture/table 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorRates {
    /// Rate the architecture's peak is normalized to.
    pub normalization: u128,
    /// FP4 rate, when the architecture has an FP4 tensor core.
    pub fp4: Option<u128>,
    /// FP8 rate.
    pub fp8: u128,
    /// BF16 rate of one F32-input dense product.
    pub dense: u128,
}

/// Compute capabilities named once so the mapping carries no magic numbers.
const SM_70: (u32, u32) = (7, 0);
const SM_75: (u32, u32) = (7, 5);
const SM_80: (u32, u32) = (8, 0);
const SM_86: (u32, u32) = (8, 6);
const SM_89: (u32, u32) = (8, 9);
const SM_90: (u32, u32) = (9, 0);
const SM_100: (u32, u32) = (10, 0);
const SM_103: (u32, u32) = (10, 3);
const SM_120: (u32, u32) = (12, 0);

impl Arch {
    /// Map a driver-reported compute capability to a known profile.
    #[must_use]
    pub const fn from_sm(major: u32, minor: u32) -> Self {
        match (major, minor) {
            SM_70 => Self::Sm70,
            SM_75 => Self::Sm75,
            SM_80 => Self::Sm80,
            SM_86 => Self::Sm86,
            SM_89 => Self::Sm89,
            SM_90 => Self::Sm90,
            SM_100 => Self::Sm100,
            SM_103 => Self::Sm103,
            SM_120 => Self::Sm120,
            _ => Self::Other { major, minor },
        }
    }

    /// Parse the `sm_<major><minor>` spelling reported by the driver; the minor is one digit.
    #[must_use]
    pub fn from_sm_name(name: &str) -> Option<Self> {
        let digits = name.strip_prefix("sm_")?;
        let (major, minor) = digits.split_at(digits.len().checked_sub(1)?);
        Some(Self::from_sm(major.parse().ok()?, minor.parse().ok()?))
    }

    /// Compute dtypes this architecture executes natively.
    ///
    /// This is a capability, not a policy: which precision a checkpoint uses is declared by its
    /// model provider and resolved against this answer.
    #[must_use]
    pub const fn compute_dtypes(self) -> &'static [DType] {
        match self {
            Self::Sm120 => &[
                DType::F32,
                DType::Bf16,
                DType::Fp8E4M3,
                DType::Fp8E5M2,
                DType::Fp4E2M1,
            ],
            Self::Sm90 => &[DType::F32, DType::Bf16, DType::Fp8E4M3, DType::Fp8E5M2],
            _ => &[DType::F32, DType::Bf16],
        }
    }

    /// Whether this architecture executes `dtype` natively.
    #[must_use]
    pub fn supports_compute(self, dtype: DType) -> bool {
        self.compute_dtypes().contains(&dtype)
    }

    /// Gate the FP4 tile type before allocating launch output.
    /// # Errors
    /// Hopper cannot execute native FP4; other profiles require validation.
    pub fn require_native_nvfp4(self) -> Result<()> {
        match self {
            Self::Sm120 => Ok(()),
            Self::Sm90 => Err(Error::unsupported(
                "Hopper/H200 has no native FP4 type; use the DecodedBf16 model-loading policy",
            )),
            other => Err(Error::unsupported(format!(
                "native NVFP4 target {} has not been validated by this backend",
                other.name()
            ))),
        }
    }

    /// Whether the Rust SIMT attention, GDN and recurrent artifacts were built for this device.
    #[must_use]
    pub const fn has_simt_artifacts(self) -> bool {
        matches!(self, Self::Sm120)
    }

    /// Whether the fused draft projection has been qualified on this device.
    #[must_use]
    pub const fn qualifies_fused_draft(self) -> bool {
        matches!(self, Self::Sm120)
    }

    /// Whether narrow prompt graphs have been qualified to use the dense layout on this device.
    #[must_use]
    pub const fn qualifies_narrow_dense(self) -> bool {
        matches!(self, Self::Sm120)
    }

    /// Tensor core rates, when they have been sourced for this architecture.
    #[must_use]
    pub const fn tensor_rates(self) -> Option<TensorRates> {
        match self {
            Self::Sm120 => Some(TensorRates {
                normalization: SM120_FP4,
                fp4: Some(SM120_FP4),
                fp8: SM120_FP8,
                dense: SM120_BF16 / DENSE_PRODUCTS,
            }),
            Self::Sm90 => Some(TensorRates {
                normalization: SM90_FP8,
                fp4: None,
                fp8: SM90_FP8,
                dense: SM90_BF16 / DENSE_PRODUCTS,
            }),
            _ => None,
        }
    }

    /// Select an image program by compute capability, including future architectures.
    /// # Errors
    /// Rejects a driver too old to load the portable program.
    pub fn image_kernel(self, driver: u32) -> Result<Option<ImageKernel>> {
        let sm = self.compute_capability();
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

    const fn compute_capability(self) -> u32 {
        let (major, minor) = match self {
            Self::Sm70 => SM_70,
            Self::Sm75 => SM_75,
            Self::Sm80 => SM_80,
            Self::Sm86 => SM_86,
            Self::Sm89 => SM_89,
            Self::Sm90 => SM_90,
            Self::Sm100 => SM_100,
            Self::Sm103 => SM_103,
            Self::Sm120 => SM_120,
            Self::Other { major, minor } => (major, minor),
        };
        major * SM_MAJOR_SCALE + minor
    }

    /// `sm_<major><minor>` spelling used by artifacts and capability reports.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::Sm70 => "sm_70".to_owned(),
            Self::Sm75 => "sm_75".to_owned(),
            Self::Sm80 => "sm_80".to_owned(),
            Self::Sm86 => "sm_86".to_owned(),
            Self::Sm89 => "sm_89".to_owned(),
            Self::Sm90 => "sm_90".to_owned(),
            Self::Sm100 => "sm_100".to_owned(),
            Self::Sm103 => "sm_103".to_owned(),
            Self::Sm120 => "sm_120".to_owned(),
            Self::Other { major, minor } => format!("sm_{major}{minor}"),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/target.rs"]
mod tests;
