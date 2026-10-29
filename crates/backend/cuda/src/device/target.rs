//! CUDA capability interpretation owned by the device boundary.
use super::architecture::CudaArchitecture;
use crate::target::CudaTarget;
use infer_core::{Error, Result};
use infer_ir::DType;

/// Derive BF16 support from a CUDA SM version, with optional compiler target qualifiers.
/// NVIDIA documents BF16 for compute capability 8.0 and newer in the CUDA programming guide.
#[inline(never)]
const fn supports_native_bf16(name: &str) -> bool {
    const MIN_SM_NAME_LEN: usize = 5;
    // Two signed 32-bit decimal attributes, the `sm_` prefix and an optional qualifier.
    const MAX_SM_NAME_LEN: usize = 24;
    const MAJOR_OFFSET: usize = 3;
    let bytes = name.as_bytes();
    if bytes.len() < MIN_SM_NAME_LEN
        || bytes.len() > MAX_SM_NAME_LEN
        || bytes[0] != b's'
        || bytes[1] != b'm'
        || bytes[2] != b'_'
        || bytes[MAJOR_OFFSET] == b'0'
    {
        return false;
    }
    let suffix = bytes[bytes.len() - 1];
    let qualified = suffix == b'a' || suffix == b'f';
    let digits_end = bytes.len() - if qualified { 1 } else { 0 };
    if digits_end < MIN_SM_NAME_LEN {
        return false;
    }
    let mut index = MAJOR_OFFSET;
    while index < digits_end {
        if !bytes[index].is_ascii_digit() {
            return false;
        }
        index += 1;
    }
    let double_digit_major = digits_end > MIN_SM_NAME_LEN;
    if qualified {
        // Architecture-specific targets start at 9.x; family-specific targets at 10.x.
        if suffix == b'a' && !double_digit_major && bytes[MAJOR_OFFSET] < b'9' {
            return false;
        }
        if suffix == b'f' && !double_digit_major {
            return false;
        }
    }
    double_digit_major || bytes[MAJOR_OFFSET] >= b'8'
}

impl CudaTarget {
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
            Self::Other(name) if supports_native_bf16(name.as_str()) => &[DType::F32, DType::Bf16],
            Self::Other(_) => &[DType::F32],
        }
    }

    /// Whether this architecture executes `dtype` natively.
    #[must_use]
    pub fn supports_compute(&self, dtype: DType) -> bool {
        self.compute_dtypes().contains(&dtype)
    }

    #[must_use]
    pub fn from_sm_name(name: &str) -> Self {
        match CudaArchitecture::from_name(name) {
            CudaArchitecture::Sm90 => Self::HopperSm90,
            CudaArchitecture::Sm120 => Self::BlackwellSm120,
            CudaArchitecture::Sm100 | CudaArchitecture::Sm103 | CudaArchitecture::Other => {
                Self::Other(name.to_owned())
            }
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
#[path = "../../tests/unit/target.rs"]
mod tests;
