//! Storage encodings of attention KV caches: a runtime policy, never a weight encoding.
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Bytes of one single-precision element.
const F32_BYTES: usize = 4;
/// Bytes of one half-precision element.
const F16_BYTES: usize = 2;
/// Bytes of one E4M3 element.
const FP8_BYTES: usize = 1;

/// How a backend stores attention keys and values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheEncoding {
    /// Unquantized single precision.
    F32,
    /// IEEE half precision.
    F16,
    /// E4M3 with per-tensor static scales the checkpoint declares.
    Fp8E4m3,
}

impl KvCacheEncoding {
    /// Operator-facing name, as `--kv-cache-dtype` accepts it.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Fp8E4m3 => "fp8",
        }
    }

    /// Device bytes of one stored element.
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::F32 => F32_BYTES,
            Self::F16 => F16_BYTES,
            Self::Fp8E4m3 => FP8_BYTES,
        }
    }
}

impl FromStr for KvCacheEncoding {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        match text {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "fp8" | "fp8_e4m3" => Ok(Self::Fp8E4m3),
            other => Err(Error::invalid(format!(
                "KV cache dtype {other}: expected f32, f16 or fp8"
            ))),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/hardware_kv_cache.rs"]
mod tests;
