//! FA4 numerical profile and hardware candidates, separate from canonical F32 attention.
use crate::device::platform::is_linux_x86_64;
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};

pub const HEADS: usize = 24;
pub const KV_HEADS: usize = 4;
pub const DIM: usize = 256;
pub const LANE_MULTIPLE: usize = 64;
pub const OPERANDS: usize = 6;
pub const METADATA_FIELDS: usize = 3;
const BF16_PAIR_BYTES: usize = 4;
pub const PROFILE: &str = "fa4-bf16-qkv:f32-acc:bf16-output:gpu-seqlen-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Variant {
    Sm120,
    Hopper,
    Sm100,
    Sm100Pair,
    Sm103,
    Sm103Pair,
}
impl Variant {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sm120 => "sm120",
            Self::Hopper => "sm90a",
            Self::Sm100 => "sm100a-1cta",
            Self::Sm100Pair => "sm100a-2cta",
            Self::Sm103 => "sm103a-1cta",
            Self::Sm103Pair => "sm103a-2cta",
        }
    }
    #[cfg(test)]
    pub const fn cluster(self) -> usize {
        match self {
            Self::Sm100Pair | Self::Sm103Pair => 2,
            _ => 1,
        }
    }
    pub fn candidates(architecture: &str) -> &'static [Self] {
        match architecture {
            "sm_120" => &[Self::Sm120],
            "sm_90" => &[Self::Hopper],
            "sm_100" => &[Self::Sm100, Self::Sm100Pair],
            "sm_103" => &[Self::Sm103, Self::Sm103Pair],
            _ => &[],
        }
    }
    pub const fn bytes(self) -> &'static [u8] {
        match self {
            Self::Sm120 => include_bytes!("../../../kernels/fa4-attention/sm120/attention.so"),
            Self::Hopper => include_bytes!("../../../kernels/fa4-attention/sm90a/attention.so"),
            Self::Sm100 => {
                include_bytes!("../../../kernels/fa4-attention/sm100a-1cta/attention.so")
            }
            Self::Sm100Pair => {
                include_bytes!("../../../kernels/fa4-attention/sm100a-2cta/attention.so")
            }
            Self::Sm103 => {
                include_bytes!("../../../kernels/fa4-attention/sm103a-1cta/attention.so")
            }
            Self::Sm103Pair => {
                include_bytes!("../../../kernels/fa4-attention/sm103a-2cta/attention.so")
            }
        }
    }
}
pub fn supported(
    architecture: &str,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    lanes: usize,
    window: usize,
) -> bool {
    is_linux_x86_64()
        && !Variant::candidates(architecture).is_empty()
        && geometry(heads, kv_heads, dim, lanes, window)
}
pub fn geometry(heads: usize, kv_heads: usize, dim: usize, lanes: usize, window: usize) -> bool {
    (heads, kv_heads, dim) == (HEADS, KV_HEADS, DIM)
        && lanes > 0
        && lanes.is_multiple_of(LANE_MULTIPLE)
        && i32::try_from(lanes).is_ok()
        && lanes
            .checked_mul(HEADS)
            .and_then(|n| n.checked_mul(DIM))
            .is_some()
        && window == 0
}
/// One Q/O pair and one K/V pair per program, shared across its attention layers.
pub fn workspace_bytes(capacity: usize, lanes: usize) -> Result<usize> {
    capacity
        .checked_mul(KV_HEADS)
        .and_then(|n| n.checked_mul(DIM))
        .and_then(|n| n.checked_add(lanes.checked_mul(HEADS)?.checked_mul(DIM)?))
        .and_then(|n| n.checked_mul(BF16_PAIR_BYTES))
        .and_then(|n| n.checked_add(2 * size_of::<i32>()))
        .ok_or_else(|| Error::invalid("MMA attention workspace size overflow"))
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_attention_mma_catalog.rs"]
mod tests;
