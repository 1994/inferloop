//! GGML grouped-integer weight formats and their backend-neutral geometry.
//!
//! Every format decodes to one unsigned code per value plus one F32 factor per group. A value is
//! `factor * (code - zero)` for symmetric formats and `factor * code - bias` for formats that
//! carry a per-group minimum. Backends choose their own code packing and kernels from this
//! geometry; container decoders produce codes, factors and biases in this form.
use serde::{Deserialize, Serialize};

/// Values in one k-quant super-block, the unit every grouped-int layout is tiled by.
pub const GROUPED_INT_SPAN: usize = 256;

/// Values sharing one factor in the 32-value formats.
const WIDE_GROUP: usize = 32;
/// Values sharing one factor in `Q6_K`.
const NARROW_GROUP: usize = 16;
/// Most groups any format places in one span.
pub const GROUPED_INT_MAX_GROUPS: usize = GROUPED_INT_SPAN / NARROW_GROUP;
/// Code widths, in bits.
const NIBBLE_BITS: u32 = 4;
const FIVE_BITS: u32 = 5;
const SIX_BITS: u32 = 6;
const BYTE_BITS: u32 = 8;
/// Zero points of the symmetric formats: the midpoint of each code range.
const NIBBLE_ZERO: u8 = 1 << (NIBBLE_BITS - 1);
const FIVE_BIT_ZERO: u8 = 1 << (FIVE_BITS - 1);
const SIX_BIT_ZERO: u8 = 1 << (SIX_BITS - 1);
const BYTE_ZERO: u8 = 1 << (BYTE_BITS - 1);

/// A grouped-integer block family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupedIntFormat {
    /// 32-value blocks, 4-bit codes, one scale, zero point 8.
    Q4_0,
    /// 32-value blocks, 4-bit codes, one scale and one minimum.
    Q4_1,
    /// 32-value blocks, 5-bit codes, one scale, zero point 16.
    Q5_0,
    /// 32-value blocks, 5-bit codes, one scale and one minimum.
    Q5_1,
    /// 32-value blocks, signed 8-bit codes, one scale.
    Q8_0,
    /// 256-value super-blocks, 4-bit codes, 32-value scale/min groups.
    Q4K,
    /// 256-value super-blocks, 5-bit codes, 32-value scale/min groups.
    Q5K,
    /// 256-value super-blocks, 6-bit codes, 16-value signed scale groups.
    Q6K,
}

/// How a group's codes map back to values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeOffset {
    /// `factor * (code - zero)`; the format stores no per-group bias.
    Zero(u8),
    /// `factor * code - bias`, with one bias per group.
    Bias,
}

impl GroupedIntFormat {
    /// Every format, in GGML type order.
    pub const ALL: [Self; 8] = [
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
    ];

    /// GGML type name, used in errors, diagnostics and tuning keys.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
        }
    }

    /// Significant bits of one unsigned code.
    #[must_use]
    pub const fn code_bits(self) -> u32 {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q4K => NIBBLE_BITS,
            Self::Q5_0 | Self::Q5_1 | Self::Q5K => FIVE_BITS,
            Self::Q6K => SIX_BITS,
            Self::Q8_0 => BYTE_BITS,
        }
    }

    /// Values sharing one factor (and bias).
    #[must_use]
    pub const fn group_values(self) -> usize {
        match self {
            Self::Q6K => NARROW_GROUP,
            _ => WIDE_GROUP,
        }
    }

    /// Groups in one [`GROUPED_INT_SPAN`].
    #[must_use]
    pub const fn span_groups(self) -> usize {
        GROUPED_INT_SPAN / self.group_values()
    }

    /// How codes map back to values.
    #[must_use]
    pub const fn offset(self) -> CodeOffset {
        match self {
            Self::Q4_0 => CodeOffset::Zero(NIBBLE_ZERO),
            Self::Q5_0 => CodeOffset::Zero(FIVE_BIT_ZERO),
            Self::Q6K => CodeOffset::Zero(SIX_BIT_ZERO),
            Self::Q8_0 => CodeOffset::Zero(BYTE_ZERO),
            Self::Q4_1 | Self::Q5_1 | Self::Q4K | Self::Q5K => CodeOffset::Bias,
        }
    }

    /// Whether the format carries a per-group bias.
    #[must_use]
    pub const fn has_biases(self) -> bool {
        matches!(self.offset(), CodeOffset::Bias)
    }

    /// One decoded value, rounded like the GGML reference decoder.
    #[must_use]
    #[expect(
        clippy::suboptimal_flops,
        reason = "the reference decoder rounds the multiply and subtract separately"
    )]
    pub fn value(self, code: u8, factor: f32, bias: f32) -> f32 {
        match self.offset() {
            CodeOffset::Zero(zero) => factor * f32::from(i16::from(code) - i16::from(zero)),
            CodeOffset::Bias => factor * f32::from(code) - bias,
        }
    }
}
