//! Storage of the gated-delta recurrent state.
use serde::Serialize;

/// The element type the target program stores its gated-delta states in.
///
/// Every recurrence computes in F32 whichever is selected; `Bf16` widens the state on load and
/// rounds it to nearest even on store, halving the largest per-sequence allocation of a hybrid
/// model. It changes numerics, so it stays the numerical-acceptance profile until the quality
/// gate passes; convolution, KV and draft states are unaffected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum RecurrentProfile {
    /// F32 states, unchanged.
    #[default]
    F32,
    /// BF16 states computed in F32.
    Bf16,
}

impl RecurrentProfile {
    /// Bytes of one stored state element.
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::Bf16 => size_of::<u16>(),
        }
    }

    /// Stable readback/log name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }
}
