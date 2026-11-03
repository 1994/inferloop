//! The vision encoder family: the launches of one image's tower and position interpolation.
#![expect(
    unsafe_code,
    reason = "Position taps state their index bound once, so the gather method can be safe"
)]
use super::buffer::{Bf16, Buffer};
use super::recording::Recording;
use infer_core::Result;

/// `out = input · weightᵀ + bias` over rows of `columns`.
#[derive(Clone, Copy, Debug)]
pub struct VisionDense<'a> {
    pub out: Buffer<'a, f32>,
    pub input: Buffer<'a, f32>,
    pub weight: Buffer<'a, Bf16>,
    pub bias: Buffer<'a, Bf16>,
    pub columns: usize,
}

/// Row-wise normalization of `dim` channels padded to `tile`.
#[derive(Clone, Copy, Debug)]
pub struct VisionNorm<'a> {
    pub out: Buffer<'a, f32>,
    pub input: Buffer<'a, f32>,
    pub weight: Buffer<'a, Bf16>,
    pub bias: Buffer<'a, Bf16>,
    pub dim: usize,
    pub tile: usize,
    pub epsilon: f32,
}

/// Axial rotation of `rows` rows.
#[derive(Clone, Copy, Debug)]
pub struct VisionRope<'a> {
    pub out: Buffer<'a, f32>,
    pub input: Buffer<'a, f32>,
    pub cosines: Buffer<'a, f32>,
    pub sines: Buffer<'a, f32>,
    pub rows: usize,
    pub half: usize,
    /// Real half width of a compact head stored wider; `None` for a whole power-of-two head.
    pub real_half: Option<usize>,
}

/// Bilinear interpolation taps into a learned `[rows, hidden]` position table.
///
/// The gather reads the table at addresses the device computes from `indices`, so the bound on
/// those indices is a precondition no launch can check; [`Self::new`] carries it.
#[derive(Clone, Copy, Debug)]
pub struct PositionTaps<'a> {
    table: Buffer<'a, Bf16>,
    indices: Buffer<'a, i32>,
    weights: Buffer<'a, f32>,
    hidden: usize,
}

impl<'a> PositionTaps<'a> {
    /// Taps over `table` rows of `hidden` channels; `indices` and `weights` are tap-major
    /// `[4, padded]`.
    ///
    /// # Safety
    /// Every element of `indices` is non-negative and below `table.len() / hidden`, padded
    /// entries included, until the recorded work has completed.
    #[must_use]
    pub const unsafe fn new(
        table: Buffer<'a, Bf16>,
        indices: Buffer<'a, i32>,
        weights: Buffer<'a, f32>,
        hidden: usize,
    ) -> Self {
        Self {
            table,
            indices,
            weights,
            hidden,
        }
    }

    /// The table.
    #[must_use]
    pub const fn table(self) -> Buffer<'a, Bf16> {
        self.table
    }

    /// Tap-major row indices.
    #[must_use]
    pub const fn indices(self) -> Buffer<'a, i32> {
        self.indices
    }

    /// Tap-major interpolation weights.
    #[must_use]
    pub const fn weights(self) -> Buffer<'a, f32> {
        self.weights
    }

    /// Channels of one table row.
    #[must_use]
    pub const fn hidden(self) -> usize {
        self.hidden
    }
}

/// The activation of a vision MLP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gelu {
    /// The hyperbolic-tangent approximation.
    Tanh,
    /// The exact error-function form.
    Erf,
}

/// Direction of the compact-head packing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pack {
    /// Compact heads into power-of-two ones.
    Wide,
    /// Power-of-two heads back into compact ones.
    Compact,
}

/// Vision kernels. Every method records and never waits.
///
/// # Errors of every method
/// A buffer disagrees with its plan, or the launch is rejected.
pub trait Vision: Send + Sync {
    /// Project rows with a bias.
    /// # Errors
    /// See the trait.
    fn dense_bias(&self, recording: &Recording<'_>, dense: VisionDense<'_>) -> Result<()>;

    /// `out = left + right` over rows of `columns`.
    /// # Errors
    /// See the trait.
    fn add(
        &self,
        recording: &Recording<'_>,
        columns: usize,
        out: Buffer<'_, f32>,
        left: Buffer<'_, f32>,
        right: Buffer<'_, f32>,
    ) -> Result<()>;

    /// Activate rows of `columns`.
    /// # Errors
    /// See the trait.
    fn gelu(
        &self,
        recording: &Recording<'_>,
        form: Gelu,
        columns: usize,
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
    ) -> Result<()>;

    /// Normalize rows into a flat output.
    /// # Errors
    /// See the trait.
    fn layernorm(&self, recording: &Recording<'_>, norm: VisionNorm<'_>) -> Result<()>;

    /// Rotate head rows through their tables.
    /// # Errors
    /// See the trait.
    fn rope(&self, recording: &Recording<'_>, rope: VisionRope<'_>) -> Result<()>;

    /// Pack `rows` rows between compact and power-of-two heads.
    /// # Errors
    /// See the trait.
    fn pack(
        &self,
        recording: &Recording<'_>,
        direction: Pack,
        rows: usize,
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
    ) -> Result<()>;

    /// `out = embedded + Σ weight · table[index]` over the four taps of each row.
    /// # Errors
    /// See the trait.
    fn add_positions(
        &self,
        recording: &Recording<'_>,
        out: Buffer<'_, f32>,
        embedded: Buffer<'_, f32>,
        taps: PositionTaps<'_>,
    ) -> Result<()>;
}
