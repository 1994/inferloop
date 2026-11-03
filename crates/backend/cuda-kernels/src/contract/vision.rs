//! The vision encoder family: the launches of one image's tower and position interpolation.
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
}
