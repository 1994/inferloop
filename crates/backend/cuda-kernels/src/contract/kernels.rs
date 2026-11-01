//! Operator-family traits implemented by each kernel source.

use crate::contract::buffer::{Bf16, Buffer};
use crate::contract::plan::{PoolGemvPlan, RopePlan};
use crate::contract::recording::Recording;
use infer_core::Result;

/// Row-major projection kernels.
///
/// Methods are split by weight dtype instead of a generic parameter so the trait stays
/// object-safe and unsupported dtypes are rejected before a launch.
pub trait Projection: Send + Sync {
    /// Record one BF16 row-independent projection; no work completes before the graph runs.
    ///
    /// `out`, `x` and `w` are row-major with shapes `[rows, outputs]`, `[rows, columns]` and
    /// `[outputs, columns]`.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, the recording form does not belong to this
    /// implementation, or the launch is rejected.
    fn pool_gemv_bf16(
        &self,
        recording: &Recording<'_>,
        plan: &PoolGemvPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        w: Buffer<'_, Bf16>,
    ) -> Result<()>;
}

/// Operands of one rotary launch.
#[derive(Clone, Copy, Debug)]
pub struct RopeBuffers<'a> {
    /// `[rows, head_dim]`.
    pub out: Buffer<'a, f32>,
    /// `[rows, head_dim]`, distinct from `out`.
    pub x: Buffer<'a, f32>,
    /// `[half]`.
    pub frequencies: Buffer<'a, f32>,
    /// `[half]`; present exactly when the plan's mode reads per-axis positions.
    pub axes: Option<Buffer<'a, i32>>,
    /// Position fields, one row per lane.
    pub metadata: Buffer<'a, i32>,
}

/// Rotary position kernels.
pub trait Rope: Send + Sync {
    /// Record one rotary launch; no work completes before the graph runs.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, the recording form does not belong to this
    /// implementation, or the launch is rejected.
    fn rope(
        &self,
        recording: &Recording<'_>,
        plan: &RopePlan,
        buffers: RopeBuffers<'_>,
    ) -> Result<()>;
}

/// One device's kernels, selected once when a model loads; no call path chooses a source.
pub struct Kernels {
    pub projection: Box<dyn Projection>,
    pub rope: Box<dyn Rope>,
}
