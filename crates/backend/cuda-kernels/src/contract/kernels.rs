//! Operator-family traits implemented by each kernel source.

use crate::contract::buffer::{Bf16, Buffer, Fp4Pair, Fp8};
use crate::contract::plan::{
    BinaryOp, DownPlan, ExpertPlan, FusionNormPlan, GreedyPlan, NormPlan, PoolGemvPlan,
    QuantNormPlan, RopePlan, RoutePlan, RowNormPlan, SplitPlan, UnaryOp,
};
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

/// Operands of one normalization launch.
#[derive(Clone, Copy, Debug)]
pub struct NormBuffers<'a> {
    /// `[rows, dimension]`.
    pub out: Buffer<'a, f32>,
    /// `[rows, dimension]`, distinct from `out`.
    pub x: Buffer<'a, f32>,
    /// `[dimension]`.
    pub weight: Buffer<'a, f32>,
    /// `[rows, dimension]`; read only when the plan is gated.
    pub gate: Buffer<'a, f32>,
}

/// Operands of one draft fusion normalization.
#[derive(Clone, Copy, Debug)]
pub struct FusionNormBuffers<'a> {
    /// `[2 * lanes, dimension]`.
    pub out: Buffer<'a, f32>,
    pub embedding: Buffer<'a, f32>,
    pub hidden: Buffer<'a, f32>,
    /// `[2, dimension]`.
    pub norms: Buffer<'a, f32>,
}

/// Operands of a normalization that records each row's largest magnitude.
#[derive(Clone, Copy, Debug)]
pub struct NormMaximumBuffers<'a> {
    /// `[rows, dimension]`.
    pub out: Buffer<'a, f32>,
    /// `[rows, 1]`.
    pub maximum: Buffer<'a, f32>,
    /// `[rows, dimension]`.
    pub x: Buffer<'a, f32>,
    /// `[dimension]`.
    pub weight: Buffer<'a, f32>,
}

/// Operands of a normalization that also quantizes to FP4.
#[derive(Clone, Copy, Debug)]
pub struct NormFp4Buffers<'a> {
    /// `[rows, dimension]`.
    pub out: Buffer<'a, f32>,
    /// `[rows, dimension / 2]` code pairs.
    pub codes: Buffer<'a, Fp4Pair>,
    /// `[rows, dimension / group]`.
    pub scales: Buffer<'a, Fp8>,
    /// `[rows, dimension]`.
    pub x: Buffer<'a, f32>,
    /// `[dimension]`.
    pub weight: Buffer<'a, f32>,
}

/// Normalization kernels. Every method records and never waits.
pub trait Norm: Send + Sync {
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn norm(
        &self,
        recording: &Recording<'_>,
        plan: &NormPlan,
        buffers: NormBuffers<'_>,
    ) -> Result<()>;

    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn fusion(
        &self,
        recording: &Recording<'_>,
        plan: &FusionNormPlan,
        buffers: FusionNormBuffers<'_>,
    ) -> Result<()>;

    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn maximum(
        &self,
        recording: &Recording<'_>,
        plan: &QuantNormPlan,
        buffers: NormMaximumBuffers<'_>,
    ) -> Result<()>;

    /// `input_global` is the activation's global FP4 scale.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn fp4(
        &self,
        recording: &Recording<'_>,
        plan: &QuantNormPlan,
        input_global: f32,
        buffers: NormFp4Buffers<'_>,
    ) -> Result<()>;
}

/// A down projection's weight, in the storage its checkpoint uses.
#[derive(Clone, Copy, Debug)]
pub enum DownWeight<'a> {
    /// `[outputs, columns]`.
    Dense(Buffer<'a, Bf16>),
    /// `[outputs, columns]` and per-channel scales.
    Fp8 {
        weight: Buffer<'a, Fp8>,
        scales: Buffer<'a, f32>,
    },
    /// `[outputs, columns / 2]` code pairs, block scales and the inverse global scale.
    Fp4 {
        weight: Buffer<'a, Fp4Pair>,
        scales: Buffer<'a, Fp8>,
        inverse_global: f32,
    },
}

/// Operands of an activation product: `out = silu(gate) * up`.
#[derive(Clone, Copy, Debug)]
pub struct ProductBuffers<'a> {
    pub out: Buffer<'a, f32>,
    pub gate: Buffer<'a, f32>,
    pub up: Buffer<'a, f32>,
}

/// Fused MLP kernels. Every method records and never waits.
pub trait Mlp: Send + Sync {
    /// Normalize one flat row.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn row_norm(
        &self,
        recording: &Recording<'_>,
        plan: &RowNormPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: Buffer<'_, f32>,
    ) -> Result<()>;

    /// # Errors
    /// Fails when the launch is rejected.
    fn product(&self, recording: &Recording<'_>, buffers: ProductBuffers<'_>) -> Result<()>;

    /// `out = down + x`.
    /// # Errors
    /// Fails when the launch is rejected.
    fn residual(
        &self,
        recording: &Recording<'_>,
        out: Buffer<'_, f32>,
        down: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
    ) -> Result<()>;

    /// [`Self::product`] that signals [`Self::down_after_product`] as rows become ready.
    /// # Errors
    /// Fails when the launch is rejected.
    fn signalling_product(
        &self,
        recording: &Recording<'_>,
        buffers: ProductBuffers<'_>,
    ) -> Result<()>;

    /// A down projection that starts before its input is complete, waiting on the producer.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn down_after_product(
        &self,
        recording: &Recording<'_>,
        plan: &DownPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: DownWeight<'_>,
    ) -> Result<()>;
}

/// Element-wise and slicing kernels on activations. Every method records and never waits.
pub trait State: Send + Sync {
    /// # Errors
    /// Fails when the launch is rejected.
    fn unary(
        &self,
        recording: &Recording<'_>,
        op: UnaryOp,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
    ) -> Result<()>;

    /// # Errors
    /// Fails when the launch is rejected.
    fn binary(
        &self,
        recording: &Recording<'_>,
        op: BinaryOp,
        out: Buffer<'_, f32>,
        a: Buffer<'_, f32>,
        b: Buffer<'_, f32>,
    ) -> Result<()>;

    /// `out = x * scalar`, with `scalar` holding one element.
    /// # Errors
    /// Fails when the launch is rejected.
    fn scale(
        &self,
        recording: &Recording<'_>,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        scalar: Buffer<'_, f32>,
    ) -> Result<()>;

    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn split(
        &self,
        recording: &Recording<'_>,
        plan: &SplitPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
    ) -> Result<()>;
}

/// An expert bank's weight, in the storage its checkpoint uses.
#[derive(Clone, Copy, Debug)]
pub enum ExpertWeight<'a> {
    /// `[experts, rows, input]`.
    Dense(Buffer<'a, Bf16>),
    /// `[experts, rows, input]` with one scale per row.
    Fp8Channel {
        weight: Buffer<'a, Fp8>,
        scales: Buffer<'a, f32>,
        quantized_inputs: bool,
    },
    /// `[experts, rows, input]` with one scale per `block_columns` of a row.
    Fp8Block {
        weight: Buffer<'a, Fp8>,
        scales: Buffer<'a, f32>,
        block_columns: usize,
        quantized_inputs: bool,
    },
}

/// Mixture-of-experts kernels. Every method records and never waits.
pub trait Moe: Send + Sync {
    /// # Errors
    /// Fails when the launch is rejected.
    fn route(
        &self,
        recording: &Recording<'_>,
        plan: &RoutePlan,
        out: Buffer<'_, f32>,
        scores: Buffer<'_, f32>,
    ) -> Result<()>;

    /// `x` is `[active, hidden]`, the two values of `geometry`.
    /// # Errors
    /// Fails when a buffer disagrees with the geometry, or the launch is rejected.
    fn combine(
        &self,
        recording: &Recording<'_>,
        geometry: (usize, usize),
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        routes: Buffer<'_, f32>,
    ) -> Result<()>;

    /// `out` is `[active, rows]`.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn experts(
        &self,
        recording: &Recording<'_>,
        plan: &ExpertPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        routes: Buffer<'_, f32>,
        weight: ExpertWeight<'_>,
    ) -> Result<()>;
}

/// Reduction state between the two greedy kernels, each `[rows, blocks]`.
#[derive(Clone, Copy, Debug)]
pub struct GreedyPartials<'a> {
    pub keys: Buffer<'a, u32>,
    pub indices: Buffer<'a, i32>,
    pub invalid: Buffer<'a, i32>,
}

/// Device-side token selection. Every method records and never waits.
pub trait Sampling: Send + Sync {
    /// Reduce each logits block; `logits` is `[rows, vocabulary]`.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn greedy_partials(
        &self,
        recording: &Recording<'_>,
        plan: &GreedyPlan,
        partials: GreedyPartials<'_>,
        logits: Buffer<'_, f32>,
    ) -> Result<()>;

    /// Finish into `output` rows of two elements, starting at row `offset`.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn greedy_finish(
        &self,
        recording: &Recording<'_>,
        plan: &GreedyPlan,
        offset: usize,
        output: Buffer<'_, i32>,
        partials: GreedyPartials<'_>,
    ) -> Result<()>;
}

/// One device's kernels, selected once when a model loads; no call path chooses a source.
pub struct Kernels {
    pub projection: Box<dyn Projection>,
    pub rope: Box<dyn Rope>,
    pub norm: Box<dyn Norm>,
    pub mlp: Box<dyn Mlp>,
    pub state: Box<dyn State>,
    pub moe: Box<dyn Moe>,
    pub sampling: Box<dyn Sampling>,
}
