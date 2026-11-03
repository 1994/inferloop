//! Operator-family traits implemented by each kernel source.

use super::attention::Attention;
use super::draft::Draft;
use super::vision::Vision;
use crate::contract::buffer::{Bf16, Buffer, Fp4Pair, Fp8};
use crate::contract::plan::{
    BinaryOp, DeltaKernel, DenseRowsPlan, DownPlan, EmbeddingBatchPlan, EmbeddingPlan, ExpertPlan,
    Fp4QuantPlan, Fp8QuantPlan, FusionNormPlan, GemmPlan, GreedyPlan, LastRowsPlan, LinearRowsPlan,
    NormPlan, PoolGemvPlan, PromptPlan, QuantNormPlan, RopePlan, RoutePlan, RowNormPlan, RowsPlan,
    SplitGemmPlan, SplitPlan, THREE_ROWS, UnaryOp,
};
use crate::contract::plan::{
    ConvKernel, DeltaStepPlan, PoolConvPlan, PoolDeltaPlan, RecurrentShape,
};
use crate::contract::recording::Recording;
use infer_core::Result;

/// Row-major projection kernels.
///
/// Methods are split by weight dtype instead of a generic parameter so the trait stays
/// object-safe and unsupported dtypes are rejected before a launch.
pub trait Projection: Send + Sync {
    /// Project `[rows, columns]` rows into `[rows, outputs]`, by the kernel the plan names.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, the plan's kernel does not take this weight's
    /// storage, or the launch is rejected.
    fn rows(
        &self,
        recording: &Recording<'_>,
        plan: &RowsPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: WeightRef<'_>,
    ) -> Result<()>;

    /// Project one flat row by the tiles of `plan`.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn row(
        &self,
        recording: &Recording<'_>,
        plan: &DownPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: WeightRef<'_>,
    ) -> Result<()>;

    /// Copy flat `input` into row `row` of `out`, whose shape is `[rows, columns]`.
    /// # Errors
    /// Fails when the launch is rejected.
    fn pack_row(
        &self,
        recording: &Recording<'_>,
        shape: (usize, usize),
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
        row: usize,
    ) -> Result<()>;

    /// Copy row `row` of `input`, whose shape is `[rows, columns]`, into flat `out`.
    /// # Errors
    /// Fails when the launch is rejected.
    fn unpack_row(
        &self,
        recording: &Recording<'_>,
        shape: (usize, usize),
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
        row: usize,
    ) -> Result<()>;

    /// [`Self::pack_row`] of exactly three rows in one launch.
    /// # Errors
    /// Fails when the launch is rejected.
    fn pack_three(
        &self,
        recording: &Recording<'_>,
        columns: usize,
        out: Buffer<'_, f32>,
        inputs: [Buffer<'_, f32>; THREE_ROWS],
    ) -> Result<()>;

    /// [`Self::unpack_row`] of exactly three rows in one launch.
    /// # Errors
    /// Fails when the launch is rejected.
    fn unpack_three(
        &self,
        recording: &Recording<'_>,
        columns: usize,
        outs: [Buffer<'_, f32>; THREE_ROWS],
        input: Buffer<'_, f32>,
    ) -> Result<()>;

    /// BF16 projection by the kernel the plan names.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn dense(
        &self,
        recording: &Recording<'_>,
        plan: &DenseRowsPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        w: Buffer<'_, Bf16>,
    ) -> Result<()>;

    /// One token's embedding row, or the external input the plan allows.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn embedding(
        &self,
        recording: &Recording<'_>,
        plan: &EmbeddingPlan,
        out: Buffer<'_, f32>,
        weight: Buffer<'_, Bf16>,
        external: Buffer<'_, f32>,
        metadata: Buffer<'_, i32>,
    ) -> Result<()>;

    /// Embedding rows of a whole metadata batch, taken from `external` where a row allows it.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn embedding_batch(
        &self,
        recording: &Recording<'_>,
        plan: &EmbeddingBatchPlan,
        out: Buffer<'_, f32>,
        weight: Buffer<'_, Bf16>,
        external: Option<Buffer<'_, f32>>,
        metadata: Buffer<'_, i32>,
    ) -> Result<()>;

    /// BF16 projection of a pool's rows over an explicit output and depth tile.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn linear_rows(
        &self,
        recording: &Recording<'_>,
        plan: &LinearRowsPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        w: Buffer<'_, Bf16>,
    ) -> Result<()>;

    /// Two prompt BF16 projections of one input in one launch.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn dense_pair(
        &self,
        recording: &Recording<'_>,
        plan: &PromptPlan,
        outs: (Buffer<'_, f32>, Buffer<'_, f32>),
        x: Buffer<'_, f32>,
        weights: (Buffer<'_, Bf16>, Buffer<'_, Bf16>),
    ) -> Result<()>;

    /// Prompt FP8 projection of unquantized activations.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, the weight is not FP8, or the launch is
    /// rejected.
    fn prompt_fp8(
        &self,
        recording: &Recording<'_>,
        plan: &PromptPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: WeightRef<'_>,
    ) -> Result<()>;

    /// Split-K FP4 prompt projection accumulating into `partial`, then reducing into `out`.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, the weight is not FP4, or the launch is
    /// rejected.
    fn prompt_fp4_split(
        &self,
        recording: &Recording<'_>,
        plan: &SplitGemmPlan,
        out: Buffer<'_, f32>,
        partial: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: WeightRef<'_>,
    ) -> Result<()>;

    /// Quantize `[rows, columns]` activations into FP4 codes and block scales.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn quantize_fp4(
        &self,
        recording: &Recording<'_>,
        plan: &Fp4QuantPlan,
        codes: Buffer<'_, Fp4Pair>,
        scales: Buffer<'_, Fp8>,
        input: Buffer<'_, f32>,
        scale: f32,
    ) -> Result<()>;

    /// [`Self::quantize_fp4`] of `silu(gate) * up`, row groups only.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn quantize_swiglu(
        &self,
        recording: &Recording<'_>,
        plan: &Fp4QuantPlan,
        codes: Buffer<'_, Fp4Pair>,
        scales: Buffer<'_, Fp8>,
        inputs: (Buffer<'_, f32>, Buffer<'_, f32>),
        scale: f32,
    ) -> Result<()>;

    /// W4A4 GEMM over prepared activation codes.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, the weight is not FP4, or the launch is
    /// rejected.
    fn gemm_fp4(
        &self,
        recording: &Recording<'_>,
        plan: &GemmPlan,
        out: Buffer<'_, f32>,
        codes: (Buffer<'_, Fp4Pair>, Buffer<'_, Fp8>),
        weight: WeightRef<'_>,
        inverse_scale: f32,
    ) -> Result<()>;

    /// Gate and up GEMMs of a gated MLP, writing `silu(gate) * up` straight into FP4 codes.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, a weight is not FP4, or the launch is
    /// rejected.
    fn gemm_swiglu(
        &self,
        recording: &Recording<'_>,
        plan: &GemmPlan,
        out: (Buffer<'_, Fp4Pair>, Buffer<'_, Fp8>),
        codes: (Buffer<'_, Fp4Pair>, Buffer<'_, Fp8>),
        weights: (WeightRef<'_>, WeightRef<'_>),
        scales: (f32, f32, f32),
    ) -> Result<()>;

    /// Quantize `[rows, columns]` activations into FP8 codes and scales.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn quantize_fp8(
        &self,
        recording: &Recording<'_>,
        plan: &Fp8QuantPlan,
        codes: Buffer<'_, Fp8>,
        scales: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
    ) -> Result<()>;

    /// FP8 GEMM over prepared activation codes. Without `block`, `codes.1` holds one scale per row
    /// and the weight's scales one per output; with it, both hold one per `block` columns.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, the weight is not FP8, or the launch is
    /// rejected.
    fn gemm_fp8(
        &self,
        recording: &Recording<'_>,
        plan: &GemmPlan,
        out: Buffer<'_, f32>,
        codes: (Buffer<'_, Fp8>, Buffer<'_, f32>),
        weight: WeightRef<'_>,
        block: Option<usize>,
    ) -> Result<()>;

    /// Copy each lane's last active row into `out`.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn last_rows(
        &self,
        recording: &Recording<'_>,
        plan: &LastRowsPlan,
        out: Buffer<'_, f32>,
        rows: Buffer<'_, f32>,
        metadata: Buffer<'_, i32>,
    ) -> Result<()>;

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

/// A projection's weight, in the storage its checkpoint uses.
#[derive(Clone, Copy, Debug)]
pub enum WeightRef<'a> {
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
        weight: WeightRef<'_>,
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

/// History rows a convolution keeps.
pub const CONV_HISTORY: usize = 3;

/// Operands of the chunked prompt rule. The scratch is the caller's, and outlives the graph.
#[derive(Clone, Copy)]
pub struct ChunkBuffers<'a> {
    pub queries: Buffer<'a, Bf16>,
    pub keys: Buffer<'a, Bf16>,
    pub u: Buffer<'a, Bf16>,
    pub w: Buffer<'a, Bf16>,
    pub p: Buffer<'a, Bf16>,
    pub cumulative: Buffer<'a, f32>,
    pub qkv: Buffer<'a, f32>,
    pub betas: Buffer<'a, f32>,
    pub decays: Buffer<'a, f32>,
    pub state: Buffer<'a, f32>,
    pub output: Buffer<'a, f32>,
    pub metadata: Buffer<'a, i32>,
}

/// Operands of a recurrence that derives its gates per token.
#[derive(Clone, Copy)]
pub struct DeltaBuffers<'a> {
    pub state: Buffer<'a, f32>,
    pub output: Buffer<'a, f32>,
    pub qkv: Buffer<'a, f32>,
    pub beta: Buffer<'a, f32>,
    pub alpha: Buffer<'a, f32>,
    pub decay: Buffer<'a, f32>,
    pub bias: Buffer<'a, f32>,
    pub metadata: Buffer<'a, i32>,
}

/// Operands of the recurrence over precomputed normalization and gates.
#[derive(Clone, Copy)]
pub struct PreparedDeltaBuffers<'a> {
    pub state: Buffer<'a, f32>,
    pub output: Buffer<'a, f32>,
    pub qkv: Buffer<'a, f32>,
    pub normalized: Buffer<'a, f32>,
    pub betas: Buffer<'a, f32>,
    pub decays: Buffer<'a, f32>,
    pub metadata: Buffer<'a, i32>,
}

/// Operands of the shared gate evaluation.
#[derive(Clone, Copy)]
pub struct GateBuffers<'a> {
    pub betas: Buffer<'a, f32>,
    pub decays: Buffer<'a, f32>,
    pub beta: Buffer<'a, f32>,
    pub alpha: Buffer<'a, f32>,
    pub decay: Buffer<'a, f32>,
    pub bias: Buffer<'a, f32>,
}

/// Operands of a convolution: the three history rows are the state.
#[derive(Clone, Copy)]
pub struct ConvBuffers<'a> {
    pub output: Buffer<'a, f32>,
    pub history: [Buffer<'a, f32>; CONV_HISTORY],
    pub input: Buffer<'a, f32>,
    pub weight: Buffer<'a, f32>,
    pub metadata: Buffer<'a, i32>,
}

/// Candidate lanes the fused single-slot lowering covers.
pub const FUSED_LANES: usize = 3;

/// One buffer per fused candidate lane.
pub type LaneBuffers<'a, T> = [Buffer<'a, T>; FUSED_LANES];

/// Operands of the fused single-slot convolution verification.
#[derive(Clone, Copy)]
pub struct FusedConvBuffers<'a> {
    pub history: [Buffer<'a, f32>; CONV_HISTORY],
    pub inputs: LaneBuffers<'a, f32>,
    pub weight: Buffer<'a, f32>,
    pub metadata: LaneBuffers<'a, i32>,
}

/// Operands of the fused single-slot delta verification and commit.
#[derive(Clone, Copy)]
pub struct FusedDeltaBuffers<'a> {
    pub state: Buffer<'a, f32>,
    pub qkv: LaneBuffers<'a, f32>,
    pub beta: LaneBuffers<'a, f32>,
    pub alpha: LaneBuffers<'a, f32>,
    pub decay: Buffer<'a, f32>,
    pub bias: Buffer<'a, f32>,
    pub metadata: LaneBuffers<'a, i32>,
}

/// Operands of pool-wide delta verification or commit.
#[derive(Clone, Copy)]
pub struct PoolDeltaBuffers<'a> {
    pub state: Buffer<'a, f32>,
    pub qkv: Buffer<'a, f32>,
    pub beta: Buffer<'a, f32>,
    pub alpha: Buffer<'a, f32>,
    pub parameters: Buffer<'a, f32>,
    pub metadata: Buffer<'a, i32>,
}

/// Fused and pool-wide speculative verification of the recurrence.
pub trait RecurrentPool: Send + Sync {
    /// Read-only three-candidate convolution of one slot.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn conv_verify_fused(
        &self,
        recording: &Recording<'_>,
        channels: usize,
        outputs: LaneBuffers<'_, f32>,
        buffers: FusedConvBuffers<'_>,
    ) -> Result<()>;

    /// Commit the first `count` fused candidates into one slot's convolution history.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn conv_commit_fused(
        &self,
        recording: &Recording<'_>,
        count: i32,
        history: [Buffer<'_, f32>; CONV_HISTORY],
        inputs: LaneBuffers<'_, f32>,
        metadata: LaneBuffers<'_, i32>,
    ) -> Result<()>;

    /// Read-only three-candidate recurrence of one slot.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn delta_verify_fused(
        &self,
        recording: &Recording<'_>,
        plan: &DeltaStepPlan,
        outputs: LaneBuffers<'_, f32>,
        buffers: FusedDeltaBuffers<'_>,
    ) -> Result<()>;

    /// Commit the first `count` fused candidates into one slot's recurrence state.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn delta_commit_fused(
        &self,
        recording: &Recording<'_>,
        plan: &DeltaStepPlan,
        count: i32,
        buffers: FusedDeltaBuffers<'_>,
    ) -> Result<()>;

    /// Read-only verification across the slot axis.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn delta_verify_pool(
        &self,
        recording: &Recording<'_>,
        plan: &PoolDeltaPlan,
        output: Buffer<'_, f32>,
        buffers: PoolDeltaBuffers<'_>,
    ) -> Result<()>;

    /// Commit each slot's accepted prefix across the slot axis.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn delta_commit_pool(
        &self,
        recording: &Recording<'_>,
        plan: &PoolDeltaPlan,
        counts: Buffer<'_, i32>,
        buffers: PoolDeltaBuffers<'_>,
    ) -> Result<()>;

    /// Shift each slot's convolution window by its accepted prefix.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn conv_commit_pool(
        &self,
        recording: &Recording<'_>,
        plan: &PoolConvPlan,
        history: [Buffer<'_, f32>; CONV_HISTORY],
        rows: Buffer<'_, f32>,
        metadata: Buffer<'_, i32>,
        counts: Buffer<'_, i32>,
    ) -> Result<()>;
}

/// Gated delta recurrence and its convolution.
pub trait Recurrent: Send + Sync {
    /// One token's recurrence.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn delta_step(
        &self,
        recording: &Recording<'_>,
        plan: &DeltaStepPlan,
        buffers: DeltaBuffers<'_>,
    ) -> Result<()>;

    /// One token's recurrence, keeping the prior state in `snapshot`.
    /// # Errors
    /// Fails when a buffer disagrees with the plan, or the launch is rejected.
    fn delta_checkpoint(
        &self,
        recording: &Recording<'_>,
        plan: &DeltaStepPlan,
        snapshot: Buffer<'_, f32>,
        buffers: DeltaBuffers<'_>,
    ) -> Result<()>;

    /// One token's convolution, keeping the prior history in `snapshot`.
    /// # Errors
    /// Fails when a buffer disagrees with the kernel, or the launch is rejected.
    fn conv_checkpoint(
        &self,
        recording: &Recording<'_>,
        channels: usize,
        snapshot: [Buffer<'_, f32>; CONV_HISTORY],
        buffers: ConvBuffers<'_>,
    ) -> Result<()>;

    /// A convolution of `channels`.
    /// # Errors
    /// Fails when a buffer disagrees with the kernel, or the launch is rejected.
    fn conv(
        &self,
        recording: &Recording<'_>,
        kernel: ConvKernel,
        channels: usize,
        buffers: ConvBuffers<'_>,
    ) -> Result<()>;

    /// The chunked BF16 prompt rule.
    /// # Errors
    /// Fails when a buffer disagrees with the shape, or the launch is rejected.
    fn chunk(
        &self,
        recording: &Recording<'_>,
        shape: RecurrentShape,
        chunk: usize,
        buffers: ChunkBuffers<'_>,
    ) -> Result<()>;

    /// Normalize every query and key per head and token.
    /// # Errors
    /// Fails when a buffer disagrees with the shape, or the launch is rejected.
    fn normalize(
        &self,
        recording: &Recording<'_>,
        shape: RecurrentShape,
        normalized: Buffer<'_, f32>,
        qkv: Buffer<'_, f32>,
    ) -> Result<()>;

    /// Evaluate beta and decay once per head and token.
    /// # Errors
    /// Fails when a buffer disagrees with the shape, or the launch is rejected.
    fn gates(
        &self,
        recording: &Recording<'_>,
        shape: RecurrentShape,
        buffers: GateBuffers<'_>,
    ) -> Result<()>;

    /// The ordered recurrence over [`Recurrent::normalize`] and [`Recurrent::gates`].
    /// # Errors
    /// Fails when a buffer disagrees with the shape, or the launch is rejected.
    fn delta_prepared(
        &self,
        recording: &Recording<'_>,
        shape: RecurrentShape,
        block: usize,
        buffers: PreparedDeltaBuffers<'_>,
    ) -> Result<()>;

    /// A recurrence that derives its gates per token.
    /// # Errors
    /// Fails when a buffer disagrees with the shape, or the launch is rejected.
    fn delta(
        &self,
        recording: &Recording<'_>,
        kernel: DeltaKernel,
        shape: RecurrentShape,
        block: usize,
        buffers: DeltaBuffers<'_>,
    ) -> Result<()>;

    /// Reorder head-major scratch into lane-major output.
    /// # Errors
    /// Fails when a buffer disagrees with the shape, or the launch is rejected.
    fn transpose(
        &self,
        recording: &Recording<'_>,
        shape: RecurrentShape,
        output: Buffer<'_, f32>,
        scratch: Buffer<'_, f32>,
    ) -> Result<()>;
}

/// One device's kernels, selected once when a model loads; no call path chooses a source.
pub struct Kernels {
    pub vision: Box<dyn Vision>,
    pub draft: Box<dyn Draft>,
    pub attention: Box<dyn Attention>,
    pub projection: Box<dyn Projection>,
    pub rope: Box<dyn Rope>,
    pub norm: Box<dyn Norm>,
    pub mlp: Box<dyn Mlp>,
    pub state: Box<dyn State>,
    pub moe: Box<dyn Moe>,
    pub sampling: Box<dyn Sampling>,
    pub recurrent: Box<dyn Recurrent>,
    pub recurrent_pool: Box<dyn RecurrentPool>,
}
