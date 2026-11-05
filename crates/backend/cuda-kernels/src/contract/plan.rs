//! Geometry passed to a kernel launch; owns no device handles.

/// Which positions a rotary launch reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    /// One sequence at one position.
    Single,
    /// One sequence with per-axis positions.
    SingleAxes,
    /// A prompt: each lane of `heads` rows at its own position.
    Prompt,
    /// A prompt with per-axis positions.
    PromptAxes,
    /// Pooled rows, each lane's position `fields` apart in the metadata.
    Pooled { fields: usize },
}

/// Rotary update of `rows` rows of `head_dim`, over the first `2 * half` elements of each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopePlan {
    pub mode: RopeMode,
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub half: usize,
}

/// Row-independent projection: `output[row, n] = sum_k weight[n, k] * input[row, k]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolGemvPlan {
    /// Independent activation rows.
    pub rows: usize,
    /// Output elements per row and weight rows.
    pub outputs: usize,
    /// Reduction length and weight columns.
    pub columns: usize,
    /// Output elements covered by one tile; only cuTile uses it.
    pub tile_rows: usize,
    /// Reduction tile the kernel loads per step; only cuTile uses it.
    pub tile_columns: usize,
}

/// How a normalization launch partitions its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormLayout {
    /// One row per block, over the next power of two of the dimension.
    Block,
    /// `tile` rows per block; the kernel masks a final partial tile.
    Rows { tile: usize },
}

/// RMS normalization of `rows` rows of `dimension`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormPlan {
    pub layout: NormLayout,
    pub rows: usize,
    pub dimension: usize,
    /// Multiply by the gate activation as well; [`NormLayout`] is unchanged by it.
    pub gated: bool,
    pub epsilon: f32,
    pub offset: f32,
    /// Measured compiler occupancy; `None` leaves the compiler's choice.
    pub occupancy: Option<i32>,
}

/// Which rows a draft fusion normalizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionShape {
    /// One embedding row and one hidden row, both flat.
    Single,
    /// `lanes` embedding rows and as many hidden rows, each `dimension` wide.
    Lanes { lanes: usize },
}

/// Normalization of a draft's embedding and hidden rows into `[2 * lanes, dimension]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionNormPlan {
    pub shape: FusionShape,
    pub dimension: usize,
    pub epsilon: f32,
    pub offset: f32,
}

/// Normalization that also produces what the next projection quantizes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantNormPlan {
    pub rows: usize,
    pub dimension: usize,
    pub epsilon: f32,
    pub offset: f32,
}

/// Tiles of a down projection that waits on its producer's activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DownPlan {
    pub tile_rows: usize,
    pub tile_columns: usize,
    /// Reduction length and weight columns.
    pub columns: usize,
}

/// Norm of one flat activation row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowNormPlan {
    pub dimension: usize,
    pub epsilon: f32,
    pub offset: f32,
}

/// Element-wise activation of one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Silu,
    Sigmoid,
}

/// Element-wise combination of two operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Multiply,
}

/// One part of `[rows, stride]` rows into `[rows, width]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitPlan {
    pub rows: usize,
    pub stride: usize,
    pub width: usize,
    /// Which `width`-wide part of each row.
    pub part: usize,
}

/// Top-scoring expert selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutePlan {
    pub experts: usize,
    pub active: usize,
    pub normalize: bool,
}

/// Projection of `active` routed experts, each `[rows, input]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertPlan {
    pub experts: usize,
    pub active: usize,
    pub rows: usize,
    pub input: usize,
    /// Each active expert reads its own input row rather than one shared row.
    pub routed_input: bool,
}

/// Greedy selection over `rows` rows of `vocabulary` logits, split into `blocks` of `tile`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GreedyPlan {
    pub rows: usize,
    pub vocabulary: usize,
    pub tile: usize,
    pub blocks: usize,
}

/// Which kernel projects a row matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowsKernel {
    /// Scalar row reductions, for few rows against a wide output.
    Small,
    /// Tensor-core products over tiles of the output and reduction.
    Tiled {
        output_tile: [usize; 2],
        tile_output: usize,
        tile_columns: usize,
        /// FP4 code and scale depths; read only for FP4 weights.
        fp4_depths: [usize; 2],
    },
}

/// Projection of `[rows, columns]` into `[rows, outputs]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowsPlan {
    pub kernel: RowsKernel,
    pub rows: usize,
    pub outputs: usize,
    pub columns: usize,
}

/// Which lanes' last active rows a prompt reads back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LastRowsKind {
    /// The one final row of a prompt chunk.
    One,
    /// One row per burst segment, `segment_rows` lanes each.
    Segments { segment_rows: usize },
}

/// Copy of `[lanes, hidden]` rows' last active rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LastRowsPlan {
    pub kind: LastRowsKind,
    pub lanes: usize,
    pub hidden: usize,
}

/// Rows of the fixed three-lane pack and unpack, one per verified MTP lane.
pub const THREE_ROWS: usize = 3;

/// Elements one narrowing tile converts; a [`Projection::narrow_bf16`] length is a multiple.
///
/// Chosen from prompt measurements: 5.2 us at 1024 elements versus 8.1 us at 8192.
///
/// [`Projection::narrow_bf16`]: crate::contract::Projection::narrow_bf16
pub const NARROW_GRANULE: usize = 1024;

/// How activations reach FP4 codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp4QuantKind {
    /// One scale search per row.
    Whole,
    /// Row groups in one launch.
    RowGroups { rows: usize },
}

/// An FP4 projection `[rows, columns] x [outputs, columns]` whose FP8 block scales a vendor
/// GEMM reads in 128-row by four-group blocks, weight scales first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp4ScalePlan {
    pub rows: usize,
    pub outputs: usize,
    pub columns: usize,
}

/// FP4 activation quantization of `[rows, columns]` into codes and block scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp4QuantPlan {
    pub kind: Fp4QuantKind,
    pub rows: usize,
    pub columns: usize,
    pub codes_tile: [usize; 2],
    pub scales_tile: [usize; 2],
}

/// How activations reach FP8 codes and their scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8QuantKind {
    /// Row maximum and conversion in one kernel.
    Whole,
    /// Row maximum, then conversion over `conversion_columns` tiles.
    Parallel {
        conversion_columns: usize,
        /// The preceding norm already wrote the maximum.
        maximum_ready: bool,
    },
    /// One scale per `block` columns.
    Block { block: usize },
}

/// FP8 activation quantization of `[rows, columns]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8QuantPlan {
    pub kind: Fp8QuantKind,
    pub rows: usize,
    pub columns: usize,
}

/// Main loop of a quantized GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmLoop {
    Plain,
    Prefetch,
}

/// Quantized `[rows, outputs]` GEMM over `columns` by the output tile `tile`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmPlan {
    pub main_loop: GemmLoop,
    pub rows: usize,
    pub outputs: usize,
    pub columns: usize,
    pub tile: [usize; 2],
}

/// Which dense row kernel a BF16 projection records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseKernel {
    /// One tile per row, by the linear tiling.
    Gemv {
        tile_rows: usize,
        tile_columns: usize,
    },
    /// Row groups of `group`, by the linear tiling.
    Shared {
        group: usize,
        tile_rows: usize,
        tile_columns: usize,
    },
    /// The small-matrix tile.
    Native,
    /// The small-matrix kernel with `width` outputs in a tile of `rows`.
    NativeWide { rows: usize, width: usize },
    /// The prompt GEMM by an explicit tile.
    PromptTile { rows: usize, outputs: usize },
    /// The prompt GEMM over `tile`, for its measured default shape.
    Prompt { tile: [usize; 2] },
    /// The prompt GEMM over a tile of sixteen outputs.
    PromptNarrow { tile: [usize; 2] },
    /// The small-matrix tile over a three-term BF16 split of the input with compensated
    /// accumulation: an arithmetic diagnostic, not a production choice.
    Compensated,
}

/// Token embedding lookup, or the external input when `external` and the metadata asks for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddingPlan {
    pub dimension: usize,
    pub external: bool,
}

/// Embedding of `dimension` channels over a metadata batch of `fields` per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddingBatchPlan {
    pub dimension: usize,
    pub fields: usize,
    /// Output tile columns.
    pub tile: usize,
}

/// BF16 projection of `[rows, columns]` over `[row_tile, output_tile]` tiles of depth `depth`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearRowsPlan {
    pub rows: usize,
    pub row_tile: usize,
    pub output_tile: usize,
    pub depth: usize,
    pub columns: usize,
}

/// BF16 projection of `[rows, columns]` into `[rows, outputs]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseRowsPlan {
    pub kernel: DenseKernel,
    pub rows: usize,
    pub outputs: usize,
    pub columns: usize,
}

/// Prompt GEMM of `[rows, columns]` over the output tile `tile`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptPlan {
    pub rows: usize,
    pub outputs: usize,
    pub columns: usize,
    pub tile: [usize; 2],
}

/// Split-K FP4 prompt GEMM: partial windows, then their reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitGemmPlan {
    pub prompt: PromptPlan,
    pub k_tiles: i32,
    pub splits: i32,
}

/// Heads, dimension and lanes of a recurrent prompt lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentShape {
    pub key_heads: usize,
    pub value_heads: usize,
    pub dim: usize,
    pub lanes: usize,
}

/// Per-token recurrence that derives its own gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaKernel {
    /// One state update per token.
    Scan,
    /// The WY factorization of a chunk.
    Wy,
}

/// Single-step recurrence, per one sequence or across a pool's lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaStepPlan {
    pub key_heads: usize,
    pub value_heads: usize,
    pub dim: usize,
    pub block: usize,
    /// `Some` for one launch across this many lanes.
    pub lanes: Option<usize>,
}

/// How a convolution lowers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvKernel {
    /// One sequence, one token.
    Step,
    /// One launch across a pool's lanes.
    Lanes { lanes: usize },
    /// A prompt chunk: forward all rows, then commit the history.
    Prefill { lanes: usize },
    /// Read-only candidates of every slot; `fields` metadata fields per lane.
    Verify {
        slots: usize,
        lanes: usize,
        fields: usize,
        /// The kernel specialized for the fused candidate count.
        fused: bool,
    },
}

/// Verification or commit of a whole pool across its slot axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolDeltaPlan {
    pub key_heads: usize,
    pub value_heads: usize,
    pub dim: usize,
    pub block: usize,
    pub slots: usize,
    pub lanes: usize,
    /// Metadata fields per lane.
    pub fields: usize,
    /// The kernel specialized for the fused candidate count.
    pub fused: bool,
}

/// Convolution commit of a whole pool across its slot axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolConvPlan {
    pub channels: usize,
    pub lanes: usize,
    /// Metadata fields per lane.
    pub fields: usize,
    /// The kernel specialized for the fused candidate count.
    pub fused: bool,
}
