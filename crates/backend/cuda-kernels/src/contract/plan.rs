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
