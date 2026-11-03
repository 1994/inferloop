//! The block-draft family: the launches of one `DFlash2` proposal and its ring ingestion.
use super::buffer::{Bf16, Buffer, Fp8};
use super::recording::Recording;
use infer_core::Result;

/// Row tile of a draft projection.
pub const DRAFT_ROW_TILE: usize = 32;
/// Output-channel tile of a draft projection.
pub const DRAFT_KEY_TILE: usize = 64;

/// A draft projection's weight, in the storage its checkpoint uses.
#[derive(Clone, Copy, Debug)]
pub enum DraftWeight<'a> {
    /// `[outputs, columns]`.
    Dense(Buffer<'a, Bf16>),
    /// `[outputs, columns]` and per-channel scales.
    Fp8 {
        weight: Buffer<'a, Fp8>,
        scales: Buffer<'a, f32>,
    },
    /// `[outputs, columns]` and `[outputs, scale_columns]` block scales.
    Fp8Block {
        weight: Buffer<'a, Fp8>,
        scales: Buffer<'a, f32>,
        scale_columns: usize,
    },
}

/// `out[rows, n] = input[rows, columns] · weightᵀ`.
#[derive(Clone, Copy, Debug)]
pub struct DraftProjection<'a> {
    pub out: Buffer<'a, f32>,
    pub input: Buffer<'a, f32>,
    pub rows: usize,
    pub columns: usize,
}

/// One projection split across its K tiles, then reduced through the scales.
#[derive(Clone, Copy, Debug)]
pub struct DraftSplit<'a> {
    pub projection: DraftProjection<'a>,
    pub weight: Buffer<'a, Fp8>,
    pub scales: Buffer<'a, f32>,
    pub partials: Buffer<'a, f32>,
    pub splits: usize,
    pub k_tiles: usize,
}

/// Normalization of rows of `dim` written in tiles of `tile` channels.
#[derive(Clone, Copy, Debug)]
pub struct DraftNorm<'a> {
    pub out: Buffer<'a, f32>,
    pub input: Buffer<'a, f32>,
    pub weight: Buffer<'a, f32>,
    pub dim: usize,
    pub tile: usize,
    pub epsilon: f32,
}

/// Residual update into `stream`, normalized into `out`.
#[derive(Clone, Copy, Debug)]
pub struct DraftAddNorm<'a> {
    pub norm: DraftNorm<'a>,
    pub stream: Buffer<'a, f32>,
    pub delta: Buffer<'a, f32>,
}

/// One side of a block's dynamic convolution.
#[derive(Clone, Copy, Debug)]
pub struct DraftConv<'a> {
    pub out: Buffer<'a, f32>,
    pub input: Buffer<'a, f32>,
    pub coefficients: Buffer<'a, f32>,
    pub base: Buffer<'a, f32>,
    pub rows: usize,
    pub hidden: usize,
    pub tile: usize,
    pub group_size: usize,
    pub groups: usize,
    pub block: usize,
    pub side: i32,
}

/// Rotary rows of `heads` heads per row.
#[derive(Clone, Copy, Debug)]
pub struct DraftRope<'a> {
    pub out: Buffer<'a, f32>,
    pub input: Buffer<'a, f32>,
    pub frequencies: Buffer<'a, f32>,
    pub positions: Buffer<'a, i32>,
    pub dim: usize,
    pub heads: usize,
}

/// `[rows, heads, dim]` stored into a `[heads, capacity, dim]` ring set.
#[derive(Clone, Copy, Debug)]
pub struct DraftRingStore<'a> {
    pub ring: Buffer<'a, Bf16>,
    pub input: Buffer<'a, f32>,
    pub rings: Buffer<'a, i32>,
    pub positions: Buffer<'a, i32>,
    pub rows: usize,
    pub heads: usize,
    pub dim: usize,
    pub capacity: usize,
    pub ring_rows: usize,
}

/// Non-causal attention of one block's rows over its ring.
#[derive(Clone, Copy, Debug)]
pub struct DraftAttention<'a> {
    pub out: Buffer<'a, f32>,
    pub queries: Buffer<'a, f32>,
    pub keys: Buffer<'a, Bf16>,
    pub values: Buffer<'a, Bf16>,
    pub bases: Buffer<'a, i32>,
    pub rings: Buffer<'a, i32>,
    pub rows: usize,
    pub kv_heads: usize,
    pub group: usize,
    pub dim: usize,
    pub capacity: usize,
    pub block: usize,
    pub ring_rows: usize,
    pub window: usize,
}

/// Top candidates of every row.
#[derive(Clone, Copy, Debug)]
pub struct DraftTop<'a> {
    pub partial_values: Buffer<'a, f32>,
    pub partial_ids: Buffer<'a, i32>,
    pub unary: Buffer<'a, f32>,
    pub candidates: Buffer<'a, i32>,
    pub logits: Buffer<'a, f32>,
    pub rows: usize,
    pub vocabulary: usize,
    /// Logits one partial launch ranks.
    pub top_block: usize,
    pub top_k: usize,
}

/// The selector's walk over the candidates, one block per slot.
#[derive(Clone, Copy, Debug)]
pub struct DraftWalk<'a> {
    pub tokens: Buffer<'a, i32>,
    pub candidates: Buffer<'a, i32>,
    pub unary: Buffer<'a, f32>,
    pub selector: Buffer<'a, f32>,
    pub predecessor: Buffer<'a, Bf16>,
    pub successor: Buffer<'a, Bf16>,
    pub anchors: Buffer<'a, i32>,
    pub rows: usize,
    pub slots: usize,
    pub rank: usize,
    pub top_k: usize,
    pub block: usize,
}

/// Block-draft kernels. Every method records and never waits.
///
/// # Errors of every method
/// A buffer disagrees with its plan, or the launch is rejected.
pub trait Draft: Send + Sync {
    /// Project rows through a dense or block-scaled weight.
    /// # Errors
    /// See the trait.
    fn project(
        &self,
        recording: &Recording<'_>,
        weight: DraftWeight<'_>,
        projection: DraftProjection<'_>,
    ) -> Result<()>;

    /// Project narrow outputs across idle multiprocessors.
    /// # Errors
    /// See the trait.
    fn project_split(&self, recording: &Recording<'_>, split: DraftSplit<'_>) -> Result<()>;

    /// Normalize rows.
    /// # Errors
    /// See the trait.
    fn norm(&self, recording: &Recording<'_>, norm: DraftNorm<'_>) -> Result<()>;

    /// Add a residual and normalize the sum.
    /// # Errors
    /// See the trait.
    fn add_norm(&self, recording: &Recording<'_>, add: DraftAddNorm<'_>) -> Result<()>;

    /// Embed tokens from a table of `shape[0]` columns, in tiles of `shape[1]` channels.
    /// # Errors
    /// See the trait.
    fn embed(
        &self,
        recording: &Recording<'_>,
        shape: [usize; 2],
        out: Buffer<'_, f32>,
        table: Buffer<'_, Bf16>,
        tokens: Buffer<'_, i32>,
    ) -> Result<()>;

    /// One side of a dynamic convolution.
    /// # Errors
    /// See the trait.
    fn conv(&self, recording: &Recording<'_>, conv: DraftConv<'_>) -> Result<()>;

    /// Rotate head rows by their positions.
    /// # Errors
    /// See the trait.
    fn rope(&self, recording: &Recording<'_>, rope: DraftRope<'_>) -> Result<()>;

    /// Store rows into a ring set.
    /// # Errors
    /// See the trait.
    fn ring_store(&self, recording: &Recording<'_>, store: DraftRingStore<'_>) -> Result<()>;

    /// Attend each block over its ring.
    /// # Errors
    /// See the trait.
    fn attend(&self, recording: &Recording<'_>, attention: DraftAttention<'_>) -> Result<()>;

    /// `out = silu(gate) * up` over flat rows.
    /// # Errors
    /// See the trait.
    fn gated_product(
        &self,
        recording: &Recording<'_>,
        out: Buffer<'_, f32>,
        gate: Buffer<'_, f32>,
        up: Buffer<'_, f32>,
    ) -> Result<()>;

    /// Rank each row's logits in blocks, then merge to its top candidates.
    /// # Errors
    /// See the trait.
    fn top_candidates(&self, recording: &Recording<'_>, top: DraftTop<'_>) -> Result<()>;

    /// Choose each block's tokens among the candidates.
    /// # Errors
    /// See the trait.
    fn walk(&self, recording: &Recording<'_>, walk: DraftWalk<'_>) -> Result<()>;
}
