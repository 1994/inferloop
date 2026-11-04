//! The attention family: KV append and decode, whatever the cache's element type.
//!
//! A family method takes one launch's buffers and a plan, so a kernel's rank, grid and generics
//! stay beside its source. Launches that only make sense together, a split decode's partial and
//! merge, are one method.
use super::buffer::{Buffer, Fp8, Owner};
use super::recording::Recording;
use infer_core::Result;

/// Padded query rows of one lane's split partials.
pub const GROUPED_ROWS: usize = 16;
/// Query rows of the stacked three-lane tile: three lanes plus one padding lane.
pub const STACKED_ROWS: usize = 32;
/// Rows each verification lane owns in the stacked query tile.
pub const STACKED_LANE_ROWS: usize = 8;
/// Lanes of one verification step sharing a grouped launch.
pub const VERIFY_LANES: usize = 3;
/// Fields of a dense attention tile choice: query rows, key rows, pipelining.
pub const DENSE_TILES: usize = 3;
/// Page tables one pooled control block packs.
pub const POOLED_TABLES: usize = 4;

/// Keys and values of one attention state.
#[derive(Clone, Copy, Debug)]
pub enum KvBuffers<'a> {
    /// Unquantized cache.
    F32 {
        keys: Buffer<'a, f32>,
        values: Buffer<'a, f32>,
    },
    /// FP8 cache, whose reads rescale through the cache's scales.
    Fp8 {
        keys: Buffer<'a, Fp8>,
        values: Buffer<'a, Fp8>,
    },
}

impl<'a> KvBuffers<'a> {
    /// Elements of the key and the value buffer.
    #[must_use]
    pub const fn lens(&self) -> [usize; 2] {
        match self {
            Self::F32 { keys, values } => [keys.len(), values.len()],
            Self::Fp8 { keys, values } => [keys.len(), values.len()],
        }
    }

    /// Device addresses of the key and the value buffer.
    #[must_use]
    pub const fn addresses(&self) -> [u64; 2] {
        match self {
            Self::F32 { keys, values } => [keys.address(), values.address()],
            Self::Fp8 { keys, values } => [keys.address(), values.address()],
        }
    }

    /// What keeps the key and the value memory alive, when the buffers know.
    #[must_use]
    pub const fn owners(&self) -> [Option<&'a Owner>; 2] {
        match self {
            Self::F32 { keys, values } => [keys.owner(), values.owner()],
            Self::Fp8 { keys, values } => [keys.owner(), values.owner()],
        }
    }
}

/// Element types an attention cache can hold.
pub trait KvElement: Sized {
    /// Name a typed key and value pair.
    fn kv<'a>(keys: Buffer<'a, Self>, values: Buffer<'a, Self>) -> KvBuffers<'a>;
}

impl KvElement for f32 {
    fn kv<'a>(keys: Buffer<'a, Self>, values: Buffer<'a, Self>) -> KvBuffers<'a> {
        KvBuffers::F32 { keys, values }
    }
}

impl KvElement for Fp8 {
    fn kv<'a>(keys: Buffer<'a, Self>, values: Buffer<'a, Self>) -> KvBuffers<'a> {
        KvBuffers::Fp8 { keys, values }
    }
}

/// One KV pair and how a launch addresses it: `[groups, rows, dim]`, rows following from its size.
#[derive(Clone, Copy, Debug)]
pub struct KvCache<'a> {
    pub buffers: KvBuffers<'a>,
    pub groups: usize,
    pub dim: usize,
    pub scales: [f32; 2],
}

impl KvCache<'_> {
    /// Rows of the `[groups, rows, dim]` layout, when keys and values both hold whole rows of it.
    #[must_use]
    pub fn rows(&self) -> Option<usize> {
        let row = self.groups.checked_mul(self.dim).filter(|&row| row > 0)?;
        let [keys, values] = self.buffers.lens();
        (keys == values && keys.is_multiple_of(row)).then_some(keys / row)
    }
}

/// One launch's cache writes.
#[derive(Clone, Copy, Debug)]
pub struct AppendBuffers<'a> {
    pub cache: KvCache<'a>,
    pub keys: Buffer<'a, f32>,
    pub values: Buffer<'a, f32>,
    pub metadata: Buffer<'a, i32>,
    pub table: Buffer<'a, i32>,
}

/// One launch's cache read.
#[derive(Clone, Copy, Debug)]
pub struct DecodeBuffers<'a> {
    pub output: Buffer<'a, f32>,
    pub query: Buffer<'a, f32>,
    pub cache: KvCache<'a>,
    pub metadata: Buffer<'a, i32>,
    pub table: Buffer<'a, i32>,
}

/// Scratch a split decode's partial writes and its merge reads.
#[derive(Clone, Copy, Debug)]
pub struct Partials<'a> {
    pub numerator: Buffer<'a, f32>,
    pub maxima: Buffer<'a, f32>,
    pub sums: Buffer<'a, f32>,
}

/// The three candidates of one verification step over one cache.
#[derive(Clone, Copy, Debug)]
pub struct VerifyDecodeBuffers<'a> {
    pub outputs: [Buffer<'a, f32>; VERIFY_LANES],
    pub queries: [Buffer<'a, f32>; VERIFY_LANES],
    pub metadata: [Buffer<'a, i32>; VERIFY_LANES],
    pub cache: KvCache<'a>,
    pub table: Buffer<'a, i32>,
}

/// One group's control block and page writes across pooled slots.
#[derive(Clone, Copy, Debug)]
pub struct PooledAppendBuffers<'a> {
    pub control: Buffer<'a, i32>,
    pub tables: [Buffer<'a, i32>; POOLED_TABLES],
    pub header: Buffer<'a, i32>,
    /// The whole pool's metadata; the plan's group selects its fragment.
    pub metadata: Buffer<'a, i32>,
    pub keys: Buffer<'a, Fp8>,
    pub values: Buffer<'a, Fp8>,
    /// The whole pool's key rows.
    pub key_rows: Buffer<'a, f32>,
    /// The whole pool's value rows.
    pub value_rows: Buffer<'a, f32>,
}

/// One group's cache read across pooled slots.
#[derive(Clone, Copy, Debug)]
pub struct PooledDecodeBuffers<'a> {
    pub output: Buffer<'a, f32>,
    pub partials: Partials<'a>,
    /// The whole pool's queries.
    pub queries: Buffer<'a, f32>,
    pub keys: Buffer<'a, Fp8>,
    pub values: Buffer<'a, Fp8>,
    pub control: Buffer<'a, i32>,
}

/// Dense attention operands, `[rows, columns]` each.
#[derive(Clone, Copy, Debug)]
pub struct DenseBuffers<'a> {
    pub out: Buffer<'a, f32>,
    pub query: Buffer<'a, f32>,
    pub key: Buffer<'a, f32>,
    pub value: Buffer<'a, f32>,
}

/// Compensated-F32 dense attention over one sequence.
#[derive(Clone, Copy, Debug)]
pub struct DensePlan {
    pub head_width: usize,
    /// Streaming accumulation rather than global-maximum-first.
    pub online: bool,
    /// Rows of the query and key tiles; pipelining the third field is reserved.
    pub tiles: [usize; DENSE_TILES],
    pub group: usize,
    pub value_width: usize,
    pub mask: i32,
    /// Rows of the query and output operands.
    pub query_rows: usize,
    /// Rows of the key and value operands.
    pub key_rows: usize,
    pub tokens: i32,
    pub scale: f32,
    pub frame_tokens: i32,
    pub queries: i32,
    pub query_start: i32,
    pub left: i32,
    pub right: i32,
}

/// What every decode launch agrees on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    /// Sliding window in tokens; zero reads the whole cache.
    pub window: i32,
    /// Tokens of one KV page.
    pub block: usize,
}

/// Decode of a prompt chunk's lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillDecodePlan {
    pub decode: DecodePlan,
    pub lanes: usize,
}

/// Prompt append across lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillAppendPlan {
    pub lanes: usize,
    /// Rows one launch block owns: a page, or the whole head.
    pub rows: usize,
    pub block: usize,
}

/// Append of one launch across a pool's lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LanesAppendPlan {
    pub kv_heads: usize,
    pub lanes: usize,
    /// One row per lane of a pooled root, rather than pages shared by every lane.
    pub per_lane: bool,
    /// Rows of the two-axis page table.
    pub table_rows: usize,
}

/// Decode split across a window's parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitDecodePlan {
    pub decode: DecodePlan,
    pub parts: usize,
    /// KV heads of the grouped partial layout, or `None` for one partial per query head.
    pub groups: Option<usize>,
}

/// Grouped decode of one launch across a pool's lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LanesDecodePlan {
    pub decode: DecodePlan,
    pub parts: usize,
    pub lanes: usize,
    pub per_lane: bool,
    pub table_rows: usize,
}

/// One group of a pooled launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PooledPlan {
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub parts: usize,
    /// Columns of the control block.
    pub columns: usize,
    /// Metadata fields per lane.
    pub fields: usize,
    /// Slots of a full group, which sizes the partials.
    pub full_batch: usize,
    /// First slot of this group.
    pub first: usize,
    /// Slots of this group.
    pub batch: usize,
}

/// KV append and decode.
///
/// Every method records and never waits. # Errors below: a buffer disagrees with the plan, or the
/// launch is rejected.
pub trait Attention: Send + Sync {
    /// Append one lane's row.
    /// # Errors
    /// See the trait.
    fn append(&self, recording: &Recording<'_>, buffers: AppendBuffers<'_>) -> Result<()>;

    /// Append a prompt chunk's rows.
    /// # Errors
    /// See the trait.
    fn append_prefill(
        &self,
        recording: &Recording<'_>,
        plan: PrefillAppendPlan,
        buffers: AppendBuffers<'_>,
    ) -> Result<()>;

    /// Append every lane of a pool in one launch.
    /// # Errors
    /// See the trait.
    fn append_lanes(
        &self,
        recording: &Recording<'_>,
        plan: LanesAppendPlan,
        buffers: AppendBuffers<'_>,
    ) -> Result<()>;

    /// Dense attention over every query.
    /// # Errors
    /// See the trait.
    fn dense(
        &self,
        recording: &Recording<'_>,
        plan: DensePlan,
        buffers: DenseBuffers<'_>,
    ) -> Result<()>;

    /// Dense attention of one query row.
    /// # Errors
    /// See the trait.
    fn dense_decode(
        &self,
        recording: &Recording<'_>,
        plan: DensePlan,
        buffers: DenseBuffers<'_>,
    ) -> Result<()>;

    /// Decode one lane with one launch.
    /// # Errors
    /// See the trait.
    fn decode(
        &self,
        recording: &Recording<'_>,
        plan: DecodePlan,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()>;

    /// Decode a prompt chunk's lanes block by block.
    /// # Errors
    /// See the trait.
    fn decode_prefill(
        &self,
        recording: &Recording<'_>,
        plan: PrefillDecodePlan,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()>;

    /// Decode a prompt chunk's lanes in query tiles of `query_tile` rows.
    /// # Errors
    /// See the trait.
    fn decode_prefill_tiled(
        &self,
        recording: &Recording<'_>,
        plan: PrefillDecodePlan,
        query_tile: usize,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()>;

    /// Decode through a split's partials, then merge them.
    /// # Errors
    /// See the trait.
    fn decode_split(
        &self,
        recording: &Recording<'_>,
        plan: SplitDecodePlan,
        partials: Partials<'_>,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()>;

    /// One grouped launch for three candidates, then a merge for each.
    /// # Errors
    /// See the trait.
    fn decode_verify(
        &self,
        recording: &Recording<'_>,
        plan: SplitDecodePlan,
        partials: Partials<'_>,
        buffers: VerifyDecodeBuffers<'_>,
    ) -> Result<()>;

    /// One grouped launch and merge across a pool's lanes.
    /// # Errors
    /// See the trait.
    fn decode_lanes(
        &self,
        recording: &Recording<'_>,
        plan: LanesDecodePlan,
        partials: Partials<'_>,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()>;

    /// Pack one group's control block and write its pages.
    /// # Errors
    /// See the trait.
    fn append_pooled(
        &self,
        recording: &Recording<'_>,
        plan: PooledPlan,
        buffers: PooledAppendBuffers<'_>,
    ) -> Result<()>;

    /// Decode one group's slots and merge into their rows.
    /// # Errors
    /// See the trait.
    fn decode_pooled(
        &self,
        recording: &Recording<'_>,
        plan: PooledPlan,
        buffers: PooledDecodeBuffers<'_>,
    ) -> Result<()>;
}
