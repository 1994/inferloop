//! `DFlash2` block drafting beside the target's pooled verification.
//!
//! The draft reads the target through one fused feature per committed token (the target graph's
//! `features` result). Every feature is turned into draft K/V once, in [`Engine::ingest_prompt`] or
//! [`Engine::ingest_verified`], and lands in its sequence's sliding-window ring; a proposal then
//! runs the five draft layers over one block `[anchor, mask * (block - 1)]` per pooled slot, picks
//! the top candidates of every masked row from the target's head and walks the selector lattice.
//! Rings belong to sequences, not pool slots, because a prompt is ingested before its sequence
//! is bound to a slot.
pub mod budget;
#[cfg(test)]
#[path = "../../../tests/unit/resident_dflash2_budget.rs"]
mod budget_tests;
#[cfg(test)]
mod kernels;
mod weights;
use infer_cuda_kernels::contract::{
    Buffer, DRAFT_KEY_TILE as KEY_TILE, DRAFT_ROW_TILE as ROW_TILE, Draft, DraftAddNorm,
    DraftAttention, DraftConv, DraftNorm, DraftProjection, DraftRingStore, DraftRope, DraftSplit,
    DraftTop, DraftWalk, DraftWeight, Recording,
};
use infer_cuda_kernels::tile::kernels as tile_kernels;
use infer_cuda_kernels::tile::runtime::{Allocation, Graph};
pub use weights::DraftWeights;

use super::buffers::buffer;
use super::capture::error;
use crate::constants::WIDEST_PREFILL_LANES;
use crate::device::{CudaDevice, PinnedUpload, Readbacks, device_error};
use crate::mlp::ProjectionWeight;
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, Result};
use std::collections::BTreeMap;
use std::ops::Range;
use std::result::Result as StdResult;
use std::sync::Arc;
use weights::Geometry;

/// Channels one embedding or convolution block handles.
const CHANNEL_TILE: usize = 512;
/// Vocabulary entries one top-k block reduces.
const TOP_BLOCK: usize = 2048;
/// Prompt features one ingestion replay consumes.
const PROMPT_ROWS: usize = 256;
/// Pinned metadata uploads a prompt ingestion queues before it fences: the widest prompt chunk
/// split into [`PROMPT_ROWS`] replays, plus one.
const PROMPT_UPLOADS: usize = WIDEST_PREFILL_LANES / PROMPT_ROWS + 1;

/// Per-row and per-slot fields of the proposal metadata, in upload order.
#[derive(Clone, Copy)]
enum Field {
    Tokens,
    Positions,
    RowRings,
    Bases,
    Anchors,
    SlotRings,
}

/// Row fields precede the slot fields in the proposal metadata.
const ROW_FIELDS: usize = 3;
/// Slot fields follow the row fields in the proposal metadata.
const SLOT_FIELDS: usize = 3;

/// Elements of the proposal metadata for `rows` rows over `slots` slots.
const fn meta_len(rows: usize, slots: usize) -> usize {
    ROW_FIELDS * rows + SLOT_FIELDS * slots
}

/// Element range of one metadata field.
const fn meta_range(field: Field, rows: usize, slots: usize) -> Range<usize> {
    let (offset, len) = match field {
        Field::Tokens => (0, rows),
        Field::Positions => (rows, rows),
        Field::RowRings => (2 * rows, rows),
        Field::Bases => (ROW_FIELDS * rows, slots),
        Field::Anchors => (ROW_FIELDS * rows + slots, slots),
        Field::SlotRings => (ROW_FIELDS * rows + 2 * slots, slots),
    };
    offset..offset + len
}

/// Split-K partials keyed by `(N, K)`: the buffer, its split count and K tiles per split.
type Partials = BTreeMap<(usize, usize), (Allocation<f32>, usize, usize)>;

/// One sequence's place in the rings and how much of its timeline the draft has seen.
#[derive(Debug)]
pub struct Cursor {
    pub ring: usize,
    /// Positions `0..context` have draft K/V in the ring.
    pub context: usize,
}

/// Per-layer K/V rings, `[kv_heads, rings * ring, head_dim]` each.
struct Rings {
    keys: Vec<Allocation<bf16>>,
    values: Vec<Allocation<bf16>>,
    /// Rows of one ring: the window plus one block, in whole key tiles.
    ring: usize,
    count: usize,
    free: Vec<usize>,
}

/// K/V projection of target features for `rows` rows.
struct Ingest {
    rows: usize,
    /// `[rows]` ring ids then `[rows]` positions; a negative position skips the row.
    meta: Allocation<i32>,
    normed: Allocation<f32>,
    k: Allocation<f32>,
    v: Allocation<f32>,
    k_normed: Allocation<f32>,
    k_rotated: Allocation<f32>,
}

/// Prompt ingestion owns its feature rows; the target's prompt graph writes elsewhere.
struct PromptIngest {
    features: Allocation<f32>,
    work: Ingest,
    graph: Graph,
    uploads: PinnedUpload<i32>,
}

/// Verified-row ingestion reads the pooled verify graph's feature rows in place.
struct VerifiedIngest {
    graph: Graph,
    work: Ingest,
    uploads: PinnedUpload<i32>,
    _features: Arc<Tensor<f32>>,
}

/// Activations of one block proposal for `slots` slots.
struct ProposeBuffers {
    slots: usize,
    rows: usize,
    /// `[rows]` tokens, `[rows]` positions, `[rows]` ring ids, `[slots]` block bases,
    /// `[slots]` anchors, `[slots]` ring ids.
    meta: Allocation<i32>,
    stream: [Allocation<f32>; 2],
    normed: Allocation<f32>,
    coefficients: Allocation<f32>,
    conv: Allocation<f32>,
    q: Allocation<f32>,
    k: Allocation<f32>,
    v: Allocation<f32>,
    q_normed: Allocation<f32>,
    k_normed: Allocation<f32>,
    q_rotated: Allocation<f32>,
    k_rotated: Allocation<f32>,
    attention: Allocation<f32>,
    projected: Allocation<f32>,
    gate: Allocation<f32>,
    up: Allocation<f32>,
    activated: Allocation<f32>,
    logits: Allocation<f32>,
    partial_values: Allocation<f32>,
    partial_ids: Allocation<i32>,
    unary: Allocation<f32>,
    candidates: Allocation<i32>,
    selector: Allocation<f32>,
    tokens: Allocation<i32>,
    /// Split-K partials keyed by `(N, K)` of the narrow FP8 projections: buffer, splits and
    /// K tiles per split.
    partials: Partials,
}

struct Propose {
    graph: Graph,
    buffers: ProposeBuffers,
    uploads: PinnedUpload<i32>,
    readbacks: Readbacks<i32>,
    upload: Vec<i32>,
}

/// One slot's block request: pool slot, the sequence's ring, the anchor token and its position.
pub type BlockRequest = (usize, usize, u32, usize);

/// One verified run of feature rows to ingest: first lane in the verify graph, ring, first
/// position and row count.
pub type VerifiedRows = (usize, usize, usize, usize);

/// The pooled graphs captured against one verify graph of `slots` slots.
struct Pooled {
    slots: usize,
    propose: Propose,
    verified: VerifiedIngest,
}

/// Draft state shared by every sequence of one executor.
pub struct Engine {
    /// Verified blocks and the tokens they committed, for the acceptance-length report.
    verified_blocks: u64,
    committed_tokens: u64,
    // Graphs drop before the buffers and rings their kernels reference.
    /// One proposal and ingestion pair per pooled verify graph, narrowest first.
    pooled: Vec<Pooled>,
    prompt: PromptIngest,
    rings: Rings,
    weights: Arc<DraftWeights>,
    draft: Box<dyn Draft>,
}

impl Engine {
    /// Allocate `rings` sequence rings and capture prompt ingestion.
    /// # Errors
    /// Rejects an empty ring count and reports CUDA allocation or capture failures.
    pub fn new(device: &CudaDevice, weights: Arc<DraftWeights>, rings: usize) -> Result<Self> {
        if rings == 0 {
            return Err(Error::invalid("DFlash2 needs at least one ring"));
        }
        let geometry = weights.geometry;
        let ring = (geometry.window + geometry.block).div_ceil(KEY_TILE) * KEY_TILE;
        let shape = [geometry.kv_heads, rings * ring, geometry.head_dim];
        let layers = weights.layers.len();
        let mut keys = Vec::with_capacity(layers);
        let mut values = Vec::with_capacity(layers);
        for _ in 0..layers {
            keys.push(zeros::<bf16>(device, &shape)?);
            values.push(zeros::<bf16>(device, &shape)?);
        }
        let rings = Rings {
            keys,
            values,
            ring,
            count: rings,
            free: (0..rings).rev().collect(),
        };
        let features = zeros::<f32>(device, &[PROMPT_ROWS, geometry.hidden])?;
        let work = Ingest::new(device, &geometry, PROMPT_ROWS)?;
        let draft = tile_kernels(device.ordinal()).draft;
        let graph = device.runtime().capture(|recording| {
            let recorder = Recorder::new(&*draft, recording);
            record_ingest(&recorder, &weights, &rings, &work, features.operand())
                .map_err(device_error)
        })?;
        let prompt = PromptIngest {
            features,
            work,
            graph,
            uploads: PinnedUpload::slots(device, 2 * PROMPT_ROWS, PROMPT_UPLOADS)?,
        };
        Ok(Self {
            verified_blocks: 0,
            committed_tokens: 0,
            pooled: Vec::new(),
            prompt,
            rings,
            weights,
            draft,
        })
    }

    /// Lease a ring to a new sequence, if one is free.
    pub fn acquire(&mut self) -> Option<Cursor> {
        self.rings
            .free
            .pop()
            .map(|ring| Cursor { ring, context: 0 })
    }

    /// Return a sequence's ring.
    pub fn release(&mut self, cursor: &Cursor) {
        if cursor.ring < self.rings.count && !self.rings.free.contains(&cursor.ring) {
            self.rings.free.push(cursor.ring);
        }
    }

    /// Capture one pooled proposal per verify graph of the pool and bind verified-row
    /// ingestion to that graph's feature rows. `graphs` lists each graph's slot count and
    /// feature tensor.
    /// # Errors
    /// Rejects a feature tensor that is not `slots * block` rows wide and reports CUDA failures.
    pub fn attach_pool(
        &mut self,
        device: &CudaDevice,
        graphs: &[(usize, Arc<Tensor<f32>>)],
    ) -> Result<()> {
        self.detach_pool();
        let mut pooled = Vec::with_capacity(graphs.len());
        for (slots, features) in graphs {
            pooled.push(self.capture_pooled(device, *slots, Arc::clone(features))?);
        }
        pooled.sort_by_key(|graphs| graphs.slots);
        self.pooled = pooled;
        Ok(())
    }

    fn capture_pooled(
        &self,
        device: &CudaDevice,
        slots: usize,
        features: Arc<Tensor<f32>>,
    ) -> Result<Pooled> {
        let geometry = self.weights.geometry;
        let rows = slots * geometry.block;
        if features.size() != rows * geometry.hidden {
            return Err(Error::invalid("verify features do not match the pool"));
        }
        let work = Ingest::new(device, &geometry, rows)?;
        let weights = &self.weights;
        let rings = &self.rings;
        let draft = &*self.draft;
        let graph = device
            .runtime()
            .capture(|recording| {
                let recorder = Recorder::new(draft, recording);
                buffer(features.as_ref())
                    .and_then(|features| record_ingest(&recorder, weights, rings, &work, features))
                    .map_err(device_error)
            })
            .map_err(|error| device_error(format!("{slots}-slot verified ingestion: {error}")))?;
        let verified = VerifiedIngest {
            graph,
            work,
            uploads: PinnedUpload::single(device, 2 * rows)?,
            _features: features,
        };
        let buffers = ProposeBuffers::new(device, &geometry, slots)?;
        let graph = device
            .runtime()
            .capture(|recording| {
                record_propose(&Recorder::new(draft, recording), weights, rings, &buffers)
                    .map_err(device_error)
            })
            .map_err(|error| device_error(format!("{slots}-slot proposal: {error}")))?;
        let meta = buffers.meta.len();
        let propose = Propose {
            graph,
            buffers,
            uploads: PinnedUpload::single(device, meta)?,
            readbacks: Readbacks::allocated(device, rows)?,
            upload: Vec::with_capacity(meta),
        };
        Ok(Pooled {
            slots,
            propose,
            verified,
        })
    }

    /// Drop the pooled graphs; the pool they read is going away.
    pub fn detach_pool(&mut self) {
        self.pooled.clear();
    }

    pub const fn attached(&self) -> bool {
        !self.pooled.is_empty()
    }

    /// Ingest `rows` prompt features starting at `position` into one ring.
    /// # Errors
    /// Rejects a short feature tensor or ring and reports CUDA failures.
    #[expect(
        unsafe_code,
        reason = "Audited replay: single-stream ordering after the staged copies, \
                  engine-owned buffers, drained before the engine drops"
    )]
    pub async fn ingest_prompt(
        &mut self,
        device: &CudaDevice,
        features: &Tensor<f32>,
        rows: usize,
        cursor: &mut Cursor,
        position: usize,
    ) -> Result<()> {
        let hidden = self.weights.geometry.hidden;
        if features.size() < rows * hidden || cursor.ring >= self.rings.count {
            return Err(Error::invalid("DFlash2 prompt ingestion rows"));
        }
        let mut upload = vec![0i32; 2 * PROMPT_ROWS];
        let mut start = 0;
        while start < rows {
            let count = (rows - start).min(PROMPT_ROWS);
            let source = buffer(features)
                .map_err(device_error)?
                .slice(start * hidden..(start + count) * hidden)?;
            let target = self.prompt.features.operand().slice(0..count * hidden)?;
            device.enqueue(|transfers| transfers.copy(target, source))?;
            for row in 0..PROMPT_ROWS {
                upload[row] = i32::try_from(cursor.ring).map_err(device_error)?;
                upload[PROMPT_ROWS + row] = if row < count {
                    i32::try_from(position + start + row).map_err(device_error)?
                } else {
                    -1
                };
            }
            self.prompt
                .uploads
                .copy(device, &self.prompt.work.meta, &upload)?;
            // SAFETY: ordered on the engine stream after the copies above; every buffer the
            // graph touches is engine-owned and the engine drains the stream before dropping.
            unsafe { self.prompt.graph.launch().async_on(&device.stream) }.map_err(device_error)?;
            start += count;
        }
        // The pinned ring above holds one chunk; fence it before the next chunk reuses it.
        let _ = device
            .read_allocation(&self.prompt.work.meta, &mut Readbacks::default())
            .await?;
        cursor.context = cursor.context.max(position + rows);
        Ok(())
    }

    /// Ingest verified rows from the pooled verify graph's feature lanes.
    /// # Errors
    /// Rejects an unattached pool or out-of-range rows and reports CUDA failures.
    #[expect(
        unsafe_code,
        reason = "Audited replay: single-stream ordering after the verify graph and the staged \
                  metadata, engine-owned buffers, drained before the engine drops"
    )]
    pub fn ingest_verified(
        &mut self,
        device: &CudaDevice,
        runs: &[VerifiedRows],
        slots: usize,
    ) -> Result<()> {
        let verified = &mut self
            .pooled
            .iter_mut()
            .find(|graphs| graphs.slots == slots)
            .ok_or_else(|| Error::invalid("DFlash2 pool is not attached at this width"))?
            .verified;
        let rows = verified.work.rows;
        let mut upload = vec![-1i32; 2 * rows];
        upload[..rows].fill(0);
        for &(lane, ring, position, count) in runs {
            if lane + count > rows || ring >= self.rings.count {
                return Err(Error::invalid("DFlash2 verified rows"));
            }
            for offset in 0..count {
                upload[lane + offset] = i32::try_from(ring).map_err(device_error)?;
                upload[rows + lane + offset] =
                    i32::try_from(position + offset).map_err(device_error)?;
            }
        }
        verified
            .uploads
            .copy(device, &verified.work.meta, &upload)?;
        // SAFETY: ordered after the verify replay and the metadata copy on the engine stream;
        // the graph reads pool-owned features retained by `_features` and engine buffers.
        unsafe { verified.graph.launch().async_on(&device.stream) }.map_err(device_error)?;
        self.record_acceptance(runs);
        Ok(())
    }

    /// Count committed tokens per verified block and report the running mean periodically.
    fn record_acceptance(&mut self, runs: &[VerifiedRows]) {
        /// Verified blocks between two acceptance reports.
        const REPORT_BLOCKS: u64 = 1024;
        for &(.., count) in runs {
            self.verified_blocks += 1;
            self.committed_tokens += count as u64;
            if self.verified_blocks.is_multiple_of(REPORT_BLOCKS) {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a diagnostic mean of bounded token counts"
                )]
                let mean = self.committed_tokens as f64 / self.verified_blocks as f64;
                tracing::debug!(target: "infer::dflash2", blocks = self.verified_blocks,
                    tokens = self.committed_tokens, mean_acceptance_length = mean,
                    "DFlash2 acceptance");
            }
        }
    }

    /// Propose one block per request; returns the `block - 1` proposed tokens of each, in
    /// request order.
    /// # Errors
    /// Rejects an unattached pool, duplicate or out-of-range slots and reports CUDA failures.
    #[expect(
        unsafe_code,
        reason = "Audited replay: single-stream ordering after the staged metadata, \
                  engine-owned buffers, fenced by the token readback"
    )]
    pub async fn propose(
        &mut self,
        device: &CudaDevice,
        requests: &[BlockRequest],
    ) -> Result<Vec<Vec<u32>>> {
        let geometry = self.weights.geometry;
        let busy = requests
            .iter()
            .map(|&(slot, ..)| slot + 1)
            .max()
            .unwrap_or(0);
        let propose = &mut self
            .pooled
            .iter_mut()
            .find(|graphs| graphs.slots >= busy)
            .ok_or_else(|| Error::invalid("DFlash2 pool is not attached"))?
            .propose;
        let ProposeBuffers { slots, rows, .. } = propose.buffers;
        let block = geometry.block;
        let mask = i32::try_from(geometry.mask_token).map_err(device_error)?;
        let upload = &mut propose.upload;
        upload.clear();
        upload.resize(meta_len(rows, slots), 0);
        for row in 0..rows {
            upload[row] = mask;
            upload[rows + row] = -1;
        }
        for slot in 0..slots {
            upload[meta_range(Field::Bases, rows, slots).start + slot] = -1;
        }
        let mut seen = vec![false; slots];
        for &(slot, ring, anchor, position) in requests {
            if slot >= slots || seen[slot] || ring >= self.rings.count {
                return Err(Error::invalid("DFlash2 block request"));
            }
            seen[slot] = true;
            let ring = i32::try_from(ring).map_err(device_error)?;
            let base = i32::try_from(position).map_err(device_error)?;
            let anchor = i32::try_from(anchor).map_err(device_error)?;
            for offset in 0..block {
                let row = slot * block + offset;
                if offset == 0 {
                    upload[row] = anchor;
                }
                upload[rows + row] = base + i32::try_from(offset).map_err(device_error)?;
                upload[meta_range(Field::RowRings, rows, slots).start + row] = ring;
            }
            upload[meta_range(Field::Bases, rows, slots).start + slot] = base;
            upload[meta_range(Field::Anchors, rows, slots).start + slot] = anchor;
            upload[meta_range(Field::SlotRings, rows, slots).start + slot] = ring;
        }
        propose
            .uploads
            .copy(device, &propose.buffers.meta, upload)?;
        // SAFETY: ordered after the metadata copy on the engine stream; the readback below
        // fences the replay before any of its buffers is reused.
        unsafe { propose.graph.launch().async_on(&device.stream) }.map_err(device_error)?;
        let tokens = device
            .read_allocation(&propose.buffers.tokens, &mut propose.readbacks)
            .await?;
        requests
            .iter()
            .map(|&(slot, ..)| {
                tokens[slot * block + 1..(slot + 1) * block]
                    .iter()
                    .map(|&token| {
                        u32::try_from(token)
                            .ok()
                            .filter(|&token| (token as usize) < geometry.vocabulary)
                            .ok_or_else(|| Error::invariant("DFlash2 proposed token"))
                    })
                    .collect()
            })
            .collect()
    }
}

fn zeros<T: DType>(device: &CudaDevice, shape: &[usize]) -> Result<Allocation<T>> {
    device.runtime().zeros(shape.iter().product())
}

impl Ingest {
    fn new(device: &CudaDevice, geometry: &Geometry, rows: usize) -> Result<Self> {
        let kv = geometry.kv_heads * geometry.head_dim;
        Ok(Self {
            rows,
            meta: zeros::<i32>(device, &[2 * rows])?,
            normed: zeros::<f32>(device, &[rows, geometry.hidden])?,
            k: zeros::<f32>(device, &[rows, kv])?,
            v: zeros::<f32>(device, &[rows, kv])?,
            k_normed: zeros::<f32>(device, &[rows * geometry.kv_heads, geometry.head_dim])?,
            k_rotated: zeros::<f32>(device, &[rows * geometry.kv_heads, geometry.head_dim])?,
        })
    }
}

impl ProposeBuffers {
    fn new(device: &CudaDevice, geometry: &Geometry, slots: usize) -> Result<Self> {
        let rows = slots * geometry.block;
        let multiprocessors =
            usize::try_from(device.profile()?.multiprocessors).map_err(device_error)?;
        let mut partials = BTreeMap::new();
        let query = geometry.heads * geometry.head_dim;
        let kv = geometry.kv_heads * geometry.head_dim;
        for (n, k) in [
            (query, geometry.hidden),
            (kv, geometry.hidden),
            (geometry.hidden, query),
            (geometry.hidden, geometry.intermediate),
        ] {
            if let Some((splits, k_tiles)) = split_plan(rows, n, k, multiprocessors) {
                partials.insert(
                    (n, k),
                    (
                        zeros::<f32>(device, &[splits * ROW_TILE, n])?,
                        splits,
                        k_tiles,
                    ),
                );
            }
        }
        let hidden = geometry.hidden;
        let query = geometry.heads * geometry.head_dim;
        let kv = geometry.kv_heads * geometry.head_dim;
        let coefficients = 2 * 2 * geometry.groups;
        let top_blocks = geometry.vocabulary.div_ceil(TOP_BLOCK);
        let f = |shape: &[usize]| zeros::<f32>(device, shape);
        Ok(Self {
            slots,
            rows,
            meta: zeros::<i32>(device, &[meta_len(slots * geometry.block, slots)])?,
            stream: [f(&[rows, hidden])?, f(&[rows, hidden])?],
            normed: f(&[rows, hidden])?,
            coefficients: f(&[rows, coefficients])?,
            conv: f(&[rows, hidden])?,
            q: f(&[rows, query])?,
            k: f(&[rows, kv])?,
            v: f(&[rows, kv])?,
            q_normed: f(&[rows * geometry.heads, geometry.head_dim])?,
            k_normed: f(&[rows * geometry.kv_heads, geometry.head_dim])?,
            q_rotated: f(&[rows * geometry.heads, geometry.head_dim])?,
            k_rotated: f(&[rows * geometry.kv_heads, geometry.head_dim])?,
            attention: f(&[
                rows,
                geometry.kv_heads,
                geometry.heads / geometry.kv_heads * geometry.head_dim,
            ])?,
            projected: f(&[rows, hidden])?,
            gate: f(&[rows, geometry.intermediate])?,
            up: f(&[rows, geometry.intermediate])?,
            activated: f(&[rows * geometry.intermediate])?,
            logits: f(&[rows, geometry.vocabulary])?,
            partial_values: f(&[rows, top_blocks * geometry.top_k])?,
            partial_ids: zeros::<i32>(device, &[rows, top_blocks * geometry.top_k])?,
            unary: f(&[rows, geometry.top_k])?,
            candidates: zeros::<i32>(device, &[rows, geometry.top_k])?,
            selector: f(&[rows, geometry.rank])?,
            tokens: zeros::<i32>(device, &[slots, geometry.block])?,
            partials,
        })
    }
}

/// Where one capture's draft launches go.
struct Recorder<'a> {
    draft: &'a dyn Draft,
    recording: &'a Recording<'a>,
}

impl<'a> Recorder<'a> {
    const fn new(draft: &'a dyn Draft, recording: &'a Recording<'a>) -> Self {
        Self { draft, recording }
    }
}

/// A weight in the forms a draft projection supports.
fn draft_weight(weight: &ProjectionWeight) -> StdResult<DraftWeight<'_>, DeviceError> {
    match weight {
        ProjectionWeight::Dense(w) => Ok(DraftWeight::Dense(buffer(w)?)),
        ProjectionWeight::Fp8(w, s) => Ok(DraftWeight::Fp8 {
            weight: buffer(w)?,
            scales: buffer(s)?,
        }),
        ProjectionWeight::Fp8Block(w, s) => Ok(DraftWeight::Fp8Block {
            weight: buffer(w)?,
            scales: buffer(s)?,
            scale_columns: s.columns(),
        }),
        ProjectionWeight::Fp4(..) => Err(DeviceError::Launch(
            "DFlash2 draft projections do not support NVFP4".into(),
        )),
    }
}

/// One field of the metadata buffer.
fn meta_field(
    meta: &Allocation<i32>,
    field: Range<usize>,
) -> StdResult<Buffer<'_, i32>, DeviceError> {
    buffer(meta)?.slice(field).map_err(error)
}

/// Record `out[rows, N] = input[rows, K] · weightᵀ` with the prompt GEMM tiles.
fn record_gemm(
    r: &Recorder<'_>,
    weight: &ProjectionWeight,
    input: Buffer<'_, f32>,
    out: Buffer<'_, f32>,
    (rows, columns): (usize, usize),
) -> StdResult<(), DeviceError> {
    r.draft
        .project(
            r.recording,
            draft_weight(weight)?,
            DraftProjection {
                out,
                input,
                rows,
                columns,
            },
        )
        .map_err(error)
}

/// Split-K geometry `(splits, k_tiles)` for a projection whose output tiles alone would
/// leave most multiprocessors idle; `None` when one row tile cannot use it or the grid is wide.
fn split_plan(rows: usize, n: usize, k: usize, multiprocessors: usize) -> Option<(usize, usize)> {
    let tiles = n / KEY_TILE;
    if rows > ROW_TILE || tiles == 0 || 2 * tiles >= multiprocessors {
        return None;
    }
    let total = k.div_ceil(KEY_TILE);
    let k_tiles = total.div_ceil(multiprocessors.div_ceil(tiles).min(total));
    Some((total.div_ceil(k_tiles), k_tiles))
}

/// [`record_gemm`] that splits K across idle multiprocessors for narrow per-channel FP8
/// outputs, when the proposal owns a partial buffer for this output width.
fn record_projection(
    r: &Recorder<'_>,
    weight: &ProjectionWeight,
    input: Buffer<'_, f32>,
    out: Buffer<'_, f32>,
    (rows, k, partials): (usize, usize, &Partials),
) -> StdResult<(), DeviceError> {
    let n = out.len() / rows;
    let (ProjectionWeight::Fp8(codes, scales), Some((partial, splits, k_tiles))) =
        (weight, partials.get(&(n, k)))
    else {
        return record_gemm(r, weight, input, out, (rows, k));
    };
    r.draft
        .project_split(
            r.recording,
            DraftSplit {
                projection: DraftProjection {
                    out,
                    input,
                    rows,
                    columns: k,
                },
                weight: buffer(codes)?,
                scales: buffer(scales)?,
                partials: buffer(partial)?,
                splits: *splits,
                k_tiles: *k_tiles,
            },
        )
        .map_err(error)
}

fn record_head_norm(
    r: &Recorder<'_>,
    out: Buffer<'_, f32>,
    input: Buffer<'_, f32>,
    weight: Buffer<'_, f32>,
    geometry: &Geometry,
) -> StdResult<(), DeviceError> {
    r.draft
        .norm(
            r.recording,
            DraftNorm {
                out,
                input,
                weight,
                dim: geometry.head_dim,
                tile: geometry.head_dim,
                epsilon: geometry.epsilon,
            },
        )
        .map_err(error)
}

/// Normalization of `hidden` rows, written in whole power-of-two tiles.
const fn hidden_norm<'a>(
    out: Buffer<'a, f32>,
    input: Buffer<'a, f32>,
    weight: Buffer<'a, f32>,
    geometry: &Geometry,
) -> DraftNorm<'a> {
    DraftNorm {
        out,
        input,
        weight,
        dim: geometry.hidden,
        tile: geometry.hidden.next_power_of_two(),
        epsilon: geometry.epsilon,
    }
}

fn record_rope(
    r: &Recorder<'_>,
    (out, input): (Buffer<'_, f32>, Buffer<'_, f32>),
    weights: &DraftWeights,
    positions: Buffer<'_, i32>,
    heads: usize,
) -> StdResult<(), DeviceError> {
    r.draft
        .rope(
            r.recording,
            DraftRope {
                out,
                input,
                frequencies: buffer(weights.frequencies.as_ref())?,
                positions,
                dim: weights.geometry.head_dim,
                heads,
            },
        )
        .map_err(error)
}

/// Store one layer's keys and values, `[rows, kv_heads, head_dim]`, into its rings.
fn record_ring_stores(
    r: &Recorder<'_>,
    (rings, index): (&Rings, usize),
    (keys, values): (Buffer<'_, f32>, Buffer<'_, f32>),
    (ring_ids, positions): (Buffer<'_, i32>, Buffer<'_, i32>),
    (rows, geometry): (usize, &Geometry),
) -> StdResult<(), DeviceError> {
    for (ring, input) in [(&rings.keys[index], keys), (&rings.values[index], values)] {
        r.draft
            .ring_store(
                r.recording,
                DraftRingStore {
                    ring: buffer(ring)?,
                    input,
                    rings: ring_ids,
                    positions,
                    rows,
                    heads: geometry.kv_heads,
                    dim: geometry.head_dim,
                    capacity: rings.count * rings.ring,
                    ring_rows: rings.ring,
                },
            )
            .map_err(error)?;
    }
    Ok(())
}

/// `hidden_norm`, then each layer's K/V projection, K norm and `RoPE`, stored into the rings.
fn record_ingest(
    r: &Recorder<'_>,
    weights: &DraftWeights,
    rings: &Rings,
    work: &Ingest,
    features: Buffer<'_, f32>,
) -> StdResult<(), DeviceError> {
    let geometry = weights.geometry;
    let rows = work.rows;
    let hidden = geometry.hidden;
    let normed = buffer(&work.normed)?;
    r.draft
        .norm(
            r.recording,
            hidden_norm(
                normed,
                features,
                buffer(weights.hidden_norm.as_ref())?,
                &geometry,
            ),
        )
        .map_err(error)?;
    let ring_ids = meta_field(&work.meta, 0..rows)?;
    let positions = meta_field(&work.meta, rows..2 * rows)?;
    let (values_k, values_v) = (buffer(&work.k)?, buffer(&work.v)?);
    let (k_normed, k_rotated) = (buffer(&work.k_normed)?, buffer(&work.k_rotated)?);
    for (index, layer) in weights.layers.iter().enumerate() {
        record_gemm(r, &layer.k, normed, values_k, (rows, hidden))?;
        record_gemm(r, &layer.v, normed, values_v, (rows, hidden))?;
        record_head_norm(
            r,
            k_normed,
            values_k,
            buffer(layer.k_norm.as_ref())?,
            &geometry,
        )?;
        record_rope(
            r,
            (k_rotated, k_normed),
            weights,
            positions,
            geometry.kv_heads,
        )?;
        record_ring_stores(
            r,
            (rings, index),
            (k_rotated, values_v),
            (ring_ids, positions),
            (rows, &geometry),
        )?;
    }
    Ok(())
}

fn record_conv(
    r: &Recorder<'_>,
    (out, input): (Buffer<'_, f32>, Buffer<'_, f32>),
    coefficients: Buffer<'_, f32>,
    base: Buffer<'_, f32>,
    (rows, side): (usize, i32),
    geometry: &Geometry,
) -> StdResult<(), DeviceError> {
    r.draft
        .conv(
            r.recording,
            DraftConv {
                out,
                input,
                coefficients,
                base,
                rows,
                hidden: geometry.hidden,
                tile: CHANNEL_TILE,
                group_size: geometry.group_size,
                groups: geometry.groups,
                block: geometry.block,
                side,
            },
        )
        .map_err(error)
}

/// Record the whole block proposal: embedding, every draft layer, candidates and the walk.
fn record_propose(
    r: &Recorder<'_>,
    weights: &DraftWeights,
    rings: &Rings,
    w: &ProposeBuffers,
) -> StdResult<(), DeviceError> {
    let g = weights.geometry;
    let first = buffer(&w.stream[0])?;
    r.draft
        .embed(
            r.recording,
            [g.hidden, CHANNEL_TILE],
            first,
            buffer(weights.embedding.as_ref())?,
            meta_field(&w.meta, meta_range(Field::Tokens, w.rows, w.slots))?,
        )
        .map_err(error)?;
    r.draft
        .norm(
            r.recording,
            hidden_norm(
                buffer(&w.normed)?,
                first,
                buffer(weights.layers[0].input_norm.as_ref())?,
                &g,
            ),
        )
        .map_err(error)?;
    let mut current = 0;
    for index in 0..weights.layers.len() {
        record_attention_block(r, weights, rings, w, (index, current))?;
        current = 1 - current;
        record_mlp_block(r, weights, w, (index, current))?;
        current = 1 - current;
    }
    record_candidates(r, weights, w)
}

/// The attention block's inputs: the first dynamic convolution, Q/K/V with their norms and
/// `RoPE`, and the block's keys and values stored into the rings.
fn record_attention_inputs(
    r: &Recorder<'_>,
    weights: &DraftWeights,
    rings: &Rings,
    w: &ProposeBuffers,
    index: usize,
) -> StdResult<(), DeviceError> {
    let g = weights.geometry;
    let (rows, slots, hidden) = (w.rows, w.slots, g.hidden);
    let layer = &weights.layers[index];
    let positions = meta_field(&w.meta, meta_range(Field::Positions, rows, slots))?;
    let row_rings = meta_field(&w.meta, meta_range(Field::RowRings, rows, slots))?;
    // Attention block: dynamic convolution around the non-causal block attention.
    let conv = &layer.attention_conv;
    let normed = buffer(&w.normed)?;
    let convolved = buffer(&w.conv)?;
    let coefficients = buffer(&w.coefficients)?;
    let base = buffer(conv.base.as_ref())?;
    record_gemm(r, &conv.projection, normed, coefficients, (rows, hidden))?;
    record_conv(r, (convolved, normed), coefficients, base, (rows, 0), &g)?;
    let split = (rows, hidden, &w.partials);
    let (queries, keys, values) = (buffer(&w.q)?, buffer(&w.k)?, buffer(&w.v)?);
    let (q_normed, k_normed) = (buffer(&w.q_normed)?, buffer(&w.k_normed)?);
    let (q_rotated, k_rotated) = (buffer(&w.q_rotated)?, buffer(&w.k_rotated)?);
    record_projection(r, &layer.q, convolved, queries, split)?;
    record_projection(r, &layer.k, convolved, keys, split)?;
    record_projection(r, &layer.v, convolved, values, split)?;
    record_head_norm(r, q_normed, queries, buffer(layer.q_norm.as_ref())?, &g)?;
    record_head_norm(r, k_normed, keys, buffer(layer.k_norm.as_ref())?, &g)?;
    record_rope(r, (q_rotated, q_normed), weights, positions, g.heads)?;
    record_rope(r, (k_rotated, k_normed), weights, positions, g.kv_heads)?;
    record_ring_stores(
        r,
        (rings, index),
        (k_rotated, values),
        (row_rings, positions),
        (rows, &g),
    )
}

/// The stream holding the residual and the one the update writes.
fn streams(
    w: &ProposeBuffers,
    current: usize,
) -> StdResult<(Buffer<'_, f32>, Buffer<'_, f32>), DeviceError> {
    let [first, second] = &w.stream;
    let (first, second) = (buffer(first)?, buffer(second)?);
    Ok(if current == 0 {
        (first, second)
    } else {
        (second, first)
    })
}

/// One draft layer's attention block: dynamic convolution around the non-causal block
/// attention, then the residual update into the other stream buffer.
fn record_attention_block(
    r: &Recorder<'_>,
    weights: &DraftWeights,
    rings: &Rings,
    w: &ProposeBuffers,
    (index, current): (usize, usize),
) -> StdResult<(), DeviceError> {
    let g = weights.geometry;
    let (rows, slots) = (w.rows, w.slots);
    let query = g.heads * g.head_dim;
    let layer = &weights.layers[index];
    record_attention_inputs(r, weights, rings, w, index)?;
    let (residual, next) = streams(w, current)?;
    let conv = &layer.attention_conv;
    r.draft
        .attend(
            r.recording,
            DraftAttention {
                out: buffer(&w.attention)?,
                queries: buffer(&w.q_rotated)?,
                keys: buffer(&rings.keys[index])?,
                values: buffer(&rings.values[index])?,
                bases: meta_field(&w.meta, meta_range(Field::Bases, rows, slots))?,
                rings: meta_field(&w.meta, meta_range(Field::SlotRings, rows, slots))?,
                rows,
                kv_heads: g.kv_heads,
                group: g.heads / g.kv_heads,
                dim: g.head_dim,
                capacity: rings.count * rings.ring,
                block: g.block,
                ring_rows: rings.ring,
                window: g.window,
            },
        )
        .map_err(error)?;
    let (projected, convolved) = (buffer(&w.projected)?, buffer(&w.conv)?);
    record_projection(
        r,
        &layer.o,
        buffer(&w.attention)?,
        projected,
        (rows, query, &w.partials),
    )?;
    record_conv(
        r,
        (convolved, projected),
        buffer(&w.coefficients)?,
        buffer(conv.base.as_ref())?,
        (rows, 1),
        &g,
    )?;
    r.draft
        .add_norm(
            r.recording,
            DraftAddNorm {
                norm: hidden_norm(
                    buffer(&w.normed)?,
                    residual,
                    buffer(layer.post_norm.as_ref())?,
                    &g,
                ),
                stream: next,
                delta: convolved,
            },
        )
        .map_err(error)
}

/// One draft layer's MLP block inside its dynamic convolution, then the residual update and
/// the next layer's (or the final) norm.
fn record_mlp_block(
    r: &Recorder<'_>,
    weights: &DraftWeights,
    w: &ProposeBuffers,
    (index, current): (usize, usize),
) -> StdResult<(), DeviceError> {
    let g = weights.geometry;
    let (rows, hidden) = (w.rows, g.hidden);
    let layer = &weights.layers[index];
    let (residual, next) = streams(w, current)?;
    // MLP block, wrapped in its own dynamic convolution.
    let conv = &layer.mlp_conv;
    let normed = buffer(&w.normed)?;
    let convolved = buffer(&w.conv)?;
    let coefficients = buffer(&w.coefficients)?;
    let base = buffer(conv.base.as_ref())?;
    let (gate, up) = (buffer(&w.gate)?, buffer(&w.up)?);
    let (activated, projected) = (buffer(&w.activated)?, buffer(&w.projected)?);
    record_gemm(r, &conv.projection, normed, coefficients, (rows, hidden))?;
    record_conv(r, (convolved, normed), coefficients, base, (rows, 0), &g)?;
    record_gemm(r, &layer.gate, convolved, gate, (rows, hidden))?;
    record_gemm(r, &layer.up, convolved, up, (rows, hidden))?;
    r.draft
        .gated_product(r.recording, activated, gate, up)
        .map_err(error)?;
    record_projection(
        r,
        &layer.down,
        activated,
        projected,
        (rows, g.intermediate, &w.partials),
    )?;
    record_conv(r, (convolved, projected), coefficients, base, (rows, 1), &g)?;
    let norm = weights
        .layers
        .get(index + 1)
        .map_or(&weights.norm, |layer| &layer.input_norm);
    r.draft
        .add_norm(
            r.recording,
            DraftAddNorm {
                norm: hidden_norm(normed, residual, buffer(norm.as_ref())?, &g),
                stream: next,
                delta: convolved,
            },
        )
        .map_err(error)
}

/// Top candidates of every row from the target head, then the selector walk over them.
fn record_candidates(
    r: &Recorder<'_>,
    weights: &DraftWeights,
    w: &ProposeBuffers,
) -> StdResult<(), DeviceError> {
    let g = weights.geometry;
    let (rows, slots, hidden) = (w.rows, w.slots, g.hidden);
    let normed = buffer(&w.normed)?;
    record_gemm(r, &weights.head, normed, buffer(&w.logits)?, (rows, hidden))?;
    r.draft
        .top_candidates(
            r.recording,
            DraftTop {
                partial_values: buffer(&w.partial_values)?,
                partial_ids: buffer(&w.partial_ids)?,
                unary: buffer(&w.unary)?,
                candidates: buffer(&w.candidates)?,
                logits: buffer(&w.logits)?,
                rows,
                vocabulary: g.vocabulary,
                top_block: TOP_BLOCK,
                top_k: g.top_k,
            },
        )
        .map_err(error)?;
    record_gemm(
        r,
        &weights.selector,
        normed,
        buffer(&w.selector)?,
        (rows, hidden),
    )?;
    r.draft
        .walk(
            r.recording,
            DraftWalk {
                tokens: buffer(&w.tokens)?,
                candidates: buffer(&w.candidates)?,
                unary: buffer(&w.unary)?,
                selector: buffer(&w.selector)?,
                predecessor: buffer(weights.predecessor.as_ref())?,
                successor: buffer(weights.successor.as_ref())?,
                anchors: meta_field(&w.meta, meta_range(Field::Anchors, rows, slots))?,
                rows,
                slots,
                rank: g.rank,
                top_k: g.top_k,
                block: g.block,
            },
        )
        .map_err(error)
}
