//! `DFlash2` block drafting beside the target's pooled verification.
//!
//! The draft reads the target through one fused feature per committed token (the target graph's
//! `features` result). Every feature is turned into draft K/V once, in [`Engine::ingest_prompt`] or
//! [`Engine::ingest_verified`], and lands in its sequence's sliding-window ring; a proposal then
//! runs the five draft layers over one block `[anchor, mask * (block - 1)]` per pooled slot, picks
//! the top candidates of every masked row from the target's head and walks the selector lattice.
//! Rings belong to sequences, not pool slots, because a prompt is ingested before its sequence
//! is bound to a slot.
pub(crate) mod kernels;
mod weights;
pub(crate) use weights::DraftWeights;

use super::prefill_gemm::gemm;
use crate::device::{CudaDevice, PinnedUpload, Readbacks, device_error};
use crate::mlp::ProjectionWeight;
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, Result};
use kernels::draft;
use std::result::Result as StdResult;
use std::sync::Arc;
use weights::Geometry;

/// Keys one attention step reads; ring lengths are whole multiples of it.
const KEY_TILE: usize = 64;
/// Rows of one GEMM output tile.
const ROW_TILE: usize = 32;
/// Channels one embedding or convolution block handles.
const CHANNEL_TILE: usize = 512;
/// Vocabulary entries one top-k block reduces.
const TOP_BLOCK: usize = 2048;
/// Prompt features one ingestion replay consumes.
const PROMPT_ROWS: usize = 256;
/// Pinned metadata uploads a prompt ingestion queues before it fences: the widest prompt chunk
/// split into [`PROMPT_ROWS`] replays, plus one.
const PROMPT_UPLOADS: usize = crate::constants::WIDEST_PREFILL_LANES / PROMPT_ROWS + 1;

/// One sequence's place in the rings and how much of its timeline the draft has seen.
#[derive(Debug)]
pub(crate) struct Cursor {
    pub ring: usize,
    /// Positions `0..context` have draft K/V in the ring.
    pub context: usize,
}

/// Per-layer K/V rings, `[kv_heads, rings * ring, head_dim]` each.
struct Rings {
    keys: Vec<Tensor<bf16>>,
    values: Vec<Tensor<bf16>>,
    /// Rows of one ring: the window plus one block, in whole key tiles.
    ring: usize,
    count: usize,
    free: Vec<usize>,
}

/// K/V projection of target features for `rows` rows.
struct Ingest {
    rows: usize,
    /// `[rows]` ring ids then `[rows]` positions; a negative position skips the row.
    meta: Tensor<i32>,
    normed: Tensor<f32>,
    k: Tensor<f32>,
    v: Tensor<f32>,
    k_normed: Tensor<f32>,
    k_rotated: Tensor<f32>,
}

/// Prompt ingestion owns its feature rows; the target's prompt graph writes elsewhere.
struct PromptIngest {
    features: Tensor<f32>,
    work: Ingest,
    graph: CudaGraph<()>,
    uploads: PinnedUpload<i32>,
}

/// Verified-row ingestion reads the pooled verify graph's feature rows in place.
struct VerifiedIngest {
    graph: CudaGraph<()>,
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
    meta: Tensor<i32>,
    stream: [Tensor<f32>; 2],
    normed: Tensor<f32>,
    coefficients: Tensor<f32>,
    conv: Tensor<f32>,
    q: Tensor<f32>,
    k: Tensor<f32>,
    v: Tensor<f32>,
    q_normed: Tensor<f32>,
    k_normed: Tensor<f32>,
    q_rotated: Tensor<f32>,
    k_rotated: Tensor<f32>,
    attention: Tensor<f32>,
    projected: Tensor<f32>,
    gate: Tensor<f32>,
    up: Tensor<f32>,
    activated: Tensor<f32>,
    logits: Tensor<f32>,
    partial_values: Tensor<f32>,
    partial_ids: Tensor<i32>,
    unary: Tensor<f32>,
    candidates: Tensor<i32>,
    selector: Tensor<f32>,
    tokens: Tensor<i32>,
}

struct Propose {
    graph: CudaGraph<()>,
    buffers: ProposeBuffers,
    uploads: PinnedUpload<i32>,
    readbacks: Readbacks<i32>,
    upload: Vec<i32>,
}

/// One slot's block request: pool slot, the sequence's ring, the anchor token and its position.
pub(crate) type BlockRequest = (usize, usize, u32, usize);

/// One verified run of feature rows to ingest: first lane in the verify graph, ring, first
/// position and row count.
pub(crate) type VerifiedRows = (usize, usize, usize, usize);

/// The pooled graphs captured against one verify graph of `slots` slots.
struct Pooled {
    slots: usize,
    propose: Propose,
    verified: VerifiedIngest,
}

/// Draft state shared by every sequence of one executor.
pub(crate) struct Engine {
    /// Verified blocks and the tokens they committed, for the acceptance-length report.
    verified_blocks: u64,
    committed_tokens: u64,
    // Graphs drop before the buffers and rings their kernels reference.
    /// One proposal and ingestion pair per pooled verify graph, narrowest first.
    pooled: Vec<Pooled>,
    prompt: PromptIngest,
    rings: Rings,
    weights: Arc<DraftWeights>,
}

impl Engine {
    /// Allocate `rings` sequence rings and capture prompt ingestion.
    /// # Errors
    /// Rejects an empty ring count and reports CUDA allocation or capture failures.
    pub(crate) fn new(
        device: &CudaDevice,
        weights: Arc<DraftWeights>,
        rings: usize,
    ) -> Result<Self> {
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
        let mut rings = Rings {
            keys,
            values,
            ring,
            count: rings,
            free: (0..rings).rev().collect(),
        };
        let mut features = zeros::<f32>(device, &[PROMPT_ROWS, geometry.hidden])?;
        let mut work = Ingest::new(device, &geometry, PROMPT_ROWS)?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            record_ingest(scope, &weights, &mut rings, &mut work, &mut features)
        })
        .map_err(device_error)?;
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
        })
    }

    /// Device bytes one ring of every layer occupies.
    pub(crate) fn ring_bytes(
        geometry_window: usize,
        block: usize,
        layers: usize,
        kv: usize,
    ) -> u64 {
        let ring = (geometry_window + block).div_ceil(KEY_TILE) * KEY_TILE;
        (2 * layers * ring * kv * size_of::<bf16>()) as u64
    }

    /// Lease a ring to a new sequence, if one is free.
    pub(crate) fn acquire(&mut self) -> Option<Cursor> {
        self.rings
            .free
            .pop()
            .map(|ring| Cursor { ring, context: 0 })
    }

    /// Return a sequence's ring.
    pub(crate) fn release(&mut self, cursor: Cursor) {
        if cursor.ring < self.rings.count && !self.rings.free.contains(&cursor.ring) {
            self.rings.free.push(cursor.ring);
        }
    }

    /// Capture one pooled proposal per verify graph of the pool and bind verified-row
    /// ingestion to that graph's feature rows. `graphs` lists each graph's slot count and
    /// feature tensor.
    /// # Errors
    /// Rejects a feature tensor that is not `slots * block` rows wide and reports CUDA failures.
    pub(crate) fn attach_pool(
        &mut self,
        device: &CudaDevice,
        graphs: Vec<(usize, Arc<Tensor<f32>>)>,
    ) -> Result<()> {
        self.detach_pool();
        let mut pooled = Vec::with_capacity(graphs.len());
        for (slots, features) in graphs {
            pooled.push(self.capture_pooled(device, slots, features)?);
        }
        pooled.sort_by_key(|graphs| graphs.slots);
        self.pooled = pooled;
        Ok(())
    }

    fn capture_pooled(
        &mut self,
        device: &CudaDevice,
        slots: usize,
        features: Arc<Tensor<f32>>,
    ) -> Result<Pooled> {
        let geometry = self.weights.geometry;
        let rows = slots * geometry.block;
        if features.size() != rows * geometry.hidden {
            return Err(Error::invalid("verify features do not match the pool"));
        }
        let mut work = Ingest::new(device, &geometry, rows)?;
        let weights = Arc::clone(&self.weights);
        let rings = &mut self.rings;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            let input = features.view(&[rows, geometry.hidden])?;
            record_ingest_from(scope, &weights, rings, &mut work, &input)
        })
        .map_err(device_error)?;
        let verified = VerifiedIngest {
            graph,
            work,
            uploads: PinnedUpload::single(device, 2 * rows)?,
            _features: features,
        };
        let mut buffers = ProposeBuffers::new(device, &geometry, slots)?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            record_propose(scope, &weights, rings, &mut buffers)
        })
        .map_err(device_error)?;
        let meta = buffers.meta.size();
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
    pub(crate) fn detach_pool(&mut self) {
        self.pooled.clear();
    }

    pub(crate) fn attached(&self) -> bool {
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
    pub(crate) async fn ingest_prompt(
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
            device.copy_d2d_at(
                &mut self.prompt.features,
                features,
                0,
                start * hidden,
                count * hidden,
            )?;
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
                .copy(device, &mut self.prompt.work.meta, &upload)?;
            // SAFETY: ordered on the engine stream after the copies above; every buffer the
            // graph touches is engine-owned and the engine drains the stream before dropping.
            unsafe { self.prompt.graph.launch().async_on(&device.stream) }.map_err(device_error)?;
            start += count;
        }
        // The pinned ring above holds one chunk; fence it before the next chunk reuses it.
        let _ = device
            .read_async(&self.prompt.work.meta, &mut Readbacks::default())
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
    pub(crate) fn ingest_verified(
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
        for row in 0..rows {
            upload[row] = 0;
        }
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
            .copy(device, &mut verified.work.meta, &upload)?;
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
    pub(crate) async fn propose(
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
        upload.resize(3 * rows + 3 * slots, 0);
        for row in 0..rows {
            upload[row] = mask;
            upload[rows + row] = -1;
        }
        for slot in 0..slots {
            upload[3 * rows + slot] = -1;
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
                upload[2 * rows + row] = ring;
            }
            upload[3 * rows + slot] = base;
            upload[3 * rows + slots + slot] = anchor;
            upload[3 * rows + 2 * slots + slot] = ring;
        }
        propose
            .uploads
            .copy(device, &mut propose.buffers.meta, upload)?;
        // SAFETY: ordered after the metadata copy on the engine stream; the readback below
        // fences the replay before any of its buffers is reused.
        unsafe { propose.graph.launch().async_on(&device.stream) }.map_err(device_error)?;
        let tokens = device
            .read_async(&propose.buffers.tokens, &mut propose.readbacks)
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

fn zeros<T: DType>(device: &CudaDevice, shape: &[usize]) -> Result<Tensor<T>> {
    api::zeros::<T>(shape)
        .sync_on(&device.stream)
        .map_err(device_error)
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
        let hidden = geometry.hidden;
        let query = geometry.heads * geometry.head_dim;
        let kv = geometry.kv_heads * geometry.head_dim;
        let coefficients = 2 * 2 * geometry.groups;
        let top_blocks = geometry.vocabulary.div_ceil(TOP_BLOCK);
        let f = |shape: &[usize]| zeros::<f32>(device, shape);
        Ok(Self {
            slots,
            rows,
            meta: zeros::<i32>(device, &[3 * rows + 3 * slots])?,
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
        })
    }
}

/// Record `out[rows, N] = input[rows, K] · weightᵀ` with the prompt GEMM tiles.
fn record_gemm(
    scope: &Scope,
    weight: &ProjectionWeight,
    input: &TensorView<'_, f32>,
    out: &mut Tensor<f32>,
    k: usize,
) -> StdResult<(), DeviceError> {
    let generics = vec![k.to_string()];
    match weight {
        ProjectionWeight::Dense(w) => {
            scope.record(
                gemm::dense(out.partition([ROW_TILE, KEY_TILE]), input, w).generics(generics),
            )?;
        }
        ProjectionWeight::Fp8(w, s) => {
            scope.record(
                gemm::fp8(out.partition([ROW_TILE, KEY_TILE]), input, w, s).generics(generics),
            )?;
        }
        ProjectionWeight::Fp8Block(w, s) => {
            scope.record(
                draft::fp8_block(out.partition([ROW_TILE, KEY_TILE]), input, w, s)
                    .generics(generics),
            )?;
        }
        ProjectionWeight::Fp4(..) => {
            return Err(DeviceError::Launch(
                "DFlash2 draft projections do not support NVFP4".into(),
            ));
        }
    }
    Ok(())
}

fn record_head_norm(
    scope: &Scope,
    out: &mut Tensor<f32>,
    input: &TensorView<'_, f32>,
    weight: &Tensor<f32>,
    geometry: &Geometry,
) -> StdResult<(), DeviceError> {
    scope.record(
        draft::norm(
            out.partition([1, geometry.head_dim]),
            input,
            weight,
            geometry.epsilon,
        )
        .generics(vec![
            geometry.head_dim.to_string(),
            geometry.head_dim.to_string(),
        ]),
    )?;
    Ok(())
}

fn record_rope(
    scope: &Scope,
    out: &mut Tensor<f32>,
    input: &TensorView<'_, f32>,
    weights: &DraftWeights,
    positions: &TensorView<'_, i32>,
    heads: usize,
) -> StdResult<(), DeviceError> {
    let half = weights.geometry.head_dim / 2;
    scope.record(
        draft::rope_rows(
            out.partition([1, half]),
            input,
            &weights.frequencies,
            positions,
        )
        .generics(vec![half.to_string(), heads.to_string()]),
    )?;
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "one ring store binds the ring, rows and their per-row metadata"
)]
fn record_ring_store(
    scope: &Scope,
    ring: &mut Tensor<bf16>,
    rows: &TensorView<'_, f32>,
    count: usize,
    rings: &TensorView<'_, i32>,
    positions: &TensorView<'_, i32>,
    capacity: usize,
    geometry: &Geometry,
    ring_rows: usize,
) -> StdResult<(), DeviceError> {
    scope.record(
        draft::ring_store(
            ring.partition([1, capacity, geometry.head_dim]),
            rows,
            rings,
            positions,
        )
        .generics(vec![
            geometry.head_dim.to_string(),
            capacity.to_string(),
            ring_rows.to_string(),
            count.to_string(),
        ]),
    )?;
    Ok(())
}

fn record_ingest(
    scope: &Scope,
    weights: &DraftWeights,
    rings: &mut Rings,
    work: &mut Ingest,
    features: &mut Tensor<f32>,
) -> StdResult<(), DeviceError> {
    let input = features.view(&[work.rows, weights.geometry.hidden])?;
    record_ingest_from(scope, weights, rings, work, &input)
}

/// `hidden_norm`, then each layer's K/V projection, K norm and `RoPE`, stored into the rings.
fn record_ingest_from(
    scope: &Scope,
    weights: &DraftWeights,
    rings: &mut Rings,
    work: &mut Ingest,
    features: &TensorView<'_, f32>,
) -> StdResult<(), DeviceError> {
    let geometry = weights.geometry;
    let rows = work.rows;
    let hidden = geometry.hidden;
    scope.record(
        draft::norm(
            (&mut work.normed).partition([1, hidden.next_power_of_two()]),
            features,
            &weights.hidden_norm,
            geometry.epsilon,
        )
        .generics(vec![
            hidden.to_string(),
            hidden.next_power_of_two().to_string(),
        ]),
    )?;
    let capacity = rings.count * rings.ring;
    for (index, layer) in weights.layers.iter().enumerate() {
        record_gemm(
            scope,
            &layer.k,
            &work.normed.view(&[rows, hidden])?,
            &mut work.k,
            hidden,
        )?;
        record_gemm(
            scope,
            &layer.v,
            &work.normed.view(&[rows, hidden])?,
            &mut work.v,
            hidden,
        )?;
        let k_heads = work
            .k
            .view(&[rows * geometry.kv_heads, geometry.head_dim])?;
        record_head_norm(
            scope,
            &mut work.k_normed,
            &k_heads,
            &layer.k_norm,
            &geometry,
        )?;
        let ring_ids = work.meta.slice(&[0..rows])?;
        let positions = work.meta.slice(&[rows..2 * rows])?;
        record_rope(
            scope,
            &mut work.k_rotated,
            &work
                .k_normed
                .view(&[rows * geometry.kv_heads, geometry.head_dim])?,
            weights,
            &positions,
            geometry.kv_heads,
        )?;
        let keys = work
            .k_rotated
            .view(&[rows, geometry.kv_heads, geometry.head_dim])?;
        let values = work.v.view(&[rows, geometry.kv_heads, geometry.head_dim])?;
        record_ring_store(
            scope,
            &mut rings.keys[index],
            &keys,
            rows,
            &ring_ids,
            &positions,
            capacity,
            &geometry,
            rings.ring,
        )?;
        record_ring_store(
            scope,
            &mut rings.values[index],
            &values,
            rows,
            &ring_ids,
            &positions,
            capacity,
            &geometry,
            rings.ring,
        )?;
    }
    Ok(())
}

fn record_conv(
    scope: &Scope,
    out: &mut Tensor<f32>,
    input: &TensorView<'_, f32>,
    coefficients: &Tensor<f32>,
    base: &Tensor<f32>,
    side: i32,
    geometry: &Geometry,
) -> StdResult<(), DeviceError> {
    let base = base.view(&[4, geometry.hidden])?;
    scope.record(
        draft::grouped_conv(
            out.partition([1, CHANNEL_TILE]),
            input,
            coefficients,
            &base,
            side,
        )
        .generics(vec![
            CHANNEL_TILE.to_string(),
            geometry.group_size.to_string(),
            (CHANNEL_TILE / geometry.group_size).to_string(),
            geometry.groups.to_string(),
            geometry.block.to_string(),
        ]),
    )?;
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "one straight-line recording of the five draft layers, head and selector"
)]
fn record_propose(
    scope: &Scope,
    weights: &DraftWeights,
    rings: &mut Rings,
    w: &mut ProposeBuffers,
) -> StdResult<(), DeviceError> {
    let g = weights.geometry;
    let rows = w.rows;
    let slots = w.slots;
    let hidden = g.hidden;
    let query = g.heads * g.head_dim;
    let group = g.heads / g.kv_heads;
    let capacity = rings.count * rings.ring;
    let meta = &w.meta;
    let tokens = meta.slice(&[0..rows])?;
    let positions = meta.slice(&[rows..2 * rows])?;
    let row_rings = meta.slice(&[2 * rows..3 * rows])?;
    let bases = meta.slice(&[3 * rows..3 * rows + slots])?;
    let anchors = meta.slice(&[3 * rows + slots..3 * rows + 2 * slots])?;
    let slot_rings = meta.slice(&[3 * rows + 2 * slots..3 * rows + 3 * slots])?;
    let [first, second] = &mut w.stream;
    scope.record(
        draft::embed(
            first.partition([1, CHANNEL_TILE]),
            &weights.embedding,
            &tokens,
        )
        .generics(vec![CHANNEL_TILE.to_string()]),
    )?;
    let input_norm = &weights.layers[0].input_norm;
    scope.record(
        draft::norm(
            (&mut w.normed).partition([1, hidden.next_power_of_two()]),
            &*first,
            input_norm,
            g.epsilon,
        )
        .generics(vec![
            hidden.to_string(),
            hidden.next_power_of_two().to_string(),
        ]),
    )?;
    let (mut residual, mut next) = (first, second);
    for (index, layer) in weights.layers.iter().enumerate() {
        // Attention block: dynamic convolution around the non-causal block attention.
        let conv = &layer.attention_conv;
        record_gemm(
            scope,
            &conv.projection,
            &w.normed.view(&[rows, hidden])?,
            &mut w.coefficients,
            hidden,
        )?;
        record_conv(
            scope,
            &mut w.conv,
            &w.normed.view(&[rows, hidden])?,
            &w.coefficients,
            &conv.base,
            0,
            &g,
        )?;
        record_gemm(
            scope,
            &layer.q,
            &w.conv.view(&[rows, hidden])?,
            &mut w.q,
            hidden,
        )?;
        record_gemm(
            scope,
            &layer.k,
            &w.conv.view(&[rows, hidden])?,
            &mut w.k,
            hidden,
        )?;
        record_gemm(
            scope,
            &layer.v,
            &w.conv.view(&[rows, hidden])?,
            &mut w.v,
            hidden,
        )?;
        let q_heads = w.q.view(&[rows * g.heads, g.head_dim])?;
        record_head_norm(scope, &mut w.q_normed, &q_heads, &layer.q_norm, &g)?;
        let k_heads = w.k.view(&[rows * g.kv_heads, g.head_dim])?;
        record_head_norm(scope, &mut w.k_normed, &k_heads, &layer.k_norm, &g)?;
        record_rope(
            scope,
            &mut w.q_rotated,
            &w.q_normed.view(&[rows * g.heads, g.head_dim])?,
            weights,
            &positions,
            g.heads,
        )?;
        record_rope(
            scope,
            &mut w.k_rotated,
            &w.k_normed.view(&[rows * g.kv_heads, g.head_dim])?,
            weights,
            &positions,
            g.kv_heads,
        )?;
        let keys = w.k_rotated.view(&[rows, g.kv_heads, g.head_dim])?;
        let values = w.v.view(&[rows, g.kv_heads, g.head_dim])?;
        record_ring_store(
            scope,
            &mut rings.keys[index],
            &keys,
            rows,
            &row_rings,
            &positions,
            capacity,
            &g,
            rings.ring,
        )?;
        record_ring_store(
            scope,
            &mut rings.values[index],
            &values,
            rows,
            &row_rings,
            &positions,
            capacity,
            &g,
            rings.ring,
        )?;
        let queries = w.q_rotated.view(&[rows, g.kv_heads, group * g.head_dim])?;
        let key_ring = rings.keys[index].view(&[g.kv_heads, capacity, g.head_dim])?;
        let value_ring = rings.values[index].view(&[g.kv_heads, capacity, g.head_dim])?;
        scope.record(
            draft::block_attention(
                (&mut w.attention).partition([g.block, 1, group * g.head_dim]),
                &queries,
                &key_ring,
                &value_ring,
                &bases,
                &slot_rings,
            )
            .generics(vec![
                g.head_dim.to_string(),
                group.to_string(),
                (group * g.head_dim).to_string(),
                g.block.to_string(),
                (g.block * group).to_string(),
                rings.ring.to_string(),
                g.window.to_string(),
            ]),
        )?;
        let attended = w.attention.view(&[rows, query])?;
        record_gemm(scope, &layer.o, &attended, &mut w.projected, query)?;
        record_conv(
            scope,
            &mut w.conv,
            &w.projected.view(&[rows, hidden])?,
            &w.coefficients,
            &conv.base,
            1,
            &g,
        )?;
        scope.record(
            draft::add_norm(
                (&mut w.normed).partition([1, hidden.next_power_of_two()]),
                (&mut *next).partition([1, hidden.next_power_of_two()]),
                &*residual,
                &w.conv,
                &layer.post_norm,
                g.epsilon,
            )
            .generics(vec![
                hidden.to_string(),
                hidden.next_power_of_two().to_string(),
            ]),
        )?;
        (residual, next) = (next, residual);
        // MLP block, wrapped in its own dynamic convolution.
        let conv = &layer.mlp_conv;
        record_gemm(
            scope,
            &conv.projection,
            &w.normed.view(&[rows, hidden])?,
            &mut w.coefficients,
            hidden,
        )?;
        record_conv(
            scope,
            &mut w.conv,
            &w.normed.view(&[rows, hidden])?,
            &w.coefficients,
            &conv.base,
            0,
            &g,
        )?;
        record_gemm(
            scope,
            &layer.gate,
            &w.conv.view(&[rows, hidden])?,
            &mut w.gate,
            hidden,
        )?;
        record_gemm(
            scope,
            &layer.up,
            &w.conv.view(&[rows, hidden])?,
            &mut w.up,
            hidden,
        )?;
        let gate = w.gate.view(&[rows * g.intermediate])?;
        let up = w.up.view(&[rows * g.intermediate])?;
        scope.record(draft::silu_mul(
            (&mut w.activated).partition([256]),
            &gate,
            &up,
        ))?;
        let activated = w.activated.view(&[rows, g.intermediate])?;
        record_gemm(
            scope,
            &layer.down,
            &activated,
            &mut w.projected,
            g.intermediate,
        )?;
        record_conv(
            scope,
            &mut w.conv,
            &w.projected.view(&[rows, hidden])?,
            &w.coefficients,
            &conv.base,
            1,
            &g,
        )?;
        let norm = weights
            .layers
            .get(index + 1)
            .map_or(&weights.norm, |layer| &layer.input_norm);
        scope.record(
            draft::add_norm(
                (&mut w.normed).partition([1, hidden.next_power_of_two()]),
                (&mut *next).partition([1, hidden.next_power_of_two()]),
                &*residual,
                &w.conv,
                norm,
                g.epsilon,
            )
            .generics(vec![
                hidden.to_string(),
                hidden.next_power_of_two().to_string(),
            ]),
        )?;
        (residual, next) = (next, residual);
    }
    // Candidates from the target head, then the selector walk over them.
    record_gemm(
        scope,
        &weights.head,
        &w.normed.view(&[rows, hidden])?,
        &mut w.logits,
        hidden,
    )?;
    let top_blocks = g.vocabulary.div_ceil(TOP_BLOCK);
    scope.record(
        draft::top_partial(
            (&mut w.partial_values).partition([1, g.top_k]),
            (&mut w.partial_ids).partition([1, g.top_k]),
            &w.logits,
        )
        .generics(vec![
            g.vocabulary.to_string(),
            TOP_BLOCK.to_string(),
            g.top_k.to_string(),
        ]),
    )?;
    let merged = top_blocks * g.top_k;
    scope.record(
        draft::top_finish(
            (&mut w.unary).partition([1, g.top_k]),
            (&mut w.candidates).partition([1, g.top_k]),
            &w.partial_values,
            &w.partial_ids,
        )
        .generics(vec![
            merged.to_string(),
            merged.next_power_of_two().to_string(),
            g.top_k.to_string(),
        ]),
    )?;
    record_gemm(
        scope,
        &weights.selector,
        &w.normed.view(&[rows, hidden])?,
        &mut w.selector,
        hidden,
    )?;
    scope.record(
        draft::select_walk(
            (&mut w.tokens).partition([1, g.block]),
            &w.candidates,
            &w.unary,
            &w.selector,
            &weights.predecessor,
            &weights.successor,
            &anchors,
        )
        .generics(vec![
            g.rank.to_string(),
            g.top_k.to_string(),
            g.block.to_string(),
        ]),
    )?;
    Ok(())
}
