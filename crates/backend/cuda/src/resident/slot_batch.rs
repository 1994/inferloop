//! Continuous-batching slot pool: `width` pre-allocated state sets and one shared
//! decode graph whose lane `i` is baked onto slot `i`. A sequence binds a slot by
//! device-copying its covered state rows in, then every leased sequence advances
//! through a single graph replay per step.
use super::{
    batch::{BatchOutput, SlotDecodeGraph, SlotLane, States},
    fp8_cache::{Fp8Caches, allocate},
    program::ProgramWeights,
    slot_states::SlotStates,
    slot_verify::{SlotVerifyGraph, SlotVerifyLane},
};
use crate::constants::F32_BYTES;
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::MAX_POOL_SLOTS;
use crate::constants::MAX_VERIFICATION_WIDTH;
use crate::device::PinnedUpload;
#[cfg(test)]
use crate::device::asynchronous::block_on;
use crate::device::{CudaDevice, DeviceProfile, device_error};
use cuda_core::{CudaContext, PinnedHostBuffer, f8e4m3fn};
use cutile::prelude::*;
use infer_core::{Error, Result, TensorId};
use infer_cuda_kernels::contract::Owner;
use infer_ir::{DataflowGraph, StateKind, TensorStorage};
use infer_state::blocks::{BlockLease, BlockPool};
#[cfg(test)]
use std::cell::Cell;
use std::iter::successors;

/// Widest plain-decode rung; above it the full-width graph's batched state kernels win.
const MAX_NARROW_DECODE_SLOTS: usize = 4;

/// Slot counts of the narrow graphs a pool of `width` slots captures: powers of two below
/// the width. A plain decode pool replays one lane per busy slot, so its rungs keep a step
/// with few live sequences from paying for every lane the configured concurrency allows. A
/// verify pool needs fused recurrent state before narrower verification is valid.
pub fn narrow_widths(graph: &DataflowGraph, width: usize, verify: usize) -> Vec<usize> {
    if verify < 2 {
        return successors(Some(1usize), |slots| slots.checked_mul(2))
            .take_while(|&slots| slots <= MAX_NARROW_DECODE_SLOTS && slots < width)
            .collect();
    }
    if !super::slot_verify::fused_recurrent(graph, verify) {
        return Vec::new();
    }
    successors(Some(1usize), |slots| slots.checked_mul(2))
        .take_while(|&slots| slots < width)
        .collect()
}

/// A single slot is useful for speculative verification; plain batching needs two.
fn valid_pool_geometry(
    width: usize,
    verify: usize,
    capacity: usize,
    hidden: usize,
    vocabulary: usize,
) -> bool {
    let minimum = if verify >= 2 { 1 } else { 2 };
    (minimum..=MAX_POOL_SLOTS).contains(&width) && capacity > 0 && hidden > 0 && vocabulary > 0
}

/// NVFP4 workspace for every lane count the pool's graphs replay.
fn pool_workspace(
    device: &CudaDevice,
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    (lanes, verify): (usize, usize),
    narrow: &[usize],
) -> Result<super::nvfp4_gemm::Workspace> {
    let mut widths = vec![1, lanes];
    // A pool without verification has `verify == 0` and one lane per slot.
    widths.extend(narrow.iter().map(|&slots| slots * verify.max(1)));
    let mut workspace = super::nvfp4_gemm::Workspace::new(device, graph, weights, &widths)?;
    for &rows in &widths[1..] {
        workspace.prepare_pool_rows(device, graph, weights, rows)?;
    }
    Ok(workspace)
}

/// Decode graphs over the pool's first 1, 2, ... slots, narrowest first.
fn build_narrow_decode(
    builder: &mut super::batch::BatchBuilder<'_>,
    states: &mut SlotStates,
    fp8: &mut [Fp8Caches],
    widths: &[usize],
) -> Result<Vec<SlotDecodeGraph>> {
    let mut graphs = Vec::with_capacity(widths.len());
    for &slots in widths {
        graphs.push(builder.build_slots(&mut states[..slots], &mut fp8[..slots])?);
    }
    Ok(graphs)
}

fn reject_fused_verification(weights: &ProgramWeights, verify: usize) -> Result<()> {
    if weights.fusion.is_some() && verify >= 2 {
        return Err(Error::unsupported(
            "pooled verification requires unfused target weights",
        ));
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Keeps per-lane attention and rotary launches, to compare a pool against them.
    pub static LANES_DISABLED: Cell<bool> = const { Cell::new(false) };
}

/// Lane-batched attention scratch; a pooled verification graph has its own grouped path.
fn lanes_workspace(
    device: &CudaDevice,
    graph: &DataflowGraph,
    width: usize,
    verify: usize,
) -> Result<super::attention_lanes::Workspace> {
    if verify < 2 {
        super::attention_lanes::Workspace::new(device, graph, width)
    } else {
        Ok(super::attention_lanes::Workspace::default())
    }
}

/// The full-width decode graph and its narrower rungs.
fn build_pool_decode<'a>(
    builder: &mut super::batch::BatchBuilder<'a>,
    states: &'a mut SlotStates,
    fp8: &mut [Fp8Caches],
    narrow: &[usize],
    table_root: &'a Tensor<i32>,
) -> Result<(SlotDecodeGraph, Vec<SlotDecodeGraph>)> {
    // Narrow rungs stay per-lane: at one to four lanes the launch count is small, and their
    // state views are prefixes the batch kernel cannot address directly.
    let decode_narrow = build_narrow_decode(builder, states, fp8, narrow)?;
    // The full-width graph records one batched state kernel per node instead of one per lane;
    // its builder therefore holds the pooled state roots, not a lane.
    let (lane_states, roots) = states.parts();
    builder.states = roots;
    #[cfg(test)]
    let lanes = !LANES_DISABLED.get();
    #[cfg(not(test))]
    let lanes = true;
    builder.table_root = lanes.then_some(table_root);
    let decode = builder.build_slots(lane_states, fp8)?;
    Ok((decode, decode_narrow))
}

/// The pool-wide verify graph and one narrower graph per `narrow` slot count, with the root
/// views each narrow graph reads.
#[expect(
    unsafe_code,
    reason = "Audited narrow verify roots: prefix views of `states`, which the pool owns \
              and drops after its graphs"
)]
fn build_verify_graphs(
    builder: &mut super::batch::BatchBuilder<'_>,
    (states, fp8): (&mut SlotStates, &mut [Fp8Caches]),
    verify: usize,
    narrow: &[usize],
) -> Result<(SlotVerifyGraph, Vec<SlotVerifyGraph>, Vec<States>)> {
    let full = builder.build_slot_verify(states, fp8, verify)?;
    let mut graphs = Vec::with_capacity(narrow.len());
    let mut roots = Vec::with_capacity(narrow.len());
    for &slots in narrow {
        // SAFETY: the pool keeps `states` alive behind `_narrow_roots` and drops its graphs
        // first; every graph replays on the pool's one stream.
        let mut prefix = unsafe { prefix_roots(states, slots)? };
        graphs.push(builder.build_slot_verify_parts(
            (&mut states.parts().0[..slots], &mut prefix),
            &mut fp8[..slots],
            verify,
        )?);
        roots.push(prefix);
    }
    Ok((full, graphs, roots))
}

/// Views of the first `slots` rows of every pooled root family, for a verify graph narrower
/// than the pool.
///
/// # Safety
/// The returned roots alias `states`: the pool must keep it alive for every graph captured
/// through them and replay them on its single stream.
#[expect(
    unsafe_code,
    reason = "Audited prefix alias: the pool owns the roots and drops its graphs first"
)]
unsafe fn prefix_roots(states: &SlotStates, slots: usize) -> Result<States> {
    let width = states.len();
    if slots == 0 || slots > width {
        return Err(Error::invalid("pooled state prefix"));
    }
    let mut roots = States::new();
    for (id, owners) in states.roots() {
        let mut views = Vec::with_capacity(owners.len());
        for owner in owners {
            let mut shape = owner
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim).map_err(device_error))
                .collect::<Result<Vec<_>>>()?;
            shape[0] = slots;
            // SAFETY: forwarded from this function's contract.
            views.push(unsafe { super::arena::prefix_view(owner, width, slots, &shape)? });
        }
        roots.insert(*id, views);
    }
    Ok(roots)
}

/// Pinned ring depth per external-hidden lane: one upload per draft or catch-up step,
/// and a lane's chain between stream barriers never exceeds the maximum verify width.
const EXTERNAL_UPLOAD_RING: usize = MAX_VERIFICATION_WIDTH + 1;

/// Per-lane external hidden rows and their pinned upload rings.
type ExternalLanes = (Vec<Tensor<f32>>, Vec<Vec<PinnedHostBuffer<f32>>>);

/// One pool's per-slot recurrent states, KV caches and KV position tables.
/// The pool's external row, lane rows, pinned staging and fusion workspaces.
type LaneInputs = (
    Tensor<f32>,
    Vec<Tensor<f32>>,
    Vec<Vec<PinnedHostBuffer<f32>>>,
    Vec<Option<super::program::FusionWorkspace>>,
);
type SlotStorage = (SlotStates, Vec<Fp8Caches>, Vec<Tensor<i32>>, Tensor<i32>);

/// How one state family moves into a slot at bind time: a token-row prefix of a
/// capacity-sized KV cache (f32 or fp8), or the whole tensor (conv/delta state).
/// KV layout is discovered from the captured tensors at bind time, never assumed.
#[derive(Clone, Copy)]
enum SlotCopy {
    Kv,
    KvFp8,
    Whole,
}

/// Dimensions of the captured head-major KV state layout `[kv_heads, capacity, head_dim]`.
const KV_STATE_DIMS: usize = 3;

/// Copy plan over the dataflow graph's state tensors, in declaration order.
fn copy_plan(graph: &DataflowGraph, weights: &ProgramWeights) -> Result<Vec<(TensorId, SlotCopy)>> {
    let mut plan = Vec::new();
    for spec in &graph.tensors {
        let TensorStorage::State { kind, .. } = &spec.storage else {
            continue;
        };
        let fp8 = weights.kv_scales.contains_key(&spec.id);
        let copy = match (kind, fp8) {
            (StateKind::AttentionKv, false) if spec.shape.len() == 2 => SlotCopy::Kv,
            (StateKind::AttentionKv, true) if spec.shape.len() == 2 => SlotCopy::KvFp8,
            (StateKind::Conv | StateKind::LinearAttention, false) => SlotCopy::Whole,
            _ => return Err(Error::unsupported("slot pool state recipe")),
        };
        plan.push((spec.id, copy));
    }
    Ok(plan)
}

/// Whole-tensor aliases of every cache in `caches`.
/// # Safety
/// The caches must outlive the aliases; see [`SlotPool::slot_views`].
#[expect(
    unsafe_code,
    reason = "Audited cache alias: forwarded from the pool's view contracts"
)]
unsafe fn alias_caches(caches: &Fp8Caches) -> Result<Fp8Caches> {
    let mut views = Fp8Caches::new();
    for (id, (keys, values)) in caches {
        // SAFETY: forwarded from this function's contract.
        let keys = unsafe { alias(keys)? };
        // SAFETY: forwarded from this function's contract.
        let values = unsafe { alias(values)? };
        views.insert(*id, (keys, values));
    }
    Ok(views)
}

/// Whether two tensors start at the same device address: a slot-resident program's view and the
/// slot it aliases.
fn same_storage<T: DType>(left: &Tensor<T>, right: &Tensor<T>) -> bool {
    left.device_pointer().cu_deviceptr() == right.device_pointer().cu_deviceptr()
}

/// A tensor's dimensions as host sizes.
fn dims<T: DType>(tensor: &Tensor<T>) -> Result<Vec<usize>> {
    tensor
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).map_err(device_error))
        .collect()
}

/// Gather the first `rows` positions of a head-major `[kv_heads, rows, head_dim]` KV state into
/// a new contiguous tensor: block by block through `blocks` when the state is paged, otherwise
/// one run per head.
fn gather_kv<T: DType>(
    src: &Tensor<T>,
    rows: usize,
    blocks: Option<&[i32]>,
    device: &CudaDevice,
) -> Result<Tensor<T>> {
    let shape = dims(src)?;
    let [heads, physical, dim] = shape[..] else {
        return Err(Error::invariant("slot export KV layout"));
    };
    if blocks.is_none() && physical < rows
        || blocks.is_some_and(|blocks| rows.div_ceil(KV_BLOCK_TOKENS) > blocks.len())
    {
        return Err(Error::invariant("slot export beyond the slot's KV"));
    }
    let mut dst = api::zeros::<T>(&[heads, rows, dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    for head in 0..heads {
        match blocks {
            None => device.copy_d2d_at(
                &mut dst,
                src,
                head * rows * dim,
                head * physical * dim,
                rows * dim,
            )?,
            Some(blocks) => {
                for (index, &block) in blocks
                    .iter()
                    .enumerate()
                    .take(rows.div_ceil(KV_BLOCK_TOKENS))
                {
                    let start = index * KV_BLOCK_TOKENS;
                    let count = KV_BLOCK_TOKENS.min(rows - start);
                    let block = usize::try_from(block).map_err(device_error)?;
                    device.copy_d2d_at(
                        &mut dst,
                        src,
                        (head * rows + start) * dim,
                        (head * physical + block * KV_BLOCK_TOKENS) * dim,
                        count * dim,
                    )?;
                }
            }
        }
    }
    Ok(dst)
}

/// A whole-tensor alias with the source's shape.
/// # Safety
/// The source must outlive the alias; see [`SlotPool::slot_views`].
#[expect(
    unsafe_code,
    reason = "Audited slot alias: forwarded from SlotPool::slot_views"
)]
unsafe fn alias<T: DType>(tensor: &Tensor<T>) -> Result<Tensor<T>> {
    let shape = tensor
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).map_err(device_error))
        .collect::<Result<Vec<_>>>()?;
    // SAFETY: forwarded from this function's contract; one lane spans the whole tensor.
    let view = unsafe { super::arena::row_view(tensor, 1, 0)? };
    view.reshape(&shape).map_err(device_error)
}

/// Device bytes one slot's state tensors occupy: f32 states plus fp8 KV caches.
fn state_bytes(states: &States, fp8: &Fp8Caches) -> u64 {
    let float_elements: u64 = states
        .values()
        .flatten()
        .map(|tensor| tensor.size() as u64)
        .sum();
    let packed_elements: u64 = fp8
        .values()
        .map(|(keys, values)| (keys.size() + values.size()) as u64)
        .sum();
    float_elements.saturating_mul(F32_BYTES as u64) + packed_elements
}

/// Free-slot stack and the pool's lifetime lease counters.
struct SlotLedger {
    free: Vec<usize>,
    allocations: u64,
    releases: u64,
}
impl SlotLedger {
    fn new(width: usize) -> Self {
        Self {
            free: (0..width).rev().collect(),
            allocations: 0,
            releases: 0,
        }
    }
    /// Take a free slot, if any remain.
    fn take(&mut self) -> Option<usize> {
        // The lowest free slot keeps busy slots inside the narrowest verify graph.
        let index = self
            .free
            .iter()
            .enumerate()
            .min_by_key(|&(_, &slot)| slot)
            .map(|(index, _)| index)?;
        let slot = self.free.swap_remove(index);
        self.allocations = self.allocations.saturating_add(1);
        Some(slot)
    }
    /// Return a slot to the free set; the next bind overwrites its state prefix.
    /// Returns false when the slot was already free.
    fn release(&mut self, slot: usize) -> bool {
        if self.free.contains(&slot) {
            return false;
        }
        self.free.push(slot);
        self.releases = self.releases.saturating_add(1);
        true
    }
}

/// One FP8 KV cache per attention layer shared by every slot, addressed through each slot's
/// block table: a slot holds blocks for the tokens its sequence can reach, not the pool's
/// whole per-slot capacity.
struct Paging {
    /// Owner of the shared caches; each slot's entry in `SlotPool::fp8` aliases it.
    shared: Fp8Caches,
    /// The shared page ledger; this pool only maps its leases to device rows.
    pages: BlockPool,
    leases: Vec<Vec<BlockLease>>,
    /// Each slot's leased pages as device table entries.
    owned: Vec<Vec<i32>>,
}

impl Paging {
    /// A slot's view of the shared caches; the slot's table selects its blocks.
    #[expect(
        unsafe_code,
        reason = "Audited pool alias: the pool owns the shared caches and drops its graphs and \
                  slot views first"
    )]
    fn slot_caches(&self) -> Result<Fp8Caches> {
        // SAFETY: see the expectation above.
        unsafe { alias_caches(&self.shared) }
    }

    /// Return a slot's pages to the ledger.
    fn release(&mut self, slot: usize) -> Result<()> {
        let leases = self
            .leases
            .get_mut(slot)
            .ok_or_else(|| Error::invalid("slot assignment index"))?;
        self.pages.release(leases)?;
        leases.clear();
        self.owned[slot].clear();
        Ok(())
    }

    /// Lease `blocks` pages for a slot. False when the ledger lacks them; the slot then owns none.
    fn assign(&mut self, slot: usize, blocks: usize) -> Result<bool> {
        self.release(slot)?;
        if self.pages.free_blocks() < blocks {
            return Ok(false);
        }
        let leases = self.pages.allocate(blocks)?;
        self.owned[slot] = leases
            .iter()
            .map(|lease| i32::try_from(lease.index).map_err(device_error))
            .collect::<Result<_>>()?;
        self.leases[slot] = leases;
        Ok(true)
    }
}

pub struct SlotPool {
    _weights: Vec<Owner>,
    // Exactly one of the two graphs exists: plain decode without a draft, pooled
    // speculation with one. Declared first so the graphs drop before the state sets and
    // arena buffers their kernels reference.
    decode: Option<SlotDecodeGraph>,
    verify: Option<SlotVerifyGraph>,
    /// Decode graphs over the pool's first 2, 4, ... slots, narrowest first: a step whose busy
    /// slots are all low replays only their lanes. `decode` stays the timeline of record.
    narrow_decode: Vec<SlotDecodeGraph>,
    /// Verify graphs over the pool's first 1, 2, 4, ... slots, narrowest first: a step whose
    /// busy slots are all low replays only their lanes. `verify` stays the timeline of record.
    narrow: Vec<SlotVerifyGraph>,
    /// The narrow graph the last verification replayed, `None` for the full graph.
    active_verify: Option<usize>,
    /// Root-family views the narrow graphs read; they alias `states` and drop before it.
    _narrow_roots: Vec<States>,
    states: SlotStates,
    fp8: Vec<Fp8Caches>,
    _lanes_attention: super::attention_lanes::Workspace,
    _table_root: Tensor<i32>,
    /// Per-slot identity block tables; declared with the states so they outlive the captures.
    tables: Vec<Tensor<i32>>,
    table_staging: PinnedUpload<i32>,
    plan: Vec<(TensorId, SlotCopy)>,
    /// Shared physical KV blocks behind every slot's table, when the pool pages its KV.
    paging: Option<Paging>,
    leases: SlotLedger,
    slots: usize,
    capacity: usize,
    vocabulary: usize,
    // The captured graph bakes these buffers' device pointers; declared last so the
    // graph drops before anything it references.
    _external: Tensor<f32>,
    _nvfp4: super::nvfp4_gemm::Workspace,
    _attention: super::attention_decode::Workspace,
    lane_external: Vec<Tensor<f32>>,
    /// Whether a lane's external hidden row was uploaded at least once.
    external_uploaded: Vec<bool>,
    /// Pinned staging rings feeding `lane_external`: uploads enqueue without a sync, so a
    /// ring slot must not be rewritten while an earlier copy from it could still be in
    /// flight. The ring outlasts the longest unsynced upload chain (a draft catch-up).
    pinned_external: Vec<Vec<PinnedHostBuffer<f32>>>,
    pinned_cursor: Vec<usize>,
    _lane_fusion: Vec<Option<super::program::FusionWorkspace>>,
}

impl SlotPool {
    pub(crate) fn parameter_bytes(graph: &DataflowGraph, verify: usize) -> Result<u64> {
        if super::slot_verify::fused_recurrent(graph, verify) {
            super::slot_states::parameter_bytes(&graph.nodes)
        } else {
            Ok(0)
        }
    }

    pub(crate) fn attention_bytes(
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        geometry: (usize, usize, usize),
        profile: &DeviceProfile,
    ) -> Result<u64> {
        let (slots, verify, _) = geometry;
        if verify < 2 {
            return super::attention_lanes::required_bytes(graph, slots, profile);
        }
        super::attention_decode::Workspace::pooled_bytes(graph, weights, geometry, profile)
    }

    /// Device staging or rollback storage used by the same recurrent policy as capture.
    pub(crate) fn rollback_bytes(
        graph: &DataflowGraph,
        width: usize,
        verify: usize,
    ) -> Result<u64> {
        let elements = if super::slot_verify::fused_recurrent(graph, verify) {
            let mut elements = 0_u64;
            for node in graph
                .nodes
                .iter()
                .filter(|node| super::slot_recurrent::covers(node))
            {
                for id in super::slot_recurrent::commit_inputs(node) {
                    let spec = graph
                        .tensors
                        .iter()
                        .find(|spec| spec.id == *id)
                        .ok_or_else(|| Error::invariant("pooled staging tensor"))?;
                    elements = elements
                        .checked_add(spec.elements()? as u64)
                        .ok_or_else(|| Error::invalid("pooled staging overflow"))?;
                }
            }
            elements.checked_mul(verify as u64)
        } else {
            let elements = graph
                .tensors
                .iter()
                .filter(|spec| {
                    matches!(
                        spec.storage,
                        TensorStorage::State {
                            kind: StateKind::Conv | StateKind::LinearAttention,
                            ..
                        }
                    )
                })
                .try_fold(0_u64, |sum, spec| {
                    sum.checked_add(spec.elements()? as u64)
                        .ok_or_else(|| Error::invalid("pooled snapshot overflow"))
                })?;
            elements.checked_mul(verify.saturating_sub(1) as u64)
        }
        .and_then(|n| n.checked_mul(width as u64))
        .and_then(|n| n.checked_mul(F32_BYTES as u64))
        .ok_or_else(|| Error::invalid("pooled rollback overflow"))?;
        Ok(elements)
    }

    /// Allocate `width` slots of `capacity` tokens each and capture the shared graph.
    /// `verify` ≥ 2 captures the pooled speculation graph with `verify` candidate lanes
    /// per slot; `verify` < 2 captures the plain decode graph.
    /// # Errors
    /// Rejects invalid geometry, fused verification graphs, an exceeded device budget
    /// or CUDA capture failures; the caller disables batching on any error.
    #[expect(
        clippy::too_many_arguments,
        reason = "Pool construction binds model, geometry, readout and KV paging explicitly"
    )]
    pub fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        width: usize,
        capacity: usize,
        hidden: usize,
        output: (usize, usize),
        kv_tokens: Option<usize>,
    ) -> Result<Self> {
        let (vocabulary, verify) = output;
        let capacity = super::capacity::padded(capacity)?;
        if !valid_pool_geometry(width, verify, capacity, hidden, vocabulary) {
            return Err(Error::invalid("slot pool width, capacity or geometry"));
        }
        reject_fused_verification(weights, verify)?;
        let plan = copy_plan(graph, weights)?;
        let headroom = device.profile()?.device_headroom_bytes();
        let budget = device.allocatable_memory_bytes()?.saturating_sub(headroom);
        let paging = Self::paging(device, graph, weights, (width, capacity), kv_tokens)?;
        let (mut states, mut fp8, tables, table_root) = Self::allocate_slots(
            device,
            graph,
            weights,
            (width, capacity),
            paging.as_ref(),
            budget,
        )?;
        let lanes = if verify >= 2 { width * verify } else { width };
        let (external, mut lane_external, pinned_external, mut lane_fusion) =
            Self::lane_inputs(device, weights.fusion.is_some(), lanes, hidden)?;
        let mut attention = super::attention_decode::Workspace::new(device, graph, capacity)?;
        let geometry = (width, verify, capacity);
        attention.prepare_pooled(device, graph, weights, geometry, paging.is_some())?;
        let narrow_slots = narrow_widths(graph, width, verify);
        let mut nvfp4 = pool_workspace(device, graph, weights, (lanes, verify), &narrow_slots)?;
        let mut placeholder_states = States::new();
        let mut placeholder_fp8 = Fp8Caches::new();
        let mut lanes_attention = lanes_workspace(device, graph, width, verify)?;
        let (decode, verify_graph, decode_narrow, narrow, narrow_roots) = {
            let mut builder = super::batch::BatchBuilder {
                device,
                graph,
                weights,
                nvfp4: &mut nvfp4,
                delegated: None,
                attention: &mut attention,
                states: &mut placeholder_states,
                fp8_states: &mut placeholder_fp8,
                external: &external,
                lane_external: &mut lane_external,
                lane_external_owner: None,
                lane_fusion: &mut lane_fusion,
                tables: &tables,
                capacity,
                width,
                table_root: None,
                lanes_attention: None,
                burst: None,
            };
            if verify >= 2 {
                let (full, graphs, roots) = build_verify_graphs(
                    &mut builder,
                    (&mut states, &mut fp8),
                    verify,
                    &narrow_slots,
                )?;
                (None, Some(full), Vec::new(), graphs, roots)
            } else {
                builder.lanes_attention = Some(&mut lanes_attention);
                let (decode, decode_narrow) = build_pool_decode(
                    &mut builder,
                    &mut states,
                    &mut fp8,
                    &narrow_slots,
                    &table_root,
                )?;
                (Some(decode), None, decode_narrow, Vec::new(), Vec::new())
            }
        };
        Ok(Self {
            _weights: weights.owners(),
            decode,
            verify: verify_graph,
            narrow_decode: decode_narrow,
            narrow,
            active_verify: None,
            _narrow_roots: narrow_roots,
            states,
            fp8,
            tables,
            table_staging: PinnedUpload::slots(device, capacity.div_ceil(KV_BLOCK_TOKENS), width)?,
            plan,
            paging,
            leases: SlotLedger::new(width),
            slots: width,
            capacity,
            vocabulary,
            _external: external,
            _nvfp4: nvfp4,
            _attention: attention,
            _lanes_attention: lanes_attention,
            _table_root: table_root,
            lane_external,
            external_uploaded: vec![false; lanes],
            pinned_external,
            pinned_cursor: vec![0; lanes],
            _lane_fusion: lane_fusion,
        })
    }

    /// The shared external row, each lane's own row and each lane's fusion workspace.
    fn lane_inputs(
        device: &CudaDevice,
        fused: bool,
        lanes: usize,
        hidden: usize,
    ) -> Result<LaneInputs> {
        let external = api::zeros::<f32>(&[hidden])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let (lane_external, pinned_external) = Self::external_lanes(device, fused, lanes, hidden)?;
        let lane_fusion = (0..lanes)
            .map(|_| super::program::FusionWorkspace::allocate(device, hidden, fused))
            .collect::<Result<_>>()?;
        Ok((external, lane_external, pinned_external, lane_fusion))
    }

    /// Allocate each slot's recurrent states and KV cache, checking the running total against
    /// the device budget before the pool captures any graph.
    #[expect(
        unsafe_code,
        reason = "Audited table rows: the pool stores their owner beside them"
    )]
    fn allocate_slots(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        (width, capacity): (usize, usize),
        paging: Option<&Paging>,
        budget: u64,
    ) -> Result<SlotStorage> {
        let paging_bytes = paging.map_or(0, |paging| state_bytes(&States::new(), &paging.shared));
        let states = SlotStates::allocate(
            device,
            graph,
            weights,
            (width, capacity),
            budget.saturating_sub(paging_bytes),
        )?;
        let mut fp8 = Vec::with_capacity(width);
        let mut tables = Vec::with_capacity(width);
        let mut bytes = paging.map_or(0, |paging| state_bytes(&States::new(), &paging.shared));
        for slot_states in states.iter() {
            let slot_fp8 = match paging {
                Some(paging) => paging.slot_caches()?,
                None => allocate(device, graph, capacity, &weights.kv_scales)?,
            };
            let slot_bytes = paging.map_or_else(
                || state_bytes(slot_states, &slot_fp8),
                |_| state_bytes(slot_states, &Fp8Caches::new()),
            );
            bytes = bytes
                .checked_add(slot_bytes)
                .ok_or_else(|| Error::invalid("slot pool budget overflow"))?;
            if bytes > budget {
                return Err(Error::new(
                    infer_core::ErrorCode::Capacity,
                    "slot decode pool exceeds available device memory minus headroom",
                ));
            }
            fp8.push(slot_fp8);
        }
        // Step 1 of the KV refactor: a per-slot identity mapping over that slot's own cache.
        // A paged pool fills the table from its shared blocks when the slot is assigned.
        let root = super::metadata::identity_tables(device, capacity, width)?;
        for slot in 0..width {
            // SAFETY: the pool stores `root` beside these views; the rows are disjoint.
            tables.push(unsafe { super::arena::row_view(&root, width, slot)? });
        }
        Ok((states, fp8, tables, root))
    }

    /// Per-lane external hidden rows plus, for an external-hidden pool, their pinned
    /// upload rings. Uploads enqueue without a sync, so a ring slot must not be rewritten
    /// while an earlier copy from it could still be in flight; the ring depth outlasts
    /// the longest unsynced upload chain (a draft catch-up).
    fn external_lanes(
        device: &CudaDevice,
        external_hidden: bool,
        lanes: usize,
        hidden: usize,
    ) -> Result<ExternalLanes> {
        let mut lane_external = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            lane_external.push(
                api::zeros::<f32>(&[hidden])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            );
        }
        let context = external_hidden
            .then(|| CudaContext::new(device.stream.device().ordinal()))
            .transpose()
            .map_err(device_error)?;
        let mut pinned_external = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            let ring = match &context {
                Some(context) => (0..EXTERNAL_UPLOAD_RING)
                    .map(|_| PinnedHostBuffer::zeroed(context, hidden).map_err(device_error))
                    .collect::<Result<Vec<_>>>()?,
                None => Vec::new(),
            };
            pinned_external.push(ring);
        }
        Ok((lane_external, pinned_external))
    }

    /// Drain in-flight detached work before the pool's buffers free.
    fn barrier(&self) -> Result<()> {
        match (&self.decode, &self.verify) {
            (Some(graph), _) => graph.barrier(),
            (None, Some(graph)) => graph.barrier(),
            (None, None) => Ok(()),
        }
    }

    #[cfg(test)]
    #[expect(
        clippy::used_underscore_binding,
        reason = "This workspace is retained for capture lifetime in production and inspected only by the graph test"
    )]
    pub(crate) const fn quant_reuse_stats(&self) -> (usize, usize) {
        self._nvfp4.quant_reuse_totals()
    }

    #[cfg(test)]
    pub(crate) fn test_batched_commit(&self) -> bool {
        self.verify
            .as_ref()
            .is_some_and(SlotVerifyGraph::has_batched_commit)
    }

    #[cfg(test)]
    pub(crate) fn test_states(&self, slot: usize) -> (&States, &Fp8Caches) {
        (&self.states[slot], &self.fp8[slot])
    }

    #[cfg(test)]
    pub(crate) fn test_hidden(&self, device: &CudaDevice, slot: usize) -> Result<Vec<f32>> {
        self.decode
            .as_ref()
            .ok_or_else(|| Error::invalid("test hidden needs decode pool"))?
            .test_hidden(device, slot)
    }

    /// Slots (state sets) in the pool.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.slots
    }

    /// Candidate lanes per slot when the pool speculates (`None` on a plain decode pool).
    #[must_use]
    pub fn verify(&self) -> Option<usize> {
        self.verify.as_ref().map(|graph| graph.verify)
    }

    /// Device draft features of every verify graph, narrowest first, with its slot count;
    /// see [`SlotVerifyGraph::features`].
    pub(crate) fn verify_features(&self) -> Option<Vec<(usize, Arc<Tensor<f32>>)>> {
        self.narrow
            .iter()
            .chain(self.verify.as_ref())
            .map(|graph| graph.features().map(|features| (graph.slots, features)))
            .collect()
    }

    /// Slots of the verify graph the last verification replayed.
    pub(crate) fn verified_slots(&self) -> usize {
        self.active_verify
            .and_then(|index| self.narrow.get(index))
            .or(self.verify.as_ref())
            .map_or(0, |graph| graph.slots)
    }

    /// Slots no sequence currently leases.
    #[must_use]
    pub(crate) const fn free_slots(&self) -> usize {
        self.leases.free.len()
    }

    /// Physical KV pages and pages not reserved by any request, including resident homes.
    pub(crate) fn kv_pages(&self) -> (usize, usize) {
        self.paging.as_ref().map_or_else(
            || {
                let per_slot = self.capacity.div_ceil(KV_BLOCK_TOKENS);
                (self.width() * per_slot, self.free_slots() * per_slot)
            },
            |paging| (paging.pages.capacity(), paging.pages.free_blocks()),
        )
    }

    /// Lifetime slot leases granted and returned, in slots.
    #[must_use]
    pub(crate) const fn lease_counters(&self) -> (u64, u64) {
        (self.leases.allocations, self.leases.releases)
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Host-side state cursors of every slot.
    #[must_use]
    pub fn cursors(&self) -> &[usize] {
        match (&self.decode, &self.verify) {
            (Some(graph), _) => graph.cursors(),
            (None, Some(graph)) => graph.cursors(),
            (None, None) => &[],
        }
    }

    /// Take a free slot, if any remain.
    pub fn take_free(&mut self) -> Option<usize> {
        self.leases.take()
    }

    /// Lease one specific free slot; false when it is already leased.
    pub(crate) fn take_slot(&mut self, slot: usize) -> bool {
        let Some(index) = self.leases.free.iter().position(|&free| free == slot) else {
            return false;
        };
        self.leases.free.remove(index);
        self.leases.allocations = self.leases.allocations.saturating_add(1);
        true
    }

    /// Views of one slot's recurrent states and KV caches, for a program that prefills
    /// straight into the slot ([`super::program::DeviceProgram::new_on_slot`]).
    /// # Safety
    /// The pool must outlive the views and every graph captured through them, and only the
    /// slot's current owner may replay those graphs, serialized on the device stream.
    /// # Errors
    /// Rejects an unknown slot.
    #[expect(
        unsafe_code,
        reason = "Audited slot alias: the views share the slot's storage under the caller's contract"
    )]
    pub(crate) unsafe fn slot_views(
        &self,
        slot: usize,
    ) -> Result<(States, Fp8Caches, Tensor<i32>)> {
        let states = self
            .states
            .get(slot)
            .ok_or_else(|| Error::invalid("slot view index"))?;
        let fp8 = self
            .fp8
            .get(slot)
            .ok_or_else(|| Error::invalid("slot view index"))?;
        let mut state_views = States::new();
        for (id, tensors) in states {
            let mut views = Vec::with_capacity(tensors.len());
            for tensor in tensors {
                // SAFETY: forwarded from this function's contract.
                views.push(unsafe { alias(tensor)? });
            }
            state_views.insert(*id, views);
        }
        // SAFETY: forwarded from this function's contract.
        let table = unsafe {
            alias(
                self.tables
                    .get(slot)
                    .ok_or_else(|| Error::invalid("slot view index"))?,
            )?
        };
        let mut fp8_views = Fp8Caches::new();
        for (id, (keys, values)) in fp8 {
            // SAFETY: forwarded from this function's contract.
            let keys = unsafe { alias(keys)? };
            // SAFETY: forwarded from this function's contract.
            let values = unsafe { alias(values)? };
            fp8_views.insert(*id, (keys, values));
        }
        Ok((state_views, fp8_views, table))
    }

    /// Return a slot to the free set; the next bind overwrites its state prefix.
    pub fn release(&mut self, slot: usize) {
        if slot < self.width() && self.leases.release(slot) {
            if let Some(graph) = self.verify.as_mut() {
                graph.discard(slot);
            }
            if let Some(paging) = self.paging.as_mut()
                && let Err(error) = paging.release(slot)
            {
                tracing::error!(target: "infer::executor", %error, slot, "slot pages not released");
            }
        }
    }

    /// Give a leased slot the KV blocks for `tokens` positions and point its table at them.
    /// Unpaged pools pre-own their whole capacity. False when the shared pool lacks blocks; the
    /// caller releases the slot and the request waits.
    /// # Errors
    /// Rejects a request beyond the slot capacity and table upload failures.
    pub(crate) fn assign(
        &mut self,
        slot: usize,
        tokens: usize,
        device: &CudaDevice,
    ) -> Result<bool> {
        if tokens > self.capacity {
            return Err(Error::invalid("slot assignment beyond capacity"));
        }
        let Some(paging) = self.paging.as_mut() else {
            return Ok(true);
        };
        let blocks = tokens.div_ceil(KV_BLOCK_TOKENS).max(1);
        if !paging.assign(slot, blocks)? {
            return Ok(false);
        }
        let owned = &paging.owned[slot];
        let table = self
            .tables
            .get_mut(slot)
            .ok_or_else(|| Error::invariant("slot block table"))?;
        // Positions past the request's reach never resolve; point them at its first block.
        let mut entries = owned.clone();
        entries.resize(table.size(), owned[0]);
        self.table_staging.copy(device, table, &entries)?;
        Ok(true)
    }

    /// Views of the shared paged KV caches, for the prompt worker; `None` when unpaged.
    /// # Safety
    /// The pool must outlive the views and every graph captured through them.
    /// # Errors
    /// Propagates view failures.
    #[expect(
        unsafe_code,
        reason = "Audited pool alias: the views share the paged caches under the caller's contract"
    )]
    pub(crate) unsafe fn kv_views(&self) -> Result<Option<Fp8Caches>> {
        self.paging
            .as_ref()
            // SAFETY: forwarded from this function's contract.
            .map(|paging| unsafe { alias_caches(&paging.shared) })
            .transpose()
    }

    /// The physical KV pool for a pool of `width` slots of `capacity` tokens: `kv_tokens`
    /// shared positions, or `None` to give each slot its own whole-capacity cache.
    fn paging(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        (width, capacity): (usize, usize),
        kv_tokens: Option<usize>,
    ) -> Result<Option<Paging>> {
        let Some(tokens) = kv_tokens.filter(|_| !weights.kv_scales.is_empty()) else {
            return Ok(None);
        };
        let block = KV_BLOCK_TOKENS;
        let blocks = tokens.div_ceil(block).max(capacity.div_ceil(block));
        let shared = allocate(device, graph, blocks * block, &weights.kv_scales)?;
        Ok(Some(Paging {
            shared,
            pages: BlockPool::new(blocks)?,
            leases: vec![Vec::new(); width],
            owned: vec![Vec::new(); width],
        }))
    }

    /// Forget a slot's pending verification without freeing its lease: a slot-resident
    /// sequence that resets keeps its slot.
    pub(crate) fn discard(&mut self, slot: usize) {
        if let Some(graph) = self.verify.as_mut() {
            graph.discard(slot);
        }
    }

    /// Copy a sequence's covered state into `slot`: KV token rows up to `rows` (both sides
    /// are `[kv_heads, capacity, head_dim]` with their own capacity, so each head block
    /// copies separately), conv/delta and fp8 tensors whole. The slot cursor is seeded to
    /// `rows` for the next replay.
    /// # Errors
    /// Rejects unknown slots, prefixes beyond the slot capacity, layout mismatches
    /// between source and slot tensors, or driver copy failures.
    pub fn bind(
        &mut self,
        slot: usize,
        source: &States,
        source_fp8: &Fp8Caches,
        rows: usize,
        device: &CudaDevice,
    ) -> Result<()> {
        if slot >= self.width() || rows > self.capacity {
            return Err(Error::invalid("slot bind index or prefix length"));
        }
        // Step 1 of the KV refactor: the kernels resolve KV positions through this table, so it
        // must cover the slot's whole capacity. Step 2 replaces the identity contents with the
        // allocator's blocks and rewrites them here on every bind.
        let blocks = i32::try_from(self.capacity.div_ceil(KV_BLOCK_TOKENS))
            .map_err(|_| Error::invalid("slot block count"))?;
        if self
            .tables
            .get(slot)
            .is_none_or(|table| table.shape() != [blocks])
        {
            return Err(Error::invariant("slot block table shape"));
        }
        for (id, copy) in &self.plan {
            match *copy {
                SlotCopy::Kv => Self::copy_kv_states(
                    self.states
                        .get_mut(slot)
                        .ok_or_else(|| Error::invariant("slot states"))?,
                    source,
                    *id,
                    rows,
                    self.capacity,
                    device,
                )?,
                SlotCopy::KvFp8 => {
                    let dst = self
                        .fp8
                        .get_mut(slot)
                        .ok_or_else(|| Error::invariant("slot fp8 caches"))?;
                    match self.paging.as_ref() {
                        Some(paging) => Self::scatter_kv_fp8(
                            dst,
                            source_fp8,
                            *id,
                            (rows, &paging.owned[slot]),
                            device,
                        )?,
                        None => {
                            Self::copy_kv_fp8(dst, source_fp8, *id, rows, self.capacity, device)?;
                        }
                    }
                }
                SlotCopy::Whole => Self::copy_whole(
                    self.states
                        .get_mut(slot)
                        .ok_or_else(|| Error::invariant("slot states"))?,
                    source,
                    *id,
                    device,
                )?,
            }
        }
        match (&mut self.decode, &mut self.verify) {
            (Some(graph), _) => {
                graph.set_cursor(slot, rows)?;
                for rung in &mut self.narrow_decode {
                    if slot < rung.width {
                        rung.set_cursor(slot, rows)?;
                    }
                }
            }
            (None, Some(graph)) => graph.set_cursor(slot, rows)?,
            (None, None) => return Err(Error::invariant("slot pool graph")),
        }
        Ok(())
    }

    /// Copy one slot's first `rows` positions out in a private program's layout: KV as
    /// `[kv_heads, rows, head_dim]` gathered through the slot's blocks, conv/delta whole.
    ///
    /// The inverse of [`Self::bind`], so a slot-resident sequence's prompt prefix can be kept
    /// as a snapshot and later bound into any slot or restored into a private program.
    /// # Errors
    /// Rejects unknown slots, prefixes beyond the slot's blocks, layout mismatches and driver
    /// allocation or copy failures.
    pub fn export(
        &self,
        slot: usize,
        rows: usize,
        device: &CudaDevice,
    ) -> Result<(States, Fp8Caches)> {
        let exported = self.export_queued(slot, rows, device);
        // A partial export has queued copies into tensors it is about to drop; settle them first.
        if exported.is_err() {
            device.drain()?;
        }
        exported
    }

    fn export_queued(
        &self,
        slot: usize,
        rows: usize,
        device: &CudaDevice,
    ) -> Result<(States, Fp8Caches)> {
        if slot >= self.width() || rows == 0 || rows > self.capacity {
            return Err(Error::invalid("slot export index or prefix length"));
        }
        let states = self
            .states
            .get(slot)
            .ok_or_else(|| Error::invariant("slot states"))?;
        let caches = self
            .fp8
            .get(slot)
            .ok_or_else(|| Error::invariant("slot fp8 caches"))?;
        let blocks = match &self.paging {
            Some(paging) => Some(
                paging
                    .owned
                    .get(slot)
                    .ok_or_else(|| Error::invariant("slot blocks"))?
                    .as_slice(),
            ),
            None => None,
        };
        let mut out = States::new();
        let mut out_fp8 = Fp8Caches::new();
        for (id, copy) in &self.plan {
            match *copy {
                SlotCopy::Kv => {
                    let tensors = states
                        .get(id)
                        .ok_or_else(|| Error::invariant("slot export state"))?;
                    let copies = tensors
                        .iter()
                        .map(|tensor| gather_kv(tensor, rows, None, device))
                        .collect::<Result<_>>()?;
                    out.insert(*id, copies);
                }
                SlotCopy::KvFp8 => {
                    let (keys, values) = caches
                        .get(id)
                        .ok_or_else(|| Error::invariant("slot export fp8 cache"))?;
                    out_fp8.insert(
                        *id,
                        (
                            gather_kv(keys, rows, blocks, device)?,
                            gather_kv(values, rows, blocks, device)?,
                        ),
                    );
                }
                SlotCopy::Whole => {
                    let tensors = states
                        .get(id)
                        .ok_or_else(|| Error::invariant("slot export state"))?;
                    let mut copies = Vec::with_capacity(tensors.len());
                    for tensor in tensors {
                        let mut copy = api::zeros::<f32>(&dims(tensor)?)
                            .sync_on(&device.stream)
                            .map_err(device_error)?;
                        device.copy_d2d(&mut copy, tensor, tensor.size())?;
                        copies.push(copy);
                    }
                    out.insert(*id, copies);
                }
            }
        }
        Ok((out, out_fp8))
    }

    /// Replay one decode step for the active lanes; see [`SlotDecodeGraph::run_lanes`].
    /// # Errors
    /// Rejects invalid lanes or positions, a speculation pool and CUDA replay failures.
    #[cfg(test)]
    pub fn run_lanes(&mut self, device: &CudaDevice, lanes: &[SlotLane]) -> Result<BatchOutput> {
        Ok(self.run_lanes_readout(device, lanes, false)?.rows)
    }

    #[cfg(test)]
    pub(crate) fn run_lanes_readout(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        plain_greedy: bool,
    ) -> Result<super::batch::BatchReadout> {
        block_on(self.run_lanes_readout_async(device, lanes, plain_greedy))
    }

    pub(crate) async fn run_lanes_readout_async(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        plain_greedy: bool,
    ) -> Result<super::batch::BatchReadout> {
        if self
            .decode
            .as_ref()
            .is_some_and(|graph| graph.external_hidden)
            && lanes
                .iter()
                .any(|&(slot, ..)| !self.external_uploaded.get(slot).copied().unwrap_or(false))
        {
            return Err(Error::invalid("draft lane has no external hidden input"));
        }
        // Lanes only occupy low slots while few sequences are live, so the narrowest rung that
        // covers the highest busy slot replays the step without charging for every lane the
        // configured concurrency allows.
        let required = lanes
            .iter()
            .map(|&(slot, ..)| slot.saturating_add(1))
            .max()
            .unwrap_or(0);
        let selected = self
            .narrow_decode
            .iter()
            .position(|graph| graph.width >= required);
        let graph = match selected {
            Some(index) => &mut self.narrow_decode[index],
            None => self
                .decode
                .as_mut()
                .ok_or_else(|| Error::invalid("slot pool has no decode graph"))?,
        };
        // A sequence can move between the full graph and a narrow rung when concurrency
        // changes; the lane's staged position is authoritative, so restore the rung's cursor
        // before it validates the step.
        for &(slot, _, _, state_pos) in lanes {
            graph.set_cursor(slot, state_pos)?;
        }
        let readout = graph
            .run_lanes_readout_async(
                device,
                lanes,
                self.capacity,
                self.vocabulary,
                true,
                plain_greedy,
            )
            .await?;
        // Advance every rung to the position the step ended at, so a later step can use any
        // rung without failing its cursor check.
        let advanced: Vec<(usize, usize)> = lanes
            .iter()
            .map(|&(slot, ..)| (slot, graph.cursors()[slot]))
            .collect();
        for (slot, position) in advanced {
            if let Some(full) = self.decode.as_mut()
                && slot < full.width
            {
                full.set_cursor(slot, position)?;
            }
            for rung in &mut self.narrow_decode {
                if slot < rung.width {
                    rung.set_cursor(slot, position)?;
                }
            }
        }
        Ok(readout)
    }

    /// Replay one pooled speculation step; see [`SlotVerifyGraph::run_verify_readout`].
    /// # Errors
    /// Rejects invalid lanes or positions, a decode-only pool and CUDA replay failures.
    #[cfg(test)]
    pub fn run_verify(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotVerifyLane],
    ) -> Result<BatchOutput> {
        block_on(self.run_verify_async(device, lanes))
    }

    #[cfg(test)]
    pub(crate) async fn run_verify_async(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotVerifyLane],
    ) -> Result<BatchOutput> {
        Ok(self
            .run_verify_readout_async(device, lanes, false)
            .await?
            .rows)
    }

    #[cfg(test)]
    pub(crate) fn run_verify_readout(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotVerifyLane],
        plain_greedy: bool,
    ) -> Result<super::batch::BatchReadout> {
        block_on(self.run_verify_readout_async(device, lanes, plain_greedy))
    }

    pub(crate) async fn run_verify_readout_async(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotVerifyLane],
        plain_greedy: bool,
    ) -> Result<super::batch::BatchReadout> {
        let full = self
            .verify
            .as_mut()
            .ok_or_else(|| Error::invalid("slot pool has no verify graph"))?;
        let busy = lanes.iter().map(|&(slot, ..)| slot + 1).max().unwrap_or(0);
        self.active_verify = self.narrow.iter().position(|graph| graph.slots >= busy);
        let Some(graph) = self.active_verify.map(|index| &mut self.narrow[index]) else {
            return full
                .run_verify_readout_async(
                    device,
                    lanes,
                    self.capacity,
                    self.vocabulary,
                    plain_greedy,
                )
                .await;
        };
        graph.sync_from(full);
        let readout = graph
            .run_verify_readout_async(device, lanes, self.capacity, self.vocabulary, plain_greedy)
            .await;
        full.sync_from(graph);
        readout
    }

    /// Run `commit` on the graph the last verification replayed, keeping the full graph's
    /// cursors the timeline of record.
    fn commit_active(
        &mut self,
        commit: impl FnOnce(&mut SlotVerifyGraph) -> Result<()>,
    ) -> Result<()> {
        let full = self
            .verify
            .as_mut()
            .ok_or_else(|| Error::invalid("slot pool has no verify graph"))?;
        let Some(graph) = self
            .active_verify
            .and_then(|index| self.narrow.get_mut(index))
        else {
            return commit(full);
        };
        graph.sync_from(full);
        let result = commit(graph);
        full.sync_from(graph);
        result
    }

    /// Commit one slot's accepted candidates; see [`SlotVerifyGraph::commit`].
    /// # Errors
    /// Rejects unknown slots, out-of-range counts and CUDA restore failures.
    pub fn commit_verify(
        &mut self,
        device: &CudaDevice,
        slot: usize,
        accepted: usize,
    ) -> Result<()> {
        self.commit_active(|graph| graph.commit(device, slot, accepted))
    }

    /// Commit independent accepted prefixes in one pooled update.
    /// # Errors
    /// Rejects unknown or duplicate slots, invalid counts and CUDA restore failures.
    pub(crate) fn commit_verify_many(
        &mut self,
        device: &CudaDevice,
        commits: &[(usize, usize)],
    ) -> Result<()> {
        if let [(slot, accepted)] = commits {
            return self.commit_verify(device, *slot, *accepted);
        }
        self.commit_active(|graph| graph.commit_many(device, commits))
    }

    /// Copy the covered token-row prefix of one KV state family into the slot.
    fn copy_kv_states(
        dst: &mut States,
        src: &States,
        id: TensorId,
        rows: usize,
        slot_capacity: usize,
        device: &CudaDevice,
    ) -> Result<()> {
        let src_tensors = src
            .get(&id)
            .ok_or_else(|| Error::invariant("slot bind source state"))?;
        let dst_tensors = dst
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("slot bind target state"))?;
        if src_tensors.len() != dst_tensors.len() {
            return Err(Error::invariant("slot bind state tensors"));
        }
        for (dst, src) in dst_tensors.iter_mut().zip(src_tensors) {
            Self::copy_kv_tensor(dst, src, rows, slot_capacity, device)?;
        }
        Ok(())
    }

    /// Copy a contiguous `[kv_heads, capacity, head_dim]` FP8 cache's first `rows` positions
    /// into a paged slot's blocks, block by block. A slot-resident program's views are the
    /// slot's own blocks and move nothing.
    fn scatter_kv_fp8(
        dst: &mut Fp8Caches,
        src: &Fp8Caches,
        id: TensorId,
        (rows, blocks): (usize, &[i32]),
        device: &CudaDevice,
    ) -> Result<()> {
        let (src_keys, src_values) = src
            .get(&id)
            .ok_or_else(|| Error::invariant("slot bind source fp8 cache"))?;
        let (dst_keys, dst_values) = dst
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("slot bind target fp8 cache"))?;
        let block = KV_BLOCK_TOKENS;
        if rows.div_ceil(block) > blocks.len() {
            return Err(Error::invariant("slot bind beyond assigned blocks"));
        }
        for (dst, src) in [(dst_keys, src_keys), (dst_values, src_values)] {
            if same_storage(dst, src) {
                continue;
            }
            let dims = |tensor: &Tensor<f8e4m3fn>| -> Result<Vec<usize>> {
                tensor
                    .shape()
                    .iter()
                    .map(|&dim| usize::try_from(dim).map_err(device_error))
                    .collect()
            };
            let (dst_dims, src_dims) = (dims(dst)?, dims(src)?);
            if dst_dims.len() != KV_STATE_DIMS
                || src_dims.len() != KV_STATE_DIMS
                || dst_dims[0] != src_dims[0]
                || dst_dims[2] != src_dims[2]
                || src_dims[1] < rows
            {
                return Err(Error::invariant("slot bind paged KV layout"));
            }
            let dim = dst_dims[2];
            for head in 0..dst_dims[0] {
                for (index, &physical) in blocks.iter().enumerate().take(rows.div_ceil(block)) {
                    let count = block.min(rows - index * block);
                    let physical = usize::try_from(physical).map_err(device_error)?;
                    device.copy_d2d_at(
                        dst,
                        src,
                        (head * dst_dims[1] + physical * block) * dim,
                        (head * src_dims[1] + index * block) * dim,
                        count * dim,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// fp8 twin of [`Self::copy_kv_states`].
    fn copy_kv_fp8(
        dst: &mut Fp8Caches,
        src: &Fp8Caches,
        id: TensorId,
        rows: usize,
        slot_capacity: usize,
        device: &CudaDevice,
    ) -> Result<()> {
        let (src_keys, src_values) = src
            .get(&id)
            .ok_or_else(|| Error::invariant("slot bind source fp8 cache"))?;
        let (dst_keys, dst_values) = dst
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("slot bind target fp8 cache"))?;
        for (dst, src) in [(dst_keys, src_keys), (dst_values, src_values)] {
            Self::copy_kv_tensor(dst, src, rows, slot_capacity, device)?;
        }
        Ok(())
    }

    /// One KV tensor's covered rows. Captured KV state is head-major
    /// `[kv_heads, capacity, head_dim]` and the two sides differ in capacity, so the
    /// token prefix moves per head block with each side's own stride. Anything but that
    /// exact layout is an invariant violation, never a silent partial copy.
    fn copy_kv_tensor<T: DType>(
        dst: &mut Tensor<T>,
        src: &Tensor<T>,
        rows: usize,
        slot_capacity: usize,
        device: &CudaDevice,
    ) -> Result<()> {
        // A slot-resident program's views are the slot's own storage: nothing to move.
        if same_storage(dst, src) {
            return Ok(());
        }
        let dims = |tensor: &Tensor<T>| {
            tensor
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim).map_err(device_error))
                .collect::<Result<Vec<_>>>()
        };
        // Pool caches are stored flat, so the head axis comes from the program side of the copy.
        let dst_dims = dims(dst)?;
        let src_dims = dims(src)?;
        let shaped = if dst_dims.len() == KV_STATE_DIMS {
            &dst_dims
        } else {
            &src_dims
        };
        if shaped.len() != KV_STATE_DIMS {
            return Err(Error::invariant("slot bind KV layout"));
        }
        let (heads, head_dim) = (shaped[0], shaped[2]);
        let row_elements = heads * head_dim;
        if row_elements == 0
            || dst.size() != row_elements * slot_capacity
            || src.size() < row_elements * rows
            || !src.size().is_multiple_of(row_elements)
        {
            return Err(Error::invariant("slot bind KV layout"));
        }
        let dst_stride = slot_capacity
            .checked_mul(head_dim)
            .ok_or_else(|| Error::invalid("slot bind copy overflow"))?;
        let src_stride = src.size() / heads;
        let elements = rows
            .checked_mul(head_dim)
            .ok_or_else(|| Error::invalid("slot bind copy overflow"))?;
        for head in 0..heads {
            device.copy_d2d_at(
                dst,
                src,
                head.checked_mul(dst_stride)
                    .ok_or_else(|| Error::invalid("slot bind copy overflow"))?,
                head.checked_mul(src_stride)
                    .ok_or_else(|| Error::invalid("slot bind copy overflow"))?,
                elements,
            )?;
        }
        Ok(())
    }

    /// Copy one conv/delta state family whole; both sides share one capacity-free layout.
    fn copy_whole(dst: &mut States, src: &States, id: TensorId, device: &CudaDevice) -> Result<()> {
        let src_tensors = src
            .get(&id)
            .ok_or_else(|| Error::invariant("slot bind source state"))?;
        let dst_tensors = dst
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("slot bind target state"))?;
        if src_tensors.len() != dst_tensors.len() {
            return Err(Error::invariant("slot bind state tensors"));
        }
        for (dst, src) in dst_tensors.iter_mut().zip(src_tensors) {
            if src.size() != dst.size() {
                return Err(Error::invariant("slot bind tensor size"));
            }
            if !same_storage(dst, src) {
                device.copy_d2d(dst, src, src.size())?;
            }
        }
        Ok(())
    }
}

impl Drop for SlotPool {
    fn drop(&mut self) {
        // Detached draft catch-up replays and verify restores may still be in flight on
        // the graph stream; the pool's states and arenas must not free beneath them.
        if let Err(error) = self.barrier() {
            eprintln!("CUDA slot pool drop barrier failed: {error}");
        }
    }
}

mod draft;

#[cfg(test)]
#[path = "../../tests/unit/resident_slot_batch_draft_tests.rs"]
mod draft_tests;
