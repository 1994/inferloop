mod checkpoint;
mod completion;
mod execution;
mod forward;
mod inspection;
mod loading;
mod prefix;
mod provider;
mod setup;
mod state;
#[cfg(test)]
mod state_tests;
mod submission;
use crate::constants::F32_BYTES_U64;
use crate::constants::MIB;
use crate::constants::MIB_U64;
use crate::{MetalKernels, device::MetalDevice};
use forward::ForwardNode;
use infer_core::{Error, ErrorCode, OpId, RequestId, Result, StateId, StepId, TensorId};
use infer_ir::{DataflowGraph, LayerProbe, ModelIr, OpTrace, StateKind, TensorSpec, TensorStorage};
use infer_models::{TensorDtype, WeightLoadPlan};
use infer_state::{blocks::BlockLease, kv::KvPages, kv::KvPrefix};
use metal::{Buffer, CommandBuffer};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, collections::VecDeque, sync::Arc, time::Instant};

/// Default token rows in a layer-major prefill chunk.
const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 32;
/// Default temporary upload staging budget (4 MiB).
const DEFAULT_STAGING_BYTES: usize = 4 * MIB;
/// Default device memory budget (512 MiB).
const DEFAULT_MEMORY_BYTES: u64 = 512 * MIB_U64;
/// Default tokens covered by one KV page.
const DEFAULT_PAGE_TOKENS: usize = 16;
/// Default op-trace ring capacity.
const DEFAULT_TRACE_CAPACITY: usize = 8192;
/// Default prefix-cache budget (16 MiB).
const DEFAULT_PREFIX_CACHE_BYTES: u64 = 16 * MIB_U64;
/// Cap on the automatically sized KV pool when no block count is configured.
const MAX_AUTO_KV_BLOCKS: u64 = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalConfig {
    pub memory_bytes: u64,
    pub block_size: usize,
    pub trace_capacity: usize,
    pub prefix_cache_bytes: u64,
    pub probe_bytes: u64,
    /// None chooses a bounded pool from the available device budget.
    #[serde(default)]
    pub kv_cache_blocks: Option<usize>,
    /// Maximum rows in a layer-major prefill chunk. Decode uses one row.
    #[serde(default = "default_prefill_chunk")]
    pub prefill_chunk_tokens: usize,
    /// Temporary upload buffer budget, independent of device-resident weight memory.
    #[serde(default = "default_staging_bytes")]
    pub upload_staging_bytes: usize,
}
const fn default_prefill_chunk() -> usize {
    DEFAULT_PREFILL_CHUNK_TOKENS
}
const fn default_staging_bytes() -> usize {
    DEFAULT_STAGING_BYTES
}
impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            memory_bytes: DEFAULT_MEMORY_BYTES,
            block_size: DEFAULT_PAGE_TOKENS,
            trace_capacity: DEFAULT_TRACE_CAPACITY,
            prefix_cache_bytes: DEFAULT_PREFIX_CACHE_BYTES,
            probe_bytes: 0,
            kv_cache_blocks: None,
            prefill_chunk_tokens: default_prefill_chunk(),
            upload_staging_bytes: default_staging_bytes(),
        }
    }
}
struct Sequence {
    readback: Option<infer_ir::ModelOutput>,
    hidden_spares: Vec<Vec<f32>>,
    readout: infer_ir::OutputReadout,
    capacity: usize,
    reserved_bytes: u64,
    tokens: Vec<u32>,
    tensors: BTreeMap<TensorId, Buffer>,
    hidden: Buffer,
    logits: Buffer,
    probes: Option<Buffer>,
    blocks: Vec<BlockLease>,
    page_table: Buffer,
    token_buffer: Buffer,
    table_mirror: Vec<u32>,
    pending_prefix: Option<CachedPrefix>,
}
#[derive(Clone)]
struct CachedPrefix {
    readout: infer_ir::OutputReadout,
    tokens: Vec<u32>,
    blocks: Vec<BlockLease>,
    tensors: BTreeMap<TensorId, Buffer>,
    hidden: Buffer,
    logits: Buffer,
    bytes: u64,
}
impl KvPrefix for CachedPrefix {
    fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    fn blocks(&self) -> &[BlockLease] {
        &self.blocks
    }
    fn bytes(&self) -> u64 {
        self.bytes
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct SavedSequence {
    readout: infer_ir::OutputReadout,
    capacity: usize,
    tokens: Vec<u32>,
    tensors: BTreeMap<TensorId, Vec<f32>>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    blocks: Vec<BlockLease>,
}
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    identity: String,
    sequences: BTreeMap<StateId, SavedSequence>,
    tokens_executed: u64,
    prefix_hits: u64,
    block_size: usize,
    pool_blocks: usize,
}
struct RestoredBlock {
    lease: BlockLease,
    payload: BTreeMap<TensorId, Vec<f32>>,
    tokens: Vec<u32>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct CommandTrace {
    pub step: StepId,
    pub requests: Vec<RequestId>,
    pub gpu_start_ns: u64,
    pub gpu_duration_ns: u64,
    pub encoded_dispatches: usize,
    pub elapsed_wall_ns: u64,
}
#[derive(Clone, Copy)]
struct TicketWork {
    request: RequestId,
    state: StateId,
    readout: infer_ir::OutputReadout,
}
pub struct MetalTicket {
    owner: Arc<()>,
    command: CommandBuffer,
    step: StepId,
    tasks: Vec<TicketWork>,
    starts: Vec<usize>,
    submitted: Instant,
    dispatches: usize,
    done: bool,
    pending_cache: Vec<CachedPrefix>,
}
pub struct MetalBackend {
    owner: Arc<()>,
    resources: infer_spi::ResourcePool,
    gpu: MetalDevice,
    model: ModelIr,
    graph: DataflowGraph,
    specs: BTreeMap<TensorId, TensorSpec>,
    weights: BTreeMap<TensorId, Buffer>,
    slots: BTreeMap<TensorId, usize>,
    weight_formats: BTreeMap<TensorId, TensorDtype>,
    load_plan: WeightLoadPlan,
    forward_nodes: Vec<ForwardNode>,
    scratch_bytes: u64,
    scratch: Vec<Buffer>,
    dummy: Buffer,
    identity: String,
    config: MetalConfig,
    sequences: BTreeMap<StateId, Sequence>,
    busy: Option<StepId>,
    inflight_states: Vec<StateId>,
    inflight_pins: Vec<BlockLease>,
    ticket_work: Vec<TicketWork>,
    completion_pool: Vec<Vec<infer_ir::TaskOutput>>,
    tokens_executed: u64,
    prefix_hits: u64,
    kv: KvPages<CachedPrefix>,
    kv_buffers: BTreeMap<TensorId, Buffer>,
    kv_block_bytes: u64,
    state_recipe: infer_ir::StateRecipe,
    traces: VecDeque<OpTrace>,
    trace_dropped: u64,
    probes: VecDeque<LayerProbe>,
    probe_dropped: u64,
    commands: VecDeque<CommandTrace>,
    transfers: Vec<CommandBuffer>,
    layer_outputs: BTreeMap<OpId, usize>,
    origin: Instant,
}
fn bytes(n: usize) -> Result<u64> {
    (n as u64)
        .checked_mul(F32_BYTES_U64)
        .ok_or_else(|| Error::invalid("Metal tensor size overflow"))
}
fn u32_size(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::unsupported("Metal shape exceeds u32 ABI"))
}
impl Drop for MetalBackend {
    fn drop(&mut self) {
        if self.busy.is_some() || !self.transfers.is_empty() {
            self.gpu.synchronize();
        }
    }
}

struct KvAllocation {
    blocks: usize,
    buffers: BTreeMap<TensorId, Buffer>,
    block_bytes: u64,
}
fn allocate_kv(
    gpu: &MetalDevice,
    graph: &DataflowGraph,
    config: &MetalConfig,
    retained: u64,
) -> Result<KvAllocation> {
    let kv_block_bytes = graph.tensors.iter().try_fold(0u64, |sum, spec| {
        if matches!(
            spec.storage,
            TensorStorage::State {
                kind: StateKind::AttentionKv,
                ..
            }
        ) {
            sum.checked_add(bytes(
                config
                    .block_size
                    .checked_mul(spec.shape[1])
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(|| Error::invalid("KV block shape overflow"))?,
            )?)
            .ok_or_else(|| Error::invalid("KV block bytes overflow"))
        } else {
            Ok(sum)
        }
    })?;
    let remaining = config
        .memory_bytes
        .saturating_sub(retained)
        .saturating_sub(config.prefix_cache_bytes.saturating_mul(2))
        .saturating_sub(config.probe_bytes);
    let max_blocks = remaining.checked_div(kv_block_bytes).unwrap_or(0);
    let blocks = config
        .kv_cache_blocks
        .unwrap_or_else(|| (max_blocks / 2).min(MAX_AUTO_KV_BLOCKS) as usize);
    if blocks as u64 > max_blocks || blocks >= u32::MAX as usize {
        return Err(Error::new(
            ErrorCode::Capacity,
            "KV pool exceeds device budget",
        ));
    }
    let mut kv_buffers = BTreeMap::new();
    for spec in &graph.tensors {
        if matches!(
            spec.storage,
            TensorStorage::State {
                kind: StateKind::AttentionKv,
                ..
            }
        ) {
            let elements = blocks
                .checked_mul(config.block_size)
                .and_then(|n| n.checked_mul(spec.shape[1]))
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| Error::invalid("KV pool shape overflow"))?;
            u32_size(elements)?;
            // A zero-capacity pool has no storage; admission will reject its first KV page.
            if blocks > 0 {
                kv_buffers.insert(spec.id, gpu.zeros(elements)?);
            }
        }
    }
    Ok(KvAllocation {
        blocks,
        buffers: kv_buffers,
        block_bytes: kv_block_bytes,
    })
}
