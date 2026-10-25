use super::ActivationArena;
use crate::device::ReadbackSource;
use crate::device::asynchronous::block_on;
use crate::device::cublaslt::Support;
use crate::device::{PinnedUpload, Readbacks};
use crate::tuning::gated_norm::prepare;
use crate::tuning::resident_gemm::ResidentTuningSession;
use infer_models::mrope::ROTARY_AXES;
use std::collections::BTreeSet;
use std::future::Future;
use std::slice::from_ref;
mod geometry;
mod greedy;
use crate::{
    constants::{
        CONV_KERNEL_SIZE, CONV_STATE_TENSORS, F32_BYTES, KV_CACHE_RANK, MAX_HIDDEN_SIZE,
        METADATA_FIELDS, MIB, PREFILL_LANES,
    },
    device::{CudaDevice, device_error},
    mlp::ProjectionWeight,
    strategy::LinearTiling,
};
use cutile::{half::bf16, prelude::*};
pub use geometry::ProgramGeometry;
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, StateKind, TensorStorage};
use std::collections::BTreeMap;

type SlotState = (
    super::batch::States,
    super::fp8_cache::Fp8Caches,
    Tensor<i32>,
);
type InitialState = (
    super::batch::States,
    super::fp8_cache::Fp8Caches,
    Option<Tensor<i32>>,
);

pub struct FusionWeights {
    pub projection: ProjectionWeight,
    /// Tile for the fusion projection, selected like every other projection weight.
    pub tiling: LinearTiling,
    pub norms: Arc<Tensor<f32>>,
    pub epsilon: f32,
    pub offset: f32,
}

impl FusionWeights {
    fn validate(&self, hidden: usize) -> Result<()> {
        self.projection.validate(hidden, 2 * hidden)?;
        if self.norms.shape() != [2, i32::try_from(hidden).map_err(device_error)?]
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.offset.is_finite()
        {
            return Err(Error::invalid("MTP fusion dimensions or normalization"));
        }
        Ok(())
    }
}

pub(super) struct FusionWorkspace {
    pub embedding: Tensor<f32>,
    pub normalized: Option<Tensor<f32>>,
}

impl FusionWorkspace {
    pub(super) fn allocate(
        device: &CudaDevice,
        hidden: usize,
        enabled: bool,
    ) -> Result<Option<Self>> {
        let fusion = if enabled {
            Some(Self {
                embedding: api::zeros::<f32>(&[hidden])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
                normalized: Some(
                    api::zeros::<f32>(&[2 * hidden])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                ),
            })
        } else {
            None
        };
        Ok(fusion)
    }
}

pub struct ProgramWeights {
    /// Measure uncovered resident GEMM geometries before graph capture.
    pub autotune: bool,
    /// One finite new-shape tuning allowance for this loaded model, shared with its draft.
    pub resident_tuning: Arc<ResidentTuningSession>,
    pub batch_width: usize,
    /// Widest captured prompt graph.
    pub prefill_width: usize,
    /// Second, narrower prompt graph captured alongside [`Self::prefill_width`]; zero when the
    /// program captures one prompt graph. A prompt chunk costs one fixed-width replay, so a
    /// 51-token prompt wants the narrow graph while a 511-token prompt wants the wide one.
    pub narrow_prefill_width: usize,
    pub kv_scales: BTreeMap<TensorId, [f32; 2]>,
    pub tiling: BTreeMap<TensorId, LinearTiling>,
    pub fusion: Option<FusionWeights>,
    /// Target prefill attention numerical profile, resolved once at load. The draft proposal
    /// program keeps its existing BF16 behavior regardless of this selection.
    pub attention_profile: crate::AttentionProfile,
    pub projections: BTreeMap<TensorId, ProjectionWeight>,
    pub input_scales: BTreeMap<TensorId, f32>,
    pub fp8_inputs: BTreeSet<TensorId>,
    /// Capture the chunked gated-delta prompt kernel rather than the per-token one.
    pub chunked_recurrent: bool,
    pub expert_banks: BTreeMap<TensorId, super::moe::ExpertBank>,
    pub constants: BTreeMap<TensorId, Arc<Tensor<f32>>>,
    pub embeddings: BTreeMap<TensorId, Arc<Tensor<bf16>>>,
    pub rope_axes: BTreeMap<infer_core::OpId, Arc<Tensor<i32>>>,
    pub rope_frequencies: BTreeMap<infer_core::OpId, Arc<Tensor<f32>>>,
    /// Whether the model's `RoPE` carries three-axis (`t`, `h`, `w`) positions.
    pub mrope: bool,
    /// Prompt-graph scratch shared by every sequence of this program.
    pub prompt_scratch: super::prompt_scratch::Registry,
}

#[derive(Clone, Copy)]
pub(super) enum ActivationQuantization {
    Fp4(f32),
    Fp8Token,
}
impl ProgramWeights {
    /// Prompt graph widths this program captures, widest first; empty when it captures none.
    ///
    /// A width below `PREFILL_LANES` is not a prompt graph: it is the verification width the
    /// program falls back to.
    #[must_use]
    pub fn prompt_widths(&self) -> Vec<usize> {
        let mut widths = Vec::with_capacity(2);
        for width in [self.prefill_width, self.narrow_prefill_width] {
            if width >= PREFILL_LANES && !widths.contains(&width) {
                widths.push(width);
            }
        }
        widths
    }

    /// Widest captured prompt graph, or zero when the program captures none.
    #[must_use]
    pub fn widest_prompt_width(&self) -> usize {
        self.prompt_widths().into_iter().max().unwrap_or(0)
    }

    /// Total prompt lanes across every captured prompt graph.
    #[must_use]
    pub fn prompt_lane_total(&self) -> usize {
        self.prompt_widths().into_iter().sum()
    }

    pub(super) fn activation_quantization(&self, id: TensorId) -> Option<ActivationQuantization> {
        self.input_scales
            .get(&id)
            .copied()
            .map(ActivationQuantization::Fp4)
            .or_else(|| {
                self.fp8_inputs
                    .contains(&id)
                    .then_some(ActivationQuantization::Fp8Token)
            })
    }
}

/// Captures a complete single-token forward pass; all state and activations stay on CUDA.
pub struct DeviceProgram {
    /// Slot target handles own state views and reset only; their forward runs in the pool.
    forward_captured: bool,
    batch: Option<super::batch::BatchGraph>,
    last_batch: Option<(usize, usize)>,
    prompt_batch: Option<super::batch::BatchGraph>,
    /// Second, narrower prompt graph; chunks at most this wide replay here instead.
    prompt_narrow: Option<super::batch::BatchGraph>,
    graph: CudaGraph<()>,
    greedy: Option<greedy::DeviceGreedy>,
    token_readout: Option<super::token_readout::TokenReadout>,
    readbacks: Readbacks,
    prefill: CudaGraph<()>,
    metadata: Tensor<i32>,
    /// Identity block table over this program's own KV cache; see `capture::Capture::table`.
    table: Tensor<i32>,
    nvfp4: super::nvfp4_gemm::Workspace,
    delegated: Support,
    attention: super::attention_decode::Workspace,
    external: Tensor<f32>,
    uploads: ProgramUploads,
    // Declared after the graphs: the state tensors outlive every capture that references them,
    // and prefix caching reads them back for snapshots.
    states: super::batch::States,
    kv_states: BTreeSet<TensorId>,
    fp8_states: super::fp8_cache::Fp8Caches,
    lane_external: Vec<Tensor<f32>>,
    // Mutable upload handle for one contiguous external hidden per prompt lane. Every row in
    // `lane_external` keeps the backing storage alive, so this handle is only the transfer
    // target and is written before the prompt replay on the device stream.
    lane_external_owner: Tensor<f32>,
    // Hidden row of the position before the next device-primed chunk.
    prime_previous: Tensor<f32>,
    // Fusion workspaces are referenced by the captured graphs and never touched afterwards.
    lane_fusion: Vec<Option<FusionWorkspace>>,
    reset_graph: CudaGraph<()>,
    hidden: Arc<Tensor<f32>>,
    logits: Arc<Tensor<f32>>,
    device: CudaDevice,
    capacity: usize,
    vocabulary: usize,
    next_position: usize,
    attention_only: bool,
    requires_external: bool,
    pub activation_bytes: usize,
    pub(crate) reclaimable_bytes: u64,
    released_verification_bytes: u64,
}

enum ExternalInput<'a> {
    Host(&'a [f32]),
    Device(&'a Tensor<f32>),
}
impl ExternalInput<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Host(values) => values.len(),
            Self::Device(tensor) => tensor.size(),
        }
    }
    fn update(
        self,
        device: &CudaDevice,
        destination: &mut Tensor<f32>,
        staging: Option<&mut PinnedUpload<f32>>,
    ) -> Result<()> {
        match self {
            Self::Host(values) => {
                let staging =
                    staging.ok_or_else(|| Error::invariant("external hidden pinned staging"))?;
                if values.len() != destination.size() {
                    return Err(Error::invalid("external hidden shape"));
                }
                staging.copy(device, destination, values)
            }
            Self::Device(tensor) => {
                if tensor.device_id() != device.stream.device().ordinal() {
                    return Err(Error::invalid("external hidden device mismatch"));
                }
                device.copy_d2d(destination, tensor, tensor.size())
            }
        }
    }
}

/// Conservative lower bound on uniquely owned device storage; excludes graph metadata,
/// readback buffers and auxiliary workspaces. Idle-state admission can reclaim this much.
/// `verify`/`prompt` are the lane counts actually captured; a pool-bound program captured
/// no verification graph (and possibly no wide prompt graph), so those snapshots and arena
/// shares are not retained either.
fn retained_tensor_bytes(
    graph: &DataflowGraph,
    states: &super::batch::States,
    fp8: &super::fp8_cache::Fp8Caches,
    activation_bytes: usize,
    verify: usize,
    prompt: usize,
) -> u64 {
    let verify = if verify > 1 { verify } else { 0 };
    let activations = activation_bytes as u64 * (1 + verify + prompt) as u64;
    let state_bytes = states
        .values()
        .flatten()
        .map(|t| t.size() as u64 * F32_BYTES as u64)
        .sum::<u64>();
    let packed_bytes = fp8
        .values()
        .map(|(k, v)| (k.size() + v.size()) as u64)
        .sum::<u64>();
    let recurrent = graph
        .tensors
        .iter()
        .filter(|t| {
            matches!(
                t.storage,
                TensorStorage::State {
                    kind: StateKind::Conv | StateKind::LinearAttention,
                    ..
                }
            )
        })
        .filter_map(|t| states.get(&t.id))
        .flatten()
        .map(|t| t.size() as u64 * F32_BYTES as u64)
        .sum::<u64>();
    activations + state_bytes + packed_bytes + recurrent * verify as u64
}

/// One contiguous external-hidden owner, its per-lane row views and fusion workspaces.
type LaneFusion = (Tensor<f32>, Vec<Tensor<f32>>, Vec<Option<FusionWorkspace>>);

/// Per-lane external hidden row views and fusion workspaces for batched captures.
///
/// A fused program consumes one external hidden per lane. One contiguous owner makes a
/// prompt chunk one host transfer; the captured graphs keep pointing at the same disjoint
/// rows, so draft priming no longer uploads and rebinds one buffer per token. Each row
/// holds the backing storage itself, so the owner handle is free to remain the upload target.
fn allocate_lane_fusion(
    device: &CudaDevice,
    hidden: usize,
    weights: &ProgramWeights,
    geometry: ProgramGeometry,
) -> Result<LaneFusion> {
    let width = geometry.widths().into_iter().max().unwrap_or(1);
    let enabled = weights.fusion.is_some();
    let elements = width
        .checked_mul(hidden)
        .ok_or_else(|| Error::invalid("external hidden size overflow"))?;
    let owner = api::zeros::<f32>(&[elements])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let external = super::arena::batch_rows(&owner, width)?;
    let fused = fused_lanes(device, weights)?;
    let mut fusion = Vec::with_capacity(width);
    for lane in 0..width {
        fusion.push(FusionWorkspace::allocate(
            device,
            hidden,
            enabled && lane < fused,
        )?);
    }
    Ok((owner, external, fusion))
}

/// Lanes that need their own fusion workspace: every lane, unless batched prompt fusion covers
/// the prompt graph and only decode and verification lanes record the fusion per lane.
pub fn fused_lanes(device: &CudaDevice, weights: &ProgramWeights) -> Result<usize> {
    let width = weights.prefill_width.max(weights.batch_width).max(1);
    Ok(
        if super::fusion_batch::batched(weights, weights.prefill_width, device.profile()?) {
            weights.batch_width.max(1)
        } else {
            width
        },
    )
}

/// Activation arena budget for this device, as a host `usize`.
fn arena_budget(device: &CudaDevice) -> Result<usize> {
    usize::try_from(device.profile()?.arena_budget_bytes()).map_err(device_error)
}
/// Validate one program request and return the padded capacity it will be captured at.
/// Copy one state tensor between programs of equal capacity: with `kv = (rows, capacity)` a
/// head-major `[kv_heads, capacity, head_dim]` cache moves its first `rows` positions per head,
/// otherwise the tensor moves whole.
fn copy_prefix<T: DType>(
    device: &CudaDevice,
    dst: &mut Tensor<T>,
    src: &Tensor<T>,
    kv: Option<(usize, usize)>,
) -> Result<()> {
    if dst.shape() != src.shape() {
        return Err(Error::invariant("program state copy shape"));
    }
    if dst.device_pointer().cu_deviceptr() == src.device_pointer().cu_deviceptr() {
        return Ok(());
    }
    let shape = src.shape();
    if let Some((rows, capacity)) = kv {
        if shape.len() != KV_CACHE_RANK || usize::try_from(shape[1]).ok() != Some(capacity) {
            return Err(Error::invariant("program KV copy layout"));
        }
        return device.copy_head_prefix(dst, src, rows);
    }
    device.copy_d2d(dst, src, src.size())
}

fn program_capacity(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    capacity: usize,
    hidden: usize,
    vocabulary: usize,
) -> Result<usize> {
    if vocabulary == 0 || hidden == 0 || hidden > MAX_HIDDEN_SIZE {
        return Err(Error::invalid("resident capacity or vocabulary"));
    }
    if graph
        .nodes
        .iter()
        .any(|node| node.inputs.iter().any(|id| Some(*id) == graph.logits))
    {
        return Err(Error::unsupported("logits must be a terminal graph output"));
    }
    if let Some(fusion) = &weights.fusion {
        fusion.validate(hidden)?;
    }
    super::capacity::padded(capacity)
}

impl DeviceProgram {
    /// A leased sequence verifies through the shared pool. Its private rollback
    /// graph is idle and can be rebuilt if the sequence later returns to serial use.
    pub(crate) fn discard_verification(&mut self) -> bool {
        if self.batch.take().is_none() {
            return false;
        }
        self.last_batch = None;
        let before = self.reclaimable_bytes;
        let state_bytes = self
            .states
            .values()
            .flatten()
            .map(|t| t.size() as u64 * F32_BYTES as u64)
            .sum::<u64>();
        let packed_bytes = self
            .fp8_states
            .values()
            .map(|(k, v)| (k.size() + v.size()) as u64)
            .sum::<u64>();
        let prompt: usize = [self.prompt_batch.as_ref(), self.prompt_narrow.as_ref()]
            .into_iter()
            .flatten()
            .map(|g| g.width)
            .sum();
        self.reclaimable_bytes =
            state_bytes + packed_bytes + self.activation_bytes as u64 * (1 + prompt) as u64;
        self.released_verification_bytes = before - self.reclaimable_bytes;
        true
    }

    /// Remove private verification before pooled reuse. Cached admission retains its old
    /// charge, so no second release credit may survive this transition.
    pub(crate) fn forfeit_verification(&mut self) {
        self.discard_verification();
        self.released_verification_bytes = 0;
    }

    pub(crate) const fn released_verification_bytes(&self) -> u64 {
        self.released_verification_bytes
    }

    pub(crate) fn ensure_verification(
        &mut self,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
    ) -> Result<()> {
        if self.batch.is_some() || weights.batch_width < 2 {
            return Ok(());
        }
        self.nvfp4
            .ensure_width(&self.device, graph, weights, weights.batch_width)?;
        let mut builder = super::batch::BatchBuilder {
            device: &self.device,
            graph,
            weights,
            nvfp4: &mut self.nvfp4,
            delegated: Some(&mut self.delegated),
            attention: &mut self.attention,
            states: &mut self.states,
            fp8_states: &mut self.fp8_states,
            external: &self.external,
            lane_external: &mut self.lane_external,
            lane_external_owner: Some(&self.lane_external_owner),
            lane_fusion: &mut self.lane_fusion,
            tables: from_ref(&self.table),
            capacity: self.capacity,
            width: weights.batch_width,
        };
        self.batch = Some(builder.build()?);
        let prompt: usize = [self.prompt_batch.as_ref(), self.prompt_narrow.as_ref()]
            .into_iter()
            .flatten()
            .map(|graph| graph.width)
            .sum();
        self.reclaimable_bytes = retained_tensor_bytes(
            graph,
            &self.states,
            &self.fp8_states,
            self.activation_bytes,
            weights.batch_width,
            prompt,
        );
        self.released_verification_bytes = 0;
        Ok(())
    }

    /// # Errors
    /// Rejects unsupported graph shapes, storage budgets or CUDA capture failures.
    pub fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        capacity: usize,
        hidden: usize,
        vocabulary: usize,
    ) -> Result<Self> {
        let geometry = ProgramGeometry::new(weights, true);
        Self::new_impl(
            device,
            graph,
            weights,
            (capacity, hidden, vocabulary),
            geometry,
            None,
        )
    }

    /// A sequence that will verify through the shared slot pool skips its private
    /// verification graph: the pool's captured graph replaces it, so capturing one here
    /// would cost memory, capture time and admission budget the pool makes useless.
    /// # Errors
    /// Rejects unsupported graph shapes, storage budgets or CUDA capture failures.
    pub(crate) fn new_pooled(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        capacity: usize,
        hidden: usize,
        vocabulary: usize,
    ) -> Result<Self> {
        let geometry = ProgramGeometry::new(weights, false);
        Self::new_impl(
            device,
            graph,
            weights,
            (capacity, hidden, vocabulary),
            geometry,
            None,
        )
    }

    /// A pooled program with the wide prompt graph that prompts for slot-resident sequences:
    /// a sequence's state moves in ([`Self::copy_state_from`]), a wide chunk replays, and the
    /// state moves back to its slot. One worker replaces a wide graph per slot.
    /// # Errors
    /// Rejects invalid capacity or CUDA graph capture failures.
    pub(crate) fn new_worker(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        geometry: (usize, usize, usize),
        kv: Option<super::fp8_cache::Fp8Caches>,
        prompt_width: usize,
    ) -> Result<Self> {
        // A paged pool's worker reads and writes the shared KV blocks of whichever sequence it
        // runs, through a table it copies from that sequence; its recurrent state is its own.
        let slot = match kv {
            Some(kv) => {
                let capacity = super::capacity::padded(geometry.0)?;
                Some((
                    allocate_states(device, graph, capacity, weights)?,
                    kv,
                    super::metadata::identity_table(device, capacity)?,
                ))
            }
            None => None,
        };
        let wide = ProgramGeometry::worker(weights, prompt_width);
        if wide.prompts[0] == 0 {
            return Err(Error::invalid("prompt worker width"));
        }
        Self::new_impl(device, graph, weights, geometry, wide, slot)
    }

    /// Copy `source`'s recurrent states and its first `rows` KV positions into this program,
    /// whose capacity must match, and continue from `rows`.
    /// # Errors
    /// Rejects mismatched capacities or state layouts and driver copy failures.
    pub(crate) fn copy_state_from(&mut self, source: &Self, rows: usize) -> Result<()> {
        if source.capacity != self.capacity || rows > self.capacity {
            return Err(Error::invalid("program state copy geometry"));
        }
        let device = &self.device;
        for (id, tensors) in &mut self.states {
            let from = source
                .states
                .get(id)
                .ok_or_else(|| Error::invariant("program state copy source"))?;
            if from.len() != tensors.len() {
                return Err(Error::invariant("program state copy layout"));
            }
            // StateKind supplies the layout contract for unscaled KV as well as FP8.
            // Recurrent state moves whole; untouched KV tails are never read by attention.
            let kv = self.kv_states.contains(id).then_some((rows, self.capacity));
            for (dst, src) in tensors.iter_mut().zip(from) {
                copy_prefix(device, dst, src, kv)?;
            }
        }
        for (id, (keys, values)) in &mut self.fp8_states {
            let (from_keys, from_values) = source
                .fp8_states
                .get(id)
                .ok_or_else(|| Error::invariant("program state copy source"))?;
            copy_prefix(device, keys, from_keys, Some((rows, self.capacity)))?;
            copy_prefix(device, values, from_values, Some((rows, self.capacity)))?;
        }
        // Paged sequences share KV storage and differ in their block tables.
        copy_prefix(device, &mut self.table, &source.table, None)?;
        self.next_position = rows;
        self.last_batch = None;
        Ok(())
    }

    /// A pooled program whose recurrent states and KV caches are `slot`, one slot pool slot's
    /// storage seen through aliasing views ([`super::slot_batch::SlotPool::slot_views`]): a
    /// sequence prefills straight into the slot it then decodes from, with no private copy.
    /// # Errors
    /// Rejects invalid capacity, mismatched slot layouts or CUDA graph capture failures.
    pub(crate) fn new_on_slot(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        (capacity, hidden, vocabulary): (usize, usize, usize),
        slot: SlotState,
    ) -> Result<Self> {
        // Every prompt chunk runs on the prompt worker, so a slot's programs capture no prompt
        // graph: the draft's wide one alone retained 160 MB of projection scratch per slot.
        let narrow = ProgramGeometry::from_flags(
            weights.batch_width,
            weights.prefill_width,
            0,
            false,
            false,
        );
        Self::new_impl(
            device,
            graph,
            weights,
            (capacity, hidden, vocabulary),
            narrow,
            Some(slot),
        )
    }

    /// Zeroed decode metadata tensor and external hidden row shared by every capture.
    fn constant_inputs(
        device: &CudaDevice,
        hidden: usize,
        capacity: usize,
    ) -> Result<(Tensor<i32>, Tensor<f32>, Tensor<i32>)> {
        let metadata = device.upload(vec![0_i32; METADATA_FIELDS * 2], &[METADATA_FIELDS * 2])?;
        let metadata =
            Arc::try_unwrap(metadata).map_err(|_| Error::invariant("unique metadata"))?;
        let external = api::zeros::<f32>(&[hidden])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        // Step 1 of the KV refactor: an identity mapping over this program's own cache.
        let table = super::metadata::identity_table(device, capacity)?;
        Ok((metadata, external, table))
    }

    /// Retained bytes priced from the graphs this program actually captured.
    fn program_retained_bytes(
        graph: &DataflowGraph,
        states: &super::batch::States,
        fp8: &super::fp8_cache::Fp8Caches,
        activation_bytes: usize,
        captures: &super::batch::ProgramCaptures,
    ) -> u64 {
        retained_tensor_bytes(
            graph,
            states,
            fp8,
            activation_bytes,
            captures.batch.as_ref().map_or(0, |graph| graph.width),
            [
                captures.prompt_batch.as_ref(),
                captures.prompt_narrow.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(|graph| graph.width)
            .sum(),
        )
    }

    fn program_workspaces(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        geometry: ProgramGeometry,
        capacity: usize,
        forward_captured: bool,
    ) -> Result<(
        super::nvfp4_gemm::Workspace,
        Support,
        super::attention_decode::Workspace,
    )> {
        let empty = DataflowGraph::default();
        let captured_graph = if forward_captured { graph } else { &empty };
        let mut attention =
            super::attention_decode::Workspace::new(device, captured_graph, capacity)?;
        let (nvfp4, delegated) = if forward_captured {
            attention.prepare_mma(
                device,
                graph,
                weights,
                capacity,
                geometry.prompts.into_iter().max().unwrap_or(0),
            )?;
            prepare(device, graph, weights, geometry.prompts);
            let nvfp4 = super::nvfp4_gemm::Workspace::program(device, graph, weights, geometry)?;
            let mut delegated = Support::new(device);
            warm_delegated(&mut delegated, weights, geometry);
            (nvfp4, delegated)
        } else {
            (
                super::nvfp4_gemm::Workspace::new(device, &empty, weights, &[])?,
                Support::new(device),
            )
        };
        Ok((nvfp4, delegated, attention))
    }

    fn new_impl(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        (capacity, hidden, vocabulary): (usize, usize, usize),
        geometry: ProgramGeometry,
        slot: Option<SlotState>,
    ) -> Result<Self> {
        let forward_captured = !(slot.is_some()
            && weights.fusion.is_none()
            && geometry.verification == 0
            && geometry.prompts == [0, 0]);
        super::validation::validate(graph, weights)?;
        let capacity = program_capacity(graph, weights, capacity, hidden, vocabulary)?;
        let mut arena = ActivationArena::new(device, graph, arena_budget(device)?)?;
        let activation_bytes = arena.bytes();
        let (mut nvfp4, mut delegated, mut attention) =
            Self::program_workspaces(device, graph, weights, geometry, capacity, forward_captured)?;
        let (mut states, mut fp8_states, slot_table) =
            initial_state(device, graph, weights, capacity, slot)?;
        let (metadata, external, identity) = Self::constant_inputs(device, hidden, capacity)?;
        // A slot program addresses its slot's KV blocks through the slot's own table.
        let table = slot_table.unwrap_or(identity);
        let mut fusion = FusionWorkspace::allocate(device, hidden, weights.fusion.is_some())?;
        let (lane_external_owner, mut lane_external, mut lane_fusion) =
            allocate_lane_fusion(device, hidden, weights, geometry)?;
        let prime_previous = api::zeros::<f32>(&[hidden])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut builder = super::batch::BatchBuilder {
            device,
            graph,
            weights,
            nvfp4: &mut nvfp4,
            delegated: Some(&mut delegated),
            attention: &mut attention,
            states: &mut states,
            fp8_states: &mut fp8_states,
            external: &external,
            lane_external: &mut lane_external,
            lane_external_owner: Some(&lane_external_owner),
            lane_fusion: &mut lane_fusion,
            tables: from_ref(&table),
            capacity,
            width: weights.batch_width,
        };
        let captures = builder.build_program(
            geometry,
            &mut arena,
            &metadata,
            &mut fusion,
            forward_captured,
        )?;
        // Forward captures finalize the resident state bindings before reset records them.
        let reset_graph = capture_reset(device, &mut states)?;
        let reclaimable_bytes =
            Self::program_retained_bytes(graph, &states, &fp8_states, activation_bytes, &captures);
        let readouts = greedy::ProgramReadouts::new(
            device,
            &mut arena,
            graph,
            weights,
            vocabulary,
            forward_captured,
        )?;
        Ok(Self {
            forward_captured,
            table,
            nvfp4,
            delegated,
            attention,
            batch: captures.batch,
            prompt_batch: captures.prompt_batch,
            prompt_narrow: captures.prompt_narrow,
            last_batch: None,
            graph: captures.forward,
            greedy: readouts.greedy,
            token_readout: readouts.token_readout,
            readbacks: Readbacks::pair(device, &readouts.hidden, &readouts.logits)?,
            prefill: captures.prefill,
            metadata,
            uploads: ProgramUploads::new(
                device,
                external.size(),
                lane_external_owner.size(),
                weights.fusion.as_ref(),
            )?,
            external,
            lane_external,
            lane_external_owner,
            prime_previous,
            states,
            kv_states: attention_states(graph),
            fp8_states,
            lane_fusion,
            reset_graph,
            hidden: readouts.hidden,
            logits: readouts.logits,
            device: device.clone(),
            capacity,
            vocabulary,
            next_position: 0,
            requires_external: weights.fusion.is_some(),
            attention_only: attention_only(graph),
            activation_bytes,
            reclaimable_bytes,
            released_verification_bytes: u64::default(),
        })
    }

    /// # Errors
    /// Rejects out-of-range positions/tokens, invalid overrides or CUDA failures.
    pub fn step(
        &mut self,
        token: u32,
        position: usize,
        kv_position: usize,
        external: Option<&[f32]>,
        read_logits: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        block_on(self.step_async(token, position, kv_position, external, read_logits))
    }

    pub(crate) async fn step_async(
        &mut self,
        token: u32,
        position: usize,
        kv_position: usize,
        external: Option<&[f32]>,
        read_logits: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.step_readout_async(token, position, kv_position, external, read_logits, true)
            .await
    }

    pub(crate) async fn step_readout_async(
        &mut self,
        token: u32,
        position: usize,
        kv_position: usize,
        external: Option<&[f32]>,
        read_logits: bool,
        read_hidden: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.execute_step_async(
            token,
            [position; ROTARY_AXES],
            kv_position,
            external,
            read_logits,
            read_hidden,
        )
        .await
    }

    #[cfg(test)]
    pub(crate) fn step_selected(
        &mut self,
        token: u32,
        position: usize,
        read_hidden: bool,
    ) -> Result<super::batch::BatchReadout> {
        block_on(self.step_selected_async(token, position, read_hidden))
    }

    pub(crate) async fn step_selected_async(
        &mut self,
        token: u32,
        position: usize,
        read_hidden: bool,
    ) -> Result<super::batch::BatchReadout> {
        if self.token_readout.is_none() {
            return self
                .step_readout_async(token, position, position, None, true, read_hidden)
                .await
                .map(|row| super::batch::BatchReadout::host(vec![row]));
        }
        self.prepare_step_async(token, [position; ROTARY_AXES], position, None, true)
            .await?;
        let sources = [
            read_hidden.then(|| ReadbackSource::whole(Arc::clone(&self.hidden))),
            None,
        ];
        let (rows, tokens) = self
            .token_readout
            .as_mut()
            .ok_or_else(|| Error::invariant("program token readout"))?
            .run_async(
                &self.device,
                &self.graph,
                &mut self.readbacks,
                &sources,
                &[0],
            )
            .await?;
        self.last_batch = None;
        self.next_position = position + 1;
        Ok(super::batch::BatchReadout {
            rows,
            tokens: Some(tokens.into_iter().map(Some).collect()),
        })
    }

    /// Execute with independent time/height/width coordinates and a sequential KV index.
    /// # Errors
    /// Rejects invalid tokens, coordinates, overrides or device operations.
    pub fn step_positioned(
        &mut self,
        token: u32,
        position: [usize; ROTARY_AXES],
        kv_position: usize,
        external: Option<&[f32]>,
        read_logits: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        block_on(self.step_positioned_async(token, position, kv_position, external, read_logits))
    }

    pub(crate) async fn step_positioned_async(
        &mut self,
        token: u32,
        position: [usize; ROTARY_AXES],
        kv_position: usize,
        external: Option<&[f32]>,
        read_logits: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.execute_step_async(token, position, kv_position, external, read_logits, true)
            .await
    }

    async fn execute_step_async(
        &mut self,
        token: u32,
        position: [usize; ROTARY_AXES],
        kv_position: usize,
        external: Option<&[f32]>,
        read_logits: bool,
        read_hidden: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.prepare_step_async(
            token,
            position,
            kv_position,
            external.map(ExternalInput::Host),
            read_logits,
        )
        .await?;
        let graph = if read_logits {
            &self.graph
        } else {
            &self.prefill
        };
        let mut rows = self
            .readbacks
            .run_async(
                &self.device,
                graph,
                &[
                    read_hidden.then(|| ReadbackSource::whole(Arc::clone(&self.hidden))),
                    read_logits.then(|| ReadbackSource::whole(Arc::clone(&self.logits))),
                ],
            )
            .await?
            .into_iter();
        let result = (
            rows.next()
                .ok_or_else(|| Error::invariant("hidden readback"))?,
            rows.next()
                .ok_or_else(|| Error::invariant("logits readback"))?,
        );
        self.last_batch = None;
        self.next_position = kv_position + 1;
        Ok(result)
    }

    async fn prepare_step_async(
        &mut self,
        token: u32,
        position: [usize; ROTARY_AXES],
        kv_position: usize,
        external: Option<ExternalInput<'_>>,
        read_logits: bool,
    ) -> Result<()> {
        if !self.forward_captured {
            return Err(Error::invalid(
                "slot target forwards require the shared pool",
            ));
        }
        if kv_position >= self.capacity
            || usize::try_from(token).map_err(device_error)? >= self.vocabulary
        {
            return Err(Error::invalid(
                "resident token or KV position out of bounds",
            ));
        }
        if self.requires_external && external.is_none() {
            return Err(Error::invalid("MTP requires previous hidden state"));
        }
        if external
            .as_ref()
            .is_some_and(|values| values.len() != self.external.size())
        {
            return Err(Error::invalid("external hidden dimensions"));
        }
        if kv_position == 0 {
            self.reset_async().await?;
        }
        if kv_position != self.next_position {
            return Err(Error::invalid("nonsequential device state position"));
        }
        let metadata = [
            i32::try_from(position[0]).map_err(device_error)?,
            i32::try_from(token).map_err(device_error)?,
            i32::try_from(kv_position).map_err(device_error)?,
            i32::from(external.is_some()),
        ];
        let graph = if read_logits {
            &self.graph
        } else {
            &self.prefill
        };
        super::metadata::update_rotary(graph, &mut self.metadata, metadata, position)?;
        if let Some(external) = external {
            external.update(
                &self.device,
                &mut self.external,
                self.uploads.external.as_mut(),
            )?;
        }
        Ok(())
    }

    #[must_use]
    pub fn batch_width(&self) -> usize {
        self.batch.as_ref().map_or(1, |batch| batch.width)
    }

    /// Execute several target inputs in node order, then retain checkpoints for prefix commit.
    /// # Errors
    /// Rejects invalid positions, unsupported widths or failed device execution.
    pub fn step_batch(&mut self, tokens: &[u32], position: usize) -> Result<super::BatchOutput> {
        block_on(self.step_batch_async(tokens, position))
    }

    pub(crate) async fn step_batch_async(
        &mut self,
        tokens: &[u32],
        position: usize,
    ) -> Result<super::BatchOutput> {
        Ok(self
            .step_batch_readout_async(tokens, position, false)
            .await?
            .rows)
    }

    #[cfg(test)]
    pub(crate) fn step_batch_readout(
        &mut self,
        tokens: &[u32],
        position: usize,
        plain_greedy: bool,
    ) -> Result<super::batch::BatchReadout> {
        block_on(self.step_batch_readout_async(tokens, position, plain_greedy))
    }

    pub(crate) async fn step_batch_readout_async(
        &mut self,
        tokens: &[u32],
        position: usize,
        plain_greedy: bool,
    ) -> Result<super::batch::BatchReadout> {
        let width = self.batch_width();
        if tokens.is_empty()
            || tokens.len() > width
            || self.batch.is_none()
            || position
                .checked_add(width)
                .is_none_or(|end| end > self.capacity)
            || tokens
                .iter()
                .any(|t| usize::try_from(*t).map_or(true, |n| n >= self.vocabulary))
        {
            return Err(Error::invalid("verification tokens, width or capacity"));
        }
        if position == 0 {
            self.reset_async().await?;
        }
        if position != self.next_position {
            return Err(Error::invalid("verification state position"));
        }
        let output = self
            .batch
            .as_mut()
            .ok_or_else(|| Error::invariant("batch graph"))?
            .run_readout_async(&self.device, tokens, position, None, true, plain_greedy)
            .await?;
        self.next_position = position + width;
        self.last_batch = Some((position, tokens.len()));
        self.commit_batch_async(tokens.len()).await?;
        Ok(output)
    }

    /// Seed the logical position after a prefix restore, so the next prefill resumes there.
    pub const fn set_position(&mut self, position: usize) {
        self.next_position = position;
    }

    /// Resident state tensors of this program, one vector per state id.
    ///
    /// Prefix caching snapshots these with device-to-device copies and restores them into a
    /// later sequence, so the buffers must stay reachable after capture.
    #[must_use]
    pub const fn states(&self) -> &super::batch::States {
        &self.states
    }

    /// Mutable resident state tensors, for prefix restore.
    #[must_use]
    pub const fn states_mut(&mut self) -> &mut super::batch::States {
        &mut self.states
    }

    /// FP8 scale side tables paired with the resident state tensors.
    #[must_use]
    pub const fn fp8_states(&self) -> &super::fp8_cache::Fp8Caches {
        &self.fp8_states
    }

    /// Mutable quantized KV storage for a prefix restore.
    #[must_use]
    pub const fn fp8_states_mut(&mut self) -> &mut super::fp8_cache::Fp8Caches {
        &mut self.fp8_states
    }

    /// Device that owns this program's state and graphs.
    pub(crate) const fn device(&self) -> &CudaDevice {
        &self.device
    }

    #[must_use]
    pub fn prefill_width(&self) -> usize {
        let widest = self
            .prompt_batch
            .as_ref()
            .map_or(0, |prompt| prompt.width)
            .max(self.prompt_narrow.as_ref().map_or(0, |prompt| prompt.width));
        if widest > 0 {
            widest
        } else {
            self.batch_width()
        }
    }

    #[cfg(test)]
    pub(crate) const fn attention_taps(&self) -> &super::attention_decode::taps::Taps {
        &self.attention.taps
    }

    /// Test-only preparation evidence; configurations may include strategies a replay does not use.
    #[cfg(test)]
    pub(crate) fn prepared_gemm_configs(&self) -> serde_json::Value {
        self.nvfp4.prepared_configs()
    }

    #[cfg(test)]
    pub(crate) const fn quant_reuse_stats(&self) -> (usize, usize) {
        self.nvfp4.quant_reuse_totals()
    }

    /// Actual retained FC owners; capture counters alone cannot prove route selection.
    #[cfg(test)]
    pub(crate) fn test_fusion_batch_scratch_bytes(&self) -> usize {
        [
            self.batch.as_ref(),
            self.prompt_batch.as_ref(),
            self.prompt_narrow.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(super::batch::BatchGraph::test_fusion_scratch_bytes)
        .sum()
    }

    /// Read just-replayed prompt hidden rows immediately after `prime_batch`, without replay.
    #[cfg(test)]
    pub(crate) fn test_retained_prompt_hidden(
        &self,
        position: usize,
        rows: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let batch = if self.prefers_narrow_prompt(rows) {
            self.prompt_narrow.as_ref()
        } else {
            self.prompt_batch.as_ref()
        }
        .ok_or_else(|| Error::invariant("test prompt graph"))?;
        batch.test_retained_prompt_hidden(&self.device, position, rows)
    }

    /// Whether a chunk of `lanes` tokens replays on the narrow prompt graph.
    ///
    /// Both captured prompt graphs are fixed width, so the narrow one is preferred whenever the
    /// chunk fits it: running a 51-token chunk through a 128-lane graph spends 60% of every
    /// replay on masked rows.
    /// Actual captured shapes, including the optional tail graph.
    #[must_use]
    pub fn prefill_geometry(&self) -> infer_ir::PrefillGeometry {
        infer_ir::PrefillGeometry {
            width: self.prefill_width(),
            narrow_width: self.prompt_narrow.as_ref().map_or(0, |prompt| prompt.width),
        }
    }

    fn prefers_narrow_prompt(&self, lanes: usize) -> bool {
        self.prompt_narrow
            .as_ref()
            .is_some_and(|prompt| lanes <= prompt.width)
    }

    /// State positions this program writes behind its `RoPE` positions.
    fn batch_state_lag(&self) -> Result<usize> {
        let offset = self
            .prompt_batch
            .as_ref()
            .or(self.prompt_narrow.as_ref())
            .or(self.batch.as_ref())
            .map_or(0, super::batch::BatchGraph::state_offset);
        usize::try_from(offset.unsigned_abs()).map_err(device_error)
    }

    /// Device hidden rows written by the prompt graph that runs a `lanes`-token chunk.
    pub(crate) fn prompt_hidden(&self, lanes: usize) -> Option<Arc<Tensor<f32>>> {
        let prompt = if self.prefers_narrow_prompt(lanes) {
            self.prompt_narrow.as_ref()
        } else {
            self.prompt_batch.as_ref()
        };
        prompt
            .or(self.batch.as_ref())
            .map(super::batch::BatchGraph::hidden_rows)
    }

    /// Prime a fused draft from a target chunk's device hidden rows, without a host round trip.
    ///
    /// Lane `i` embeds `tokens[i]` with the hidden of the position before it: the previous
    /// chunk's last row for lane zero (`previous` when the host holds it, the chunk's own first
    /// row at position zero where the state write is masked), then `rows[i - 1]`. The copies and
    /// replays are ordered on the shared device stream ahead of the target's next chunk.
    /// # Errors
    /// Rejects short row tensors or a missing prompt graph, then reports CUDA failures.
    pub(crate) async fn prime_from_target_async(
        &mut self,
        tokens: &[u32],
        position: usize,
        rows: &Tensor<f32>,
        previous: Option<&[f32]>,
    ) -> Result<()> {
        let hidden = self.external.size();
        if tokens.is_empty()
            || rows.size() < tokens.len() * hidden
            || previous.is_some_and(|row| row.len() != hidden)
        {
            return Err(Error::invalid("draft priming rows"));
        }
        if let Some(row) = previous {
            ExternalInput::Host(row).update(
                &self.device,
                &mut self.prime_previous,
                self.uploads.external.as_mut(),
            )?;
        } else if position == 0 {
            self.device
                .copy_d2d_at(&mut self.prime_previous, rows, 0, 0, hidden)?;
        }
        let width = self.prefill_width().max(PREFILL_LANES);
        let mut start = 0;
        while start < tokens.len() {
            let count = width.min(tokens.len() - start);
            if start == 0 {
                self.device.copy_d2d_at(
                    &mut self.lane_external_owner,
                    &self.prime_previous,
                    0,
                    0,
                    hidden,
                )?;
            } else {
                self.device.copy_d2d_at(
                    &mut self.lane_external_owner,
                    rows,
                    0,
                    (start - 1) * hidden,
                    hidden,
                )?;
            }
            if count > 1 {
                self.device.copy_d2d_at(
                    &mut self.lane_external_owner,
                    rows,
                    hidden,
                    start * hidden,
                    (count - 1) * hidden,
                )?;
            }
            self.prefill_batch_readout_async(
                &tokens[start..start + count],
                position + start,
                false,
                false,
            )
            .await?;
            start += count;
        }
        self.device.copy_d2d_at(
            &mut self.prime_previous,
            rows,
            0,
            (tokens.len() - 1) * hidden,
            hidden,
        )
    }

    /// Host copy of the hidden row the next device-primed chunk would start from.
    /// # Errors
    /// Reports CUDA readback failures.
    pub(crate) fn primed_hidden_async(
        &mut self,
    ) -> impl Future<Output = Result<Vec<f32>>> + Send + '_ {
        self.device
            .read_async(&self.prime_previous, &mut self.uploads.prime)
    }

    /// Ingest a prompt chunk with one external hidden per lane, as a fused draft needs.
    /// # Errors
    /// Rejects a short hidden slice or a missing prompt graph, then reports CUDA failures.
    #[cfg(test)]
    pub(crate) fn prime_batch(
        &mut self,
        tokens: &[u32],
        position: usize,
        externals: &[f32],
    ) -> Result<()> {
        block_on(self.prime_batch_async(tokens, position, externals))
    }

    pub(crate) async fn prime_batch_async(
        &mut self,
        tokens: &[u32],
        position: usize,
        externals: &[f32],
    ) -> Result<()> {
        let hidden = self.external.size();
        let lanes = tokens
            .len()
            .checked_mul(hidden)
            .ok_or_else(|| Error::invalid("prompt externals"))?;
        if externals.len() != lanes {
            return Err(Error::invalid("prompt externals"));
        }
        // The prompt graphs captured row views that share this owner's storage. One contiguous
        // transfer replaces 128 allocations, host-to-device copies, stream syncs and per-lane
        // graph rebinds per chunk; row order and draft state order are unchanged. `&mut self`
        // excludes concurrent uploads/replays, and the copy stays on the device stream ahead of
        // the prompt replay, so the shared owner obeys the same exclusivity contract as before.
        self.uploads
            .lanes
            .as_mut()
            .ok_or_else(|| Error::invariant("draft prompt pinned staging"))?
            .copy(&self.device, &mut self.lane_external_owner, externals)?;
        // Replay completion fences this pinned source before the next prompt upload.
        let narrow_fits = self.prefers_narrow_prompt(tokens.len());
        let batch = if narrow_fits {
            self.prompt_narrow.take()
        } else {
            self.prompt_batch.take()
        }
        .ok_or_else(|| Error::invariant("prompt batch graph"))?;
        if narrow_fits {
            self.prompt_narrow = Some(batch);
        } else {
            self.prompt_batch = Some(batch);
        }
        self.prefill_batch_readout_async(tokens, position, false, false)
            .await?;
        Ok(())
    }

    /// Ingest prompt tokens without rollback copies; 32-token graphs mask inactive tail lanes.
    /// # Errors
    /// Rejects invalid widths, positions/tokens or CUDA failures.
    pub fn prefill_batch(
        &mut self,
        tokens: &[u32],
        position: usize,
        read_logits: bool,
    ) -> Result<super::BatchOutput> {
        block_on(self.prefill_batch_async(tokens, position, read_logits))
    }

    pub(crate) async fn prefill_batch_async(
        &mut self,
        tokens: &[u32],
        position: usize,
        read_logits: bool,
    ) -> Result<super::BatchOutput> {
        self.prefill_batch_readout_async(tokens, position, read_logits, true)
            .await
    }

    #[cfg(test)]
    pub(crate) fn prefill_batch_readout(
        &mut self,
        tokens: &[u32],
        position: usize,
        read_logits: bool,
        read_hidden: bool,
    ) -> Result<super::BatchOutput> {
        block_on(self.prefill_batch_readout_async(tokens, position, read_logits, read_hidden))
    }

    pub(crate) async fn prefill_batch_readout_async(
        &mut self,
        tokens: &[u32],
        position: usize,
        read_logits: bool,
        read_hidden: bool,
    ) -> Result<super::BatchOutput> {
        let width = self.prefill_width();
        if (self.batch.is_none() && self.prompt_batch.is_none() && self.prompt_narrow.is_none())
            || tokens.is_empty()
            || tokens.len() > width
            || (width < PREFILL_LANES && tokens.len() != width)
            || position
                .checked_add(tokens.len())
                .is_none_or(|end| end > self.capacity)
            || tokens
                .iter()
                .any(|t| usize::try_from(*t).map_or(true, |n| n >= self.vocabulary))
        {
            return Err(Error::invalid("prefill tokens, width or capacity"));
        }
        let lag = self.batch_state_lag()?;
        if position == 0 {
            self.reset_async().await?;
        }
        if position.saturating_sub(lag) != self.next_position {
            return Err(Error::invalid("prefill state position"));
        }
        let lanes = tokens.len();
        let prompt = if self.prefers_narrow_prompt(lanes) {
            self.prompt_narrow.as_mut()
        } else {
            self.prompt_batch.as_mut()
        };
        let output = prompt
            .or(self.batch.as_mut())
            .ok_or_else(|| Error::invariant("prefill batch graph"))?
            .run_async(
                &self.device,
                tokens,
                position,
                Some(read_logits),
                read_hidden,
            )
            .await?;
        let end = position
            .checked_add(tokens.len())
            .ok_or_else(|| Error::invalid("prefill position overflow"))?;
        self.next_position = end.saturating_sub(lag);
        self.last_batch = None;
        Ok(output)
    }

    /// Ingest one prompt chunk whose media lanes carry precomputed embeddings.
    ///
    /// Row copies and the replay share the device stream, so the chunk runs once instead of one
    /// decode step per media token. The wide prompt graph is used deliberately: the narrow graph
    /// would split media chunks at its own fixed width.
    #[expect(
        clippy::too_many_arguments,
        reason = "The chunk names its media tensor, row map, width, rotary table and readout explicitly"
    )]
    pub(crate) fn prefill_media_chunk(
        &mut self,
        tokens: &[u32],
        position: usize,
        media: &Tensor<f32>,
        rows: &[(usize, usize)],
        hidden: usize,
        rotary: Option<&[[usize; ROTARY_AXES]]>,
        read_logits: bool,
    ) -> Result<super::BatchOutput> {
        block_on(self.prefill_media_chunk_async(
            tokens,
            position,
            media,
            rows,
            hidden,
            rotary,
            read_logits,
        ))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The existing media chunk API names its tensor, row map, rotary coordinates and readout explicitly"
    )]
    pub(crate) async fn prefill_media_chunk_async(
        &mut self,
        tokens: &[u32],
        position: usize,
        media: &Tensor<f32>,
        rows: &[(usize, usize)],
        hidden: usize,
        rotary: Option<&[[usize; ROTARY_AXES]]>,
        read_logits: bool,
    ) -> Result<super::BatchOutput> {
        let width = self.prefill_width();
        if (self.prompt_batch.is_none() && self.batch.is_none())
            || tokens.is_empty()
            || tokens.len() > width
            || position
                .checked_add(tokens.len())
                .is_none_or(|end| end > self.capacity)
            || tokens
                .iter()
                .any(|t| usize::try_from(*t).map_or(true, |n| n >= self.vocabulary))
        {
            return Err(Error::invalid("media prefill tokens, width or capacity"));
        }
        if rotary.is_some_and(|rotary| rotary.len() != tokens.len()) {
            return Err(Error::invalid("media prefill rotary width"));
        }
        let lag = self.batch_state_lag()?;
        if position == 0 {
            self.reset_async().await?;
        }
        if position.saturating_sub(lag) != self.next_position {
            return Err(Error::invalid("media prefill state position"));
        }
        let mut overrides = vec![false; tokens.len()];
        for &(lane, row) in rows {
            if lane >= tokens.len() || overrides[lane] {
                return Err(Error::invalid("media prefill lane"));
            }
            let offset = row
                .checked_mul(hidden)
                .ok_or_else(|| Error::invalid("media row offset overflow"))?;
            let slot = self
                .lane_external
                .get_mut(lane)
                .ok_or_else(|| Error::invalid("media prefill lane"))?;
            if slot.size() != hidden {
                return Err(Error::invalid("media external width"));
            }
            self.device.copy_d2d_at(slot, media, 0, offset, hidden)?;
            overrides[lane] = true;
        }
        let prompt = self
            .prompt_batch
            .as_mut()
            .ok_or_else(|| Error::invariant("media prefill requires a prompt graph"))?;
        let output = prompt
            .run_media_async(
                &self.device,
                tokens,
                position,
                read_logits,
                super::batch::PromptOverrides {
                    lanes: &overrides,
                    rotary,
                },
            )
            .await?;
        let end = position
            .checked_add(tokens.len())
            .ok_or_else(|| Error::invalid("media prefill position overflow"))?;
        self.next_position = end.saturating_sub(lag);
        self.last_batch = None;
        Ok(output)
    }

    /// Commit a nonempty prefix of the most recent verification batch.
    /// # Errors
    /// Rejects forward commits, stale checkpoints or failed state restoration.
    pub fn commit_batch(&mut self, count: usize) -> Result<()> {
        block_on(self.commit_batch_async(count))
    }

    pub(crate) async fn commit_batch_async(&mut self, count: usize) -> Result<()> {
        let (start, available) = self
            .last_batch
            .ok_or_else(|| Error::invalid("no pending verification batch"))?;
        if count == 0 || count > available {
            return Err(Error::invalid("invalid verification prefix"));
        }
        if self.next_position != start + count {
            self.batch
                .as_mut()
                .ok_or_else(|| Error::invariant("batch graph"))?
                .restore_async(&self.device, count)
                .await?;
        }
        self.next_position = start + count;
        self.last_batch = Some((start, count));
        Ok(())
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.next_position
    }

    /// Validate an externally executed attention prefix before copying its state.
    pub(super) fn external_attention_capacity(&self, position: usize) -> Result<usize> {
        if !self.attention_only || position > self.capacity {
            return Err(Error::invalid("external attention prefix is incompatible"));
        }
        Ok(self.capacity)
    }
    /// Called only after slot state has been copied into this program's buffers.
    pub(super) fn adopt_attention_prefix(&mut self, position: usize) -> Result<()> {
        self.external_attention_capacity(position)?;
        self.next_position = position;
        self.last_batch = None;
        Ok(())
    }

    /// Rewind an attention-only draft. Replay overwrites the suffix; attention masks stale rows.
    /// # Errors
    /// Rejects recurrent state and forward skips, which need a different rollback strategy.
    pub fn rewind_attention(&mut self, position: usize) -> Result<()> {
        if !self.attention_only || position > self.next_position {
            return Err(Error::invalid("invalid attention-only rollback"));
        }
        self.next_position = position;
        Ok(())
    }

    /// # Errors
    /// Returns failed CUDA state resets; graph addresses remain unchanged.
    pub fn reset(&mut self) -> Result<()> {
        block_on(self.reset_async())
    }
    pub(crate) async fn reset_async(&mut self) -> Result<()> {
        self.device.run_op(self.reset_graph.launch()).await?;
        self.next_position = 0;
        self.last_batch = None;
        Ok(())
    }
}

fn capture_reset(
    device: &CudaDevice,
    states: &mut BTreeMap<TensorId, Vec<Tensor<f32>>>,
) -> Result<CudaGraph<()>> {
    CudaGraph::scope(&device.stream, |scope| {
        for tensors in states.values_mut() {
            for tensor in tensors {
                scope.record(super::reset::Clear(tensor))?;
            }
        }
        Ok(())
    })
    .map_err(device_error)
}

/// Allocation shapes shared by private and pooled state storage.
pub(super) fn state_shapes(
    graph: &DataflowGraph,
    capacity: usize,
    weights: &ProgramWeights,
) -> Result<Vec<(TensorId, Vec<Vec<usize>>)>> {
    let mut plan = Vec::new();
    for spec in &graph.tensors {
        let TensorStorage::State { kind, .. } = &spec.storage else {
            continue;
        };
        if weights.kv_scales.contains_key(&spec.id) {
            continue;
        }
        let shapes = match kind {
            StateKind::Conv if spec.shape.len() == 2 && spec.shape[1] == CONV_KERNEL_SIZE => {
                vec![vec![spec.shape[0]]; CONV_STATE_TENSORS]
            }
            StateKind::LinearAttention => vec![spec.shape.clone()],
            StateKind::AttentionKv if spec.shape.len() == 2 => {
                vec![
                    vec![
                        capacity
                            .checked_mul(spec.shape[1])
                            .ok_or_else(|| Error::invalid("KV allocation overflow"))?
                    ];
                    2
                ]
            }
            _ => return Err(Error::unsupported("resident state recipe")),
        };
        plan.push((spec.id, shapes));
    }
    Ok(plan)
}

pub fn allocate_states(
    device: &CudaDevice,
    graph: &DataflowGraph,
    capacity: usize,
    weights: &ProgramWeights,
) -> Result<BTreeMap<TensorId, Vec<Tensor<f32>>>> {
    let plan = state_shapes(graph, capacity, weights)?;
    let mut total = 0usize;
    let budget = device
        .allocatable_memory_bytes()?
        .saturating_sub(device.profile()?.device_headroom_bytes());
    for (_, shapes) in &plan {
        for shape in shapes {
            let elements = shape.iter().try_fold(1usize, |a, b| {
                a.checked_mul(*b)
                    .ok_or_else(|| Error::invalid("state overflow"))
            })?;
            total = total
                .checked_add(elements)
                .ok_or_else(|| Error::invalid("state budget overflow"))?;
            if total as u64 > budget / F32_BYTES as u64 {
                return Err(Error::new(
                    infer_core::ErrorCode::Capacity,
                    format!(
                        "resident F32 state exceeds available device memory minus 1 GiB headroom: \
                         need {} MiB, budget {} MiB",
                        total as u64 * F32_BYTES as u64 / MIB as u64,
                        budget / MIB as u64
                    ),
                ));
            }
        }
    }
    let mut states = BTreeMap::new();
    for (id, shapes) in plan {
        let tensors = shapes
            .iter()
            .map(|shape| {
                api::zeros::<f32>(shape)
                    .sync_on(&device.stream)
                    .map_err(device_error)
            })
            .collect::<Result<Vec<_>>>()?;
        states.insert(id, tensors);
    }
    Ok(states)
}

fn attention_only(graph: &DataflowGraph) -> bool {
    graph.tensors.iter().all(|spec| {
        !matches!(
            spec.storage,
            TensorStorage::State {
                kind: StateKind::Conv | StateKind::LinearAttention,
                ..
            }
        )
    })
}

/// Exercise every delegated projection shape before a graph records it.
///
/// cuBLAS chooses workspace and algorithm the first time it sees a configuration, and a capturing
/// stream rejects that host-side work, so the warmup has to happen here.
fn warm_delegated(delegated: &mut Support, weights: &ProgramWeights, geometry: ProgramGeometry) {
    if !delegated.is_enabled() {
        return;
    }
    // The fused draft projection is bound outside `projections` but prompt chunks record it too.
    let fusion = weights.fusion.as_ref().map(|fusion| &fusion.projection);
    for weight in weights.projections.values().chain(fusion) {
        let ProjectionWeight::Dense(dense) = weight else {
            continue;
        };
        let [rows, columns] = dense.shape()[..] else {
            continue;
        };
        let rows = usize::try_from(rows).unwrap_or(0);
        let columns = usize::try_from(columns).unwrap_or(0);
        for width in geometry.widths() {
            let _ = delegated.warm(width, dense, rows, columns);
        }
    }
}

fn external_staging(
    device: &CudaDevice,
    hidden: usize,
    fusion: Option<&FusionWeights>,
) -> Result<Option<PinnedUpload<f32>>> {
    if fusion.is_none() {
        return Ok(None);
    }
    PinnedUpload::single(device, hidden).map(Some)
}

fn initial_state(
    device: &CudaDevice,
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    capacity: usize,
    slot: Option<SlotState>,
) -> Result<InitialState> {
    match slot {
        Some((states, fp8, table)) => Ok((states, fp8, Some(table))),
        None => Ok((
            allocate_states(device, graph, capacity, weights)?,
            super::fp8_cache::allocate(device, graph, capacity, &weights.kv_scales)?,
            None,
        )),
    }
}

struct ProgramUploads {
    external: Option<PinnedUpload<f32>>,
    lanes: Option<PinnedUpload<f32>>,
    prime: Readbacks,
}
impl ProgramUploads {
    fn new(
        device: &CudaDevice,
        hidden: usize,
        lanes: usize,
        fusion: Option<&FusionWeights>,
    ) -> Result<Self> {
        Ok(Self {
            external: external_staging(device, hidden, fusion)?,
            lanes: external_staging(device, lanes, fusion)?,
            prime: Readbacks::allocated(device, hidden)?,
        })
    }
}
fn attention_states(graph: &DataflowGraph) -> BTreeSet<TensorId> {
    graph
        .tensors
        .iter()
        .filter_map(|spec| {
            matches!(
                spec.storage,
                TensorStorage::State {
                    kind: StateKind::AttentionKv,
                    ..
                }
            )
            .then_some(spec.id)
        })
        .collect()
}
