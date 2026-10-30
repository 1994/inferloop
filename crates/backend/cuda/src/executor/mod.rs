//! Engine adapter for shared weights and isolated resident CUDA graphs.
//!
//! An owned execution future separates submission from completion across every compute path.
//! Tickets publish committed state only after the complete device flight has fenced.
mod admission_pool;
mod capabilities;
mod drafting;
mod execution;
mod owner;
mod submission;
pub use owner::CudaBackend;
mod pool;
pub mod prefix;
mod prefix_state;
mod profiling;
mod provider;
use crate::constants::KV_BLOCK_TOKENS;
use crate::registry::CudaKernels;
pub use pool::PoolInspection;
mod state;
mod validation;
use crate::resident::dflash2::budget::fixed_bytes;
use crate::resident::dflash2::{Cursor, Engine};
use crate::{
    loading::{LoadedModel, PooledServingQuote},
    resident::DeviceProgram,
    resident::slot_batch::SlotPool,
};
use cutile::prelude::Tensor;
use infer_core::{Error, Result, StateId};
use infer_ir::{DeviceCapabilities, ExecutionProgram, ModelOutput, OutputReadout};
use infer_models::MediaLayout;
use infer_state::cache::SnapshotScan;
use infer_workloads::SamplingWorkspace;
use std::collections::BTreeMap;
use std::mem::{forget, take};
use std::sync::Arc;

pub(super) struct CudaExecutor {
    loaded: LoadedModel,
    capabilities: DeviceCapabilities,
    operations: Vec<infer_ir::CompiledOp>,
    workspace_bytes: u64,
    identity: String,
    states: BTreeMap<StateId, Sequence>,
    budget: u64,
    maximum_states: usize,
    roofline_bound: Option<usize>,
    memory_bound: usize,
    /// Pure funded quote before the first request, or the last complete capture quote.
    pool_quote: Option<PooledServingQuote>,
    pool_envelope: u64,
    pool: pool::StatePool,
    prefix: prefix::PrefixCache,
    /// KV bytes one 32-token block of the target graph occupies; the physical page price the
    /// engine's block accounting is reported in. Load-time constant of the graph and scales.
    kv_block_bytes: u64,
    /// Wide prompt programs slot-resident sequences prompt through; see [`Worker`].
    worker: Option<Worker>,
    /// Per slot, the target and draft programs that prefill straight into that slot while no
    /// sequence holds them. Declared before the pools so they drop first: their states are
    /// views of the slots' storage.
    homes: Vec<Option<Home>>,
    /// `DFlash2` draft engine: sequence rings and, once a pool exists, its pooled proposal.
    dflash: Option<Engine>,
    /// Continuous-batching decode slots, captured lazily on the first batchable step;
    /// `None` with `slots_disabled` means initialization failed and decode stays serial.
    slots: Option<SlotPool>,
    draft_slots: Option<SlotPool>,
    /// A rejected concurrent reservation may bootstrap a pool when that same
    /// intent retries after the solo private request drains. A different intent
    /// clears the hint, so cancellation cannot force an unrelated pool capture.
    pool_retry: Option<(StateId, usize)>,
    slots_disabled: bool,
    deferred_pool_plan: Option<(usize, u64)>,
    /// Reason the last pool attempt failed, reported when batching is given up for good.
    pool_failure_reason: Option<String>,
    busy: bool,
    fatal: Option<Error>,
}
/// Wide-prompt target and draft programs shared by every slot-resident sequence. A slot's own
/// programs capture only the narrow prompt graph (a wide one per slot does not fit), so a long
/// prompt chunk moves the sequence's state into the worker, replays there, and moves it back:
/// a few hundred megabytes of copies against a chunk that runs for hundreds of milliseconds.
struct Worker {
    program: DeviceProgram,
    draft: Option<DeviceProgram>,
}
/// A pool slot's resident programs: prompts prefill into the slot's own storage, so a sequence
/// that holds them never owns a private copy of its state.
struct Home {
    program: DeviceProgram,
    draft: Option<DeviceProgram>,
}
struct Sequence {
    /// The slot whose resident programs this sequence holds; it keeps that slot leased.
    home: Option<usize>,
    program: DeviceProgram,
    speculation: Option<Speculation>,
    capacity: usize,
    readout: OutputReadout,
    history: Vec<u32>,
    hidden: Vec<Vec<f32>>,
    media: Option<CachedMedia>,
    /// Rotary coordinate of the first decoded token and the prompt length it starts after.
    rotary_base: usize,
    media_prompt_len: usize,
    budget: u64,
    /// Request-owned vision working set, released after the device drains.
    media_budget: u64,
    poisoned: bool,
    slot: Option<SlotLease>,
    /// Whether `budget` included private verification storage; a slot-lease credit and a
    /// reset restore only apply to bytes admission actually charged.
    verification_priced: bool,
    /// `DFlash2` timeline of this sequence, when it speculates with the draft engine.
    draft: Option<DraftState>,
}
/// A sequence's `DFlash2` ring and the sampling state its acceptance needs.
pub(super) struct DraftState {
    cursor: Cursor,
    /// Index where generated tokens start, captured on the first decode step.
    prompt_len: Option<usize>,
    /// The ring missed part of the timeline (a serial step or a restored prefix): the sequence
    /// decodes without proposals rather than drafting from a gap.
    stale: bool,
    scratch: SamplingWorkspace,
}
impl DraftState {
    fn new(cursor: Cursor) -> Self {
        Self {
            cursor,
            prompt_len: None,
            stale: false,
            scratch: SamplingWorkspace::default(),
        }
    }
    /// Start a new timeline in the same ring.
    const fn reset(&mut self) {
        self.cursor.context = 0;
        self.prompt_len = None;
        self.stale = false;
    }
}
/// Device-resident encoded media for one prompt, reused across every prefill chunk.
pub(super) struct CachedMedia {
    pub tensor: Arc<Tensor<f32>>,
    pub layout: MediaLayout,
    pub bases: Vec<usize>,
}
/// A sequence's lease on one continuous-batching decode slot. `batched` is the sticky
/// mode: a leased sequence always decodes through the shared slot graph and never
/// returns to its private target graph; the pool may also verify draft proposals.
pub(super) struct SlotLease {
    slot: usize,
    position: usize,
    batched: bool,
}
/// Mutable draft state for greedy MTP speculation on one resident sequence.
pub(super) struct Speculation {
    program: DeviceProgram,
    depth: usize,
    /// Index where generated tokens start, captured on the first decode step.
    prompt_len: Option<usize>,
    /// Target hidden of the most recently consumed token; the draft fuses the token before it.
    last_hidden: Vec<f32>,
    scratch: SamplingWorkspace,
}
impl Speculation {
    pub(super) fn new(program: DeviceProgram, depth: usize) -> Self {
        Self {
            program,
            depth,
            prompt_len: None,
            last_hidden: Vec::new(),
            scratch: SamplingWorkspace::default(),
        }
    }
    /// # Errors
    /// Rejects failed CUDA state resets.
    pub(super) fn reset(&mut self) -> Result<()> {
        self.program.reset()?;
        self.prompt_len = None;
        self.last_hidden.clear();
        Ok(())
    }
}
pub use submission::CudaTicket;
/// Prefix-cache capacity derived from the state budget: default on, a quarter of the budget,
/// with a floor so small devices still cache one prefix and a disabled cache when nothing fits.
const fn prefix_cache_bytes(state_budget: u64) -> u64 {
    const DIVISOR: u64 = 4;
    const FLOOR: u64 = 128 * 1024 * 1024;
    let share = state_budget / DIVISOR;
    if share < FLOOR { 0 } else { share }
}

impl CudaExecutor {
    /// Prefix cache occupancy and hit counters, for observability.
    #[must_use]
    pub fn prefix_inspection(&self) -> (u64, u64, u64, usize) {
        (
            self.prefix.hits,
            self.prefix.lookups,
            self.prefix.bytes(),
            self.prefix.len(),
        )
    }

    /// Effective load-time choices, so a benchmark report can prove which graph widths and
    /// speculation depth actually ran instead of restating the request.
    #[must_use]
    pub fn execution_profile(&self) -> infer_ir::ExecutionProfileInspection {
        let mut profile = self.loaded.execution_profile();
        profile.concurrency = Some(infer_ir::ConcurrencyInspection {
            memory_bound: self.memory_bound,
            roofline_bound: self.roofline_bound,
            selected: self.slots.as_ref().map_or_else(
                || self.pool_quote.as_ref().map_or(1, |quote| quote.width),
                SlotPool::width,
            ),
        });
        profile.pooled_prefill_width = self
            .worker
            .as_ref()
            .map(|worker| worker.program.prefill_width());
        profile
    }

    /// # Errors
    /// Rejects an empty state limit/budget or unavailable device attributes.
    #[cfg(test)]
    pub fn new(loaded: LoadedModel, state_budget: u64, maximum_states: usize) -> Result<Self> {
        let mut backend = Self::with_roofline(loaded, state_budget, maximum_states, None)?;
        backend.pool_quote = None;
        backend.warm_slot_pool();
        Ok(backend)
    }
    pub(super) fn new_bounded(
        loaded: LoadedModel,
        state_budget: u64,
        maximum_states: usize,
    ) -> Result<Self> {
        Self::with_roofline(loaded, state_budget, maximum_states, None)
    }
    pub fn new_automatic(
        loaded: LoadedModel,
        state_budget: u64,
        maximum_states: usize,
    ) -> Result<Self> {
        let roofline = loaded.concurrency_roofline()?;
        Self::with_roofline(loaded, state_budget, maximum_states, roofline)
    }
    fn with_roofline(
        loaded: LoadedModel,
        state_budget: u64,
        maximum_states: usize,
        roofline_bound: Option<usize>,
    ) -> Result<Self> {
        if state_budget == 0 || maximum_states == 0 {
            return Err(Error::invalid("CUDA state limits must be nonzero"));
        }
        let mut capabilities = capabilities::query(loaded.device())?;
        // The model provider declares what the device must support; fail at load, not per step.
        capabilities.require(loaded.requirements())?;
        // Greedy-only speculation, and only when a draft was actually loaded.
        capabilities.speculation = infer_ir::SpeculationCapability {
            draft_depth: loaded.draft_depth(),
            greedy_only: true,
        };
        // Rings are sequence state the engine owns outright: fund them before admission.
        let (dflash, state_budget) = match loaded.dflash2() {
            Some(weights) => {
                let geometry = weights.geometry;
                let rings = maximum_states;
                let bytes = fixed_bytes(&geometry, weights.layers.len(), rings)?;
                if bytes >= state_budget {
                    return Err(Error::new(
                        infer_core::ErrorCode::Capacity,
                        "DFlash2 engine exceeds the state budget",
                    ));
                }
                let engine = Engine::new(loaded.device(), Arc::clone(weights), rings)?;
                tracing::info!(target: "infer::executor", rings, engine_bytes = bytes,
                    "DFlash2 draft engine ready");
                (Some(engine), state_budget - bytes)
            }
            None => (None, state_budget),
        };
        let mut registry = infer_kernel_api::KernelRegistry::default();
        registry.register(&CudaKernels)?;
        let program = infer_compiler::compile(
            infer_core::ProgramId::ONE,
            infer_compiler::lower(
                loaded.model(),
                loaded.graph().clone(),
                infer_ir::PrecisionPlan::f32(),
            )?,
            &registry,
            &capabilities,
            state_budget,
        )?;
        let identity = format!("cuda-resident/{}", loaded.device().name()?);
        let kv_block_bytes = loaded.kv_block_bytes()?;
        // A cache belongs to this immutable loaded backend; snapshots never cross instances.
        let prefix = prefix::PrefixCache::with_policy(
            identity.as_bytes(),
            // Whole KV blocks: requested prefix boundaries publish between the regular cadence.
            KV_BLOCK_TOKENS,
            prefix_cache_bytes(state_budget),
            usize::MAX,
            SnapshotScan,
        )?;
        let mut backend = Self {
            loaded,
            capabilities,
            operations: program.operations,
            workspace_bytes: program.workspace_bytes,
            identity,
            states: BTreeMap::new(),
            budget: state_budget,
            maximum_states,
            roofline_bound,
            memory_bound: 1,
            pool_quote: None,
            pool_envelope: 0,
            pool: pool::StatePool::default(),
            prefix,
            kv_block_bytes,
            worker: None,
            homes: Vec::new(),
            dflash,
            slots: None,
            draft_slots: None,
            pool_retry: None,
            slots_disabled: false,
            deferred_pool_plan: None,
            pool_failure_reason: None,
            busy: false,
            fatal: None,
        };
        backend.defer_request_pool()?;
        Ok(backend)
    }
    #[must_use]
    pub const fn model(&self) -> &infer_ir::ModelIr {
        self.loaded.model()
    }

    /// Compile registrations for the actual loaded graph, including package-bound weights.
    /// # Errors
    /// Rejects invalid lowering, unavailable kernels or insufficient workspace.
    pub fn compile(&self, id: infer_core::ProgramId) -> Result<ExecutionProgram> {
        let mut registry = infer_kernel_api::KernelRegistry::default();
        registry.register(&CudaKernels)?;
        let ir = infer_compiler::lower(
            self.model(),
            self.loaded.graph().clone(),
            infer_ir::PrecisionPlan::f32(),
        )?;
        infer_compiler::compile(id, ir, &registry, &self.capabilities, self.budget)
    }
    fn idle(&self) -> Result<()> {
        if let Some(error) = &self.fatal {
            return Err(error.clone());
        }
        if self.busy {
            return Err(Error::new(
                infer_core::ErrorCode::Conflict,
                "CUDA completion must be consumed first",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/executor_mod.rs"]
mod prefix_capacity_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_mtp_fusion_model.rs"]
mod mtp_fusion_model_tests;

impl Drop for CudaExecutor {
    #[expect(
        clippy::mem_forget,
        reason = "Unknown CUDA completion requires retaining physical pool owners: graph row aliases do not own their allocations"
    )]
    fn drop(&mut self) {
        if self.busy
            && let Err(error) = self.loaded.device().drain()
        {
            tracing::error!(target: "infer::cuda", %error, "retaining CUDA flight storage after failed drop drain");
            // Graphs can retain non-owning row aliases. Keep their physical pools,
            // homes and sequence programs together if the driver cannot fence them.
            forget((
                self.slots.take(),
                self.draft_slots.take(),
                self.worker.take(),
                take(&mut self.homes),
                take(&mut self.states),
                self.dflash.take(),
            ));
        }
    }
}
