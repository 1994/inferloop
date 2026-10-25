//! Engine adapter for shared weights and isolated resident CUDA graphs.
//!
//! An owned execution future separates submission from completion across every compute path.
//! Tickets publish committed state only after the complete device flight has fenced.
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
use crate::{loading::LoadedModel, resident::DeviceProgram, resident::slot_batch::SlotPool};
use infer_core::{Error, Result, StateId};
use infer_ir::{DeviceCapabilities, ExecutionProgram, ModelOutput, OutputReadout};

/// Automatic concurrency is the smallest of the memory, measured knee and explicit state bounds.
pub(super) fn bound_concurrency(
    memory_bound: usize,
    knee_bound: Option<usize>,
    maximum_states: usize,
) -> usize {
    let knee = knee_bound.unwrap_or(usize::MAX);
    let value = if memory_bound < knee {
        memory_bound
    } else {
        knee
    };
    let value = if value < maximum_states {
        value
    } else {
        maximum_states
    };
    let value = if value < infer_gpu_api::MAX_SUBMISSION_BATCH {
        value
    } else {
        infer_gpu_api::MAX_SUBMISSION_BATCH
    };
    if value == 0 { 1 } else { value }
}
use infer_state::cache::SnapshotScan;
use std::collections::BTreeMap;
use std::mem::{forget, take};

pub(super) struct CudaExecutor {
    loaded: LoadedModel,
    capabilities: DeviceCapabilities,
    operations: Vec<infer_ir::CompiledOp>,
    workspace_bytes: u64,
    identity: String,
    states: BTreeMap<StateId, Sequence>,
    budget: u64,
    maximum_states: usize,
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
    /// Continuous-batching decode slots, captured lazily on the first batchable step;
    /// `None` with `slots_disabled` means initialization failed and decode stays serial.
    slots: Option<SlotPool>,
    draft_slots: Option<SlotPool>,
    /// Geometry of an idle pool evicted for a long request. Restore it before
    /// pricing the next compatible request, so short batches keep pooled admission.
    reclaimed_slot_capacity: Option<usize>,
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
    budget: u64,
    poisoned: bool,
    slot: Option<SlotLease>,
    /// Whether `budget` included private verification storage; a slot-lease credit and a
    /// reset restore only apply to bytes admission actually charged.
    verification_priced: bool,
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
    scratch: infer_workloads::SamplingWorkspace,
}
impl Speculation {
    pub(super) fn new(program: DeviceProgram, depth: usize) -> Self {
        Self {
            program,
            depth,
            prompt_len: None,
            last_hidden: Vec::new(),
            scratch: infer_workloads::SamplingWorkspace::default(),
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
        profile.pooled_prefill_width = self
            .worker
            .as_ref()
            .map(|worker| worker.program.prefill_width());
        profile
    }

    /// # Errors
    /// Rejects an empty state limit/budget or unavailable device attributes.
    pub fn new(loaded: LoadedModel, state_budget: u64, maximum_states: usize) -> Result<Self> {
        if state_budget == 0 || maximum_states == 0 {
            return Err(Error::invalid("CUDA state limits must be nonzero"));
        }
        let mut capabilities = capabilities::query(loaded.device())?;
        // The model provider declares what the device must support; fail at load, not per step.
        capabilities.require(loaded.requirements())?;
        // Greedy-only speculation, and only when a draft was actually loaded.
        capabilities.speculation = infer_ir::SpeculationCapability {
            draft_depth: loaded.mtp_depth(),
            greedy_only: true,
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
            prefix::PREFIX_GRANULARITY,
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
            pool: pool::StatePool::default(),
            prefix,
            kv_block_bytes,
            worker: None,
            homes: Vec::new(),
            slots: None,
            draft_slots: None,
            reclaimed_slot_capacity: None,
            slots_disabled: false,
            deferred_pool_plan: None,
            pool_failure_reason: None,
            busy: false,
            fatal: None,
        };
        backend.warm_slot_pool();
        Ok(backend)
    }
    /// Price the loaded model's minimum per-sequence state and bound concurrency by the
    /// actual state budget and the configured decode step width.
    pub(super) fn scheduling_recommendation(&self) -> Result<infer_spi::SchedulingRecommendation> {
        let per_sequence =
            self.loaded
                .sequence_budget(KV_BLOCK_TOKENS, OutputReadout::Logits, true)?;
        let memory_bound = usize::try_from(self.budget / per_sequence.max(1))
            .unwrap_or(usize::MAX)
            .max(1);
        let knee_bound = None;
        let max_sequences = bound_concurrency(memory_bound, knee_bound, self.maximum_states);
        Ok(infer_spi::SchedulingRecommendation {
            max_sequences,
            memory_bound,
            knee_bound,
        })
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
            ));
        }
    }
}
