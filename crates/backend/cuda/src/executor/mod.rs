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
use crate::constants::CB_DECODE_SLOTS;
use crate::constants::CB_SLOT_TOKENS;
use crate::device::CudaDevice;
use crate::registry::CudaKernels;
use crate::tuning::driver_build;
use crate::tuning::resident_gemm::measured_compute;
use execution::paged_tokens;
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
    // The current pooled serving graph validates a batch width of 2..=CB_DECODE_SLOTS.
    // Do not recommend slots the backend cannot actually execute.
    if value < CB_DECODE_SLOTS {
        value
    } else {
        CB_DECODE_SLOTS
    }
}
/// Highest width whose exact funded quote fits. The closure is the actual pooled
/// `funded_pooled_serving` ledger, so nonlinear table/prompt/control padding is validated at
/// every legal width rather than extrapolated from widths 1 and 2.
pub(super) fn funded_width(
    maximum: usize,
    mut quote: impl FnMut(usize) -> Result<Option<u64>>,
) -> Result<usize> {
    for width in (1..=maximum).rev() {
        if quote(width)?.is_some() {
            return Ok(width);
        }
    }
    Ok(0)
}

/// Formula `N_knee = (compute / bandwidth) * (weight bytes / parameters) / 2 / tokens`
/// per sequence per step. All inputs are checked; zero/missing inputs return `None`.
pub(super) fn knee_concurrency(
    compute_flops: u64,
    bandwidth_bytes_per_second: u64,
    weight_bytes: u64,
    parameters: u64,
    tokens_per_step: u64,
) -> Result<Option<usize>> {
    if compute_flops == 0
        || bandwidth_bytes_per_second == 0
        || weight_bytes == 0
        || parameters == 0
        || tokens_per_step == 0
    {
        return Ok(None);
    }
    let numerator = u128::from(compute_flops)
        .checked_mul(u128::from(weight_bytes))
        .ok_or_else(|| Error::invalid("knee numerator overflow"))?;
    let denominator = u128::from(bandwidth_bytes_per_second)
        .checked_mul(u128::from(parameters))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(u128::from(tokens_per_step)))
        .ok_or_else(|| Error::invalid("knee denominator overflow"))?;
    let value = numerator / denominator;
    usize::try_from(value)
        .map(Some)
        .map_err(|_| Error::invalid("knee concurrency exceeds usize"))
}

pub(super) fn recommendation(
    loaded: &LoadedModel,
    budget: u64,
    maximum_states: usize,
) -> Result<infer_spi::SchedulingRecommendation> {
    let prompt_width = loaded.pooled_prompt_width().max(1);
    let capacity = CB_SLOT_TOKENS.min(loaded.model().max_sequence);
    let envelope = budget
        .checked_add(loaded.shared_scratch_bytes()?)
        .ok_or_else(|| Error::invalid("pooled recommendation envelope overflow"))?;
    let legal_cap = CB_DECODE_SLOTS.min(infer_gpu_api::MAX_SUBMISSION_BATCH);
    let maximum = maximum_states.min(legal_cap);
    let memory_bound = funded_width(maximum, |width| {
        loaded
            .funded_pooled_serving(
                width,
                capacity,
                paged_tokens(width, capacity),
                prompt_width,
                envelope,
            )
            .map(|quote| quote.map(|quote| quote.total_bytes))
    })?;
    let minimum_kv_bytes = loaded.kv_block_bytes()?;
    let profile = loaded.device().profile()?;
    let metrics = loaded.weight_metrics()?;
    let driver = CudaDevice::driver_version()?;
    let driver_build = driver_build();
    let compute = measured_compute(profile, driver, driver_build.as_deref())?;
    let tokens_per_step = u64::try_from(loaded.mtp_depth() + 1)
        .map_err(|_| Error::invalid("MTP token count overflow"))?;
    let knee_bound = compute
        .as_ref()
        .map(|compute| {
            knee_concurrency(
                compute.flops_per_second,
                profile.memory_bandwidth_bytes_per_second(),
                metrics.bytes,
                metrics.parameters,
                tokens_per_step,
            )
        })
        .transpose()?
        .flatten();
    let max_sequences = bound_concurrency(memory_bound, knee_bound, maximum_states);
    tracing::info!(
        target: "infer::load",
        state_budget_bytes = budget,
        pool_capacity = capacity,
        pool_prompt_width = prompt_width,
        pooled_envelope_bytes = envelope,
        minimum_kv_bytes,
        memory_bound,
        compute_flops = ?compute.as_ref().map(|value| value.flops_per_second),
        compute_mode = ?compute.as_ref().map(|value| value.mode.as_str()),
        compute_kernel = ?compute.as_ref().map(|value| value.kernel_fingerprint.as_str()),
        compute_shape = ?compute.as_ref().map(|value| (value.rows, value.columns, value.reduction)),
        compute_ms = ?compute.as_ref().map(|value| value.milliseconds),
        compute_device = profile.identity(),
        driver_api = driver,
        driver_build = ?driver_build,
        dram_bytes_per_second = profile.memory_bandwidth_bytes_per_second(),
        weight_bytes = metrics.bytes,
        parameters = metrics.parameters,
        tokens_per_step,
        knee_bound = ?knee_bound,
        legal_cap,
        maximum_states = maximum_states,
        max_sequences,
        "roofline concurrency inputs"
    );
    Ok(infer_spi::SchedulingRecommendation {
        max_sequences,
        memory_bound,
        knee_bound,
    })
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
        recommendation(&self.loaded, self.budget, self.maximum_states)
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
