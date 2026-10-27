//! Construction responsibilities.
use super::{Engine, RuntimeConfig};
use crate::cpu::storage::HostStorage;
use crate::observation::Observations;
use crate::observation::collector::Collector;
use crate::pipeline::completion::cpu::OutputFlights;
use crate::resource::ResourceWaiters;
use crate::runner::RunnerConfig;
use crate::runner::ThreadedBackend;
use crate::stages::worker::OutputWorker;
use infer_core::map::BoundedMap;
use infer_core::{Error, ErrorCode, IdAllocator, ProgramId, Result, event::event_ring};
use infer_ir::{ModelIr, PrecisionPlan};
use infer_kernel_api::KernelRegistry;
use infer_quality::ProgressGuard;
use infer_scheduler::{CalibratedCosts, CostAwarePolicy, ResourceAdmission};
use infer_spi::{BackendProvider, SchedulingPolicy, WorkloadProvider};
use infer_state::SequenceStateManager;
use infer_workloads::NativeWorkloads;
use std::mem::replace;
use std::time::Duration;
use std::{collections::BTreeSet, collections::VecDeque};

/// Upper bound on parent trace contexts the event collector retains.
const MAX_TRACKED_PARENTS: usize = 8192;

impl<B: BackendProvider> Engine<B, CostAwarePolicy> {
    /// # Errors
    /// Returns an invalid-input error for invalid configuration, or a capacity error if the requested resources cannot be reserved.
    pub fn new(
        backend: B,
        model: ModelIr,
        precision: PrecisionPlan,
        registry: &KernelRegistry,
        config: RuntimeConfig,
    ) -> Result<Self> {
        Self::with_policy(backend, model, precision, registry, config, CostAwarePolicy)
    }
}

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    /// # Errors
    /// Returns a validation, unsupported, or capacity error if configuration, model compilation, or backend initialization fails.
    pub fn with_policy(
        backend: B,
        model: ModelIr,
        precision: PrecisionPlan,
        registry: &KernelRegistry,
        config: RuntimeConfig,
        policy: P,
    ) -> Result<Self> {
        let placement = config.cpu.placement.scheduler.clone();
        placement.scope(|| Self::initialize(backend, model, precision, registry, config, policy))
    }
    fn initialize(
        backend: B,
        model: ModelIr,
        precision: PrecisionPlan,
        registry: &KernelRegistry,
        config: RuntimeConfig,
        policy: P,
    ) -> Result<Self> {
        let mut config = config;
        config.resolve_concurrency(backend.execution_profile().as_ref())?;
        config.validate()?;
        // Resolve the deployment length contract before any resource is planned, so every entry
        // point, the workload plan and capacity planning share one set of numbers.
        let limits = crate::ResolvedLengthLimits::resolve(model.max_sequence, &config)?;
        let program = infer_compiler::compile(
            ProgramId::new(1)?,
            infer_compiler::lower(&model, backend.execution_graph(&model)?, precision)?,
            registry,
            &backend.capabilities(),
            config.workspace_bytes,
        )?;
        backend.validate_program(&model, &program)?;
        let state = SequenceStateManager::with_capacity(
            config.num_gpu_blocks,
            config.block_size,
            config.max_requests,
        )?;
        let guard = ProgressGuard::new(config.progress_limit)?;
        let (events, event_reader) = event_ring(config.event_capacity);
        let costs = CalibratedCosts::new(
            format!("{}:{}", backend.identity(), program.id),
            config.cost_model.clone(),
        )?;
        let admission_policy = ResourceAdmission::new(config.admission.clone())?;
        let observations = Observations::new(config.history_capacity)?;
        let host = HostStorage::new(&config, program.id, program.backend, model.vocab_size)?;
        let mut planning_workspace = P::Workspace::default();
        policy.reserve_workspace(
            &mut planning_workspace,
            config.candidate_limit.min(config.max_requests),
            config.max_num_seqs,
        )?;
        let tenants = BoundedMap::new(config.max_requests)?;
        let resources_pending = ResourceWaiters::new(config.max_requests)?;
        let output_owners = BoundedMap::new(config.max_requests)?;
        let history_capacity = config.history_capacity;
        Ok(Self {
            backend,
            policy,
            planning_workspace,
            host,
            output_worker: None,
            output_pending: OutputFlights::default(),
            output_owners,
            workloads: Box::new(NativeWorkloads),
            costs: Box::new(costs),
            admission_policy: Box::new(admission_policy),
            pending_cost_observations: VecDeque::with_capacity(config.history_capacity),
            replaying: false,
            preemption_focus: None,
            model,
            program,
            config,
            limits,
            state,
            resource_epoch: 0,
            backend_resource_epoch: 0,
            resources_pending,
            seen_requests: BTreeSet::new(),
            retired_request_floor: 0,
            tenants,
            ids: IdAllocator::default(),
            inflight: None,
            global_progress_epoch: 0,
            now_us: 0,
            guard,
            events,
            event_reader: Some(event_reader),
            decisions: VecDeque::with_capacity(history_capacity),
            actions: VecDeque::with_capacity(history_capacity),
            dropped_actions: 0,
            diagnostic_snapshot: None,
            fault: None,
            observations,
        })
    }
    /// # Errors
    /// Returns a conflict error if provider installation is attempted after requests or replay actions have been recorded.
    pub fn with_workloads(
        mut self,
        provider: impl WorkloadProvider + Send + Sync + 'static,
    ) -> Result<Self> {
        if !self.host.requests.is_empty() || !self.actions.is_empty() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "providers must be installed before admission",
            ));
        }
        self.workloads = Box::new(provider);
        Ok(self)
    }
}

impl<B: BackendProvider + Send + 'static, P: SchedulingPolicy> Engine<B, P>
where
    B::Ticket: Send,
{
    /// Move device execution to a dedicated submission owner before service admission.
    /// # Errors
    /// Rejects active requests/tickets or failure to start the device thread.
    pub fn into_threaded(
        mut self,
        queue: usize,
        timeout: Duration,
    ) -> Result<Engine<ThreadedBackend<B>, P>> {
        if self.inflight.is_some()
            || !self.output_pending.is_empty()
            || !self.resources_pending.is_empty()
            || self.backend.pending_resource_releases()
        {
            return Err(Error::invalid(
                "start runner at a quiescent resource boundary",
            ));
        }
        let states: Vec<_> = self
            .checkpoint_states()?
            .into_iter()
            .map(|state| state.0)
            .collect();
        self.config.resource_timeout_us = u64::try_from(timeout.as_micros())
            .map_err(|_| Error::invalid("resource timeout overflow"))?;
        self.config.validate()?;
        self.collect_observations();
        let store = replace(
            &mut self.observations.store,
            infer_observe::ObservationStore::new(self.config.history_capacity)?,
        );
        let reader = self
            .event_reader
            .take()
            .ok_or_else(|| Error::invariant("event reader already transferred"))?;
        self.observations.collector = Some(Collector::new(
            reader,
            store,
            self.events.published(),
            self.config
                .history_capacity
                .saturating_add(self.config.max_requests)
                .min(MAX_TRACKED_PARENTS),
        )?);
        let output_worker = OutputWorker::placed(
            &self.fork_workloads()?.into(),
            self.config.max_num_seqs.min(self.config.max_requests),
            self.model.vocab_size,
            &self.config.cpu.placement.output,
        )?;
        let backend = ThreadedBackend::with_states_placed(
            self.backend,
            self.model.clone(),
            self.program.clone(),
            RunnerConfig {
                control_capacity: queue,
                max_num_seqs: self.config.max_num_seqs,
                max_states: self.config.max_requests,
                batch_slots: 2,
                poll_min_us: self.config.cpu.poll_min_us,
                poll_max_us: self.config.cpu.poll_max_us,
            },
            &states,
            &self.config.cpu.placement.device,
        )?;
        Ok(Engine {
            backend,
            policy: self.policy,
            planning_workspace: self.planning_workspace,
            host: self.host,
            output_worker: Some(output_worker),
            output_pending: self.output_pending,
            output_owners: self.output_owners,
            workloads: self.workloads,
            costs: self.costs,
            admission_policy: self.admission_policy,
            pending_cost_observations: self.pending_cost_observations,
            replaying: self.replaying,
            preemption_focus: self.preemption_focus,
            model: self.model,
            program: self.program,
            config: self.config,
            limits: self.limits,
            state: self.state,
            resources_pending: self.resources_pending,
            resource_epoch: self.resource_epoch,
            backend_resource_epoch: 0,
            seen_requests: self.seen_requests,
            retired_request_floor: self.retired_request_floor,
            tenants: self.tenants,
            ids: self.ids,
            inflight: None,
            global_progress_epoch: self.global_progress_epoch,
            now_us: self.now_us,
            guard: self.guard,
            events: self.events,
            event_reader: self.event_reader,
            decisions: self.decisions,
            actions: self.actions,
            dropped_actions: self.dropped_actions,
            diagnostic_snapshot: self.diagnostic_snapshot,
            fault: self.fault,
            observations: self.observations,
        })
    }
}
