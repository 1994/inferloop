//! Scheduler-owned preallocated CPU storage; no backend-specific implementation belongs here.
use crate::persistence::checkpoint::CheckpointFlight;
use crate::pipeline::dispatch::PreparedDispatch;
use crate::stages::output::OutputJob;
use crate::{
    RuntimeConfig, pipeline::scheduling::index::ReadyIndex,
    pipeline::scheduling::index::ReadyWindow, pipeline::scheduling::overlap::PreparedNext,
    requests::RequestArena,
};
use infer_core::credits::CreditPool;
use infer_core::placement::PlacementReport;
use infer_core::{Error, ErrorCode, ProgramId, RequestId, Result};
use infer_ir::{BackendKind, CostEstimate, CostQuery, ExecutionRole};

pub struct HostStorage {
    pub placement: PlacementReport,
    pub preemption: infer_scheduler::PreemptionWorkspace,
    pub history: super::history::HistoryStorage,
    pub decisions: infer_spi::DecisionStorage,
    pub steps: super::plans::StepPool,
    pub queries: super::plans::QueryPool,
    pub ready_buffer: Option<ReadyWindow>,
    pub ready_index: ReadyIndex,
    pub cost_epoch: u64,
    pub history_tokens: usize,
    pub history_bytes: usize,
    pub validation: infer_scheduler::ValidationScratch,
    pub dispatch: Option<PreparedDispatch>,
    pub sampling: infer_workloads::SamplingWorkspace,
    pub output_jobs: Vec<(usize, OutputJob)>,
    pub resource_poll_buffer: Vec<RequestId>,
    pub credits: CreditPool,
    pub bytes: CreditPool,
    pub wake_buffer: Vec<RequestId>,
    pub expiry_buffer: Vec<RequestId>,
    pub prepared_next: Option<PreparedNext>,
    pub prepared_dispatched: u64,
    pub prepared_discarded: u64,
    pub checkpoint: Option<CheckpointFlight>,
    pub prepare_versions: Vec<u64>,
    pub requests: RequestArena,
    pub queues: infer_scheduler::RequestQueue,
}
fn scratch<T>(capacity: usize) -> Result<Vec<T>> {
    let mut storage = Vec::new();
    storage
        .try_reserve_exact(capacity)
        .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
    Ok(storage)
}
impl HostStorage {
    pub fn new(
        config: &RuntimeConfig,
        program: ProgramId,
        backend: BackendKind,
        vocabulary: usize,
    ) -> Result<Self> {
        config
            .cpu
            .placement
            .scheduler
            .scope(|| Self::initialize(config, program, backend, vocabulary))
    }
    fn initialize(
        config: &RuntimeConfig,
        program: ProgramId,
        backend: BackendKind,
        vocabulary: usize,
    ) -> Result<Self> {
        let candidates = config.candidate_limit.min(config.max_requests);
        let query = CostQuery::from_unit(
            program,
            backend,
            ExecutionRole::Prefill,
            1,
            1,
            CostEstimate::default(),
        );
        config.cpu.fixed_bytes_for(config, vocabulary)?;
        Ok(Self {
            placement: config.cpu.placement.scheduler.report()?,
            preemption: infer_scheduler::PreemptionWorkspace::new(candidates)?,
            history: super::history::HistoryStorage::new(
                config.history_capacity,
                candidates,
                config.max_num_seqs,
                program,
            )?,
            decisions: infer_spi::DecisionStorage::new(candidates, config.max_num_seqs)?,
            steps: super::plans::StepPool::new(config.max_num_seqs, program)?,
            queries: super::plans::QueryPool::new(config.history_capacity, config.max_num_seqs)?,
            ready_buffer: Some(ReadyWindow::new(candidates, query)?),
            ready_index: ReadyIndex::new(config.max_requests, candidates, query)?,
            cost_epoch: 0,
            history_tokens: 0,
            history_bytes: 0,
            validation: infer_scheduler::ValidationScratch::new(candidates, config.max_num_seqs)?,
            dispatch: Some(PreparedDispatch::new(config.max_num_seqs)?),
            sampling: infer_workloads::SamplingWorkspace::with_capacity(vocabulary)?,
            output_jobs: scratch(config.max_num_seqs)?,
            credits: CreditPool::new(config.max_requests, config.cpu.host_tokens)?,
            bytes: CreditPool::tracking(config.max_requests)?,
            prepared_next: None,
            prepared_dispatched: 0,
            prepared_discarded: 0,
            checkpoint: None,
            resource_poll_buffer: scratch(config.max_requests)?,
            wake_buffer: scratch(config.max_requests)?,
            expiry_buffer: scratch(config.max_requests)?,
            prepare_versions: scratch(candidates)?,
            requests: RequestArena::new(config.max_requests)?,
            queues: infer_scheduler::RequestQueue::new(config.max_requests)?,
        })
    }
}
