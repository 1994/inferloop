use crate::constants::MAX_CALL_HISTORY;
use crate::{experiment::ExperimentResult, registry::CommandDescriptor};
use infer_core::{ErrorCode, IdAllocator, Result, SessionId};
use infer_ir::{ExecutionStats, LayerProbe, OpTrace};
use infer_kernel_api::KernelRegistry;
use infer_runtime::Engine;
use infer_spi::BackendProvider;
use serde::Serialize;
use serde_json::Value;
use std::collections::VecDeque;

/// A cold-path diagnostic adapter; scheduler and inference never depend on Agent.
pub trait AgentBackend: BackendProvider + Sized + 'static {
    /// # Errors
    /// Returns backend initialization errors while creating independent execution state.
    fn fresh(&self) -> Result<Self>;
    /// # Errors
    /// Returns registration errors for the backend's kernel catalog.
    fn registry(&self) -> Result<KernelRegistry>;
    fn inspection(&self) -> Value;
    fn traces(&self) -> Vec<OpTrace>;
    fn trace_timing_scope(&self) -> &'static str;
    fn probes(&self) -> Vec<LayerProbe>;
    fn profile(&self) -> Value;
    fn execution_stats(&self) -> Option<ExecutionStats>;
}

#[derive(Debug, Clone, Serialize)]
pub struct CallRecord {
    pub sequence: u64,
    pub method: String,
    pub error: Option<ErrorCode>,
    pub rpc_error: Option<i32>,
    pub elapsed_us: u64,
    pub notification: bool,
}

pub struct AgentContext<B: AgentBackend> {
    pub(crate) engine: Engine<B>,
    pub(crate) backend_catalog: Value,
    pub(crate) descriptors: Vec<CommandDescriptor>,
    pub(crate) calls: VecDeque<CallRecord>,
    pub(crate) call_sequence: u64,
    pub(crate) dropped_calls: u64,
    pub(crate) experiments: VecDeque<ExperimentResult>,
    pub(crate) ids: IdAllocator,
    pub session: SessionId,
    /// Requests a temporarily full device refused, kept until capacity frees.
    ///
    /// Rejection happens before the engine retains the request identity, so retrying the same
    /// id is safe and the client is told the request was queued rather than failed.
    pub(crate) pending: VecDeque<(infer_ir::CanonicalRequest, Option<u64>)>,
    /// Queued submissions that waited past the budget and were dropped.
    pub(crate) expired: Vec<infer_core::RequestId>,
}

impl<B: AgentBackend> AgentContext<B> {
    pub(crate) fn new(engine: Engine<B>, backend_catalog: Value) -> Self {
        Self {
            engine,
            backend_catalog,
            descriptors: Vec::new(),
            calls: VecDeque::new(),
            call_sequence: 0,
            dropped_calls: 0,
            experiments: VecDeque::new(),
            ids: IdAllocator::default(),
            session: SessionId::ONE,
            pending: VecDeque::new(),
            expired: Vec::new(),
        }
    }

    #[must_use]
    pub const fn engine(&self) -> &Engine<B> {
        &self.engine
    }

    pub const fn engine_mut(&mut self) -> &mut Engine<B> {
        &mut self.engine
    }

    pub(crate) fn record(&mut self, record: CallRecord) {
        if self.calls.len() == MAX_CALL_HISTORY {
            self.calls.pop_front();
            self.dropped_calls = self.dropped_calls.saturating_add(1);
        }
        self.calls.push_back(record);
    }
}
