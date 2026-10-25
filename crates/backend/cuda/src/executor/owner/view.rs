//! Committed host inspection remains available while the executor is suspended.
use super::super::{CudaExecutor, PoolInspection};
use infer_core::{Error, Result, StateId};
use infer_ir::{
    DataflowGraph, DeviceCapabilities, ExecutionProfileInspection, KvCacheInspection, ModelIr,
    PageGrowth, PrefillGeometry, TaskOutput,
};
use std::collections::BTreeMap;

pub(super) struct StateView {
    pub capacity: usize,
    pub cursor: usize,
    pub growth: Option<PageGrowth>,
    pub geometry: PrefillGeometry,
    pub program_width: usize,
    pub poisoned: bool,
}
pub(super) struct View {
    pub model: ModelIr,
    pub graph: DataflowGraph,
    pub identity: String,
    pub capabilities: DeviceCapabilities,
    pub profile: ExecutionProfileInspection,
    pub free: Option<u64>,
    pub capacity: Option<u64>,
    pub kv: Option<KvCacheInspection>,
    pub pool: PoolInspection,
    pub prefix: (u64, u64, u64, usize),
    pub states: BTreeMap<StateId, StateView>,
    pub result: Option<Result<Vec<TaskOutput>>>,
}
impl View {
    pub(super) fn new(executor: &CudaExecutor) -> Result<Self> {
        let mut view = Self {
            model: executor.model().clone(),
            graph: executor.loaded.graph().clone(),
            identity: executor.identity.clone(),
            capabilities: executor.capabilities.clone(),
            profile: executor.execution_profile(),
            free: None,
            capacity: None,
            kv: None,
            pool: executor.pool_inspection(),
            prefix: executor.prefix_inspection(),
            states: BTreeMap::new(),
            result: None,
        };
        view.refresh(executor)?;
        Ok(view)
    }
    pub(super) fn state(&self, id: StateId) -> Result<&StateView> {
        self.states
            .get(&id)
            .ok_or_else(|| Error::invalid("unknown CUDA state"))
    }
    pub(super) fn refresh(&mut self, executor: &CudaExecutor) -> Result<()> {
        self.free = executor.free_state_bytes()?;
        self.capacity = executor.state_capacity_bytes()?;
        self.kv = executor.kv_cache();
        self.pool = executor.pool_inspection();
        self.prefix = executor.prefix_inspection();
        self.profile = executor.execution_profile();
        self.states.retain(|id, _| executor.states.contains_key(id));
        for (id, state) in &executor.states {
            self.states.insert(
                *id,
                StateView {
                    capacity: state.capacity,
                    cursor: state.history.len(),
                    growth: executor.state_page_growth(*id)?,
                    geometry: executor.prefill_geometry_for(*id)?,
                    program_width: executor.prefill_width_for(*id)?,
                    poisoned: state.poisoned,
                },
            );
        }
        Ok(())
    }
}
