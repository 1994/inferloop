//! Fixed generational cold slots, hot `SoA` columns, and coalesced request mutations.
pub mod record;
mod tenant;
use infer_core::{
    Error, ErrorCode, ProgramId, RequestHandle, RequestId, RequestStatus, Result, StateId, StepId,
    arena::Arena, map::BoundedMap,
};
use infer_ir::{ExecutionRole, Workload};
use std::collections::BTreeMap;
use std::mem::take;
use std::sync::Arc;

pub use record::{RequestRecord, TokenContext};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HotStatus {
    Runnable,
    Waiting,
    Running,
    Terminal,
}
pub struct RequestHotSoA {
    pub ids: Vec<Option<RequestId>>,
    pub handles: Vec<Option<RequestHandle>>,
    pub status: Vec<HotStatus>,
    pub state: Vec<Option<StateId>>,
    pub program: Vec<Option<ProgramId>>,
    pub phase: Vec<ExecutionRole>,
    pub computed: Vec<usize>,
    pub reserved: Vec<usize>,
    pub remaining: Vec<usize>,
    pub tenant: Vec<Option<Arc<str>>>,
    pub weight: Vec<u32>,
    pub deadline: Vec<Option<u64>>,
    pub latency_deadline: Vec<Option<u64>>,
    pub last_service: Vec<u64>,
    pub progress: Vec<u64>,
    pub flight: Vec<Option<StepId>>,
    pub cancelled: Vec<bool>,
}
pub struct RequestArena {
    records: Arena<RequestHandle, RequestRecord>,
    index: BoundedMap<RequestId, RequestHandle>,
    order: Vec<RequestId>,
    pub hot: RequestHotSoA,
    dirty: Vec<usize>,
    marked: Vec<bool>,
}
impl RequestArena {
    pub fn new(capacity: usize) -> Result<Self> {
        let records = Arena::with_capacity(capacity)?;
        let index = BoundedMap::new(capacity)?;
        let mut order = Vec::new();
        let mut dirty = Vec::new();
        order
            .try_reserve_exact(capacity)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        dirty
            .try_reserve_exact(capacity)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        Ok(Self {
            records,
            index,
            order,
            hot: RequestHotSoA {
                ids: vec![None; capacity],
                handles: vec![None; capacity],
                status: vec![HotStatus::Terminal; capacity],
                state: vec![None; capacity],
                program: vec![None; capacity],
                phase: vec![ExecutionRole::Forward; capacity],
                computed: vec![0; capacity],
                reserved: vec![0; capacity],
                remaining: vec![0; capacity],
                tenant: vec![None; capacity],
                weight: vec![0; capacity],
                deadline: vec![None; capacity],
                latency_deadline: vec![None; capacity],
                last_service: vec![0; capacity],
                progress: vec![0; capacity],
                flight: vec![None; capacity],
                cancelled: vec![false; capacity],
            },
            dirty,
            marked: vec![false; capacity],
        })
    }
    pub const fn len(&self) -> usize {
        self.records.len()
    }
    pub const fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    pub fn contains_key(&self, id: RequestId) -> bool {
        self.index.contains_key(&id)
    }
    pub fn get(&self, id: RequestId) -> Option<&RequestRecord> {
        self.records.get(*self.index.get(&id)?)
    }
    pub fn known(&self, id: RequestId) -> Result<&RequestRecord> {
        self.get(id)
            .ok_or_else(|| Error::invariant("request arena identity missing"))
    }
    pub fn slot(&self, id: RequestId) -> Option<usize> {
        self.records.index(*self.index.get(&id)?)
    }
    pub fn get_mut(&mut self, id: RequestId) -> Option<&mut RequestRecord> {
        let handle = *self.index.get(&id)?;
        let slot = self.records.index(handle)?;
        self.mark(slot);
        self.records.get_mut(handle)
    }
    fn mark(&mut self, slot: usize) {
        if !self.marked[slot] {
            self.marked[slot] = true;
            self.dirty.push(slot);
        }
    }
    pub fn insert(&mut self, id: RequestId, record: RequestRecord) -> Result<()> {
        if id != record.request.id || self.index.contains_key(&id) {
            return Err(Error::invariant("invalid request arena insertion"));
        }
        let handle = self.records.insert(record)?;
        let slot = self
            .records
            .index(handle)
            .ok_or_else(|| Error::invariant("new request slot missing"))?;
        self.index.insert(id, handle)?;
        let at = self.order.binary_search(&id).unwrap_or_else(|at| at);
        self.order.insert(at, id);
        self.hot.ids[slot] = Some(id);
        self.hot.handles[slot] = Some(handle);
        self.mark(slot);
        Ok(())
    }
    pub fn remove(&mut self, id: RequestId) -> Option<RequestRecord> {
        let handle = self.index.remove(&id)?;
        let slot = self.records.index(handle)?;
        let at = self.order.binary_search(&id).ok()?;
        self.order.remove(at);
        self.hot.ids[slot] = None;
        self.hot.handles[slot] = None;
        self.mark(slot);
        self.records.remove(handle)
    }
    pub fn keys(&self) -> impl Iterator<Item = &RequestId> {
        self.order.iter()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&RequestId, &RequestRecord)> {
        self.order
            .iter()
            .filter_map(|id| self.get(*id).map(|record| (id, record)))
    }
    pub fn values(&self) -> impl Iterator<Item = &RequestRecord> {
        self.iter().map(|(_, r)| r)
    }
    pub fn snapshot(&self) -> BTreeMap<RequestId, RequestRecord> {
        self.iter().map(|(id, r)| (*id, r.clone())).collect()
    }
    pub fn take_dirty(&mut self) -> Vec<usize> {
        take(&mut self.dirty)
    }
    pub fn restore_dirty(&mut self, mut dirty: Vec<usize>) {
        dirty.clear();
        self.dirty = dirty;
    }
    pub fn sync(&mut self, slot: usize, state: &infer_state::SequenceStateManager) -> Result<()> {
        self.marked[slot] = false;
        let Some(record) = self.hot.handles[slot].and_then(|h| self.records.get(h)) else {
            self.hot.status[slot] = HotStatus::Terminal;
            self.hot.state[slot] = None;
            self.hot.program[slot] = None;
            self.hot.remaining[slot] = 0;
            self.hot.computed[slot] = 0;
            self.hot.reserved[slot] = 0;
            self.hot.tenant[slot] = None;
            self.hot.deadline[slot] = None;
            self.hot.latency_deadline[slot] = None;
            self.hot.flight[slot] = None;
            self.hot.cancelled[slot] = false;
            return Ok(());
        };
        self.hot.status[slot] = match &record.status {
            RequestStatus::Runnable => HotStatus::Runnable,
            RequestStatus::Waiting(_) => HotStatus::Waiting,
            RequestStatus::Running { .. } => HotStatus::Running,
            RequestStatus::Finished(_) => HotStatus::Terminal,
        };
        self.hot.state[slot] = record.state;
        self.hot.program[slot] = Some(record.plan.program);
        self.hot.computed[slot] = record
            .state
            .map(|id| state.get(id).map(|state| state.committed_tokens))
            .transpose()?
            .unwrap_or(0);
        self.hot.tenant[slot].clone_from(&Some(record.tenant.clone()));
        self.hot.weight[slot] = record.request.qos.weight;
        self.hot.deadline[slot] = record.request.qos.deadline_us;
        self.hot.latency_deadline[slot] = if record.first_token_us.is_none() {
            record
                .request
                .qos
                .ttft_slo_us
                .map(|slo| record.accepted_us.saturating_add(slo))
        } else {
            record
                .request
                .qos
                .tpot_slo_us
                .and_then(|slo| record.last_token_us.map(|time| time.saturating_add(slo)))
        };
        self.hot.last_service[slot] = record.last_service_us;
        self.hot.progress[slot] = record.progress_epoch;
        self.hot.flight[slot] = if let RequestStatus::Running { step } = record.status {
            Some(step)
        } else {
            None
        };
        self.hot.cancelled[slot] = record.pending_finish.is_some();
        self.hot.reserved[slot] = record.plan.reserved_tokens;
        self.hot.phase[slot] = if matches!(record.request.workload, Workload::Generate { .. }) {
            if record.prefill_done == record.prefill_target {
                ExecutionRole::Decode
            } else {
                ExecutionRole::Prefill
            }
        } else {
            ExecutionRole::Forward
        };
        self.hot.remaining[slot] = if self.hot.phase[slot] == ExecutionRole::Decode {
            1
        } else {
            record
                .pending_prefix_stop()
                .unwrap_or_else(|| record.context.len())
                .saturating_sub(record.prefill_done)
        };
        Ok(())
    }
}
