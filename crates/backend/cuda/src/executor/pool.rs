//! Bounded idle graph/state retention. Live request IDs never enter this cache.
use super::{CudaExecutor, Sequence};
use infer_core::{Error, Result};
use infer_ir::OutputReadout;
use std::collections::VecDeque;

#[derive(Default)]
pub(super) struct StatePool {
    entries: VecDeque<Sequence>,
    bytes: u64,
    pub(super) reclaimable_bytes: u64,
    pub allocations: u64,
    pub reuses: u64,
    evictions: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PoolInspection {
    pub active_sequences: usize,
    pub cached_sequences: usize,
    pub cached_admission_bytes: u64,
    pub sequence_allocations: u64,
    pub sequence_reuses: u64,
    pub sequence_evictions: u64,
}

impl Sequence {
    fn reclaimable_bytes(&self) -> u64 {
        self.program.reclaimable_bytes
            + self
                .speculation
                .as_ref()
                .map_or(0, |s| s.program.reclaimable_bytes)
    }
}

impl StatePool {
    fn matching(&self, capacity: usize, readout: OutputReadout) -> Option<usize> {
        self.entries.iter().position(|s| {
            !s.poisoned
                && s.capacity.next_power_of_two() == capacity.next_power_of_two()
                && s.readout == readout
        })
    }
    pub(super) fn cached_budget(&self, capacity: usize, readout: OutputReadout) -> Option<u64> {
        self.matching(capacity, readout)
            .map(|index| self.entries[index].budget)
    }
    pub fn take(&mut self, capacity: usize, readout: OutputReadout) -> Option<Sequence> {
        let index = self.matching(capacity, readout)?;
        let sequence = self.entries.remove(index)?;
        self.bytes -= sequence.budget;
        self.reclaimable_bytes -= sequence.reclaimable_bytes();
        Some(sequence)
    }
    pub fn retain(&mut self, sequence: Sequence) {
        self.bytes += sequence.budget;
        self.reclaimable_bytes += sequence.reclaimable_bytes();
        self.entries.push_back(sequence);
    }
    pub fn evict(&mut self) -> bool {
        let Some(sequence) = self.entries.pop_front() else {
            return false;
        };
        self.bytes -= sequence.budget;
        self.reclaimable_bytes -= sequence.reclaimable_bytes();
        self.evictions = self.evictions.saturating_add(1);
        drop(sequence);
        true
    }
}

impl CudaExecutor {
    /// Cached programs can retain wider prompt graphs than a newly captured pooled program.
    /// Reuse keeps their original charge; a requested private recapture must also be covered.
    pub(super) fn reservation_budget(
        &self,
        capacity: usize,
        readout: OutputReadout,
        private_verification: bool,
    ) -> Result<u64> {
        let fresh = self
            .loaded
            .sequence_budget(capacity, readout, private_verification)?;
        Ok(self
            .pool
            .cached_budget(capacity, readout)
            .map_or(fresh, |retained| retained.max(fresh)))
    }
    #[must_use]
    pub fn pool_inspection(&self) -> PoolInspection {
        PoolInspection {
            active_sequences: self.states.len(),
            cached_sequences: self.pool.entries.len(),
            cached_admission_bytes: self.pool.bytes,
            sequence_allocations: self.pool.allocations,
            sequence_reuses: self.pool.reuses,
            sequence_evictions: self.pool.evictions,
        }
    }
    pub(super) fn active_budget(&self) -> Result<u64> {
        self.states.values().try_fold(0_u64, |sum, s| {
            sum.checked_add(s.budget)
                .ok_or_else(|| Error::invariant("CUDA reservation accounting overflow"))
        })
    }
    pub(super) fn make_room(&mut self, needed: u64, scratch: u64) -> Result<()> {
        let physical_needed = needed.saturating_add(scratch);
        let active = self.active_budget()?;
        while active
            .checked_add(self.pool.bytes)
            .and_then(|n| n.checked_add(self.prefix.bytes()))
            .and_then(|n| n.checked_add(needed))
            .is_none_or(|n| n > self.budget)
            || self.states.len() + self.pool.entries.len() >= self.maximum_states
            || physical_needed > self.physical_available()?
        {
            if !self.pool.evict() && self.prefix.evict_oldest().is_none() {
                let physical_available = self.physical_available()?;
                let (pool_slots, pool_free_slots) = self
                    .slots
                    .as_ref()
                    .map_or((0, 0), |pool| (pool.width(), pool.free_slots()));
                tracing::debug!(
                    target: "infer::cuda_admission",
                    phase = "memory_after_reclaim",
                    needed_bytes = needed,
                    pending_scratch_bytes = scratch,
                    physical_needed_bytes = physical_needed,
                    active_bytes = active,
                    cached_bytes = self.pool.bytes,
                    prefix_bytes = self.prefix.bytes(),
                    physical_available_bytes = physical_available,
                    budget_bytes = self.budget,
                    active_states = self.states.len(),
                    maximum_states = self.maximum_states,
                    pool_slots,
                    pool_free_slots,
                    "CUDA resident reservation rejected"
                );
                return Err(Error::new(
                    infer_core::ErrorCode::Capacity,
                    format!(
                        "CUDA resident state admission budget exhausted: needed={needed}, scratch={scratch}, physical_needed={physical_needed}, active={active}, cached={}, prefix={}, physical={}, budget={}",
                        self.pool.bytes,
                        self.prefix.bytes(),
                        physical_available,
                        self.budget
                    ),
                ));
            }
            self.loaded.device().reclaim_barrier()?;
        }
        Ok(())
    }
    /// Release only cached graphs. Live states and completion tickets remain owned.
    /// # Errors
    /// Rejects an in-flight completion or a failed device drain.
    pub fn trim_state_pool(&mut self) -> Result<()> {
        self.idle()?;
        self.loaded.device().drain()?;
        while self.pool.evict() {}
        self.loaded.device().reclaim_barrier()
    }
}
