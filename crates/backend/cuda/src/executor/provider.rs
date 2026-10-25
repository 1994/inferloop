use super::{CudaBackend, CudaTicket, prefix::Restore};
use crate::constants::KV_BLOCK_TOKENS;
use infer_core::{Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, KvCacheInspection, ModelIr, OutputReadout,
    PageGrowth, StepPlan, TaskOutput,
};
use infer_spi::BackendProvider;
impl BackendProvider for CudaBackend {
    type Ticket = CudaTicket;
    fn identity(&self) -> &str {
        &self.identity
    }
    fn weight_backed_dataflow(&self) -> bool {
        true
    }
    fn execution_profile(&self) -> Option<infer_ir::ExecutionProfileInspection> {
        Some(self.loaded.execution_profile())
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }
    fn speculation_capability(&self) -> infer_ir::SpeculationCapability {
        self.capabilities.speculation
    }
    fn control_ready(&self) -> bool {
        !self.busy && self.fatal.is_none()
    }
    fn maintenance(&mut self) -> Result<()> {
        self.fatal
            .as_ref()
            .map_or(Ok(()), |error| Err(error.clone()))
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        if model != self.model() {
            return Err(infer_core::Error::invalid("execution graph model mismatch"));
        }
        Ok(self.loaded.graph().clone())
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.check_program(model, program)
    }
    fn prepare_state_reservation(&mut self, capacity: usize, readout: OutputReadout) -> Result<()> {
        self.prepare_reservation_for(capacity, readout)
    }
    fn state_reservation_bytes(&self, capacity: usize) -> Result<Option<u64>> {
        self.state_reservation_bytes_for(capacity, OutputReadout::Full)
    }
    fn state_reservation_bytes_for(
        &self,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<Option<u64>> {
        // Prefill still needs the sequence's own program even when decode will lease a slot.
        // Capture and cached-shell accounting must use the same price as reserve().
        let private = self.private_verification_for(capacity, readout);
        self.reservation_budget(capacity, readout, private)
            .map(Some)
    }
    fn state_capacity_bytes(&self) -> Result<Option<u64>> {
        self.capacity_budget().map(Some)
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        self.available().map(Some)
    }
    /// Report the slot pool in the engine's block vocabulary. The pool is slot-granular and
    /// pre-paid: a leased slot's untouched tail is where its sequence's growth comes from, so
    /// it counts as available; a free slot's blocks are entirely available. The prefix cache
    /// holds whole-state snapshots outside the pool, so its coverage is reported in block
    /// units for observability without ever feeding the available count. `total_blocks` is
    /// floored at the private path's single-sequence ceiling: a sequence too long for the
    /// pool still gets a private reservation up to the model context, and the admission
    /// isolation check compares against this field.
    fn kv_cache(&self) -> Option<KvCacheInspection> {
        let pool = self.slots.as_ref()?;
        let block = KV_BLOCK_TOKENS;
        let per_slot = pool.capacity().div_ceil(block);
        let pool_blocks = pool.width().saturating_mul(per_slot);
        let active_blocks = self
            .states
            .values()
            .filter(|state| state.slot.is_some())
            .map(|state| state.history.len().div_ceil(block))
            .sum();
        let (allocations, releases) = pool.lease_counters();
        Some(KvCacheInspection {
            pinned_blocks: 0,
            block_size: block,
            total_blocks: pool_blocks.max(self.loaded.model().max_sequence.div_ceil(block)),
            free_blocks: pool.free_slots().saturating_mul(per_slot),
            active_blocks,
            cached_blocks: self.prefix.cached_tokens().div_ceil(block),
            shared_blocks: 0,
            available_blocks: pool_blocks.saturating_sub(active_blocks),
            pool_bytes: u64::try_from(pool_blocks)
                .unwrap_or(u64::MAX)
                .saturating_mul(self.kv_block_bytes),
            bytes_per_block: self.kv_block_bytes,
            allocations,
            releases,
            cow_copies: 0,
            cache_evictions: self.prefix.evictions,
            reused_tokens: self.prefix.reused_tokens,
        })
    }
    /// Physical page demand of a sequence's next step. A slot-leased sequence grows into its
    /// slot's pre-paid tail, so the scheduler sees the content blocks it commits; a private
    /// sequence reserved its whole capacity at admission and never grows at all. The draft
    /// pool lags the target by one token and crosses the same 32-token boundaries, so it is
    /// not separately reported.
    fn state_page_growth(&self, state: StateId) -> Result<Option<PageGrowth>> {
        let sequence = self
            .states
            .get(&state)
            .ok_or_else(|| infer_core::Error::invalid("unknown CUDA state"))?;
        let block = KV_BLOCK_TOKENS;
        let allocated_pages = if sequence.slot.is_some() {
            sequence.history.len().div_ceil(block)
        } else {
            sequence.capacity.div_ceil(block)
        };
        Ok(Some(PageGrowth {
            block_size: block,
            allocated_pages,
            bytes_per_page: self.kv_block_bytes,
            cow_tail: false,
        }))
    }
    fn state_prefill_geometry(&self, state: StateId) -> Result<Option<infer_ir::PrefillGeometry>> {
        self.prefill_geometry_for(state).map(Some)
    }
    fn reserve_state(&mut self, state: StateId, capacity: usize) -> Result<()> {
        self.reserve(state, capacity, OutputReadout::Full)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.reserve(state, capacity, readout)
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.reset(state)
    }
    /// Restore a cached prompt prefix into `state` and report how many tokens it covered.
    ///
    /// Returns zero whenever the snapshot cannot be used verbatim (no match, non-logits readout,
    /// unknown state), which makes the engine fall back to a full prefill.
    fn reuse_prefix_shared(
        &mut self,
        state: StateId,
        tokens: infer_ir::TokenBuffer,
        maximum: usize,
    ) -> Result<usize> {
        self.idle()?;
        let Some(sequence) = self.states.get_mut(&state) else {
            return Ok(0);
        };
        if !Restore::supports(sequence) {
            return Ok(0);
        }
        let maximum = maximum.min(sequence.capacity);
        self.prefix
            .reuse(
                &tokens,
                maximum,
                &mut Restore {
                    device: self.loaded.device(),
                    sequence,
                },
            )
            .map(|matched| matched.unwrap_or(0))
    }

    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.release(state)
    }
    fn validate_state_ownership(&self, states: &[(StateId, usize, usize)]) -> Result<()> {
        self.ownership(states)
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<CudaTicket> {
        self.execute(program, step, &tasks)
    }
    fn submit_borrowed(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<CudaTicket> {
        self.execute(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut CudaTicket) -> Result<Option<Vec<TaskOutput>>> {
        self.complete(ticket)
    }
}
