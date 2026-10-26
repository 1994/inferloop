use super::{CudaExecutor, prefix::Restore};
use crate::constants::KV_BLOCK_TOKENS;
use infer_core::{Result, StateId};
use infer_ir::{KvCacheInspection, OutputReadout, PageGrowth};
impl CudaExecutor {
    pub(super) fn prepare_state_reservation(
        &mut self,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.prepare_reservation_for(capacity, readout)
    }

    pub(super) fn prepare_state_reservation_for_media(
        &mut self,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.prepare_reservation_for(capacity, readout)
    }

    pub(super) fn state_reservation_bytes_for(
        &self,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<Option<u64>> {
        // Prefill still needs the sequence's own program even when decode will lease a slot.
        // Capture and cached-shell accounting must use the same price as reserve().
        let private = self.private_verification_for(capacity, readout, true);
        self.reservation_budget(capacity, readout, private)
            .map(Some)
    }

    pub(super) fn state_reservation_bytes_for_media(
        &self,
        capacity: usize,
        readout: OutputReadout,
        media: &infer_ir::MediaPlan,
    ) -> Result<Option<u64>> {
        // Media prefill bypasses slot/home programs and additionally retains encoded rows plus
        // the bounded encoder transient peak; price the same allowance reserve_media charges.
        let private = true;
        let encoded = self
            .loaded
            .vision()
            .map_or(Ok(0), |vision| vision.footprint(media))?;
        self.reservation_budget(capacity, readout, private)
            .and_then(|bytes| {
                bytes
                    .checked_add(encoded)
                    .ok_or_else(|| infer_core::Error::invalid("media quote footprint overflow"))
            })
            .map(Some)
    }
    pub(super) fn state_capacity_bytes(&self) -> Result<Option<u64>> {
        self.capacity_budget().map(Some)
    }
    pub(super) fn free_state_bytes(&self) -> Result<Option<u64>> {
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
    pub(super) fn kv_cache(&self) -> Option<KvCacheInspection> {
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
    pub(super) fn state_page_growth(&self, state: StateId) -> Result<Option<PageGrowth>> {
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

    #[cfg(test)]
    pub(super) fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.reserve(state, capacity, readout, true)
    }

    /// Restore a cached prompt prefix into `state` and report how many tokens it covered.
    ///
    /// Returns zero whenever the snapshot cannot be used verbatim (no match, non-logits readout,
    /// unknown state), which makes the engine fall back to a full prefill.
    pub(super) fn reuse_prefix_shared(
        &mut self,
        state: StateId,
        tokens: &infer_ir::TokenBuffer,
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
                tokens,
                maximum,
                &mut Restore {
                    device: self.loaded.device(),
                    sequence,
                },
            )
            .map(|matched| matched.unwrap_or(0))
    }

    #[cfg(test)]
    pub(super) fn release_state(&mut self, state: StateId) -> Result<()> {
        self.release(state)
    }
}
