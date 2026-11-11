use super::{CudaExecutor, prefix::Restore};
use crate::constants::KV_BLOCK_TOKENS;
use crate::resident::slot_batch::SlotPool;
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
    /// Report physically allocated/free pages, not logical slot tails. A paged pool's pages come
    /// from the engine's ledger; a pool whose slots own whole-capacity caches reports its slots
    /// as pages. Total capacity retains the private path's isolated context ceiling; its
    /// separately funded pages never inflate free pooled pages. Prefix snapshots remain
    /// separately owned.
    pub(super) fn kv_cache(&self) -> KvCacheInspection {
        let pool = self.slots.as_ref();
        let block = KV_BLOCK_TOKENS;
        let active_blocks = self
            .states
            .values()
            .filter(|state| state.slot.is_some() || state.home.is_some())
            .map(|state| state.history.len().div_ceil(block))
            .sum();
        let pending = self
            .pool_quote
            .as_ref()
            .filter(|_| pool.is_none())
            .map(|quote| (quote.width, quote.capacity));
        let max_sequence = self.loaded.model().max_sequence;
        let cached_blocks = self.prefix.cached_tokens().div_ceil(block);
        if let Some(ledger) = pool.and_then(SlotPool::kv_ledger) {
            return KvCacheInspection {
                total_blocks: logical_kv_blocks(ledger.total_blocks, pending, max_sequence),
                active_blocks,
                cached_blocks,
                cache_evictions: self.prefix.evictions,
                reused_tokens: self.prefix.reused_tokens,
                ..ledger
            };
        }
        let (pool_blocks, free_blocks) = pool.map_or((0, 0), SlotPool::kv_pages);
        let (allocations, releases) = pool.map_or((0, 0), SlotPool::lease_counters);
        KvCacheInspection {
            pinned_blocks: 0,
            block_size: block,
            total_blocks: logical_kv_blocks(pool_blocks, pending, max_sequence),
            free_blocks,
            active_blocks,
            cached_blocks,
            shared_blocks: 0,
            available_blocks: free_blocks,
            pool_bytes: u64::try_from(pool_blocks)
                .unwrap_or(u64::MAX)
                .saturating_mul(self.kv_block_bytes),
            bytes_per_block: self.kv_block_bytes,
            allocations,
            releases,
            cow_copies: 0,
            cache_evictions: self.prefix.evictions,
            reused_tokens: self.prefix.reused_tokens,
        }
    }
    /// A sequence on a paged slot owns the pages it has reached and leases more as it grows,
    /// so the engine charges each step the pages it adds. A private sequence prepaid its whole
    /// capacity, which is never charged again.
    pub(super) fn state_page_growth(&self, state: StateId) -> Result<Option<PageGrowth>> {
        let sequence = self
            .states
            .get(&state)
            .ok_or_else(|| infer_core::Error::invalid("unknown CUDA state"))?;
        let block = KV_BLOCK_TOKENS;
        let slot = sequence
            .slot
            .as_ref()
            .map(|lease| lease.slot)
            .or(sequence.home);
        let paged = slot.and_then(|slot| {
            let pool = self.slots.as_ref()?;
            let pages = pool.slot_pages(slot)?;
            // A verify step writes its lanes past the token the engine counts; keep one page
            // of them out of the report so the engine never under-charges a step.
            Some(pages.saturating_sub(usize::from(pool.verify().is_some_and(|lanes| lanes > 1))))
        });
        let allocated_pages = paged.unwrap_or_else(|| sequence.capacity.div_ceil(block));
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
        let pools = if let Some(slot) = sequence.home {
            let Some(target) = self.slots.as_mut() else {
                return Ok(0);
            };
            // Restored rows land in the slot's pages; without them the prompt prefills instead.
            if !target.grow(slot, maximum, self.loaded.device())? {
                return Ok(0);
            }
            Some((target, self.draft_slots.as_mut()))
        } else {
            None
        };
        self.prefix
            .reuse(
                tokens,
                maximum,
                &mut Restore {
                    device: self.loaded.device(),
                    sequence,
                    pools,
                },
            )
            .map(|matched| matched.unwrap_or(0))
    }

    #[cfg(test)]
    pub(super) fn release_state(&mut self, state: StateId) -> Result<()> {
        self.release(state)
    }
}

/// The runtime allocates logical descriptors once, before a demand-driven pool exists.
/// Its ceiling must cover any funded future slot layout; these are not free physical pages.
fn logical_kv_blocks(
    physical: usize,
    pending: Option<(usize, usize)>,
    private_capacity: usize,
) -> usize {
    let pooled = pending.map_or(physical, |(width, capacity)| {
        width.saturating_mul(capacity).div_ceil(KV_BLOCK_TOKENS)
    });
    pooled.max(private_capacity.div_ceil(KV_BLOCK_TOKENS))
}

#[cfg(test)]
#[path = "../../tests/unit/executor_kv_inspection.rs"]
mod tests;
