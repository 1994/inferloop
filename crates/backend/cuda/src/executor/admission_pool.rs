//! Select physical serving resources from real request lengths at idle boundaries.
use super::CudaExecutor;
use crate::constants::{CB_SLOT_TOKENS, KV_BLOCK_TOKENS, MAX_POOL_SLOTS};
use crate::executor::execution::pooled_envelope;
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::OutputReadout;

impl CudaExecutor {
    pub(super) fn pool_funding(&self) -> Result<(u64, u64)> {
        let scratch = self.loaded.shared_scratch_bytes()?;
        let physical = self.physical_available()?;
        Ok((
            pooled_envelope(physical, self.budget, self.active_budget()?, scratch),
            physical,
        ))
    }

    /// The funded pool's memory split, so capacity work can name its largest item.
    pub(super) fn log_pool_memory(&self) {
        let (Some(quote), Some(pool)) = (self.pool_quote.as_ref(), self.slots.as_ref()) else {
            return;
        };
        let kv_pages = u64::try_from(pool.kv_pages().0).unwrap_or(u64::MAX);
        tracing::info!(target: "infer::cuda_admission",
            kv_bytes = kv_pages.saturating_mul(self.kv_block_bytes),
            target_pool_bytes = quote.target_pool_bytes,
            draft_pool_bytes = quote.draft_pool_bytes,
            target_worker_bytes = quote.target_worker_bytes,
            draft_worker_bytes = quote.draft_worker_bytes,
            target_homes_bytes = quote.target_homes_bytes,
            draft_homes_bytes = quote.draft_homes_bytes,
            graph_headroom_bytes = quote.graph_headroom_bytes,
            total_bytes = quote.total_bytes,
            "CUDA serving pool memory split");
    }

    /// Capture the zero-demand funded pool unless one exists (`CudaBackend::with_pool_captured`).
    pub(super) fn capture_funded_pool(&mut self) {
        if self.slots.is_none() && self.pool_quote.is_some() {
            self.warm_slot_pool_for(CB_SLOT_TOKENS.min(self.loaded.model().max_sequence), 0);
        }
    }

    /// A pure quote bounds runtime resources; capture waits for actual request demand.
    pub(super) fn defer_request_pool(&mut self) -> Result<()> {
        if self.dflash.is_none() && (self.maximum_states < 2 || self.roofline_bound == Some(1)) {
            self.warm_slot_pool();
            return Ok(());
        }
        let capacity = CB_SLOT_TOKENS.min(self.loaded.model().max_sequence);
        let (envelope, _) = self.pool_funding()?;
        let maximum = self.maximum_states.min(MAX_POOL_SLOTS);
        if let Some(plan) = self.pooled_plan_for(capacity, maximum, envelope, 0)? {
            self.memory_bound = plan.memory_bound;
            self.maximum_states = self.maximum_states.min(plan.quote.width);
            self.pool_quote = Some(plan.quote);
            self.pool_envelope = envelope;
            tracing::info!(target: "infer::cuda_admission",
                width = plan.quote.width, capacity, envelope_bytes = envelope,
                "funded CUDA concurrency pending actual request length");
        } else {
            self.pool_failure_reason = Some("no complete deferred CUDA serving quote".into());
        }
        Ok(())
    }

    /// Reservation owns the incoming request. Quotes and live requests never recapture.
    pub(super) fn resolve_request_pool(
        &mut self,
        id: StateId,
        tokens: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        let capacity = CB_SLOT_TOKENS.min(self.loaded.model().max_sequence);
        if readout != OutputReadout::Logits
            || tokens > capacity
            || (self.dflash.is_none()
                && (self.maximum_states < 2 || self.roofline_bound == Some(1)))
        {
            return Ok(());
        }
        self.idle()?;
        let Some(demand) = self.defer_solo_pool(id, tokens)? else {
            return Ok(());
        };
        if !self.pool_is_idle() {
            return Ok(());
        }
        // A shorter request does not justify rebuilding a working pool. Retain the
        // current width whenever its actual physical pages can fund that demand.
        if self.slots.as_ref().is_some_and(|pool| {
            let (pages, _) = pool.kv_pages();
            tokens <= pool.capacity()
                && pool
                    .width()
                    .saturating_mul(tokens.div_ceil(KV_BLOCK_TOKENS))
                    <= pages
        }) {
            return Ok(());
        }
        let maximum = self.maximum_states.min(MAX_POOL_SLOTS);
        // An accepted pool's envelope remains an upper bound for the idle replacement.
        // Actual free memory is measured again after retirement, before any allocation.
        let envelope = self.pool_envelope.max(self.pool_funding()?.0);
        let Some(plan) = self.pooled_plan_for(capacity, maximum, envelope, demand)? else {
            if self.dflash.is_some() {
                return Err(Error::new(
                    ErrorCode::Capacity,
                    "DFlash2 proposal and verification graphs exceed the serving budget",
                ));
            }
            // A single long request may still use the current pool or private execution.
            if self.slots.is_none() {
                self.pool_quote = None;
            }
            return Ok(());
        };
        if self.slots.is_some()
            && self.pool_quote.as_ref().is_some_and(|current| {
                current.width == plan.quote.width
                    && current.capacity == plan.quote.capacity
                    && current.prompt_width == plan.quote.prompt_width
            })
        {
            return Ok(());
        }
        self.retire_request_pool()?;
        self.warm_slot_pool_for(capacity, demand);
        if self.slots.is_none() {
            return Err(Error::new(
                ErrorCode::Backend,
                format!(
                    "funded request-length CUDA capture failed: {}",
                    self.pool_failure_reason
                        .as_deref()
                        .unwrap_or("no captured pool")
                ),
            ));
        }
        Ok(())
    }

    /// A solo request uses a reusable private program. Only a previously observed
    /// concurrent reservation may pay for pool capture once that solo state drains.
    fn defer_solo_pool(&mut self, id: StateId, tokens: usize) -> Result<Option<usize>> {
        // A separate draft has no private proposal graph. Its first request must fund the
        // shared verification/proposal pair before prefill, including a solo request.
        if self.slots.is_some() || (self.dflash.is_some() && self.states.is_empty()) {
            return Ok(Some(tokens));
        }
        if self.states.is_empty() {
            return Ok(self
                .pool_retry
                .take()
                .and_then(|(waiting, demand)| (waiting == id).then_some(demand.max(tokens))));
        }
        if self.states.len() == 1
            && self.states.values().all(|state| {
                state.readout == OutputReadout::Logits
                    && state.capacity <= CB_SLOT_TOKENS
                    && state.media.is_none()
            })
        {
            let demand = self
                .states
                .values()
                .map(|state| state.capacity)
                .max()
                .unwrap_or(tokens)
                .max(tokens);
            let waiting = self.pool_retry.get_or_insert((id, demand));
            waiting.1 = waiting.1.max(demand);
            return Err(Error::new(
                ErrorCode::Capacity,
                "concurrent request waits for the solo state before pooled capture",
            ));
        }
        // Incompatible private states retain the existing funded lazy batching path.
        Ok(None)
    }

    fn pool_is_idle(&self) -> bool {
        self.states.is_empty()
            && self
                .slots
                .as_ref()
                .is_none_or(|pool| pool.free_slots() == pool.width())
            && self
                .draft_slots
                .as_ref()
                .is_none_or(|pool| pool.free_slots() == pool.width())
    }

    fn retire_request_pool(&mut self) -> Result<()> {
        tracing::info!(target: "infer::cuda_admission", "retired idle CUDA request pool");
        self.worker = None;
        self.homes.clear();
        if let Some(engine) = self.dflash.as_mut() {
            engine.detach_pool();
        }
        drop(self.draft_slots.take());
        drop(self.slots.take());
        self.pool_quote = None;
        while self.pool.evict() {}
        while self.prefix.evict_oldest().is_some() {}
        if let Err(error) = self.loaded.device().reclaim_barrier() {
            self.fatal = Some(error.clone());
            return Err(error);
        }
        self.loaded.reclaim_idle_prompt_scratch()?;
        self.slots_disabled = false;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/executor_request_kv.rs"]
mod tests;
