//! Select physical serving resources from real request lengths at idle boundaries.
use super::CudaExecutor;
use crate::constants::{CB_SLOT_TOKENS, MAX_POOL_SLOTS};
use crate::executor::execution::pooled_envelope;
use infer_core::{Error, ErrorCode, Result};
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

    /// A pure quote bounds runtime resources; capture waits for actual request demand.
    pub(super) fn defer_request_pool(&mut self) -> Result<()> {
        if self.maximum_states < 2 || self.roofline_bound == Some(1) {
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
        tokens: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        let capacity = CB_SLOT_TOKENS.min(self.loaded.model().max_sequence);
        if readout != OutputReadout::Logits
            || tokens > capacity
            || self.maximum_states < 2
            || self.roofline_bound == Some(1)
            || !self.pool_is_idle()
        {
            return Ok(());
        }
        self.idle()?;
        let maximum = self.maximum_states.min(MAX_POOL_SLOTS);
        // An accepted pool's envelope remains an upper bound for the idle replacement.
        // Actual free memory is measured again after retirement, before any allocation.
        let envelope = self.pool_envelope.max(self.pool_funding()?.0);
        let Some(plan) = self.pooled_plan_for(capacity, maximum, envelope, tokens)? else {
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
        self.warm_slot_pool_for(capacity, tokens);
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
