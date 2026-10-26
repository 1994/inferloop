use crate::Engine;
use crate::resource::ResourcePhase;
use infer_core::{Error, Result};
use infer_ir::ResourceSnapshot;
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::mem::take;

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn execution_state_bytes(&self) -> Result<u64> {
        let free = self.backend.free_state_bytes()?.unwrap_or(u64::MAX);
        Ok(free.saturating_add(
            self.backend
                .kv_cache()
                .map_or(0, |c| c.available_blocks as u64 * c.bytes_per_block),
        ))
    }
    pub(crate) fn resources(&self) -> Result<ResourceSnapshot> {
        Ok(ResourceSnapshot {
            max_num_seqs: self.config.max_num_seqs,
            max_num_batched_tokens: self.config.max_num_batched_tokens,
            gpu_budget_us: self.config.gpu_budget_us,
            workspace_bytes: self.config.workspace_bytes,
            free_state_pages: self
                .backend
                .kv_cache()
                .map_or(usize::MAX, |c| c.available_blocks),
            free_logical_pages: self.state.free_pages(),
            free_state_bytes: self.execution_state_bytes()?,
            transfer_budget_us: self.config.transfer_budget_us,
            encoder_budget_us: self.config.encoder_budget_us,
            graphs: vec![],
            scheduler: self.config.scheduler.clone(),
        })
    }
    pub(crate) fn reuse_ready_prefixes(&mut self) -> Result<()> {
        self.fill_candidate_ids()?;
        let ids = take(&mut self.host.ready_index.candidates);
        let result = ids.iter().copied().try_for_each(|id| self.reuse_prefix(id));
        self.host.ready_index.candidates = ids;
        result
    }
    fn reuse_prefix(&mut self, id: infer_core::RequestId) -> Result<()> {
        let record = self.host.requests.known(id)?;
        if record.prefill_done != 0
            || record.prefix_attempted
            || record.media.is_some()
            || self.preemption_focus.is_some_and(|focus| focus != id)
        {
            return Ok(());
        }
        let state = record
            .state
            .ok_or_else(|| Error::invariant("prefix owner has no state"))?;
        let ticket = match self
            .backend
            .begin_resource(infer_spi::ResourceCommand::Prefix {
                state,
                tokens: record.context.prompt.clone(),
                maximum: record
                    .context
                    .len()
                    .saturating_sub(1)
                    .min(record.context.prompt.len()),
            }) {
            Ok(ticket) => ticket,
            Err(error) if error.code == infer_core::ErrorCode::Capacity => return Ok(()),
            Err(error) => return Err(error),
        };
        self.host
            .requests
            .get_mut(id)
            .ok_or_else(|| Error::invariant("prefix owner lost"))?
            .prefix_attempted = true;
        // The same acknowledgement path applies to direct and threaded backend owners.
        self.park_resource(id, ticket, ResourcePhase::Prefix)?;
        Ok(())
    }
}
