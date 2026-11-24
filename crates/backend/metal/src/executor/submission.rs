use super::{CachedPrefix, MetalBackend, MetalDevice};
use crate::constants::MAX_PREFIX_ENTRIES;
use infer_core::{Error, Result, StateId};
use infer_ir::{ExecutionProgram, ExecutionTask, StepPlan};
use infer_state::blocks::BlockLease;

pub(super) struct EncodingUndo {
    pages: Vec<(StateId, Vec<BlockLease>)>,
    pinned_tails: Vec<BlockLease>,
}
pub(super) struct EncodedBatch {
    pub starts: Vec<usize>,
    pub dispatches: usize,
    pub pending_cache: Vec<CachedPrefix>,
}
impl MetalBackend {
    pub(super) fn begin_encoding(&mut self, tasks: &[ExecutionTask]) -> Result<EncodingUndo> {
        let mut pages = Vec::with_capacity(tasks.len());
        for task in tasks {
            let s = &self.sequences[&task.state];
            pages.push((task.state, s.blocks.clone()));
        }
        let pinned_tails = self.kv.pin_shared_tails(tasks.iter().map(|task| {
            let s = &self.sequences[&task.state];
            (s.blocks.as_slice(), s.tokens.len())
        }))?;
        // Protect old shared tails from cache eviction until commit or rollback.
        // Unshared tails stay unpinned so this guard does not introduce extra COW.
        Ok(EncodingUndo {
            pages,
            pinned_tails,
        })
    }
    pub(super) fn rollback_encoding(&mut self, undo: EncodingUndo) -> Result<()> {
        for (id, old) in undo.pages {
            let sequence = self
                .sequences
                .get_mut(&id)
                .ok_or_else(|| Error::invariant("encoding rollback sequence missing"))?;
            self.kv.rollback_append(&mut sequence.blocks, old)?;
            self.update_table(id)?;
        }
        self.kv.release(&undo.pinned_tails)
    }
    pub(super) fn finish_encoding(&mut self, undo: EncodingUndo) {
        // Source pages of queued COW copies remain pinned until GPU completion.
        self.inflight_pins = undo.pinned_tails;
    }
    pub(super) fn release_encoding_pins(&mut self) -> Result<()> {
        self.kv.release(&self.inflight_pins)?;
        self.inflight_pins.clear();
        Ok(())
    }
    pub(super) fn encode_batch(
        &mut self,
        command: &metal::CommandBufferRef,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<EncodedBatch> {
        let mut starts = Vec::with_capacity(tasks.len());
        let mut dispatches = 0;
        let mut pending_cache = vec![];
        let mut pending_bytes = 0u64;
        for task in tasks {
            self.grow_pages(task.state, task.tokens.len(), command)?;
            let s = &self.sequences[&task.state];
            self.encode_pending_prefix(command, s)?;
            let start = s.tokens.len();
            starts.push(start);
            MetalDevice::write_indices_range_idle(
                &s.token_buffer,
                start,
                task.tokens.delta(start)?,
            )?;
            let mut position = start;
            while position < task.tokens.len() {
                let chunk = self.next_chunk(position, task.tokens.len());
                let end = chunk.end;
                dispatches += self.encode_chunk(command, program, step, task, chunk)?;
                let s = &self.sequences[&task.state];
                if end.is_multiple_of(self.config.block_size)
                    && pending_bytes < self.config.prefix_cache_bytes
                    && pending_cache.len() < MAX_PREFIX_ENTRIES
                    && let Some(cached) = self.cache_boundary(
                        command,
                        s,
                        &task.tokens.prefix(&s.tokens, end)?,
                        self.config.prefix_cache_bytes - pending_bytes,
                    )?
                {
                    pending_bytes += cached.bytes;
                    pending_cache.push(cached);
                }
                position = end;
            }
        }
        Ok(EncodedBatch {
            starts,
            dispatches,
            pending_cache,
        })
    }
}

#[cfg(test)]
mod tests;
