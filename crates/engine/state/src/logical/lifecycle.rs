//! Lifecycle responsibilities.
use super::SequenceStates;
use infer_core::{Error, ErrorCode, Result, StateId, StatePageId};
use std::mem::take;

impl SequenceStates {
    /// # Errors
    /// Returns a not-found error for unknown sequence state.
    pub fn reset(&mut self, id: StateId) -> Result<()> {
        self.get(id)?;
        let s = self
            .sequences
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("state disappeared during reset"))?;
        s.committed_tokens = 0;
        let mut pages = take(&mut s.pages);
        self.unreference(&pages);
        pages.clear();
        self.sequences
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("reset owner missing"))?
            .pages = pages;
        Ok(())
    }
    /// # Errors
    /// Returns a not-found error for unknown state or an invariant error for a non-monotonic or unallocated commit cursor.
    pub fn commit(&mut self, id: StateId, tokens: usize) -> Result<()> {
        let state = self
            .sequences
            .get_mut(&id)
            .ok_or_else(|| Error::new(ErrorCode::NotFound, "unknown sequence state"))?;
        if tokens < state.committed_tokens
            || tokens > state.capacity_tokens
            || tokens > state.pages.len() * self.block_size
        {
            return Err(Error::invariant("invalid state commit cursor"));
        }
        state.committed_tokens = tokens;
        Ok(())
    }
    #[expect(
        clippy::expect_used,
        reason = "Only validated private page tables reach this infallible ownership cleanup; missing pages indicate internal corruption"
    )]
    pub(super) fn unreference(&mut self, pages: &[StatePageId]) {
        for page in pages {
            let p = self.pages.get_mut(page).expect("validated reference");
            p.references -= 1;
            if p.references == 0 {
                self.pages.remove(page);
            }
        }
    }
    /// # Errors
    /// Returns a not-found or invariant error for unknown identities or inconsistent ownership.
    pub fn release(&mut self, id: StateId) -> Result<()> {
        let mut state = self
            .sequences
            .remove(&id)
            .ok_or_else(|| Error::new(ErrorCode::NotFound, "unknown sequence state"))?;
        self.unreference(&state.pages);
        self.owners.remove(&state.owner);
        state.pages.clear();
        self.spare_tables.push(state.pages);
        Ok(())
    }
}
