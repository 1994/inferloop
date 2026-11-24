//! Inspection responsibilities.
use super::{SequenceState, SequenceStates, StateSnapshot};
use infer_core::{Error, ErrorCode, Result, StateId, StatePageId};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

impl SequenceStates {
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }
    /// Constant-time page counters for scheduling; no reference-table scan.
    #[must_use]
    pub const fn allocated_pages(&self) -> usize {
        self.pages.len()
    }
    #[must_use]
    pub const fn free_pages(&self) -> usize {
        self.total_pages.saturating_sub(self.pages.len())
    }
    /// # Errors
    /// Returns a not-found error for an unknown sequence identity.
    pub fn get(&self, id: StateId) -> Result<&SequenceState> {
        self.sequences
            .get(&id)
            .ok_or_else(|| Error::new(ErrorCode::NotFound, "unknown sequence state"))
    }
    pub fn states(&self) -> impl Iterator<Item = (StateId, &SequenceState)> {
        self.sequences.iter().map(|(id, s)| (*id, s))
    }
    #[must_use]
    pub fn snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            total_pages: self.total_pages,
            allocated_pages: self.pages.len(),
            free_pages: self.total_pages.saturating_sub(self.pages.len()),
            sequence_count: self.sequences.len(),
            cached_prefix_count: self.prefixes.len(),
            references: self.pages.values().map(|p| p.references).sum(),
        }
    }
    /// # Errors
    /// Returns an invariant error for inconsistent identities, ownership, capacity, or execution cursors.
    pub fn check_invariants(&self) -> Result<()> {
        self.ids.validate_after(
            self.pages
                .keys()
                .map(|id| id.get())
                .chain(self.sequences.keys().map(|id| id.get()))
                .max()
                .unwrap_or(0),
        )?;
        if self.block_size == 0 || self.total_pages == 0 || self.pages.len() > self.total_pages {
            return Err(Error::invariant("invalid state capacity"));
        }
        if self.pages.capacity() != self.total_pages
            || self.owners.len() != self.sequences.len()
            || self.owners.capacity() != self.sequences.capacity()
        {
            return Err(Error::invariant(
                "logical page pool ownership dimensions differ",
            ));
        }
        for (id, state) in self.sequences.iter() {
            if self.owners.get(&state.owner) != Some(id) {
                return Err(Error::invariant("state owner index mismatch"));
            }
        }
        let mut actual = BTreeMap::<StatePageId, usize>::new();
        let mut owners = BTreeSet::new();
        for sequence in self.sequences.values() {
            if !owners.insert(sequence.owner)
                || sequence.capacity_tokens == 0
                || sequence.committed_tokens > sequence.capacity_tokens
                || sequence.pages.len() > sequence.capacity_tokens.div_ceil(self.block_size)
                || sequence.committed_tokens > sequence.pages.len() * self.block_size
            {
                return Err(Error::invariant("invalid sequence ownership/cursor"));
            }
            let mut seen = BTreeSet::new();
            for page in &sequence.pages {
                if !seen.insert(page) {
                    return Err(Error::invariant("duplicate sequence page"));
                }
                *actual.entry(*page).or_default() += 1;
            }
        }
        let mut keys = BTreeSet::new();
        for (key, pages) in &self.prefixes {
            if !keys.insert(key)
                || key.tokens.is_empty()
                || key.tokens.len() != pages.len() * self.block_size
            {
                return Err(Error::invariant("invalid prefix index"));
            }
            for page in pages {
                if !self.pages.get(page).is_some_and(|p| p.sealed) {
                    return Err(Error::invariant("prefix page is not sealed"));
                }
                *actual.entry(*page).or_default() += 1;
            }
        }
        if actual.len() != self.pages.len()
            || self
                .pages
                .iter()
                .any(|(id, p)| actual.get(id) != Some(&p.references) || p.references == 0)
        {
            return Err(Error::invariant(
                "allocated pages do not equal owned/shared/cached pages",
            ));
        }
        Ok(())
    }
}
