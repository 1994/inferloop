//! Prefix responsibilities.
use super::{PrefixKey, SequenceStates};
use infer_core::{Error, Result, StateId};

impl SequenceStates {
    /// Cache only fully committed pages; a mutable partial tail is never shared.
    ///
    /// # Errors
    /// Returns an invalid-input error for partial or uncommitted token blocks, or an invariant error for invalid page ownership.
    pub fn cache_prefix(&mut self, id: StateId, key: PrefixKey) -> Result<()> {
        let state = self.get(id)?;
        if key.tokens.is_empty()
            || !key.tokens.len().is_multiple_of(self.block_size)
            || key.tokens.len() > state.committed_tokens
        {
            return Err(Error::invalid("prefix must contain committed full pages"));
        }
        if self.prefixes.iter().any(|(k, _)| k == &key) {
            return Ok(());
        }
        let pages = state.pages[..key.tokens.len() / self.block_size].to_vec();
        for page in &pages {
            let p = self
                .pages
                .get(page)
                .ok_or_else(|| Error::invariant("prefix page missing"))?;
            if p.references == usize::MAX {
                return Err(Error::invariant("page reference overflow"));
            }
        }
        for page in &pages {
            let p = self
                .pages
                .get_mut(page)
                .ok_or_else(|| Error::invariant("owned page disappeared"))?;
            p.references += 1;
            p.sealed = true;
        }
        self.prefixes.push((key, pages));
        Ok(())
    }
    pub fn evict_prefix(&mut self, key: &PrefixKey) -> bool {
        if let Some(index) = self.prefixes.iter().position(|(k, _)| k == key) {
            let (_, pages) = self.prefixes.remove(index);
            self.unreference(&pages);
            true
        } else {
            false
        }
    }
}
