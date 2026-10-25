//! Page-backed adapter for the common prefix publication/reuse lifecycle.
use super::{KvCacheManager, KvPrefix};
use crate::{
    blocks::BlockPool,
    cache::{PrefixReuse, PrefixSnapshot},
};
use infer_core::{Error, Result};

struct PageReuse<'a, F> {
    pool: &'a mut BlockPool,
    compatible: F,
}
impl<P: KvPrefix, F: Fn(&P) -> bool> PrefixReuse<P> for PageReuse<'_, F> {
    type Output = P;
    type Error = Error;
    fn compatible(&self, _: &[u32], prefix: &P) -> bool {
        (self.compatible)(prefix)
    }
    fn reuse(&mut self, _: &[u32], prefix: &P) -> Result<P> {
        let acquired = prefix.clone();
        self.pool.retain(acquired.blocks())?;
        Ok(acquired)
    }
}
impl<P: KvPrefix> KvCacheManager<P> {
    /// Acquire only snapshots compatible with the consumer's recurrent/readout layout.
    /// # Errors
    /// Returns stale-lease or reference overflow errors.
    pub fn attach_prefix_where(
        &mut self,
        tokens: &[u32],
        maximum: usize,
        compatible: impl Fn(&P) -> bool,
    ) -> Result<Option<P>> {
        self.prefixes.reuse(
            tokens,
            maximum,
            &mut PageReuse {
                pool: &mut self.pool,
                compatible,
            },
        )
    }
    /// Acquire active ownership before returning a cached prefix to a sequence.
    /// # Errors
    /// Returns stale-cache or reference overflow errors; a miss returns None.
    pub fn attach_prefix(&mut self, tokens: &[u32], maximum: usize) -> Result<Option<P>> {
        self.attach_prefix_where(tokens, maximum, |_| true)
    }
    /// Publish a completed device snapshot and acquire its cache page references atomically.
    /// # Errors
    /// Returns invalid-prefix or ownership errors. A prefix exceeding the cache budget is declined.
    pub fn publish_prefix(&mut self, prefix: P) -> Result<bool> {
        let tokens = prefix.tokens();
        if tokens.is_empty()
            || !tokens.len().is_multiple_of(self.config.block_size)
            || prefix.blocks().len() != tokens.len() / self.config.block_size
        {
            return Err(Error::invalid(
                "KV prefix must own exactly its complete pages",
            ));
        }
        if !self.prefixes.can_publish(tokens, prefix.bytes())? {
            return Ok(false);
        }
        let tokens = tokens.to_vec();
        let bytes = prefix.bytes();
        let leases = prefix.blocks().to_vec();
        self.pool.retain(&leases)?;
        let removed = match self.prefixes.publish(PrefixSnapshot {
            tokens,
            value: prefix,
            bytes,
        }) {
            Ok(removed) => removed,
            Err(error) => {
                self.pool.release(&leases)?;
                return Err(error);
            }
        };
        for entry in removed {
            self.pool.release(entry.blocks())?;
        }
        Ok(true)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/prefix_pages.rs"]
mod tests;
