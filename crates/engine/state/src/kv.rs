//! Physical KV ownership and prefix lifecycle. Device buffers and GPU copies live in the backend.
use crate::{blocks::BlockLease, blocks::BlockPool, cache::PrefixCache};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::{KvCacheInspection, PageGrowth};
use std::collections::BTreeSet;
mod prefix;

#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    pub namespace: Vec<u8>,
    pub block_size: usize,
    pub blocks: usize,
    pub bytes_per_block: u64,
    pub prefix_bytes: u64,
    pub max_prefixes: usize,
}
/// A backend snapshot includes the KV leases and any recurrent/history/readout state.
/// Its lease references are acquired and released exclusively by the manager.
pub trait KvPrefix: Clone {
    fn tokens(&self) -> &[u32];
    fn blocks(&self) -> &[BlockLease];
    fn bytes(&self) -> u64;
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCopy {
    pub source: BlockLease,
    pub destination: BlockLease,
}
/// Owns the physical page ledger, prefix namespace/index, cache references and eviction policy.
///
/// Active sequence tables hold manager-issued leases. Backends own buffer storage and execute
/// returned copy plans before kernels use new pages. Scheduler queries have no mutation side effects.
pub struct KvCacheManager<P: KvPrefix> {
    config: KvCacheConfig,
    pool: BlockPool,
    prefixes: PrefixCache<P>,
    allocation: Vec<BlockLease>,
}
impl<P: KvPrefix> KvCacheManager<P> {
    /// # Errors
    /// Returns invalid-input errors for an invalid namespace, page layout, or device block ABI.
    pub fn new(config: KvCacheConfig) -> Result<Self> {
        config
            .bytes_per_block
            .checked_mul(config.blocks as u64)
            .ok_or_else(|| Error::invalid("KV pool byte size overflow"))?;
        if config.blocks > 0 && config.bytes_per_block == 0 {
            return Err(Error::invalid("KV blocks require a physical byte layout"));
        }
        let pool = BlockPool::new(config.blocks)?;
        let prefixes = PrefixCache::new(
            &config.namespace,
            config.block_size,
            config.prefix_bytes,
            config.max_prefixes,
        )?;
        let mut allocation = Vec::new();
        allocation
            .try_reserve_exact(config.blocks)
            .map_err(|error| Error::new(ErrorCode::Capacity, error.to_string()))?;
        Ok(Self {
            config,
            pool,
            prefixes,
            allocation,
        })
    }
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.pool.capacity()
    }
    #[must_use]
    pub fn free_blocks(&self) -> usize {
        self.pool.free_blocks()
    }
    #[must_use]
    pub fn prefix_count(&self) -> usize {
        self.prefixes.len()
    }
    #[must_use]
    pub const fn prefix_bytes(&self) -> u64 {
        self.prefixes.bytes()
    }
    #[must_use]
    pub fn contains_prefix(&self, tokens: &[u32]) -> bool {
        self.prefixes.contains(tokens)
    }
    #[must_use]
    pub fn contains_prefix_where(&self, tokens: &[u32], compatible: impl Fn(&P) -> bool) -> bool {
        self.prefixes.contains_where(tokens, compatible)
    }
    #[must_use]
    pub fn matched_tokens(&self, tokens: &[u32], maximum: usize) -> usize {
        self.prefixes.matched_tokens(tokens, maximum)
    }
    #[must_use]
    pub fn matched_tokens_where(
        &self,
        tokens: &[u32],
        maximum: usize,
        compatible: impl Fn(&P) -> bool,
    ) -> usize {
        self.prefixes
            .matched_tokens_where(tokens, maximum, compatible)
    }
    /// # Errors
    /// Returns stale-lease, reference overflow, or duplicate-table errors.
    pub fn retain(&mut self, leases: &[BlockLease]) -> Result<()> {
        self.pool.retain(leases)
    }
    /// # Errors
    /// Returns stale-lease, reference underflow, or duplicate-table errors.
    pub fn release(&mut self, leases: &[BlockLease]) -> Result<()> {
        self.pool.release(leases)
    }
    /// # Errors
    /// Returns an invalid-input error for stale or unowned page leases.
    pub fn references(&self, lease: BlockLease) -> Result<usize> {
        self.pool.references(lease)
    }
    /// Acquire a second active owner without copying immutable/full pages.
    /// # Errors
    /// Returns duplicate-table, stale-lease, or reference overflow errors.
    pub fn fork_pages(&mut self, source: &[BlockLease]) -> Result<Vec<BlockLease>> {
        self.pool.retain(source)?;
        Ok(source.to_vec())
    }
    /// Protect shared original tails until an encoding batch commits or rolls back.
    /// # Errors
    /// Returns cursor, stale-lease, or reference overflow errors without retaining a partial set.
    pub fn pin_shared_tails<'a>(
        &mut self,
        tables: impl Iterator<Item = (&'a [BlockLease], usize)>,
    ) -> Result<Vec<BlockLease>> {
        let mut tails = BTreeSet::new();
        for (table, committed) in tables {
            if self.page_growth(table, committed)?.cow_tail
                && let Some(tail) = table.last()
            {
                tails.insert(*tail);
            }
        }
        let pins: Vec<_> = tails.into_iter().collect();
        self.pool.retain(&pins)?;
        Ok(pins)
    }
    /// Restore a protected table after an uncommitted append and release its new pages.
    /// # Errors
    /// Returns stale-lease or ownership errors without dropping the original references.
    pub fn rollback_append(
        &mut self,
        table: &mut Vec<BlockLease>,
        original: Vec<BlockLease>,
    ) -> Result<()> {
        self.pool.retain(&original)?;
        if let Err(error) = self.pool.release(table) {
            self.pool.release(&original)?;
            return Err(error);
        }
        *table = original;
        Ok(())
    }
    /// Allocate pages after reclaiming only cache-owned references.
    /// # Errors
    /// Returns capacity errors when active owners pin all usable pages, or ownership errors.
    pub fn allocate(&mut self, count: usize) -> Result<Vec<BlockLease>> {
        self.reclaim(count)?;
        self.pool.allocate(count)
    }
    /// # Errors
    /// Returns capacity errors when no reclaimable pages remain, or stale-cache ownership errors.
    pub fn reclaim(&mut self, count: usize) -> Result<()> {
        if count > self.pool.capacity() {
            return Err(Error::new(
                ErrorCode::Capacity,
                "KV growth exceeds total pool capacity",
            ));
        }
        while self.pool.free_blocks() < count {
            let entry = self.prefixes.evict_oldest().ok_or_else(|| {
                Error::new(
                    ErrorCode::Capacity,
                    "active KV owners exhaust physical pool",
                )
            })?;
            self.pool.release(entry.blocks())?;
        }
        Ok(())
    }
    /// Scheduler-facing growth estimate including a shared partial-tail copy.
    /// # Errors
    /// Returns invalid-input errors for a table/cursor mismatch or stale page leases.
    pub fn page_growth(&self, table: &[BlockLease], committed: usize) -> Result<PageGrowth> {
        if table.len() != committed.div_ceil(self.config.block_size) {
            return Err(Error::invalid("KV table/cursor mismatch"));
        }
        let tail_references = table
            .last()
            .map(|lease| self.pool.references(*lease))
            .transpose()?;
        let cow_tail = !committed.is_multiple_of(self.config.block_size)
            && tail_references.is_some_and(|references| references > 1);
        Ok(PageGrowth {
            block_size: self.config.block_size,
            allocated_pages: table.len(),
            bytes_per_page: self.config.bytes_per_block,
            cow_tail,
        })
    }
    /// Reserve append pages and return the device copy required for a shared tail.
    /// The caller must keep the old table until its encoding transaction commits.
    /// # Errors
    /// Returns invalid cursor, capacity, generation, or ownership errors.
    pub fn prepare_append(
        &mut self,
        table: &mut Vec<BlockLease>,
        committed: usize,
        end: usize,
    ) -> Result<Option<KvCopy>> {
        if end < committed {
            return Err(Error::invalid("KV append cursor moved backwards"));
        }
        if end == committed {
            self.page_growth(table, committed)?;
            return Ok(None);
        }
        let growth = self.page_growth(table, committed)?;
        let required = growth
            .required_pages(end)
            .ok_or_else(|| Error::invalid("KV growth overflow"))?;
        let count = end
            .div_ceil(self.config.block_size)
            .saturating_sub(table.len());
        table
            .try_reserve(count)
            .map_err(|error| Error::new(ErrorCode::Capacity, error.to_string()))?;
        self.reclaim(required)?;
        // Cache reclamation may have removed the other owner of a shared tail.
        let cow = growth.cow_tail
            && table
                .last()
                .is_some_and(|l| self.pool.references(*l).is_ok_and(|n| n > 1));
        self.allocation.clear();
        self.pool
            .allocate_into(count + usize::from(cow), &mut self.allocation)?;
        let mut allocated = self.allocation.drain(..);
        let copy = if cow {
            let destination = allocated
                .next()
                .ok_or_else(|| Error::invariant("reserved COW page missing"))?;
            let source = *table
                .last()
                .ok_or_else(|| Error::invariant("validated COW tail missing"))?;
            self.pool.release(&[source])?;
            let tail = table
                .last_mut()
                .ok_or_else(|| Error::invariant("validated COW tail missing"))?;
            *tail = destination;
            self.pool.cow_copies = self.pool.cow_copies.saturating_add(1);
            Some(KvCopy {
                source,
                destination,
            })
        } else {
            None
        };
        table.extend(allocated);
        Ok(copy)
    }
    /// Validate all active and cached owners against the physical page ledger.
    /// # Errors
    /// Returns an invariant error for duplicate ownership, stale generations, or untracked references.
    pub fn check_owners<'a>(
        &'a self,
        active: impl Iterator<Item = &'a [BlockLease]>,
    ) -> Result<()> {
        self.pool
            .check_owners(active.chain(self.prefixes.values().map(KvPrefix::blocks)))
    }
    #[must_use]
    pub fn inspect<'a>(&self, active: impl Iterator<Item = &'a [BlockLease]>) -> KvCacheInspection {
        self.inspect_with_pins(active, &[])
    }
    #[must_use]
    pub fn inspect_with_pins<'a>(
        &self,
        active: impl Iterator<Item = &'a [BlockLease]>,
        pins: &[BlockLease],
    ) -> KvCacheInspection {
        let active: BTreeSet<_> = active.flatten().map(|l| l.index).collect();
        let cached: BTreeSet<_> = self
            .prefixes
            .values()
            .flat_map(KvPrefix::blocks)
            .map(|l| l.index)
            .collect();
        let shared = active
            .iter()
            .filter(|index| self.pool.reference_count(**index).is_some_and(|n| n > 1))
            .count();
        let pinned: BTreeSet<_> = pins.iter().map(|lease| lease.index).collect();
        KvCacheInspection {
            pinned_blocks: pinned.len(),
            block_size: self.config.block_size,
            total_blocks: self.pool.capacity(),
            free_blocks: self.pool.free_blocks(),
            active_blocks: active.len(),
            cached_blocks: cached.len(),
            shared_blocks: shared,
            available_blocks: self.pool.free_blocks()
                + cached
                    .difference(&active)
                    .filter(|index| !pinned.contains(index))
                    .count(),
            pool_bytes: self.config.bytes_per_block * self.pool.capacity() as u64,
            bytes_per_block: self.config.bytes_per_block,
            allocations: self.pool.allocations,
            releases: self.pool.releases,
            cow_copies: self.pool.cow_copies,
            cache_evictions: self.prefixes.evictions,
            reused_tokens: self.prefixes.reused_tokens,
        }
    }
}
