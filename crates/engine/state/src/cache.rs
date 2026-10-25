//! Shared prefix publication, matching and deterministic LRU. Values own device/host state;
//! callers release their page leases when entries are evicted.
mod index;
mod recency;
mod reuse;
use ahash::{AHashMap, AHasher, RandomState};
pub use index::{BlockHash, PrefixCandidate, PrefixPolicy, PrefixSearch, SnapshotScan};
use infer_core::{Error, Result};
pub use recency::{PrefixId, PrefixRecency, SnapshotRecency, TreeRecency};
use reuse::CloneReuse;
pub use reuse::{PrefixReuse, PrefixSnapshot};
use std::array::from_fn;
use std::hash::{BuildHasher, Hash};
use std::result::Result as StdResult;

/// Independently seeded hash lanes keep the process-local prefix index at 128 bits.
const FINGERPRINT_LANES: usize = 2;
type Fingerprint = [u64; FINGERPRINT_LANES];
struct Entry<V> {
    tokens: Vec<u32>,
    value: V,
    bytes: u64,
    touched: u64,
}
pub struct PrefixCache<V, M: PrefixPolicy = BlockHash> {
    policy: M,
    root: [AHasher; FINGERPRINT_LANES],
    block_size: usize,
    budget: u64,
    max_entries: usize,
    // Fingerprints select candidates; full token equality establishes identity on every hit.
    entries: AHashMap<Fingerprint, Entry<V>>,
    lru: M::Recency,
    clock: u64,
    bytes: u64,
    cached_tokens: usize,
    // A conservative lower bound for resident prefix lengths. Eviction may make it stale
    // downward, which only loses a shortcut; reset it when the store becomes empty.
    minimum_tokens_hint: usize,
    /// Reuse queries submitted to this cache, including misses and backend failures.
    pub lookups: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub evictions: u64,
}
impl<V> PrefixCache<V> {
    /// # Errors
    /// Rejects an empty namespace, zero block size or zero entry limit.
    pub fn new(binding: &[u8], block_size: usize, budget: u64, max_entries: usize) -> Result<Self> {
        Self::with_policy(binding, block_size, budget, max_entries, BlockHash)
    }
}
impl<V, M: PrefixPolicy> PrefixCache<V, M> {
    /// Select matching and recency indexes without duplicating lifecycle or resource policy.
    /// # Errors
    /// Rejects an empty namespace, zero block size or zero entry limit.
    pub fn with_policy(
        binding: &[u8],
        block_size: usize,
        budget: u64,
        max_entries: usize,
        policy: M,
    ) -> Result<Self> {
        if binding.is_empty() || block_size == 0 || max_entries == 0 {
            return Err(Error::invalid("invalid prefix cache namespace/budget"));
        }
        let mut root: [AHasher; FINGERPRINT_LANES] = from_fn(|_| RandomState::new().build_hasher());
        for hasher in &mut root {
            ("kv-prefix-local-v1", binding, block_size).hash(hasher);
        }
        Ok(Self {
            policy,
            root,
            block_size,
            budget,
            max_entries,
            entries: AHashMap::new(),
            lru: M::Recency::default(),
            clock: 0,
            bytes: 0,
            cached_tokens: 0,
            minimum_tokens_hint: usize::MAX,
            lookups: 0,
            hits: 0,
            reused_tokens: 0,
            evictions: 0,
        })
    }
    // Both publication and lookup hash the same block boundaries. The state stays local to this
    // instance: fingerprints are neither persisted nor used as proof of token identity.
    fn keys(&self, tokens: &[u32]) -> impl Iterator<Item = (usize, Fingerprint)> {
        index::keys(self.root.clone(), self.block_size, tokens)
    }
    fn key(&self, tokens: &[u32]) -> Fingerprint {
        self.keys(tokens)
            .last()
            .map_or([0; FINGERPRINT_LANES], |(_, key)| key)
    }
    fn tick(&mut self) -> u64 {
        if self.clock == u64::MAX {
            self.lru.clear();
            let mut order: Vec<_> = self.entries.iter().map(|(k, e)| (e.touched, *k)).collect();
            order.sort_unstable();
            for (i, (_, key)) in order.into_iter().enumerate() {
                let t = i as u64;
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.touched = t;
                    self.lru.insert(PrefixId(key), t);
                }
            }
            self.clock = self.entries.len() as u64;
        }
        self.clock += 1;
        self.clock
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    #[must_use]
    pub const fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }
    /// Cached values in arbitrary order; eviction order is maintained by the LRU index.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.values().map(|e| &e.value)
    }
    #[must_use]
    pub fn contains(&self, tokens: &[u32]) -> bool {
        self.entries
            .get(&self.key(tokens))
            .is_some_and(|e| e.tokens == tokens)
    }
    #[must_use]
    pub fn contains_where(&self, tokens: &[u32], compatible: impl Fn(&V) -> bool) -> bool {
        self.entries
            .get(&self.key(tokens))
            .is_some_and(|entry| entry.tokens == tokens && compatible(&entry.value))
    }
    #[must_use]
    pub fn matched_tokens(&self, tokens: &[u32], maximum: usize) -> usize {
        self.matched_tokens_where(tokens, maximum, |_| true)
    }
    pub fn matched_tokens_where(
        &self,
        tokens: &[u32],
        maximum: usize,
        compatible: impl Fn(&V) -> bool,
    ) -> usize {
        self.matched_prefix(tokens, maximum, |_, value| compatible(value))
            .map_or(0, |candidate| candidate.tokens.len())
    }
    /// Select the longest compatible complete prefix through the configured matcher.
    /// Shared candidate access guards maximum, exact token identity and compatibility.
    fn matched_prefix<'a>(
        &'a self,
        tokens: &'a [u32],
        maximum: usize,
        compatible: impl Fn(&[u32], &V) -> bool,
    ) -> Option<PrefixCandidate<'a, V>> {
        if tokens.len().min(maximum) < self.minimum_tokens_hint {
            return None;
        }
        self.policy.longest(PrefixSearch {
            entries: &self.entries,
            root: self.root.clone(),
            block_size: self.block_size,
            tokens,
            maximum,
            compatible,
        })
    }
    pub fn lookup(&mut self, tokens: &[u32], maximum: usize) -> Option<V>
    where
        V: Clone,
    {
        self.lookup_where(tokens, maximum, |_| true)
    }
    pub fn lookup_where(
        &mut self,
        tokens: &[u32],
        maximum: usize,
        compatible: impl Fn(&V) -> bool,
    ) -> Option<V>
    where
        V: Clone,
    {
        match self.reuse(tokens, maximum, &mut CloneReuse(compatible)) {
            Ok(value) => value,
            Err(never) => match never {},
        }
    }
    /// Match the longest compatible complete prefix and let the backend acquire or restore it.
    /// A failed hook does not count as a hit, touch the LRU, or remove the source snapshot.
    /// # Errors
    /// Propagates the backend error after its rollback/poison handling.
    pub fn reuse<R: PrefixReuse<V>>(
        &mut self,
        tokens: &[u32],
        maximum: usize,
        backend: &mut R,
    ) -> StdResult<Option<R::Output>, R::Error> {
        self.lookups = self.lookups.saturating_add(1);
        let Some(candidate) = self.matched_prefix(tokens, maximum, |prefix, value| {
            backend.compatible(prefix, value)
        }) else {
            return Ok(None);
        };
        let length = candidate.tokens.len();
        let key = candidate.key;
        let output = backend.reuse(candidate.tokens, candidate.value)?;
        let tick = self.tick();
        // Only an immutable payload was lent to the hook; it cannot remove this entry.
        if let Some(entry) = self.entries.get_mut(&key) {
            self.lru.touch(PrefixId(key), entry.touched, tick);
            entry.touched = tick;
        }
        self.hits = self.hits.saturating_add(1);
        self.reused_tokens = self.reused_tokens.saturating_add(length as u64);
        Ok(Some(output))
    }
    /// Check publication policy before a backend acquires any additional resource references.
    /// # Errors
    /// Rejects empty or incomplete prefixes.
    pub fn can_publish(&self, tokens: &[u32], bytes: u64) -> Result<bool> {
        if tokens.is_empty() || !tokens.len().is_multiple_of(self.block_size) {
            return Err(Error::invalid(
                "only complete token blocks may be published",
            ));
        }
        Ok(self.budget != 0 && bytes <= self.budget)
    }
    /// Publish a backend's completed snapshot through the same budget and eviction policy.
    /// Returned values own the rejected/replaced/evicted backend storage for caller cleanup.
    /// # Errors
    /// Rejects empty or incomplete prefixes, or inconsistent eviction metadata.
    pub fn publish(&mut self, snapshot: PrefixSnapshot<V>) -> Result<Vec<V>> {
        self.insert(snapshot.tokens, snapshot.value, snapshot.bytes)
    }
    pub fn evict_oldest(&mut self) -> Option<V> {
        let PrefixId(key) = self.lru.pop_oldest()?;
        let e = self.entries.remove(&key)?;
        self.bytes -= e.bytes;
        self.cached_tokens -= e.tokens.len();
        if self.entries.is_empty() {
            self.minimum_tokens_hint = usize::MAX;
        }
        self.evictions = self.evictions.saturating_add(1);
        Some(e.value)
    }
    /// # Errors
    /// Returns an invalid-input error for empty or partial token blocks, or an invariant error if eviction metadata is inconsistent.
    pub fn insert(&mut self, tokens: Vec<u32>, value: V, bytes: u64) -> Result<Vec<V>> {
        let mut removed = vec![];
        if !self.can_publish(&tokens, bytes)? {
            removed.push(value);
            return Ok(removed);
        }
        let key = self.key(&tokens);
        if let Some(old) = self.entries.remove(&key) {
            self.lru.remove(PrefixId(key), old.touched);
            self.bytes -= old.bytes;
            self.cached_tokens -= old.tokens.len();
            removed.push(old.value);
        }
        if self.entries.is_empty() {
            self.minimum_tokens_hint = usize::MAX;
        }
        while self.entries.len() >= self.max_entries || self.bytes > self.budget - bytes {
            removed.push(
                self.evict_oldest().ok_or_else(|| {
                    Error::invariant("prefix cache eviction index is inconsistent")
                })?,
            );
        }
        let touched = self.tick();
        self.bytes += bytes;
        self.cached_tokens += tokens.len();
        self.minimum_tokens_hint = self.minimum_tokens_hint.min(tokens.len());
        self.entries.insert(
            key,
            Entry {
                tokens,
                value,
                bytes,
                touched,
            },
        );
        self.lru.insert(PrefixId(key), touched);
        Ok(removed)
    }
}

#[cfg(test)]
#[path = "../tests/unit/cache.rs"]
mod tests;
