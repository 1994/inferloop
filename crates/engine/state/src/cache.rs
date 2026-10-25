//! Hash-chain block index and deterministic LRU. Values own device/host state;
//! callers release their page leases when entries are evicted.
use ahash::{AHashMap, AHasher, RandomState};
use infer_core::{Error, Result};
use std::array::from_fn;
use std::collections::BTreeSet;
use std::hash::{BuildHasher, Hash, Hasher};

/// Independently seeded hash lanes keep the process-local prefix index at 128 bits.
const FINGERPRINT_LANES: usize = 2;
type Fingerprint = [u64; FINGERPRINT_LANES];
struct Entry<V> {
    tokens: Vec<u32>,
    value: V,
    bytes: u64,
    touched: u64,
}
pub struct PrefixCache<V> {
    root: [AHasher; FINGERPRINT_LANES],
    block_size: usize,
    budget: u64,
    max_entries: usize,
    // Fingerprints select candidates; full token equality establishes identity on every hit.
    entries: AHashMap<Fingerprint, Entry<V>>,
    lru: BTreeSet<(u64, Fingerprint)>,
    clock: u64,
    bytes: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub evictions: u64,
}
impl<V> PrefixCache<V> {
    /// # Errors
    /// Returns an invalid-input error for an empty namespace, zero page size, or zero entry limit.
    pub fn new(binding: &[u8], block_size: usize, budget: u64, max_entries: usize) -> Result<Self> {
        if binding.is_empty() || block_size == 0 || max_entries == 0 {
            return Err(Error::invalid("invalid prefix cache namespace/budget"));
        }
        let mut root: [AHasher; FINGERPRINT_LANES] = from_fn(|_| RandomState::new().build_hasher());
        for hasher in &mut root {
            ("kv-prefix-local-v1", binding, block_size).hash(hasher);
        }
        Ok(Self {
            root,
            block_size,
            budget,
            max_entries,
            entries: AHashMap::new(),
            lru: BTreeSet::new(),
            clock: 0,
            bytes: 0,
            hits: 0,
            reused_tokens: 0,
            evictions: 0,
        })
    }
    // Both publication and lookup hash the same block boundaries. The state stays local to this
    // instance: fingerprints are neither persisted nor used as proof of token identity.
    fn keys(&self, tokens: &[u32]) -> impl Iterator<Item = (usize, Fingerprint)> {
        let mut hashers = self.root.clone();
        let mut length = 0;
        tokens.chunks(self.block_size).map(move |block| {
            length += block.len();
            for hasher in &mut hashers {
                block.hash(hasher);
            }
            (length, hashers.each_ref().map(Hasher::finish))
        })
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
                    self.lru.insert((t, key));
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
        self.matched_prefix(tokens, maximum, compatible).0
    }
    /// Longest cached block-aligned prefix paired with its process-local fingerprint. Hash
    /// states advance once per block; exact token comparison guards every candidate hit.
    fn matched_prefix(
        &self,
        tokens: &[u32],
        maximum: usize,
        compatible: impl Fn(&V) -> bool,
    ) -> (usize, Fingerprint) {
        let end = tokens.len().min(maximum) / self.block_size * self.block_size;
        let mut matched = (0, [0; FINGERPRINT_LANES]);
        for (length, key) in self.keys(&tokens[..end]) {
            if self
                .entries
                .get(&key)
                .is_some_and(|e| e.tokens == tokens[..length] && compatible(&e.value))
            {
                matched = (length, key);
            }
        }
        matched
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
        let (length, key) = self.matched_prefix(tokens, maximum, compatible);
        if length == 0 {
            return None;
        }
        let tick = self.tick();
        let e = self.entries.get_mut(&key)?;
        self.lru.remove(&(e.touched, key));
        e.touched = tick;
        self.lru.insert((tick, key));
        self.hits = self.hits.saturating_add(1);
        self.reused_tokens = self.reused_tokens.saturating_add(length as u64);
        Some(e.value.clone())
    }
    pub fn evict_oldest(&mut self) -> Option<V> {
        let (_, key) = self.lru.pop_first()?;
        let e = self.entries.remove(&key)?;
        self.bytes -= e.bytes;
        self.evictions = self.evictions.saturating_add(1);
        Some(e.value)
    }
    /// # Errors
    /// Returns an invalid-input error for empty or partial token blocks, or an invariant error if eviction metadata is inconsistent.
    pub fn insert(&mut self, tokens: Vec<u32>, value: V, bytes: u64) -> Result<Vec<V>> {
        if tokens.is_empty() || !tokens.len().is_multiple_of(self.block_size) {
            return Err(Error::invalid(
                "only complete token blocks may be published",
            ));
        }
        let mut removed = vec![];
        if bytes > self.budget || self.budget == 0 {
            removed.push(value);
            return Ok(removed);
        }
        let key = self.key(&tokens);
        if let Some(old) = self.entries.remove(&key) {
            self.lru.remove(&(old.touched, key));
            self.bytes -= old.bytes;
            removed.push(old.value);
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
        self.entries.insert(
            key,
            Entry {
                tokens,
                value,
                bytes,
                touched,
            },
        );
        self.lru.insert((touched, key));
        Ok(removed)
    }
}

#[cfg(test)]
#[path = "../tests/unit/cache.rs"]
mod tests;
