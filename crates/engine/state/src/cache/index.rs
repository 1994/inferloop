//! Interchangeable lookup strategies over the same validated snapshots and lifecycle.
use super::{Entry, FINGERPRINT_LANES, Fingerprint, PrefixRecency, SnapshotRecency, TreeRecency};
use ahash::{AHashMap, AHasher};
use std::hash::{Hash, Hasher};

/// A candidate borrowed from the cache. Its private identity cannot be fabricated by a matcher.
pub struct PrefixCandidate<'a, V> {
    pub tokens: &'a [u32],
    pub value: &'a V,
    pub(super) key: Fingerprint,
}

/// Read-only search access. Both candidate iterators enforce maximum, exact prefix equality and
/// the backend's compatibility predicate before exposing a snapshot to a lookup strategy.
pub struct PrefixSearch<'a, V, F> {
    pub(super) entries: &'a AHashMap<Fingerprint, Entry<V>>,
    pub(super) root: [AHasher; FINGERPRINT_LANES],
    pub(super) block_size: usize,
    pub(super) tokens: &'a [u32],
    pub(super) maximum: usize,
    pub(super) compatible: F,
}
impl<'a, V, F: Fn(&[u32], &V) -> bool> PrefixSearch<'a, V, F> {
    fn candidate(&self, key: Fingerprint, entry: &'a Entry<V>) -> Option<PrefixCandidate<'a, V>> {
        (entry.tokens.len() <= self.maximum
            && self.tokens.starts_with(&entry.tokens)
            && (self.compatible)(&entry.tokens, &entry.value))
        .then_some(PrefixCandidate {
            tokens: &entry.tokens,
            value: &entry.value,
            key,
        })
    }
    /// Enumerate compatible snapshots without hashing the query. Useful for few large snapshots.
    pub fn snapshots(&self) -> impl Iterator<Item = PrefixCandidate<'a, V>> + '_ {
        self.entries
            .iter()
            .filter_map(|(key, entry)| self.candidate(*key, entry))
    }
    /// Enumerate compatible block-index hits in increasing covered-token order.
    pub fn blocks(&self) -> impl Iterator<Item = PrefixCandidate<'a, V>> + '_ {
        let end = self.tokens.len().min(self.maximum) / self.block_size * self.block_size;
        keys(self.root.clone(), self.block_size, &self.tokens[..end]).filter_map(|(length, key)| {
            let entry = self.entries.get(&key)?;
            if entry.tokens.len() != length {
                return None;
            }
            self.candidate(key, entry)
        })
    }
}
/// Select the longest compatible candidate. Policies customize search, while publication,
/// equality/compatibility guards, resource reuse, accounting and LRU stay in the shared cache.
pub trait PrefixPolicy {
    type Recency: PrefixRecency;
    fn longest<'a, V: 'a, F: Fn(&[u32], &V) -> bool>(
        &self,
        search: PrefixSearch<'a, V, F>,
    ) -> Option<PrefixCandidate<'a, V>>;
}
/// Incremental fingerprint lookup for a cache containing many block-aligned prefixes.
#[derive(Clone, Copy, Default)]
pub struct BlockHash;
impl PrefixPolicy for BlockHash {
    type Recency = TreeRecency;
    fn longest<'a, V: 'a, F: Fn(&[u32], &V) -> bool>(
        &self,
        search: PrefixSearch<'a, V, F>,
    ) -> Option<PrefixCandidate<'a, V>> {
        search.blocks().last()
    }
}
/// Scan the small snapshot set directly, avoiding a full query hash on an early-token miss.
#[derive(Clone, Copy, Default)]
pub struct SnapshotScan;
impl PrefixPolicy for SnapshotScan {
    type Recency = SnapshotRecency;
    fn longest<'a, V: 'a, F: Fn(&[u32], &V) -> bool>(
        &self,
        search: PrefixSearch<'a, V, F>,
    ) -> Option<PrefixCandidate<'a, V>> {
        search
            .snapshots()
            .max_by_key(|candidate| candidate.tokens.len())
    }
}
pub(super) fn keys(
    mut hashers: [AHasher; FINGERPRINT_LANES],
    block_size: usize,
    tokens: &[u32],
) -> impl Iterator<Item = (usize, Fingerprint)> {
    let mut length = 0;
    tokens.chunks(block_size).map(move |block| {
        length += block.len();
        for hasher in &mut hashers {
            block.hash(hasher);
        }
        (length, hashers.each_ref().map(Hasher::finish))
    })
}
