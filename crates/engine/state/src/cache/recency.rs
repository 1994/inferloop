//! Recency-index representations; lifecycle decisions remain in the shared cache.
use super::Fingerprint;
use std::collections::{BTreeSet, VecDeque};

/// Opaque identity within one cache. It is neither a persisted key nor a device/page lease.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrefixId(pub(super) Fingerprint);

/// Keep insertion/touch order and return the least recently used identity on eviction.
///
/// The cache invokes these methods only for its own identities and after successful reuse.
/// Removing/reinserting a known identity must preserve all other identities' relative order.
pub trait PrefixRecency: Default {
    fn insert(&mut self, id: PrefixId, clock: u64);
    fn remove(&mut self, id: PrefixId, clock: u64);
    fn touch(&mut self, id: PrefixId, previous: u64, clock: u64);
    fn pop_oldest(&mut self) -> Option<PrefixId>;
    fn clear(&mut self);
}

/// Logarithmic recency operations for large block-prefix caches.
#[derive(Default)]
pub struct TreeRecency(BTreeSet<(u64, PrefixId)>);
impl PrefixRecency for TreeRecency {
    fn insert(&mut self, id: PrefixId, clock: u64) {
        self.0.insert((clock, id));
    }
    fn remove(&mut self, id: PrefixId, clock: u64) {
        self.0.remove(&(clock, id));
    }
    fn touch(&mut self, id: PrefixId, previous: u64, clock: u64) {
        self.remove(id, previous);
        self.insert(id, clock);
    }
    fn pop_oldest(&mut self) -> Option<PrefixId> {
        self.0.pop_first().map(|(_, id)| id)
    }
    fn clear(&mut self) {
        self.0.clear();
    }
}

/// Compact recency metadata for a small snapshot set. Touch reuses its retained queue capacity.
#[derive(Default)]
pub struct SnapshotRecency(VecDeque<PrefixId>);
impl PrefixRecency for SnapshotRecency {
    fn insert(&mut self, id: PrefixId, _: u64) {
        self.0.push_back(id);
    }
    fn remove(&mut self, id: PrefixId, _: u64) {
        if let Some(index) = self.0.iter().position(|key| *key == id) {
            self.0.remove(index);
        }
    }
    fn touch(&mut self, id: PrefixId, _: u64, _: u64) {
        self.remove(id, 0);
        self.insert(id, 0);
    }
    fn pop_oldest(&mut self) -> Option<PrefixId> {
        self.0.pop_front()
    }
    fn clear(&mut self) {
        self.0.clear();
    }
}
