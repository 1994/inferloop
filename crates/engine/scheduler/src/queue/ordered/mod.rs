//! Library B-trees backed by reusable slab nodes; no local balancing algorithm.
use btree_slab::generic::map::BTreeExt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result;
use std::iter::from_fn;
use std::ops::{Bound, RangeBounds};

pub(super) struct OrderedSet<K> {
    tree: btree_slab::BTreeMap<K, ()>,
    capacity: usize,
    seed: fn(usize) -> K,
}
impl<K: Copy + Ord> OrderedSet<K> {
    pub fn new(capacity: usize, seed: fn(usize) -> K) -> Self {
        let mut tree = btree_slab::BTreeMap::new();
        // Pinned btree-slab 0.6.1 has order 8: non-root nodes have at least 3
        // keys. Sequentially inserting twice the item limit provisions more
        // slab slots than any valid tree at the limit, including a split.
        // clear retains the slab allocation; leaf/branch SmallVecs stay inline.
        for slot in 0..capacity * 2 {
            tree.insert(seed(slot), ());
        }
        tree.clear();
        Self {
            tree,
            capacity,
            seed,
        }
    }
    pub fn insert(&mut self, key: K) -> bool {
        if self.tree.len() == self.capacity {
            return false;
        }
        self.tree.insert(key, ()).is_none()
    }
    pub fn remove(&mut self, key: &K) -> bool {
        self.tree.remove(key).is_some()
    }
    pub fn first(&self) -> Option<&K> {
        self.tree.first_key_value().map(|(key, ())| key)
    }
    pub fn len(&self) -> usize {
        self.tree.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = &K> {
        self.tree.keys()
    }
    pub fn range(&self, range: impl RangeBounds<K>) -> impl Iterator<Item = &K> {
        // 0.6.1's Range uses the first address for an unbounded end and does
        // not normalize missing-key insertion addresses. Use its public cursor
        // API instead; addresses never survive a mutation or leave this borrow.
        let mut next = match range.start_bound() {
            Bound::Unbounded => self.tree.first_item_address(),
            Bound::Included(key) => match self.tree.address_of(key) {
                Ok(address) => Some(address),
                Err(address) => self.tree.normalize(address),
            },
            Bound::Excluded(key) => match self.tree.address_of(key) {
                Ok(address) => self.tree.next_item_address(address),
                Err(address) => self.tree.normalize(address),
            },
        };
        let end = range.end_bound().cloned();
        from_fn(move || {
            let address = next?;
            let key = self.tree.item(address)?.key();
            let within = match end {
                Bound::Unbounded => true,
                Bound::Included(end) => *key <= end,
                Bound::Excluded(end) => *key < end,
            };
            next = if within {
                self.tree.next_item_address(address)
            } else {
                None
            };
            within.then_some(key)
        })
    }
}
impl<K: Copy + Ord> Clone for OrderedSet<K> {
    fn clone(&self) -> Self {
        let mut cloned = Self::new(self.capacity, self.seed);
        for key in self.tree.keys() {
            cloned.tree.insert(*key, ());
        }
        cloned
    }
}
impl<K: Copy + Ord> PartialEq for OrderedSet<K> {
    fn eq(&self, other: &Self) -> bool {
        self.capacity == other.capacity && self.tree.keys().eq(other.tree.keys())
    }
}
impl<K: Copy + Ord> Eq for OrderedSet<K> {}

impl<K: Copy + Ord + Debug> Debug for OrderedSet<K> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result {
        formatter.debug_set().entries(self.tree.keys()).finish()
    }
}
#[cfg(test)]
mod tests;
