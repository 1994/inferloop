//! Fixed-capacity membership without allocation on insertion/removal.
use crate::{Result, map::BoundedMap};
use std::hash::Hash;
#[derive(Clone)]
pub struct BoundedSet<K> {
    entries: BoundedMap<K, ()>,
}
impl<K: Eq + Hash> BoundedSet<K> {
    /// # Errors
    /// Rejects invalid dimensions or failure to allocate the fixed lookup table.
    pub fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            entries: BoundedMap::new(capacity)?,
        })
    }
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }
    /// # Errors
    /// Returns capacity when adding a new member to a full set; existing members are unchanged.
    pub fn insert(&mut self, key: K) -> Result<bool> {
        if self.contains(&key) {
            return Ok(false);
        }
        self.entries.insert(key, ())?;
        Ok(true)
    }
    pub fn remove(&mut self, key: &K) -> bool {
        self.entries.remove(key).is_some()
    }
    pub fn iter(&self) -> impl Iterator<Item = &K> {
        self.entries.iter().map(|(key, ())| key)
    }
}
impl<K: Eq + Hash + Ord + Clone> BoundedSet<K> {
    /// Control-only reclamation chooses a stable identity; dispatch never scans this set.
    ///
    /// `BoundedMap::iter` walks every bucket rather than only live entries, so the empty case is
    /// rejected up front: callers poll this on an idle control loop, where a full bucket sweep per
    /// poll would dominate the cost of finding nothing.
    pub fn pop_first(&mut self) -> Option<K> {
        if self.is_empty() {
            return None;
        }
        let key = self.iter().min()?.clone();
        self.remove(&key);
        Some(key)
    }
}
#[cfg(test)]
mod tests;
