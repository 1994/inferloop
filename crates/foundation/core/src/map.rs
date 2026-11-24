//! Fixed Robin Hood lookup with backward-shift deletion; churn never grows or rehashes storage.
use crate::{Error, ErrorCode, Result};
use ahash::RandomState;
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::mem::replace;
use std::mem::swap;
use std::result::Result as StdResult;
use std::{borrow::Borrow, hash::Hash};

/// Largest fixed lookup capacity accepted by [`BoundedMap::new`].
const MAX_MAP_CAPACITY: usize = 1_048_576;

#[derive(Clone, Debug)]
struct Bucket<K, V> {
    hash: u64,
    distance: usize,
    key: K,
    value: V,
}
#[derive(Clone, Debug)]
pub struct BoundedMap<K, V> {
    buckets: Vec<Option<Bucket<K, V>>>,
    hash: RandomState,
    capacity: usize,
    len: usize,
}
impl<K: Eq + Hash, V> BoundedMap<K, V> {
    /// # Errors
    /// Rejects invalid capacity or failure to allocate fixed lookup slots.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > MAX_MAP_CAPACITY {
            return Err(Error::invalid("invalid fixed map capacity"));
        }
        let slots = capacity
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| Error::invalid("fixed map size overflow"))?;
        let mut buckets = Vec::new();
        buckets
            .try_reserve_exact(slots)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        buckets.resize_with(slots, || None);
        Ok(Self {
            buckets,
            hash: RandomState::new(),
            capacity,
            len: 0,
        })
    }
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.buckets
            .iter_mut()
            .filter_map(|bucket| bucket.as_mut().map(|bucket| &mut bucket.value))
    }
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Clear entries while retaining every lookup bucket for the next publication.
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            *bucket = None;
        }
        self.len = 0;
    }
    fn locate<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
    {
        let hash = self.hash.hash_one(key);
        let mask = self.buckets.len() - 1;
        let mut slot = usize::try_from(hash & mask as u64).ok()?;
        for distance in 0..self.buckets.len() {
            let bucket = self.buckets[slot].as_ref()?;
            if bucket.distance < distance {
                return None;
            }
            if bucket.hash == hash && bucket.key.borrow() == key {
                return Some(slot);
            }
            slot = (slot + 1) & mask;
        }
        None
    }
    pub fn get<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.buckets[self.locate(key)?].as_ref().map(|b| &b.value)
    }
    pub fn get_mut<Q: Eq + Hash + ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        let slot = self.locate(key)?;
        self.buckets[slot].as_mut().map(|b| &mut b.value)
    }
    pub fn contains_key<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.locate(key).is_some()
    }
    /// # Errors
    /// Returns capacity when adding a new key to a full map. Existing keys can still be replaced.
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        if let Some(slot) = self.locate(&key) {
            return Ok(self.buckets[slot]
                .as_mut()
                .map(|b| replace(&mut b.value, value)));
        }
        if self.len == self.capacity {
            return Err(Error::new(ErrorCode::Capacity, "fixed map exhausted"));
        }
        let hash = self.hash.hash_one(&key);
        let mask = self.buckets.len() - 1;
        let mut slot = usize::try_from(hash & mask as u64)
            .map_err(|_| Error::invariant("fixed map index overflow"))?;
        let mut incoming = Bucket {
            hash,
            distance: 0,
            key,
            value,
        };
        loop {
            if let Some(bucket) = &mut self.buckets[slot] {
                if bucket.distance < incoming.distance {
                    swap(bucket, &mut incoming);
                }
            } else {
                self.buckets[slot] = Some(incoming);
                self.len += 1;
                return Ok(None);
            }
            incoming.distance += 1;
            slot = (slot + 1) & mask;
        }
    }
    pub fn remove<Q: Eq + Hash + ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        let mut hole = self.locate(key)?;
        let removed = self.buckets[hole].take()?;
        let mask = self.buckets.len() - 1;
        loop {
            let next = (hole + 1) & mask;
            if self.buckets[next].as_ref().is_none_or(|b| b.distance == 0) {
                break;
            }
            let mut bucket = self.buckets[next].take()?;
            bucket.distance -= 1;
            self.buckets[hole] = Some(bucket);
            hole = next;
        }
        self.len -= 1;
        Some(removed.value)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.buckets.iter().flatten().map(|b| (&b.key, &b.value))
    }
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }
}
impl<K: Eq + Hash, V: PartialEq> PartialEq for BoundedMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.capacity == other.capacity
            && self.len == other.len
            && self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}
impl<K: Eq + Hash, V: Eq> Eq for BoundedMap<K, V> {}
#[derive(Serialize, Deserialize)]
struct Snapshot<K, V> {
    capacity: usize,
    entries: Vec<(K, V)>,
}
impl<K: Eq + Hash + Serialize, V: Serialize> Serialize for BoundedMap<K, V> {
    fn serialize<S: Serializer>(&self, serializer: S) -> StdResult<S::Ok, S::Error> {
        Snapshot {
            capacity: self.capacity,
            entries: self.iter().collect::<Vec<_>>(),
        }
        .serialize(serializer)
    }
}
impl<'de, K: Eq + Hash + Deserialize<'de>, V: Deserialize<'de>> Deserialize<'de>
    for BoundedMap<K, V>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> StdResult<Self, D::Error> {
        let snapshot = Snapshot::<K, V>::deserialize(deserializer)?;
        let mut map = Self::new(snapshot.capacity).map_err(DeError::custom)?;
        for (key, value) in snapshot.entries {
            if map.insert(key, value).map_err(DeError::custom)?.is_some() {
                return Err(DeError::custom("duplicate fixed map key"));
            }
        }
        Ok(map)
    }
}
#[cfg(test)]
mod tests;
