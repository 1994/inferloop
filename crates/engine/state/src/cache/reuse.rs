//! Typed publication and backend reuse hooks; no device calls in the shared cache.
use std::convert::Infallible;
use std::iter::Chain;
use std::option::IntoIter as One;
use std::result::Result;
use std::vec::IntoIter as Many;

/// A completed snapshot ready for publication. Device storage stays opaque to the cache.
pub struct PrefixSnapshot<V> {
    pub tokens: Vec<u32>,
    pub value: V,
    pub bytes: u64,
}

/// Backend-specific reuse within the shared match/compatibility/LRU transaction.
///
/// An implementation acquires page leases or restores device storage. It returns success only
/// after the returned owner can safely retain the snapshot or its queued reads. On failure it
/// must settle its device work and roll back resources, or poison the destination as appropriate.
/// The cache retains the source and commits hit/LRU metadata only after this hook succeeds.
pub trait PrefixReuse<V> {
    type Output;
    type Error;
    fn compatible(&self, tokens: &[u32], value: &V) -> bool;
    /// # Errors
    /// Returns the backend's acquisition/restore error; the cache remains published and untouched.
    fn reuse(&mut self, tokens: &[u32], value: &V) -> Result<Self::Output, Self::Error>;
}

pub(super) struct CloneReuse<F>(pub F);
impl<V: Clone, F: Fn(&V) -> bool> PrefixReuse<V> for CloneReuse<F> {
    type Output = V;
    type Error = Infallible;
    fn compatible(&self, _: &[u32], value: &V) -> bool {
        (self.0)(value)
    }
    fn reuse(&mut self, _: &[u32], value: &V) -> Result<V, Infallible> {
        Ok(value.clone())
    }
}

/// Rejected/replaced/evicted payloads for backend cleanup. The common single-retirement case
/// stays inline; larger eviction batches allocate only their overflow list.
pub struct RetiredPrefixes<V> {
    first: Option<V>,
    rest: Vec<V>,
}
impl<V> RetiredPrefixes<V> {
    pub(super) const fn new() -> Self {
        Self {
            first: None,
            rest: Vec::new(),
        }
    }
    pub(super) fn push(&mut self, value: V) {
        if self.first.is_none() {
            self.first = Some(value);
        } else {
            self.rest.push(value);
        }
    }
    #[must_use]
    pub const fn len(&self) -> usize {
        if self.first.is_some() {
            self.rest.len() + 1
        } else {
            self.rest.len()
        }
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.first.is_none() && self.rest.is_empty()
    }
}
impl<V> IntoIterator for RetiredPrefixes<V> {
    type Item = V;
    type IntoIter = Chain<One<V>, Many<V>>;
    fn into_iter(self) -> Self::IntoIter {
        self.first.into_iter().chain(self.rest)
    }
}
