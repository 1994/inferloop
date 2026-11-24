//! An intrusive FIFO gives bounded, fair resource polling without heap nodes or empty-slot scans.
use super::PendingResource;
use infer_core::{Error, RequestHandle, RequestId, Result, arena::Arena, map::BoundedMap};
use std::borrow::Borrow;
struct Entry {
    id: RequestId,
    value: PendingResource,
    previous: Option<RequestHandle>,
    next: Option<RequestHandle>,
}
pub struct ResourceWaiters {
    entries: Arena<RequestHandle, Entry>,
    index: BoundedMap<RequestId, RequestHandle>,
    head: Option<RequestHandle>,
    tail: Option<RequestHandle>,
    cursor: Option<RequestHandle>,
}
impl ResourceWaiters {
    pub fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            entries: Arena::with_capacity(capacity)?,
            index: BoundedMap::new(capacity)?,
            head: None,
            tail: None,
            cursor: None,
        })
    }
    pub const fn len(&self) -> usize {
        self.index.len()
    }
    pub const fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
    pub fn contains_key(&self, id: impl Borrow<RequestId>) -> bool {
        self.index.contains_key(id.borrow())
    }
    pub fn get(&self, id: impl Borrow<RequestId>) -> Option<&PendingResource> {
        self.entries
            .get(*self.index.get(id.borrow())?)
            .map(|entry| &entry.value)
    }
    pub fn get_mut(&mut self, id: impl Borrow<RequestId>) -> Option<&mut PendingResource> {
        self.entries
            .get_mut(*self.index.get(id.borrow())?)
            .map(|entry| &mut entry.value)
    }
    pub fn insert(&mut self, id: RequestId, value: PendingResource) -> Result<()> {
        if self.index.contains_key(&id) {
            return Err(Error::invariant("resource waiter inserted twice"));
        }
        let handle = self.entries.insert(Entry {
            id,
            value,
            previous: self.tail,
            next: None,
        })?;
        self.index.insert(id, handle)?;
        if let Some(tail) = self.tail {
            self.entries
                .get_mut(tail)
                .ok_or_else(|| Error::invariant("resource tail missing"))?
                .next = Some(handle);
        } else {
            self.head = Some(handle);
        }
        self.tail = Some(handle);
        Ok(())
    }
    pub fn remove(&mut self, id: impl Borrow<RequestId>) -> Option<PendingResource> {
        let handle = self.index.remove(id.borrow())?;
        let entry = self.entries.remove(handle)?;
        if let Some(previous) = entry.previous {
            self.entries.get_mut(previous)?.next = entry.next;
        } else {
            self.head = entry.next;
        }
        if let Some(next) = entry.next {
            self.entries.get_mut(next)?.previous = entry.previous;
        } else {
            self.tail = entry.previous;
        }
        if self.cursor == Some(handle) {
            self.cursor = entry.next.or(self.head);
        }
        Some(entry.value)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&RequestId, &PendingResource)> {
        self.index.iter().filter_map(|(_, handle)| {
            self.entries
                .get(*handle)
                .map(|entry| (&entry.id, &entry.value))
        })
    }
    pub fn poll_window(&mut self, limit: usize, output: &mut Vec<RequestId>) -> Result<()> {
        output.clear();
        let count = limit.min(self.len());
        if output.capacity() < count {
            return Err(Error::invariant("resource poll scratch too small"));
        }
        let mut cursor = self.cursor.or(self.head);
        for _ in 0..count {
            let entry = self
                .entries
                .get(cursor.ok_or_else(|| Error::invariant("resource cursor missing"))?)
                .ok_or_else(|| Error::invariant("resource poll entry missing"))?;
            output.push(entry.id);
            cursor = entry.next.or(self.head);
        }
        self.cursor = cursor;
        Ok(())
    }
}
#[cfg(test)]
mod tests;
