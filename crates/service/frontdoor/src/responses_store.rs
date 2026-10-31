//! Bounded in-process implementation of the Responses state extension.
//!
//! This is the reference backend installed by `--enable-response-store`. It is deliberately
//! process-local: it exists so a single-node deployment can use `store`, `previous_response_id`,
//! retrieval and background responses without an external service. An external SDK can install
//! its own [`infer_spi::ResponseStore`] through [`infer_spi::ResponsesExtensions`].
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{
    RESPONSES_SPI_VERSION, ResponseEvent, ResponseFuture, ResponseId, ResponseStatus,
    ResponseStore, ResponseStoreCapabilities, StoredResponse, WriteCondition,
};
use std::collections::HashMap;
use std::future::ready;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Default retained response count.
const DEFAULT_MAX_ENTRIES: usize = 256;
/// Default retained payload budget.
const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
/// Default largest single record.
const DEFAULT_MAX_RECORD_BYTES: usize = 4 * 1024 * 1024;
/// Default retained background event count per response.
const DEFAULT_MAX_EVENTS: usize = 4096;
/// Default retention after the last access.
const DEFAULT_TTL: Duration = Duration::from_secs(60 * 60);

/// Tuning for [`InMemoryResponseStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InMemoryResponseStoreConfig {
    pub max_entries: usize,
    pub max_bytes: usize,
    pub max_record_bytes: usize,
    pub max_events: usize,
    /// `None` keeps terminal responses until eviction.
    pub ttl: Option<Duration>,
}

impl Default for InMemoryResponseStoreConfig {
    fn default() -> Self {
        Self {
            max_entries: DEFAULT_MAX_ENTRIES,
            max_bytes: DEFAULT_MAX_BYTES,
            max_record_bytes: DEFAULT_MAX_RECORD_BYTES,
            max_events: DEFAULT_MAX_EVENTS,
            ttl: Some(DEFAULT_TTL),
        }
    }
}

struct Entry {
    response: StoredResponse,
    last_used: u64,
    expires_at: Option<Instant>,
    bytes: usize,
}

struct Store {
    responses: HashMap<ResponseId, Entry>,
    events: HashMap<ResponseId, Vec<ResponseEvent>>,
    total_bytes: usize,
    clock: u64,
}

/// Process-local, bounded Responses store.
pub struct InMemoryResponseStore {
    config: InMemoryResponseStoreConfig,
    store: Mutex<Store>,
}

impl InMemoryResponseStore {
    /// Build a store with the supplied bounds.
    ///
    /// # Errors
    /// Returns an invalid-input error for zero limits or a per-record limit above the total
    /// budget.
    pub fn new(config: InMemoryResponseStoreConfig) -> Result<Self> {
        if config.max_entries == 0
            || config.max_bytes == 0
            || config.max_record_bytes == 0
            || config.max_events == 0
        {
            return Err(Error::invalid("response store limits must be nonzero"));
        }
        if config.max_record_bytes > config.max_bytes {
            return Err(Error::invalid(
                "response store per-record limit exceeds its total budget",
            ));
        }
        Ok(Self {
            config,
            store: Mutex::new(Store {
                responses: HashMap::new(),
                events: HashMap::new(),
                total_bytes: 0,
                clock: 0,
            }),
        })
    }

    /// The active tuning.
    #[must_use]
    pub const fn config(&self) -> InMemoryResponseStoreConfig {
        self.config
    }

    fn lock(&self) -> Result<MutexGuard<'_, Store>> {
        self.store
            .lock()
            .map_err(|_| Error::invariant("response store lock poisoned"))
    }

    const fn tick(store: &mut Store) -> u64 {
        store.clock = store.clock.wrapping_add(1);
        store.clock
    }

    fn purge_expired(&self, store: &mut Store) {
        if self.config.ttl.is_none() {
            return;
        }
        let now = Instant::now();
        let expired: Vec<ResponseId> = store
            .responses
            .iter()
            .filter(|(_, entry)| entry.response.status.is_terminal())
            .filter(|(_, entry)| entry.expires_at.is_some_and(|at| at <= now))
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            Self::remove(store, &id);
        }
    }

    fn remove(store: &mut Store, id: &ResponseId) {
        if let Some(entry) = store.responses.remove(id) {
            store.total_bytes = store.total_bytes.saturating_sub(entry.bytes);
        }
        store.events.remove(id);
    }

    fn eviction_candidates(
        store: &Store,
        keep: Option<&ResponseId>,
        needed_entries: usize,
        needed_bytes: usize,
    ) -> Result<Vec<ResponseId>> {
        let mut candidates: Vec<&ResponseId> = store
            .responses
            .iter()
            .filter(|(id, entry)| Some(*id) != keep && entry.response.status.is_terminal())
            .map(|(id, _)| id)
            .collect();
        candidates.sort_by_key(|id| store.responses.get(*id).map_or(0, |entry| entry.last_used));
        // `keep` is about to be replaced, so it is not part of the budget this write has to
        // free. Excluding it also avoids evicting an active response we are about to update.
        let mut entries = store.responses.len();
        let mut bytes = store.total_bytes;
        if let Some(entry) = keep.and_then(|id| store.responses.get(id)) {
            entries = entries.saturating_sub(1);
            bytes = bytes.saturating_sub(entry.bytes);
        }
        let mut selected = Vec::new();
        for id in candidates {
            if entries <= needed_entries && bytes <= needed_bytes {
                break;
            }
            let Some(entry) = store.responses.get(id) else {
                continue;
            };
            entries = entries.saturating_sub(1);
            bytes = bytes.saturating_sub(entry.bytes);
            selected.push(id.clone());
        }
        if entries > needed_entries || bytes > needed_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "response store retention budget is exhausted by active responses",
            ));
        }
        Ok(selected)
    }

    fn get_sync(&self, id: &ResponseId) -> Result<Option<StoredResponse>> {
        let mut store = self.lock()?;
        self.purge_expired(&mut store);
        let clock = Self::tick(&mut store);
        let Some(entry) = store.responses.get_mut(id) else {
            return Ok(None);
        };
        entry.last_used = clock;
        let response = entry.response.clone();
        drop(store);
        Ok(Some(response))
    }

    fn put_sync(&self, response: StoredResponse, condition: WriteCondition) -> Result<()> {
        response.validate()?;
        let bytes = response
            .request_input
            .len()
            .checked_add(response.response.len())
            .ok_or_else(|| Error::invalid("stored response size overflow"))?;
        if bytes > self.config.max_record_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "stored response exceeds the per-record byte limit",
            ));
        }
        let mut store = self.lock()?;
        self.purge_expired(&mut store);
        if let Some(entry) = store.responses.get(&response.id)
            && match condition {
                WriteCondition::Always => false,
                WriteCondition::UnlessStatus(status) => entry.response.status == status,
                WriteCondition::UnlessTerminal => entry.response.status.is_terminal(),
            }
        {
            return Ok(());
        }
        let replacing = store
            .responses
            .get(&response.id)
            .map_or(0, |entry| entry.bytes);
        let existing = usize::from(store.responses.contains_key(&response.id));
        let needed_entries = self.config.max_entries.saturating_sub(1);
        let needed_bytes = self.config.max_bytes.saturating_sub(bytes);
        let base_entries = store.responses.len().saturating_sub(existing);
        let base_bytes = store.total_bytes.saturating_sub(replacing);
        let candidates = if base_entries.saturating_add(1) > self.config.max_entries
            || base_bytes > needed_bytes
        {
            Self::eviction_candidates(&store, Some(&response.id), needed_entries, needed_bytes)?
        } else {
            Vec::new()
        };
        for id in candidates {
            Self::remove(&mut store, &id);
        }
        Self::remove(&mut store, &response.id);
        let expires_at = self
            .config
            .ttl
            .and_then(|ttl| Instant::now().checked_add(ttl));
        let clock = Self::tick(&mut store);
        store.responses.insert(
            response.id.clone(),
            Entry {
                response,
                last_used: clock,
                expires_at,
                bytes,
            },
        );
        store.total_bytes = store.total_bytes.checked_add(bytes).ok_or_else(|| {
            Error::new(
                ErrorCode::Capacity,
                "response store byte accounting overflow",
            )
        })?;
        drop(store);
        Ok(())
    }

    fn delete_sync(&self, id: &ResponseId) -> Result<()> {
        let mut store = self.lock()?;
        self.purge_expired(&mut store);
        Self::remove(&mut store, id);
        Ok(())
    }

    fn transition_sync(
        &self,
        id: &ResponseId,
        allowed: &[ResponseStatus],
        next: ResponseStatus,
    ) -> Result<StoredResponse> {
        let mut store = self.lock()?;
        self.purge_expired(&mut store);
        let clock = Self::tick(&mut store);
        let entry = store
            .responses
            .get_mut(id)
            .ok_or_else(|| Error::new(ErrorCode::NotFound, "stored response not found"))?;
        let current = entry.response.status;
        if !allowed.contains(&current) || !current.can_transition_to(next) {
            return Err(Error::new(
                ErrorCode::Conflict,
                format!("response status {current} cannot transition to {next}"),
            ));
        }
        entry.response.status = next;
        entry.last_used = clock;
        entry.expires_at = self
            .config
            .ttl
            .and_then(|ttl| Instant::now().checked_add(ttl));
        let response = entry.response.clone();
        drop(store);
        Ok(response)
    }

    fn append_event_sync(&self, id: &ResponseId, event: ResponseEvent) -> Result<()> {
        let mut store = self.lock()?;
        self.purge_expired(&mut store);
        if !store.responses.contains_key(id) {
            return Err(Error::new(ErrorCode::NotFound, "stored response not found"));
        }
        let events = store.events.entry(id.clone()).or_default();
        if events
            .last()
            .is_some_and(|last| last.sequence >= event.sequence)
        {
            return Err(Error::invalid(
                "response events must have increasing sequence numbers",
            ));
        }
        if events.len() >= self.config.max_events {
            return Err(Error::new(
                ErrorCode::Capacity,
                "response event retention is full",
            ));
        }
        events.push(event);
        drop(store);
        Ok(())
    }

    fn events_sync(&self, id: &ResponseId, after: Option<u64>) -> Result<Vec<ResponseEvent>> {
        let mut store = self.lock()?;
        self.purge_expired(&mut store);
        let Some(events) = store.events.get(id) else {
            return Ok(Vec::new());
        };
        let events = events
            .iter()
            .filter(|event| after.is_none_or(|after| event.sequence > after))
            .cloned()
            .collect();
        drop(store);
        Ok(events)
    }
}

impl Default for InMemoryResponseStore {
    fn default() -> Self {
        Self {
            config: InMemoryResponseStoreConfig::default(),
            store: Mutex::new(Store {
                responses: HashMap::new(),
                events: HashMap::new(),
                total_bytes: 0,
                clock: 0,
            }),
        }
    }
}

impl ResponseStore for InMemoryResponseStore {
    fn capabilities(&self) -> ResponseStoreCapabilities {
        ResponseStoreCapabilities {
            retrieve: true,
            previous_response_id: true,
            background: true,
            cancel: true,
            event_replay: true,
            persistent: false,
            shared: false,
            max_record_bytes: self.config.max_record_bytes,
        }
    }

    fn spi_version(&self) -> u32 {
        RESPONSES_SPI_VERSION
    }

    fn get<'a>(&'a self, id: &'a ResponseId) -> ResponseFuture<'a, Option<StoredResponse>> {
        Box::pin(ready(self.get_sync(id)))
    }

    fn put(&self, response: StoredResponse, condition: WriteCondition) -> ResponseFuture<'_, ()> {
        Box::pin(ready(self.put_sync(response, condition)))
    }

    fn delete<'a>(&'a self, id: &'a ResponseId) -> ResponseFuture<'a, ()> {
        Box::pin(ready(self.delete_sync(id)))
    }

    fn transition<'a>(
        &'a self,
        id: &'a ResponseId,
        allowed: &'a [ResponseStatus],
        next: ResponseStatus,
    ) -> ResponseFuture<'a, StoredResponse> {
        Box::pin(ready(self.transition_sync(id, allowed, next)))
    }

    fn append_event<'a>(
        &'a self,
        id: &'a ResponseId,
        event: ResponseEvent,
    ) -> ResponseFuture<'a, ()> {
        Box::pin(ready(self.append_event_sync(id, event)))
    }

    fn events<'a>(
        &'a self,
        id: &'a ResponseId,
        after: Option<u64>,
    ) -> ResponseFuture<'a, Vec<ResponseEvent>> {
        Box::pin(ready(self.events_sync(id, after)))
    }
}

#[cfg(test)]
#[path = "../tests/unit/responses_store.rs"]
mod tests;
