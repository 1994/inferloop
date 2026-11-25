//! A bounded, process-local [`RecordStore`].
use crate::clock::now;
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{
    LogRecord, Precondition, PutOutcome, Record, RecordCapabilities, RecordFuture, RecordKey,
    RecordState, RecordStore, RecordWrite, Retention, Version,
};
use std::collections::HashMap;
use std::future::ready;
use std::sync::{Mutex, MutexGuard};

/// Default retained record count.
const DEFAULT_MAX_RECORDS: usize = 256;
/// Default retained bytes of values and logs.
const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
/// Default largest value.
const DEFAULT_MAX_VALUE_BYTES: usize = 4 * 1024 * 1024 + 64 * 1024;

/// Bounds of a [`MemoryRecordStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRecordStoreConfig {
    pub max_records: usize,
    /// Bytes of all values and log entries together.
    pub max_bytes: usize,
    pub max_value_bytes: usize,
}

impl Default for MemoryRecordStoreConfig {
    fn default() -> Self {
        Self {
            max_records: DEFAULT_MAX_RECORDS,
            max_bytes: DEFAULT_MAX_BYTES,
            max_value_bytes: DEFAULT_MAX_VALUE_BYTES,
        }
    }
}

struct Entry {
    record: Record,
    log: Vec<LogRecord>,
    bytes: usize,
    last_used: u64,
}

impl Entry {
    const fn expired(&self, now: u64) -> bool {
        matches!(self.record.retention, Retention::Disposable(Some(at)) if at.get() <= now)
    }
}

/// What a write needs to know about the record it replaces.
#[derive(Clone, Copy)]
struct Current {
    version: Version,
    state: RecordState,
    bytes: usize,
    log_bytes: usize,
}

#[derive(Default)]
struct Records {
    entries: HashMap<RecordKey, Entry>,
    bytes: usize,
    clock: u64,
    last_version: u64,
}

/// Records kept in process memory: lost on restart and not shared.
///
/// Within its record and byte bounds it keeps everything; past them it drops
/// [`Retention::Disposable`] records least recently used first, and refuses a write that only
/// pinned records stand in the way of.
#[derive(Default)]
pub struct MemoryRecordStore {
    config: MemoryRecordStoreConfig,
    records: Mutex<Records>,
}

impl MemoryRecordStore {
    /// # Errors
    /// Returns an invalid-input error for zero bounds or a value bound above the byte bound.
    pub fn new(config: MemoryRecordStoreConfig) -> Result<Self> {
        if config.max_records == 0 || config.max_bytes == 0 || config.max_value_bytes == 0 {
            return Err(Error::invalid("memory record store bounds must be nonzero"));
        }
        if config.max_value_bytes > config.max_bytes {
            return Err(Error::invalid(
                "memory record store value bound exceeds its byte bound",
            ));
        }
        Ok(Self {
            config,
            records: Mutex::default(),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, Records>> {
        self.records
            .lock()
            .map_err(|_| Error::invariant("memory record store lock poisoned"))
    }

    /// The live entry for `key`, after dropping it if it has expired.
    fn live<'a>(records: &'a mut Records, key: &RecordKey, now: u64) -> Option<&'a mut Entry> {
        if records
            .entries
            .get(key)
            .is_some_and(|entry| entry.expired(now))
        {
            Self::remove(records, key);
        }
        records.clock += 1;
        let clock = records.clock;
        let entry = records.entries.get_mut(key)?;
        entry.last_used = clock;
        Some(entry)
    }

    fn remove(records: &mut Records, key: &RecordKey) {
        if let Some(entry) = records.entries.remove(key) {
            records.bytes -= entry.bytes;
        }
    }

    /// Make room for `extra_records` more records and `extra_bytes` more bytes, dropping
    /// expired then least recently used disposable records other than `keep`.
    fn make_room(
        &self,
        records: &mut Records,
        keep: &RecordKey,
        extra_records: usize,
        extra_bytes: usize,
        now: u64,
    ) -> Result<()> {
        let fits = |records: &Records| {
            records.entries.len() + extra_records <= self.config.max_records
                && records.bytes + extra_bytes <= self.config.max_bytes
        };
        if fits(records) {
            return Ok(());
        }
        let mut candidates: Vec<(bool, u64, RecordKey)> = records
            .entries
            .iter()
            .filter(|(key, entry)| {
                *key != keep && matches!(entry.record.retention, Retention::Disposable(_))
            })
            .map(|(key, entry)| (!entry.expired(now), entry.last_used, key.clone()))
            .collect();
        candidates.sort_unstable();
        for (_, _, key) in candidates {
            if fits(records) {
                break;
            }
            Self::remove(records, &key);
        }
        if fits(records) {
            Ok(())
        } else {
            Err(Error::new(
                ErrorCode::Capacity,
                "memory record store is full of pinned records",
            ))
        }
    }

    fn put_now(&self, key: &RecordKey, write: RecordWrite<'_>) -> Result<PutOutcome> {
        if write.value.len() > self.config.max_value_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "record value exceeds the store's value limit",
            ));
        }
        let now = now()?;
        let mut records = self.lock()?;
        let current = Self::live(&mut records, key, now).map(|entry| Current {
            version: entry.record.version,
            state: entry.record.state,
            bytes: entry.bytes,
            log_bytes: entry.bytes - entry.record.value.len(),
        });
        let allowed = match (write.condition, current) {
            (Precondition::Any, _) | (Precondition::Absent | Precondition::StateNotIn(_), None) => {
                true
            }
            (Precondition::Absent, Some(_)) => false,
            (Precondition::Version(version), current) => {
                current.is_some_and(|current| current.version == version)
            }
            (Precondition::StateNotIn(excluded), Some(current)) => {
                !excluded.contains(current.state)
            }
        };
        if !allowed {
            return Ok(PutOutcome::Rejected(
                current.map(|current| (current.version, current.state)),
            ));
        }
        // A rewrite keeps the record's log, so the log's bytes stay with the new value.
        let new_bytes = write.value.len() + current.map_or(0, |current| current.log_bytes);
        let old_bytes = current.map_or(0, |current| current.bytes);
        let extra_records = usize::from(current.is_none());
        self.make_room(
            &mut records,
            key,
            extra_records,
            new_bytes.saturating_sub(old_bytes),
            now,
        )?;
        records.last_version += 1;
        let version = Version::new(records.last_version);
        records.clock += 1;
        let last_used = records.clock;
        let log = records
            .entries
            .remove(key)
            .map(|entry| {
                records.bytes -= entry.bytes;
                entry.log
            })
            .unwrap_or_default();
        records.bytes += new_bytes;
        records.entries.insert(
            key.clone(),
            Entry {
                record: Record {
                    version,
                    state: write.state,
                    retention: write.retention,
                    value: write.value.to_vec(),
                },
                log,
                bytes: new_bytes,
                last_used,
            },
        );
        drop(records);
        Ok(PutOutcome::Written(version))
    }

    fn append_now(
        &self,
        key: &RecordKey,
        sequence: u64,
        value: &[u8],
        max_entries: usize,
    ) -> Result<()> {
        let now = now()?;
        let mut records = self.lock()?;
        let Some(entry) = Self::live(&mut records, key, now) else {
            return Err(Error::new(
                ErrorCode::NotFound,
                format!("record {key} not found"),
            ));
        };
        if entry
            .log
            .last()
            .is_some_and(|last| last.sequence >= sequence)
        {
            return Err(Error::invalid(format!(
                "log sequence {sequence} of {key} does not follow the last entry"
            )));
        }
        if entry.log.len() >= max_entries {
            return Err(Error::new(
                ErrorCode::Capacity,
                format!("log of {key} is full"),
            ));
        }
        self.make_room(&mut records, key, 0, value.len(), now)?;
        let Some(entry) = records.entries.get_mut(key) else {
            return Err(Error::invariant("record vanished while making room"));
        };
        entry.log.push(LogRecord {
            sequence,
            value: value.to_vec(),
        });
        entry.bytes += value.len();
        records.bytes += value.len();
        drop(records);
        Ok(())
    }
}

impl RecordStore for MemoryRecordStore {
    fn capabilities(&self) -> RecordCapabilities {
        RecordCapabilities {
            durable: false,
            shared: false,
            evicts_disposable: true,
            state_predicates: true,
            max_value_bytes: self.config.max_value_bytes,
        }
    }

    fn get<'a>(&'a self, key: &'a RecordKey) -> RecordFuture<'a, Option<Record>> {
        let result = now().and_then(|now| {
            let mut records = self.lock()?;
            Ok(Self::live(&mut records, key, now).map(|entry| entry.record.clone()))
        });
        Box::pin(ready(result))
    }

    fn put<'a>(
        &'a self,
        key: &'a RecordKey,
        write: RecordWrite<'a>,
    ) -> RecordFuture<'a, PutOutcome> {
        Box::pin(ready(self.put_now(key, write)))
    }

    fn delete<'a>(&'a self, key: &'a RecordKey) -> RecordFuture<'a, ()> {
        let result = self
            .lock()
            .map(|mut records| Self::remove(&mut records, key));
        Box::pin(ready(result))
    }

    fn append<'a>(
        &'a self,
        key: &'a RecordKey,
        sequence: u64,
        value: &'a [u8],
        max_entries: usize,
    ) -> RecordFuture<'a, ()> {
        Box::pin(ready(self.append_now(key, sequence, value, max_entries)))
    }

    fn log<'a>(
        &'a self,
        key: &'a RecordKey,
        after: Option<u64>,
    ) -> RecordFuture<'a, Vec<LogRecord>> {
        let result = now().and_then(|now| {
            let mut records = self.lock()?;
            Ok(
                Self::live(&mut records, key, now).map_or_else(Vec::new, |entry| {
                    entry
                        .log
                        .iter()
                        .filter(|entry| after.is_none_or(|after| entry.sequence > after))
                        .cloned()
                        .collect()
                }),
            )
        });
        Box::pin(ready(result))
    }
}
