//! The record storage primitive shared by every persistent domain.
//!
//! Domain stores ([`crate::ResponseStore`], and later conversations and archive indexes) are
//! written once over [`RecordStore`]; a backend (in-process, file, SQL, Redis...) implements only
//! this trait. Every domain operation is a conditional write to one key or an append to that
//! key's log, so the contract needs no cross-key transactions and maps onto stores that offer
//! only single-key atomicity.
use infer_core::{Error, ErrorCode, Result};
use std::fmt;
use std::future::Future;
use std::pin::Pin;

/// Version of the record storage contract.
pub const RECORDS_SPI_VERSION: u32 = 1;

/// A boxed record-store operation; no executor type is part of the contract. Dropping it
/// cancels the operation, which then either took effect whole or not at all.
pub type RecordFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// The domain a key belongs to, fixed by the domain store (for example `responses`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Namespace(&'static str);

impl Namespace {
    /// Largest accepted namespace, in bytes.
    pub const MAX_BYTES: usize = 32;

    /// Name a domain.
    ///
    /// # Errors
    /// Returns an invalid-input error for an empty or oversized name, or one with characters
    /// other than lowercase ASCII letters, digits and underscores.
    pub fn new(name: &'static str) -> Result<Self> {
        if name.is_empty()
            || name.len() > Self::MAX_BYTES
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        {
            return Err(Error::invalid(format!(
                "record namespace {name:?} must be 1..{} lowercase letters, digits or underscores",
                Self::MAX_BYTES
            )));
        }
        Ok(Self(name))
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// The key of one record: a domain and an identifier within it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordKey {
    namespace: Namespace,
    id: Box<str>,
}

impl RecordKey {
    /// Largest accepted identifier, in bytes.
    pub const MAX_ID_BYTES: usize = 128;

    /// Build a key.
    ///
    /// # Errors
    /// Returns an invalid-input error for an empty or oversized identifier, or one with
    /// characters other than ASCII letters, digits, underscores or dashes.
    pub fn new(namespace: Namespace, id: &str) -> Result<Self> {
        if id.is_empty()
            || id.len() > Self::MAX_ID_BYTES
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(Error::invalid(format!(
                "record id must be 1..{} ASCII letters, digits, underscores or dashes",
                Self::MAX_ID_BYTES
            )));
        }
        Ok(Self {
            namespace,
            id: id.into(),
        })
    }

    #[must_use]
    pub const fn namespace(&self) -> Namespace {
        self.namespace
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl fmt::Display for RecordKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.namespace.as_str(), self.id)
    }
}

/// The version a store gave one write. Versions increase across the whole store and are never
/// reused, so a version names exactly one write even after the key is deleted and recreated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(u64);

impl Version {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A small label the domain gives a record (for example a lifecycle status), so a write can be
/// conditioned on it without reading the record first. The store never interprets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordState(u8);

impl RecordState {
    /// Labels a [`StateSet`] can hold. Each label's bit then fits a positive signed 64-bit
    /// integer, so a SQL store can test a set with one integer parameter.
    pub const LIMIT: u8 = 63;

    /// # Errors
    /// Returns an invalid-input error for a label of [`Self::LIMIT`] or more.
    pub fn new(value: u8) -> Result<Self> {
        if value >= Self::LIMIT {
            return Err(Error::invalid(format!(
                "record state {value} must be below {}",
                Self::LIMIT
            )));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// This label's bit in a [`StateSet`].
    #[must_use]
    pub const fn bit(self) -> u64 {
        1 << self.0
    }
}

/// A set of [`RecordState`] labels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct StateSet(u64);

impl StateSet {
    #[must_use]
    pub fn of(states: &[RecordState]) -> Self {
        Self(states.iter().fold(0, |bits, state| bits | state.bit()))
    }

    #[must_use]
    pub const fn contains(self, state: RecordState) -> bool {
        self.0 & state.bit() != 0
    }

    /// The set as bits, one per label; always a positive `i64`.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }
}

/// A point in time, in milliseconds since the Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnixMillis(u64);

impl UnixMillis {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// How long a record must be kept: its durability class, chosen by the domain per write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    /// Kept until it is deleted or rewritten; never expires and is never evicted.
    Pinned,
    /// May be dropped: it disappears once the time passes (if any), and a store with bounded
    /// capacity may drop it earlier, least recently used first.
    Disposable(Option<UnixMillis>),
}

/// When a write takes effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precondition {
    /// Always; creates or replaces the record.
    Any,
    /// Only when no record exists, so two creators cannot overwrite each other.
    Absent,
    /// Only when the current record has this version.
    Version(Version),
    /// When the record is absent or its state is not in the set.
    StateNotIn(StateSet),
}

/// One write of a record.
#[derive(Debug, Clone, Copy)]
pub struct RecordWrite<'a> {
    pub value: &'a [u8],
    pub state: RecordState,
    pub retention: Retention,
    pub condition: Precondition,
}

/// What a write did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PutOutcome {
    Written(Version),
    /// The precondition did not hold; the current record's version and state, if one exists.
    Rejected(Option<(Version, RecordState)>),
}

/// One record as read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub version: Version,
    pub state: RecordState,
    pub retention: Retention,
    pub value: Vec<u8>,
}

/// One entry of a record's log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogRecord {
    pub sequence: u64,
    pub value: Vec<u8>,
}

/// What a store guarantees, checked by the deployment against each domain's needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent guarantees a deployment checks one by one"
)]
pub struct RecordCapabilities {
    /// Writes survive a process restart.
    pub durable: bool,
    /// Several processes, including on other hosts, may use the store at once.
    pub shared: bool,
    /// [`Retention::Disposable`] records may be dropped before they expire.
    pub evicts_disposable: bool,
    /// [`Precondition::StateNotIn`] is evaluated by the store in one atomic step; otherwise
    /// callers fall back to [`put_by_version`].
    pub state_predicates: bool,
    /// Largest accepted value, in bytes.
    pub max_value_bytes: usize,
}

/// Keyed records with store-assigned versions and one append-only log per key.
///
/// Each operation is atomic and linearizable for its key. A record's log lives as long as the
/// record: rewriting the record keeps it, deleting or expiring the record removes it.
pub trait RecordStore: Send + Sync {
    fn capabilities(&self) -> RecordCapabilities;

    /// The contract version the implementation was built against.
    fn spi_version(&self) -> u32 {
        RECORDS_SPI_VERSION
    }

    /// Read one record; an absent or expired record is `None`.
    ///
    /// # Errors
    /// Returns a backend error for unavailable storage.
    fn get<'a>(&'a self, key: &'a RecordKey) -> RecordFuture<'a, Option<Record>>;

    /// Write one record when its precondition holds.
    ///
    /// # Errors
    /// Returns a capacity error for a value over [`RecordCapabilities::max_value_bytes`] or a
    /// full store, and a backend error for unavailable storage.
    fn put<'a>(
        &'a self,
        key: &'a RecordKey,
        write: RecordWrite<'a>,
    ) -> RecordFuture<'a, PutOutcome>;

    /// Delete one record and its log; deleting an absent record succeeds.
    ///
    /// # Errors
    /// Returns a backend error for unavailable storage.
    fn delete<'a>(&'a self, key: &'a RecordKey) -> RecordFuture<'a, ()>;

    /// Append one entry to a record's log. `sequence` must exceed the last entry's, and the
    /// log may hold at most `max_entries`.
    ///
    /// # Errors
    /// Returns not-found for an absent or expired record, invalid-input for a sequence that does
    /// not increase, capacity for a full log, and a backend error for unavailable storage.
    fn append<'a>(
        &'a self,
        key: &'a RecordKey,
        sequence: u64,
        value: &'a [u8],
        max_entries: usize,
    ) -> RecordFuture<'a, ()>;

    /// Read a record's log entries after `after` (all of them when `None`), in order.
    ///
    /// # Errors
    /// Returns a backend error for unavailable storage. An absent record has an empty log.
    fn log<'a>(
        &'a self,
        key: &'a RecordKey,
        after: Option<u64>,
    ) -> RecordFuture<'a, Vec<LogRecord>>;
}

/// Attempts [`put_by_version`] makes before reporting a conflict. Writers of one key are its
/// owner and a canceller, so more than a few rounds means something else is wrong.
pub const STATE_PREDICATE_ATTEMPTS: usize = 8;

/// Evaluate [`Precondition::StateNotIn`] by reading the record and writing on its version, for
/// stores without [`RecordCapabilities::state_predicates`]. Other preconditions pass through.
///
/// # Errors
/// Returns the store's errors, and a conflict error when the record keeps changing between the
/// read and the write.
pub async fn put_by_version(
    store: &dyn RecordStore,
    key: &RecordKey,
    write: RecordWrite<'_>,
) -> Result<PutOutcome> {
    let Precondition::StateNotIn(excluded) = write.condition else {
        return store.put(key, write).await;
    };
    for _ in 0..STATE_PREDICATE_ATTEMPTS {
        let condition = match store.get(key).await? {
            Some(current) if excluded.contains(current.state) => {
                return Ok(PutOutcome::Rejected(Some((current.version, current.state))));
            }
            Some(current) => Precondition::Version(current.version),
            None => Precondition::Absent,
        };
        match store.put(key, RecordWrite { condition, ..write }).await? {
            PutOutcome::Rejected(_) => {}
            written @ PutOutcome::Written(_) => return Ok(written),
        }
    }
    Err(Error::new(
        ErrorCode::Conflict,
        format!("record {key} kept changing during a conditional write"),
    ))
}
