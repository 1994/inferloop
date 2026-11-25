//! [`ResponseStore`] over any [`RecordStore`]: the one place Responses persistence rules live.
use crate::clock::now;
use crate::envelope::{decode, encode};
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{
    Namespace, Precondition, PutOutcome, RecordKey, RecordState, RecordStore, RecordWrite,
    ResponseEvent, ResponseFuture, ResponseId, ResponseStatus, ResponseStore,
    ResponseStoreCapabilities, Retention, STATE_PREDICATE_ATTEMPTS, StateSet, StoredResponse,
    UnixMillis, WriteCondition, put_by_version,
};
use std::sync::Arc;
use std::time::Duration;

/// Records of this store, in the record store's key space.
const NAMESPACE: &str = "responses";

/// Default largest `request_input + response` payload, as before records existed.
const DEFAULT_MAX_RECORD_BYTES: usize = 4 * 1024 * 1024;
/// Default background events kept per response.
const DEFAULT_MAX_EVENTS: usize = 4096;
/// Default retention of a terminal response after its last write.
const DEFAULT_TTL: Duration = Duration::from_secs(60 * 60);

/// Every status, in the order of its record state label.
const STATUSES: [ResponseStatus; 6] = [
    ResponseStatus::Queued,
    ResponseStatus::InProgress,
    ResponseStatus::Completed,
    ResponseStatus::Incomplete,
    ResponseStatus::Failed,
    ResponseStatus::Cancelled,
];

/// How long and how much of a response is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseRetention {
    /// Largest `request_input + response` payload of one response, in bytes.
    pub max_record_bytes: usize,
    /// Background events kept per response.
    pub max_events: usize,
    /// How long a terminal response is kept after its last write; `None` keeps it until it is
    /// deleted or a bounded store needs the room. Active responses are always kept.
    pub ttl: Option<Duration>,
}

impl Default for ResponseRetention {
    fn default() -> Self {
        Self {
            max_record_bytes: DEFAULT_MAX_RECORD_BYTES,
            max_events: DEFAULT_MAX_EVENTS,
            ttl: Some(DEFAULT_TTL),
        }
    }
}

/// Responses state kept in a [`RecordStore`].
///
/// Each response is one record whose state label is its status, so a terminal snapshot that
/// must not overwrite a cancelled response is a single conditional write. Active responses are
/// [`Retention::Pinned`]; terminal ones are [`Retention::Disposable`] with the configured TTL,
/// so a bounded store drops finished responses and never running ones. Background events are
/// the record's log.
pub struct RecordResponseStore {
    records: Arc<dyn RecordStore>,
    retention: ResponseRetention,
    namespace: Namespace,
}

impl RecordResponseStore {
    /// Keep Responses state in `records`.
    ///
    /// # Errors
    /// Returns an invalid-input error for zero limits, or when the record store cannot hold a
    /// response of `max_record_bytes`.
    pub fn new(records: Arc<dyn RecordStore>, retention: ResponseRetention) -> Result<Self> {
        if retention.max_record_bytes == 0
            || retention.max_events == 0
            || retention.ttl.is_some_and(|ttl| ttl.is_zero())
        {
            return Err(Error::invalid("response retention limits must be nonzero"));
        }
        if retention.max_record_bytes > records.capabilities().max_value_bytes {
            return Err(Error::invalid(
                "the record store's value limit is below the response record limit",
            ));
        }
        Ok(Self {
            records,
            retention,
            namespace: Namespace::new(NAMESPACE)?,
        })
    }

    fn key(&self, id: &ResponseId) -> Result<RecordKey> {
        RecordKey::new(self.namespace, id.as_str())
    }

    fn retention_for(&self, status: ResponseStatus) -> Result<Retention> {
        if !status.is_terminal() {
            return Ok(Retention::Pinned);
        }
        let expires = self
            .retention
            .ttl
            .map(|ttl| {
                let ttl = u64::try_from(ttl.as_millis())
                    .map_err(|_| Error::invalid("response retention is too long"))?;
                now()?
                    .checked_add(ttl)
                    .map(UnixMillis::new)
                    .ok_or_else(|| Error::invalid("response retention is too long"))
            })
            .transpose()?;
        Ok(Retention::Disposable(expires))
    }

    fn check_size(&self, response: &StoredResponse) -> Result<()> {
        let bytes = response
            .request_input
            .len()
            .checked_add(response.response.len())
            .ok_or_else(|| Error::invalid("stored response size overflow"))?;
        if bytes > self.retention.max_record_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "stored response exceeds the per-record byte limit",
            ));
        }
        Ok(())
    }

    async fn write(&self, key: &RecordKey, write: RecordWrite<'_>) -> Result<PutOutcome> {
        if self.records.capabilities().state_predicates {
            self.records.put(key, write).await
        } else {
            put_by_version(self.records.as_ref(), key, write).await
        }
    }

    async fn put_response(
        &self,
        response: StoredResponse,
        condition: WriteCondition,
    ) -> Result<()> {
        response.validate()?;
        self.check_size(&response)?;
        let key = self.key(&response.id)?;
        let value = encode(&response)?;
        let condition = match condition {
            WriteCondition::Always => Precondition::Any,
            WriteCondition::UnlessStatus(status) => {
                Precondition::StateNotIn(StateSet::of(&[state_of(status)?]))
            }
            WriteCondition::UnlessTerminal => Precondition::StateNotIn(terminal_states()?),
        };
        let write = RecordWrite {
            value: &value,
            state: state_of(response.status)?,
            retention: self.retention_for(response.status)?,
            condition,
        };
        // A rejected conditional write is the success case the condition asked for.
        self.write(&key, write).await.map(drop)
    }

    async fn get_response(&self, id: &ResponseId) -> Result<Option<StoredResponse>> {
        let key = self.key(id)?;
        let Some(record) = self.records.get(&key).await? else {
            return Ok(None);
        };
        let response = decode(&record.value)?;
        if state_of(response.status)? != record.state {
            return Err(Error::invariant(format!(
                "record {key} has state {} but holds a {} response",
                record.state.get(),
                response.status
            )));
        }
        Ok(Some(response))
    }

    async fn transition_response(
        &self,
        id: &ResponseId,
        allowed: &[ResponseStatus],
        next: ResponseStatus,
    ) -> Result<StoredResponse> {
        let key = self.key(id)?;
        for _ in 0..STATE_PREDICATE_ATTEMPTS {
            let Some(record) = self.records.get(&key).await? else {
                return Err(Error::new(ErrorCode::NotFound, "stored response not found"));
            };
            let mut response = decode(&record.value)?;
            let current = response.status;
            if !allowed.contains(&current) || !current.can_transition_to(next) {
                return Err(Error::new(
                    ErrorCode::Conflict,
                    format!("response status {current} cannot transition to {next}"),
                ));
            }
            response.status = next;
            let value = encode(&response)?;
            let write = RecordWrite {
                value: &value,
                state: state_of(next)?,
                retention: self.retention_for(next)?,
                condition: Precondition::Version(record.version),
            };
            if matches!(self.records.put(&key, write).await?, PutOutcome::Written(_)) {
                return Ok(response);
            }
        }
        Err(Error::new(
            ErrorCode::Conflict,
            "stored response kept changing during a status transition",
        ))
    }
}

/// The record state label of a status.
fn state_of(status: ResponseStatus) -> Result<RecordState> {
    let index = STATUSES
        .iter()
        .position(|candidate| *candidate == status)
        .ok_or_else(|| Error::invariant(format!("status {status} has no record state")))?;
    RecordState::new(u8::try_from(index).map_err(|_| Error::invariant("status label overflow"))?)
}

fn terminal_states() -> Result<StateSet> {
    let states = STATUSES
        .iter()
        .filter(|status| status.is_terminal())
        .map(|status| state_of(*status))
        .collect::<Result<Vec<_>>>()?;
    Ok(StateSet::of(&states))
}

impl ResponseStore for RecordResponseStore {
    fn capabilities(&self) -> ResponseStoreCapabilities {
        let records = self.records.capabilities();
        ResponseStoreCapabilities {
            retrieve: true,
            previous_response_id: true,
            background: true,
            cancel: true,
            event_replay: true,
            persistent: records.durable,
            shared: records.shared,
            max_record_bytes: self.retention.max_record_bytes,
        }
    }

    fn get<'a>(&'a self, id: &'a ResponseId) -> ResponseFuture<'a, Option<StoredResponse>> {
        Box::pin(self.get_response(id))
    }

    fn put(&self, response: StoredResponse, condition: WriteCondition) -> ResponseFuture<'_, ()> {
        Box::pin(self.put_response(response, condition))
    }

    fn delete<'a>(&'a self, id: &'a ResponseId) -> ResponseFuture<'a, ()> {
        Box::pin(async move { self.records.delete(&self.key(id)?).await })
    }

    fn transition<'a>(
        &'a self,
        id: &'a ResponseId,
        allowed: &'a [ResponseStatus],
        next: ResponseStatus,
    ) -> ResponseFuture<'a, StoredResponse> {
        Box::pin(self.transition_response(id, allowed, next))
    }

    fn append_event<'a>(
        &'a self,
        id: &'a ResponseId,
        event: ResponseEvent,
    ) -> ResponseFuture<'a, ()> {
        Box::pin(async move {
            let key = self.key(id)?;
            self.records
                .append(
                    &key,
                    event.sequence,
                    &event.payload,
                    self.retention.max_events,
                )
                .await
                .map_err(|error| match error.code {
                    ErrorCode::NotFound => {
                        Error::new(ErrorCode::NotFound, "stored response not found")
                    }
                    _ => error,
                })
        })
    }

    fn events<'a>(
        &'a self,
        id: &'a ResponseId,
        after: Option<u64>,
    ) -> ResponseFuture<'a, Vec<ResponseEvent>> {
        Box::pin(async move {
            let log = self.records.log(&self.key(id)?, after).await?;
            Ok(log
                .into_iter()
                .map(|entry| ResponseEvent {
                    sequence: entry.sequence,
                    payload: entry.value,
                })
                .collect())
        })
    }
}
