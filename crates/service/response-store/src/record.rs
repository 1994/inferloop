//! Retention limits and the row form of a stored response, shared by the SQL stores.
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_spi::{
    IncompleteReason, ResponseError, ResponseId, ResponseStatus, ResponseUsage, StoredResponse,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Default largest single record, matching the in-process store.
const DEFAULT_MAX_RECORD_BYTES: usize = 4 * 1024 * 1024;
/// Default retained background event count per response, matching the in-process store.
const DEFAULT_MAX_EVENTS: usize = 4096;
/// Default retention of a terminal response after its last write, matching the in-process store.
const DEFAULT_TTL: Duration = Duration::from_secs(60 * 60);

/// Token counters retained with a response: input, output, reasoning, cached and cache write.
pub const USAGE_COLUMNS: usize = 5;

/// Version of the table layout the SQL stores create; a database at another version is refused.
pub const SCHEMA_VERSION: i64 = 1;

/// The statuses after which a response never changes again; only these expire.
pub const TERMINAL: [ResponseStatus; 4] = [
    ResponseStatus::Completed,
    ResponseStatus::Incomplete,
    ResponseStatus::Failed,
    ResponseStatus::Cancelled,
];

/// Retention limits of a persistent response store.
///
/// Unlike the in-process store there is no entry or total byte budget: the database owns its
/// capacity. Terminal responses expire `ttl` after their last write; active ones never expire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistentStoreConfig {
    /// Largest `request_input + response` payload of one record, in bytes.
    pub max_record_bytes: usize,
    /// Background events retained per response.
    pub max_events: usize,
    /// `None` keeps terminal responses until they are deleted.
    pub ttl: Option<Duration>,
}

impl Default for PersistentStoreConfig {
    fn default() -> Self {
        Self {
            max_record_bytes: DEFAULT_MAX_RECORD_BYTES,
            max_events: DEFAULT_MAX_EVENTS,
            ttl: Some(DEFAULT_TTL),
        }
    }
}

impl PersistentStoreConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.max_record_bytes == 0 || self.max_events == 0 {
            return Err(Error::invalid("response store limits must be nonzero"));
        }
        if self.ttl.is_some_and(|ttl| ttl.is_zero()) {
            return Err(Error::invalid("response store retention must be nonzero"));
        }
        Ok(())
    }

    /// When a record written now expires once terminal, in Unix milliseconds.
    pub(crate) fn expires_at(&self) -> Result<Option<i64>> {
        self.ttl
            .map(|ttl| {
                let ttl = i64::try_from(ttl.as_millis())
                    .map_err(|_| Error::invalid("response store retention is too long"))?;
                now_millis()?
                    .checked_add(ttl)
                    .ok_or_else(|| Error::invalid("response store retention is too long"))
            })
            .transpose()
    }

    pub(crate) fn check_size(&self, response: &StoredResponse) -> Result<()> {
        let bytes = response
            .request_input
            .len()
            .checked_add(response.response.len())
            .ok_or_else(|| Error::invalid("stored response size overflow"))?;
        if bytes > self.max_record_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "stored response exceeds the per-record byte limit",
            ));
        }
        Ok(())
    }
}

/// The current time in Unix milliseconds.
pub fn now_millis() -> Result<i64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::invariant("system clock is before the Unix epoch"))?;
    i64::try_from(elapsed.as_millis()).map_err(|_| Error::invariant("system clock overflow"))
}

/// The statuses a response may currently have for `transition(allowed, next)` to succeed.
pub fn sources(allowed: &[ResponseStatus], next: ResponseStatus) -> Vec<&'static str> {
    allowed
        .iter()
        .filter(|status| status.can_transition_to(next))
        .map(|status| status.as_str())
        .collect()
}

/// The error for a transition whose response exists but is in `current`.
pub fn transition_conflict(current: &str, next: ResponseStatus) -> Error {
    Error::new(
        ErrorCode::Conflict,
        format!("response status {current} cannot transition to {next}"),
    )
}

pub fn not_found() -> Error {
    Error::new(ErrorCode::NotFound, "stored response not found")
}

/// The error for an event sequence that does not follow the last one.
pub fn unordered_event() -> Error {
    Error::invalid("response events must have increasing sequence numbers")
}

pub fn events_full() -> Error {
    Error::new(ErrorCode::Capacity, "response event retention is full")
}

/// Convert an unsigned count or identifier to the signed integer SQL stores.
pub fn to_sql(value: u64, what: &str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| Error::invalid(format!("{what} {value} is too large to store")))
}

pub fn from_sql(value: i64, what: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::invariant(format!("stored {what} {value} is negative")))
}

/// One stored response as SQL column values.
#[derive(Debug)]
pub struct Row {
    pub id: String,
    pub request_id: i64,
    pub model: String,
    pub created_at: i64,
    pub completed_at: Option<i64>,
    pub status: &'static str,
    pub incomplete: Option<&'static str>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    /// Input, output, reasoning, cached and cache-write tokens, when usage was recorded.
    pub usage: Option<[i64; USAGE_COLUMNS]>,
    pub previous_response_id: Option<String>,
    pub request_input: Vec<u8>,
    pub response: Vec<u8>,
    pub payload_version: i64,
}

/// One row as read back, with owned text columns.
#[derive(Debug)]
pub struct ReadRow {
    pub id: String,
    pub request_id: i64,
    pub model: String,
    pub created_at: i64,
    pub completed_at: Option<i64>,
    pub status: String,
    pub incomplete: Option<String>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub usage: [Option<i64>; USAGE_COLUMNS],
    pub previous_response_id: Option<String>,
    pub request_input: Vec<u8>,
    pub response: Vec<u8>,
    pub payload_version: i64,
}

/// The selected columns, in [`ReadRow`] order.
pub const COLUMNS: &str = "id, request_id, model, created_at, completed_at, status, incomplete, \
    error_code, error_message, input_tokens, output_tokens, reasoning_tokens, cached_tokens, \
    cache_write_tokens, previous_response_id, request_input, response, payload_version";

impl Row {
    pub fn new(response: StoredResponse) -> Result<Self> {
        let usage = response
            .usage
            .map(|usage| {
                Ok::<_, Error>([
                    to_sql(usage.input_tokens, "input tokens")?,
                    to_sql(usage.output_tokens, "output tokens")?,
                    to_sql(usage.reasoning_tokens, "reasoning tokens")?,
                    to_sql(usage.cached_tokens, "cached tokens")?,
                    to_sql(usage.cache_write_tokens, "cache write tokens")?,
                ])
            })
            .transpose()?;
        let (error_code, error_message) = response.error.map_or((None, None), |error| {
            (Some(error.code), Some(error.message))
        });
        Ok(Self {
            id: response.id.into(),
            request_id: to_sql(response.request_id.get(), "request id")?,
            model: response.model,
            created_at: to_sql(response.created_at, "created_at")?,
            completed_at: response
                .completed_at
                .map(|at| to_sql(at, "completed_at"))
                .transpose()?,
            status: response.status.as_str(),
            incomplete: response.incomplete.map(IncompleteReason::as_str),
            error_code,
            error_message,
            usage,
            previous_response_id: response.previous_response_id.map(String::from),
            request_input: response.request_input,
            response: response.response,
            payload_version: i64::from(response.payload_version),
        })
    }
}

impl ReadRow {
    pub fn into_response(self) -> Result<StoredResponse> {
        let error = match (self.error_code, self.error_message) {
            (Some(code), Some(message)) => Some(ResponseError::new(code, message)?),
            (None, None) => None,
            _ => return Err(Error::invariant("stored response has half an error")),
        };
        let usage = match self.usage {
            [
                Some(input),
                Some(output),
                Some(reasoning),
                Some(cached),
                Some(cache_write),
            ] => Some(ResponseUsage {
                input_tokens: from_sql(input, "input tokens")?,
                output_tokens: from_sql(output, "output tokens")?,
                reasoning_tokens: from_sql(reasoning, "reasoning tokens")?,
                cached_tokens: from_sql(cached, "cached tokens")?,
                cache_write_tokens: from_sql(cache_write, "cache write tokens")?,
            }),
            [None, None, None, None, None] => None,
            _ => return Err(Error::invariant("stored response has partial usage")),
        };
        let response = StoredResponse {
            id: ResponseId::new(self.id)?,
            request_id: RequestId::new(from_sql(self.request_id, "request id")?)?,
            model: self.model,
            created_at: from_sql(self.created_at, "created_at")?,
            completed_at: self
                .completed_at
                .map(|at| from_sql(at, "completed_at"))
                .transpose()?,
            status: self.status.parse()?,
            incomplete: self
                .incomplete
                .as_deref()
                .map(incomplete_reason)
                .transpose()?,
            error,
            usage,
            previous_response_id: self.previous_response_id.map(ResponseId::new).transpose()?,
            request_input: self.request_input,
            response: self.response,
            payload_version: u32::try_from(self.payload_version)
                .map_err(|_| Error::invariant("stored payload version is out of range"))?,
        };
        response.validate()?;
        Ok(response)
    }
}

fn incomplete_reason(value: &str) -> Result<IncompleteReason> {
    [
        IncompleteReason::MaxOutputTokens,
        IncompleteReason::MaxMessages,
        IncompleteReason::ContentFilter,
    ]
    .into_iter()
    .find(|reason| reason.as_str() == value)
    .ok_or_else(|| Error::invariant(format!("unknown stored incomplete reason {value:?}")))
}
