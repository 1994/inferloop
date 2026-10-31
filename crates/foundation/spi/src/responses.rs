//! Responses API state extension contracts.
//!
//! The `OpenAI` Responses API can retain a response so a later request can address it by
//! `previous_response_id`, retrieve it, cancel it, or poll a background job. The engine does not
//! own that policy: an installed [`ResponseStore`] owns persistence and atomicity, while the
//! frontdoor owns the wire format stored in the opaque payload fields.
//!
//! Implementations are `Send + Sync` and object-safe. Each method returns a boxed future so a
//! store can use its own runtime bridge (in-memory, Redis, Postgres, an external SDK) without the
//! engine depending on that runtime. The frontdoor calls these methods from its bounded service
//! tasks; implementations must honour cancellation by dropping the returned future and must not
//! spawn unbounded background work of their own.
use infer_core::{Error, RequestId, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;

/// Version of the Responses state extension contract.
pub const RESPONSES_SPI_VERSION: u32 = 1;
/// Version of the opaque payloads the frontdoor writes into a stored response.
///
/// The payload format is owned by the frontdoor, not by the store. A store returns the bytes it
/// was given; a future payload version is a coordinated frontdoor/store change.
pub const RESPONSES_PAYLOAD_VERSION: u32 = 1;

/// A boxed response-store operation. The frontdoor awaits these on its own task; no executor or
/// runtime type is part of the contract.
pub type ResponseFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// Stable identifier of one stored response.
///
/// The identifier is opaque to the store. It is validated as a bounded ASCII token so a store can
/// use it as a key and a caller cannot smuggle path or control characters into a backend.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ResponseId(String);

impl ResponseId {
    /// Largest accepted identifier, in bytes.
    pub const MAX_BYTES: usize = 128;
    /// Prefix the frontdoor currently generates; shown in diagnostics and error messages.
    pub const PREFIX: &'static str = "resp_";

    /// Validate and wrap a response identifier.
    ///
    /// # Errors
    /// Returns an invalid-input error for an empty, oversized, or non-token identifier.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.is_empty() || value.len() > Self::MAX_BYTES {
            return Err(Error::invalid(format!(
                "response id must be 1..{} bytes",
                Self::MAX_BYTES
            )));
        }
        if !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(Error::invalid(
                "response id must contain only ASCII letters, digits, underscores or dashes",
            ));
        }
        Ok(Self(value))
    }

    /// The token as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ResponseId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}
impl TryFrom<String> for ResponseId {
    type Error = Error;
    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}
impl From<ResponseId> for String {
    fn from(value: ResponseId) -> Self {
        value.0
    }
}
impl FromStr for ResponseId {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

/// Lifecycle state of a stored response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Incomplete,
    Failed,
    Cancelled,
}

impl ResponseStatus {
    /// Whether no further transition may occur.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Incomplete | Self::Failed | Self::Cancelled
        )
    }

    /// The wire spelling used by the `OpenAI` Responses protocol.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether a store may move from `self` to `next`.
    ///
    /// A store applies the caller's explicit allowed list as well; this method rejects impossible
    /// or backwards transitions before they reach persistence.
    #[must_use]
    pub fn can_transition_to(self, next: Self) -> bool {
        match (self, next) {
            (Self::Queued, Self::InProgress | Self::Failed | Self::Cancelled)
            | (
                Self::InProgress,
                Self::Completed | Self::Incomplete | Self::Failed | Self::Cancelled,
            ) => true,
            (current, next) => current == next,
        }
    }
}
impl fmt::Display for ResponseStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(formatter)
    }
}
impl FromStr for ResponseStatus {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "queued" => Ok(Self::Queued),
            "in_progress" => Ok(Self::InProgress),
            "completed" => Ok(Self::Completed),
            "incomplete" => Ok(Self::Incomplete),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(Error::invalid(format!("unknown response status {value:?}"))),
        }
    }
}

/// Why a response ended `incomplete`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum IncompleteReason {
    MaxOutputTokens,
    MaxMessages,
    ContentFilter,
}

impl IncompleteReason {
    /// The wire spelling used by the `OpenAI` Responses protocol.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxOutputTokens => "max_output_tokens",
            Self::MaxMessages => "max_messages",
            Self::ContentFilter => "content_filter",
        }
    }
}

/// Error attached to a failed response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}
impl ResponseError {
    /// # Errors
    /// Returns an invalid-input error for an empty code or message.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Result<Self> {
        let code = code.into();
        let message = message.into();
        if code.is_empty() || message.is_empty() {
            return Err(Error::invalid(
                "response error code and message are required",
            ));
        }
        Ok(Self { code, message })
    }
}

/// Token usage retained with a response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub cached_tokens: u64,
    pub cache_write_tokens: u64,
}
impl ResponseUsage {
    /// Sum of input and output tokens, or `None` if the sum overflows.
    #[must_use]
    pub const fn checked_total(self) -> Option<u64> {
        self.input_tokens.checked_add(self.output_tokens)
    }
}

/// One retained response and the opaque payloads needed to continue it.
///
/// `request_input` and `response` are serialized by the frontdoor. A store must treat them as
/// opaque bytes: persist them, return them unchanged, and use [`Self::validate`] only for the
/// typed metadata. This keeps protocol evolution out of every storage backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredResponse {
    pub id: ResponseId,
    /// Engine request identity, used to cancel a running generation.
    pub request_id: RequestId,
    pub model: String,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub status: ResponseStatus,
    pub incomplete: Option<IncompleteReason>,
    pub error: Option<ResponseError>,
    pub usage: Option<ResponseUsage>,
    pub previous_response_id: Option<ResponseId>,
    /// Effective conversation input, as a JSON array of Responses input items.
    pub request_input: Vec<u8>,
    /// Serialized Responses `response` object, including its `output` array.
    pub response: Vec<u8>,
    /// Payload format version; currently [`RESPONSES_PAYLOAD_VERSION`].
    pub payload_version: u32,
}

impl StoredResponse {
    /// Validate the typed metadata independently of the payload format.
    ///
    /// # Errors
    /// Returns an invalid-input error for an empty model, inconsistent terminal fields, a missing
    /// payload, or a different payload version.
    pub fn validate(&self) -> Result<()> {
        if self.model.is_empty() {
            return Err(Error::invalid("stored response model must not be empty"));
        }
        if self.request_input.is_empty() || self.response.is_empty() {
            return Err(Error::invalid(
                "stored response requires non-empty request and response payloads",
            ));
        }
        if self.payload_version != RESPONSES_PAYLOAD_VERSION {
            return Err(Error::unsupported(format!(
                "unsupported response payload version {}",
                self.payload_version
            )));
        }
        match self.status {
            ResponseStatus::Queued | ResponseStatus::InProgress => {
                if self.completed_at.is_some() || self.incomplete.is_some() || self.error.is_some()
                {
                    return Err(Error::invalid(
                        "active stored response cannot carry terminal details",
                    ));
                }
            }
            ResponseStatus::Completed => {
                if self.completed_at.is_none() || self.incomplete.is_some() || self.error.is_some()
                {
                    return Err(Error::invalid(
                        "completed stored response needs completed_at and no failure details",
                    ));
                }
            }
            ResponseStatus::Incomplete => {
                if self.incomplete.is_none() || self.error.is_some() {
                    return Err(Error::invalid(
                        "incomplete stored response needs an incomplete reason",
                    ));
                }
            }
            ResponseStatus::Failed => {
                if self.error.is_none() || self.incomplete.is_some() {
                    return Err(Error::invalid(
                        "failed stored response needs an error and no incomplete reason",
                    ));
                }
            }
            ResponseStatus::Cancelled => {
                if self.incomplete.is_some() || self.error.is_some() {
                    return Err(Error::invalid(
                        "cancelled stored response cannot carry failure details",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// One replayed background streaming event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseEvent {
    pub sequence: u64,
    /// Serialized SSE `data` JSON, opaque to the store.
    pub payload: Vec<u8>,
}

/// Capabilities an installed store advertises to the frontdoor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "capability flags are independent switches negotiated by the frontdoor"
)]
pub struct ResponseStoreCapabilities {
    pub retrieve: bool,
    pub previous_response_id: bool,
    pub background: bool,
    pub cancel: bool,
    pub event_replay: bool,
    /// Survives a process restart and can be shared by replicas.
    pub persistent: bool,
    pub shared: bool,
    /// Largest accepted `request_input + response` payload, in bytes.
    pub max_record_bytes: usize,
}

/// Conditional write that lets a store avoid overwriting a cancelled response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteCondition {
    Always,
    /// Skip the write when the stored response already has this status.
    UnlessStatus(ResponseStatus),
    /// Skip the write when the stored response already reached any terminal state.
    UnlessTerminal,
}

/// Persistence and atomic transition contract for Responses state.
///
/// The frontdoor calls [`Self::put`] for the initial and terminal snapshots, [`Self::transition`]
/// for cancel/background status changes, and [`Self::get`] for retrieval and chaining. A store
/// that advertises a capability but cannot honour it must return an `unsupported` error rather
/// than silently dropping data.
pub trait ResponseStore: Send + Sync {
    fn capabilities(&self) -> ResponseStoreCapabilities;

    /// The contract version the implementation was built against.
    fn spi_version(&self) -> u32 {
        RESPONSES_SPI_VERSION
    }

    /// Load one response.
    ///
    /// # Errors
    /// Returns a backend error for unavailable storage. A missing response is `Ok(None)`.
    fn get<'a>(&'a self, id: &'a ResponseId) -> ResponseFuture<'a, Option<StoredResponse>>;

    /// Insert or replace one response, optionally keeping a cancelled record.
    ///
    /// # Errors
    /// Returns a capacity or backend error. A condition that is not met is a successful no-op.
    fn put(&self, response: StoredResponse, condition: WriteCondition) -> ResponseFuture<'_, ()>;

    /// Remove one response and its events. Missing identifiers are a successful no-op.
    ///
    /// # Errors
    /// Returns a backend error.
    fn delete<'a>(&'a self, id: &'a ResponseId) -> ResponseFuture<'a, ()>;

    /// Atomically move one response to `next` when its current status is in `allowed`.
    ///
    /// # Errors
    /// Returns `not_found` for an unknown identifier and `conflict` when the transition is not
    /// allowed or the response is already terminal. Implementations must make this atomic across
    /// concurrent writers.
    fn transition<'a>(
        &'a self,
        id: &'a ResponseId,
        allowed: &'a [ResponseStatus],
        next: ResponseStatus,
    ) -> ResponseFuture<'a, StoredResponse>;

    /// Append one background streaming event.
    ///
    /// # Errors
    /// Returns an `unsupported` error by default, or a capacity/backend error.
    fn append_event<'a>(
        &'a self,
        id: &'a ResponseId,
        event: ResponseEvent,
    ) -> ResponseFuture<'a, ()> {
        let _ = (id, event);
        Box::pin(async { Err(Error::unsupported("response event replay is not supported")) })
    }

    /// Read background streaming events after `after`.
    ///
    /// # Errors
    /// Returns an `unsupported` error by default, or a backend error.
    fn events<'a>(
        &'a self,
        id: &'a ResponseId,
        after: Option<u64>,
    ) -> ResponseFuture<'a, Vec<ResponseEvent>> {
        let _ = (id, after);
        Box::pin(async { Err(Error::unsupported("response event replay is not supported")) })
    }
}

/// Installed Responses extensions.
///
/// The struct is built through [`Self::with_store`] and kept private so future extension families
/// can be added without changing existing constructors. An empty value preserves the stateless
/// behavior: `store`, `previous_response_id`, `background` and retrieval remain explicit errors.
#[derive(Clone, Default)]
pub struct ResponsesExtensions {
    store: Option<Arc<dyn ResponseStore>>,
}

impl ResponsesExtensions {
    /// No stateful Responses extension is installed.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Install one response store.
    #[must_use]
    pub fn with_store(mut self, store: Arc<dyn ResponseStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// The installed store, if any.
    #[must_use]
    pub fn store(&self) -> Option<&Arc<dyn ResponseStore>> {
        self.store.as_ref()
    }

    /// Store capabilities, if a store is installed.
    #[must_use]
    pub fn capabilities(&self) -> Option<ResponseStoreCapabilities> {
        self.store.as_ref().map(|store| store.capabilities())
    }

    /// Whether any stateful Responses feature is available.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.store.is_some()
    }

    /// Validate the installed extension against this contract version.
    ///
    /// # Errors
    /// Returns an unsupported error for a store built against a different contract version, or an
    /// invalid-input error for capabilities that cannot serve the routes they advertise.
    pub fn validate(&self) -> Result<()> {
        let Some(store) = &self.store else {
            return Ok(());
        };
        if store.spi_version() != RESPONSES_SPI_VERSION {
            return Err(Error::unsupported(format!(
                "response store SPI version {} is not {RESPONSES_SPI_VERSION}",
                store.spi_version()
            )));
        }
        let capabilities = store.capabilities();
        if !capabilities.retrieve {
            return Err(Error::invalid(
                "an installed response store must support retrieval",
            ));
        }
        if capabilities.max_record_bytes == 0 {
            return Err(Error::invalid(
                "an installed response store needs a record byte limit",
            ));
        }
        if capabilities.background && !capabilities.retrieve {
            return Err(Error::invalid(
                "background responses need retrieval support",
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for ResponsesExtensions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResponsesExtensions")
            .field("enabled", &self.is_enabled())
            .field("capabilities", &self.capabilities())
            .finish()
    }
}

#[cfg(test)]
#[path = "../tests/unit/responses.rs"]
mod tests;
