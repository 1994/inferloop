//! The protocol response body as the agentic layer sees it: typed metadata plus opaque bytes.
use infer_core::{Error, RequestId, Result};
use infer_spi::{IncompleteReason, ResponseError, ResponseId, ResponseStatus, ResponseUsage};
use std::sync::Arc;

/// Store metadata carried by a response body at one point in its lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentMetadata {
    pub status: ResponseStatus,
    pub completed_at: Option<u64>,
    pub incomplete: Option<IncompleteReason>,
    pub error: Option<ResponseError>,
    pub usage: Option<ResponseUsage>,
}

/// A protocol response body persisted without this crate knowing its wire shape.
///
/// The protocol adapter owns the payload encoding and its `payload_version`; this crate only
/// reads the typed metadata a store record needs and stores the encoded bytes.
pub trait ResponseDocument: Send + Sync + 'static {
    /// The store metadata this body reports.
    ///
    /// # Errors
    /// Returns an invariant error when the body carries a status or reason the store contract
    /// cannot represent.
    fn metadata(&self) -> Result<DocumentMetadata>;

    /// The stored payload bytes.
    ///
    /// # Errors
    /// Returns an invalid-input error when the body cannot be serialized.
    fn encode(&self) -> Result<Vec<u8>>;
}

/// Builds the failed body for an error; shared with the disconnect cleanup task.
pub type FailedDocument<D> = Arc<dyn Fn(&Error) -> D + Send + Sync>;

/// Everything needed to retain one admitted response in a store.
pub struct Retention<D> {
    /// Identifier the record is stored under.
    pub id: ResponseId,
    /// Engine request generating the response; cancellation targets it.
    pub request: RequestId,
    pub model: String,
    pub created: u64,
    pub previous: Option<ResponseId>,
    /// Opaque request input a chained request replays.
    pub request_input: Vec<u8>,
    /// Builds the failed body for a generation, store or disconnect failure.
    pub failed: FailedDocument<D>,
}
