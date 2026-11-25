//! Persistence of one response through an installed [`ResponseStore`].
use crate::document::{FailedDocument, ResponseDocument, Retention};
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_spi::{
    RESPONSES_PAYLOAD_VERSION, ResponseId, ResponseStatus, ResponseStore, StoredResponse,
    WriteCondition,
};
use std::sync::Arc;
use tokio::runtime::Handle;

/// One response's retained state handles. Cloneable so a background task or the disconnect
/// cleanup can finish the store write after the protocol response has returned.
struct StoreSink<D> {
    store: Arc<dyn ResponseStore>,
    id: ResponseId,
    request: RequestId,
    model: String,
    created: u64,
    previous: Option<ResponseId>,
    request_input: Vec<u8>,
    failed: FailedDocument<D>,
}

impl<D> Clone for StoreSink<D> {
    fn clone(&self) -> Self {
        Self {
            store: Arc::clone(&self.store),
            id: self.id.clone(),
            request: self.request,
            model: self.model.clone(),
            created: self.created,
            previous: self.previous.clone(),
            request_input: self.request_input.clone(),
            failed: Arc::clone(&self.failed),
        }
    }
}

impl<D: ResponseDocument> StoreSink<D> {
    fn new(store: Arc<dyn ResponseStore>, retention: Retention<D>) -> Self {
        Self {
            store,
            id: retention.id,
            request: retention.request,
            model: retention.model,
            created: retention.created,
            previous: retention.previous,
            request_input: retention.request_input,
            failed: retention.failed,
        }
    }

    /// Persist the terminal response unless it was cancelled while generating.
    async fn finish(&self, value: &D) -> Result<()> {
        if let Err(error) = self
            .put(
                value,
                WriteCondition::UnlessStatus(ResponseStatus::Cancelled),
            )
            .await
        {
            let persist_error = Error::new(
                ErrorCode::Backend,
                format!("failed to persist a completed response: {error}"),
            );
            if self.fail(&persist_error).await.is_err()
                && let Err(error) = self.store.delete(&self.id).await
            {
                tracing::warn!(target: "infer::agentic", %error, "unpersisted response could not be removed");
            }
            return Err(error);
        }
        // The conditional write is a no-op when another writer already recorded a terminal
        // state (cancel, or a disconnect cleanup). Report that so the caller does not publish a
        // completion for a response another writer terminated.
        if self.store.get(&self.id).await?.is_some_and(|stored| {
            matches!(
                stored.status,
                ResponseStatus::Cancelled | ResponseStatus::Failed
            )
        }) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "response was terminated while completing",
            ));
        }
        Ok(())
    }

    /// Persist a failed response unless it was cancelled while generating.
    async fn fail(&self, error: &Error) -> Result<()> {
        let value = (self.failed)(error);
        match self
            .put(
                &value,
                WriteCondition::UnlessStatus(ResponseStatus::Cancelled),
            )
            .await
        {
            Ok(()) => Ok(()),
            Err(error) => {
                if let Err(error) = self.store.delete(&self.id).await {
                    tracing::warn!(target: "infer::agentic", %error, "unpersisted response could not be removed");
                }
                Err(error)
            }
        }
    }

    async fn put(&self, value: &D, condition: WriteCondition) -> Result<()> {
        let record = self.record(value)?;
        let limit = self.store.capabilities().max_record_bytes;
        let bytes = record
            .request_input
            .len()
            .checked_add(record.response.len())
            .ok_or_else(|| Error::invalid("stored response size overflow"))?;
        if bytes > limit {
            return Err(Error::new(
                ErrorCode::Capacity,
                "stored response exceeds the store record limit",
            ));
        }
        self.store.put(record, condition).await
    }

    fn record(&self, value: &D) -> Result<StoredResponse> {
        let metadata = value.metadata()?;
        let record = StoredResponse {
            id: self.id.clone(),
            request_id: self.request,
            model: self.model.clone(),
            created_at: self.created,
            completed_at: metadata.completed_at,
            status: metadata.status,
            incomplete: metadata.incomplete,
            error: metadata.error,
            usage: metadata.usage,
            previous_response_id: self.previous.clone(),
            request_input: self.request_input.clone(),
            response: value.encode()?,
            payload_version: RESPONSES_PAYLOAD_VERSION,
        };
        record.validate()?;
        Ok(record)
    }
}

/// Armed persistence guard for one response.
///
/// If the owner is dropped before a terminal write (a client disconnect drops the HTTP future
/// or SSE stream), the guard spawns a best-effort failure write, so the store never keeps an
/// active record forever merely because a client went away.
pub struct StoreLease<D: ResponseDocument> {
    sink: StoreSink<D>,
    armed: bool,
}

impl<D: ResponseDocument> StoreLease<D> {
    pub(crate) fn new(store: Arc<dyn ResponseStore>, retention: Retention<D>) -> Self {
        Self {
            sink: StoreSink::new(store, retention),
            armed: false,
        }
    }

    /// The engine request this lease retains.
    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.sink.request
    }

    /// Write the initial snapshot and arm the disconnect cleanup.
    pub(crate) async fn start(&mut self, initial: &D) -> Result<()> {
        self.armed = true;
        let result = self.sink.put(initial, WriteCondition::Always).await;
        if result.is_err() {
            self.armed = false;
        }
        result
    }

    /// Write the terminal snapshot and disarm cleanup.
    ///
    /// # Errors
    /// Returns a capacity or backend error from the store, or a conflict when another writer
    /// (cancellation or disconnect cleanup) already terminated the response.
    pub async fn finish(&mut self, value: &D) -> Result<()> {
        let result = self.sink.finish(value).await;
        self.armed = false;
        result
    }

    /// Write a failed snapshot and disarm cleanup.
    ///
    /// # Errors
    /// Returns a capacity or backend error from the store.
    pub async fn fail(&mut self, error: &Error) -> Result<()> {
        let result = self.sink.fail(error).await;
        self.armed = false;
        result
    }
}

impl<D: ResponseDocument> Drop for StoreLease<D> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let sink = self.sink.clone();
        if let Ok(handle) = Handle::try_current() {
            handle.spawn(async move {
                let error = Error::new(
                    ErrorCode::Backend,
                    "response was abandoned before completion",
                );
                if let Err(store_error) = sink.fail(&error).await {
                    tracing::error!(
                        target: "infer::agentic",
                        error = %store_error,
                        "failed to retain an abandoned response"
                    );
                }
            });
        }
    }
}
