//! Response lifecycles over an installed store: retention, chaining, background runs, cancel.
use crate::constants::MAX_BACKGROUND_RESPONSES;
use crate::document::Retention;
use crate::persist::StoreLease;
use crate::upstream::Upstream;
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_spi::{
    ResponseId, ResponseStatus, ResponseStore, ResponseStoreCapabilities, ResponsesExtensions,
    StoredResponse,
};
use std::future::Future;
use std::result::Result as StdResult;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// The response lifecycle service shared by every Responses request of one deployment.
///
/// Cloning is cheap: the store and the background capacity are shared.
#[derive(Clone)]
pub struct Responses {
    extensions: ResponsesExtensions,
    background: Arc<Semaphore>,
}

/// An admitted background generation with the store identity and initial body it starts with.
pub struct BackgroundStart<U: Upstream> {
    pub admitted: U::Admitted,
    pub retention: Retention<U::Document>,
    pub initial: U::Document,
}

impl Responses {
    #[must_use]
    pub fn new(extensions: ResponsesExtensions) -> Self {
        Self {
            extensions,
            background: Arc::new(Semaphore::new(MAX_BACKGROUND_RESPONSES)),
        }
    }

    /// Whether a response store is installed.
    #[must_use]
    pub fn has_store(&self) -> bool {
        self.extensions.store().is_some()
    }

    /// Capabilities of the installed store, or the defaults when none is installed.
    #[must_use]
    pub fn capabilities(&self) -> ResponseStoreCapabilities {
        self.extensions.capabilities().unwrap_or_default()
    }

    fn require_store(&self, purpose: &str) -> Result<&Arc<dyn ResponseStore>> {
        self.extensions.store().ok_or_else(|| {
            Error::unsupported(format!("{purpose} needs an installed response store"))
        })
    }

    /// Load the response a chained request continues.
    ///
    /// # Errors
    /// Returns unsupported without a store, not-found for an unknown id, conflict for a
    /// response that did not complete or end incomplete, and invalid input for a malformed id
    /// or a different model.
    pub async fn prior(
        &self,
        previous: Option<&str>,
        model: &str,
    ) -> Result<Option<StoredResponse>> {
        let Some(previous) = previous else {
            return Ok(None);
        };
        let store = self.require_store("previous_response_id")?;
        let id = ResponseId::new(previous)?;
        let stored = store.get(&id).await?.ok_or_else(|| {
            Error::new(ErrorCode::NotFound, "response not found; see store policy")
        })?;
        if !matches!(
            stored.status,
            ResponseStatus::Completed | ResponseStatus::Incomplete
        ) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "previous response must have completed or ended incomplete",
            ));
        }
        if stored.model != model {
            return Err(Error::invalid(
                "previous_response_id belongs to a different model",
            ));
        }
        Ok(Some(stored))
    }

    /// Read a retained response.
    ///
    /// # Errors
    /// Returns unsupported without a store, invalid input for a malformed id and not-found for
    /// an unknown one.
    pub async fn retrieve(&self, id: &str) -> Result<StoredResponse> {
        let store = self.require_store("response retrieval")?;
        let id = ResponseId::new(id)?;
        store
            .get(&id)
            .await?
            .ok_or_else(|| Error::new(ErrorCode::NotFound, "response not found"))
    }

    /// Mark an active response cancelled, then stop its generation.
    ///
    /// The store transition decides the race with completion; a generation that has already
    /// stopped is not an error.
    ///
    /// # Errors
    /// Returns unsupported without a store, not-found for an unknown id, invalid input for a
    /// terminal response, and the store's conflict when the transition loses a race.
    pub async fn cancel<U: Upstream>(&self, upstream: &U, id: &str) -> Result<StoredResponse> {
        let store = self.require_store("response cancellation")?;
        let id = ResponseId::new(id)?;
        let stored = store
            .get(&id)
            .await?
            .ok_or_else(|| Error::new(ErrorCode::NotFound, "response not found"))?;
        if stored.status.is_terminal() {
            return Err(Error::invalid("a terminal response cannot be cancelled"));
        }
        let updated = store
            .transition(
                &id,
                &[ResponseStatus::Queued, ResponseStatus::InProgress],
                ResponseStatus::Cancelled,
            )
            .await?;
        if let Err(error) = upstream.cancel(updated.request_id).await {
            tracing::debug!(
                target: "infer::agentic",
                error = %error,
                "response was marked cancelled after generation had already stopped"
            );
        }
        Ok(updated)
    }

    /// Reserve capacity, admit a generation, retain it and finish it on its own task.
    ///
    /// Capacity is reserved before `admit` runs, so a full deployment rejects the request
    /// without admitting a generation. The task keeps the reservation until its terminal write
    /// is done; clients poll the store for the result.
    ///
    /// # Errors
    /// Returns unsupported without a store, capacity when every background slot is busy, the
    /// admission error unchanged, or the initial store write error (after cancelling the
    /// generation).
    pub async fn start_background<U: Upstream, E: From<Error>>(
        &self,
        upstream: U,
        admit: impl Future<Output = StdResult<BackgroundStart<U>, E>>,
    ) -> StdResult<U::Document, E> {
        self.require_store("background")?;
        let permit = Arc::clone(&self.background)
            .try_acquire_owned()
            .map_err(|_| {
                Error::new(
                    ErrorCode::Capacity,
                    "background response capacity is exhausted",
                )
            })?;
        let BackgroundStart {
            admitted,
            retention,
            initial,
        } = admit.await?;
        let lease = self.retain(&upstream, retention, &initial).await?;
        tokio::spawn(async move {
            let _permit = permit;
            finish_background(&upstream, admitted, lease).await;
        });
        Ok(initial)
    }

    /// Write the initial in-progress snapshot of an admitted response and arm its cleanup.
    ///
    /// A store failure cancels the admitted generation, so nothing runs unrecorded.
    ///
    /// # Errors
    /// Returns unsupported without a store, or the store's capacity or backend error.
    pub async fn retain<U: Upstream>(
        &self,
        upstream: &U,
        retention: Retention<U::Document>,
        initial: &U::Document,
    ) -> Result<StoreLease<U::Document>> {
        let store = self.require_store("store")?;
        let request = retention.request;
        let mut lease = StoreLease::new(Arc::clone(store), retention);
        if let Err(error) = lease.start(initial).await {
            cancel_after_store_failure(upstream, request).await;
            return Err(error);
        }
        Ok(lease)
    }

    /// Run an admitted generation to completion and record its terminal state.
    ///
    /// # Errors
    /// Returns the generation error (after recording the failure) or the terminal store write
    /// error.
    pub async fn complete<U: Upstream>(
        &self,
        upstream: &U,
        admitted: U::Admitted,
        lease: Option<StoreLease<U::Document>>,
    ) -> Result<U::Document> {
        let mut lease = lease;
        let value = match upstream.complete(admitted).await {
            Ok(value) => value,
            Err(error) => {
                if let Some(lease) = &mut lease
                    && let Err(store_error) = lease.fail(&error).await
                {
                    tracing::error!(
                        target: "infer::agentic",
                        error = %store_error,
                        "failed to retain a failed response"
                    );
                }
                return Err(error);
            }
        };
        if let Some(lease) = &mut lease {
            lease.finish(&value).await?;
        }
        Ok(value)
    }
}

/// Record a background generation's terminal state; no client waits, so store errors are logged.
async fn finish_background<U: Upstream>(
    upstream: &U,
    admitted: U::Admitted,
    mut lease: StoreLease<U::Document>,
) {
    match upstream.complete(admitted).await {
        Ok(value) => {
            if let Err(error) = lease.finish(&value).await {
                tracing::error!(
                    target: "infer::agentic",
                    error = %error,
                    "failed to retain a completed background response"
                );
            }
        }
        Err(error) => {
            if let Err(store_error) = lease.fail(&error).await {
                tracing::error!(
                    target: "infer::agentic",
                    error = %store_error,
                    "failed to retain a failed background response"
                );
            }
        }
    }
}

/// Best effort: the store failure is what the caller reports, so a cancel failure is only logged.
async fn cancel_after_store_failure<U: Upstream>(upstream: &U, request: RequestId) {
    if let Err(error) = upstream.cancel(request).await {
        tracing::warn!(target: "infer::agentic", %error, "generation cancel after a store failure failed");
    }
}
