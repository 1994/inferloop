//! The model side of the agentic layer: admitted generations and their cancellation.
use crate::document::ResponseDocument;
use infer_core::{RequestId, Result};
use std::future::Future;

/// Runs admitted generations for one protocol request.
///
/// The protocol adapter implements this over the engine runtime: it has already encoded and
/// admitted the generation, and it turns the completed generation into its own wire document.
/// Tests substitute a scripted implementation.
pub trait Upstream: Send + Sync + 'static {
    /// One generation the engine has admitted.
    type Admitted: Send + 'static;
    /// The protocol response body this upstream produces.
    type Document: ResponseDocument;

    /// Wait for an admitted generation to finish and build its response body.
    ///
    /// # Errors
    /// Returns the backend, decoding or parsing error that ended the generation.
    fn complete(
        &self,
        admitted: Self::Admitted,
    ) -> impl Future<Output = Result<Self::Document>> + Send;

    /// Cancel an engine request.
    ///
    /// # Errors
    /// Returns an error when the request is unknown or has already stopped.
    fn cancel(&self, request: RequestId) -> impl Future<Output = Result<()>> + Send;
}
