//! Server HTTP adapter.
use super::{RuntimeHandle, logging::log_request, router, router_with_text_and_media};
use crate::media::MediaService;
use axum::middleware::from_fn;
use infer_core::{Error, ErrorCode, Result};
use std::sync::Arc;
use tokio::net::TcpListener;

/// # Errors
/// Returns an I/O error if the HTTP server fails.
pub async fn serve(
    listener: TcpListener,
    handle: RuntimeHandle,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    axum::serve(listener, router(handle).layer(from_fn(log_request)))
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))
}
/// # Errors
/// Returns an I/O error if the HTTP server fails.
pub async fn serve_with_text(
    listener: TcpListener,
    handle: RuntimeHandle,
    assets: Arc<infer_models::TextAssets>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    serve_with_text_and_media(listener, handle, assets, None, shutdown).await
}
/// # Errors
/// Returns an I/O error if the HTTP server fails.
pub async fn serve_with_text_and_media(
    listener: TcpListener,
    handle: RuntimeHandle,
    assets: Arc<infer_models::TextAssets>,
    media: Option<Arc<MediaService>>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    axum::serve(
        listener,
        router_with_text_and_media(handle, assets, media).layer(from_fn(log_request)),
    )
    .with_graceful_shutdown(shutdown)
    .await
    .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))
}
