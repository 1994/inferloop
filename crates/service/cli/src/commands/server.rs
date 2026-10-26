//! Server commands.
use super::{ServeOptions, axum_serve, backend};
#[cfg(cuda_backend)]
use crate::support::flush_logs;
use crate::support::{configured_engine, read_json};
use infer_core::ErrorCode;
use infer_core::{Error, Result};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;

pub fn serve(options: ServeOptions, backend_choice: backend::Selection) -> Result<()> {
    #[cfg(cuda_backend)]
    let (backend_choice, tuning_worker) = super::tuning_worker::prepare(backend_choice)?;
    let ServeOptions {
        listen,
        config: config_path,
        package,
        host_memory_mib,
    } = options;
    let media_limits = backend_choice.media_limits;
    let text_only = backend_choice.text_only;
    let engine = configured_engine(
        config_path.as_deref().map(read_json).transpose()?,
        None,
        package.as_deref(),
        host_memory_mib,
        backend_choice,
    )?;
    #[cfg(cuda_backend)]
    if tuning_worker {
        tracing::info!(target: "infer::tuning", "CUDA tuning preparation complete; releasing preparation resources");
        flush_logs();
        return Ok(());
    }
    let assets = package
        .as_deref()
        .filter(|p| p.join("tokenizer.json").exists())
        .map(|p| infer_models::TextAssets::open(p, engine.model().max_sequence))
        .transpose()?
        .map(Arc::new);
    let media = if text_only {
        None
    } else {
        package
            .as_deref()
            .map(infer_models::MultimodalAssets::open)
            .transpose()?
            .flatten()
            .map(|assets| Arc::new(infer_frontdoor::MediaService::new(assets, media_limits)))
    };
    Runtime::new()
        .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))?
        .block_on(async move {
            let listener = TcpListener::bind(listen)
                .await
                .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))?;
            let handle = infer_frontdoor::RuntimeHandle::start(engine)?;
            let shutdown = handle.clone();
            tracing::info!(target: "infer::serve", listen = %listen, "listening for requests");
            axum_serve(listener, handle, shutdown, assets, media).await
        })?;

    Ok(())
}
