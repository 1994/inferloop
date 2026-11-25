//! Server commands.
#[cfg(cuda_backend)]
use super::tuning_worker;
use super::{ServeOptions, axum_serve, backend};
#[cfg(cuda_backend)]
use crate::support::flush_logs;
use crate::support::{configured_engine, read_json};
use infer_core::ErrorCode;
use infer_core::{Error, Result};
use infer_spi::{BackendProvider, ResponsesExtensions};
use infer_storage::{MemoryRecordStore, RecordResponseStore, ResponseRetention};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;

pub fn serve(options: ServeOptions, backend_choice: backend::Selection) -> Result<()> {
    #[cfg(cuda_backend)]
    let (mut backend_choice, tuning_worker) = tuning_worker::prepare(backend_choice)?;
    #[cfg(not(cuda_backend))]
    let mut backend_choice = backend_choice;
    // The server pays the capture before it listens rather than inside its first concurrent
    // request. A tuning worker captures too: pool geometries are only measured by the process
    // that builds them with autotune on, and the server only reads what the worker cached.
    backend_choice.pool_capture = backend::PoolCapture::AtLoad;
    #[cfg(cuda_backend)]
    let device = backend_choice.device;
    let ServeOptions {
        listen,
        config: config_path,
        package,
        mm_processor_kwargs,
        image_decoder,
        host_memory_mib,
        enable_response_store,
    } = options;
    let responses = if enable_response_store {
        ResponsesExtensions::new().with_store(Arc::new(RecordResponseStore::new(
            Arc::new(MemoryRecordStore::default()),
            ResponseRetention::default(),
        )?))
    } else {
        ResponsesExtensions::default()
    };
    let image_options = mm_processor_kwargs
        .as_deref()
        .map(infer_frontdoor::ImageProcessingOptions::parse)
        .transpose()?;
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
        tuning_worker::record_prepared(
            device,
            engine
                .backend()
                .execution_profile()
                .and_then(|profile| profile.concurrency.as_ref().map(|limit| limit.selected)),
        );
        tracing::info!(target: "infer::tuning", "CUDA tuning preparation complete; releasing preparation resources");
        flush_logs();
        return Ok(());
    }
    let source = package
        .as_deref()
        .map(infer_models::PackageSource::open)
        .transpose()?;
    let assets = text_assets(engine.model().max_sequence, source.as_ref())?;
    let image_preprocessing = engine.backend().image_preprocessing();
    if !text_only
        && image_decoder == super::ImageDecoder::Auto
        && let Some(capability) = &image_preprocessing
    {
        capability.initialize_encoded()?;
    }
    let media = if text_only {
        None
    } else {
        source
            .as_ref()
            .map(infer_models::MultimodalAssets::from_source)
            .transpose()?
            .flatten()
            .map(|assets| {
                let mut media = infer_frontdoor::MediaIngest::new(assets, media_limits)
                    .with_image_decoder(match image_decoder {
                        super::ImageDecoder::Auto => infer_frontdoor::ImageDecoder::Auto,
                        super::ImageDecoder::Cpu => infer_frontdoor::ImageDecoder::Cpu,
                    });
                if let Some(options) = image_options {
                    media = media.with_image_options(options);
                }
                let media = match image_preprocessing {
                    Some(capability) => media.with_image_preprocessing(capability),
                    None => media,
                };
                Arc::new(media)
            })
    };
    // The text and media assets no longer borrow the source; release its cached metadata before
    // the server starts, so a raw tokenizer does not stay resident beside the parsed one.
    drop(source);
    Runtime::new()
        .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))?
        .block_on(async move {
            let listener = TcpListener::bind(listen)
                .await
                .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))?;
            let handle = infer_frontdoor::RuntimeHandle::start(engine)?;
            let shutdown = handle.clone();
            tracing::info!(target: "infer::serve", listen = %listen, "listening for requests");
            axum_serve(listener, handle, shutdown, assets, media, responses).await
        })?;

    Ok(())
}

/// Open the package's text assets only when its detected tokenizer is present.
fn text_assets(
    max_sequence: usize,
    source: Option<&infer_models::PackageSource>,
) -> Result<Option<Arc<infer_models::TextAssets>>> {
    source
        .filter(|source| source.has_text_assets())
        .map(|source| {
            infer_models::TextAssets::from_source(
                infer_models::default_registry(),
                source,
                max_sequence,
            )
        })
        .transpose()
        .map(|assets| assets.map(Arc::new))
}
