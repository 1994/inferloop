//! Shared asynchronous transport for remote multimodal sources.
use super::{MAX_VIDEO_BYTES, MediaRequest};
use crate::cpu::CpuPool;
use futures_util::future::try_join_all;
use infer_core::Result;
use infer_ir::Modality;
use infer_media_input::HttpSource;
use tokio::sync::OnceCell;

static CLIENT: OnceCell<HttpSource> = OnceCell::const_new();

async fn client(pool: Option<&CpuPool>) -> Result<&'static HttpSource> {
    CLIENT
        .get_or_try_init(|| async {
            if let Some(pool) = pool {
                pool.run(0, |_| HttpSource::new()).await
            } else {
                HttpSource::new()
            }
        })
        .await
}

/// Download remote inputs concurrently while leaving inline inputs for the CPU worker.
pub(super) async fn resolve(
    requests: &[MediaRequest],
    pool: &CpuPool,
) -> Result<Vec<Option<Vec<u8>>>> {
    let has_remote = requests
        .iter()
        .any(|request| request.url.starts_with("http://") || request.url.starts_with("https://"));
    if !has_remote {
        return Ok(vec![None; requests.len()]);
    }
    let client = client(Some(pool)).await?;
    try_join_all(requests.iter().map(|request| async move {
        if request.url.starts_with("http://") || request.url.starts_with("https://") {
            let limit = (request.modality == Modality::Video).then_some(MAX_VIDEO_BYTES);
            client.fetch(&request.url, limit).await.map(Some)
        } else {
            Ok(None)
        }
    }))
    .await
}

pub(super) async fn fetch(url: &str, limit: Option<usize>) -> Result<Vec<u8>> {
    client(None).await?.fetch(url, limit).await
}

#[cfg(test)]
#[path = "../../tests/unit/media_source.rs"]
mod tests;
