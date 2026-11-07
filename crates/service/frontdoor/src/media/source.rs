//! Shared asynchronous transport for remote multimodal sources.
use super::MediaRequest;
use crate::cpu::CpuPool;
use futures_util::future::try_join_all;
use infer_core::Result;
use infer_ir::Modality;
use infer_media_input::{HttpDownload, HttpSource};
use tokio::sync::OnceCell;

static CLIENT: OnceCell<HttpSource> = OnceCell::const_new();

async fn client(pool: Option<&CpuPool>) -> Result<&'static HttpSource> {
    CLIENT
        .get_or_try_init(|| async {
            if let Some(pool) = pool {
                pool.run_when_available(0, |_| HttpSource::new()).await
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
        .any(|request| request.modality != Modality::Video && is_remote(&request.url));
    if !has_remote {
        return Ok(vec![None; requests.len()]);
    }
    let client = client(Some(pool)).await?;
    try_join_all(requests.iter().map(|request| async move {
        if request.modality != Modality::Video && is_remote(&request.url) {
            client.fetch(&request.url, None).await.map(Some)
        } else {
            Ok(None)
        }
    }))
    .await
}

pub(super) fn is_remote(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

pub(super) async fn stream(url: &str, pool: &CpuPool) -> Result<HttpDownload> {
    client(Some(pool)).await?.stream(url, None).await
}

pub(super) async fn fetch(url: &str, limit: Option<usize>) -> Result<Vec<u8>> {
    client(None).await?.fetch(url, limit).await
}

#[cfg(test)]
#[path = "../../tests/unit/media_source.rs"]
mod tests;
