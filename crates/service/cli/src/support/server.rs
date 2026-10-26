//! Server CLI support.
use infer_core::Result;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal::ctrl_c;

pub async fn axum_serve(
    listener: TcpListener,
    handle: infer_frontdoor::RuntimeHandle,
    shutdown: infer_frontdoor::RuntimeHandle,
    assets: Option<Arc<infer_models::TextAssets>>,
    media: Option<Arc<infer_frontdoor::MediaService>>,
) -> Result<()> {
    let stop = async move {
        let _ = ctrl_c().await;
        let _ = shutdown.shutdown().await;
    };
    if let Some(assets) = assets {
        infer_frontdoor::serve_with_text_and_media(listener, handle, assets, media, stop).await
    } else {
        infer_frontdoor::serve(listener, handle, stop).await
    }
}
