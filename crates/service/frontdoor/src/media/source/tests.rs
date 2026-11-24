use super::*;
use crate::cpu::{CpuConfig, CpuPool};
use axum::extract::Path;
use axum::{Router, routing::get};
use infer_core::{Error, ErrorCode};
use infer_ir::Modality;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::{Barrier, Notify, oneshot};
use tokio::time::timeout;

#[tokio::test]
async fn remote_media_downloads_run_concurrently_in_request_order() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let app = Router::new().route(
        "/{item}",
        get(move |Path(item): Path<String>| {
            let barrier = barrier.clone();
            async move {
                barrier.wait().await;
                item
            }
        }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let requests = ["first", "second"].map(|item| MediaRequest {
        modality: Modality::Image,
        url: format!("http://{address}/{item}"),
    });
    let pool = CpuPool::new(CpuConfig::default()).unwrap();
    let result = timeout(Duration::from_secs(5), resolve(&requests, &pool))
        .await
        .expect("downloads must run concurrently")
        .unwrap();
    assert_eq!(
        result,
        vec![Some(b"first".to_vec()), Some(b"second".to_vec())]
    );
    server.abort();
}

#[tokio::test]
async fn waiting_for_remote_media_leaves_cpu_workers_available() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let app = Router::new().route(
        "/media",
        get({
            let entered = entered.clone();
            let release = release.clone();
            move || {
                let entered = entered.clone();
                let release = release.clone();
                async move {
                    entered.notify_one();
                    release.notified().await;
                    "image"
                }
            }
        }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        ..CpuConfig::default()
    })
    .unwrap();
    let source_pool = pool.clone();
    let download = tokio::spawn(async move {
        resolve(
            &[MediaRequest {
                modality: Modality::Image,
                url: format!("http://{address}/media"),
            }],
            &source_pool,
        )
        .await
    });
    timeout(Duration::from_secs(5), entered.notified())
        .await
        .expect("download reached server");
    assert_eq!(
        timeout(Duration::from_secs(1), pool.run(0, |_| Ok(7)))
            .await
            .expect("network wait must not occupy the CPU worker")
            .unwrap(),
        7
    );
    release.notify_one();
    assert_eq!(
        download.await.unwrap().unwrap(),
        vec![Some(b"image".to_vec())]
    );
    server.abort();
}

#[tokio::test]
async fn remote_media_limit_and_http_errors_are_reported() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().route("/media", get(|| async { "123456" }));
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let error = fetch(&format!("http://{address}/media"), Some(5))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Capacity);
    let error = fetch(&format!("http://{address}/missing"), None)
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidInput);
    server.abort();
}

#[tokio::test]
async fn incremental_work_waits_for_worker_capacity_and_wakes() {
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        ..CpuConfig::default()
    })
    .unwrap();
    let (started, ready) = oneshot::channel();
    let (release, gate) = mpsc::channel();
    let first_pool = pool.clone();
    let first = tokio::spawn(async move {
        first_pool
            .run(0, move |_| {
                started
                    .send(())
                    .map_err(|()| Error::invariant("waiter dropped"))?;
                gate.recv()
                    .map_err(|error| Error::invariant(error.to_string()))?;
                Ok(())
            })
            .await
    });
    ready.await.unwrap();
    let next_pool = pool.clone();
    let mut next = tokio::spawn(async move { next_pool.run_when_available(0, |_| Ok(7)).await });
    assert!(timeout(Duration::from_millis(20), &mut next).await.is_err());
    release.send(()).unwrap();
    first.await.unwrap().unwrap();
    assert_eq!(
        timeout(Duration::from_secs(2), next)
            .await
            .unwrap()
            .unwrap()
            .unwrap(),
        7
    );
    assert_eq!(pool.inspect().retained_jobs, 0);
}
