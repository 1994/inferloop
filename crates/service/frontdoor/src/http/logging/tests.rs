use super::*;
use crate::ingress::Registry;
use axum::Router;
use axum::body::to_bytes;
use axum::http::Uri;
use axum::middleware::from_fn;
use axum::routing::get;
use futures_util::{StreamExt, stream};
use std::convert::Infallible;
use std::io::{Error as IoError, Result as IoResult, Write};
use std::sync::{Arc, Mutex};
use tower::ServiceExt;
use tracing::subscriber::{DefaultGuard, set_default};

/// Captured `infer::http` lines.
#[derive(Clone, Default)]
struct Lines(Arc<Mutex<Vec<u8>>>);

impl Write for Lines {
    fn write(&mut self, bytes: &[u8]) -> IoResult<usize> {
        self.0
            .lock()
            .map_err(|_| IoError::other("captured log poisoned"))?
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

impl Lines {
    fn install(&self) -> DefaultGuard {
        let writer = self.clone();
        set_default(
            tracing_subscriber::fmt()
                .with_writer(move || writer.clone())
                .with_ansi(false)
                .finish(),
        )
    }
    fn http(&self) -> Vec<String> {
        self.0
            .lock()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_default()
            .lines()
            .filter(|line| line.contains("infer::http"))
            .map(str::to_owned)
            .collect()
    }
}

fn app() -> Router {
    let registry = Registry::new(1);
    Router::new()
        .route("/whole", get(|| async { "ok" }))
        .route(
            "/stream",
            get(move || {
                let registry = registry.clone();
                async move {
                    // The reservation is what names the request on the access line; the
                    // response keeps it alive, as a streaming handler does.
                    let Ok(pending) = RequestId::new(7).and_then(|id| registry.reserve(id, 0))
                    else {
                        return Body::empty();
                    };
                    let chunks = stream::iter(["a", "b", "c"]).map(move |chunk| {
                        let _held = &pending;
                        Ok::<_, Infallible>(chunk)
                    });
                    Body::from_stream(chunks)
                }
            }),
        )
        .layer(from_fn(log_request))
}

fn get_request(path: &'static str) -> Request {
    let mut request = Request::new(Body::empty());
    *request.uri_mut() = Uri::from_static(path);
    request
}

/// A response of known length is logged with its headers and is not a stream.
#[tokio::test]
async fn a_whole_response_is_logged_once_when_its_headers_are_ready() {
    let lines = Lines::default();
    let _guard = lines.install();
    let response = app().oneshot(get_request("/whole")).await.unwrap();
    assert_eq!(lines.http().len(), 1);
    drop(response);
    let logged = lines.http();
    assert_eq!(logged.len(), 1, "{logged:?}");
    assert!(logged[0].contains("stream=false"), "{}", logged[0]);
    assert!(!logged[0].contains("request="), "{}", logged[0]);
}

/// A streamed response is logged when its body ends, naming the request it carried.
#[tokio::test]
async fn a_finished_stream_is_logged_at_its_end_with_its_request() {
    let lines = Lines::default();
    let _guard = lines.install();
    let response = app().oneshot(get_request("/stream")).await.unwrap();
    assert!(
        lines.http().is_empty(),
        "a stream is not over at its headers"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(&body[..], b"abc");
    let logged = lines.http();
    assert_eq!(logged.len(), 1, "{logged:?}");
    for field in [
        "request=7",
        "stream=true",
        "client_gone=false",
        "status=200",
    ] {
        assert!(logged[0].contains(field), "{field} in {}", logged[0]);
    }
}

/// A client that stops reading mid-stream turns the access line into a warning, instead of the
/// `200` that used to be written at header time.
#[tokio::test]
async fn a_stream_dropped_by_its_client_says_so() {
    let lines = Lines::default();
    let _guard = lines.install();
    let response = app().oneshot(get_request("/stream")).await.unwrap();
    drop(response);
    let logged = lines.http();
    assert_eq!(logged.len(), 1, "{logged:?}");
    for field in ["WARN", "client_gone=true", "request=7"] {
        assert!(logged[0].contains(field), "{field} in {}", logged[0]);
    }
}
