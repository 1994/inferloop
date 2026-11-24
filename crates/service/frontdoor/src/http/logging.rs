//! HTTP request logging.
//!
//! One line per request, produced by a router layer so every route is covered. Success is
//! `info`, a client rejection is `warn` and a server failure is `error`, so an operator sees a
//! rejected request without turning on debug logging. A response whose length is known is logged
//! when its headers are ready; a streamed one (SSE) is logged when its body ends, and says whether
//! the client left first. The line carries the request identity the exchange reserved, so it joins
//! the engine's `infer::request` line for the same request.
use crate::ingress::correlate;
use axum::body::{Body, BodyDataStream, HttpBody};
use axum::extract::Request;
use axum::http::{Method, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use futures_util::Stream;
use infer_core::RequestId;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

/// Log method, path, status, request identity and latency for one request.
pub(super) async fn log_request(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let started = Instant::now();
    let (response, request) = correlate(next.run(request)).await;
    let access = Access {
        method,
        path,
        status: response.status(),
        request,
        started,
        headers_ms: elapsed_ms(started),
    };
    if response.body().size_hint().exact().is_some() {
        access.log(Ending::Whole);
        return response;
    }
    let (parts, body) = response.into_parts();
    let body = Streamed {
        inner: body.into_data_stream(),
        access: Some(access),
    };
    Response::from_parts(parts, Body::from_stream(body))
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// How the exchange's body ended.
#[derive(Clone, Copy)]
enum Ending {
    /// The length was known; the body was handed over with the headers.
    Whole,
    /// A streamed body reached its end.
    Streamed,
    /// The client stopped reading a streamed body before its end.
    ClientGone,
}

/// What one access line reports.
struct Access {
    method: Method,
    path: String,
    status: StatusCode,
    request: Option<RequestId>,
    started: Instant,
    headers_ms: u64,
}

impl Access {
    fn log(&self, ending: Ending) {
        let line = Line {
            method: &self.method,
            path: self.path.as_str(),
            status: self.status.as_u16(),
            request: self.request.map(RequestId::get),
            stream: !matches!(ending, Ending::Whole),
            client_gone: matches!(ending, Ending::ClientGone),
            headers_ms: self.headers_ms,
            elapsed_ms: elapsed_ms(self.started),
        };
        if self.status.is_server_error() {
            line.error();
        } else if self.status.is_client_error() {
            line.warn("request rejected");
        } else if line.client_gone {
            line.warn("request stream closed by client");
        } else {
            line.info();
        }
    }
}

/// The fields of one access line; their names are part of the logging contract.
struct Line<'a> {
    method: &'a Method,
    path: &'a str,
    status: u16,
    request: Option<u64>,
    stream: bool,
    client_gone: bool,
    headers_ms: u64,
    elapsed_ms: u64,
}

impl Line<'_> {
    fn error(&self) {
        tracing::error!(
            target: "infer::http",
            method = %self.method, path = self.path, status = self.status, request = self.request,
            stream = self.stream, client_gone = self.client_gone, headers_ms = self.headers_ms,
            elapsed_ms = self.elapsed_ms,
            "request failed"
        );
    }
    fn warn(&self, message: &'static str) {
        tracing::warn!(
            target: "infer::http",
            method = %self.method, path = self.path, status = self.status, request = self.request,
            stream = self.stream, client_gone = self.client_gone, headers_ms = self.headers_ms,
            elapsed_ms = self.elapsed_ms,
            "{message}"
        );
    }
    fn info(&self) {
        tracing::info!(
            target: "infer::http",
            method = %self.method, path = self.path, status = self.status, request = self.request,
            stream = self.stream, client_gone = self.client_gone, headers_ms = self.headers_ms,
            elapsed_ms = self.elapsed_ms,
            "request"
        );
    }
}

/// A streamed body that logs its exchange when it ends or is dropped unfinished.
struct Streamed {
    inner: BodyDataStream,
    /// Present until the line is written, so it is written exactly once.
    access: Option<Access>,
}

impl Stream for Streamed {
    type Item = <BodyDataStream as Stream>::Item;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = Pin::new(&mut self.inner).poll_next(context);
        if matches!(polled, Poll::Ready(None))
            && let Some(access) = self.access.take()
        {
            access.log(Ending::Streamed);
        }
        polled
    }
}

impl Drop for Streamed {
    fn drop(&mut self) {
        if let Some(access) = self.access.take() {
            access.log(Ending::ClientGone);
        }
    }
}

#[cfg(test)]
mod tests;
