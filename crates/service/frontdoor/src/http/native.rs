//! Native HTTP adapter.
use super::{
    HttpError, HttpResult, RuntimeHandle, diagnostics, events, metrics, observability, otlp,
    timeline,
};
use crate::constants::MAX_REQUEST_BODY_BYTES;
use axum::extract::rejection::JsonRejection;
use axum::{
    Json, Router, extract::DefaultBodyLimit, extract::Path, extract::State, http::HeaderMap,
    http::StatusCode, response::IntoResponse, response::Response, response::sse::Event,
    response::sse::KeepAlive, response::sse::Sse, routing::delete, routing::get, routing::post,
};
use futures_util::stream::unfold;
use infer_core::{Error, ErrorCode, RequestId};
use infer_ir::CanonicalRequest;
use infer_observe::trace::TraceContext;
use infer_runtime::{CompletedRequest, EngineOutput, RuntimeInspection};
use serde::Serialize;
use std::convert::Infallible;

pub fn router(handle: RuntimeHandle) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/native/v1/requests", post(infer))
        .route("/native/v1/stream", post(stream))
        .route("/native/v1/requests/{id}", delete(cancel))
        .route("/native/v1/runtime", get(inspect))
        .route("/metrics", get(metrics))
        .route("/native/v1/observability", get(observability))
        .route("/native/v1/events", get(events))
        .route("/native/v1/diagnostics", get(diagnostics))
        .route("/native/v1/timeline", get(timeline))
        .route("/native/v1/traces/otlp", get(otlp))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(handle)
}
#[derive(Serialize)]
struct SseError {
    message: String,
}

pub(super) fn trace_parent(headers: &HeaderMap) -> Option<TraceContext> {
    headers
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .and_then(|parent| TraceContext::parse(parent).ok())
}
#[derive(Serialize)]
struct HealthResponse<'a> {
    ready: bool,
    fault: Option<&'a infer_runtime::EngineFault>,
    resource_release_pending: bool,
    weight_backed_dataflow: bool,
    backend: &'a str,
    backend_kind: infer_ir::BackendKind,
}

pub(super) async fn health(State(handle): State<RuntimeHandle>) -> HttpResult<Response> {
    let inspection = handle.inspect().await?;
    let body = HealthResponse {
        ready: inspection.ready,
        fault: inspection.fault.as_ref(),
        resource_release_pending: inspection.resource_release_pending,
        weight_backed_dataflow: inspection.weight_backed_dataflow,
        backend: &inspection.backend,
        backend_kind: inspection.backend_kind,
    };
    Ok((
        if inspection.ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(body),
    )
        .into_response())
}
pub(super) async fn inspect(
    State(handle): State<RuntimeHandle>,
) -> HttpResult<Json<RuntimeInspection>> {
    Ok(Json(handle.inspect().await?))
}
pub(super) async fn cancel(
    State(handle): State<RuntimeHandle>,
    Path(id): Path<u64>,
) -> HttpResult<StatusCode> {
    handle.cancel(RequestId::new(id)?).await?;
    Ok(StatusCode::NO_CONTENT)
}
pub(super) async fn infer(
    State(handle): State<RuntimeHandle>,
    headers: HeaderMap,
    request: Result<Json<CanonicalRequest>, JsonRejection>,
) -> HttpResult<Json<CompletedRequest>> {
    let Json(request) = request.map_err(HttpError::from)?;
    let mut receiver = handle
        .submit_with_trace(request, trace_parent(&headers))
        .await?;
    while let Some(event) = receiver.recv().await {
        if let EngineOutput::Finished(result) = event? {
            return Ok(Json(result));
        }
    }
    Err(Error::new(
        ErrorCode::Backend,
        "request stream closed before completion",
    )
    .into())
}
pub(super) async fn stream(
    State(handle): State<RuntimeHandle>,
    headers: HeaderMap,
    request: Result<Json<CanonicalRequest>, JsonRejection>,
) -> HttpResult<Response> {
    let Json(request) = request.map_err(HttpError::from)?;
    let receiver = handle
        .submit_with_trace(request, trace_parent(&headers))
        .await?;
    let events = unfold((receiver, false), |(mut receiver, finished)| async move {
        if finished {
            return None;
        }
        let incoming = receiver.recv().await.unwrap_or_else(|| {
            Err(Error::new(
                ErrorCode::Backend,
                "request stream closed before completion",
            ))
        });
        let mut finished = matches!(incoming, Ok(EngineOutput::Finished(_)) | Err(_));
        Some({
            let event = incoming;
            let (name, data) = match event {
                Ok(event) => {
                    let name = if matches!(event, EngineOutput::Finished(_)) {
                        "finished"
                    } else {
                        "token"
                    };
                    (name, serde_json::to_string(&event))
                }
                Err(error) => ("error", serde_json::to_string(&error)),
            };
            let (name, data) = match data {
                Ok(data) => (name, data),
                Err(error) => {
                    finished = true;
                    (
                        "error",
                        serde_json::to_string(&SseError {
                            message: error.to_string(),
                        })
                        .unwrap_or_default(),
                    )
                }
            };
            (
                Ok::<_, Infallible>(Event::default().event(name).data(data)),
                (receiver, finished),
            )
        })
    });
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}
