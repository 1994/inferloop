//! Observability HTTP adapter.
use super::{HttpResult, RuntimeHandle};
use axum::extract::rejection::QueryRejection;
use axum::{Json, extract::Query, extract::State, response::IntoResponse, response::Response};
use infer_core::{Error, RequestId};
use infer_observe::ObservationQuery;

/// Default semantic-event count returned when an event query omits `limit`.
const DEFAULT_EVENT_LIMIT: usize = 256;
/// Default device timeline count returned when a query omits `limit`.
const DEFAULT_DEVICE_TIMELINE_LIMIT: usize = 8;

#[derive(serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct DeviceTimelineParams {
    after: u64,
    limit: usize,
}
impl Default for DeviceTimelineParams {
    fn default() -> Self {
        Self {
            after: 0,
            limit: DEFAULT_DEVICE_TIMELINE_LIMIT,
        }
    }
}

/// Raw sampled device timelines; every device-time view is built from these records.
pub(super) async fn device_timelines(
    State(handle): State<RuntimeHandle>,
    query: Result<Query<DeviceTimelineParams>, QueryRejection>,
) -> HttpResult<Json<serde_json::Value>> {
    let Query(query) = query.map_err(|error| Error::invalid(error.body_text()))?;
    Ok(Json(
        handle
            .observe(ObservationQuery::DeviceTimelines {
                after: query.after,
                limit: query.limit,
            })
            .await?,
    ))
}

#[derive(serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct EventParams {
    after: u64,
    limit: usize,
    request: Option<RequestId>,
}
impl Default for EventParams {
    fn default() -> Self {
        Self {
            after: 0,
            limit: DEFAULT_EVENT_LIMIT,
            request: None,
        }
    }
}
pub(super) async fn metrics(State(handle): State<RuntimeHandle>) -> HttpResult<Response> {
    let output = handle.observe(ObservationQuery::Metrics).await?;
    Ok((
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        output.as_str().unwrap_or_default().to_string(),
    )
        .into_response())
}
pub(super) async fn observability(
    State(handle): State<RuntimeHandle>,
) -> HttpResult<Json<serde_json::Value>> {
    Ok(Json(handle.observe(ObservationQuery::Summary).await?))
}
pub(super) async fn diagnostics(
    State(handle): State<RuntimeHandle>,
) -> HttpResult<Json<serde_json::Value>> {
    Ok(Json(handle.observe(ObservationQuery::Diagnostics).await?))
}
pub(super) async fn timeline(
    State(handle): State<RuntimeHandle>,
) -> HttpResult<Json<serde_json::Value>> {
    Ok(Json(handle.observe(ObservationQuery::Timeline).await?))
}
pub(super) async fn otlp(
    State(handle): State<RuntimeHandle>,
) -> HttpResult<Json<serde_json::Value>> {
    Ok(Json(handle.observe(ObservationQuery::Otlp).await?))
}
pub(super) async fn events(
    State(handle): State<RuntimeHandle>,
    query: Result<Query<EventParams>, QueryRejection>,
) -> HttpResult<Json<serde_json::Value>> {
    let Query(query) = query.map_err(|error| Error::invalid(error.body_text()))?;
    Ok(Json(
        handle
            .observe(ObservationQuery::Events {
                after: query.after,
                limit: query.limit,
                request: query.request,
            })
            .await?,
    ))
}
