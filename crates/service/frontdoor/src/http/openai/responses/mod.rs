//! `POST /v1/responses`: the text and function-tool subset of the `OpenAI` Responses API.
//!
//! The route accepts a string or an item list, converts it once into the same internal chat
//! generation the `/v1/chat/completions` route uses, and reports the result as typed output
//! items and typed streaming events. Stateful features are delegated to an installed
//! [`infer_spi::ResponseStore`]; without one they remain explicit errors.
mod events;
mod execute;
mod output;
mod request;
mod state;
mod stream;
#[cfg(test)]
mod tests;
mod upstream;

use super::execution;
use super::ids;
use super::request::GenerationRequest;
use super::{ApiResult, TextState, parse_json_body};
use axum::{
    Json,
    extract::{Path, Request, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use infer_agentic::BackgroundStart;
use request::{ResponsesRequest, StoreOptions};
use upstream::{Admitted, TextUpstream};

pub(super) async fn create(
    State(state): State<TextState>,
    headers: HeaderMap,
    request: Request,
) -> ApiResult<Response> {
    let request = parse_json_body::<ResponsesRequest>(&state, request).await?;
    let options = StoreOptions {
        installed: state.responses.has_store(),
        capabilities: state.responses.capabilities(),
    };
    let prior = state
        .responses
        .prior(request.previous_response_id.as_deref(), &request.model)
        .await?
        .map(|stored| state::prior_state(&stored))
        .transpose()?;
    let prepared = request.prepare(options, prior)?;
    let request::Prepared {
        generation,
        mut spec,
        request_input,
    } = prepared;
    let is_stream = generation.stream == Some(true);

    if spec.background {
        return start_background(state, headers, generation, spec, request_input).await;
    }

    let generation = execution::prepare(&state, &headers, generation, true).await?;
    // A fresh UUID per response: a client that resumes a conversation across a restart
    // replays distinct ids instead of colliding with the previous life's `resp_<request>`.
    spec.response_id = Some(ids::response());
    let responses = state.responses.clone();
    let upstream = TextUpstream { state };
    let lease = if responses.has_store() && spec.store {
        let retention = state::retention(&generation, &spec, request_input)?;
        let initial = state::initial(&generation, &spec);
        Some(responses.retain(&upstream, retention, &initial).await?)
    } else {
        None
    };
    if is_stream {
        Ok(stream::stream(&upstream.state, generation, spec, lease))
    } else {
        let admitted = Admitted { generation, spec };
        let value = responses.complete(&upstream, admitted, lease).await?;
        Ok(Json(value).into_response())
    }
}

async fn start_background(
    state: TextState,
    headers: HeaderMap,
    generation: GenerationRequest,
    mut spec: request::Spec,
    request_input: Vec<u8>,
) -> ApiResult<Response> {
    let responses = state.responses.clone();
    // Boxed: admission's future is large and would otherwise be stored inline in the
    // background start future. One allocation per background request.
    let admit = Box::pin(async {
        let generation = execution::prepare(&state, &headers, generation, true).await?;
        spec.response_id = Some(ids::response());
        let retention = state::retention(&generation, &spec, request_input)?;
        let initial = state::initial(&generation, &spec);
        ApiResult::Ok(BackgroundStart {
            admitted: Admitted { generation, spec },
            retention,
            initial,
        })
    });
    let upstream = TextUpstream {
        state: state.clone(),
    };
    let initial = responses.start_background(upstream, admit).await?;
    Ok(Json(initial).into_response())
}

pub(super) async fn retrieve(
    State(state): State<TextState>,
    Path(id): Path<String>,
) -> ApiResult<Json<output::ResponseBody>> {
    let stored = state.responses.retrieve(&id).await?;
    Ok(Json(state::retrieved(&stored)?))
}

pub(super) async fn cancel(
    State(state): State<TextState>,
    Path(id): Path<String>,
) -> ApiResult<Json<output::ResponseBody>> {
    let responses = state.responses.clone();
    let upstream = TextUpstream { state };
    let updated = responses.cancel(&upstream, &id).await?;
    Ok(Json(state::retrieved(&updated)?))
}
