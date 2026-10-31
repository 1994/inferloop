//! `POST /v1/responses`: the text and function-tool subset of the `OpenAI` Responses API.
//!
//! The route accepts a string or an item list, converts it once into the same internal chat
//! generation the `/v1/chat/completions` route uses, and reports the result as typed output
//! items and typed streaming events.
mod execute;
mod output;
mod request;
mod stream;
#[cfg(test)]
#[path = "../../../../tests/unit/http_responses.rs"]
mod tests;

use super::{ApiResult, TextState, parse_json_body};
use axum::{
    Json,
    extract::{Request, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use request::ResponsesRequest;

pub(super) async fn create(
    State(state): State<TextState>,
    headers: HeaderMap,
    request: Request,
) -> ApiResult<Response> {
    let request = parse_json_body::<ResponsesRequest>(&state, request).await?;
    let prepared = request.prepare()?;
    let is_stream = prepared.generation.stream == Some(true);
    if is_stream {
        stream::stream(state, headers, prepared).await
    } else {
        execute::generate(state, headers, prepared)
            .await
            .map(|value| Json(value).into_response())
    }
}
