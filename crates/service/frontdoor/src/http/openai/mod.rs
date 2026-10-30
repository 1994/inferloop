//! A deliberately bounded subset of the `OpenAI` text generation protocol.
mod decisions;
mod error;
mod execution;
mod request;
mod stream;
#[cfg(test)]
#[path = "../../../tests/unit/http_openai.rs"]
mod tests;

use super::text::TextState;
use axum::extract::rejection::JsonRejection;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
    routing::get,
    routing::post,
};
use error::ApiError;
use request::{GenerationRequest, ToolPolicy};
use serde_json::{Value, json};

type ApiResult<T> = Result<T, ApiError>;
type Payload = Result<Json<GenerationRequest>, JsonRejection>;

pub(super) fn routes() -> Router<TextState> {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/models/{model}", get(retrieve_model))
        .route(
            "/v1/chat/completions",
            post(chat).layer(DefaultBodyLimit::disable()),
        )
        .route("/v1/completions", post(completion))
        .route("/v1/decisions", post(decision))
}

async fn models(State(state): State<TextState>) -> ApiResult<Json<Value>> {
    // The served name is configuration, not the internal model identity.
    let model = state.handle.inspect().await?.model_name;
    Ok(Json(json!({"object":"list", "data":[{
        "id":model, "object":"model", "created":0, "owned_by":"inference.rs"
    }]})))
}

async fn retrieve_model(
    State(state): State<TextState>,
    Path(model): Path<String>,
) -> ApiResult<Json<Value>> {
    let served = state.handle.inspect().await?.model_name;
    if model != served {
        return Err(infer_core::Error::new(
            infer_core::ErrorCode::NotFound,
            "model is not served; see /v1/models",
        )
        .into());
    }
    Ok(Json(
        json!({"id":served, "object":"model", "created":0, "owned_by":"inference.rs"}),
    ))
}

async fn chat(
    State(state): State<TextState>,
    headers: HeaderMap,
    request: Payload,
) -> ApiResult<Response> {
    let Json(request) = request.map_err(ApiError::from)?;
    reply(state, headers, request, true).await
}

async fn completion(
    State(state): State<TextState>,
    headers: HeaderMap,
    request: Payload,
) -> ApiResult<Response> {
    let Json(request) = request.map_err(ApiError::from)?;
    reply(state, headers, request, false).await
}

async fn decision(
    State(state): State<TextState>,
    headers: HeaderMap,
    request: Result<Json<decisions::DecisionRequest>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    let Json(request) = request.map_err(ApiError::from)?;
    decisions::decide(state, headers, request).await.map(Json)
}

/// Route one validated request to the complete or the streamed reply.
async fn reply(
    state: TextState,
    headers: HeaderMap,
    request: GenerationRequest,
    chat: bool,
) -> ApiResult<Response> {
    if request.stream == Some(true) {
        stream::stream(state, headers, request, chat).await
    } else {
        execution::generate(state, headers, request, chat)
            .await
            .map(|value| Json(value).into_response())
    }
}
