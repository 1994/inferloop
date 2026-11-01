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
#[path = "../../../../tests/unit/http_responses.rs"]
mod tests;

use super::execution;
use super::request::GenerationRequest;
use super::{ApiResult, TextState, parse_json_body};
use axum::{
    Json,
    extract::{Path, Request, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use infer_core::{Error, ErrorCode};
use infer_spi::{ResponseId, ResponseStatus, ResponseStore};
use request::{PriorState, ResponsesRequest, StoreOptions};
use state::{StoreLease, StoreSink};
use std::sync::Arc;

pub(super) async fn create(
    State(state): State<TextState>,
    headers: HeaderMap,
    request: Request,
) -> ApiResult<Response> {
    let request = parse_json_body::<ResponsesRequest>(&state, request).await?;
    let store = state.responses.store().cloned();
    let options = StoreOptions {
        installed: store.is_some(),
        capabilities: state.responses.capabilities().unwrap_or_default(),
    };
    let prior = load_prior(
        store.as_ref(),
        request.previous_response_id.as_deref(),
        &request.model,
    )
    .await?;
    let prepared = request.prepare(options, prior)?;
    let request::Prepared {
        generation,
        mut spec,
        request_input,
    } = prepared;
    let is_stream = generation.stream == Some(true);

    if spec.background {
        return start_background(state, headers, generation, spec, request_input, store).await;
    }

    let generation = execution::prepare(&state, &headers, generation, true).await?;
    if spec.store {
        spec.response_id = Some(format!(
            "resp_{}_{}",
            state.response_instance,
            generation.request.get()
        ));
    }
    let mut lease = match store {
        Some(store) if spec.store => Some(StoreLease::new(StoreSink::new(
            store,
            &generation,
            &spec,
            request_input,
        )?)),
        _ => None,
    };
    if let Some(lease) = &mut lease
        && let Err(error) = lease.start().await
    {
        let _ = state.handle.cancel(generation.request).await;
        return Err(error.into());
    }
    if is_stream {
        Ok(stream::stream(&state, generation, spec, lease))
    } else {
        let value = match execute::generate_prepared(&state, generation, &spec).await {
            Ok(value) => value,
            Err(error) => {
                if let Some(lease) = &mut lease
                    && let Err(store_error) = lease.fail(&error).await
                {
                    tracing::error!(
                        target: "infer::openai",
                        error = %store_error,
                        "failed to retain a failed response"
                    );
                }
                return Err(error.into());
            }
        };
        if let Some(lease) = &mut lease {
            lease.finish(&value).await?;
        }
        Ok(Json(value).into_response())
    }
}

async fn start_background(
    state: TextState,
    headers: HeaderMap,
    generation: GenerationRequest,
    mut spec: request::Spec,
    request_input: Vec<u8>,
    store: Option<Arc<dyn ResponseStore>>,
) -> ApiResult<Response> {
    let store = store.ok_or_else(|| Error::invariant("background response has no store"))?;
    let permit = state
        .background_permits
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            Error::new(
                ErrorCode::Capacity,
                "background response capacity is exhausted",
            )
        })?;
    let generation = execution::prepare(&state, &headers, generation, true).await?;
    if spec.store {
        spec.response_id = Some(format!(
            "resp_{}_{}",
            state.response_instance,
            generation.request.get()
        ));
    }
    let mut lease = StoreLease::new(StoreSink::new(store, &generation, &spec, request_input)?);
    let initial = match lease.start().await {
        Ok(value) => value,
        Err(error) => {
            let _ = state.handle.cancel(generation.request).await;
            return Err(error.into());
        }
    };
    let task_state = state.clone();
    let task_spec = spec;
    tokio::spawn(async move {
        let _permit = permit;
        let mut lease = lease;
        match execute::generate_prepared(&task_state, generation, &task_spec).await {
            Ok(value) => {
                if let Err(error) = lease.finish(&value).await {
                    tracing::error!(
                        target: "infer::openai",
                        error = %error,
                        "failed to retain a completed background response"
                    );
                }
            }
            Err(error) => {
                if let Err(store_error) = lease.fail(&error).await {
                    tracing::error!(
                        target: "infer::openai",
                        error = %store_error,
                        "failed to retain a failed background response"
                    );
                }
            }
        }
    });
    Ok(Json(initial).into_response())
}

async fn load_prior(
    store: Option<&Arc<dyn ResponseStore>>,
    previous: Option<&str>,
    model: &str,
) -> ApiResult<Option<PriorState>> {
    let Some(previous) = previous else {
        return Ok(None);
    };
    let store = store.ok_or_else(|| {
        Error::unsupported("previous_response_id needs an installed response store")
    })?;
    let id = ResponseId::new(previous)?;
    let stored = store
        .get(&id)
        .await?
        .ok_or_else(|| Error::new(ErrorCode::NotFound, "response not found; see store policy"))?;
    Ok(Some(state::prior_state(&stored, model)?))
}

pub(super) async fn retrieve(
    State(state): State<TextState>,
    Path(id): Path<String>,
) -> ApiResult<Json<output::ResponseBody>> {
    let store = state.responses.store().cloned().ok_or_else(|| {
        Error::unsupported("response retrieval needs an installed response store")
    })?;
    let id = ResponseId::new(id)?;
    let stored = store
        .get(&id)
        .await?
        .ok_or_else(|| Error::new(ErrorCode::NotFound, "response not found"))?;
    Ok(Json(state::retrieved(&stored)?))
}

pub(super) async fn cancel(
    State(state): State<TextState>,
    Path(id): Path<String>,
) -> ApiResult<Json<output::ResponseBody>> {
    let store = state.responses.store().cloned().ok_or_else(|| {
        Error::unsupported("response cancellation needs an installed response store")
    })?;
    let id = ResponseId::new(id)?;
    let stored = store
        .get(&id)
        .await?
        .ok_or_else(|| Error::new(ErrorCode::NotFound, "response not found"))?;
    if stored.status.is_terminal() {
        return Err(Error::invalid("a terminal response cannot be cancelled").into());
    }
    let updated = store
        .transition(
            &id,
            &[ResponseStatus::Queued, ResponseStatus::InProgress],
            ResponseStatus::Cancelled,
        )
        .await?;
    if let Err(error) = state.handle.cancel(updated.request_id).await {
        tracing::debug!(
            target: "infer::openai",
            error = %error,
            "response was marked cancelled after generation had already stopped"
        );
    }
    Ok(Json(state::retrieved(&updated)?))
}
