//! Text HTTP adapter.
use super::{HttpError, HttpResult, RuntimeHandle, cpu, router, trace_parent};
use crate::constants::MAX_BACKGROUND_RESPONSES;
use crate::constants::MAX_REQUEST_BODY_BYTES;
use crate::constants::TEXT_STAGING_EXPANSION;
use crate::constants::TEXT_STAGING_OVERHEAD_BYTES;
use crate::constants::TOKEN_STAGING_BYTES;
use crate::media::MediaService;
use axum::extract::rejection::JsonRejection;
use axum::{
    Json, Router, extract::DefaultBodyLimit, extract::State, http::HeaderMap, routing::post,
};
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_ir::{CanonicalRequest, WorkloadOutput};
use infer_runtime::EngineOutput;
use infer_spi::ResponsesExtensions;
use std::collections::BTreeMap;
use std::process::id;
use std::result::Result as StdResult;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Semaphore;

#[derive(Clone)]
pub(super) struct TextState {
    pub(super) handle: RuntimeHandle,
    pub(super) assets: Arc<infer_models::TextAssets>,
    pub(super) media: Option<Arc<MediaService>>,
    pub(super) delivery: cpu::CpuPool,
    pub(super) responses: ResponsesExtensions,
    pub(super) background_permits: Arc<Semaphore>,
    /// Process-instance namespace for retained response ids.
    pub(super) response_instance: Arc<str>,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTextRequest {
    pub id: RequestId,
    pub model: infer_core::ModelId,
    pub prompt: Option<String>,
    pub messages: Option<Vec<infer_models::ChatMessage>>,
    pub workload: infer_ir::Workload,
    #[serde(default)]
    pub qos: infer_ir::Qos,
    #[serde(default)]
    pub sampling: infer_models::SamplingOverrides,
    #[serde(default)]
    pub options: Option<infer_models::ChatOptions>,
}
pub fn router_with_text(handle: RuntimeHandle, assets: Arc<infer_models::TextAssets>) -> Router {
    router_with_text_and_media(handle, assets, None)
}

pub fn router_with_text_and_media(
    handle: RuntimeHandle,
    assets: Arc<infer_models::TextAssets>,
    media: Option<Arc<MediaService>>,
) -> Router {
    router_with_text_media_and_responses(handle, assets, media, ResponsesExtensions::default())
}

pub fn router_with_text_media_and_responses(
    handle: RuntimeHandle,
    assets: Arc<infer_models::TextAssets>,
    media: Option<Arc<MediaService>>,
    responses: ResponsesExtensions,
) -> Router {
    let state = TextState {
        handle: handle.clone(),
        assets,
        media,
        delivery: handle.delivery_pool(),
        responses,
        background_permits: Arc::new(Semaphore::new(MAX_BACKGROUND_RESPONSES)),
        response_instance: response_instance(),
    };
    router(handle).merge(
        Router::new()
            .route("/native/v1/text", post(text_infer))
            .merge(super::openai::routes())
            .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
            .with_state(state),
    )
}
fn response_instance() -> Arc<str> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let ordinal = NEXT.fetch_add(1, Ordering::Relaxed);
    Arc::from(format!("{:x}{nanos:x}{ordinal:x}", id()))
}

pub(super) fn text_preparation_bytes(request: &NativeTextRequest) -> Result<usize> {
    let bytes = request
        .prompt
        .as_ref()
        .map_or(0, String::len)
        .checked_add(
            request
                .messages
                .as_ref()
                .map_or(0, |messages| messages.iter().map(|m| m.content.len()).sum()),
        )
        .ok_or_else(|| Error::invalid("text preparation size overflow"))?;
    bytes
        .checked_mul(TEXT_STAGING_EXPANSION)
        .and_then(|n| n.checked_add(TEXT_STAGING_OVERHEAD_BYTES))
        .ok_or_else(|| Error::invalid("text staging size overflow"))
}

#[derive(serde::Serialize)]
pub(super) struct NativeTextResponse {
    result: infer_runtime::CompletedRequest,
    text: Option<String>,
    tokenizer_fingerprint: String,
}

pub(super) async fn text_infer(
    State(state): State<TextState>,
    headers: HeaderMap,
    request: StdResult<Json<NativeTextRequest>, JsonRejection>,
) -> HttpResult<Json<NativeTextResponse>> {
    let Json(request) = request.map_err(HttpError::from)?;
    let assets = state.assets.clone();
    let bytes = text_preparation_bytes(&request)?;
    let mut receiver = state
        .handle
        .submit_preparing_until(
            request.id,
            bytes,
            trace_parent(&headers),
            request.qos.deadline_us,
            move |context| {
                context.check()?;
                let mut overrides = request.sampling;
                if let Some(options) = &request.options {
                    overrides.enable_thinking = Some(options.enable_thinking);
                }
                let resolved = if request.messages.is_some() && request.prompt.is_none() {
                    assets.resolve_chat_generation(&overrides)?
                } else {
                    assets.generation.resolve(&overrides)?
                };
                let options = request
                    .options
                    .unwrap_or_else(|| infer_models::ChatOptions {
                        enable_thinking: resolved.enable_thinking,
                        ..Default::default()
                    });
                let tokens = match (request.prompt, request.messages) {
                    (Some(prompt), None) => assets.encode(&prompt, true)?,
                    (None, Some(messages)) => assets.encode_chat(&messages, &options)?,
                    _ => return Err(Error::invalid("provide exactly one of prompt/messages")),
                };
                Ok(CanonicalRequest {
                    id: request.id,
                    model: request.model,
                    session: None,
                    input: infer_ir::RequestInput::Sequence {
                        tokens: tokens.into(),
                        media: vec![],
                    },
                    workload: request.workload,
                    qos: request.qos,
                    sampling: resolved.sampling,
                    prefix_boundary: None,
                    extensions: BTreeMap::new(),
                })
            },
        )
        .await?;
    while let Some(event) = receiver.recv().await {
        if let EngineOutput::Finished(result) = event? {
            let text = match &result.output {
                Some(WorkloadOutput::Tokens(tokens)) => Some({
                    let assets = state.assets.clone();
                    let tokens = tokens.clone();
                    state
                        .delivery
                        .run(
                            tokens.len().saturating_mul(TOKEN_STAGING_BYTES),
                            move |_| assets.decode(&tokens, true),
                        )
                        .await?
                }),
                _ => None,
            };
            return Ok(Json(NativeTextResponse {
                result,
                text,
                tokenizer_fingerprint: state.assets.fingerprint.clone(),
            }));
        }
    }
    Err(Error::new(ErrorCode::Backend, "text stream closed before completion").into())
}
