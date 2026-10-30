use super::{ApiResult, GenerationRequest, TextState, ToolPolicy};
use crate::constants::TOKEN_STAGING_BYTES;
use crate::cpu::CpuContext;
use crate::media::{MediaRequest, MediaService};
use axum::http::HeaderMap;
use infer_core::{Error, ErrorCode, FinishReason, RequestId};
use infer_ir::{CanonicalRequest, RequestInput, Workload, WorkloadOutput};
use infer_observe::trace::TraceContext;
use infer_runtime::{CompletedRequest, EngineOutput};
use serde_json::{Value, json};
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use std::{
    collections::BTreeMap,
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::sync::mpsc;

/// One admitted generation, shared by the complete and the streamed reply so both are produced
/// from exactly the same validation, limits and preparation.
pub(super) struct Prepared {
    pub(super) request: RequestId,
    pub(super) receiver: mpsc::Receiver<infer_core::Result<EngineOutput>>,
    pub(super) prompt_tokens: Arc<AtomicUsize>,
    pub(super) model_name: String,
    pub(super) created: u64,
    pub(super) chat: bool,
    pub(super) policy: ToolPolicy,
    pub(super) enable_thinking: bool,
    pub(super) reasoning_open: Arc<AtomicBool>,
    pub(super) dialect: infer_models::ToolDialect,
}

/// Everything the preparation closure needs to encode one request.
struct Encoding {
    constraint: Option<infer_models::ToolConstraints>,
    vocabulary: usize,
    assets: Arc<infer_models::TextAssets>,
    messages: Option<Vec<infer_models::ChatMessage>>,
    media_requests: Vec<MediaRequest>,
    media: Option<Arc<MediaService>>,
    image_options: Option<crate::ImageProcessingOptions>,
    prompt: Option<String>,
    tools: Option<Value>,
    enable_thinking: bool,
    reasoning_open: Arc<AtomicBool>,
    count: Arc<AtomicUsize>,
    id: RequestId,
    model: infer_core::ModelId,
    max_new_tokens: usize,
    sampling: infer_ir::Sampling,
}

impl Encoding {
    /// Encode the prompt and describe the workload to run over it.
    fn encode(
        self,
        context: &CpuContext,
        downloads: Vec<Option<Vec<u8>>>,
    ) -> infer_core::Result<(CanonicalRequest, Option<infer_spi::ConstraintHandle>)> {
        context.check()?;
        let (tokens, media) = if let Some(messages) = self.messages {
            let rendered = self.assets.render_chat(
                &messages,
                &infer_models::ChatOptions {
                    enable_thinking: self.enable_thinking,
                    tools: self.tools.clone(),
                    ..Default::default()
                },
            )?;
            self.reasoning_open.store(
                self.enable_thinking
                    && infer_models::OutputStreamParser::prompt_opens_reasoning(&rendered),
                Ordering::Release,
            );
            if self.media_requests.is_empty() {
                (self.assets.encode(&rendered, false)?, Vec::new())
            } else {
                let media = self
                    .media
                    .as_ref()
                    .ok_or_else(|| Error::unsupported("this deployment is text-only"))?;
                let (tokens, inputs) = media.prepare_resolved_owned(
                    &self.assets,
                    &rendered,
                    &self.media_requests,
                    downloads,
                    self.image_options.as_ref(),
                    context,
                )?;
                (tokens, inputs)
            }
        } else {
            (
                self.assets
                    .encode(self.prompt.as_deref().unwrap_or_default(), true)?,
                Vec::new(),
            )
        };
        self.count.store(tokens.len(), Ordering::Relaxed);
        let constraint = self
            .constraint
            .as_ref()
            .map(|spec| {
                self.assets.tool_constraint(
                    spec,
                    &self.sampling,
                    self.enable_thinking,
                    self.reasoning_open.load(Ordering::Acquire),
                    self.vocabulary,
                )
            })
            .transpose()?;
        context.check()?;
        Ok((
            CanonicalRequest {
                id: self.id,
                model: self.model,
                session: None,
                input: RequestInput::Sequence {
                    tokens: tokens.into(),
                    media,
                },
                workload: Workload::Generate {
                    max_new_tokens: self.max_new_tokens,
                },
                qos: infer_ir::Qos::default(),
                sampling: self.sampling,
                prefix_boundary: None,
                extensions: BTreeMap::new(),
            },
            constraint,
        ))
    }
}

/// Parse and validate chat media before an identity is consumed.
fn chat_media(
    state: &TextState,
    request: &mut GenerationRequest,
    chat: bool,
) -> infer_core::Result<(Option<Vec<infer_models::ChatMessage>>, Vec<MediaRequest>)> {
    let (messages, media) = if chat {
        let (messages, media) = request.chat_messages()?;
        (Some(messages), media)
    } else {
        (None, Vec::new())
    };
    if !media.is_empty() && state.media.is_none() {
        return Err(Error::unsupported("this deployment is text-only"));
    }
    if let Some(first) = media.first() {
        if media
            .iter()
            .any(|request| request.modality != first.modality)
        {
            return Err(Error::unsupported(
                "mixed image and video in one prompt is not supported",
            ));
        }
        if let Some(service) = &state.media {
            service.limits().check(&first.modality, media.len())?;
        }
    }
    Ok((messages, media))
}

/// Validate, price and submit one generation.
///
/// # Errors
/// Returns a parameter, capacity or model-resolution error before a request identity is consumed.
pub(super) async fn prepare(
    state: &TextState,
    headers: &HeaderMap,
    request: GenerationRequest,
    chat: bool,
) -> ApiResult<Prepared> {
    let requested = request.validate(chat)?;
    let policy = request.tool_policy(chat)?;
    validate_capabilities(state, &request, &policy)?;
    let inspection = state.handle.inspect().await?;
    let limits = inspection.lengths;
    // An explicit budget over the service cap is a parameter error; an omitted budget uses the
    // bounded default and is only lowered by the cap. Neither silently truncates the prompt.
    if requested.explicit && requested.tokens > limits.output_cap {
        return Err(Error::invalid(format!(
            "requested output of {} tokens exceeds the service output cap {}",
            requested.tokens, limits.output_cap
        ))
        .into());
    }
    let max_new_tokens = requested.tokens.min(limits.output_cap);
    let bytes = request
        .preparation_bytes(max_new_tokens)?
        .checked_add(if policy.constraint().is_some() {
            infer_models::CONSTRAINT_REQUEST_BYTES
        } else {
            0
        })
        .ok_or_else(|| Error::invalid("constraint preparation budget overflow"))?;
    // Move inline media out of the wire history on a CPU worker before consuming an identity.
    let (mut request, messages, media_requests) =
        prepare_media_on_cpu(state, request, chat, bytes).await?;
    if request.model != inspection.model_name {
        return Err(Error::new(ErrorCode::NotFound, "model is not served; see /v1/models").into());
    }
    let model = inspection.model;
    let prompt = request.prompt.take();
    let tools = policy.template();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))?
        .as_secs();
    let prompt_tokens = Arc::new(AtomicUsize::new(0));
    let count = prompt_tokens.clone();
    let reasoning_open = Arc::new(AtomicBool::new(false));
    let assets = state.assets.clone();
    let resolved = assets
        .generation
        .resolve(&infer_models::SamplingOverrides {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            min_p: request.min_p,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            seed: request.seed,
            enable_thinking: request.enable_thinking,
            ..Default::default()
        })?;
    let enable_thinking = resolved.enable_thinking;
    let id = state.handle.allocate_request_id()?;
    // Which layer produced each effective sampling parameter is the difference between two
    // apparently identical baselines, so it is recorded per request rather than inferred.
    tracing::debug!(
        target: "infer::openai",
        request = id.get(),
        sources = ?resolved.sources,
        "resolved sampling parameters"
    );
    let encoding = Encoding {
        constraint: policy.constraint(),
        vocabulary: state.handle.vocabulary(),
        assets,
        messages,
        media_requests,
        media: state.media.clone(),
        image_options: request.mm_processor_kwargs.take(),
        prompt,
        tools,
        enable_thinking,
        reasoning_open: reasoning_open.clone(),
        count,
        id,
        model,
        max_new_tokens,
        sampling: resolved.sampling,
    };
    let trace = super::super::trace_parent(headers);
    let receiver = submit_encoding(state, id, bytes, trace, encoding).await?;
    Ok(Prepared {
        request: id,
        receiver,
        prompt_tokens,
        model_name: inspection.model_name.clone(),
        created,
        chat,
        policy,
        enable_thinking,
        reasoning_open,
        dialect: state.assets.tool_dialect(),
    })
}

async fn prepare_media_on_cpu(
    state: &TextState,
    mut request: GenerationRequest,
    chat: bool,
    bytes: usize,
) -> infer_core::Result<(
    GenerationRequest,
    Option<Vec<infer_models::ChatMessage>>,
    Vec<MediaRequest>,
)> {
    if !chat {
        return Ok((request, None, Vec::new()));
    }
    let state_for_work = state.clone();
    state
        .handle
        .cpu_pool()
        .run(bytes, move |context| {
            context.check()?;
            let (messages, media) = chat_media(&state_for_work, &mut request, true)?;
            Ok((request, messages, media))
        })
        .await
}

async fn submit_encoding(
    state: &TextState,
    id: RequestId,
    bytes: usize,
    trace: Option<TraceContext>,
    encoding: Encoding,
) -> infer_core::Result<mpsc::Receiver<infer_core::Result<EngineOutput>>> {
    if encoding.media_requests.is_empty() {
        return state
            .handle
            .submit_constrained_preparing(id, bytes, trace, None, move |context| {
                encoding.encode(context, Vec::new())
            })
            .await;
    }
    let pool = state.handle.cpu_pool();
    let source = async move {
        let media = encoding
            .media
            .as_ref()
            .ok_or_else(|| Error::unsupported("this deployment is text-only"))?;
        let downloads = media.resolve(&encoding.media_requests, &pool).await?;
        let retained = downloads
            .iter()
            .flatten()
            .try_fold(0usize, |total, bytes| {
                total
                    .checked_add(bytes.len())
                    .ok_or_else(|| Error::invalid("media download size overflow"))
            })?;
        Ok(((encoding, downloads), retained))
    };
    state
        .handle
        .submit_with_source(
            id,
            bytes,
            trace,
            source,
            move |context, (encoding, downloads)| encoding.encode(context, downloads),
        )
        .await
}

/// # Errors
/// Returns a validation, capacity or backend error.
pub(super) async fn generate(
    state: TextState,
    headers: HeaderMap,
    request: GenerationRequest,
    chat: bool,
) -> ApiResult<Value> {
    let mut prepared = prepare(&state, &headers, request, chat).await?;
    while let Some(event) = prepared.receiver.recv().await {
        if let EngineOutput::Finished(result) = event? {
            return response(
                &state,
                result,
                prepared.prompt_tokens.load(Ordering::Relaxed),
                &prepared.model_name,
                prepared.created,
                prepared.chat,
                &prepared.policy,
                prepared.enable_thinking,
                prepared.reasoning_open.load(Ordering::Acquire),
            )
            .await;
        }
    }
    Err(Error::new(
        ErrorCode::Backend,
        "request stream closed before completion",
    )
    .into())
}
#[expect(
    clippy::too_many_arguments,
    reason = "The response assembles one protocol object from the request context it was generated under"
)]
async fn response(
    state: &TextState,
    result: CompletedRequest,
    prompt_tokens: usize,
    model: &str,
    created: u64,
    chat: bool,
    policy: &ToolPolicy,
    enable_thinking: bool,
    reasoning_open: bool,
) -> ApiResult<Value> {
    let reason = finish_reason(&result.reason)?;
    let Some(WorkloadOutput::Tokens(tokens)) = result.output else {
        return Err(Error::invariant("generation completed without tokens").into());
    };
    let completion_tokens = tokens.len();
    tracing::debug!(target: "infer::openai", request = result.request.get(), ?tokens, "generated completion token ids");
    let assets = state.assets.clone();
    let text = state
        .delivery
        .run(
            tokens.len().saturating_mul(TOKEN_STAGING_BYTES),
            move |_| assets.decode(&tokens, true),
        )
        .await?;
    // Text is only split when the request could have produced a structured region: a declared
    // tool set for calls, or thinking mode for reasoning. Otherwise the decoded text is the
    // content, exactly as before.
    let choice = if chat {
        chat_choice(
            result.request,
            &text,
            state.assets.tool_dialect(),
            policy,
            enable_thinking,
            reason,
            reasoning_open,
        )?
    } else {
        json!({"index":0, "text":text, "finish_reason":reason, "logprobs":null})
    };
    let prefix = if chat { "chatcmpl" } else { "cmpl" };
    Ok(json!({
        "id":format!("{prefix}-{}", result.request),
        "object":if chat { "chat.completion" } else { "text_completion" },
        "created":created, "model":model, "choices":[choice],
        "usage":{"prompt_tokens":prompt_tokens, "completion_tokens":completion_tokens,
            "total_tokens":prompt_tokens + completion_tokens}
    }))
}

/// Assemble one chat choice from a decoded completion.
///
/// # Errors
/// Returns an invalid-input error when the response must be parsed but its generated region does
/// not decode in the package dialect.
fn chat_choice(
    request: RequestId,
    text: &str,
    dialect: infer_models::ToolDialect,
    policy: &ToolPolicy,
    enable_thinking: bool,
    reason: &'static str,
    reasoning_open: bool,
) -> infer_core::Result<Value> {
    let parsed = if policy.parses() || enable_thinking {
        let mut parser = infer_models::OutputStreamParser::new(dialect, policy.declared().cloned())
            .with_tools(policy.parses())
            .with_open_reasoning(reasoning_open);
        let mut events = parser.feed(text)?;
        events.extend(parser.finish()?);
        infer_models::assemble(events, parser.truncated())
    } else {
        infer_models::ParsedOutput {
            content: text.to_string(),
            ..Default::default()
        }
    };
    let reason = completion_reason(reason, parsed.truncated, parsed.has_calls())?;
    let infer_models::ParsedOutput {
        content,
        reasoning,
        calls,
        ..
    } = parsed;
    let tool_calls: Vec<Value> = if policy.parses() {
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                json!({
                    // Stable and unique per response, so a later `tool` result can name it.
                    "id": format!("call_{}_{}", request.get(), index),
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    let content = if content.is_empty() && !tool_calls.is_empty() {
        Value::Null
    } else {
        Value::String(content)
    };
    let mut message = json!({"role":"assistant", "content":content, "refusal":null});
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }
    if let Some(reasoning) = reasoning {
        message["reasoning_content"] = json!(reasoning);
    }
    Ok(json!({"index":0, "message":message, "finish_reason":reason, "logprobs":null}))
}

/// A partial structural region is successful only when the engine reported its length cap.
pub(super) fn completion_reason(
    reason: &'static str,
    truncated: bool,
    has_calls: bool,
) -> infer_core::Result<&'static str> {
    if truncated && reason != "length" {
        return Err(Error::new(
            ErrorCode::Backend,
            "model output ended inside a structured region",
        ));
    }
    Ok(if has_calls && reason != "length" {
        "tool_calls"
    } else {
        reason
    })
}

pub(super) fn finish_reason(reason: &FinishReason) -> infer_core::Result<&'static str> {
    match reason {
        FinishReason::Length => Ok("length"),
        FinishReason::Eos | FinishReason::Completed => Ok("stop"),
        FinishReason::Failed(message) => Err(Error::new(ErrorCode::Backend, message.clone())),
        FinishReason::Cancelled | FinishReason::Deadline => Err(Error::new(
            ErrorCode::Backend,
            "generation cancelled or deadline exceeded",
        )),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/http_openai_execution.rs"]
mod tests;

fn validate_capabilities(
    state: &TextState,
    request: &GenerationRequest,
    policy: &ToolPolicy,
) -> ApiResult<()> {
    if policy.parses() && !state.assets.supports_tool_calls() {
        return Err(
            Error::unsupported("the model template has no supported tool-call dialect").into(),
        );
    }
    // Streaming publishes text incrementally, so it needs a decoder that never rewrites a
    // published prefix. Packages without one are refused rather than degraded silently.
    if request.stream == Some(true) && !state.assets.supports_streaming_decode() {
        return Err(Error::unsupported(
            "streaming needs a byte-level tokenizer decoder; this package cannot publish a stable incremental decode",
        )
        .into());
    }
    Ok(())
}
