//! Server-sent events for the `OpenAI` chat and completion protocols.
//!
//! Both the streamed and the complete reply come from [`prepare`], so they accept the same
//! requests and resolve the same limits. Text is decoded and parsed incrementally on the CPU
//! pool: each committed token is detokenized once and fed to the parser once, and no published
//! delta is ever rewritten.
use super::error::ApiError;
use super::execution::{completion_reason, finish_reason, prepare};
use super::{ApiResult, GenerationRequest, TextState};
use crate::cpu::CpuPool;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::stream::unfold;
use infer_core::{Error, ErrorCode, RequestId};
use infer_models::{OutputStreamParser, StreamEvent, TextStreamDecoder};
use infer_runtime::EngineOutput;
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    convert::Infallible,
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::sync::mpsc;

/// Terminal marker of an `OpenAI` event stream.
const DONE: &str = "[DONE]";

/// Lifecycle of one event stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The role has not been published yet.
    Start,
    Running,
    /// The terminator has been queued.
    Done,
}

/// One streamed generation in progress.
struct Stream {
    request: RequestId,
    receiver: mpsc::Receiver<infer_core::Result<EngineOutput>>,
    delivery: CpuPool,
    /// The decoder and parser leave this owner for the duration of a CPU job and come back.
    decoder: Option<TextStreamDecoder>,
    parser: Option<OutputStreamParser>,
    /// Set by prompt encoding before the first engine event; consumed once.
    prompt_reasoning: Option<Arc<AtomicBool>>,
    /// Serialized event payloads waiting to be written.
    queue: VecDeque<String>,
    id: String,
    object: &'static str,
    model: String,
    created: u64,
    chat: bool,
    include_usage: bool,
    /// Whether generated text is split into content, reasoning and calls. False publishes the
    /// decoded text unchanged, which is what the complete reply does when neither tools nor
    /// thinking mode are in play.
    parse: bool,
    prompt_tokens: Arc<AtomicUsize>,
    completion_tokens: usize,
    call_index: usize,
    phase: Phase,
}

impl Stream {
    /// Serialize one chunk of this response.
    fn chunk(&self, delta: &Value, finish_reason: Option<&str>) -> String {
        let mut choice = json!({"index":0, "finish_reason":finish_reason, "logprobs":null});
        if self.chat {
            choice["delta"] = delta.clone();
        } else {
            choice["text"] = json!(delta["text"].as_str().unwrap_or_default());
        }
        let mut chunk = json!({
            "id":self.id, "object":self.object, "created":self.created, "model":self.model,
            "choices":[choice]
        });
        if self.include_usage {
            chunk["usage"] = Value::Null;
        }
        chunk.to_string()
    }
    fn queue_chunk(&mut self, delta: &Value, finish_reason: Option<&str>) {
        let chunk = self.chunk(delta, finish_reason);
        self.queue.push_back(chunk);
    }
    /// Publish one parser event as a delta.
    fn publish(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Text(text) => {
                let delta = if self.chat {
                    json!({"content": text})
                } else {
                    json!({"text": text})
                };
                self.queue_chunk(&delta, None);
            }
            StreamEvent::Reasoning(text) => {
                self.queue_chunk(&json!({"reasoning_content": text}), None);
            }
            StreamEvent::CallStart { index, name } => {
                self.queue_chunk(
                    &json!({"tool_calls":[{
                        "index":index, "id":format!("call_{}_{}",self.request.get(),index),
                        "type":"function", "function":{"name":name,"arguments":""}
                    }]}),
                    None,
                );
            }
            StreamEvent::CallArguments { index, arguments } => {
                self.queue_chunk(
                    &json!({"tool_calls":[{
                        "index":index, "function":{"arguments":arguments}
                    }]}),
                    None,
                );
            }
            StreamEvent::CallEnd { index } => {
                self.call_index = index + 1;
            }
            StreamEvent::Call(call) => {
                // Completed-only fallback, for string-wrapped JSON argument objects.
                let delta = json!({"tool_calls":[{
                    "index": self.call_index,
                    "id": format!("call_{}_{}", self.request.get(), self.call_index),
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }]});
                self.call_index += 1;
                self.queue_chunk(&delta, None);
            }
        }
    }
    /// Queue an error event and end the stream.
    fn fail(&mut self, error: &Error) {
        self.queue.push_back(
            ApiError::from(Error::new(error.code, error.message.clone()))
                .body()
                .to_string(),
        );
        self.queue.push_back(DONE.to_string());
        self.phase = Phase::Done;
    }
    /// Terminal chunks: the finish reason, optional usage, and the stream terminator.
    fn finish(&mut self, reason: &str) {
        self.queue_chunk(&json!({}), Some(reason));
        if self.include_usage {
            let prompt = self.prompt_tokens.load(Ordering::Relaxed);
            self.queue.push_back(
                json!({
                    "id":self.id, "object":self.object, "created":self.created, "model":self.model,
                    "choices":[],
                    "usage":{
                        "prompt_tokens":prompt, "completion_tokens":self.completion_tokens,
                        "total_tokens":prompt + self.completion_tokens
                    }
                })
                .to_string(),
            );
        }
        self.queue.push_back(DONE.to_string());
        self.phase = Phase::Done;
    }
    /// Publish everything produced by one engine event.
    ///
    /// # Errors
    /// Returns a decoding or parsing error for a response that cannot be published incrementally.
    async fn advance(&mut self, event: EngineOutput) -> infer_core::Result<()> {
        if let Some(open) = self.prompt_reasoning.take() {
            let parser = self
                .parser
                .take()
                .ok_or_else(|| Error::invariant("stream parser missing"))?;
            self.parser = Some(parser.with_open_reasoning(open.load(Ordering::Acquire)));
        }
        match event {
            EngineOutput::Token { token, .. } => {
                self.completion_tokens += 1;
                let parse = self.parse;
                let delivery = self.delivery.clone();
                let decoder = self
                    .decoder
                    .take()
                    .ok_or_else(|| Error::invariant("stream decoder missing"))?;
                let parser = self
                    .parser
                    .take()
                    .ok_or_else(|| Error::invariant("stream parser missing"))?;
                let (decoder, parser, events) = delivery
                    .run(0, move |_| {
                        let mut decoder = decoder;
                        let mut parser = parser;
                        let text = decoder.push(token)?;
                        let events = if text.is_empty() {
                            Vec::new()
                        } else if parse {
                            parser.feed(&text)?
                        } else {
                            vec![StreamEvent::Text(text.clone())]
                        };
                        Ok((decoder, parser, events))
                    })
                    .await?;
                self.decoder = Some(decoder);
                self.parser = Some(parser);

                for event in events {
                    self.publish(event);
                }
            }
            EngineOutput::Finished(result) => {
                let reason = finish_reason(&result.reason)?;
                let parse = self.parse;
                let delivery = self.delivery.clone();
                let decoder = self
                    .decoder
                    .take()
                    .ok_or_else(|| Error::invariant("stream decoder missing"))?;
                let parser = self
                    .parser
                    .take()
                    .ok_or_else(|| Error::invariant("stream parser missing"))?;
                let (decoder, parser, events) = delivery
                    .run(0, move |_| {
                        let mut decoder = decoder;
                        let mut parser = parser;
                        // Flush the held tail, then finish the parser so a withheld block is
                        // reported as truncated rather than published.
                        let tail = decoder.finish()?;
                        let mut events = if tail.is_empty() {
                            Vec::new()
                        } else if parse {
                            parser.feed(&tail)?
                        } else {
                            vec![StreamEvent::Text(tail.clone())]
                        };
                        if parse {
                            events.extend(parser.finish()?);
                        }
                        Ok((decoder, parser, events))
                    })
                    .await?;
                self.decoder = Some(decoder);
                self.parser = Some(parser);

                for event in events {
                    self.publish(event);
                }
                let reason = completion_reason(
                    reason,
                    self.parser
                        .as_ref()
                        .is_some_and(OutputStreamParser::truncated),
                    self.call_index > 0,
                )?;
                self.finish(reason);
            }
        }
        Ok(())
    }
}

/// Stream one chat or completion request as server-sent events.
///
/// # Errors
/// Returns the same validation errors as the complete reply, plus `unsupported` when the package
/// cannot publish a stable incremental decode.
pub(super) async fn stream(
    state: TextState,
    headers: HeaderMap,
    request: GenerationRequest,
    chat: bool,
) -> ApiResult<Response> {
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let prepared = prepare(&state, headers, request, chat).await?;
    let prefix = if chat { "chatcmpl" } else { "cmpl" };
    let object = if chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    let events = unfold(
        Stream {
            request: prepared.request,
            receiver: prepared.receiver,
            delivery: state.delivery.clone(),
            decoder: Some(state.assets.stream_decoder()),
            parser: Some(
                OutputStreamParser::new(prepared.dialect, prepared.policy.declared().cloned())
                    .with_tools(prepared.policy.parses())
                    .with_argument_deltas(),
            ),
            prompt_reasoning: Some(prepared.reasoning_open),
            queue: VecDeque::new(),
            id: format!("{prefix}-{}", prepared.request),
            object,
            model: prepared.model_name.clone(),
            created: prepared.created,
            chat,
            include_usage,
            parse: chat && (prepared.policy.parses() || prepared.enable_thinking),
            prompt_tokens: prepared.prompt_tokens,
            completion_tokens: 0,
            call_index: 0,
            phase: Phase::Start,
        },
        |mut stream| async move {
            loop {
                if let Some(data) = stream.queue.pop_front() {
                    return Some((Ok::<_, Infallible>(Event::default().data(data)), stream));
                }
                if stream.phase == Phase::Done {
                    return None;
                }
                if stream.phase == Phase::Start {
                    // The role is published before any content, as the protocol requires.
                    stream.phase = Phase::Running;
                    if stream.chat {
                        stream.queue_chunk(&json!({"role":"assistant"}), None);
                    }
                    continue;
                }
                match stream.receiver.recv().await {
                    None => {
                        let error = Error::new(
                            ErrorCode::Backend,
                            "request stream closed before completion",
                        );
                        stream.fail(&error);
                    }
                    Some(Err(error)) => stream.fail(&error),
                    Some(Ok(event)) => {
                        if let Err(error) = stream.advance(event).await {
                            stream.fail(&error);
                        }
                    }
                }
            }
        },
    );
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[cfg(test)]
#[path = "../../../tests/unit/http_openai_stream.rs"]
mod tests;
