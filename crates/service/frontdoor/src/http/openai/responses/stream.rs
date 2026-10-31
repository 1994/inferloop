//! Typed server-sent events for the `OpenAI` Responses protocol.
//!
//! Both the streamed and the complete reply come from [`prepare`], so they accept the same
//! requests and resolve the same limits. Each committed token is detokenized and parsed once on
//! the delivery pool, and a published delta is never rewritten.
use super::super::execution::{finish_reason, prepare};
use super::super::{ApiResult, TextState};
use super::output::{self, Snapshot, Usage};
use super::request::{Prepared, Spec};
use crate::cpu::CpuPool;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::stream::unfold;
use infer_core::{Error, ErrorCode, RequestId};
use infer_models::{OutputStreamParser, StreamEvent, TextStreamDecoder};
use infer_runtime::EngineOutput;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// Lifecycle of one event stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Running,
    /// The terminal response event has been queued.
    Done,
}

/// An assistant message whose content is still streaming.
struct OpenMessage {
    index: usize,
    item_id: String,
    text: String,
}

/// A reasoning item whose text is still streaming.
struct OpenReasoning {
    index: usize,
    item_id: String,
    text: String,
}

/// A function call whose arguments are still streaming.
struct OpenCall {
    index: usize,
    call_index: usize,
    item_id: String,
    name: String,
    arguments: String,
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
    /// SSE event name and serialized payload waiting to be written.
    queue: VecDeque<(String, String)>,
    sequence: u64,
    model: String,
    created: u64,
    spec: Spec,
    temperature: f32,
    top_p: f32,
    prompt_tokens: Arc<AtomicUsize>,
    completion_tokens: usize,
    /// Completed output items, in output order.
    output: Vec<Value>,
    message: Option<OpenMessage>,
    reasoning: Option<OpenReasoning>,
    call: Option<OpenCall>,
    /// Output index the next item receives.
    next_index: usize,
    /// Calls published so far, which is also their `call_id` ordinal.
    calls: usize,
    phase: Phase,
}

impl Stream {
    /// Serialize one event with the next sequence number.
    fn emit(&mut self, mut event: Value) {
        event["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        let name = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message")
            .to_string();
        self.queue.push_back((name, event.to_string()));
    }

    /// The response object at this point in the lifecycle.
    fn response(
        &self,
        status: &str,
        completed_at: Option<u64>,
        incomplete: Option<Value>,
        error: Option<Value>,
        usage: Option<&Usage>,
    ) -> Value {
        Snapshot {
            request: self.request,
            created: self.created,
            completed_at,
            model: &self.model,
            status,
            incomplete,
            error,
            output: &self.output,
            spec: &self.spec,
            temperature: self.temperature,
            top_p: self.top_p,
            usage,
        }
        .value()
    }

    /// Publish `response.created` and `response.in_progress` before the first token.
    fn start(&mut self) {
        let response = self.response("in_progress", None, None, None, None);
        self.emit(json!({"type":"response.created","response":response.clone()}));
        self.emit(json!({"type":"response.in_progress","response":response}));
    }

    /// Open a reasoning item and publish its arrival.
    fn open_reasoning(&mut self) {
        let index = self.next_index;
        self.next_index += 1;
        let item_id = format!("rs_{}_{}", self.request.get(), index);
        let item = output::reasoning_item(self.request, index, "", "in_progress");
        self.emit(json!({
            "type":"response.output_item.added","output_index":index,"item":item
        }));
        self.reasoning = Some(OpenReasoning {
            index,
            item_id,
            text: String::new(),
        });
    }

    /// Open an assistant message and its first content part.
    fn open_message(&mut self) {
        let index = self.next_index;
        self.next_index += 1;
        let item_id = format!("msg_{}_{}", self.request.get(), index);
        let item = output::message_item(self.request, index, "", "in_progress");
        self.emit(json!({
            "type":"response.output_item.added","output_index":index,"item":item
        }));
        self.emit(json!({
            "type":"response.content_part.added","item_id":item_id,"output_index":index,
            "content_index":0,"part":output::output_text_part("")
        }));
        self.message = Some(OpenMessage {
            index,
            item_id,
            text: String::new(),
        });
    }

    /// Open a function call item with empty arguments.
    fn open_call(&mut self, name: &str) {
        let index = self.next_index;
        self.next_index += 1;
        let call_index = self.calls;
        self.calls += 1;
        let item_id = format!("fc_{}_{}", self.request.get(), call_index);
        let item = output::function_call_item(self.request, call_index, name, "", "in_progress");
        self.emit(json!({
            "type":"response.output_item.added","output_index":index,"item":item
        }));
        self.call = Some(OpenCall {
            index,
            call_index,
            item_id,
            name: name.to_string(),
            arguments: String::new(),
        });
    }

    /// Finish and publish the open reasoning item.
    fn close_reasoning(&mut self, status: &str) {
        let Some(open) = self.reasoning.take() else {
            return;
        };
        self.emit(json!({
            "type":"response.reasoning_text.done","item_id":open.item_id,
            "output_index":open.index,"content_index":0,"text":open.text
        }));
        let item = output::reasoning_item(self.request, open.index, &open.text, status);
        self.emit(json!({
            "type":"response.output_item.done","output_index":open.index,"item":item.clone()
        }));
        self.output.push(item);
    }

    /// Finish and publish the open assistant message.
    fn close_message(&mut self, status: &str) {
        let Some(open) = self.message.take() else {
            return;
        };
        let part = output::output_text_part(&open.text);
        self.emit(json!({
            "type":"response.output_text.done","item_id":open.item_id,
            "output_index":open.index,"content_index":0,"text":open.text,"logprobs":[]
        }));
        self.emit(json!({
            "type":"response.content_part.done","item_id":open.item_id,
            "output_index":open.index,"content_index":0,"part":part
        }));
        let item = output::message_item(self.request, open.index, &open.text, status);
        self.emit(json!({
            "type":"response.output_item.done","output_index":open.index,"item":item.clone()
        }));
        self.output.push(item);
    }

    /// Finish and publish the open function call.
    fn close_call(&mut self, status: &str) {
        let Some(open) = self.call.take() else {
            return;
        };
        self.emit(json!({
            "type":"response.function_call_arguments.done","item_id":open.item_id,
            "output_index":open.index,"arguments":open.arguments
        }));
        let item = output::function_call_item(
            self.request,
            open.call_index,
            &open.name,
            &open.arguments,
            status,
        );
        self.emit(json!({
            "type":"response.output_item.done","output_index":open.index,"item":item.clone()
        }));
        self.output.push(item);
    }

    /// Finish every open item; a normal finish completes them and an interruption does not.
    fn close_open(&mut self, status: &str) {
        self.close_call(status);
        self.close_message(status);
        self.close_reasoning(status);
    }

    /// Publish one parser event.
    fn publish(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Reasoning(text) => self.publish_reasoning(&text),
            StreamEvent::Text(text) => self.publish_text(&text),
            StreamEvent::CallStart { name, .. } => self.publish_call_start(&name),
            StreamEvent::CallArguments { arguments, .. } => self.publish_arguments(&arguments),
            StreamEvent::CallEnd { .. } => self.close_call("completed"),
            StreamEvent::Call(call) => self.publish_complete_call(&call.name, &call.arguments),
        }
    }

    fn publish_reasoning(&mut self, text: &str) {
        if self.message.is_some() {
            self.close_message("completed");
        }
        if self.call.is_some() {
            self.close_call("completed");
        }
        if self.reasoning.is_none() {
            self.open_reasoning();
        }
        let delta = self.reasoning.as_mut().map(|open| {
            open.text.push_str(text);
            (open.item_id.clone(), open.index)
        });
        if let Some((item_id, index)) = delta {
            self.emit(json!({
                "type":"response.reasoning_text.delta","item_id":item_id,"output_index":index,
                "content_index":0,"delta":text
            }));
        }
    }

    fn publish_text(&mut self, text: &str) {
        if self.reasoning.is_some() {
            self.close_reasoning("completed");
        }
        if self.call.is_some() {
            self.close_call("completed");
        }
        if self.message.is_none() {
            self.open_message();
        }
        let delta = self.message.as_mut().map(|open| {
            open.text.push_str(text);
            (open.item_id.clone(), open.index)
        });
        if let Some((item_id, index)) = delta {
            self.emit(json!({
                "type":"response.output_text.delta","item_id":item_id,"output_index":index,
                "content_index":0,"delta":text,"logprobs":[]
            }));
        }
    }

    fn publish_call_start(&mut self, name: &str) {
        self.close_reasoning("completed");
        self.close_message("completed");
        if self.call.is_some() {
            self.close_call("completed");
        }
        self.open_call(name);
    }

    fn publish_arguments(&mut self, arguments: &str) {
        let delta = self.call.as_mut().map(|open| {
            open.arguments.push_str(arguments);
            (open.item_id.clone(), open.index)
        });
        if let Some((item_id, index)) = delta {
            self.emit(json!({
                "type":"response.function_call_arguments.delta","item_id":item_id,
                "output_index":index,"delta":arguments
            }));
        }
    }

    /// Publish a complete call reported as one event, for dialects that withhold a block.
    fn publish_complete_call(&mut self, name: &str, arguments: &str) {
        self.publish_call_start(name);
        self.publish_arguments(arguments);
        self.close_call("completed");
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
            EngineOutput::Token { token, .. } => self.advance_token(token).await,
            EngineOutput::Finished(result) => self.finish(result).await,
        }
    }

    async fn advance_token(&mut self, token: u32) -> infer_core::Result<()> {
        self.completion_tokens += 1;
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
                } else {
                    parser.feed(&text)?
                };
                Ok((decoder, parser, events))
            })
            .await?;
        self.decoder = Some(decoder);
        self.parser = Some(parser);
        for event in events {
            self.publish(event);
        }
        Ok(())
    }

    async fn finish(&mut self, result: infer_runtime::CompletedRequest) -> infer_core::Result<()> {
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
                let tail = decoder.finish()?;
                let mut events = if tail.is_empty() {
                    Vec::new()
                } else {
                    parser.feed(&tail)?
                };
                events.extend(parser.finish()?);
                Ok((decoder, parser, events))
            })
            .await?;
        self.decoder = Some(decoder);
        self.parser = Some(parser);
        for event in events {
            self.publish(event);
        }
        let reason = finish_reason(&result.reason)?;
        let truncated = self
            .parser
            .as_ref()
            .is_some_and(OutputStreamParser::truncated);
        if truncated && reason != "length" {
            return Err(Error::new(
                ErrorCode::Backend,
                "model output ended inside a structured region",
            ));
        }
        let (status, item_status) = if reason == "length" {
            ("incomplete", "incomplete")
        } else {
            ("completed", "completed")
        };
        self.close_open(item_status);
        // A response with no publishable item still carries the assistant message item the
        // complete route produces, so both routes expose the same output shape.
        if self.output.is_empty() {
            self.open_message();
            self.close_message(item_status);
        }
        let usage = Usage {
            input: self.prompt_tokens.load(Ordering::Relaxed),
            output: self.completion_tokens,
            reasoning: 0,
        };
        let incomplete = (reason == "length").then(|| json!({"reason":"max_output_tokens"}));
        let completed_at = (reason != "length").then(now).transpose()?;
        let response = self.response(status, completed_at, incomplete, None, Some(&usage));
        self.emit(json!({"type":format!("response.{status}"),"response":response}));
        self.phase = Phase::Done;
        Ok(())
    }

    /// Publish a terminal failure event and end the stream.
    fn fail(&mut self, error: &Error) {
        self.close_open("incomplete");
        let code = error_code(error.code);
        let value = json!({"code":code,"message":error.message});
        let response = self.response("failed", None, None, Some(value), None);
        self.emit(json!({
            "type":"error","code":code,"message":error.message,"param":null
        }));
        self.emit(json!({"type":"response.failed","response":response}));
        self.phase = Phase::Done;
    }
}

const fn error_code(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::InvalidInput => "invalid_request",
        ErrorCode::Unsupported => "unsupported_feature",
        ErrorCode::NotFound => "model_not_found",
        ErrorCode::Conflict => "request_conflict",
        ErrorCode::Capacity => "capacity_exceeded",
        ErrorCode::Backend => "backend_error",
        ErrorCode::Invariant => "internal_error",
    }
}

fn now() -> infer_core::Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| Error::new(ErrorCode::Backend, error.to_string()))
}

/// Stream one Responses request as typed server-sent events.
///
/// # Errors
/// Returns the same validation errors as the complete reply, plus `unsupported` when the package
/// cannot publish a stable incremental decode.
pub(super) async fn stream(
    state: TextState,
    headers: HeaderMap,
    request: Prepared,
) -> ApiResult<Response> {
    let spec = request.spec;
    let prepared = prepare(&state, &headers, request.generation, true).await?;
    let mut stream = Stream {
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
        sequence: 0,
        model: prepared.model_name.clone(),
        created: prepared.created,
        spec,
        temperature: prepared.temperature,
        top_p: prepared.top_p,
        prompt_tokens: prepared.prompt_tokens,
        completion_tokens: 0,
        output: Vec::new(),
        message: None,
        reasoning: None,
        call: None,
        next_index: 0,
        calls: 0,
        phase: Phase::Running,
    };
    stream.start();
    let events = unfold(stream, |mut stream| async move {
        loop {
            if let Some((name, data)) = stream.queue.pop_front() {
                return Some((
                    Ok::<_, Infallible>(Event::default().event(name).data(data)),
                    stream,
                ));
            }
            if stream.phase == Phase::Done {
                return None;
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
    });
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}
