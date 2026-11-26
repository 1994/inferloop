use super::{NoParams, add, object};
use crate::constants::QUEUE_WAIT_US;
use crate::{AgentBackend, AgentContext, registry::CommandEffect, registry::CommandRegistry};
use infer_core::{ErrorCode, RequestId, Result};
use infer_ir::CanonicalRequest;
use infer_runtime::ReplayAction;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::VecDeque;

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RequestParams {
    pub request_id: RequestId,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct TickParams {
    now_us: u64,
}

pub(super) fn request_schema() -> Value {
    object(
        json!({"request_id":{"type":"integer","minimum":1}}),
        &["request_id"],
    )
}

pub(super) fn register<B: AgentBackend>(
    registry: &mut CommandRegistry<AgentContext<B>>,
) -> Result<()> {
    add(
        registry,
        "runtime.inspect",
        &[],
        "Inspect runtime resources, progress and backend identity",
        CommandEffect::ReadOnly,
        object(json!({}), &[]),
        inspect,
    )?;
    add(
        registry,
        "runtime.query",
        &[],
        "Query compiled program and retained request records",
        CommandEffect::ReadOnly,
        object(json!({}), &[]),
        query,
    )?;
    add(
        registry,
        "runtime.submit",
        &[],
        "Submit a typed CanonicalRequest to the live engine",
        CommandEffect::Control,
        json!({"type":"object","description":"CanonicalRequest; validated by the runtime"}),
        submit,
    )?;
    add(
        registry,
        "runtime.tick",
        &[],
        "Advance live runtime control at a monotonic timestamp",
        CommandEffect::Control,
        object(
            json!({"now_us":{"type":"integer","minimum":0}}),
            &["now_us"],
        ),
        tick,
    )?;
    add(
        registry,
        "runtime.cancel",
        &[],
        "Cancel a live request and release its state",
        CommandEffect::Control,
        request_schema(),
        cancel,
    )?;
    add(
        registry,
        "runtime.explain",
        &["scheduler.explain"],
        "Explain admission, selection and wait evidence for a request",
        CommandEffect::ReadOnly,
        request_schema(),
        explain,
    )?;
    add(
        registry,
        "runtime.take_completed",
        &[],
        "Consume a terminal completion record",
        CommandEffect::Control,
        request_schema(),
        completed,
    )?;
    add(
        registry,
        "runtime.replay",
        &["scheduler.replay"],
        "Apply a typed control action journal to the live engine",
        CommandEffect::Control,
        json!({"type":"array","items":{"description":"ReplayAction"}}),
        |context, actions: Vec<ReplayAction>| replay(context, &actions),
    )?;
    add(
        registry,
        "events.drain",
        &[],
        "Consume retained semantic events",
        CommandEffect::Control,
        object(json!({}), &[]),
        drain,
    )?;
    Ok(())
}

fn inspect<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!(context.engine.inspect())
}
fn query<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!({
        "program": context.engine.program(),
        "requests": context.engine.request_records().collect::<Vec<_>>(),
        "pending": context.pending.iter().map(|(r, _)| r.id).collect::<Vec<_>>(),
        "expired": context.expired,
    })
}
fn submit<B: AgentBackend>(
    context: &mut AgentContext<B>,
    request: CanonicalRequest,
) -> Result<Value> {
    match context.engine.submit(request.clone()) {
        Ok(()) => Ok(json!({"accepted":true})),
        // A full device is a transient condition: queue instead of failing the request.
        Err(error) if error.code == ErrorCode::Capacity => {
            context.pending.push_back((request, None));
            Ok(json!({"accepted":true,"queued":true}))
        }
        Err(error) => Err(error),
    }
}
fn tick<B: AgentBackend>(context: &mut AgentContext<B>, params: TickParams) -> Result<Value> {
    // Drain queued submissions first, but keep the tick reply shape unchanged for clients.
    let expired = retry_pending(context, params.now_us)?;
    context.expired.extend(expired);
    Ok(json!(context.engine.tick(params.now_us)?))
}

/// Retry queued submissions; drop the ones that waited past the configured budget.
fn retry_pending<B: AgentBackend>(
    context: &mut AgentContext<B>,
    now_us: u64,
) -> Result<Vec<RequestId>> {
    let mut expired = Vec::new();
    let mut kept: VecDeque<(CanonicalRequest, Option<u64>)> = VecDeque::new();
    while let Some((request, queued_at)) = context.pending.pop_front() {
        let deadline = queued_at.unwrap_or(now_us);
        if now_us.saturating_sub(deadline) > QUEUE_WAIT_US {
            expired.push(request.id);
            continue;
        }
        match context.engine.submit(request.clone()) {
            Ok(()) => {}
            Err(error) if error.code == ErrorCode::Capacity => {
                kept.push_back((request, Some(deadline)));
            }
            Err(error) => return Err(error),
        }
    }
    context.pending = kept;
    Ok(expired)
}
fn cancel<B: AgentBackend>(context: &mut AgentContext<B>, params: RequestParams) -> Result<Value> {
    Ok(json!(context.engine.cancel(params.request_id)?))
}
fn explain<B: AgentBackend>(context: &mut AgentContext<B>, params: RequestParams) -> Result<Value> {
    context.engine.explain(params.request_id)
}
fn completed<B: AgentBackend>(
    context: &mut AgentContext<B>,
    params: RequestParams,
) -> Result<Value> {
    Ok(json!(context.engine.take_completed(params.request_id)?))
}
fn replay<B: AgentBackend>(
    context: &mut AgentContext<B>,
    actions: &[ReplayAction],
) -> Result<Value> {
    context.engine.replay(actions)?;
    Ok(json!(context.engine.inspect()))
}
fn drain<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!(context.engine.drain_events())
}
