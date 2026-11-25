use super::{NoParams, add, object};
use crate::constants::DEFAULT_EVENT_LIMIT;
use crate::constants::MAX_EVENT_LIMIT;
use crate::{AgentBackend, AgentContext, registry::CommandEffect, registry::CommandRegistry};
use infer_core::{RequestId, Result};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Events {
    after: u64,
    limit: usize,
    request_id: Option<RequestId>,
}
impl Default for Events {
    fn default() -> Self {
        Self {
            after: 0,
            limit: DEFAULT_EVENT_LIMIT,
            request_id: None,
        }
    }
}

pub(super) fn register<B: AgentBackend>(
    registry: &mut CommandRegistry<AgentContext<B>>,
) -> Result<()> {
    for (method, description, execute) in [
        (
            "observability.inspect",
            "Inspect always-on metrics, bounded retention, resources and engine fault",
            inspect as fn(&mut AgentContext<B>, NoParams) -> Value,
        ),
        (
            "observability.metrics",
            "Export counters, gauges and duration histograms in Prometheus text",
            metrics,
        ),
        (
            "observability.diagnostics",
            "Query retained typed admission, execution and progress failures",
            diagnostics,
        ),
        (
            "observability.timeline",
            "Export retained semantic events in Chrome trace JSON with loss metadata",
            timeline,
        ),
        (
            "observability.otlp",
            "Export complete retained request spans as OTLP JSON using W3C trace parents",
            otlp,
        ),
    ] {
        add(
            registry,
            method,
            &[],
            description,
            CommandEffect::ReadOnly,
            object(json!({}), &[]),
            execute,
        )?;
    }
    add(
        registry,
        "observability.events",
        &[],
        "Query semantic history non-destructively with a cursor and explicit loss",
        CommandEffect::ReadOnly,
        object(
            json!({"after":{"type":"integer","minimum":0},"limit":{"type":"integer","minimum":1,"maximum":MAX_EVENT_LIMIT},"request_id":{"type":"integer","minimum":1}}),
            &[],
        ),
        events,
    )
}
fn inspect<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    context.engine.observability()
}
fn metrics<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!({"format":"prometheus","text":context.engine.prometheus()})
}
fn diagnostics<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!({"diagnostics":context.engine.diagnostics(),"fault":context.engine.inspect().fault})
}
fn timeline<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    context.engine.timeline()
}
fn otlp<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    context.engine.otlp()
}
fn events<B: AgentBackend>(context: &mut AgentContext<B>, params: Events) -> Result<Value> {
    Ok(json!(context.engine.observed_events(
        params.after,
        params.limit,
        params.request_id
    )?))
}
