use super::{NoParams, add, object};
use crate::{AgentBackend, AgentContext, registry::CommandEffect, registry::CommandRegistry};
use infer_core::Result;
use infer_runtime::{Engine, RuntimeSnapshot};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureParams {
    now_us: Option<u64>,
}

pub(super) fn register<B: AgentBackend>(
    registry: &mut CommandRegistry<AgentContext<B>>,
) -> Result<()> {
    add(
        registry,
        "state.inspect",
        &[],
        "Inspect logical state ownership and free pages",
        CommandEffect::ReadOnly,
        object(json!({}), &[]),
        inspect,
    )?;
    add(
        registry,
        "kv.inspect",
        &["cache.inspect"],
        "Inspect physical page pool, cache reuse and preemption",
        CommandEffect::ReadOnly,
        object(json!({}), &[]),
        cache,
    )?;
    add(
        registry,
        "snapshot.capture",
        &[],
        "Quiesce submission and capture physical/control checkpoint",
        CommandEffect::Control,
        object(json!({"now_us":{"type":"integer","minimum":0}}), &[]),
        capture,
    )?;
    add(
        registry,
        "snapshot.replay",
        &[],
        "Validate and atomically replace live engine from checkpoint",
        CommandEffect::Control,
        json!({"type":"object","description":"RuntimeSnapshot; identities and ownership validated during restore"}),
        restore,
    )?;
    Ok(())
}
fn inspect<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!(context.engine.inspect().state)
}
fn cache<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!({"pool":context.engine.inspect().kv_cache,"preemptions":context.engine.inspect().preemptions,"logical_state":context.engine.inspect().state,"execution":context.engine.backend().execution_stats()})
}
fn capture<B: AgentBackend>(context: &mut AgentContext<B>, params: CaptureParams) -> Result<Value> {
    let (events, snapshot) = context
        .engine
        .quiesce(params.now_us.unwrap_or_else(|| context.engine.now_us()))?;
    Ok(json!({"events":events,"pending":snapshot.is_none(),"snapshot":snapshot}))
}
fn restore<B: AgentBackend>(
    context: &mut AgentContext<B>,
    snapshot: RuntimeSnapshot,
) -> Result<Value> {
    let backend = context.engine.backend().fresh()?;
    let registry = backend.registry()?;
    let restored = Engine::restore_with_workloads(
        backend,
        &registry,
        snapshot,
        context.engine.fork_workloads()?,
    )?;
    context.engine = restored;
    Ok(json!(context.engine.inspect()))
}
