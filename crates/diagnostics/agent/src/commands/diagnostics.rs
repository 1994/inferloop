use super::{NoParams, add, object, runtime::RequestParams, runtime::request_schema};
use crate::{
    AgentBackend, AgentContext, graph, registry::CommandEffect, registry::CommandRegistry,
};
use infer_core::{Error, ErrorCode, KernelId, RequestId, Result};
use infer_ir::TensorStorage;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphParams {
    request_id: Option<RequestId>,
}

pub(super) fn register<B: AgentBackend>(
    registry: &mut CommandRegistry<AgentContext<B>>,
) -> Result<()> {
    for (method, description, execute) in [
        (
            "executor.inspect",
            "Inspect physical execution and device capabilities",
            execution as fn(&mut AgentContext<B>, NoParams) -> Value,
        ),
        (
            "backend.catalog",
            "Discover backend availability and production priority",
            catalog,
        ),
        (
            "executor.trace",
            "Query retained typed per-operation execution traces",
            trace,
        ),
        (
            "executor.probes",
            "Query bounded layer hidden-state samples",
            probes,
        ),
        (
            "model.bindings",
            "Query compiled tensor weight bindings",
            bindings,
        ),
        (
            "scheduler.inspect",
            "Inspect scheduler configuration, cost learning and decisions",
            scheduler,
        ),
        (
            "runtime.diagnose",
            "Check live invariants and report structured failure evidence",
            diagnose,
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
        "kernel.sources",
        &[],
        "Resolve compiled operations to registered kernel source",
        CommandEffect::ReadOnly,
        object(json!({}), &[]),
        sources,
    )?;
    add(
        registry,
        "gpu.profile",
        &["profile"],
        "Query backend profile with actual timing scope and source",
        CommandEffect::ReadOnly,
        object(json!({}), &[]),
        profile,
    )?;
    add(
        registry,
        "kernel.profile",
        &[],
        "Resolve one kernel and its retained operation timing evidence",
        CommandEffect::ReadOnly,
        object(
            json!({"kernel_id":{"type":"integer","minimum":1}}),
            &["kernel_id"],
        ),
        kernel_profile,
    )?;
    add(
        registry,
        "request.trace",
        &[],
        "Query request decisions and executed operation evidence",
        CommandEffect::ReadOnly,
        request_schema(),
        request_trace,
    )?;
    add(
        registry,
        "runtime.graph",
        &[],
        "Query typed Request/Decision/Step/State/Op/Kernel/source relationships",
        CommandEffect::ReadOnly,
        object(json!({"request_id":{"type":"integer","minimum":1}}), &[]),
        semantic_graph,
    )?;
    Ok(())
}

fn execution<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    context.engine.backend().inspection()
}
fn catalog<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    context.backend_catalog.clone()
}
fn trace<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!(context.engine.backend().traces())
}
fn probes<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!(context.engine.backend().probes())
}
fn profile<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    context.engine.backend().profile()
}
fn bindings<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!(
        context
            .engine
            .program()
            .dataflow
            .tensors
            .iter()
            .filter(|t| matches!(t.storage, TensorStorage::Weight { .. }))
            .collect::<Vec<_>>()
    )
}
fn scheduler<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!({"config":context.engine.config().scheduler,"admission":context.engine.config().admission,"cost_model":context.engine.inspect().cost_model,"pending_cost_observations":context.engine.inspect().pending_cost_observations,"decisions":context.engine.decisions()})
}
fn sources<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Result<Value> {
    let registry = context.engine.backend().registry()?;
    Ok(json!(context.engine.program().operations.iter().map(|op|json!({"op":op.op.id,"kernel":op.kernel,"source":registry.get(op.kernel).map(|k|&k.source)})).collect::<Vec<_>>()))
}
fn request_trace<B: AgentBackend>(
    context: &mut AgentContext<B>,
    params: RequestParams,
) -> Result<Value> {
    let id = params.request_id;
    Ok(
        json!({"request":context.engine.request(id)?,"decisions":context.engine.decisions().iter().filter(|d|d.step.as_ref().is_some_and(|s|s.work.iter().any(|w|w.request==id)) || d.deferred.iter().any(|w|w.request==id)).collect::<Vec<_>>(),"operations":context.engine.backend().traces().iter().filter(|t|t.request==id).collect::<Vec<_>>()}),
    )
}
fn semantic_graph<B: AgentBackend>(
    context: &mut AgentContext<B>,
    params: GraphParams,
) -> Result<Value> {
    Ok(json!(graph::query(&context.engine, params.request_id)?))
}
fn diagnose<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    let error = context.engine.check_invariants().err();

    json!({"invariants_passed":error.is_none(),"diagnostic":error,"retained_diagnostics":context.engine.diagnostics(),"runtime":context.engine.inspect(),"scope":"runtime ownership, execution failures and lifecycle invariants"})
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct KernelParams {
    kernel_id: KernelId,
}
fn kernel_profile<B: AgentBackend>(
    context: &mut AgentContext<B>,
    params: KernelParams,
) -> Result<Value> {
    let registry = context.engine.backend().registry()?;
    let kernel = registry.get(params.kernel_id).ok_or_else(|| {
        Error::new(
            ErrorCode::NotFound,
            "kernel is not registered on this backend",
        )
    })?;
    let traces = context.engine.backend().traces();
    let operations: Vec<_> = traces
        .iter()
        .filter(|trace| trace.kernel == params.kernel_id)
        .collect();
    Ok(
        json!({"kernel":params.kernel_id,"source":kernel.source,"timing_scope":context.engine.backend().trace_timing_scope(),"operations":operations,"dropped_traces":context.engine.backend().execution_stats().map_or(0, |s|s.trace_dropped)}),
    )
}
