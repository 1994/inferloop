use super::{NoParams, add, object};
use crate::constants::MAX_BENCHMARK_REQUESTS;
use crate::{
    AgentBackend, AgentContext, experiment, experiment::BenchmarkPlan, experiment::ExperimentPlan,
    registry::CommandEffect, registry::CommandRegistry,
};
use infer_core::{Error, ErrorCode, ExperimentId, Result};
use infer_quality::BenchmarkReport;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericParams {
    reference: Vec<f32>,
    candidate: Vec<f32>,
    atol: f64,
    rtol: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompareParams {
    baseline: BenchmarkReport,
    candidate: BenchmarkReport,
    correctness_passed: bool,
    max_p99_regression: f64,
}
#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentParams {
    experiment_id: ExperimentId,
}

pub(super) fn register<B: AgentBackend>(
    registry: &mut CommandRegistry<AgentContext<B>>,
) -> Result<()> {
    let numeric = object(
        json!({"reference":{"type":"array","items":{"type":"number"}},"candidate":{"type":"array","items":{"type":"number"}},"atol":{"type":"number","minimum":0},"rtol":{"type":"number","minimum":0}}),
        &["reference", "candidate", "atol", "rtol"],
    );
    add(
        registry,
        "accuracy.verify",
        &["verify.numeric", "verify"],
        "Compare numeric arrays with finite absolute/relative tolerances",
        CommandEffect::ReadOnly,
        numeric,
        |context, params: NumericParams| verify(context, &params),
    )?;
    let workload = object(
        json!({"requests":{"type":"array","minItems":1,"maxItems":MAX_BENCHMARK_REQUESTS,"items":{"type":"object","description":"CanonicalRequest"}},"ttft_slo_us":{"type":"integer","minimum":1},"tpot_slo_us":{"type":"integer","minimum":1}}),
        &["requests", "ttft_slo_us", "tpot_slo_us"],
    );
    add(
        registry,
        "benchmark.run",
        &["benchmark.measure", "benchmark"],
        "Measure workload on a fresh engine without changing live requests",
        CommandEffect::IsolatedExecution,
        workload.clone(),
        |context, params: BenchmarkPlan| benchmark(context, &params),
    )?;
    add(
        registry,
        "benchmark.compare",
        &["experiment.compare", "compare"],
        "Compare supplied reports; correctness is caller-provided evidence",
        CommandEffect::ReadOnly,
        object(
            json!({"baseline":{"type":"object"},"candidate":{"type":"object"},"correctness_passed":{"type":"boolean"},"max_p99_regression":{"type":"number","minimum":0}}),
            &[
                "baseline",
                "candidate",
                "correctness_passed",
                "max_p99_regression",
            ],
        ),
        |context, params: CompareParams| compare(context, &params),
    )?;
    add(
        registry,
        "experiment.run",
        &[],
        "Execute baseline/candidate configs, verify actual outputs and apply acceptance constraints",
        CommandEffect::IsolatedExecution,
        object(
            json!({"baseline":{"type":"object","description":"RuntimeConfig"},"candidate":{"type":"object","description":"RuntimeConfig"},"workload":workload,"constraints":{"type":"object","description":"Accuracy, P99, goodput and state-memory constraints"}}),
            &["baseline", "candidate", "workload"],
        ),
        run,
    )?;
    add(
        registry,
        "experiment.inspect",
        &[],
        "Read retained experiment execution and acceptance evidence",
        CommandEffect::ReadOnly,
        object(
            json!({"experiment_id":{"type":"integer","minimum":1}}),
            &["experiment_id"],
        ),
        inspect,
    )?;
    add(
        registry,
        "experiment.list",
        &[],
        "List bounded experiment history and verdicts",
        CommandEffect::ReadOnly,
        object(json!({}), &[]),
        list,
    )?;
    Ok(())
}
fn verify<B: AgentBackend>(_: &mut AgentContext<B>, params: &NumericParams) -> Result<Value> {
    Ok(json!(infer_quality::compare(
        &params.reference,
        &params.candidate,
        params.atol,
        params.rtol
    )?))
}
fn benchmark<B: AgentBackend>(context: &AgentContext<B>, plan: &BenchmarkPlan) -> Result<Value> {
    let measured = experiment::measure(context, context.engine.config().clone(), plan)?;
    Ok(
        json!({"scope":"selected backend closed-loop execution","report":measured.report,"peak_state_bytes":measured.peak_state_bytes}),
    )
}
fn compare<B: AgentBackend>(_: &mut AgentContext<B>, params: &CompareParams) -> Result<Value> {
    Ok(json!(infer_quality::experiment(
        &params.baseline,
        &params.candidate,
        params.correctness_passed,
        params.max_p99_regression
    )?))
}
fn run<B: AgentBackend>(context: &mut AgentContext<B>, plan: ExperimentPlan) -> Result<Value> {
    Ok(json!(experiment::run(context, plan)?))
}
fn inspect<B: AgentBackend>(
    context: &mut AgentContext<B>,
    params: ExperimentParams,
) -> Result<Value> {
    let result = context
        .experiments
        .iter()
        .find(|e| e.id == params.experiment_id)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::NotFound,
                "experiment is not retained in this session",
            )
        })?;
    Ok(json!(result))
}
fn list<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!(context.experiments.iter().map(|e|json!({"id":e.id,"verdict":e.verdict,"workload_fingerprint":e.workload_fingerprint})).collect::<Vec<_>>())
}
