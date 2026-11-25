use crate::constants::BENCHMARK_TIMEOUT_SECS;
use crate::constants::DEFAULT_ACCURACY_ATOL;
use crate::constants::DEFAULT_ACCURACY_RTOL;
use crate::constants::DEFAULT_MAX_P99_REGRESSION;
use crate::constants::MAX_BENCHMARK_REQUESTS;
use crate::constants::MAX_BENCHMARK_TICKS;
use crate::constants::MAX_EXPERIMENT_HISTORY;
use crate::{AgentBackend, AgentContext};
use infer_core::digest::lower_hex;
use infer_core::{Error, ErrorCode, ExperimentId, RequestId, Result};
use infer_ir::{CanonicalRequest, WorkloadOutput};
use infer_quality::{BenchmarkReport, ExperimentVerdict, VerificationReport};
use infer_runtime::{CompletedRequest, Engine, RuntimeConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, collections::BTreeSet, time::Instant};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkPlan {
    pub requests: Vec<CanonicalRequest>,
    pub ttft_slo_us: u64,
    pub tpot_slo_us: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AccuracyConstraints {
    pub atol: f64,
    pub rtol: f64,
    /// Optional independently produced expected outputs, covering every request.
    pub reference_outputs: Option<BTreeMap<RequestId, WorkloadOutput>>,
}
impl Default for AccuracyConstraints {
    fn default() -> Self {
        Self {
            atol: DEFAULT_ACCURACY_ATOL,
            rtol: DEFAULT_ACCURACY_RTOL,
            reference_outputs: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExperimentConstraints {
    pub accuracy: AccuracyConstraints,
    pub max_p99_regression: f64,
    pub min_goodput_ratio: f64,
    pub max_state_bytes: Option<u64>,
}
impl Default for ExperimentConstraints {
    fn default() -> Self {
        Self {
            accuracy: AccuracyConstraints::default(),
            max_p99_regression: DEFAULT_MAX_P99_REGRESSION,
            min_goodput_ratio: 1.0,
            max_state_bytes: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentPlan {
    pub baseline: RuntimeConfig,
    pub candidate: RuntimeConfig,
    pub workload: BenchmarkPlan,
    #[serde(default)]
    pub constraints: ExperimentConstraints,
}

#[derive(Debug, Clone, Serialize)]
pub struct AccuracyEvidence {
    pub request: RequestId,
    pub passed: bool,
    pub basis: &'static str,
    pub numeric: Option<VerificationReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentResult {
    pub id: ExperimentId,
    pub workload_fingerprint: String,
    pub baseline_config: RuntimeConfig,
    pub candidate_config: RuntimeConfig,
    pub constraints: ExperimentConstraints,
    pub baseline: BenchmarkReport,
    pub candidate: BenchmarkReport,
    pub baseline_peak_state_bytes: Option<u64>,
    pub candidate_peak_state_bytes: Option<u64>,
    pub correctness: Vec<AccuracyEvidence>,
    pub verdict: ExperimentVerdict,
}

pub(crate) struct MeasuredRun {
    pub report: BenchmarkReport,
    pub results: Vec<CompletedRequest>,
    pub peak_state_bytes: Option<u64>,
}

fn validate_workload(plan: &BenchmarkPlan) -> Result<()> {
    if plan.requests.is_empty()
        || plan.requests.len() > MAX_BENCHMARK_REQUESTS
        || plan.ttft_slo_us == 0
        || plan.tpot_slo_us == 0
    {
        return Err(Error::invalid(
            "benchmark requires 1 to 256 requests and positive SLOs",
        ));
    }
    let mut ids = BTreeSet::new();
    if plan.requests.iter().any(|request| !ids.insert(request.id)) {
        return Err(Error::invalid("duplicate experiment request IDs"));
    }
    Ok(())
}

pub(crate) fn measure<B: AgentBackend>(
    context: &AgentContext<B>,
    config: RuntimeConfig,
    plan: &BenchmarkPlan,
) -> Result<MeasuredRun> {
    validate_workload(plan)?;
    let backend = context.engine.backend().fresh()?;
    let registry = backend.registry()?;
    let mut measured = Engine::new(
        backend,
        context.engine.model().clone(),
        context.engine.program().precision.clone(),
        &registry,
        config,
    )?
    .with_workloads(context.engine.fork_workloads()?)?;
    let fingerprint = fingerprint(plan)?;
    let start = Instant::now();
    for request in &plan.requests {
        measured.submit(request.clone())?;
    }
    let mut peak_state_bytes = measured
        .backend()
        .execution_stats()
        .map(|s| s.reserved_bytes.max(s.allocated_bytes));
    for _ in 0..MAX_BENCHMARK_TICKS {
        if measured.is_idle() {
            break;
        }
        if start.elapsed().as_secs() >= BENCHMARK_TIMEOUT_SECS {
            return Err(Error::new(
                ErrorCode::Capacity,
                "isolated benchmark exceeded 30 second execution budget",
            ));
        }
        measured.tick(elapsed(&start))?;
        measured.drain_events();
        if let Some(stats) = measured.backend().execution_stats() {
            peak_state_bytes = Some(
                peak_state_bytes
                    .unwrap_or(0)
                    .max(stats.reserved_bytes)
                    .max(stats.allocated_bytes),
            );
        }
    }
    if !measured.is_idle() {
        return Err(Error::invariant("isolated benchmark exceeded tick budget"));
    }
    let results: Vec<_> = plan
        .requests
        .iter()
        .map(|request| {
            measured
                .request(request.id)?
                .completed
                .clone()
                .ok_or_else(|| Error::invariant("benchmark request has no completion"))
        })
        .collect::<Result<_>>()?;
    let samples: Vec<_> = results.iter().map(|r| r.measurement.clone()).collect();
    let report = infer_quality::benchmark(
        measured.inspect().backend,
        fingerprint,
        &samples,
        elapsed(&start).max(1),
        plan.ttft_slo_us,
        plan.tpot_slo_us,
    )?;
    Ok(MeasuredRun {
        report,
        results,
        peak_state_bytes,
    })
}

fn elapsed(start: &Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn fingerprint(plan: &BenchmarkPlan) -> Result<String> {
    let bytes = serde_json::to_vec(&plan.requests).map_err(|e| Error::invalid(e.to_string()))?;
    Ok(format!("sha256:{}", lower_hex(&Sha256::digest(bytes))))
}

fn validate_plan(plan: &ExperimentPlan) -> Result<()> {
    validate_workload(&plan.workload)?;
    plan.baseline.validate()?;
    plan.candidate.validate()?;
    let constraints = &plan.constraints;
    infer_quality::compare(
        &[0.0],
        &[0.0],
        constraints.accuracy.atol,
        constraints.accuracy.rtol,
    )?;
    if !constraints.max_p99_regression.is_finite()
        || constraints.max_p99_regression < 0.0
        || !constraints.min_goodput_ratio.is_finite()
        || constraints.min_goodput_ratio < 0.0
    {
        return Err(Error::invalid("invalid experiment performance constraints"));
    }
    if let Some(expected) = &constraints.accuracy.reference_outputs {
        let ids: BTreeSet<_> = plan.workload.requests.iter().map(|r| r.id).collect();
        if expected.keys().copied().collect::<BTreeSet<_>>() != ids {
            return Err(Error::invalid(
                "reference outputs must cover exactly the experiment workload",
            ));
        }
    }
    Ok(())
}

/// Execute both configurations with independent state and derive correctness from outputs.
/// # Errors
/// Returns plan validation, isolated execution, or measurement errors without mutating live requests.
pub fn run<B: AgentBackend>(
    context: &mut AgentContext<B>,
    plan: ExperimentPlan,
) -> Result<ExperimentResult> {
    validate_plan(&plan)?;
    let constraints = &plan.constraints;
    let baseline = measure(context, plan.baseline.clone(), &plan.workload)?;
    let candidate = measure(context, plan.candidate.clone(), &plan.workload)?;
    let mut correctness = Vec::new();
    for (base, candidate) in baseline.results.iter().zip(&candidate.results) {
        correctness.push(output_evidence(
            base.request,
            base.output.as_ref(),
            candidate.output.as_ref(),
            &constraints.accuracy,
            "baseline_candidate_parity",
        )?);
        if let Some(expected) = &constraints.accuracy.reference_outputs {
            correctness.push(output_evidence(
                base.request,
                expected.get(&base.request),
                base.output.as_ref(),
                &constraints.accuracy,
                "baseline_reference",
            )?);
            correctness.push(output_evidence(
                base.request,
                expected.get(&base.request),
                candidate.output.as_ref(),
                &constraints.accuracy,
                "candidate_reference",
            )?);
        }
    }
    let mut verdict = if baseline.report.slo_goodput > 0.0 {
        infer_quality::experiment(
            &baseline.report,
            &candidate.report,
            correctness.iter().all(|e| e.passed),
            constraints.max_p99_regression,
        )?
    } else {
        ExperimentVerdict {
            accepted: false,
            goodput_change: 0.0,
            reasons: vec!["baseline produced zero SLO goodput".into()],
        }
    };
    if baseline.report.successful != baseline.report.requests {
        verdict.reasons.push("baseline requests failed".into());
    }
    if candidate.report.slo_goodput < baseline.report.slo_goodput * constraints.min_goodput_ratio {
        verdict
            .reasons
            .push("goodput optimization objective failed".into());
    }
    if let Some(limit) = constraints.max_state_bytes {
        if candidate.peak_state_bytes.is_none() {
            verdict
                .reasons
                .push("backend cannot provide state memory evidence".into());
        } else if candidate
            .peak_state_bytes
            .is_some_and(|bytes| bytes > limit)
        {
            verdict
                .reasons
                .push("state memory constraint failed".into());
        }
    }
    verdict.accepted = verdict.reasons.is_empty();
    let result = ExperimentResult {
        id: context.ids.allocate()?,
        workload_fingerprint: baseline.report.workload_fingerprint.clone(),
        baseline_config: plan.baseline,
        candidate_config: plan.candidate,
        constraints: plan.constraints,
        baseline: baseline.report,
        candidate: candidate.report,
        baseline_peak_state_bytes: baseline.peak_state_bytes,
        candidate_peak_state_bytes: candidate.peak_state_bytes,
        correctness,
        verdict,
    };
    if context.experiments.len() == MAX_EXPERIMENT_HISTORY {
        context.experiments.pop_front();
    }
    context.experiments.push_back(result.clone());
    Ok(result)
}

fn output_evidence(
    id: RequestId,
    reference: Option<&WorkloadOutput>,
    candidate: Option<&WorkloadOutput>,
    accuracy: &AccuracyConstraints,
    basis: &'static str,
) -> Result<AccuracyEvidence> {
    let (Some(reference), Some(candidate)) = (reference, candidate) else {
        return Ok(AccuracyEvidence {
            request: id,
            passed: false,
            basis,
            numeric: None,
        });
    };
    if let (WorkloadOutput::Embedding(a), WorkloadOutput::Embedding(b)) = (reference, candidate) {
        if a.is_empty() || a.len() != b.len() {
            return Ok(AccuracyEvidence {
                request: id,
                passed: false,
                basis,
                numeric: None,
            });
        }
        let numeric = infer_quality::compare(a, b, accuracy.atol, accuracy.rtol)?;
        return Ok(AccuracyEvidence {
            request: id,
            passed: numeric.passed,
            basis,
            numeric: Some(numeric),
        });
    }
    // Preserve token IDs, ranking order and decision labels exactly; compare float leaves numerically.
    let reference = serde_json::to_value(reference).map_err(|e| Error::invalid(e.to_string()))?;
    let candidate = serde_json::to_value(candidate).map_err(|e| Error::invalid(e.to_string()))?;
    Ok(AccuracyEvidence {
        request: id,
        passed: equivalent(&reference, &candidate, accuracy),
        basis,
        numeric: None,
    })
}

fn equivalent(
    a: &serde_json::Value,
    b: &serde_json::Value,
    accuracy: &AccuracyConstraints,
) -> bool {
    match (a, b) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b))
            if a.is_f64() && b.is_f64() =>
        {
            a.as_f64().zip(b.as_f64()).is_some_and(|(a, b)| {
                a.is_finite()
                    && b.is_finite()
                    && (a - b).abs() <= accuracy.rtol.mul_add(a.abs(), accuracy.atol)
            })
        }
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| equivalent(a, b, accuracy))
        }
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| equivalent(a, b, accuracy)))
        }
        _ => a == b,
    }
}
