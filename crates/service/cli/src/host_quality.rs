//! Independent golden validation of loaded weights and their executable program.
use crate::backend::Selection;
use crate::constants::DEVICE_POLL_INTERVAL;
use crate::constants::MAX_IDLE_TICKS;
use crate::constants::MAX_JSON_FILE_BYTES;
use crate::{backend::SelectedBackend, support::selected_engine};
use infer_core::{DecisionId, Error, ErrorCode, RequestId, Result, StateId, StepId};
use infer_ir::{
    CanonicalRequest, CostEstimate, ExecutionRole, ExecutionTask, LayerProbe, PlannedWork, Qos,
    RequestInput, Sampling, StepPlan, Workload, WorkloadOutput,
};
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::BackendProvider;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::iter::once;
use std::path::Path;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;

/// Bytes per F32 hidden-state element in sampled layer-probe payloads.
const F32_BYTES_U64: u64 = size_of::<f32>() as u64;
/// Wall-clock budget for one golden prefill step on a device backend.
const GOLDEN_STEP_TIMEOUT: Duration = Duration::from_secs(30);
/// Leading token budgets exercised for golden trajectory invariance.
const TRAJECTORY_CHUNK_TOKENS: [usize; 2] = [1, 3];

#[derive(Deserialize)]
struct Golden {
    prefixes: Vec<Prefix>,
    greedy_tokens: Vec<u32>,
}
#[derive(Deserialize)]
struct Prefix {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    hidden: Vec<Vec<f32>>,
    #[serde(default)]
    layers: Vec<Vec<Vec<f32>>>,
}

pub fn verify_package(
    package: &Path,
    golden_path: &Path,
    memory_mib: u64,
    atol: f64,
    rtol: f64,
    choice: &Selection,
) -> Result<serde_json::Value> {
    let file = File::open(golden_path).map_err(|e| Error::invalid(e.to_string()))?;
    if file
        .metadata()
        .map_err(|e| Error::invalid(e.to_string()))?
        .len()
        > MAX_JSON_FILE_BYTES
    {
        return Err(Error::new(
            ErrorCode::Capacity,
            "golden file exceeds 64 MiB",
        ));
    }
    let golden: Golden =
        serde_json::from_reader(file).map_err(|e| Error::invalid(e.to_string()))?;
    if golden.prefixes.is_empty() || golden.greedy_tokens.is_empty() {
        return Err(Error::invalid(
            "golden needs prefixes and expected generation",
        ));
    }
    let mut engine = selected_engine(
        RuntimeConfig::default(),
        None,
        Some(package),
        memory_mib,
        choice.clone(),
    )?;
    let model = engine.model().clone();
    let program = engine.program().clone();
    let backend = engine.backend_mut();
    let sample_bytes = (model.hidden_size as u64 * F32_BYTES_U64 + size_of::<LayerProbe>() as u64)
        .checked_mul(model.mixers.len() as u64)
        .and_then(|n| n.checked_mul(golden.prefixes.last()?.tokens.len() as u64))
        .ok_or_else(|| Error::invalid("probe size overflow"))?;
    if golden.prefixes.iter().any(|p| !p.layers.is_empty()) {
        backend.enable_layer_probes(sample_bytes)?;
    }
    let state = StateId::new(1)?;
    backend.reserve_state(
        state,
        golden
            .prefixes
            .last()
            .ok_or_else(|| Error::invariant("nonempty"))?
            .tokens
            .len(),
    )?;
    let PrefixReport {
        mut passed,
        reports,
        executed,
    } = verify_prefixes(backend, &program, &model, &golden, state, atol, rtol)?;
    backend.release_state(state)?;
    let incremental_tokens = backend
        .execution_stats()
        .ok_or_else(|| Error::invariant("package backend"))?
        .tokens_executed;
    let (trajectory_passed, trajectories) = verify_trajectories(
        package,
        memory_mib,
        choice,
        &model,
        &executed,
        &golden.greedy_tokens,
    )?;
    passed &= trajectory_passed;
    Ok(
        serde_json::json!({"scope":"selected backend weights vs independent golden; excludes trained task quality","backend":program.backend,
        "prefixes":reports,"incremental_tokens_executed":incremental_tokens,
        "trajectory_batch_chunk":trajectories,"passed":passed}),
    )
}

struct PrefixReport {
    passed: bool,
    reports: Vec<serde_json::Value>,
    executed: Vec<u32>,
}
fn verify_prefixes(
    backend: &mut SelectedBackend,
    program: &infer_ir::ExecutionProgram,
    model: &infer_ir::ModelIr,
    golden: &Golden,
    state: StateId,
    atol: f64,
    rtol: f64,
) -> Result<PrefixReport> {
    let mut executed = Vec::new();
    let mut reports = Vec::new();
    let mut passed = true;
    for prefix in &golden.prefixes {
        if !prefix.tokens.starts_with(&executed) || prefix.tokens.len() <= executed.len() {
            return Err(Error::invalid(
                "golden prefixes must strictly extend one sequence",
            ));
        }
        let request = RequestId::new(1)?;
        let step = StepPlan {
            id: StepId::new(prefix.tokens.len() as u64)?,
            decision: DecisionId::new(1)?,
            program: program.id,
            role: ExecutionRole::Prefill,
            work: vec![PlannedWork {
                request,
                state,
                token_count: prefix.tokens.len() - executed.len(),
                role: ExecutionRole::Prefill,
            }],
            cost: CostEstimate::default(),
            graph: None,
            quantum_overrun: false,
        };
        let mut ticket = backend.submit(
            program,
            &step,
            vec![ExecutionTask {
                plain_greedy: false,
                request,
                state,
                tokens: prefix.tokens.clone().into(),

                sampling: None,

                media: None,
            }],
        )?;
        let deadline = Instant::now() + GOLDEN_STEP_TIMEOUT;
        let result = loop {
            if let Some(mut output) = backend.poll(&mut ticket)? {
                break output.remove(0).output;
            }
            if Instant::now() > deadline {
                return Err(Error::new(
                    ErrorCode::Backend,
                    "golden GPU completion timeout",
                ));
            }
            sleep(DEVICE_POLL_INTERVAL);
        };
        let logits = infer_quality::compare(&prefix.logits, &result.logits, atol, rtol)?;
        let expected: Vec<_> = prefix.hidden.iter().flatten().copied().collect();
        let actual: Vec<_> = result.hidden.iter().flatten().copied().collect();
        let hidden = infer_quality::compare(&expected, &actual, atol, rtol)?;
        passed &= logits.passed && hidden.passed;
        let mut layers = Vec::new();
        let probes = backend.drain_layer_probes();
        if !prefix.layers.is_empty() {
            if prefix.layers.len() != model.mixers.len()
                || probes.len() != model.mixers.len() * (prefix.tokens.len() - executed.len())
            {
                return Err(Error::invalid("golden layer coverage/shape mismatch"));
            }
            for probe in probes {
                let expected = prefix
                    .layers
                    .get(probe.layer)
                    .and_then(|positions| positions.get(probe.position))
                    .ok_or_else(|| Error::invalid("missing golden layer position"))?;
                let metric = infer_quality::compare(expected, &probe.hidden, atol, rtol)?;
                passed &= metric.passed;
                layers.push(serde_json::json!({"layer":probe.layer,"position":probe.position,"op":probe.op,"metric":metric}));
            }
        }
        reports.push(serde_json::json!({"input_tokens":prefix.tokens.len(),"logits":logits,"hidden":hidden,"layers":layers}));
        executed.clone_from(&prefix.tokens);
    }
    Ok(PrefixReport {
        passed,
        reports,
        executed,
    })
}
fn verify_trajectories(
    package: &Path,
    memory_mib: u64,
    choice: &Selection,
    model: &infer_ir::ModelIr,
    executed: &[u32],
    expected_tokens: &[u32],
) -> Result<(bool, Vec<serde_json::Value>)> {
    let mut passed = true;
    let mut trajectories = Vec::new();
    for chunk in TRAJECTORY_CHUNK_TOKENS
        .into_iter()
        .chain(once(model.max_sequence))
    {
        let config = RuntimeConfig {
            max_num_batched_tokens: chunk,
            ..Default::default()
        };
        let mut candidate: Engine<SelectedBackend> =
            selected_engine(config, None, Some(package), memory_mib, choice.clone())?;
        let ids = [RequestId::new(1)?, RequestId::new(2)?];
        for id in ids {
            candidate.submit(CanonicalRequest {
                id,
                model: model.id,
                session: None,
                input: RequestInput::Sequence {
                    tokens: executed.to_vec().into(),
                    media: vec![],
                },
                workload: Workload::Generate {
                    max_new_tokens: expected_tokens.len(),
                },
                qos: Qos::default(),
                sampling: Sampling::default(),
                extensions: BTreeMap::new(),
            })?;
        }
        for now in 0..MAX_IDLE_TICKS {
            if candidate.is_idle() {
                break;
            }
            candidate.tick(now)?;
            candidate.drain_events();
            if candidate.inspect().inflight_step.is_some()
                && candidate.program().backend.is_device()
            {
                sleep(DEVICE_POLL_INTERVAL);
            }
        }
        if !candidate.is_idle() {
            return Err(Error::invariant(
                "golden trajectory exceeded progress limit",
            ));
        }
        let trajectory_passed = ids.iter().all(|id| {
            candidate
                .request(*id)
                .ok()
                .and_then(|record| record.completed.as_ref())
                .is_some_and(|done| {
                    done.measurement.successful
                        && done.output
                            == Some(WorkloadOutput::Tokens(expected_tokens.to_vec().into()))
                })
        });
        let state_leaks = candidate
            .backend()
            .execution_stats()
            .ok_or_else(|| Error::invariant("package backend"))?
            .sequences
            + candidate.inspect().state.sequence_count;
        passed &= trajectory_passed && state_leaks == 0;
        trajectories.push(serde_json::json!({"chunk":chunk,"requests":2,"passed":trajectory_passed,"state_leaks":state_leaks}));
    }
    Ok((passed, trajectories))
}
