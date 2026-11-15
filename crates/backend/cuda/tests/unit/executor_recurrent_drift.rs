//! Teacher-forced logit drift between recurrent-state profiles on a real checkpoint.
//!
//! One process per mode, all over the same token trajectories:
//! - `reference`: F32 states, sequences decoded together; keeps each position's top
//!   log-probabilities.
//! - `noise`: F32 states, one sequence at a time; how far batch composition alone moves logits.
//! - `candidate`: BF16 states, decoded together like the reference.
//!
//! Inputs and outputs are files named by the environment, so the diagnostic needs no product
//! option: `INFER_TEST_TRAJECTORIES` (JSON `[{"id", "prompt", "continuation"}]`),
//! `INFER_TEST_DRIFT_MODE`, `INFER_TEST_DRIFT_REFERENCE` (written by `reference`, read by the
//! others), `INFER_TEST_DRIFT_OUTPUT` (the summary of `noise` and `candidate`) and optionally
//! `INFER_TEST_DRIFT_LANES` for the batched modes, which long prompts lower to fit their KV.
use super::*;
use crate::RecurrentProfile;
use crate::device::{CudaDevice, asynchronous::block_on};
use crate::loading::{LoadOptions, LoadedModel};
use infer_core::{ModelId, RequestId, StateId};
use infer_ir::{ExecutionInput, ExecutionTask, TokenBuffer};
use serde::{Deserialize, Serialize};
use std::env::var;
use std::fs;
use std::slice::from_ref;

/// Sequences decoded together in the batched modes, unless `INFER_TEST_DRIFT_LANES` says.
const LANES: usize = 16;
/// Reference log-probabilities kept per position; the rest of its mass is one tail bucket.
const TOP: usize = 64;
/// Percentiles are read in thousandths.
const PER_MILLE: usize = 1000;
/// The KL percentile the acceptance compares.
const P99: usize = 990;
/// The median.
const P50: usize = 500;

#[derive(Deserialize)]
struct Trajectory {
    id: String,
    prompt: Vec<u32>,
    continuation: Vec<u32>,
}

/// The reference distribution at one position: its top tokens and their log-probabilities.
#[derive(Serialize, Deserialize)]
struct Position {
    tokens: Vec<u32>,
    log_probs: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct Reference {
    id: String,
    positions: Vec<Position>,
}

fn env(name: &str) -> Result<String> {
    var(name).map_err(|error| Error::invalid(format!("{name}: {error}")))
}

fn fixture(recurrent_profile: RecurrentProfile) -> Result<CudaExecutor> {
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        env("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 0,
            autotune: false,
            recurrent_profile,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let budget = loaded
        .device()
        .allocatable_memory_bytes()?
        .saturating_sub(loaded.device().profile()?.device_headroom_bytes())
        .saturating_sub(loaded.shared_scratch_bytes()?);
    let mut backend = CudaExecutor::new_automatic(loaded, budget, MAX_POOL_SLOTS)?;
    backend.capture_funded_pool();
    Ok(backend)
}

fn run(backend: &mut CudaExecutor, tasks: &[ExecutionTask]) -> Result<Vec<Vec<f32>>> {
    backend.prepare_execution()?;
    let result = block_on(backend.execute(tasks));
    let fence = block_on(backend.loaded.device().fence());
    backend.busy = false;
    fence?;
    Ok(result?.into_iter().map(|task| task.output.logits).collect())
}

fn task(state: StateId, tokens: ExecutionInput) -> ExecutionTask {
    ExecutionTask {
        plain_greedy: false,
        publish_prefix: false,
        request: RequestId::ONE,
        state,
        tokens,
        sampling: None,
        media: None,
    }
}

/// Natural-log softmax of one logits row, in F64.
fn log_softmax(logits: &[f32]) -> Result<Vec<f64>> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(Error::invariant("logits are empty or not finite"));
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f64> = logits
        .iter()
        .map(|&value| f64::from(value) - f64::from(maximum))
        .collect();
    let total = shifted.iter().map(|value| value.exp()).sum::<f64>().ln();
    Ok(shifted.into_iter().map(|value| value - total).collect())
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(index, _)| index)
}

/// Teacher-forced logits of every trajectory position, `lanes` sequences per step, handed to
/// `visit(sequence, position, log_probs)`: position 0 is the prompt's last row, position `t`
/// follows trajectory token `t - 1`.
fn force(
    backend: &mut CudaExecutor,
    trajectories: &[Trajectory],
    lanes: usize,
    mut visit: impl FnMut(usize, usize, Vec<f64>) -> Result<()>,
) -> Result<()> {
    for (group, batch) in trajectories.chunks(lanes).enumerate() {
        let states = (0..batch.len())
            .map(|lane| {
                StateId::new(u64::try_from(lane + 1).map_err(|e| Error::invalid(e.to_string()))?)
            })
            .collect::<Result<Vec<_>>>()?;
        let buffers: Vec<TokenBuffer> = batch
            .iter()
            .map(|t| TokenBuffer::from(t.prompt.clone()))
            .collect();
        let mut prefill = Vec::new();
        for ((state, buffer), trajectory) in states.iter().zip(&buffers).zip(batch) {
            backend.reserve_state_for(
                *state,
                trajectory.prompt.len() + trajectory.continuation.len() + 1,
                OutputReadout::Logits,
            )?;
            prefill.push(task(
                *state,
                ExecutionInput::Prefill {
                    span: buffer.span(0..buffer.len())?,
                    readout: OutputReadout::Logits,
                },
            ));
        }
        // Prompts one at a time: the comparison is about recurrent state, and every mode
        // must reach the first decoded position the same way.
        for (lane, prefill) in prefill.iter().enumerate() {
            let logits = run(backend, from_ref(prefill))?.remove(0);
            visit(group * lanes + lane, 0, log_softmax(&logits)?)?;
        }
        let longest = batch
            .iter()
            .map(|t| t.continuation.len())
            .max()
            .unwrap_or(0);
        for step in 1..longest {
            let mut decode = Vec::new();
            let mut owners = Vec::new();
            for (lane, trajectory) in batch.iter().enumerate() {
                if step < trajectory.continuation.len() {
                    decode.push(task(
                        states[lane],
                        ExecutionInput::Decode {
                            token: trajectory.continuation[step - 1],
                            position: trajectory.prompt.len() + step - 1,
                        },
                    ));
                    owners.push(lane);
                }
            }
            for (lane, logits) in owners.into_iter().zip(run(backend, &decode)?) {
                visit(group * lanes + lane, step, log_softmax(&logits)?)?;
            }
        }
        for state in states {
            backend.release_state(state)?;
        }
    }
    Ok(())
}

/// `KL(reference || actual)` over the reference's top tokens plus one bucket for the rest.
fn divergence(reference: &Position, actual: &[f64]) -> Result<f64> {
    let mut kl = 0.0;
    let (mut p_top, mut q_top) = (0.0, 0.0);
    for (&token, &log_p) in reference.tokens.iter().zip(&reference.log_probs) {
        let index = usize::try_from(token).map_err(|e| Error::invalid(e.to_string()))?;
        let log_q = *actual
            .get(index)
            .ok_or_else(|| Error::invalid("reference token outside the vocabulary"))?;
        let p = log_p.exp();
        kl = p.mul_add(log_p - log_q, kl);
        p_top += p;
        q_top += log_q.exp();
    }
    let (p_tail, q_tail) = ((1.0 - p_top).max(0.0), (1.0 - q_top).max(f64::MIN_POSITIVE));
    if p_tail > 0.0 {
        kl = p_tail.mul_add(p_tail.ln() - q_tail.ln(), kl);
    }
    Ok(kl.max(0.0))
}

/// The `per_mille` thousandth of a sorted sample, by nearest rank.
fn percentile(sorted: &[f64], per_mille: usize) -> f64 {
    let index = sorted.len().saturating_sub(1) * per_mille / PER_MILLE;
    sorted.get(index).copied().unwrap_or(f64::NAN)
}

/// `part / whole` for counts well inside F64's exact integers.
fn ratio(part: usize, whole: usize) -> Result<f64> {
    let exact = |count: usize| {
        u32::try_from(count)
            .map(f64::from)
            .map_err(|e| Error::invalid(e.to_string()))
    };
    Ok(exact(part)? / exact(whole)?)
}

/// The state profile and lanes per step of `mode`.
fn profile_and_lanes(mode: &str) -> Result<(RecurrentProfile, usize)> {
    let batched = match var("INFER_TEST_DRIFT_LANES") {
        Ok(lanes) => lanes
            .parse::<usize>()
            .map_err(|e| Error::invalid(format!("INFER_TEST_DRIFT_LANES: {e}")))?
            .max(1),
        Err(_) => LANES,
    };
    Ok(match mode {
        "reference" => (RecurrentProfile::F32, batched),
        "candidate" => (RecurrentProfile::Bf16, batched),
        "noise" => (RecurrentProfile::F32, 1),
        _ => {
            return Err(Error::invalid(
                "INFER_TEST_DRIFT_MODE: reference, noise or candidate",
            ));
        }
    })
}

/// Keep the reference's top log-probabilities of every position in `path`.
fn write_reference(
    backend: &mut CudaExecutor,
    trajectories: &[Trajectory],
    lanes: usize,
    path: &str,
) -> Result<()> {
    let mut kept: Vec<Reference> = trajectories
        .iter()
        .map(|t| Reference {
            id: t.id.clone(),
            positions: Vec::new(),
        })
        .collect();
    force(
        backend,
        trajectories,
        lanes,
        |sequence, position, log_probs| {
            assert_eq!(kept[sequence].positions.len(), position);
            let mut order: Vec<usize> = (0..log_probs.len()).collect();
            order.select_nth_unstable_by(TOP, |a, b| log_probs[*b].total_cmp(&log_probs[*a]));
            order.truncate(TOP);
            order.sort_by(|a, b| log_probs[*b].total_cmp(&log_probs[*a]));
            kept[sequence].positions.push(Position {
                tokens: order
                    .iter()
                    .map(|&i| u32::try_from(i).map_err(|e| Error::invalid(e.to_string())))
                    .collect::<Result<_>>()?,
                log_probs: order.iter().map(|&i| log_probs[i]).collect(),
            });
            Ok(())
        },
    )?;
    let bytes = serde_json::to_vec(&kept).map_err(|e| Error::invalid(e.to_string()))?;
    fs::write(path, bytes).map_err(|e| Error::invalid(e.to_string()))?;
    Ok(())
}

#[test]
#[ignore = "real CUDA checkpoint; INFER_TEST_MODEL, INFER_TEST_TRAJECTORIES, INFER_TEST_DRIFT_*"]
fn teacher_forced_recurrent_profile_drift() -> Result<()> {
    let mode = env("INFER_TEST_DRIFT_MODE")?;
    let trajectories: Vec<Trajectory> = serde_json::from_slice(
        &fs::read(env("INFER_TEST_TRAJECTORIES")?).map_err(|e| Error::invalid(e.to_string()))?,
    )
    .map_err(|e| Error::invalid(e.to_string()))?;
    let reference_path = env("INFER_TEST_DRIFT_REFERENCE")?;
    let (profile, lanes) = profile_and_lanes(&mode)?;
    let mut backend = fixture(profile)?;
    if mode == "reference" {
        return write_reference(&mut backend, &trajectories, lanes, &reference_path);
    }
    let reference: Vec<Reference> = serde_json::from_slice(
        &fs::read(&reference_path).map_err(|e| Error::invalid(e.to_string()))?,
    )
    .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(reference.len(), trajectories.len());
    let mut kl = Vec::new();
    let mut agree = 0usize;
    let mut first_divergence: Vec<Option<usize>> = vec![None; trajectories.len()];
    force(
        &mut backend,
        &trajectories,
        lanes,
        |sequence, position, log_probs| {
            let expected = &reference[sequence].positions[position];
            let same = u32::try_from(argmax(&log_probs)).ok() == expected.tokens.first().copied();
            agree += usize::from(same);
            if !same && first_divergence[sequence].is_none() {
                first_divergence[sequence] = Some(position);
            }
            kl.push(divergence(expected, &log_probs)?);
            Ok(())
        },
    )?;
    let positions = kl.len();
    let kl_mean = kl.iter().sum::<f64>() / ratio(positions, 1)?;
    kl.sort_by(f64::total_cmp);
    let summary = serde_json::json!({
        "mode": mode,
        "recurrent_profile": profile.name(),
        "lanes": lanes,
        "sequences": trajectories.len(),
        "positions": positions,
        "top1_agreement": ratio(agree, positions)?,
        "kl_mean": kl_mean,
        "kl_p50": percentile(&kl, P50),
        "kl_p99": percentile(&kl, P99),
        "kl_max": kl.last().copied().unwrap_or(f64::NAN),
        "kl_method": "KL(reference || actual) over the reference top 64 tokens plus one tail bucket",
        "first_divergence": trajectories.iter().zip(&first_divergence)
            .map(|(t, p)| serde_json::json!({"id": t.id, "position": p}))
            .collect::<Vec<_>>(),
    });
    println!("{summary}");
    fs::write(env("INFER_TEST_DRIFT_OUTPUT")?, summary.to_string())
        .map_err(|e| Error::invalid(e.to_string()))
}
