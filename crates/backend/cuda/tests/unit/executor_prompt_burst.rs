use super::*;
use crate::AttentionProfile;
use crate::device::{CudaDevice, asynchronous::block_on};
use crate::loading::{LoadOptions, LoadedModel};
use infer_core::{ModelId, RequestId, StateId};
use infer_ir::{ExecutionInput, ExecutionTask, TokenBuffer};
use std::env::var;
use std::slice::from_ref;

/// Prompts a replay carries, leaving slots for the idle-segment check.
const PROMPTS: usize = 6;
/// Tokens a replay decodes after its prompts, which read everything the prompts wrote.
const DECODE_STEPS: usize = 4;
/// Relative logit tolerance across the two prompt graphs' F32 reduction trees.
const LOGIT_TOLERANCE: f32 = 5.0e-3;

fn fixture() -> Result<CudaExecutor> {
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL").map_err(|error| Error::invalid(error.to_string()))?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 0,
            mtp_depth: 0,
            autotune: false,
            attention_profile: AttentionProfile::Canonical,
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

fn prompt(seed: usize, length: usize) -> Vec<u32> {
    (0..length)
        .map(|index| u32::try_from(1000 + (seed * 131 + index * 17) % 5000).unwrap_or(1000))
        .collect()
}

fn state(index: usize) -> Result<StateId> {
    StateId::new(u64::try_from(index + 1).map_err(|e| Error::invalid(e.to_string()))?)
}

fn run(backend: &mut CudaExecutor, tasks: &[ExecutionTask]) -> Result<Vec<Vec<f32>>> {
    backend.prepare_execution()?;
    let result = block_on(backend.execute(tasks));
    let fence = block_on(backend.loaded.device().fence());
    backend.busy = false;
    fence?;
    Ok(result?.into_iter().map(|task| task.output.logits).collect())
}

fn prefill_task(id: StateId, buffer: &TokenBuffer) -> Result<ExecutionTask> {
    Ok(ExecutionTask {
        plain_greedy: false,
        publish_prefix: false,
        request: RequestId::ONE,
        state: id,
        tokens: ExecutionInput::Prefill {
            span: buffer.span(0..buffer.len())?,
            readout: OutputReadout::Logits,
        },
        sampling: None,
        media: None,
    })
}

fn greedy(logits: &[f32]) -> Result<u32> {
    let best = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .ok_or_else(|| Error::invariant("empty logits"))?;
    u32::try_from(best).map_err(|e| Error::invalid(e.to_string()))
}

/// Logits of every prompt and of `DECODE_STEPS` single-sequence decodes after it.
fn trace(backend: &mut CudaExecutor, together: bool) -> Result<Vec<Vec<Vec<f32>>>> {
    let buffers: Vec<_> = (0..PROMPTS)
        .map(|seed| TokenBuffer::from(prompt(seed, 40 + seed * 4)))
        .collect();
    let mut tasks = Vec::new();
    for (index, buffer) in buffers.iter().enumerate() {
        let id = state(index)?;
        backend.reserve_state_for(id, 256, OutputReadout::Logits)?;
        assert!(backend.states[&id].home.is_some());
        tasks.push(prefill_task(id, buffer)?);
    }
    let first = if together {
        let logits = run(backend, &tasks)?;
        assert!(
            backend
                .worker
                .as_ref()
                .is_some_and(|w| !w.bursts.is_empty()),
            "fixture has no multi-sequence prompt graph"
        );
        logits
    } else {
        // One task per step never reaches the multi-sequence graph.
        let mut logits = Vec::new();
        for task in &tasks {
            logits.extend(run(backend, from_ref(task))?);
        }
        logits
    };
    let mut traces = Vec::new();
    for (index, (buffer, logits)) in buffers.iter().zip(first).enumerate() {
        let id = state(index)?;
        let mut token = greedy(&logits)?;
        let mut steps = vec![logits];
        for position in buffer.len()..buffer.len() + DECODE_STEPS {
            let task = ExecutionTask {
                plain_greedy: false,
                publish_prefix: false,
                request: RequestId::ONE,
                state: id,
                tokens: ExecutionInput::Decode { token, position },
                sampling: None,
                media: None,
            };
            let logits = run(backend, from_ref(&task))?
                .pop()
                .ok_or_else(|| Error::invariant("decode output"))?;
            token = greedy(&logits)?;
            steps.push(logits);
        }
        traces.push(steps);
    }
    for index in 0..PROMPTS {
        backend.release_state(state(index)?)?;
    }
    Ok(traces)
}

/// Every lane through the full-width decode graph at once, after prompts through one graph.
fn trace_together(backend: &mut CudaExecutor) -> Result<Vec<Vec<Vec<f32>>>> {
    let buffers: Vec<_> = (0..PROMPTS)
        .map(|seed| TokenBuffer::from(prompt(seed, 40 + seed * 4)))
        .collect();
    let mut tasks = Vec::new();
    for (index, buffer) in buffers.iter().enumerate() {
        let id = state(index)?;
        backend.reserve_state_for(id, 256, OutputReadout::Logits)?;
        tasks.push(prefill_task(id, buffer)?);
    }
    let first = run(backend, &tasks)?;
    let mut tokens = Vec::new();
    for logits in &first {
        tokens.push(greedy(logits)?);
    }
    let mut traces: Vec<Vec<Vec<f32>>> = first.into_iter().map(|logits| vec![logits]).collect();
    for step in 0..DECODE_STEPS {
        let mut decode = Vec::new();
        for (index, buffer) in buffers.iter().enumerate() {
            decode.push(ExecutionTask {
                plain_greedy: false,
                publish_prefix: false,
                request: RequestId::ONE,
                state: state(index)?,
                tokens: ExecutionInput::Decode {
                    token: tokens[index],
                    position: buffer.len() + step,
                },
                sampling: None,
                media: None,
            });
        }
        for (index, logits) in run(backend, &decode)?.into_iter().enumerate() {
            tokens[index] = greedy(&logits)?;
            traces[index].push(logits);
        }
    }
    for index in 0..PROMPTS {
        backend.release_state(state(index)?)?;
    }
    Ok(traces)
}

fn compare(reference: &[Vec<Vec<f32>>], actual: &[Vec<Vec<f32>>]) -> Result<f32> {
    let mut worst = 0.0_f32;
    for (lane, (expected, actual)) in reference.iter().zip(actual).enumerate() {
        for (step, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            assert_eq!(
                greedy(expected)?,
                greedy(actual)?,
                "lane {lane} step {step} changed its greedy token"
            );
            let scale = expected
                .iter()
                .fold(0.0_f32, |m, v| m.max(v.abs()))
                .max(1.0);
            let error = expected
                .iter()
                .zip(actual)
                .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
            worst = worst.max(error / scale);
        }
    }
    Ok(worst)
}

/// Prompts through one graph must leave the same logits and the same state as one at a time,
/// and must not disturb a resident sequence whose segment is idle during the replay.
#[test]
#[ignore = "real CUDA checkpoint; run under safe-run with INFER_TEST_MODEL"]
fn multi_sequence_prompts_match_one_at_a_time() -> Result<()> {
    let mut backend = fixture()?;
    let reference = trace(&mut backend, false)?;
    let burst = trace(&mut backend, true)?;
    let worst = compare(&reference, &burst)?;
    let together = trace_together(&mut backend)?;
    let worst_together = compare(&reference, &together)?;
    println!("worst relative logit difference with full-width decode: {worst_together}");
    assert!(
        worst_together <= LOGIT_TOLERANCE,
        "relative logit error {worst_together}"
    );
    println!("worst relative logit difference: {worst}");
    assert!(worst <= LOGIT_TOLERANCE, "relative logit error {worst}");

    // A resident sequence keeps its state while other segments replay.
    let quiet = state(PROMPTS)?;
    let buffer = TokenBuffer::from(prompt(99, 48));
    backend.reserve_state_for(quiet, 256, OutputReadout::Logits)?;
    let task = prefill_task(quiet, &buffer)?;
    let logits = run(&mut backend, from_ref(&task))?
        .pop()
        .ok_or_else(|| Error::invariant("quiet prompt"))?;
    let token = greedy(&logits)?;
    let decode = ExecutionTask {
        plain_greedy: false,
        publish_prefix: false,
        request: RequestId::ONE,
        state: quiet,
        tokens: ExecutionInput::Decode {
            token,
            position: buffer.len(),
        },
        sampling: None,
        media: None,
    };
    let _ = trace(&mut backend, true)?;
    let after = run(&mut backend, from_ref(&decode))?
        .pop()
        .ok_or_else(|| Error::invariant("quiet decode"))?;
    backend.release_state(quiet)?;

    let control = state(PROMPTS)?;
    backend.reserve_state_for(control, 256, OutputReadout::Logits)?;
    let task = prefill_task(control, &buffer)?;
    run(&mut backend, from_ref(&task))?;
    let decode = ExecutionTask {
        state: control,
        ..decode
    };
    let expected = run(&mut backend, from_ref(&decode))?
        .pop()
        .ok_or_else(|| Error::invariant("control decode"))?;
    assert_eq!(
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "an idle segment changed a resident sequence"
    );
    Ok(())
}

#[test]
fn rungs_cover_the_pool_and_padding_stays_bounded() {
    assert_eq!(prefill_burst::rungs(16), [4, 8, 16]);
    assert_eq!(prefill_burst::rungs(20), [4, 8, 16, 20]);
    assert_eq!(prefill_burst::rungs(4), [4]);
    assert_eq!(prefill_burst::rungs(2), [2]);
    assert!(burst_pays(2, 8));
    assert!(burst_pays(4, 16));
    assert!(!burst_pays(1, 4));
    assert!(!burst_pays(3, 16));
}
