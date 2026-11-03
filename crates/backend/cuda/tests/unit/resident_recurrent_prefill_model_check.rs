//! Same weights, width, convolution and GEMM policy; only Delta capture differs. The default
//! lowering is the BF16 chunked rule, so the comparison is on predictions.
use crate::resident::DeviceProgram;
use crate::{
    device::CudaDevice,
    loading::{LoadOptions, LoadedModel},
};
use infer_core::{Error, ModelId, Result};
use std::env::var;
use std::fs::read;

/// Largest KL divergence accepted at a prompt end: the chunked prompt rule multiplies in BF16 and
/// the scalar recurrence does not, so the numeric contract judges the predictions, not the bits.
const MAXIMUM_PROMPT_KL: f64 = 0.25;

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and INFER_TEST_TOKENS (JSON token array)"]
fn checkpoint_ordered_recurrence_agrees_with_scalar_on_predictions() -> Result<()> {
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = checkpoint_tokens()?;
    let width = var("INFER_TEST_PREFILL_WIDTH")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|e| Error::invalid(e.to_string()))
        })
        .transpose()?
        .unwrap_or(64);
    if !super::ORDERED_WIDTHS.contains(&width) {
        return Err(Error::invalid("checkpoint test prefill width"));
    }
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: width,
            verification_width: 3,
            autotune: false,
            // Exercise the default exact lowering; no diagnostic algorithm switch.
            chunked_recurrent: false,
            ..LoadOptions::default()
        },
    )?;
    let capacity = (tokens.len() + 32).next_power_of_two();
    let before = super::ORDERED_CAPTURE_COUNT.get();
    let mut candidate = loaded.sequence(capacity)?;
    let ordered = super::ORDERED_CAPTURE_COUNT.get() - before;
    let delta_nodes = loaded
        .graph()
        .nodes
        .iter()
        .filter(|node| matches!(node.op, infer_ir::TensorOp::Delta { .. }))
        .count();
    assert!(
        delta_nodes > 0 && ordered >= delta_nodes,
        "default ordered lowering did not capture every recurrent layer"
    );
    eprintln!(
        "DEFAULT_ORDERED_CAPTURE count={ordered} layers={delta_nodes} diagnostic_option=false"
    );
    let before = super::ORDERED_CAPTURE_COUNT.get();
    super::LEGACY_DELTA_CAPTURE.set(true);
    let legacy = loaded.sequence(capacity);
    super::LEGACY_DELTA_CAPTURE.set(false);
    let mut legacy = legacy?;
    assert_eq!(
        super::ORDERED_CAPTURE_COUNT.get(),
        before,
        "scalar control captured ordered lowering"
    );
    let prompt = &tokens;
    for (chunk, tokens) in prompt.chunks(width).enumerate() {
        let actual = candidate.prefill_batch(tokens, chunk * width, true)?;
        let expected = legacy.prefill_batch(tokens, chunk * width, true)?;
        assert_eq!(actual.len(), tokens.len());
        assert_eq!(actual.len(), expected.len());
        assert_ne!(actual.last().expect("prefill result").1.len(), 0);

        finite_states(&loaded, &candidate, "prefill");
        finite_states(&loaded, &legacy, "scalar prefill");
        for (a, b) in actual.iter().zip(&expected) {
            finite(&a.0, "hidden");
            finite(&b.0, "scalar hidden");
        }
        let (a, b) = (
            &actual.last().expect("prefill").1,
            &expected.last().expect("prefill").1,
        );
        let kl = logits_kl(a, b);
        eprintln!("PROMPT_CHUNK chunk={chunk} kl={kl:.4e}");
        // Only the real prompt end decides a next token; a chunk boundary is an arbitrary cut.
        if (chunk + 1) * width >= prompt.len() {
            assert!(kl <= MAXIMUM_PROMPT_KL, "prompt end KL {kl}");
            assert_eq!(top(a), top(b), "prompt end top-1");
        }
    }
    continue_past_prompt(&loaded, &mut candidate, &mut legacy, prompt)
}

fn continue_past_prompt(
    loaded: &LoadedModel,
    candidate: &mut DeviceProgram,
    legacy: &mut DeviceProgram,
    tokens: &[u32],
) -> Result<()> {
    // Also exercise captured verification, including rollback and continued decode.
    let position = tokens.len();
    let verify: Vec<_> = tokens
        .iter()
        .copied()
        .cycle()
        .take(candidate.batch_width())
        .collect();
    let actual = candidate.step_batch(&verify, position)?;
    let expected = legacy.step_batch(&verify, position)?;
    assert_eq!(actual.len(), verify.len());
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|row| !row.1.is_empty()));

    // The tokens below continue the prompt arbitrarily, so past the prompt end only finiteness and
    // the ability to continue are asserted; the divergence is reported.
    for (a, b) in actual.iter().zip(&expected) {
        finite(&a.1, "verify logits");
        eprintln!("VERIFY kl={:.4e}", logits_kl(&a.1, &b.1));
    }
    finite_states(loaded, candidate, "verify");
    candidate.commit_batch(1)?;
    legacy.commit_batch(1)?;
    finite_states(loaded, candidate, "rollback");
    for step in 0..16 {
        let position = tokens.len() + 1 + step;
        let token = tokens[step % tokens.len()];
        let a = candidate.step(token, position, position, None, true)?;
        let b = legacy.step(token, position, position, None, true)?;
        assert!(!a.1.is_empty() && !b.1.is_empty());
        finite(&a.1, "decode logits");
        eprintln!("DECODE step={step} kl={:.4e}", logits_kl(&a.1, &b.1));
    }
    finite_states(loaded, candidate, "decode");
    Ok(())
}

fn checkpoint_tokens() -> Result<Vec<u32>> {
    let tokens_path = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(tokens_path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    if tokens.is_empty() {
        return Err(Error::invalid("empty checkpoint test prompt"));
    }
    Ok(tokens)
}

fn finite(values: &[f32], label: &str) {
    assert!(values.iter().all(|v| v.is_finite()), "{label} nonfinite");
}

fn top(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
}

/// `KL(actual || reference)` of the distributions two logits rows define.
fn logits_kl(actual: &[f32], reference: &[f32]) -> f64 {
    let probabilities = |logits: &[f32]| {
        let peak = logits.iter().copied().fold(f32::MIN, f32::max);
        let raw: Vec<f64> = logits.iter().map(|v| f64::from(v - peak).exp()).collect();
        let total: f64 = raw.iter().sum();
        raw.into_iter().map(|v| v / total).collect::<Vec<_>>()
    };
    assert_eq!(actual.len(), reference.len());
    probabilities(actual)
        .iter()
        .zip(probabilities(reference))
        .filter(|(p, _)| **p > 0.0)
        .map(|(p, q)| p * (p / q.max(f64::MIN_POSITIVE)).ln())
        .sum()
}

fn finite_states(loaded: &LoadedModel, program: &DeviceProgram, label: &str) {
    for tensors in program.states().values() {
        for tensor in tensors {
            let values = loaded
                .device()
                .read_borrowed(tensor)
                .expect("state download");
            finite(&values, label);
        }
    }
    for (keys, values) in program.fp8_states().values() {
        for tensor in [keys, values] {
            let values = loaded
                .device()
                .read_borrowed(tensor)
                .expect("FP8 state download");
            assert!(
                values.iter().all(|x| x.0 & 127 != 127),
                "{label} FP8 nonfinite"
            );
        }
    }
}
