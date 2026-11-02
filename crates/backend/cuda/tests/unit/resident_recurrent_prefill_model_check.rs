//! Same weights, width, convolution and GEMM policy; only Delta capture differs.
use crate::resident::DeviceProgram;
use crate::{
    device::CudaDevice,
    loading::{LoadOptions, LoadedModel},
};
use infer_core::{Error, ModelId, Result};
use std::env::var;
use std::fs::read;

/// Largest relative L2 error of a state: the candidate's chunk factors multiply in BF16 and the
/// scalar recurrence does not, which the numeric contract permits.
const STATE_TOLERANCE: f64 = 1e-1;

fn relative_error(actual: &[f32], reference: &[f32]) -> f64 {
    let squared = |values: &mut dyn Iterator<Item = f32>| {
        values.map(|v| f64::from(v) * f64::from(v)).sum::<f64>()
    };
    let error = squared(&mut actual.iter().zip(reference).map(|(a, b)| a - b));
    (error / squared(&mut reference.iter().copied()).max(f64::MIN_POSITIVE)).sqrt()
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and INFER_TEST_TOKENS (JSON token array)"]
fn checkpoint_ordered_recurrence_matches_scalar_bits() -> Result<()> {
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
    for (chunk, tokens) in tokens.chunks(width).enumerate() {
        let actual = candidate.prefill_batch(tokens, chunk * width, true)?;
        let expected = legacy.prefill_batch(tokens, chunk * width, true)?;
        assert_eq!(actual.len(), tokens.len());
        assert_eq!(actual.len(), expected.len());
        assert_ne!(actual.last().expect("prefill result").1.len(), 0);

        // The two captures must agree on state, not only on the readout: when they did not, the
        // divergence was already in the recurrence and not in whatever consumes it.
        compare_all_states(&loaded, &candidate, &legacy, "prefill");
        for (a, b) in actual.iter().zip(&expected) {
            compare(&a.0, &b.0, "hidden");
            compare(&a.1, &b.1, "logits");
        }
    }
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

    for (a, b) in actual.iter().zip(&expected) {
        compare(&a.0, &b.0, "verify hidden");
        compare(&a.1, &b.1, "verify logits");
    }
    compare_all_states(&loaded, &candidate, &legacy, "verify");
    candidate.commit_batch(1)?;
    legacy.commit_batch(1)?;
    compare_all_states(&loaded, &candidate, &legacy, "rollback");
    for step in 0..16 {
        let position = tokens.len() + 1 + step;
        let token = tokens[step % tokens.len()];
        let a = candidate.step(token, position, position, None, true)?;
        let b = legacy.step(token, position, position, None, true)?;
        assert!(!a.1.is_empty() && !b.1.is_empty());
        compare(&a.0, &b.0, "decode hidden");
        compare(&a.1, &b.1, "decode logits");
        compare_all_states(&loaded, &candidate, &legacy, "decode");
    }
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

fn compare(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    let differing = a
        .iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "{label} nonfinite"
    );
    eprintln!("{label}: bit_differing={differing}/{}", a.len());
    assert_eq!(differing, 0, "{label} bit drift");
}

fn compare_all_states(
    loaded: &LoadedModel,
    candidate: &DeviceProgram,
    scalar: &DeviceProgram,
    label: &str,
) {
    assert_eq!(candidate.states().len(), scalar.states().len());
    let mut elements = 0;
    for (id, tensors) in candidate.states() {
        let reference = scalar.states().get(id).expect("scalar state id");
        assert_eq!(tensors.len(), reference.len());
        for (a, b) in tensors.iter().zip(reference) {
            let a = loaded
                .device()
                .read_borrowed(a)
                .expect("candidate state download");
            let b = loaded
                .device()
                .read_borrowed(b)
                .expect("scalar state download");
            assert_eq!(a.len(), b.len());
            elements += a.len();
            assert!(
                a.iter().chain(&b).all(|v| v.is_finite()),
                "{label} state nonfinite"
            );
            assert!(
                relative_error(&a, &b) <= STATE_TOLERANCE,
                "{label} state {id:?} drift {}",
                relative_error(&a, &b)
            );
        }
    }
    eprintln!("MODEL_STATE_BITS label={label} checked={elements} differing=0");
    compare_fp8_states(loaded, candidate, scalar, label);
}

fn compare_fp8_states(
    loaded: &LoadedModel,
    candidate: &DeviceProgram,
    scalar: &DeviceProgram,
    label: &str,
) {
    assert_eq!(candidate.fp8_states().len(), scalar.fp8_states().len());
    let mut elements = 0;
    for (id, (keys, values)) in candidate.fp8_states() {
        let (rkeys, rvalues) = &scalar.fp8_states()[id];
        for (a, b) in [(keys, rkeys), (values, rvalues)] {
            let a = loaded
                .device()
                .read_borrowed(a)
                .expect("candidate FP8 state");
            let b = loaded.device().read_borrowed(b).expect("scalar FP8 state");
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter().chain(&b).all(|x| x.0 & 127 != 127),
                "FP8 state nonfinite"
            );
            assert!(
                a.iter().zip(&b).all(|(a, b)| a.0 == b.0),
                "{label} FP8 state {id:?} bit drift"
            );
            elements += a.len();
        }
    }
    eprintln!("MODEL_FP8_BITS label={label} checked={elements} differing=0");
}
