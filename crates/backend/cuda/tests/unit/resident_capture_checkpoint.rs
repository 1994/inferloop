//! Same-weight checkpoint model gate for serial state-writer snapshot fusion.
use super::LEGACY_FLAT_CAPTURE;
use crate::resident::fp8_cache::KvState;
use crate::{
    device::CudaDevice,
    loading::{LoadOptions, LoadedModel},
    resident::{BatchOutput, DeviceProgram},
};
use infer_core::{Error, ModelId, Result};
use std::env::var;
use std::fs::read;

fn assert_bits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label} element={index}"
        );
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} element={index}"
        );
    }
}

fn compare_outputs(actual: &BatchOutput, expected: &BatchOutput) {
    assert_eq!(actual.len(), expected.len());
    assert_ne!(actual.len(), 0);
    for (actual, expected) in actual.iter().zip(expected) {
        assert_bits(&actual.0, &expected.0, "hidden");
        assert_bits(&actual.1, &expected.1, "logits");
    }
}

fn compare_states(
    loaded: &LoadedModel,
    candidate: &DeviceProgram,
    legacy: &DeviceProgram,
) -> Result<()> {
    assert_eq!(candidate.states().len(), legacy.states().len());
    assert!(!candidate.states().is_empty());
    for (id, actual) in candidate.states() {
        let expected = &legacy.states()[id];
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_bits(
                &loaded.device().read_borrowed(actual)?,
                &loaded.device().read_borrowed(expected)?,
                "state",
            );
        }
    }
    assert_eq!(candidate.fp8_states().len(), legacy.fp8_states().len());
    for (
        id,
        KvState {
            keys: actual_keys,
            values: actual_values,
            ..
        },
    ) in candidate.fp8_states()
    {
        let KvState {
            keys: expected_keys,
            values: expected_values,
            ..
        } = &legacy.fp8_states()[id];
        for (actual, expected) in [
            (actual_keys, expected_keys),
            (actual_values, expected_values),
        ] {
            let actual = loaded.device().runtime().read(actual)?;
            let expected = loaded.device().runtime().read(expected)?;
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual.iter().zip(expected).all(|(a, b)| a.0 == b.0),
                "FP8 KV codes"
            );
        }
    }
    assert_eq!(candidate.position(), legacy.position());
    Ok(())
}

fn checkpoint_input() -> Result<(String, Vec<u32>)> {
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens_path = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(tokens_path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    if tokens.is_empty() {
        return Err(Error::invalid("empty checkpoint test prompt"));
    }
    Ok((path, tokens))
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and INFER_TEST_TOKENS; run under safe-run"]
fn serial_snapshot_fusion_preserves_checkpoint_model_bits() -> Result<()> {
    let (path, tokens) = checkpoint_input()?;
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            autotune: false,
            chunked_recurrent: false,
            ..LoadOptions::default()
        },
    )?;
    let capacity = (tokens.len() + 64).next_power_of_two();
    let mut candidate = loaded.sequence(capacity)?;
    LEGACY_FLAT_CAPTURE.set(true);
    let legacy = loaded.sequence(capacity);
    LEGACY_FLAT_CAPTURE.set(false);
    let mut legacy = legacy?;
    for (chunk, input) in tokens.chunks(64).enumerate() {
        compare_outputs(
            &candidate.prefill_batch(input, chunk * 64, true)?,
            &legacy.prefill_batch(input, chunk * 64, true)?,
        );
    }
    compare_states(&loaded, &candidate, &legacy)?;
    for (count, commit) in [(3, 1), (2, 2), (1, 1), (3, 2), (3, 3)] {
        let position = candidate.position();
        let input: Vec<_> = tokens.iter().copied().cycle().take(count).collect();
        compare_outputs(
            &candidate.step_batch(&input, position)?,
            &legacy.step_batch(&input, position)?,
        );
        compare_states(&loaded, &candidate, &legacy)?;
        candidate.commit_batch(commit)?;
        legacy.commit_batch(commit)?;
        compare_states(&loaded, &candidate, &legacy)?;
        let position = candidate.position();
        let token = tokens[position % tokens.len()];
        let actual = candidate.step(token, position, position, None, true)?;
        let expected = legacy.step(token, position, position, None, true)?;
        compare_outputs(&vec![actual], &vec![expected]);
        compare_states(&loaded, &candidate, &legacy)?;
        eprintln!(
            "serial_snapshot_gate: lanes={count} commit={commit} position={} bitwise_equal=true",
            candidate.position()
        );
    }
    Ok(())
}
