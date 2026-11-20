use super::quant_reuse_model_tests::{outputs, program_states, prompt};
use super::support::states;
use super::{LoadOptions, LoadedModel};
use crate::{
    device::CudaDevice,
    resident::{
        DeviceProgram,
        fp8_quant_reduction::{captures, with_parallel_conversion},
    },
};
use infer_core::{Error, ModelId, Result};
use std::env::var;
fn pooled(
    loaded: &LoadedModel,
    actual: &DeviceProgram,
    expected: &DeviceProgram,
    tokens: &[u32],
    slots: usize,
) -> Result<()> {
    let before = captures();
    let mut separate = with_parallel_conversion(false, || loaded.slot_pool(slots, 128, None))?;
    assert_eq!(captures(), before);
    let mut shared = with_parallel_conversion(true, || loaded.slot_pool(slots, 128, None))?;
    assert!(captures() > before);
    for slot in 0..slots {
        assert_eq!(shared.take_free(), Some(slot));
        assert_eq!(separate.take_free(), Some(slot));
        shared.bind(
            slot,
            actual.states(),
            actual.fp8_states(),
            actual.position(),
            loaded.device(),
        )?;
        separate.bind(
            slot,
            expected.states(),
            expected.fp8_states(),
            expected.position(),
            loaded.device(),
        )?;
    }
    let all: Vec<_> = (0..slots).collect();
    for active in [all.clone(), vec![0], vec![slots - 1], all] {
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                let position = shared.cursors()[slot];
                (
                    slot,
                    tokens[(position + slot) % tokens.len()],
                    position,
                    position,
                )
            })
            .collect();
        assert_eq!(shared.cursors(), separate.cursors());
        let a = shared.run_lanes(loaded.device(), &lanes)?;
        let b = separate.run_lanes(loaded.device(), &lanes)?;
        // Read active logits, then compare every slot state including inactive lanes.
        outputs(&a, &b);
        for slot in 0..slots {
            let (a, af) = shared.test_states(slot);
            let (b, bf) = separate.test_states(slot);
            states(loaded.device(), a, b, af, bf)?;
        }
    }
    shared.release(slots - 1);
    separate.release(slots - 1);
    assert_eq!(shared.take_free(), Some(slots - 1));
    assert_eq!(separate.take_free(), Some(slots - 1));
    shared.bind(
        slots - 1,
        actual.states(),
        actual.fp8_states(),
        actual.position(),
        loaded.device(),
    )?;
    separate.bind(
        slots - 1,
        expected.states(),
        expected.fp8_states(),
        expected.position(),
        loaded.device(),
    )?;
    let position = actual.position();
    outputs(
        &shared.run_lanes(
            loaded.device(),
            &[(slots - 1, tokens[0], position, position)],
        )?,
        &separate.run_lanes(
            loaded.device(),
            &[(slots - 1, tokens[0], position, position)],
        )?,
    );
    for slot in 0..slots {
        let (a, af) = shared.test_states(slot);
        let (b, bf) = separate.test_states(slot);
        states(loaded.device(), a, b, af, bf)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA/checkpoint/safe-run; actual pool inactive/reactivate/rebind full states"]
fn full_model_pool_parallel_quantization() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = prompt()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            autotune: false,
            mtp_depth: 0,
            ..LoadOptions::default()
        },
    )?;
    let mut old = with_parallel_conversion(false, || loaded.sequence(128))?;
    let mut new = with_parallel_conversion(true, || loaded.sequence(128))?;
    outputs(
        &old.prefill_batch(&tokens[..51], 0, true)?,
        &new.prefill_batch(&tokens[..51], 0, true)?,
    );
    program_states(&loaded, &old, &new)?;
    for slots in [2, 4] {
        pooled(&loaded, &new, &old, &tokens, slots)?;
        println!(
            "{}",
            serde_json::json!({"parallel_quant_pool_slots":slots,
            "inactive_reactivate_rebind":true,"full_F32_FP8_states_exact":true,"different":0})
        );
    }
    Ok(())
}
