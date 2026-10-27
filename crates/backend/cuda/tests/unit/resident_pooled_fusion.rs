//! The actual draft pool versus independent original per-slot FC captures.
use super::{captures, with_row_reference};
use crate::{
    device::CudaDevice,
    loading::{LoadOptions, LoadedModel},
    resident::slot_batch::SlotPool,
};
use infer_core::{Error, ModelId, Result};
use std::env::var;

#[test]
#[ignore = "requires CUDA and INFER_TEST_MODEL; run release inside safe-run"]
fn pooled_fusion_preserves_checkpoint_model_bits() -> Result<()> {
    let loaded = checkpoint_fixture()?;
    for width in [2, 5, 16] {
        compare_pool(&loaded, width)?;
    }
    Ok(())
}

fn checkpoint_fixture() -> Result<LoadedModel> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|error| Error::invalid(error.to_string()))?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            mtp_depth: 2,
            autotune: false,
            chunked_recurrent: false,
            ..LoadOptions::default()
        },
    )?;
    assert_eq!(loaded.model().hidden_size, 5120);
    Ok(loaded)
}
#[test]
#[ignore = "CUDA/checkpoint/safe-run; extended draft pool complete output/state bits and reuse"]
fn extended_concurrency_preserves_checkpoint_draft_states() -> Result<()> {
    let loaded = checkpoint_fixture()?;
    let bound = loaded
        .concurrency_roofline()?
        .ok_or_else(|| Error::invariant("checkpoint roofline absent"))?;
    assert!(bound > 0);
    println!(
        "{}",
        serde_json::json!({"checkpoint_roofline_bound":bound,"extended_slots":[17,32,64]})
    );
    for width in [17, 32, 64] {
        compare_pool(&loaded, width)?;
    }
    Ok(())
}

fn compare_pool(loaded: &LoadedModel, width: usize) -> Result<()> {
    let before = captures();
    let mut candidate = loaded
        .draft_slot_pool(width, 64)?
        .ok_or_else(|| Error::invariant("checkpoint draft pool missing"))?;
    assert_eq!(
        captures() - before,
        if super::pooled(width) { 2 } else { 0 },
        "qualified fusion or original row captures must match the actual pool geometry"
    );
    let before = captures();
    let mut reference = with_row_reference(|| loaded.draft_slot_pool(width, 64))?
        .ok_or_else(|| Error::invariant("checkpoint reference pool missing"))?;
    assert_eq!(
        captures(),
        before,
        "original row fallback must remain available"
    );
    for round in 0..4 {
        let mut active: Vec<_> = (0..width)
            .filter(|slot| round != 1 || slot % 2 == 0)
            .collect();
        if round == 2 {
            active.reverse();
        }
        if round == 3 {
            for slot in [0, width - 1] {
                let position = candidate.cursors()[slot] - 1;
                candidate.rewind_external(slot, position)?;
                reference.rewind_external(slot, position)?;
            }
        }
        let external: Vec<_> = active
            .iter()
            .map(|&slot| {
                (0..5120)
                    .map(|index| {
                        (f32::from(u16::try_from((index + slot * 7 + round * 11) % 43).unwrap())
                            - 21.0)
                            / 37.0
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                let position = candidate.cursors()[slot];
                (
                    slot,
                    u32::try_from(100 + slot + round).unwrap(),
                    position + 1,
                    position,
                )
            })
            .collect();
        let external: Vec<_> = external.iter().map(Vec::as_slice).collect();
        let read_logits = round != 2;
        let actual =
            candidate.run_external_readout(loaded.device(), &lanes, &external, read_logits)?;
        let expected =
            reference.run_external_readout(loaded.device(), &lanes, &external, read_logits)?;
        assert_eq!(actual.len(), expected.len());
        for ((a_hidden, a_logits), (b_hidden, b_logits)) in actual.iter().zip(&expected) {
            bits(a_hidden, b_hidden);
            bits(a_logits, b_logits);
        }
        assert_eq!(candidate.cursors(), reference.cursors());
        compare_states(loaded.device(), &candidate, &reference, width)?;
    }
    Ok(())
}

fn compare_states(device: &CudaDevice, a: &SlotPool, b: &SlotPool, width: usize) -> Result<()> {
    for slot in 0..width {
        bits(&a.test_hidden(device, slot)?, &b.test_hidden(device, slot)?);
        let (a, a_packed) = a.test_states(slot);
        let (b, b_packed) = b.test_states(slot);
        assert_eq!(
            a_packed.keys().collect::<Vec<_>>(),
            b_packed.keys().collect::<Vec<_>>(),
        );
        for (id, (keys, values)) in a_packed {
            let (expected_keys, expected_values) = &b_packed[id];
            for (actual, expected) in [(keys, expected_keys), (values, expected_values)] {
                let actual = device.read_borrowed(actual)?;
                let expected = device.read_borrowed(expected)?;
                assert_eq!(actual.len(), expected.len());
                for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(a.0, b.0, "slot={slot} state={id:?} byte={index}");
                }
            }
        }
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
        for (id, actual) in a {
            let expected = &b[id];
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                bits(&device.read_borrowed(a)?, &device.read_borrowed(b)?);
            }
        }
    }
    Ok(())
}

fn bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element={index}: {a} != {b}");
    }
}
