//! Shared diagnostic helpers for the loading checkpoints.
//!
//! Each checkpoint compares private and pooled arithmetic the same way, so the census helpers live
//! here once. `pub(super)` matches the existing `quant_reuse_model_tests::bits` sharing convention.
use super::LoadedModel;
use super::quant_reuse_model_tests::bits;
use crate::resident::fp8_cache::KvState;
use crate::{
    device::CudaDevice,
    resident::{DeviceProgram, slot_batch::SlotPool},
};
use infer_core::{Error, Result, TensorId};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(super) fn agreement(left: &[f32], right: &[f32], field: &str) -> Value {
    assert_eq!(left.len(), right.len(), "{field}");
    assert!(!left.is_empty(), "{field}");
    assert!(left.iter().chain(right).all(|x| x.is_finite()), "{field}");
    let mut different = 0usize;
    let mut maximum = 0.0_f64;
    let mut first = None;
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        maximum = maximum.max((f64::from(left) - f64::from(right)).abs());
        if left.to_bits() != right.to_bits() {
            different += 1;
            first.get_or_insert_with(|| json!({"element":index,"left":left,"right":right,"left_bits":left.to_bits(),"right_bits":right.to_bits()}));
        }
    }
    json!({"field":field,"elements":left.len(),"different":different,"maximum_absolute":maximum,"first":first})
}

pub(super) fn differences(value: &Value) -> usize {
    usize::try_from(value["different"].as_u64().expect("diagnostic count"))
        .expect("count fits usize")
}

pub(super) fn state_census_slot(
    loaded: &LoadedModel,
    program: &DeviceProgram,
    pool: &SlotPool,
    slot: usize,
) -> Result<(usize, Vec<Value>)> {
    let (state, fp8) = pool.test_states(slot);
    assert_eq!(program.states().len(), state.len());
    let mut changed = 0;
    let mut details = Vec::new();
    for (id, tensors) in program.states() {
        let reference = &state[id];
        assert_eq!(tensors.len(), reference.len());
        let spec = loaded
            .graph
            .tensors
            .iter()
            .find(|s| s.id == *id)
            .ok_or_else(|| Error::invariant("state spec"))?;
        for (part, (left, right)) in tensors.iter().zip(reference).enumerate() {
            let count = agreement(
                &loaded.device().runtime().read(left.memory())?,
                &loaded.device().runtime().read(right.memory())?,
                &format!("{id:?} {:?} part={part}", spec.storage),
            );
            changed += differences(&count);
            if differences(&count) > 0 {
                details.push(count);
            }
        }
    }
    assert_eq!(program.fp8_states().len(), fp8.len());
    for (id, KvState { keys, values, .. }) in program.fp8_states() {
        let KvState {
            keys: rkeys,
            values: rvalues,
            ..
        } = &fp8[id];
        for (field, left, right) in [("keys", keys, rkeys), ("values", values, rvalues)] {
            let a = loaded.device().runtime().read(left)?;
            let b = loaded.device().runtime().read(right)?;
            assert_eq!(a.len(), b.len());
            assert!(a.iter().chain(&b).all(|x| x.0 & 127 != 127));
            let different = a.iter().zip(&b).filter(|(a, b)| a.0 != b.0).count();
            changed += different;
            if different > 0 {
                details.push(json!({"field":format!("{id:?} {field}"),"elements":a.len(),"different":different}));
            }
        }
    }
    Ok((changed, details))
}

pub(super) fn seed_prefix(
    loaded: &LoadedModel,
    aligned: &DeviceProgram,
    original: &mut DeviceProgram,
) -> Result<()> {
    // Seed every program with the same prefix before isolating decode arithmetic.
    // Copy into captured allocations without changing their addresses or F32/FP8 storage.
    for (id, source) in aligned.states() {
        let target = original
            .states_mut()
            .get_mut(id)
            .ok_or_else(|| Error::invariant("control state"))?;
        assert_eq!(target.len(), source.len());
        for (target, source) in target.iter().zip(source) {
            target.copy_from(loaded.device(), source)?;
        }
    }
    for (id, KvState { keys, values, .. }) in aligned.fp8_states() {
        let target = original
            .fp8_states()
            .get(id)
            .ok_or_else(|| Error::invariant("control FP8 cache"))?;
        for (target, source) in [(&target.keys, keys), (&target.values, values)] {
            assert_eq!(target.len(), source.len());
            loaded
                .device()
                .copy_d2d_at(target, source, 0, 0, source.len())?;
        }
    }
    loaded.device().drain()?;
    original.set_position(aligned.position());
    Ok(())
}

pub(super) type States = BTreeMap<TensorId, Vec<Shaped<f32>>>;
pub(super) use crate::resident::fp8_cache::Fp8Caches;
use crate::resident::shaped::Shaped;

pub(super) fn states(
    device: &CudaDevice,
    actual: &States,
    expected: &States,
    actual_fp8: &Fp8Caches,
    expected_fp8: &Fp8Caches,
) -> Result<()> {
    assert_eq!(actual.len(), expected.len());
    assert_ne!(actual.len(), 0, "batch output is empty");
    for (id, tensors) in actual {
        let reference = &expected[id];
        assert_eq!(tensors.len(), reference.len());
        for (a, b) in tensors.iter().zip(reference) {
            bits(
                &device.runtime().read(a.memory())?,
                &device.runtime().read(b.memory())?,
                "F32 state",
            );
        }
    }
    assert_eq!(actual_fp8.len(), expected_fp8.len());
    assert!(!actual_fp8.is_empty());
    for (id, KvState { keys, values, .. }) in actual_fp8 {
        let KvState {
            keys: ref_keys,
            values: ref_values,
            ..
        } = &expected_fp8[id];
        for (a, b) in [(keys, ref_keys), (values, ref_values)] {
            let a = device.runtime().read(a)?;
            let b = device.runtime().read(b)?;
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(&b).all(|(a, b)| a.0 == b.0), "FP8 KV");
        }
    }
    Ok(())
}
