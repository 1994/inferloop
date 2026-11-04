use super::support::{agreement, differences, seed_prefix};
use super::{LoadOptions, LoadedModel};
use crate::{
    device::CudaDevice,
    resident::{DeviceProgram, slot_batch::SlotPool},
};
use infer_core::{Error, ModelId, Result};
use serde_json::json;
use std::env::var;
use std::fs::read;

fn greedy(logits: &[f32]) -> Result<u32> {
    assert_ne!(logits.len(), 0);
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    u32::try_from(best).map_err(|e| Error::invalid(e.to_string()))
}
struct Lane {
    teacher: DeviceProgram,
    selected: DeviceProgram,
    next: u32,
}
fn make_lanes(loaded: &LoadedModel, inputs: &[Vec<u32>]) -> Result<Vec<Lane>> {
    let mut lanes = Vec::new();
    for input in inputs {
        let mut teacher = loaded.sequence(128)?;
        let out = teacher.prefill_batch(input, 0, true)?;
        let next = greedy(
            &out.last()
                .ok_or_else(|| Error::invariant("prefix logits"))?
                .1,
        )?;
        let mut selected = loaded.sequence_pooled(128)?;
        seed_prefix(loaded, &teacher, &mut selected)?;
        lanes.push(Lane {
            teacher,
            selected,
            next,
        });
    }
    Ok(lanes)
}
fn check_states(loaded: &LoadedModel, lane: &Lane, pool: &SlotPool, slot: usize) -> Result<usize> {
    let (states, fp8) = pool.test_states(slot);
    let mut diff = 0;
    for (id, parts) in lane.teacher.states() {
        for (left, right) in parts.iter().zip(&states[id]) {
            let a = loaded.device().runtime().read(left.memory())?;
            let b = loaded.device().runtime().read(right.memory())?;
            diff += differences(&agreement(&a, &b, "whole F32 state"));
        }
    }
    let rows = lane.teacher.position();
    for (id, teacher) in lane.teacher.fp8_states() {
        let pooled = &fp8[id];
        for (left, right) in [
            (&teacher.keys, &pooled.keys),
            (&teacher.values, &pooled.values),
        ] {
            let a = loaded.device().runtime().read(left)?;
            let b = loaded.device().runtime().read(right)?;
            let heads = teacher.layout.heads;
            let src_cap = teacher.layout.rows;
            let dst_cap = pooled.layout.rows;
            let d = teacher.layout.dim;
            for head in 0..heads {
                let a = &a[head * src_cap * d..head * src_cap * d + rows * d];
                let b = &b[head * dst_cap * d..head * dst_cap * d + rows * d];
                diff += a.iter().zip(b).filter(|(x, y)| x.0 != y.0).count();
            }
        }
    }
    Ok(diff)
}
fn trajectory(
    loaded: &LoadedModel,
    lanes: &mut [Lane],
    pool: &mut SlotPool,
    join: usize,
) -> Result<()> {
    let mut diffs = 0;
    for step in 0..64 {
        if step == join {
            for (slot, lane) in lanes.iter().enumerate() {
                assert_eq!(pool.take_free(), Some(slot));
                pool.bind(
                    slot,
                    lane.selected.states(),
                    lane.selected.fp8_states(),
                    lane.selected.position(),
                    loaded.device(),
                )?;
                let different = check_states(loaded, lane, pool, slot)?;
                println!(
                    "{}",
                    json!({"greedy_pool_pre_replay":step,"slot":slot,"different":different})
                );
                assert_eq!(different, 0, "covered state copied before execution");
            }
        }
        let active: Vec<_> = if step >= 56 {
            vec![0, 3]
        } else {
            (0..4).collect()
        };
        let calls: Vec<_> = active
            .iter()
            .map(|&slot| {
                let lane = &lanes[slot];
                let pos = lane.teacher.position();
                (slot, lane.next, pos, pos)
            })
            .collect();
        let mut canonical = Vec::new();
        for &(slot, token, pos, _) in &calls {
            let full = lanes[slot].teacher.step(token, pos, pos, None, true)?;
            let expected = greedy(&full.1)?;
            let got = lanes[slot].selected.step_selected(token, pos, false)?;
            let got = got
                .tokens
                .ok_or_else(|| Error::invariant("private GPU token readout absent"))?[0]
                .ok_or_else(|| Error::invariant("private GPU token"))?;
            if got != expected {
                diffs += 1;
                println!(
                    "{}",
                    json!({"greedy_private_mismatch":step,"slot":slot,"expected":expected,"actual":got})
                );
            }
            canonical.push(expected);
        }
        if step >= join {
            let output = pool.run_lanes_readout(loaded.device(), &calls, true)?;
            let tokens = output
                .tokens
                .ok_or_else(|| Error::invariant("pooled GPU token readout absent"))?;
            for ((&(slot, ..), token), &expected) in calls.iter().zip(&tokens).zip(&canonical) {
                if *token != Some(expected) {
                    diffs += 1;
                    println!(
                        "{}",
                        json!({"greedy_pool_mismatch":step,"slot":slot,"expected":expected,"actual":token})
                    );
                }
                if [join, 20, 32, 63].contains(&step) {
                    let state = check_states(loaded, &lanes[slot], pool, slot)?;
                    diffs += state;
                    println!(
                        "{}",
                        json!({"greedy_pool_state":step,"slot":slot,"different":state})
                    );
                }
            }
        }
        for (&slot, &token) in active.iter().zip(&canonical) {
            lanes[slot].next = token;
        }
    }
    println!(
        "{}",
        json!({"greedy_lifecycle_summary":true,"join_step":join,"source_capacity":128,"pool_capacity":1024,"steps":64,"different":diffs})
    );
    assert_eq!(
        diffs, 0,
        "actual GPU greedy/readout and crosscapacity migration"
    );
    Ok(())
}
#[test]
#[ignore = "requires CUDA/checkpoint/failing52token inputs/safe-run; GPUselected versus full logits/crosscapacity join"]
fn original_private_pool_gpu_greedy_lifecycle() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let path = var("INFER_TEST_ROUTE_CASES").map_err(|e| Error::invalid(e.to_string()))?;
    let inputs: Vec<Vec<u32>> =
        serde_json::from_slice(&read(path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(inputs.len(), 4);
    assert!(inputs.iter().all(|t| t.len() == 52));
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            autotune: false,
            mtp_depth: 0,
            prefill_width: 64,
            ..LoadOptions::default()
        },
    )?;
    let mut lanes = make_lanes(&loaded, &inputs)?;
    loaded.weights.prefill_width = 0;
    loaded.weights.narrow_prefill_width = 0;
    let mut pool = loaded.slot_pool(4, 1024, None)?;
    trajectory(&loaded, &mut lanes, &mut pool, 8)?;
    Ok(())
}
