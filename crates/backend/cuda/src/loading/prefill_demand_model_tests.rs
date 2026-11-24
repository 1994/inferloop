//! Exact state-effect closure experiment; both programs own the same arenas and graphs.
use super::quant_reuse_model_tests::{greedy, outputs, program_states, prompt};
use super::{LoadOptions, LoadedModel};
use crate::resident::DeviceProgram;
use crate::tuning::table_path;
use crate::{
    device::CudaDevice,
    resident::prefill_demand::{StateDemand, with_state_prefill},
};
use cutile::cutile_compiler::hints::CompileOptions;
use infer_core::{Error, ModelId, Result};
use infer_ir::TensorOp;
use serde_json::json;
use std::env::var;
use std::fs::read;
use std::time::Instant;

fn report(loaded: &LoadedModel) -> Result<()> {
    let plan = StateDemand::new(&loaded.graph);
    let mut macs = 0usize;
    let mut skipped = Vec::new();
    for (index, node) in loaded.graph.nodes.iter().enumerate() {
        if plan.retains(index)
            || node.outputs.contains(
                &loaded
                    .graph
                    .logits
                    .ok_or_else(|| Error::invariant("logits"))?,
            )
        {
            continue;
        }
        skipped.push(index);
        if node.op == TensorOp::Linear {
            let elements = |id| {
                loaded
                    .graph
                    .tensors
                    .iter()
                    .find(|spec| spec.id == id)
                    .ok_or_else(|| Error::invariant("projection spec"))?
                    .elements()
            };
            macs += elements(node.inputs[0])? * elements(node.outputs[0])?;
        }
    }
    assert!(macs > 0);
    println!(
        "{}",
        json!({"state_prefill_plan":true,"skipped_nodes":skipped,"projection_MAC_saved_per_row_beyond_existing_lm_head_skip":macs,"all_state_effects_retained":true,"arena_and_arithmetic_unchanged":true})
    );
    Ok(())
}
#[test]
#[ignore = "requires CUDA, frozen INFER_TEST_MODEL/TOKENS/table; safe-run; complete F32/FP8 state and vocabulary comparisons"]
fn checkpoint_state_prefill_demand_preserves_complete_bits() -> Result<()> {
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = prompt()?;
    let cache_path = table_path().ok_or_else(|| Error::invalid("frozen tuning table required"))?;
    let cache = read(&cache_path).map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 256,
            verification_width: 3,
            mtp_depth: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    report(&loaded)?;
    for width in [64, 256] {
        loaded.weights.prefill_width = width;
        loaded.weights.narrow_prefill_width = 0;
        // Both retain the extra graph, so admission, allocation and graph ownership match.
        let mut actual = with_state_prefill(true, || loaded.sequence(1024))?;
        let mut expected = with_state_prefill(true, || loaded.sequence(1024))?;
        let mut state_chunks = 0;
        for length in [51, 64, 255, 256, 511] {
            let chunks: Vec<_> = tokens[..length].chunks(width).collect();
            for (chunk, input) in chunks.iter().enumerate() {
                let last = chunk + 1 == chunks.len();
                let a = with_state_prefill(true, || {
                    actual.prefill_batch_readout(input, chunk * width, last, false)
                })?;
                let b = with_state_prefill(false, || {
                    expected.prefill_batch_readout(input, chunk * width, last, false)
                })?;
                outputs(&a, &b);
                program_states(&loaded, &actual, &expected)?;
                state_chunks += usize::from(!last);
            }
            let mut token = tokens[length - 1];
            for _ in 0..8 {
                let position = actual.position();
                let a = actual.step(token, position, position, None, true)?;
                let b = expected.step(token, position, position, None, true)?;
                let next = greedy(&a.1)?;
                assert_eq!(next, greedy(&b.1)?);
                outputs(&vec![a], &vec![b]);
                token = next;
            }
            program_states(&loaded, &actual, &expected)?;
        }
        // Hidden readout selects the full graph even when the internal experiment is enabled.
        outputs(
            &with_state_prefill(true, || {
                actual.prefill_batch_readout(&tokens[..51], 0, false, true)
            })?,
            &with_state_prefill(false, || {
                expected.prefill_batch_readout(&tokens[..51], 0, false, true)
            })?,
        );
        program_states(&loaded, &actual, &expected)?;
        println!(
            "{}",
            json!({"state_prefill_complete_bitwise":true,"width":width,"state_only_chunks":state_chunks,"F32_states_complete":true,"FP8_KV_complete":true,"full_vocabulary_and_hidden":true,"greedy_steps_per_length":8,"hidden_readout_fallback":true,"MTP":0,"chunked_recurrent":false})
        );
    }
    assert_eq!(
        cache,
        read(cache_path).map_err(|e| Error::invalid(e.to_string()))?
    );
    Ok(())
}

fn timed_prompt(
    program: &mut DeviceProgram,
    tokens: &[u32],
    width: usize,
    enabled: bool,
) -> Result<(f64, Vec<f32>)> {
    let start = Instant::now();
    let logits = with_state_prefill(enabled, || {
        let mut logits = Vec::new();
        for (chunk, input) in tokens.chunks(width).enumerate() {
            let last = (chunk + 1) * width >= tokens.len();
            let mut rows = program.prefill_batch_readout(input, chunk * width, last, false)?;
            if last {
                logits = rows
                    .pop()
                    .ok_or_else(|| Error::invariant("prompt output"))?
                    .1;
            }
        }
        Ok::<_, Error>(logits)
    })?;
    Ok((start.elapsed().as_secs_f64() * 1000.0, logits))
}

#[test]
#[ignore = "requires release CUDA O3, completed CI and complete bitwise gate; safe-run; internal paired prefill cost only"]
fn checkpoint_state_prefill_demand_paired_cost() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::invalid("cost requires release"));
    }
    let compile = CompileOptions::default();
    if compile.device_debug || compile.opt_level.is_some_and(|level| level != 3) {
        return Err(Error::invalid("cost requires CUDA O3"));
    }
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = prompt()?;
    let cache_path = table_path().ok_or_else(|| Error::invalid("frozen table required"))?;
    let cache = read(&cache_path).map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 256,
            verification_width: 3,
            mtp_depth: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    for width in [64, 256] {
        loaded.weights.prefill_width = width;
        loaded.weights.narrow_prefill_width = 0;
        let mut actual = with_state_prefill(true, || loaded.sequence(1024))?;
        let mut expected = with_state_prefill(true, || loaded.sequence(1024))?;
        for length in [51, 511] {
            let input = &tokens[..length];
            for _ in 0..3 {
                let a = timed_prompt(&mut actual, input, width, true)?;
                let b = timed_prompt(&mut expected, input, width, false)?;
                super::quant_reuse_model_tests::bits(&a.1, &b.1, "warmup full vocabulary");
            }
            for round in 0..3 {
                let mut baseline = Vec::new();
                let mut candidate = Vec::new();
                let mut orders = Vec::new();
                for rep in 0..16 {
                    let reverse = (rep + round) % 2 == 1;
                    let (a, b) = if reverse {
                        let a = timed_prompt(&mut actual, input, width, true)?;
                        let b = timed_prompt(&mut expected, input, width, false)?;
                        (a, b)
                    } else {
                        let b = timed_prompt(&mut expected, input, width, false)?;
                        let a = timed_prompt(&mut actual, input, width, true)?;
                        (a, b)
                    };
                    // Compare every vocabulary entry outside the measured intervals.
                    super::quant_reuse_model_tests::bits(&a.1, &b.1, "timed full vocabulary");
                    candidate.push(a.0);
                    baseline.push(b.0);
                    orders.push(if reverse { "BA" } else { "AB" });
                }
                program_states(&loaded, &actual, &expected)?;
                println!(
                    "{}",
                    json!({
                        "state_prefill_pair_cost":true,"width":width,"tokens":length,"round":round,
                        "baseline_ms":baseline,"candidate_ms":candidate,"orders":orders,
                        "same_arenas_and_graph_ownership":true,"complete_vocabulary_bitwise":true,
                        "scope":"single-sequence prompt: reset, metadata, all chunks, final full logits host readback",
                        "service_performance_claim":false,"chunked_recurrent":false,"MTP":0,
                    })
                );
            }
        }
    }
    assert_eq!(
        cache,
        read(cache_path).map_err(|e| Error::invalid(e.to_string()))?
    );
    Ok(())
}
