//! Complete model output/state comparison plus qualification reuse across serving states.
use super::{
    LoadOptions, LoadedModel,
    quant_reuse_model_tests::{outputs, program_states},
};
use crate::tuning::resident_gemm::ResidentTuningSession;
use crate::{
    AttentionProfile,
    device::{CudaDevice, vendor_fp8::nvfp4},
    mlp::ProjectionWeight,
    resident::DeviceProgram,
};
use infer_core::{Error, ModelId, Result};
use std::{env::var, fs::read, sync::Arc};

#[test]
#[ignore = "requires CUDA, 27B checkpoint and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn vendor_fp4_preserves_checkpoint_model_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Bf16, 1024, 64)
}
#[test]
#[ignore = "requires CUDA, 27B checkpoint and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn vendor_fp4_keeps_canonical_checkpoint_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Canonical, 2048, 64)
}
#[test]
#[ignore = "requires CUDA, 27B checkpoint and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn vendor_fp4_preserves_wide_bf16_checkpoint_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Bf16, 2048, 64)
}
fn loaded_model(
    profile: AttentionProfile,
    width: usize,
    narrow: usize,
) -> Result<(LoadedModel, Vec<u32>)> {
    var("INFER_CUDA_TUNING_TABLE").map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let token_path = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(token_path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(tokens.len(), 1025);
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: width,
            verification_width: 0,
            mtp_depth: 0,
            autotune: true,
            chunked_recurrent: true,
            attention_profile: profile,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    loaded.weights.narrow_prefill_width = narrow;
    Ok((loaded, tokens))
}
fn checkpoint_bits(profile: AttentionProfile, width: usize, narrow: usize) -> Result<()> {
    let (loaded, tokens) = loaded_model(profile, width, narrow)?;
    let before = nvfp4::captures();
    let mut candidate = loaded.sequence(2048)?;
    let captured = nvfp4::captures();
    if profile.is_bf16() {
        assert!(captured > before, "vendor path must actually be captured");
    } else {
        assert_eq!(captured, before, "canonical retains native FP4");
    }
    let mut reference = nvfp4::with_native(|| loaded.sequence(2048))?;
    assert_eq!(
        nvfp4::captures(),
        captured,
        "reference must retain native FP4"
    );
    compare_programs(
        &loaded,
        &mut candidate,
        &mut reference,
        &tokens,
        width,
        &[1025, 65, 1],
    )?;
    println!(
        "complete vendor/native checkpoint bits passed; captures={}",
        captured - before
    );
    Ok(())
}
fn compare_programs(
    loaded: &LoadedModel,
    a: &mut DeviceProgram,
    b: &mut DeviceProgram,
    tokens: &[u32],
    width: usize,
    lengths: &[usize],
) -> Result<()> {
    for &length in lengths {
        a.reset()?;
        b.reset()?;
        for (chunk, input) in tokens[..length].chunks(width).enumerate() {
            outputs(
                &a.prefill_batch(input, chunk * width, true)?,
                &b.prefill_batch(input, chunk * width, true)?,
            );
        }
        program_states(loaded, a, b)?;
        for step in 0..4 {
            let pos = a.position();
            let token = tokens[(length + step) % tokens.len()];
            outputs(
                &vec![a.step(token, pos, pos, None, true)?],
                &vec![b.step(token, pos, pos, None, true)?],
            );
        }
        program_states(loaded, a, b)?;
        println!("full hidden/logit/state bits passed: prompt={length}");
    }
    Ok(())
}
#[test]
#[ignore = "requires CUDA, 27B checkpoint and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn vendor_fp4_reuses_model_session_qualifications() -> Result<()> {
    let (mut loaded, tokens) = loaded_model(AttentionProfile::Bf16, 1024, 64)?;
    let before = nvfp4::confirmations();
    let mut first = loaded.sequence(2048)?;
    let confirmed = nvfp4::confirmations();
    assert!(
        confirmed - before <= 1,
        "at most one fresh performance confirmation per geometry"
    );
    let captured = nvfp4::captures();
    loaded.weights.autotune = false;
    loaded.weights.resident_tuning = Arc::new(ResidentTuningSession::default());
    let mut second = loaded.sequence(2048)?;
    assert_eq!(
        nvfp4::confirmations(),
        confirmed,
        "same model must reuse its qualifications"
    );
    assert!(
        nvfp4::captures() > captured,
        "cached state must still capture and execute vendor FP4"
    );
    compare_programs(
        &loaded,
        &mut first,
        &mut second,
        &tokens,
        1024,
        &[1025, 65, 1],
    )?;
    println!(
        "model-session qualifications reused; confirmations={}",
        confirmed - before
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA, actual checkpoint and frozen prompt; safe-run"]
fn fused_prompt_excludes_standalone_gate_up_tuning() -> Result<()> {
    use crate::resident::nvfp4_gemm::Workspace;
    let (mut loaded, _) = loaded_model(AttentionProfile::Bf16, 1024, 64)?;
    loaded.weights.autotune = false;
    let workspace = Workspace::new(
        loaded.device(),
        &loaded.graph,
        &loaded.weights,
        &[1, 3, 64, 1024],
    )?;
    let strategies = workspace.prepared_configs();
    let rows = strategies
        .as_array()
        .ok_or_else(|| Error::invariant("prepared strategies"))?;
    let contains = |m, n, k| {
        rows.iter()
            .any(|row| row["mode"] == "nvfp4" && row["m"] == m && row["n"] == n && row["k"] == k)
    };
    for m in [64, 1024] {
        assert!(
            !contains(m, 17408, 5120),
            "fused gate/up must not be tuned at M={m}"
        );
        assert!(
            contains(m, 5120, 17408),
            "live down must still be prepared at M={m}"
        );
    }
    for m in [1, 3] {
        assert!(
            contains(m, 17408, 5120),
            "unfused decode/verify must retain gate/up at M={m}"
        );
    }
    println!("live geometry only: {strategies}");
    drop(workspace);
    extra_consumer_retains_standalone(&loaded)?;
    Ok(())
}

fn extra_consumer_retains_standalone(loaded: &LoadedModel) -> Result<()> {
    use crate::resident::nvfp4_gemm::Workspace;
    use infer_core::{OpId, TensorId};
    use infer_ir::{TensorNode, TensorOp};
    let gate = loaded
        .graph
        .nodes
        .iter()
        .find(|node| {
            node.op == TensorOp::Linear
                && loaded
                    .weights
                    .projections
                    .get(&node.inputs[1])
                    .is_some_and(|weight| matches!(weight, ProjectionWeight::Fp4(codes, ..) if codes.shape() == [17408, 5120]))
        })
        .ok_or_else(|| Error::invalid("checkpoint gate projection"))?;
    let mut graph = loaded.graph.clone();
    let new_output = TensorId::new(1_000_000)?;
    let new_op = OpId::new(1_000_000)?;
    assert!(graph.tensors.iter().all(|tensor| tensor.id != new_output));
    assert!(graph.nodes.iter().all(|node| node.id != new_op));
    let mut tensor = graph
        .tensors
        .iter()
        .find(|t| t.id == gate.outputs[0])
        .ok_or_else(|| Error::invariant("gate output"))?
        .clone();
    tensor.id = new_output;
    graph.tensors.push(tensor);
    graph.nodes.push(TensorNode {
        id: new_op,
        layer: None,
        op: TensorOp::Silu,
        inputs: vec![gate.outputs[0]],
        outputs: vec![new_output],
        states: vec![],
    });
    let workspace = Workspace::new(loaded.device(), &graph, &loaded.weights, &[64, 1024])?;
    let configs = workspace.prepared_configs();
    let configs = configs
        .as_array()
        .ok_or_else(|| Error::invariant("prepared strategies"))?;
    for width in [64, 1024] {
        assert!(
            configs.iter().any(|row| row["mode"] == "nvfp4"
                && row["m"] == width
                && row["n"] == 17408
                && row["k"] == 5120),
            "a real additional reader prevents fusion and keeps standalone preparation"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA, checkpoint and private seed missing both live down geometries; safe-run"]
fn live_budget_completes_uncached_down_shapes() -> Result<()> {
    use crate::resident::nvfp4_gemm::Workspace;
    use crate::tuning::{resolve_table, table_path};
    let (mut loaded, _) = loaded_model(AttentionProfile::Bf16, 1024, 64)?;
    loaded.weights.autotune = true;
    let path = table_path().ok_or_else(|| Error::invalid("private tuning table required"))?;
    let before =
        resolve_table(Some(&path))?.ok_or_else(|| Error::invalid("private seed required"))?;
    for m in [64, 1024] {
        assert!(
            !before
                .resident_gemm
                .iter()
                .any(|e| e.key.m == m && e.key.n == 5120 && e.key.k == 17408),
            "fixture must really remove the live down shape before this fresh process"
        );
    }
    let workspace = Workspace::new(loaded.device(), &loaded.graph, &loaded.weights, &[64, 1024])?;
    let table =
        resolve_table(Some(&path))?.ok_or_else(|| Error::invalid("measured table required"))?;
    for m in [64, 1024] {
        let entry = table
            .resident_gemm
            .iter()
            .find(|e| e.key.m == m && e.key.n == 5120 && e.key.k == 17408)
            .ok_or_else(|| {
                Error::invalid(format!("missing completed actual-weight search at M={m}"))
            })?;
        assert!(entry.bitwise_verified);
        assert!(!entry.trials.is_empty());
        println!(
            "completed uncached down search: M={m}, selected={:?}",
            entry.selected
        );
    }
    println!(
        "live standalone strategies: {}",
        workspace.prepared_configs()
    );
    Ok(())
}
