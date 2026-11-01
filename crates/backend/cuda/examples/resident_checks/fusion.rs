use cutile::half::bf16;
use infer_backend_cuda::strategy::default_tiling;
use infer_backend_cuda::tuning::resident_gemm::ResidentTuningSession;
use infer_backend_cuda::{
    device::CudaDevice,
    mlp::ProjectionWeight,
    resident::{DeviceProgram, FusionWeights, ProgramWeights, PromptScratchRegistry},
};
use infer_core::{Error, OpId, Result, TensorId};
use infer_ir::{DType, DataflowGraph, TensorNode, TensorOp, TensorSpec, TensorStorage};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

pub fn run(device: &CudaDevice) -> Result<()> {
    let d = 48;
    let id = TensorId::ONE;
    let hidden = TensorId::new(2)?;
    let logits = TensorId::new(3)?;
    let graph = DataflowGraph {
        tensors: vec![
            TensorSpec {
                id,
                shape: vec![2, d],
                dtype: DType::Bf16,
                storage: TensorStorage::Weight {
                    slot: "embedding".into(),
                },
            },
            TensorSpec {
                id: hidden,
                shape: vec![d],
                dtype: DType::F32,
                storage: TensorStorage::Activation,
            },
            TensorSpec {
                id: logits,
                shape: vec![d],
                dtype: DType::F32,
                storage: TensorStorage::Activation,
            },
        ],
        nodes: vec![
            TensorNode {
                id: OpId::ONE,
                layer: None,
                op: TensorOp::Embedding,
                inputs: vec![id],
                outputs: vec![hidden],
                states: vec![],
            },
            TensorNode {
                id: OpId::new(2)?,
                layer: None,
                op: TensorOp::Add,
                inputs: vec![hidden, hidden],
                outputs: vec![logits],
                states: vec![],
            },
        ],
        hidden: Some(hidden),
        logits: Some(logits),
        features: None,
        ..Default::default()
    };
    let weights = weights(device, d, id)?;
    let mut program = DeviceProgram::new(device, &graph, &weights, 16, d, 2)?;
    for position in 0..16 {
        let token = u32::from(position % 2 != 0);
        let embedded: f32 = if token == 0 { 0.25 } else { -0.5 };
        let previous: f32 = if position % 3 == 0 { -0.1 } else { 0.3 };
        let expected = (embedded / embedded.mul_add(embedded, 1e-6).sqrt()
            + previous / previous.mul_add(previous, 1e-6).sqrt())
            * 1.5;
        let (actual, _) =
            program.step(token, position, position, Some(&vec![previous; d]), true)?;
        if actual
            .iter()
            .any(|v| !v.is_finite() || (v - expected).abs() > 1e-4)
        {
            return Err(Error::invariant(format!(
                "MTP fusion mismatch: actual={}, expected={expected}",
                actual[0]
            )));
        }
    }
    println!("PASS MTP embedding + dual norm + FC: 16 varied token/hidden replays");
    Ok(())
}

fn weights(device: &CudaDevice, d: usize, id: TensorId) -> Result<ProgramWeights> {
    let embedding: Vec<_> = [0.25, -0.5]
        .into_iter()
        .flat_map(|v| vec![bf16::from_f32(v); d])
        .collect();
    Ok(ProgramWeights {
        kernels: device.kernels(),
        mrope: false,
        prompt_scratch: PromptScratchRegistry::default(),
        autotune: false,
        resident_tuning: Arc::new(ResidentTuningSession::default()),
        chunked_recurrent: false,
        batch_width: 0,
        prefill_width: 1,
        narrow_prefill_width: 0,
        attention_profile: infer_backend_cuda::AttentionProfile::Canonical,
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        fusion: Some(FusionWeights {
            projection: ProjectionWeight::Dense(
                device.upload(vec![bf16::from_f32(0.03125); d * 2 * d], &[d, 2 * d])?,
            ),
            // This synthetic check measures nothing, so it uses the hardware-independent tile.
            tiling: default_tiling()?,
            norms: device.upload(vec![0.0; 2 * d], &[2, d])?,
            epsilon: 1e-6,
            offset: 1.0,
        }),
        fp8_inputs: BTreeSet::new(),
        input_scales: BTreeMap::new(),
        projections: BTreeMap::new(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        embeddings: BTreeMap::from([(id, device.upload(embedding, &[2, d])?)]),
    })
}
