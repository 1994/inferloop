//! Model bindings for the CUDA-resident token graph.
use super::Model;
use cutile::{
    half::bf16,
    prelude::{Arc, Tensor},
};
use infer_backend_cuda::loading::load_kv_scales;
use infer_backend_cuda::resident::{
    DeviceProgram, FusionWeights, ProgramWeights, PromptScratchRegistry,
};
use infer_backend_cuda::strategy::default_tiling;
use infer_backend_cuda::tuning::projection_key;
use infer_backend_cuda::tuning::resident_gemm::ResidentTuningSession;
use infer_core::{Error, Result};
use infer_ir::TensorOp;
use infer_models::TensorDtype;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Mutex;

pub type Embeddings = BTreeMap<infer_core::TensorId, Arc<Tensor<bf16>>>;

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Bounded rotary dimensions, intentionally rounded F32 device frequencies"
)]
pub fn prepare(
    model: &mut Model,
    shared: Option<&Embeddings>,
    fusion: Option<FusionWeights>,
) -> Result<()> {
    let mut weights = ProgramWeights {
        mrope: false,
        prompt_scratch: PromptScratchRegistry::default(),
        autotune: false,
        resident_tuning: Arc::new(ResidentTuningSession::default()),
        pool_gemv: Arc::new(Mutex::new(BTreeMap::new())),
        chunked_recurrent: false,
        batch_width: model.verify_width,
        prefill_width: model.prefill_width,
        narrow_prefill_width: 0,
        attention_profile: infer_backend_cuda::AttentionProfile::Canonical,
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        fusion,
        fp8_inputs: BTreeSet::new(),
        input_scales: BTreeMap::new(),
        projections: model
            .projections
            .iter()
            .map(|(id, p)| (*id, p.resident()))
            .collect(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        embeddings: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
    };
    if model.fp8_kv {
        weights.kv_scales = kv_scales(model)?;
    }
    for (id, projection) in &weights.projections {
        let (key, _, _) = projection_key(projection)?;
        let tiling = model.tuning.get(&key).copied().unwrap_or(default_tiling()?);
        weights.tiling.insert(*id, tiling);
    }
    for (id, data) in &model.constants {
        weights
            .constants
            .insert(*id, model.device.upload(data.clone(), &[data.len()])?);
    }
    for (id, source) in &model.embeddings {
        if let Some(embedding) = shared.and_then(|weights| weights.get(id)) {
            weights.embeddings.insert(*id, embedding.clone());
            continue;
        }
        if source.dtype != TensorDtype::BF16 {
            return Err(Error::unsupported("resident embedding requires BF16"));
        }
        // Explicit bounded staging: this model's embedding is larger than 2 GiB.
        let bytes = model.package.read(source, 4 * 1024 * 1024 * 1024)?;
        let values = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
            .collect();
        drop(bytes);
        weights
            .embeddings
            .insert(*id, model.device.upload(values, &source.shape)?);
    }
    for node in &model.package.graph.nodes {
        if let TensorOp::Rope {
            rotary_dim, theta, ..
        } = node.op
        {
            let half = rotary_dim / 2;
            let frequencies = (0..half)
                .map(|i| theta.powf(-((2 * i) as f64) / rotary_dim as f64) as f32)
                .collect();
            weights
                .rope_frequencies
                .insert(node.id, model.device.upload(frequencies, &[half])?);
        }
    }
    let config = &model.package.imported.model;
    model.resident = Some(DeviceProgram::new(
        &model.device,
        &model.package.graph,
        &weights,
        model.capacity,
        config.hidden_size,
        config.vocab_size,
    )?);
    model.device_embeddings = weights.embeddings;
    model.state.clear();
    eprintln!(
        "prepared resident token graph: {} nodes, {} activation bytes",
        model.package.graph.nodes.len(),
        model
            .resident
            .as_ref()
            .map_or(0, |program| program.activation_bytes)
    );
    Ok(())
}

fn kv_scales(model: &mut Model) -> Result<BTreeMap<infer_core::TensorId, [f32; 2]>> {
    load_kv_scales(&mut model.package)
}
