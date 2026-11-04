//! Single-layer Qwen MTP with a distinct KV cache and shared target vocabulary head.
use super::{
    Model,
    weights::{ProjectionWeight, floats},
};
use infer_backend_cuda::resident::DeviceProgram;
use infer_backend_cuda::resident::FusionWeights;
use infer_backend_cuda::strategy::default_tiling;
use infer_backend_cuda::tuning::projection_key;
use infer_core::{Error, Result, TensorId};
use infer_ir::TensorOp;
use infer_models::QuantizedPackage;
use infer_state::physical::PhysicalTensor;
use std::{collections::BTreeMap, path::Path};

pub struct Mtp {
    pub model: Model,
    pub(super) fc: ProjectionWeight,
    embedding_norm: Vec<f32>,
    hidden_norm: Vec<f32>,
}

impl Mtp {
    pub fn load(root: &Path, target: &Model, capacity: usize) -> Result<Self> {
        let mut package = QuantizedPackage::open(root, infer_core::ModelId::ONE)?;
        // The diagnostic loads the head exactly as its provider declares it.
        let plan = package
            .imported
            .speculation
            .clone()
            .ok_or_else(|| Error::unsupported("MTP draft head is not declared"))?;
        if plan.layers != 1 {
            return Err(Error::unsupported("diagnostic supports one MTP layer"));
        }
        let fusion = plan
            .fusion
            .as_ref()
            .ok_or_else(|| Error::unsupported("MTP fusion projection is not declared"))?;
        let get = |name: &str| {
            package
                .mtp
                .get(name)
                .cloned()
                .ok_or_else(|| Error::invalid(format!("missing {name}")))
        };
        let fc = get(&format!("{}{}", plan.prefix, fusion.projection))?;
        let embedding_norm = get(&format!("{}{}", plan.prefix, fusion.norms[0]))?;
        let hidden_norm = get(&format!("{}{}", plan.prefix, fusion.norms[1]))?;
        let hidden = package.imported.model.hidden_size;
        if fc.shape != [hidden, hidden * 2]
            || embedding_norm.shape != [hidden]
            || hidden_norm.shape != [hidden]
        {
            return Err(Error::invalid("MTP fusion weight shapes"));
        }
        let fc = ProjectionWeight::load(&target.device, &mut package, &fc)?;
        let embedding_norm = floats(&mut package, &embedding_norm.data)?;
        let hidden_norm = floats(&mut package, &hidden_norm.data)?;
        let (_, graph) = package.provider.draft_graph(&package.imported.model)?;
        package.graph = graph;
        let mut bindings = BTreeMap::new();
        for spec in &package.graph.tensors {
            if let infer_ir::TensorStorage::Weight { slot } = &spec.storage {
                let weight = if slot == "embed_tokens.weight" || slot == "lm_head.weight" {
                    package.weights.get(slot)
                } else {
                    package.mtp.get(&format!("{}{slot}", plan.prefix))
                }
                .ok_or_else(|| Error::invalid(format!("missing MTP binding {slot}")))?;
                if weight.shape != spec.shape {
                    return Err(Error::invalid(format!("MTP shape mismatch {slot}")));
                }
                bindings.insert(slot.clone(), weight.clone());
            }
        }
        package.weights = bindings;
        let mut model = Model::from_package(package, capacity, Some(target))?;
        model.kv_offset = 1;
        Ok(Self {
            model,
            fc,
            embedding_norm,
            hidden_norm,
        })
    }

    pub fn prepare_device_graph(&mut self, shared: &super::resident::Embeddings) -> Result<()> {
        let mut norms = self.embedding_norm.clone();
        norms.extend_from_slice(&self.hidden_norm);
        let config = &self.model.package.imported.model;
        let projection = self.fc.clone();
        let (key, _, _) = projection_key(&projection)?;
        let tiling = self
            .model
            .tuning
            .get(&key)
            .copied()
            .unwrap_or(default_tiling()?);
        let fusion = FusionWeights {
            projection,
            tiling,
            norms: self.model.device.runtime().upload(norms)?,
            epsilon: config.norm_epsilon,
            offset: config.norm_weight_offset,
        };
        super::resident::prepare(&mut self.model, Some(shared), Some(fusion))
    }

    pub fn step(
        &mut self,
        token: u32,
        previous_hidden: &[f32],
        position: usize,
        logits: bool,
    ) -> Result<Vec<f32>> {
        if self.model.resident.is_some() {
            return self
                .model
                .forward(token, position, logits, Some(previous_hidden.to_vec()));
        }
        let source = self
            .model
            .embeddings
            .values()
            .next()
            .ok_or_else(|| Error::invalid("MTP embedding"))?;
        let embedded = self
            .model
            .package
            .read_float_row(source, token as usize, 1024 * 1024)?;
        let config = &self.model.package.imported.model;
        let op = TensorOp::Norm {
            epsilon: config.norm_epsilon,
            offset: config.norm_weight_offset,
            head_dim: config.hidden_size,
        };
        let norm = |input: &[f32], weight: &[f32]| -> Result<Vec<f32>> {
            infer_reference_ops::reference_operation(&op, &[input, weight], None, 0, 0)?
                .pop()
                .ok_or_else(|| Error::invariant("MTP norm output"))
        };
        let mut fused = norm(&embedded, &self.embedding_norm)?;
        fused.extend(norm(previous_hidden, &self.hidden_norm)?);
        let fused = self.fc.apply(&self.model.device, &fused)?;
        self.model.forward(token, position, logits, Some(fused))
    }

    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            host: self.model.state.clone(),
            device_position: self.model.resident.as_ref().map(DeviceProgram::position),
        }
    }

    pub fn restore(&mut self, state: Checkpoint) -> Result<()> {
        if let (Some(program), Some(position)) = (&mut self.model.resident, state.device_position) {
            program.rewind_attention(position)?;
        }
        self.model.state = state.host;
        Ok(())
    }
}

#[derive(Clone)]
pub struct Checkpoint {
    host: BTreeMap<TensorId, PhysicalTensor>,
    device_position: Option<usize>,
}
