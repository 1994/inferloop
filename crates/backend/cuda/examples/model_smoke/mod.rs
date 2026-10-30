pub mod decode;
pub mod mtp;
pub mod options;
pub mod prefill;
mod resident;
mod resident_mlp;
pub mod suite;
pub mod tuning;
mod verification;
mod weights;
use infer_backend_cuda::device::CudaDevice;
use infer_backend_cuda::resident::DeviceProgram;
use infer_backend_cuda::strategy::LinearTiling;
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, StateKind, TensorOp, TensorStorage};
use infer_models::{QuantizedPackage, WeightSource};
use infer_state::physical::{PagedRows, PhysicalTensor};
use std::{collections::BTreeMap, path::Path};
use weights::Projection;

fn weight_roles(graph: &DataflowGraph, id: TensorId, slot: &str) -> (bool, bool) {
    let embedding = slot == "embed_tokens.weight";
    let projection = graph
        .nodes
        .iter()
        .any(|node| node.op == TensorOp::Linear && node.inputs[1] == id);
    (embedding, projection)
}

pub fn prepare_graphs(
    model: &mut Model,
    draft: Option<&mut mtp::Mtp>,
    options: &options::Options,
) -> Result<()> {
    if let Some(path) = &options.tuning {
        model.tuning = tuning::load(path, model)?;
    }
    if options.device_graph {
        model.fp8_kv = options.fp8_kv;
        model.batch_verify = options.mtp > 0 && !options.sequential_verify;
        model.verify_width = if options.mtp > 0 && !options.sequential_verify {
            options.mtp + 1
        } else {
            0
        };
        model.prefill_width = options.prefill_batch;
        if options.prefill_batch == 3 {
            model.verify_width = model.verify_width.max(3);
        }
        resident::prepare(model, None, None)?;
        if let Some(draft) = draft {
            draft.model.tuning.clone_from(&model.tuning);
            draft.prepare_device_graph(&model.device_embeddings)?;
        }
    } else if options.mlp_graph {
        model.prepare_mlp_graphs(options.mlp_pdl)?;
        if let Some(draft) = draft {
            draft.model.prepare_mlp_graphs(options.mlp_pdl)?;
        }
    }
    Ok(())
}

pub struct Model {
    resident: Option<DeviceProgram>,
    device_embeddings: resident::Embeddings,
    capacity: usize,
    fp8_kv: bool,
    verify_width: usize,
    prefill_width: usize,
    batch_verify: bool,
    tuning: BTreeMap<String, LinearTiling>,
    pub package: QuantizedPackage,
    pub device: CudaDevice,
    projections: BTreeMap<TensorId, Projection>,
    constants: BTreeMap<TensorId, Vec<f32>>,
    embeddings: BTreeMap<TensorId, WeightSource>,
    state: BTreeMap<TensorId, PhysicalTensor>,
    elements: BTreeMap<TensorId, usize>,
    pub hidden: Vec<f32>,
    kv_offset: usize,
    mlps: BTreeMap<infer_core::OpId, resident_mlp::ResidentMlp>,
}

impl Model {
    pub fn load(root: &Path, capacity: usize) -> Result<Self> {
        Self::from_package(
            QuantizedPackage::open(root, infer_core::ModelId::ONE)?,
            capacity,
            None,
        )
    }

    fn from_package(
        package: QuantizedPackage,
        capacity: usize,
        shared: Option<&Self>,
    ) -> Result<Self> {
        let mut model = Self {
            resident: None,
            device_embeddings: BTreeMap::new(),
            capacity,
            tuning: BTreeMap::new(),
            fp8_kv: false,
            verify_width: 0,
            prefill_width: 1,
            batch_verify: false,
            package,
            device: shared.map_or_else(|| CudaDevice::new(0), |model| Ok(model.device.clone()))?,
            hidden: vec![],
            kv_offset: 0,
            mlps: BTreeMap::new(),
            projections: BTreeMap::new(),
            constants: BTreeMap::new(),
            embeddings: BTreeMap::new(),
            state: BTreeMap::new(),
            elements: BTreeMap::new(),
        };
        let specs = model.package.graph.tensors.clone();
        for spec in &specs {
            model.elements.insert(spec.id, spec.elements()?);
            match &spec.storage {
                TensorStorage::Weight { slot } => {
                    let weight = model.package.weights[slot].clone();
                    let (embedding, projection) = weight_roles(&model.package.graph, spec.id, slot);
                    if embedding {
                        model.embeddings.insert(spec.id, weight.data.clone());
                    }
                    if projection {
                        let loaded = if (slot == "lm_head.weight" || embedding)
                            && let Some(shared) = shared
                        {
                            shared.projection(slot)?.clone()
                        } else {
                            Projection::load(&model.device, &mut model.package, &weight)?
                        };
                        model.projections.insert(spec.id, loaded);
                        if model.projections.len().is_multiple_of(50) {
                            eprintln!("loaded {} GPU projections", model.projections.len());
                        }
                    } else if !embedding {
                        let values = weights::floats(&mut model.package, &weight.data)?;
                        model.constants.insert(spec.id, values);
                    }
                }
                TensorStorage::State { kind, .. } => {
                    let state = match kind {
                        StateKind::AttentionKv => PhysicalTensor::Kv {
                            keys: PagedRows::new(spec.shape[1], 16, capacity)?,
                            values: PagedRows::new(spec.shape[1], 16, capacity)?,
                        },
                        StateKind::Conv => PhysicalTensor::Conv {
                            channels: spec.shape[0],
                            kernel: spec.shape[1],
                            history: vec![0.0; spec.shape[0] * (spec.shape[1] - 1)],
                        },
                        StateKind::LinearAttention => PhysicalTensor::Delta {
                            heads: spec.shape[0],
                            key_dim: spec.shape[1],
                            value_dim: spec.shape[2],
                            recurrent: vec![0.0; spec.elements()?],
                        },
                        _ => return Err(Error::unsupported("diagnostic state kind")),
                    };
                    model.state.insert(spec.id, state);
                }
                TensorStorage::Activation => {}
            }
        }
        eprintln!(
            "loaded {} projections; {} recurrent/KV states",
            model.projections.len(),
            model.state.len()
        );
        Ok(model)
    }

    pub fn step(&mut self, token: u32, position: usize, read_logits: bool) -> Result<Vec<f32>> {
        self.forward(token, position, read_logits, None)
    }

    pub fn prepare_mlp_graphs(&mut self, pdl: bool) -> Result<()> {
        self.mlps = resident_mlp::prepare(self, pdl)?;
        eprintln!("prepared {} resident MLP graphs", self.mlps.len());
        Ok(())
    }

    fn projection(&self, name: &str) -> Result<&Projection> {
        let id = self
            .package
            .graph
            .tensors
            .iter()
            .find_map(|s| match &s.storage {
                TensorStorage::Weight { slot } if slot == name => Some(s.id),
                _ => None,
            })
            .ok_or_else(|| Error::invalid("projection not found"))?;
        self.projections
            .get(&id)
            .ok_or_else(|| Error::invalid("projection not loaded"))
    }

    fn forward(
        &mut self,
        token: u32,
        position: usize,
        read_logits: bool,
        embedding: Option<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        if let Some(program) = &mut self.resident {
            let kv_position = position
                .checked_sub(self.kv_offset)
                .ok_or_else(|| Error::invalid("MTP KV offset"))?;
            let (hidden, logits) = program.step(
                token,
                position,
                kv_position,
                embedding.as_deref(),
                read_logits,
            )?;
            self.hidden = hidden;
            return Ok(logits);
        }
        self.forward_diagnostic(token, position, read_logits, embedding)
    }

    fn forward_diagnostic(
        &mut self,
        token: u32,
        position: usize,
        read_logits: bool,
        mut embedding: Option<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let mut buffers: BTreeMap<TensorId, Vec<f32>> = BTreeMap::new();
        let nodes = self.package.graph.nodes.clone();
        let mut skip = 0;
        for node in &nodes {
            if skip > 0 {
                skip -= 1;
                continue;
            }
            if let Some(mlp) = self.mlps.get_mut(&node.id) {
                let output = mlp.graph.apply(&buffers[&mlp.input])?;
                buffers.insert(mlp.output, output);
                skip = 6;
                continue;
            }
            if !read_logits
                && self
                    .package
                    .graph
                    .logits
                    .is_some_and(|id| node.outputs.contains(&id))
            {
                continue;
            }
            let outputs = match node.op {
                TensorOp::Embedding => {
                    if let Some(values) = embedding.take() {
                        buffers.insert(node.outputs[0], values);
                        continue;
                    }
                    let source = &self.embeddings[&node.inputs[0]];
                    vec![
                        self.package
                            .read_float_row(source, token as usize, 1024 * 1024)?,
                    ]
                }
                TensorOp::Linear => vec![
                    self.projections[&node.inputs[1]]
                        .apply(&self.device, &buffers[&node.inputs[0]])?,
                ],
                _ => {
                    let inputs: Vec<&[f32]> = node
                        .inputs
                        .iter()
                        .map(|id| {
                            self.constants
                                .get(id)
                                .or_else(|| buffers.get(id))
                                .map(Vec::as_slice)
                                .ok_or_else(|| Error::invariant("missing diagnostic input"))
                        })
                        .collect::<Result<_>>()?;
                    let state = node.states.first().and_then(|id| self.state.get_mut(id));
                    let state_position = if matches!(node.op, TensorOp::Attention { .. }) {
                        position
                            .checked_sub(self.kv_offset)
                            .ok_or_else(|| Error::invalid("MTP KV offset"))?
                    } else {
                        position
                    };
                    infer_reference_ops::reference_operation(
                        &node.op,
                        &inputs,
                        state,
                        token,
                        state_position,
                    )?
                }
            };
            if outputs.len() != node.outputs.len() {
                return Err(Error::invariant("diagnostic output count mismatch"));
            }
            for (id, values) in node.outputs.iter().zip(outputs) {
                if values.len() != self.elements[id] {
                    return Err(Error::invariant("diagnostic output shape mismatch"));
                }
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(Error::invariant(format!(
                        "nonfinite values at {:?}",
                        node.id
                    )));
                }
                buffers.insert(*id, values);
            }
        }
        self.hidden = self
            .package
            .graph
            .hidden
            .and_then(|id| buffers.get(&id))
            .cloned()
            .ok_or_else(|| Error::invariant("missing final hidden state"))?;
        Ok(self
            .package
            .graph
            .logits
            .and_then(|id| buffers.remove(&id))
            .unwrap_or_default())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/model_smoke_bindings.rs"]
mod tests;
