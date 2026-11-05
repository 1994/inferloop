//! Model graph adapter; CUDA kernels do not know Qwen weight names or layer layout.
use super::Model;
use infer_backend_cuda::{
    mlp::{MlpConfig, MlpGraph, MlpWeights},
    strategy::LinearTiling,
};
use infer_core::{OpId, Result, TensorId};
use infer_ir::{TensorNode, TensorOp};
use std::collections::BTreeMap;

pub struct ResidentMlp {
    pub input: TensorId,
    pub output: TensorId,
    pub graph: MlpGraph,
}

pub fn prepare(model: &Model, pdl: bool) -> Result<BTreeMap<OpId, ResidentMlp>> {
    let mut result = BTreeMap::new();
    for nodes in model.package.graph.nodes.windows(7) {
        if let Some(resident) = lower(model, nodes, pdl)? {
            result.insert(nodes[0].id, resident);
        }
    }
    Ok(result)
}

fn lower(model: &Model, nodes: &[TensorNode], pdl: bool) -> Result<Option<ResidentMlp>> {
    let [norm, gate, up, silu, mul, down, add] = nodes else {
        return Ok(None);
    };
    let TensorOp::Norm {
        epsilon,
        offset,
        head_dim,
    } = norm.op
    else {
        return Ok(None);
    };
    if nodes
        .iter()
        .any(|n| n.outputs.len() != 1 || !n.states.is_empty())
        || norm.inputs.len() != 2
        || gate.inputs.len() != 2
        || up.inputs.len() != 2
        || down.inputs.len() != 2
        || gate.op != TensorOp::Linear
        || up.op != TensorOp::Linear
        || down.op != TensorOp::Linear
        || silu.op != TensorOp::Silu
        || mul.op != TensorOp::Multiply
        || add.op != TensorOp::Add
    {
        return Ok(None);
    }
    if gate.inputs[0] != norm.outputs[0]
        || up.inputs[0] != norm.outputs[0]
        || silu.inputs != gate.outputs
        || mul.inputs != [silu.outputs[0], up.outputs[0]]
        || down.inputs[0] != mul.outputs[0]
        || add.inputs != [norm.inputs[0], down.outputs[0]]
    {
        return Ok(None);
    }
    // Refuse fusion when any intermediate is observable outside this subgraph.
    for node in &nodes[..6] {
        let id = node.outputs[0];
        if model
            .package
            .graph
            .nodes
            .iter()
            .any(|other| other.inputs.contains(&id) && !nodes.iter().any(|n| n.id == other.id))
            || model.package.graph.hidden == Some(id)
            || model.package.graph.logits == Some(id)
        {
            return Ok(None);
        }
    }
    let hidden = model.elements[&norm.inputs[0]];
    if hidden != head_dim {
        return Ok(None);
    }
    let Some(weight) = model.constants.get(&norm.inputs[1]) else {
        return Ok(None);
    };
    let weights = MlpWeights {
        norm: model.device.runtime().upload(weight.clone())?,
        gate: model.projections[&gate.inputs[1]].clone(),
        up: model.projections[&up.inputs[1]].clone(),
        down: model.projections[&down.inputs[1]].clone(),
    };
    let config = MlpConfig {
        hidden,
        intermediate: model.elements[&gate.outputs[0]],
        epsilon,
        norm_offset: offset,
        tiling: LinearTiling::new(16, 256)?,
        pdl,
    };
    Ok(Some(ResidentMlp {
        input: norm.inputs[0],
        output: add.outputs[0],
        graph: MlpGraph::new(&model.device, &config, &weights)?,
    }))
}
