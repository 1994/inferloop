//! Identify same-input dense projection siblings from graph geometry.
use crate::constants::PREFILL_GEMM_TILE_N;
use infer_core::TensorId;
use infer_ir::{DataflowGraph, TensorNode, TensorOp, TensorSpec, TensorStorage};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(in crate::resident) struct DensePlan {
    narrow: BTreeSet<usize>,
    pairs: BTreeMap<usize, usize>,
}

impl DensePlan {
    pub(in crate::resident) fn new(graph: &DataflowGraph, single_row: Option<TensorId>) -> Self {
        let specs = graph.tensors.iter().map(|spec| (spec.id, spec)).collect();
        let shapes: Vec<_> = graph
            .nodes
            .iter()
            .map(|node| {
                if node.outputs.iter().any(|id| Some(*id) == single_row) {
                    None
                } else {
                    geometry(node, &specs)
                }
            })
            .collect();
        let narrow = shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| {
                shape
                    .filter(|shape| shape[0] <= PREFILL_GEMM_TILE_N)
                    .map(|_| index)
            })
            .collect();
        let mut pairs = BTreeMap::new();
        let mut index = 0;
        while index + 1 < graph.nodes.len() {
            let first = &graph.nodes[index];
            let second = &graph.nodes[index + 1];
            if shapes[index].is_some()
                && shapes[index] == shapes[index + 1]
                && first.layer == second.layer
                && first.inputs[0] == second.inputs[0]
                && first.outputs[0] != second.outputs[0]
            {
                pairs.insert(index, index + 1);
                index += 2;
            } else {
                index += 1;
            }
        }
        Self { narrow, pairs }
    }
    pub(in crate::resident) fn pair(&self, index: usize) -> Option<usize> {
        self.pairs.get(&index).copied()
    }
    pub(in crate::resident) fn narrow(&self, index: usize) -> bool {
        self.narrow.contains(&index)
    }
}

fn geometry(node: &TensorNode, specs: &BTreeMap<TensorId, &TensorSpec>) -> Option<[usize; 2]> {
    if node.op != TensorOp::Linear
        || node.inputs.len() != 2
        || node.outputs.len() != 1
        || !node.states.is_empty()
    {
        return None;
    }
    let input = specs.get(&node.inputs[0])?;
    let weight = specs.get(&node.inputs[1])?;
    let output = specs.get(&node.outputs[0])?;
    let columns = input.elements().ok()?;
    let rows = output.elements().ok()?;
    if rows == 0
        || columns == 0
        || !matches!(input.storage, TensorStorage::Activation)
        || !matches!(weight.storage, TensorStorage::Weight { .. })
        || weight.shape.as_slice() != [rows, columns]
    {
        return None;
    }
    Some([rows, columns])
}
