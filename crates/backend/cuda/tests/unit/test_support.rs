//! Fixtures shared by CUDA checkpoints that live in more than one module tree.
//!
//! A fixture used by both the resident and the loading checkpoints belongs to neither module's
//! test tree, so it is mounted once at the crate root under `cfg(all(test, linux, cuda))`.
use infer_core::{OpId, Result, TensorId};
use infer_ir::{DType, DataflowGraph, TensorNode, TensorOp, TensorSpec, TensorStorage};

pub fn fixture_graph(width: usize) -> Result<DataflowGraph> {
    let embedding = TensorId::ONE;
    let hidden = TensorId::new(2)?;
    let logits = TensorId::new(3)?;
    Ok(DataflowGraph {
        tensors: vec![
            TensorSpec {
                id: embedding,
                shape: vec![width, width],
                dtype: DType::Bf16,
                storage: TensorStorage::Weight {
                    slot: "embedding".into(),
                },
            },
            TensorSpec {
                id: hidden,
                shape: vec![width],
                dtype: DType::F32,
                storage: TensorStorage::Activation,
            },
            TensorSpec {
                id: logits,
                shape: vec![width],
                dtype: DType::F32,
                storage: TensorStorage::Activation,
            },
        ],
        nodes: vec![
            TensorNode {
                id: OpId::ONE,
                layer: None,
                op: TensorOp::Embedding,
                inputs: vec![embedding],
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
    })
}
