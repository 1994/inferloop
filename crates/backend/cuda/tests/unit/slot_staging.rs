use super::*;
use crate::test_support::fixture_graph;
use infer_core::OpId;
use infer_cuda_kernels::tile::runtime::Graph;
use infer_ir::{DType, DataflowGraph, TensorSpec, TensorStorage};
use std::mem::take;

fn staging_fixture(
    width: usize,
) -> infer_core::Result<(DataflowGraph, [TensorId; 3], Vec<TensorNode>)> {
    let mut graph = fixture_graph(width)?;
    let first = graph
        .hidden
        .ok_or_else(|| Error::invariant("fixture hidden"))?;
    let second = graph
        .logits
        .ok_or_else(|| Error::invariant("fixture logits"))?;
    let third = TensorId::new(4)?;
    graph.tensors.push(TensorSpec {
        id: third,
        shape: vec![width],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    });
    graph.nodes.push(TensorNode {
        id: OpId::new(3)?,
        layer: None,
        op: TensorOp::Add,
        inputs: vec![first, second],
        outputs: vec![third],
        states: vec![],
    });
    graph.logits = Some(third);
    let sources = [first, second, third];
    let nodes = vec![
        TensorNode {
            id: OpId::new(4)?,
            layer: None,
            op: TensorOp::Conv {
                channels: width,
                kernel: CONV_KERNEL_SIZE,
            },
            inputs: vec![first, TensorId::ONE],
            outputs: vec![second],
            states: vec![TensorId::new(100)?],
        },
        TensorNode {
            id: OpId::new(5)?,
            layer: None,
            op: TensorOp::Delta {
                key_heads: 4,
                value_heads: 4,
                key_dim: 4,
                value_dim: 4,
            },
            inputs: vec![first, second, third, TensorId::ONE, TensorId::ONE],
            outputs: vec![second],
            states: vec![TensorId::new(101)?],
        },
    ];
    Ok((graph, sources, nodes))
}

/// Batched copies must preserve every candidate's commit input after producer storage is reused.
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn contiguous_staging_preserves_rows_after_arena_reuse() -> infer_core::Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let width = 48;
    let slots = 5;
    let lanes = slots * LANES;
    let (graph, sources, nodes) = staging_fixture(width)?;
    let mut arena = ActivationArena::new_batched(&device, &graph, lanes, 8 << 20)?;
    let mut expected = BTreeMap::new();
    for (index, id) in sources.into_iter().enumerate() {
        let data = (0..lanes * width)
            .map(|i| {
                i16::try_from(index * 4096 + i)
                    .map(f32::from)
                    .map_err(|e| Error::invalid(e.to_string()))
            })
            .collect::<infer_core::Result<Vec<_>>>()?;
        let slot = arena.slot(id)?;
        let target = arena.buffers[slot]
            .as_mut()
            .ok_or_else(|| Error::invariant("staging fixture source"))?;
        device.copy_host_slice(target, &data)?;
        expected.insert(id, data);
    }
    let mut staging = allocate_staging(&device, &nodes, &arena, (slots, LANES))?;
    assert_eq!(staging.roots.len(), 4);
    assert_eq!(staging.entries.len(), slots * nodes.len());
    let copy = Graph::scope(&device.stream, |scope| {
        for (index, node) in nodes.iter().enumerate() {
            staging.record_node(scope, node, &arena, index)?;
        }
        Ok(())
    })
    .map_err(device_error)?;
    copy.launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    for id in sources {
        let slot = arena.slot(id)?;
        let target = arena.buffers[slot]
            .as_mut()
            .ok_or_else(|| Error::invariant("staging fixture source"))?;
        device.copy_host_slice(target, &vec![0.0; lanes * width])?;
    }
    drop(copy);
    drop(arena);
    for ((slot, index), inputs) in take(&mut staging.entries) {
        for (input, candidates) in inputs.into_iter().enumerate() {
            let id = commit_inputs(&nodes[index])[input];
            for (candidate, row) in candidates.into_iter().enumerate() {
                let begin = (slot * LANES + candidate) * width;
                assert_eq!(
                    row.to_host_vec()
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                    expected[&id][begin..begin + width],
                    "slot={slot} node={index} input={input} candidate={candidate}"
                );
            }
        }
    }
    Ok(())
}
