//! Target-side features a speculative draft reads, lowered into the target's own graph.
use infer_core::{Error, OpId, Result, TensorId};
use infer_ir::{DType, DataflowGraph, TensorNode, TensorOp, TensorSpec, TensorStorage};

/// Weight slot of the projection applied to the `index`-th tapped layer.
#[must_use]
pub fn tap_projection_slot(index: usize) -> String {
    format!("draft.fc.{index}")
}

/// Append the draft's fused tap feature to a lowered decoder graph.
///
/// `x_L` is the residual stream once layer `L` returns, and `W_i` is the `i`-th `[hidden, hidden]`
/// column block of the draft's fusion projection, bound to [`tap_projection_slot`]. The appended
/// nodes compute `Σ_i W_i · x_{taps[i]}`, which is the draft's `fc(concat(x_taps))` without
/// materializing the concatenation: only one running sum stays alive across the graph instead of
/// every tapped row. Each projection is recorded right after the layer it reads, so the tapped
/// stream does not outlive its layer.
///
/// # Errors
/// Rejects an empty or unordered tap list, a tap without a layer, a layer whose last node is not
/// a residual of width `hidden`, a graph that already carries features, or overflowing IDs.
pub fn attach_tap_projection(
    graph: &mut DataflowGraph,
    taps: &[usize],
    hidden: usize,
) -> Result<()> {
    if graph.features.is_some() {
        return Err(Error::invalid("graph already carries draft features"));
    }
    if taps.is_empty() || taps.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::invalid(
            "draft taps must be a non-empty ascending list",
        ));
    }
    let largest = graph
        .tensors
        .iter()
        .map(|tensor| tensor.id.get())
        .chain(graph.nodes.iter().map(|node| node.id.get()))
        .max()
        .unwrap_or(0);
    let mut next = largest
        .checked_add(1)
        .ok_or_else(|| Error::invalid("graph identities exhausted"))?;
    let mut allocate = || -> Result<u64> {
        let id = next;
        next = next
            .checked_add(1)
            .ok_or_else(|| Error::invalid("graph identities exhausted"))?;
        Ok(id)
    };
    let mut sum: Option<TensorId> = None;
    // Insert from the deepest tap so earlier insertion points keep their indices; the running sum
    // is threaded front to back afterwards.
    let mut insertions = Vec::with_capacity(taps.len());
    for (index, &layer) in taps.iter().enumerate() {
        let position = graph
            .nodes
            .iter()
            .rposition(|node| node.layer == Some(layer))
            .ok_or_else(|| Error::invalid(format!("draft tap {layer} has no target layer")))?;
        let node = &graph.nodes[position];
        let stream = match (&node.op, node.outputs.as_slice()) {
            (TensorOp::Add, [output]) => *output,
            _ => {
                return Err(Error::invalid(format!(
                    "target layer {layer} does not end in a residual"
                )));
            }
        };
        if graph
            .tensors
            .iter()
            .find(|tensor| tensor.id == stream)
            .is_none_or(|tensor| tensor.shape != [hidden])
        {
            return Err(Error::invalid(format!(
                "target layer {layer} residual is not {hidden} wide"
            )));
        }
        let weight = TensorId::new(allocate()?)?;
        graph.tensors.push(TensorSpec {
            id: weight,
            shape: vec![hidden, hidden],
            dtype: DType::F32,
            storage: TensorStorage::Weight {
                slot: tap_projection_slot(index),
            },
        });
        let projected = activation(graph, allocate()?, hidden)?;
        let mut nodes = vec![TensorNode {
            id: OpId::new(allocate()?)?,
            layer: Some(layer),
            op: TensorOp::Linear,
            inputs: vec![stream, weight],
            outputs: vec![projected],
            states: vec![],
        }];
        sum = Some(match sum {
            None => projected,
            Some(previous) => {
                let total = activation(graph, allocate()?, hidden)?;
                nodes.push(TensorNode {
                    id: OpId::new(allocate()?)?,
                    layer: Some(layer),
                    op: TensorOp::Add,
                    inputs: vec![previous, projected],
                    outputs: vec![total],
                    states: vec![],
                });
                total
            }
        });
        insertions.push((position + 1, nodes));
    }
    for (position, nodes) in insertions.into_iter().rev() {
        graph.nodes.splice(position..position, nodes);
    }
    graph.features = sum;
    graph.plan_lifetimes()
}

fn activation(graph: &mut DataflowGraph, raw: u64, width: usize) -> Result<TensorId> {
    let id = TensorId::new(raw)?;
    graph.tensors.push(TensorSpec {
        id,
        shape: vec![width],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    });
    Ok(id)
}

#[cfg(test)]
#[path = "../tests/unit/features.rs"]
mod tests;
