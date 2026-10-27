//! A single recurrent commit graph with independently uploaded acceptance counts.
use super::{
    batch::States,
    capture::error,
    slot_recurrent::{LANES, Staging, ValueDevice, covers, fused, fused_block, value_split},
    slot_states::RecurrentParameters,
};
use crate::{
    constants::{CONV_KERNEL_TILE, CONV_STATE_TENSORS, METADATA_FIELDS},
    device::{CudaDevice, PinnedUpload, device_error},
};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::{TensorNode, TensorOp};

pub(super) struct CapturedCommit {
    graph: CudaGraph<()>,
    counts: Tensor<i32>,
    upload: Vec<i32>,
    staging: PinnedUpload<i32>,
}

struct Inputs<'a> {
    staging: &'a Staging,
    parameters: &'a RecurrentParameters,
    metadata: &'a Tensor<i32>,
    counts: &'a Tensor<i32>,
    slots: usize,
    hardware: ValueDevice,
}

impl CapturedCommit {
    pub(super) fn supported(
        nodes: &[TensorNode],
        parameters: &RecurrentParameters,
        roots: &States,
        slots: usize,
    ) -> bool {
        nodes.iter().any(covers)
            && nodes.iter().filter(|node| covers(node)).all(|node| {
                let Some(state) = roots.get(&node.states[0]) else {
                    return false;
                };
                match node.op {
                    TensorOp::Conv { channels, .. } => {
                        state.len() == CONV_STATE_TENSORS
                            && state.iter().all(|s| has_shape(s, &[slots, channels]))
                    }
                    TensorOp::Delta {
                        value_heads,
                        key_dim,
                        value_dim,
                        ..
                    } => {
                        state.len() == 1
                            && has_shape(&state[0], &[slots, value_heads * key_dim, value_dim])
                            && parameters.contains_key(&node.id)
                    }
                    _ => false,
                }
            })
    }

    pub(super) fn new(
        device: &CudaDevice,
        nodes: &[TensorNode],
        roots: &mut States,
        (staging, parameters, metadata): (&Staging, &RecurrentParameters, &Tensor<i32>),
        slots: usize,
    ) -> Result<Self> {
        let counts = api::zeros::<i32>(&[slots])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let inputs = Inputs {
            staging,
            parameters,
            metadata,
            counts: &counts,
            slots,
            hardware: ValueDevice::new(device)?,
        };
        let graph = CudaGraph::scope(&device.stream, |scope| {
            for (index, node) in nodes.iter().enumerate().filter(|(_, node)| covers(node)) {
                record_node(scope, node, index, roots, &inputs).map_err(error)?;
            }
            Ok(())
        })
        .map_err(device_error)?;
        Ok(Self {
            graph,
            counts,
            upload: vec![0; slots],
            staging: PinnedUpload::single(device, slots)?,
        })
    }

    pub(super) fn update(&mut self, device: &CudaDevice, commits: &[(usize, usize)]) -> Result<()> {
        self.upload.fill(0);
        for &(slot, accepted) in commits {
            let count = self
                .upload
                .get_mut(slot)
                .ok_or_else(|| Error::invalid("commit slot"))?;
            *count = i32::try_from(accepted + 1).map_err(device_error)?;
        }
        self.staging.copy(device, &mut self.counts, &self.upload)
    }

    pub(super) const fn graph(&self) -> &CudaGraph<()> {
        &self.graph
    }
}

fn record_node(
    scope: &Scope,
    node: &TensorNode,
    index: usize,
    roots: &mut States,
    inputs: &Inputs<'_>,
) -> Result<()> {
    let state = roots
        .get_mut(&node.states[0])
        .ok_or_else(|| Error::invariant("commit state owner"))?;
    let input = |offset| {
        inputs
            .staging
            .root(index, offset)
            .ok_or_else(|| Error::invariant("commit staging owner"))
    };
    match node.op {
        TensorOp::Conv { channels, .. } => {
            let [h0, h1, h2] = state.as_mut_slice() else {
                return Err(Error::invariant("commit convolution state"));
            };
            scope
                .record(
                    fused::conv_commit_batch(
                        h0.partition([1, CONV_KERNEL_TILE]),
                        h1.partition([1, CONV_KERNEL_TILE]),
                        h2.partition([1, CONV_KERNEL_TILE]),
                        &input(0)?
                            .view(&[inputs.slots * LANES, channels])
                            .map_err(device_error)?,
                        inputs.metadata,
                        inputs.counts,
                    )
                    .generics(vec![METADATA_FIELDS.to_string()]),
                )
                .map_err(device_error)?;
        }
        TensorOp::Delta {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } => {
            let [state] = state.as_mut_slice() else {
                return Err(Error::invariant("commit delta state"));
            };
            let parameters = inputs
                .parameters
                .get(&node.id)
                .ok_or_else(|| Error::invariant("commit delta parameters"))?;
            let split = value_split(inputs.hardware, value_heads, value_dim, inputs.slots)?;
            let block = fused_block(value_dim, split);
            scope
                .record(
                    fused::delta_commit_batch(
                        state.partition([1, key_dim, block]),
                        input(0)?,
                        input(1)?,
                        input(2)?,
                        &parameters
                            .view(&[parameters.size()])
                            .map_err(device_error)?,
                        inputs.metadata,
                        inputs.counts,
                    )
                    .generics(vec![
                        key_heads.to_string(),
                        value_heads.to_string(),
                        key_dim.to_string(),
                        block.to_string(),
                        METADATA_FIELDS.to_string(),
                    ]),
                )
                .map_err(device_error)?;
        }
        _ => return Err(Error::invariant("covered recurrent commit node")),
    }
    Ok(())
}

fn has_shape(tensor: &Tensor<f32>, expected: &[usize]) -> bool {
    tensor.shape().len() == expected.len()
        && tensor
            .shape()
            .iter()
            .zip(expected)
            .all(|(&actual, &wanted)| usize::try_from(actual).ok() == Some(wanted))
}
