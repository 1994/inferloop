//! A single recurrent commit graph with independently uploaded acceptance counts.
use super::{
    batch::States,
    buffers::buffer,
    capture::error,
    slot_recurrent::{LANES, Staging, ValueDevice, covers, fused_block, value_split},
    slot_states::RecurrentParameters,
};
use crate::{
    constants::{CONV_STATE_TENSORS, METADATA_FIELDS},
    device::{CudaDevice, PinnedUpload, device_error},
};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    Buffer, Kernels, PoolConvPlan, PoolDeltaBuffers, PoolDeltaPlan, Recording,
};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_cuda_kernels::tile::runtime::Graph;
use infer_ir::{TensorNode, TensorOp};

pub(super) struct CapturedCommit {
    graph: Graph,
    counts: Allocation<i32>,
    upload: Vec<i32>,
    staging: PinnedUpload<i32>,
}

struct Inputs<'a> {
    kernels: &'a Kernels,
    staging: &'a Staging,
    parameters: &'a RecurrentParameters,
    metadata: &'a Allocation<i32>,
    counts: &'a Allocation<i32>,
    slots: usize,
    lanes: usize,
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
        kernels: &Kernels,
        nodes: &[TensorNode],
        roots: &mut States,
        (staging, parameters, metadata): (&Staging, &RecurrentParameters, &Allocation<i32>),
        (slots, lanes): (usize, usize),
    ) -> Result<Self> {
        let counts = device.runtime().zeros(slots)?;
        let inputs = Inputs {
            kernels,
            staging,
            parameters,
            metadata,
            counts: &counts,
            slots,
            lanes,
            hardware: ValueDevice::new(device)?,
        };
        let graph = Graph::scope(&device.stream, |scope| {
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
        self.staging.copy(device, &self.counts, &self.upload)
    }

    pub(super) const fn graph(&self) -> &Graph {
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
            let rows = buffer(input(0)?).map_err(device_error)?;
            record_conv(scope, state, rows, channels, inputs)?;
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
            inputs.kernels.recurrent_pool.delta_commit_pool(
                &Recording::scope(scope),
                &PoolDeltaPlan {
                    key_heads,
                    value_heads,
                    dim: key_dim,
                    block,
                    slots: inputs.slots,
                    lanes: inputs.lanes,
                    fields: METADATA_FIELDS,
                    fused: inputs.lanes == LANES,
                },
                buffer(inputs.counts).map_err(device_error)?,
                PoolDeltaBuffers {
                    state: buffer(&*state).map_err(device_error)?,
                    qkv: buffer(input(0)?).map_err(device_error)?,
                    beta: buffer(input(1)?).map_err(device_error)?,
                    alpha: buffer(input(2)?).map_err(device_error)?,
                    parameters: buffer(parameters).map_err(device_error)?,
                    metadata: buffer(inputs.metadata).map_err(device_error)?,
                },
            )?;
        }
        _ => return Err(Error::invariant("covered recurrent commit node")),
    }
    Ok(())
}

/// Shift each slot's convolution window by its accepted prefix.
fn record_conv(
    scope: &Scope,
    state: &[Tensor<f32>],
    rows: Buffer<'_, f32>,
    channels: usize,
    inputs: &Inputs<'_>,
) -> Result<()> {
    let [h0, h1, h2] = state else {
        return Err(Error::invariant("commit convolution state"));
    };
    inputs.kernels.recurrent_pool.conv_commit_pool(
        &Recording::scope(scope),
        &PoolConvPlan {
            channels,
            lanes: inputs.lanes,
            fields: METADATA_FIELDS,
            fused: inputs.lanes == LANES,
        },
        [
            buffer(h0).map_err(device_error)?,
            buffer(h1).map_err(device_error)?,
            buffer(h2).map_err(device_error)?,
        ],
        rows,
        buffer(inputs.metadata).map_err(device_error)?,
        buffer(inputs.counts).map_err(device_error)?,
    )
}

fn has_shape(tensor: &Tensor<f32>, expected: &[usize]) -> bool {
    tensor.shape().len() == expected.len()
        && tensor
            .shape()
            .iter()
            .zip(expected)
            .all(|(&actual, &wanted)| usize::try_from(actual).ok() == Some(wanted))
}
