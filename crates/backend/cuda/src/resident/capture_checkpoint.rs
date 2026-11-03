//! State writes and their rollback snapshots share the original per-lane launch geometry.
use super::{
    buffers::buffer,
    capture::{Capture, CaptureMode, error},
    capture_state::value_block,
};
use crate::constants::CONV_KERNEL_SIZE;
use cutile::prelude::*;
use infer_cuda_kernels::contract::{ConvBuffers, DeltaBuffers, DeltaStepPlan, Recording};
use infer_ir::{TensorNode, TensorOp};
#[cfg(test)]
use std::cell::Cell;

impl Capture<'_> {
    /// Record only a conv/delta row that has a rollback snapshot. Other row captures
    /// keep their original entry point and kernels.
    pub(super) fn record_row_checkpoint(
        &mut self,
        node: &TensorNode,
        lane: usize,
        snapshot: &[Tensor<f32>],
    ) -> Result<(), DeviceError> {
        if self.mode != CaptureMode::Row(lane) || node.outputs.len() != 1 || node.states.len() != 1
        {
            return Err(error("checkpoint row capture bindings"));
        }
        let index = self.arena.slot(node.outputs[0]).map_err(error)?;
        let slot = self.arena.buffers[index]
            .take()
            .ok_or_else(|| error("missing checkpoint output buffer"))?;
        let (row, shared) =
            super::arena::row_tensor(slot, self.arena.lanes(), lane).map_err(error)?;
        let _ = self.checkpoint_state(node, row, snapshot)?;
        self.arena.buffers[index] = Some(
            Arc::try_unwrap(shared)
                .map_err(|_| error("checkpoint activation row tensor still aliased"))?,
        );
        Ok(())
    }

    /// Flat verification lanes own their activation slots; return that allocation
    /// after recording the same state writer used by pooled checkpoint rows.
    pub(super) fn record_flat_checkpoint(
        &mut self,
        node: &TensorNode,
        snapshot: &[Tensor<f32>],
    ) -> Result<(), DeviceError> {
        if self.mode != CaptureMode::Flat || node.outputs.len() != 1 || node.states.len() != 1 {
            return Err(error("checkpoint flat capture bindings"));
        }
        let index = self.arena.slot(node.outputs[0]).map_err(error)?;
        let output = self.arena.buffers[index]
            .take()
            .ok_or_else(|| error("missing checkpoint output buffer"))?;
        let shape = output
            .shape()
            .iter()
            .map(|&dimension| usize::try_from(dimension).map_err(error))
            .collect::<Result<Vec<_>, _>>()?;
        let size = output.size();
        let output = self.checkpoint_state(node, output.reshape(&[size])?, snapshot)?;
        self.arena.buffers[index] = Some(output.reshape(&shape)?);
        Ok(())
    }

    fn checkpoint_state(
        &mut self,
        node: &TensorNode,
        output: Tensor<f32>,
        snapshot: &[Tensor<f32>],
    ) -> Result<Tensor<f32>, DeviceError> {
        let id = node.states[0];
        let state = self
            .states
            .remove(&id)
            .ok_or_else(|| error("missing checkpoint state allocation"))?;
        let output = match node.op {
            TensorOp::Conv {
                channels,
                kernel: CONV_KERNEL_SIZE,
            } => self.checkpoint_conv(node, output, &state, snapshot, channels)?,
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => self.checkpoint_delta(
                node,
                output,
                &state,
                snapshot,
                (key_heads, value_heads, key_dim, value_dim),
            )?,
            _ => return Err(error("unsupported checkpoint state operation")),
        };
        self.states.insert(id, state);
        Ok(output)
    }

    fn checkpoint_conv(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
        state: &[Tensor<f32>],
        snapshot: &[Tensor<f32>],
        channels: usize,
    ) -> Result<Tensor<f32>, DeviceError> {
        let [h0, h1, h2] = state else {
            return Err(error("checkpoint convolution state layout"));
        };
        let [c0, c1, c2] = snapshot else {
            return Err(error("checkpoint convolution snapshot layout"));
        };
        self.weights
            .kernels
            .recurrent
            .conv_checkpoint(
                &Recording::scope(self.scope),
                channels,
                [buffer(c0)?, buffer(c1)?, buffer(c2)?],
                ConvBuffers {
                    output: buffer(&output)?,
                    history: [buffer(h0)?, buffer(h1)?, buffer(h2)?],
                    input: self.input_buffer(node.inputs[0])?,
                    weight: self.input_buffer(node.inputs[1])?,
                    metadata: buffer(self.metadata)?,
                },
            )
            .map_err(error)?;
        Ok(output)
    }

    fn checkpoint_delta(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
        state: &[Tensor<f32>],
        snapshot: &[Tensor<f32>],
        (key_heads, value_heads, key_dim, value_dim): (usize, usize, usize, usize),
    ) -> Result<Tensor<f32>, DeviceError> {
        if key_dim != value_dim
            || !key_dim.is_power_of_two()
            || key_heads == 0
            || !value_heads.is_multiple_of(key_heads)
        {
            return Err(error("checkpoint delta dimensions"));
        }
        let [state] = state else {
            return Err(error("checkpoint delta state layout"));
        };
        let [snapshot] = snapshot else {
            return Err(error("checkpoint delta snapshot layout"));
        };
        let block = value_block(value_dim)?;
        let output = output.reshape(&[value_heads, 1, value_dim])?;
        self.weights
            .kernels
            .recurrent
            .delta_checkpoint(
                &Recording::scope(self.scope),
                &DeltaStepPlan {
                    key_heads,
                    value_heads,
                    dim: key_dim,
                    block,
                    lanes: None,
                },
                buffer(snapshot)?,
                DeltaBuffers {
                    state: buffer(state)?,
                    output: buffer(&output)?,
                    qkv: self.input_buffer(node.inputs[0])?,
                    beta: self.input_buffer(node.inputs[1])?,
                    alpha: self.input_buffer(node.inputs[2])?,
                    decay: self.input_buffer(node.inputs[3])?,
                    bias: self.input_buffer(node.inputs[4])?,
                    metadata: buffer(self.metadata)?,
                },
            )
            .map_err(error)?;
        Ok(output)
    }
}

#[cfg(not(test))]
pub(super) const fn flat_checkpoint_enabled() -> bool {
    true
}

#[cfg(test)]
pub(super) fn flat_checkpoint_enabled() -> bool {
    !LEGACY_FLAT_CAPTURE.get()
}

#[cfg(test)]
thread_local! {
    pub(super) static LEGACY_FLAT_CAPTURE: Cell<bool> =
        const { Cell::new(false) };
}

#[cfg(test)]
#[path = "../../tests/unit/resident_capture_checkpoint.rs"]
mod tests;
