use super::{
    capture::{Capture, error},
    recurrent::recurrent,
};
use crate::constants::CONV_KERNEL_SIZE;
use crate::constants::CONV_KERNEL_TILE;
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};
#[cfg(test)]
use std::env::{var, var_os};

/// The untuned per-lane path retains one block per value head.
#[cfg(not(test))]
const fn value_split() -> usize {
    1
}

/// Internal tests can compare value partitions without adding a production tuning knob.
#[cfg(test)]
fn value_split() -> usize {
    var("INFER_CUDA_RECURRENT_VALUE_SPLIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
}

/// Whether an internal diagnostic overrides the measured chunk-prefill choice.
#[cfg(not(test))]
pub(super) const fn value_split_override() -> bool {
    false
}
#[cfg(test)]
pub(super) fn value_split_override() -> bool {
    var_os("INFER_CUDA_RECURRENT_VALUE_SPLIT").is_some()
}

/// Width of one value-dimension block, rejecting a split that does not divide the dimension.
pub(super) fn value_block(value_dim: usize) -> Result<usize, DeviceError> {
    let split = value_split();
    if split == 0 || !value_dim.is_multiple_of(split) {
        return Err(error("recurrent value split"));
    }
    Ok(value_dim / split)
}

impl Capture<'_> {
    fn conv(
        &self,
        node: &TensorNode,
        output: &mut Tensor<f32>,
        state: &mut [Tensor<f32>],
        channels: usize,
    ) -> Result<(), DeviceError> {
        let [h0, h1, h2] = state else {
            return Err(error("convolution state layout"));
        };
        self.scope.record(recurrent::conv4(
            output.partition([CONV_KERNEL_TILE]),
            h0.partition([CONV_KERNEL_TILE]),
            h1.partition([CONV_KERNEL_TILE]),
            h2.partition([CONV_KERNEL_TILE]),
            &self.input(node.inputs[0])?,
            &self
                .input(node.inputs[1])?
                .view(&[channels, CONV_KERNEL_SIZE])?,
            self.metadata,
        ))?;
        Ok(())
    }

    pub fn record_state(
        &mut self,
        node: &TensorNode,
        mut output: Tensor<f32>,
    ) -> Result<Tensor<f32>, DeviceError> {
        let id = *node
            .states
            .first()
            .ok_or_else(|| error("missing state binding"))?;
        // The pooled graph holds state roots here, not lane views: one replay advances every
        // busy lane instead of one launch per lane.
        if self.mode == super::capture::CaptureMode::Pooled && self.states.contains_key(&id) {
            match node.op {
                TensorOp::Delta { .. } => return self.record_delta_lanes(node, output),
                TensorOp::Conv { channels, .. } => {
                    return self.record_conv_lanes(node, output, channels);
                }
                _ => {}
            }
        }
        if let Some((keys, values)) = self.fp8_states.remove(&id) {
            let (output, keys, values) =
                self.record_attention(node, output, keys, values, self.weights.kv_scales[&id])?;
            self.fp8_states.insert(id, (keys, values));
            return Ok(output);
        }
        let mut state = self
            .states
            .remove(&id)
            .ok_or_else(|| error("missing state allocation"))?;
        match node.op {
            TensorOp::Conv {
                channels,
                kernel: CONV_KERNEL_SIZE,
            } => {
                self.conv(node, &mut output, &mut state, channels)?;
            }
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                if key_dim != value_dim
                    || !key_dim.is_power_of_two()
                    || key_heads == 0
                    || !value_heads.is_multiple_of(key_heads)
                {
                    return Err(error("resident delta dimensions"));
                }
                output = output.reshape(&[value_heads, value_dim])?;
                // The value dimension may be split so the grid can fill the device: the
                // established single-block launch is one block per value head, 48 of them on the
                // 27B against 170 multiprocessors, and each block moves the whole D-by-D state for
                // every token. Splitting the columns is exact - every reduction is over the key
                // dimension - and is off by default until the full matrix shows a gain.
                let block = value_block(value_dim)?;
                output = output.reshape(&[value_heads, 1, value_dim])?;
                self.scope.record(
                    recurrent::delta(
                        (&mut output).partition([1, 1, block]),
                        (&mut state[0]).partition([1, key_dim, block]),
                        &self.input(node.inputs[0])?,
                        &self.input(node.inputs[1])?,
                        &self.input(node.inputs[2])?,
                        &self.input(node.inputs[3])?,
                        &self.input(node.inputs[4])?,
                        self.metadata,
                    )
                    .generics(vec![
                        key_heads.to_string(),
                        value_heads.to_string(),
                        key_dim.to_string(),
                        block.to_string(),
                    ]),
                )?;
                output = output.reshape(&[value_heads, value_dim])?;
            }
            TensorOp::Attention { .. } => {
                let keys = state.remove(0);
                let values = state.remove(0);
                let (result, keys, values) =
                    self.record_attention(node, output, keys, values, [1.0, 1.0])?;
                output = result;
                state.push(keys);
                state.push(values);
            }
            _ => return Err(error("unsupported resident state operation")),
        }
        self.states.insert(id, state);
        Ok(output)
    }

    /// Lane-batched convolution decode: one kernel over every lane's pooled history rows.
    fn record_conv_lanes(
        &mut self,
        node: &TensorNode,
        output: Tensor<f32>,
        channels: usize,
    ) -> Result<Tensor<f32>, DeviceError> {
        let id = *node
            .states
            .first()
            .ok_or_else(|| error("missing state binding"))?;
        let lanes = self.arena.lanes();
        let mut state = self
            .states
            .remove(&id)
            .ok_or_else(|| error("missing state allocation"))?;
        let [h0, h1, h2] = state.as_mut_slice() else {
            return Err(error("convolution state layout"));
        };
        let mut out = output.reshape(&[lanes, channels])?;
        self.scope.record(recurrent::conv4_lanes(
            (&mut out).partition([1, CONV_KERNEL_TILE]),
            h0.partition([1, CONV_KERNEL_TILE]),
            h1.partition([1, CONV_KERNEL_TILE]),
            h2.partition([1, CONV_KERNEL_TILE]),
            &self.input(node.inputs[0])?.view(&[lanes, channels])?,
            &self
                .input(node.inputs[1])?
                .view(&[channels, CONV_KERNEL_SIZE])?,
            self.metadata,
        ))?;
        self.states.insert(id, state);
        Ok(out.reshape(&[lanes * channels])?)
    }

    /// Lane-batched delta decode: one kernel over `[lanes * value_heads]` blocks. The state
    /// root is lane-major, so flattening its head and lane axes addresses every lane's
    /// `[value_heads, key, value]` state without copying.
    fn record_delta_lanes(
        &mut self,
        node: &TensorNode,
        output: Tensor<f32>,
    ) -> Result<Tensor<f32>, DeviceError> {
        let TensorOp::Delta {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } = node.op
        else {
            return Err(error("expected delta operation"));
        };
        if key_dim != value_dim
            || !key_dim.is_power_of_two()
            || key_heads == 0
            || !value_heads.is_multiple_of(key_heads)
        {
            return Err(error("resident delta dimensions"));
        }
        let id = *node
            .states
            .first()
            .ok_or_else(|| error("missing state binding"))?;
        let lanes = self.arena.lanes();
        let mut state = self
            .states
            .remove(&id)
            .ok_or_else(|| error("missing state allocation"))?;
        let root = state.pop().ok_or_else(|| error("delta state layout"))?;
        let shape = root
            .shape()
            .iter()
            .map(|&dim| usize::try_from(dim).map_err(|_| error("delta state shape")))
            .collect::<Result<Vec<_>, _>>()?;
        let mut batched = root.reshape(&[
            lanes
                .checked_mul(value_heads)
                .ok_or_else(|| error("delta lane overflow"))?,
            key_dim,
            value_dim,
        ])?;
        let block = value_block(value_dim)?;
        let mut out = output.reshape(&[lanes * value_heads, 1, value_dim])?;
        self.scope.record(
            recurrent::delta_lanes(
                (&mut out).partition([1, 1, block]),
                (&mut batched).partition([1, key_dim, block]),
                &self.input(node.inputs[0])?,
                &self.input(node.inputs[1])?,
                &self.input(node.inputs[2])?,
                &self.input(node.inputs[3])?,
                &self.input(node.inputs[4])?,
                self.metadata,
            )
            .generics(vec![
                key_heads.to_string(),
                value_heads.to_string(),
                key_dim.to_string(),
                block.to_string(),
            ]),
        )?;
        state.push(batched.reshape(&shape)?);
        self.states.insert(id, state);
        Ok(out.reshape(&[lanes * value_heads * value_dim])?)
    }
}
