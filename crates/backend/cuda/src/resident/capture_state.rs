use super::recurrent_lowering::{Delta, Lowering, Mode, Recurrence, lower};
use super::shaped::Shaped;
use super::{
    arena::SlotMemory,
    buffers::{buffer, kv_cache},
    capture::{Capture, error},
};
use infer_core::{Result, TensorId};
use infer_cuda_kernels::contract::{Buffer, ConvBuffers, ConvKernel, DeltaBuffers, DeltaStepPlan};
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
pub(super) fn value_block(value_dim: usize) -> Result<usize> {
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
        output: &SlotMemory,
        state: &[Shaped<f32>],
        channels: usize,
    ) -> Result<()> {
        let [h0, h1, h2] = state else {
            return Err(error("convolution state layout"));
        };
        self.weights
            .kernels
            .recurrent
            .conv(
                self.scope,
                ConvKernel::Step,
                channels,
                ConvBuffers {
                    output: buffer(output)?,
                    history: [buffer(h0)?, buffer(h1)?, buffer(h2)?],
                    input: self.input_buffer(node.inputs[0])?,
                    weight: self.input_buffer(node.inputs[1])?,
                    metadata: self.metadata,
                },
            )
            .map_err(error)?;
        Ok(())
    }

    pub fn record_state(&mut self, node: &TensorNode, output: &SlotMemory) -> Result<()> {
        let id = *node
            .states
            .first()
            .ok_or_else(|| error("missing state binding"))?;
        if let Some(recurrence) = Recurrence::of(&node.op)? {
            // The pooled graph holds state roots here, not lane views: one replay advances every
            // busy lane instead of one launch per lane.
            let pooled =
                self.mode == super::capture::CaptureMode::Pooled && self.states.contains_key(&id);
            let mode = if pooled { Mode::Pooled } else { Mode::Lane };
            return match (lower(recurrence, mode), recurrence) {
                (Lowering::Lanes, Recurrence::Conv { channels }) => {
                    self.record_conv_lanes(node, output, channels)
                }
                (Lowering::Lanes, Recurrence::Delta(delta)) => {
                    self.record_delta_lanes(node, output, delta)
                }
                (Lowering::Step, recurrence) => self.record_step(node, output, id, recurrence),
                _ => Err(error("recurrent decode lowering")),
            };
        }
        if let Some(state) = self.fp8_states.get(&id).cloned() {
            let cache = state.cache(self.weights.kv_scales[&id]);
            return self.record_attention(node, output, cache);
        }
        let TensorOp::Attention {
            kv_heads, head_dim, ..
        } = node.op
        else {
            return Err(error("unsupported resident state operation"));
        };
        let mut state = self
            .states
            .remove(&id)
            .ok_or_else(|| error("missing state allocation"))?;
        if state.len() != 2 {
            return Err(error("attention state layout"));
        }
        // Unscaled KV is still a shaped tensor whose copies read the head-major shape.
        let rows = state[0].len() / (kv_heads * head_dim).max(1);
        let keys = state
            .remove(0)
            .reshaped(vec![kv_heads, rows, head_dim])
            .map_err(error)?;
        let values = state
            .remove(0)
            .reshaped(vec![kv_heads, rows, head_dim])
            .map_err(error)?;
        let recorded = self.record_attention(
            node,
            output,
            kv_cache((&keys, &values), kv_heads, head_dim, [1.0, 1.0])?,
        );
        state.push(keys);
        state.push(values);
        self.states.insert(id, state);
        recorded
    }

    /// One lane's convolution or delta step against its own state.
    fn record_step(
        &mut self,
        node: &TensorNode,
        output: &SlotMemory,
        id: TensorId,
        recurrence: Recurrence,
    ) -> Result<()> {
        let state = self
            .states
            .remove(&id)
            .ok_or_else(|| error("missing state allocation"))?;
        let recorded = match recurrence {
            Recurrence::Conv { channels } => self.conv(node, output, &state, channels),
            // The value dimension may be split so the grid can fill the device: the established
            // single-block launch is one block per value head, 48 of them on the 27B against 170
            // multiprocessors, and each block moves the whole D-by-D state for every token.
            // Splitting the columns is exact - every reduction is over the key dimension - and is
            // off by default until the full matrix shows a gain.
            Recurrence::Delta(delta) => value_block(delta.value_dim).and_then(|block| {
                self.record_delta_step(
                    node,
                    &DeltaStepPlan {
                        key_heads: delta.key_heads,
                        value_heads: delta.value_heads,
                        dim: delta.key_dim,
                        block,
                        lanes: None,
                    },
                    buffer(output)?,
                    buffer(&state[0])?,
                )
            }),
        };
        self.states.insert(id, state);
        recorded
    }

    fn record_delta_step(
        &self,
        node: &TensorNode,
        plan: &DeltaStepPlan,
        output: Buffer<'_, f32>,
        state: Buffer<'_, f32>,
    ) -> Result<()> {
        self.weights
            .kernels
            .recurrent
            .delta_step(
                self.scope,
                plan,
                DeltaBuffers {
                    state: state.into(),
                    output,
                    qkv: self.input_buffer(node.inputs[0])?,
                    beta: self.input_buffer(node.inputs[1])?,
                    alpha: self.input_buffer(node.inputs[2])?,
                    decay: self.input_buffer(node.inputs[3])?,
                    bias: self.input_buffer(node.inputs[4])?,
                    metadata: self.metadata,
                },
            )
            .map_err(error)
    }

    /// Lane-batched convolution decode: one kernel over every lane's pooled history rows.
    fn record_conv_lanes(
        &self,
        node: &TensorNode,
        output: &SlotMemory,
        channels: usize,
    ) -> Result<()> {
        let id = *node
            .states
            .first()
            .ok_or_else(|| error("missing state binding"))?;
        let lanes = self.arena.lanes();
        let state = self
            .states
            .get(&id)
            .ok_or_else(|| error("missing state allocation"))?;
        let [h0, h1, h2] = state.as_slice() else {
            return Err(error("convolution state layout"));
        };
        self.weights
            .kernels
            .recurrent
            .conv(
                self.scope,
                ConvKernel::Lanes { lanes },
                channels,
                ConvBuffers {
                    output: buffer(output)?,
                    history: [buffer(h0)?, buffer(h1)?, buffer(h2)?],
                    input: self.input_buffer(node.inputs[0])?,
                    weight: self.input_buffer(node.inputs[1])?,
                    metadata: self.metadata,
                },
            )
            .map_err(error)
    }

    /// Lane-batched delta decode: one kernel over `[lanes * value_heads]` blocks. The state
    /// root is lane-major, so flattening its head and lane axes addresses every lane's
    /// `[value_heads, key, value]` state without copying.
    fn record_delta_lanes(
        &self,
        node: &TensorNode,
        output: &SlotMemory,
        delta: Delta,
    ) -> Result<()> {
        let id = *node
            .states
            .first()
            .ok_or_else(|| error("missing state binding"))?;
        let lanes = self.arena.lanes();
        let root = self
            .states
            .get(&id)
            .and_then(|state| state.last())
            .ok_or_else(|| error("delta state layout"))?;
        let block = value_block(delta.value_dim)?;
        self.record_delta_step(
            node,
            &DeltaStepPlan {
                key_heads: delta.key_heads,
                value_heads: delta.value_heads,
                dim: delta.key_dim,
                block,
                lanes: Some(lanes),
            },
            buffer(output)?,
            buffer(root)?,
        )
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_recurrent_cell.rs"]
mod cell_tests;
