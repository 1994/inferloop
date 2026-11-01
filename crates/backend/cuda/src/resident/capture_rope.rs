use super::{
    buffers::buffer,
    capture::{Capture, CaptureMode, error},
};
use crate::constants::METADATA_FIELDS;
use cutile::prelude::*;
use infer_cuda_kernels::contract::{Recording, RopeBuffers, RopeMode, RopePlan};
use infer_ir::{TensorNode, TensorOp};

impl Capture<'_> {
    pub(super) fn record_rope(
        &self,
        node: &TensorNode,
        mut output: Tensor<f32>,
    ) -> Result<Tensor<f32>, DeviceError> {
        let TensorOp::Rope {
            heads,
            head_dim,
            rotary_dim,
            ..
        } = &node.op
        else {
            return Err(error("expected rotary operation"));
        };
        let half = rotary_dim / 2;
        if !half.is_power_of_two() || !head_dim.is_multiple_of(half) {
            return Err(error("unsupported rotary dimensions"));
        }
        let rows = if matches!(self.mode, CaptureMode::Prefill | CaptureMode::PrefillRotary) {
            self.arena.lanes() * heads
        } else {
            *heads
        };
        output = output.reshape(&[rows, *head_dim])?;
        let frequency = self
            .weights
            .rope_frequencies
            .get(&node.id)
            .ok_or_else(|| error("missing rotary frequencies"))?;
        let axes = self.weights.rope_axes.get(&node.id);
        let (mode, axes) = match self.mode {
            CaptureMode::PrefillRotary => (
                RopeMode::PromptAxes,
                Some(axes.ok_or_else(|| error("missing rotary axes"))?),
            ),
            CaptureMode::Prefill => (RopeMode::Prompt, None),
            _ if self.metadata.size() == METADATA_FIELDS * 2 && axes.is_some() => {
                (RopeMode::SingleAxes, axes)
            }
            _ => (RopeMode::Single, None),
        };
        let plan = RopePlan {
            mode,
            rows,
            heads: *heads,
            head_dim: *head_dim,
            half,
        };
        self.weights
            .kernels
            .rope
            .rope(
                &Recording::scope(self.scope),
                &plan,
                RopeBuffers {
                    out: buffer(&output)?,
                    x: self.input_buffer(node.inputs[0])?,
                    frequencies: buffer(frequency)?,
                    axes: axes.map(|axes| buffer(axes)).transpose()?,
                    metadata: buffer(self.metadata)?,
                },
            )
            .map_err(error)?;
        Ok(output)
    }
}
