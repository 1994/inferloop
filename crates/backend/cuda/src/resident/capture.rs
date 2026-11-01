use super::{
    ActivationArena, ProgramWeights,
    attention::attention,
    buffers::{buffer, region},
    program::FusionWeights,
};
use crate::constants::AUX_KERNEL_TILE;
use crate::constants::CB_DECODE_SLOTS;
use crate::constants::DEFAULT_TILE_COLUMNS;
use crate::constants::DEFAULT_TILE_ROWS;
#[cfg(test)]
use crate::constants::GEMV_WIDE_OUTPUT_RATIO;
#[cfg(test)]
use crate::constants::SMALL_GEMM_TILE;
#[cfg(test)]
use crate::mlp::ProjectionWeight;
use crate::strategy::LinearTiling;
use crate::tuning::gated_norm::options as gated_norm_options;
#[cfg(test)]
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use infer_core::TensorId;
use infer_cuda_kernels::contract::{
    BinaryOp, Buffer, FusionNormBuffers, FusionNormPlan, FusionShape, Kernels, NormBuffers,
    NormLayout, NormPlan, Recording, SplitPlan, UnaryOp,
};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::fmt::Display;
use std::sync::Arc;

/// How activation slots are bound for one captured node.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CaptureMode {
    /// Width 1 and verification widths: every slot is one flat `[size]` tensor.
    Flat,
    /// Prefill width 32: every slot is one `[32, size]` tensor used as a whole.
    Batched,
    /// Serving slots retain one normalization policy when physical funding narrows the pool.
    Pooled,
    /// Candidate rows retain the ordinary pool's normalization policy.
    Verification { slots: usize },
    /// Contiguous prompt rows with [position, active rows, state offset, unused] metadata.
    Prefill,
    /// Prefill width 32 with a per-lane `(t, h, w)` rotary table instead of one base position.
    PrefillRotary,
    /// Prefill width 32, one lane at a time: slots stay `[32, size]`, kernels
    /// with per-lane metadata or ordered state read and write single rows.
    Row(usize),
}

pub(super) struct Capture<'a> {
    pub scope: &'a Scope,
    pub arena: &'a mut ActivationArena,
    pub weights: &'a ProgramWeights,
    pub nvfp4: &'a mut super::nvfp4_gemm::Workspace,
    pub attention: &'a mut super::attention_decode::Workspace,
    pub fp8_states: &'a mut super::fp8_cache::Fp8Caches,
    pub states: &'a mut BTreeMap<TensorId, Vec<Tensor<f32>>>,
    pub metadata: &'a Tensor<i32>,
    /// Logical-to-physical KV block ids for the sequence this capture belongs to. Step 1 of the
    /// KV refactor keeps it an identity mapping over the sequence's own cache, so kernels resolve
    /// a position through it while every byte stays where it is today; the allocator of step 2
    /// will hand these blocks out across sequences.
    pub table: &'a Tensor<i32>,
    pub external: &'a Tensor<f32>,
    pub capacity: usize,
    pub fusion: &'a mut Option<super::program::FusionWorkspace>,
    pub mode: CaptureMode,
}

pub(super) fn error(message: impl Display) -> DeviceError {
    DeviceError::Launch(message.to_string())
}

fn identity(tensor: &Tensor<f32>) -> Result<TensorView<'_, f32>, DeviceError> {
    let shape = tensor
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).map_err(error))
        .collect::<Result<Vec<_>, _>>()?;
    tensor.view(&shape).map_err(error)
}

/// A draft's embedding and hidden rows through their fusion norm; the reduction width follows
/// the dimension alone.
fn record_fusion_norm(
    scope: &Scope,
    kernels: &Kernels,
    fusion: &FusionWeights,
    dimension: usize,
    (embedding, hidden, out): (Buffer<'_, f32>, Buffer<'_, f32>, Buffer<'_, f32>),
) -> Result<(), DeviceError> {
    kernels
        .norm
        .fusion(
            &Recording::scope(scope),
            &FusionNormPlan {
                shape: FusionShape::Single,
                dimension,
                epsilon: fusion.epsilon,
                offset: fusion.offset,
            },
            FusionNormBuffers {
                out,
                embedding,
                hidden,
                norms: buffer(fusion.norms.as_ref())?,
            },
        )
        .map_err(error)
}

/// One part of a split, which must sit on a multiple of its power-of-two width.
fn split_plan(widths: &[usize], part: usize, elements: usize) -> Result<SplitPlan, DeviceError> {
    let width = widths[part];
    let offset: usize = widths[..part].iter().sum();
    if !width.is_power_of_two() || !offset.is_multiple_of(width) {
        return Err(error("unsupported device split alignment"));
    }
    Ok(SplitPlan {
        rows: elements / width,
        stride: widths.iter().sum(),
        width,
        part: offset / width,
    })
}

impl Capture<'_> {
    /// Activation inputs resolve per mode; weight constants keep their own shape.
    pub fn input(&self, id: TensorId) -> Result<TensorView<'_, f32>, DeviceError> {
        if let Some(value) = self.weights.constants.get(&id) {
            return identity(value);
        }
        let tensor = self.arena.get(id).map_err(error)?;
        match self.mode {
            CaptureMode::Flat => identity(tensor),
            CaptureMode::Batched
            | CaptureMode::Pooled
            | CaptureMode::Verification { .. }
            | CaptureMode::Prefill
            | CaptureMode::PrefillRotary => tensor.view(&[tensor.size()]).map_err(error),
            CaptureMode::Row(lane) => self.arena.row(id, lane).map_err(error),
        }
    }

    /// [`Self::input`] as a kernel operand: a lane's row sits inside its batched slot.
    pub(super) fn input_buffer(&self, id: TensorId) -> Result<Buffer<'static, f32>, DeviceError> {
        if let Some(value) = self.weights.constants.get(&id) {
            return Ok(region(value.device_pointer().cu_deviceptr(), value.size()));
        }
        let tensor = self.arena.get(id).map_err(error)?;
        let base = tensor.device_pointer().cu_deviceptr();
        let CaptureMode::Row(lane) = self.mode else {
            return Ok(region(base, tensor.size()));
        };
        let row = tensor.size() / self.arena.lanes();
        let offset = u64::try_from(lane * row * size_of::<f32>()).map_err(error)?;
        Ok(region(base + offset, row))
    }

    pub fn record(&mut self, node: &TensorNode) -> Result<(), DeviceError> {
        for (part, id) in node.outputs.iter().enumerate() {
            let index = self.arena.slot(*id).map_err(error)?;
            let output = self.arena.buffers[index]
                .take()
                .ok_or_else(|| error("missing output buffer"))?;
            let restore = output
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim).map_err(error))
                .collect::<Result<Vec<_>, _>>()?;
            let size = output.size();
            #[cfg(test)]
            if self.mode == CaptureMode::Prefill && matches!(node.op, TensorOp::Attention { .. }) {
                let input = |id| {
                    self.weights.constants.get(&id).map_or_else(
                        || self.arena.get(id).map_err(error),
                        |constant| Ok(constant.as_ref()),
                    )
                };
                self.attention.taps.inputs(
                    self.scope,
                    (*id, self.arena.lanes()),
                    [
                        input(node.inputs[0])?,
                        input(node.inputs[1])?,
                        input(node.inputs[2])?,
                    ],
                    (self.metadata, self.table),
                )?;
            }
            let output = self.operation(node, output.reshape(&[size])?, part)?;
            #[cfg(test)]
            if self.mode == CaptureMode::Prefill && matches!(node.op, TensorOp::Attention { .. }) {
                self.attention
                    .taps
                    .output(self.scope, (*id, self.arena.lanes()), &output)?;
            }
            self.arena.buffers[index] = Some(output.reshape(&restore)?);
        }
        Ok(())
    }

    /// Record every node of a program graph, optionally skipping the logits-producing
    /// node (a prefill graph leaves the logits readout to its `_last` twin).
    pub fn record_program(
        &mut self,
        graph: &DataflowGraph,
        skip_logits: bool,
    ) -> Result<(), DeviceError> {
        let quant_plan = (self.mode == CaptureMode::Flat)
            .then(|| super::nvfp4_gemm::QuantPairPlan::new(graph, self.weights));
        self.nvfp4.begin_quant_capture();
        let mut record = || {
            for (index, node) in graph.nodes.iter().enumerate() {
                let skip = skip_logits && node.outputs.iter().any(|id| Some(*id) == graph.logits);
                let role = (self.mode == CaptureMode::Flat && !skip)
                    .then(|| quant_plan.as_ref().and_then(|plan| plan.role(index)))
                    .flatten();
                self.nvfp4.enter_quant_node(index, role);
                if !skip {
                    self.record(node)?;
                }
            }
            Ok(())
        };
        let result = record();
        self.nvfp4.end_quant_capture();
        result
    }

    /// Row-mode twin of [`Self::record`]: kernels write one lane of each batched
    /// slot through an owned row tensor; slots return unaltered.
    pub fn record_row(&mut self, node: &TensorNode, lane: usize) -> Result<(), DeviceError> {
        if self.mode != CaptureMode::Row(lane) {
            return Err(error("row capture mode mismatch"));
        }
        for (part, id) in node.outputs.iter().enumerate() {
            let index = self.arena.slot(*id).map_err(error)?;
            let slot = self.arena.buffers[index]
                .take()
                .ok_or_else(|| error("missing output buffer"))?;
            let (row, shared) =
                super::arena::row_tensor(slot, self.arena.lanes(), lane).map_err(error)?;
            let _ = self.operation(node, row, part)?;
            let slot = Arc::try_unwrap(shared)
                .map_err(|_| error("activation row tensor still aliased"))?;
            self.arena.buffers[index] = Some(slot);
        }
        Ok(())
    }

    fn operation(
        &mut self,
        node: &TensorNode,
        mut output: Tensor<f32>,
        part: usize,
    ) -> Result<Tensor<f32>, DeviceError> {
        match &node.op {
            TensorOp::MoeRoute { .. }
            | TensorOp::ExpertLinear { .. }
            | TensorOp::MoeCombine { .. } => {
                output = self.record_moe(node, output)?;
            }
            TensorOp::Scale => {
                self.weights
                    .kernels
                    .state
                    .scale(
                        &Recording::scope(self.scope),
                        buffer(&output)?,
                        self.input_buffer(node.inputs[0])?,
                        self.input_buffer(node.inputs[1])?,
                    )
                    .map_err(error)?;
            }
            TensorOp::Embedding => {
                self.embedding(node, &mut output)?;
            }
            TensorOp::Linear => {
                output = self.linear(node, output)?;
            }
            TensorOp::Silu | TensorOp::Sigmoid => {
                let op = if node.op == TensorOp::Sigmoid {
                    UnaryOp::Sigmoid
                } else {
                    UnaryOp::Silu
                };
                self.weights
                    .kernels
                    .state
                    .unary(
                        &Recording::scope(self.scope),
                        op,
                        buffer(&output)?,
                        self.input_buffer(node.inputs[0])?,
                    )
                    .map_err(error)?;
            }
            TensorOp::Add | TensorOp::Multiply => {
                let op = if node.op == TensorOp::Multiply {
                    BinaryOp::Multiply
                } else {
                    BinaryOp::Add
                };
                self.weights
                    .kernels
                    .state
                    .binary(
                        &Recording::scope(self.scope),
                        op,
                        buffer(&output)?,
                        self.input_buffer(node.inputs[0])?,
                        self.input_buffer(node.inputs[1])?,
                    )
                    .map_err(error)?;
            }
            TensorOp::Norm {
                head_dim,
                epsilon,
                offset,
            } => {
                output = self.norm(node, output, *head_dim, *epsilon, *offset, false)?;
            }
            TensorOp::GatedNorm { head_dim, epsilon } => {
                output = self.norm(node, output, *head_dim, *epsilon, 0.0, true)?;
            }
            TensorOp::Split { widths, .. } => {
                let plan = split_plan(widths, part, output.size())?;
                output = output.reshape(&[plan.rows, plan.width])?;
                self.record_split(node, buffer(&output)?, &plan)?;
            }
            TensorOp::Rope { .. } => {
                output = self.record_rope(node, output)?;
            }
            TensorOp::Conv { .. } | TensorOp::Delta { .. } | TensorOp::Attention { .. } => {
                output = self.record_state(node, output)?;
            }
        }
        Ok(output)
    }

    fn record_split(
        &self,
        node: &TensorNode,
        out: Buffer<'_, f32>,
        plan: &SplitPlan,
    ) -> Result<(), DeviceError> {
        self.weights
            .kernels
            .state
            .split(
                &Recording::scope(self.scope),
                plan,
                out,
                self.input_buffer(node.inputs[0])?,
            )
            .map_err(error)
    }

    /// Width-1 and verification-width projection: per-lane input tensor.
    /// Prefill width 32 linear nodes never reach here; see `prefill_projection`.
    fn linear(
        &mut self,
        node: &TensorNode,
        mut output: Tensor<f32>,
    ) -> Result<Tensor<f32>, DeviceError> {
        let input = self.weights.constants.get(&node.inputs[0]).map_or_else(
            || self.arena.get(node.inputs[0]).map_err(error),
            |value| Ok(value.as_ref()),
        )?;
        let weights = self
            .weights
            .projections
            .get(&node.inputs[1])
            .ok_or_else(|| error("missing projection"))?;
        if let Some(scale) = self.weights.activation_quantization(node.inputs[1]) {
            let size = output.size();
            let mut matrix = output.reshape(&[1, size])?;
            if !super::nvfp4_gemm::record_with_input_pointer(
                self.scope,
                self.nvfp4,
                weights,
                Some(scale),
                &input.view(&[1, input.size()]).map_err(error)?,
                &mut matrix,
                input.size(),
                input.device_pointer().cu_deviceptr(),
            )? {
                return Err(error(
                    "activation quantization and weight encoding disagree",
                ));
            }
            return Ok(matrix.reshape(&[size])?);
        }
        #[cfg(test)]
        if self.mode == CaptureMode::Flat
            && super::small_gemm::precise_dense_enabled()
            && output.size() <= input.size().saturating_mul(GEMV_WIDE_OUTPUT_RATIO)
            && let ProjectionWeight::Dense(weight) = weights
        {
            let size = output.size();
            let columns = input.size();
            let mut matrix = output.reshape(&[1, size])?;
            self.scope.record(
                super::small_gemm::precise_kernels::dense(
                    (&mut matrix).partition(SMALL_GEMM_TILE),
                    &input.view(&[1, columns]).map_err(error)?,
                    weight,
                )
                .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
            )?;
            return Ok(matrix.reshape(&[size])?);
        }
        weights.record(
            self.scope,
            &mut output,
            input,
            input.size(),
            self.weights.tiling.get(&node.inputs[1]).copied().unwrap_or(
                LinearTiling::new(DEFAULT_TILE_ROWS, DEFAULT_TILE_COLUMNS).map_err(error)?,
            ),
        )?;
        Ok(output)
    }

    fn embedding(
        &mut self,
        node: &TensorNode,
        output: &mut Tensor<f32>,
    ) -> Result<(), DeviceError> {
        let dimension = output.size();
        if self.weights.fusion.is_some() {
            let normalized = self.fusion_input(node, dimension)?;
            let fusion = self
                .weights
                .fusion
                .as_ref()
                .ok_or_else(|| error("missing fusion weights"))?;
            fusion.projection.record(
                self.scope,
                output,
                &normalized,
                2 * dimension,
                fusion.tiling,
            )?;
            self.fusion
                .as_mut()
                .ok_or_else(|| error("missing fusion workspace"))?
                .normalized = Some(normalized);
        } else {
            let weight = self
                .weights
                .embeddings
                .get(&node.inputs[0])
                .ok_or_else(|| error("missing device embedding"))?;
            self.scope.record(
                attention::embedding(
                    output.partition([AUX_KERNEL_TILE]),
                    weight,
                    self.external,
                    self.metadata,
                )
                .generics(vec![dimension.to_string(), "1".into()]),
            )?;
        }
        Ok(())
    }

    /// Original row embedding and fusion normalization, before its projection.
    pub(super) fn fusion_input(
        &mut self,
        node: &TensorNode,
        dimension: usize,
    ) -> Result<Tensor<f32>, DeviceError> {
        let weight = self
            .weights
            .embeddings
            .get(&node.inputs[0])
            .ok_or_else(|| error("missing device embedding"))?;
        let fusion = self
            .weights
            .fusion
            .as_ref()
            .ok_or_else(|| error("missing fusion weights"))?;
        let work = self
            .fusion
            .as_mut()
            .ok_or_else(|| error("missing fusion workspace"))?;
        self.scope.record(
            attention::embedding(
                (&mut work.embedding).partition([AUX_KERNEL_TILE]),
                weight,
                self.external,
                self.metadata,
            )
            .generics(vec![dimension.to_string(), "0".into()]),
        )?;
        let normalized = work
            .normalized
            .take()
            .ok_or_else(|| error("fusion buffer"))?
            .reshape(&[2, dimension])?;
        record_fusion_norm(
            self.scope,
            &self.weights.kernels,
            fusion,
            dimension,
            (
                buffer(&work.embedding)?,
                buffer(self.external)?,
                buffer(&normalized)?,
            ),
        )?;
        Ok(normalized.reshape(&[2 * dimension])?)
    }

    /// Fusion normalization alone, reading an already gathered embedding row.
    /// The arithmetic and reduction width are identical to `fusion_input`.
    pub(super) fn fusion_norm(
        &mut self,
        dimension: usize,
        embedding: &Tensor<f32>,
    ) -> Result<Tensor<f32>, DeviceError> {
        let fusion = self
            .weights
            .fusion
            .as_ref()
            .ok_or_else(|| error("missing fusion weights"))?;
        let work = self
            .fusion
            .as_mut()
            .ok_or_else(|| error("missing fusion workspace"))?;
        let normalized = work
            .normalized
            .take()
            .ok_or_else(|| error("fusion buffer"))?
            .reshape(&[2, dimension])?;
        record_fusion_norm(
            self.scope,
            &self.weights.kernels,
            fusion,
            dimension,
            (
                buffer(embedding)?,
                buffer(self.external)?,
                buffer(&normalized)?,
            ),
        )?;
        Ok(normalized.reshape(&[2 * dimension])?)
    }

    fn norm(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
        dimension: usize,
        epsilon: f32,
        offset: f32,
        gated: bool,
    ) -> Result<Tensor<f32>, DeviceError> {
        if dimension == 0 || !output.size().is_multiple_of(dimension) {
            return Err(error("norm dimensions"));
        }
        let heads = output.size() / dimension;
        let output = output.reshape(&[heads, dimension])?;
        let options = if gated {
            gated_norm_options(output.device_id(), dimension, heads, epsilon)
        } else {
            CompileOptions::default()
        };
        #[cfg(test)]
        let options = super::gated_norm_discovery::model_options(dimension, gated, heads, options);
        // Physical funding must not change the automatic target's arithmetic.
        // Match the established full serving pool even when fewer slots fit.
        // The row kernel masks its final partial tile.
        let normalization_heads = self.normalization_heads(heads)?;
        let layout = if dimension.is_power_of_two()
            && dimension <= NORM_ROWS_MAX_DIMENSION
            && normalization_heads >= NORM_ROWS * NORM_ROWS
            && normalization_heads.is_multiple_of(NORM_ROWS)
        {
            NormLayout::Rows { tile: NORM_ROWS }
        } else {
            NormLayout::Block
        };
        self.weights
            .kernels
            .norm
            .norm(
                &Recording::scope(self.scope),
                &NormPlan {
                    layout,
                    rows: heads,
                    dimension,
                    gated,
                    epsilon,
                    offset,
                    occupancy: options.occupancy,
                },
                NormBuffers {
                    out: buffer(&output)?,
                    x: self.input_buffer(node.inputs[0])?,
                    weight: self.input_buffer(node.inputs[if gated { 2 } else { 1 }])?,
                    gate: self.input_buffer(node.inputs[usize::from(gated)])?,
                },
            )
            .map_err(error)?;
        Ok(output)
    }

    /// Kernel selection follows logical serving rows, independently of candidate padding.
    fn normalization_heads(&self, heads: usize) -> Result<usize, DeviceError> {
        let bf16_pool = self.weights.fusion.is_none() && self.weights.attention_profile.is_bf16();
        let reference_slots = match self.mode {
            CaptureMode::Verification { slots } => {
                if bf16_pool {
                    CB_DECODE_SLOTS
                } else {
                    slots
                }
            }
            CaptureMode::Pooled if bf16_pool => CB_DECODE_SLOTS,
            _ => return Ok(heads),
        };
        let rows = self.arena.lanes();
        if rows == 0 || !heads.is_multiple_of(rows) {
            return Err(error("pooled normalization geometry"));
        }
        Ok((heads / rows) * reference_slots)
    }
}

/// Rows per block of the batched norm (fixed in its kernel), and the widest row it takes.
const NORM_ROWS: usize = 16;
pub(super) const NORM_ROWS_MAX_DIMENSION: usize = 256;
