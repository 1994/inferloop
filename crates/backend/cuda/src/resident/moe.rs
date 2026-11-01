//! Device-resident routing and selected-expert projections.
use super::buffers::buffer;
use super::capture::{Capture, error};
use crate::constants::FP8_BLOCK_COLUMNS;
use crate::mlp::ProjectionWeight;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{ExpertPlan, ExpertWeight, Recording, RoutePlan};
use infer_ir::{TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::result::Result as StdResult;

/// Existing supported routing reduction extent; this is a shape legality bound.
const MAX_ROUTING_EXPERTS: usize = 4096;

/// Contiguous expert-major matrices. Routing indexes this bank on the GPU.
pub struct ExpertBank {
    pub projection: ProjectionWeight,
    pub experts: usize,
    pub rows: usize,
    pub columns: usize,
    /// Follow the checkpoint activation quantization contract for selected experts.
    pub quantized_inputs: bool,
}

impl ExpertBank {
    pub(crate) fn validate(&self, experts: usize, rows: usize, columns: usize) -> Result<()> {
        if (self.experts, self.rows, self.columns) != (experts, rows, columns) {
            return Err(Error::invalid("expert bank geometry"));
        }
        self.projection.validate(
            experts
                .checked_mul(rows)
                .ok_or_else(|| Error::invalid("expert bank overflow"))?,
            columns,
        )
    }
}

impl Capture<'_> {
    pub(super) fn record_moe(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
    ) -> StdResult<Tensor<f32>, DeviceError> {
        match node.op {
            TensorOp::MoeRoute {
                experts,
                active,
                normalize,
            } => {
                self.weights
                    .kernels
                    .moe
                    .route(
                        &Recording::scope(self.scope),
                        &RoutePlan {
                            experts,
                            active,
                            normalize,
                        },
                        buffer(&output)?,
                        self.input_buffer(node.inputs[0])?,
                    )
                    .map_err(error)?;
                Ok(output)
            }
            TensorOp::MoeCombine { active, hidden } => {
                self.weights
                    .kernels
                    .moe
                    .combine(
                        &Recording::scope(self.scope),
                        (active, hidden),
                        buffer(&output)?,
                        self.input_buffer(node.inputs[0])?,
                        self.input_buffer(node.inputs[1])?,
                    )
                    .map_err(error)?;
                Ok(output)
            }
            TensorOp::ExpertLinear {
                experts: _,
                active,
                input,
                output: rows,
                routed_input,
            } => self.record_experts(node, output, active, input, rows, routed_input),
            _ => Err(error("MoE capture operation")),
        }
    }
    fn record_experts(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
        active: usize,
        input: usize,
        rows: usize,
        routed_input: bool,
    ) -> StdResult<Tensor<f32>, DeviceError> {
        let bank = self
            .weights
            .expert_banks
            .get(&node.inputs[2])
            .ok_or_else(|| error("missing expert bank"))?;
        let weight = match &bank.projection {
            ProjectionWeight::Dense(w) => ExpertWeight::Dense(buffer(w.as_ref())?),
            ProjectionWeight::Fp8(w, scales) => ExpertWeight::Fp8Channel {
                weight: buffer(w.as_ref())?,
                scales: buffer(scales.as_ref())?,
                quantized_inputs: bank.quantized_inputs,
            },
            ProjectionWeight::Fp8Block(w, scales) => ExpertWeight::Fp8Block {
                weight: buffer(w.as_ref())?,
                scales: buffer(scales.as_ref())?,
                block_columns: FP8_BLOCK_COLUMNS,
                quantized_inputs: bank.quantized_inputs,
            },
            ProjectionWeight::Fp4(..) => {
                return Err(error(
                    "NVFP4 expert activation quantization requires an explicit numerical contract",
                ));
            }
        };
        let output = output.reshape(&[active, rows])?;
        self.weights
            .kernels
            .moe
            .experts(
                &Recording::scope(self.scope),
                &ExpertPlan {
                    experts: bank.experts,
                    active,
                    rows,
                    input,
                    routed_input,
                },
                buffer(&output)?,
                self.input_buffer(node.inputs[0])?,
                self.input_buffer(node.inputs[1])?,
                weight,
            )
            .map_err(error)?;
        Ok(output.reshape(&[active * rows])?)
    }
}

pub(super) fn validate_node(
    node: &TensorNode,
    specs: &BTreeMap<infer_core::TensorId, &infer_ir::TensorSpec>,
    weights: &super::ProgramWeights,
    input: usize,
    output: usize,
) -> Result<()> {
    if let TensorOp::ExpertLinear {
        experts,
        active,
        input: columns,
        output: rows,
        routed_input,
    } = node.op
    {
        if experts == 0
            || active == 0
            || active > experts
            || columns == 0
            || rows == 0
            || columns.checked_mul(if routed_input { active } else { 1 }) != Some(input)
            || active.checked_mul(rows) != Some(output)
            || active.checked_mul(2) != Some(specs[&node.inputs[1]].elements()?)
        {
            return Err(Error::invalid("expert projection geometry"));
        }
        weights
            .expert_banks
            .get(&node.inputs[2])
            .ok_or_else(|| Error::invalid("missing expert bank"))?
            .validate(experts, rows, columns)?;
    }
    if let TensorOp::MoeRoute {
        experts, active, ..
    } = node.op
        && (experts == 0
            || experts > MAX_ROUTING_EXPERTS
            || active == 0
            || active > experts
            || input != experts
            || active.checked_mul(2) != Some(output))
    {
        return Err(Error::invalid("routing geometry"));
    }
    if let TensorOp::MoeCombine { active, hidden } = node.op
        && (active == 0
            || hidden == 0
            || active.checked_mul(hidden) != Some(input)
            || output != hidden
            || active.checked_mul(2) != Some(specs[&node.inputs[1]].elements()?))
    {
        return Err(Error::invalid("expert combine geometry"));
    }
    if node.op == TensorOp::Scale && (input != output || specs[&node.inputs[1]].elements()? != 1) {
        return Err(Error::invalid("broadcast scale geometry"));
    }
    Ok(())
}

pub use infer_cuda_kernels::tile::moe::moe as kernels;

#[cfg(test)]
#[path = "../../tests/unit/moe.rs"]
mod tests;
