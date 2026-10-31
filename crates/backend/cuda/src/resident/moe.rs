//! Device-resident routing and selected-expert projections.
use super::capture::{Capture, error};
use crate::constants::AUX_KERNEL_TILE;
use crate::constants::FP8_BLOCK_COLUMNS;
use crate::mlp::ProjectionWeight;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::{TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::result::Result as StdResult;

/// Matches the fixed output Tensor tile declared by all selected-expert kernel entries.
const EXPERT_PROJECTION_TILE: [usize; 2] = [1, 4];
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
                let mut output = output;
                self.scope.record(
                    kernels::route(
                        (&mut output).partition([1]),
                        &self.input(node.inputs[0])?,
                        f32::NEG_INFINITY,
                    )
                    .generics(vec![
                        experts.to_string(),
                        experts.next_power_of_two().to_string(),
                        active.to_string(),
                        i32::from(normalize).to_string(),
                    ]),
                )?;
                Ok(output)
            }
            TensorOp::MoeCombine { active, hidden } => {
                let mut output = output;
                self.scope.record(
                    kernels::combine(
                        (&mut output).partition([AUX_KERNEL_TILE]),
                        &self
                            .input(node.inputs[0])?
                            .view(&[active, hidden])
                            .map_err(error)?,
                        &self.input(node.inputs[1])?,
                    )
                    .generics(vec![active.to_string(), hidden.to_string()]),
                )?;
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
        let x = self.input(node.inputs[0])?;
        let x = x
            .view(&[if routed_input { active } else { 1 }, input])
            .map_err(error)?;
        let routes = self.input(node.inputs[1])?;
        let mut output = output.reshape(&[active, rows])?;
        let mut generics = vec![
            rows.to_string(),
            input.to_string(),
            i32::from(routed_input).to_string(),
        ];
        match &bank.projection {
            ProjectionWeight::Dense(w) => {
                self.scope.record(
                    kernels::dense(
                        (&mut output).partition(EXPERT_PROJECTION_TILE),
                        &x,
                        &routes,
                        &w.view(&[bank.experts, rows, input]).map_err(error)?,
                    )
                    .generics(generics),
                )?;
            }
            ProjectionWeight::Fp8(w, scales) => {
                generics.extend([
                    "1".into(),
                    i32::from(bank.quantized_inputs).to_string(),
                    input.next_power_of_two().to_string(),
                ]);
                self.scope.record(
                    kernels::fp8(
                        (&mut output).partition(EXPERT_PROJECTION_TILE),
                        &x,
                        &routes,
                        &w.view(&[bank.experts, rows, input]).map_err(error)?,
                        &scales.view(&[bank.experts, rows, 1]).map_err(error)?,
                    )
                    .generics(generics),
                )?;
            }
            ProjectionWeight::Fp8Block(w, scales) => {
                generics.extend([
                    "0".into(),
                    i32::from(bank.quantized_inputs).to_string(),
                    input.next_power_of_two().to_string(),
                ]);
                self.scope.record(
                    kernels::fp8(
                        (&mut output).partition(EXPERT_PROJECTION_TILE),
                        &x,
                        &routes,
                        &w.view(&[bank.experts, rows, input]).map_err(error)?,
                        &scales
                            .view(&[bank.experts, rows, input / FP8_BLOCK_COLUMNS])
                            .map_err(error)?,
                    )
                    .generics(generics),
                )?;
            }
            ProjectionWeight::Fp4(..) => {
                return Err(error(
                    "NVFP4 expert activation quantization requires an explicit numerical profile",
                ));
            }
        }
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
