//! A prompt RMS norm fused into the low-precision quantizer of the projection it feeds.
use super::super::norm_lowering::{Consumer, Fusion, Norm, fusion};
use super::super::projection_lowering::Format;
use super::super::{ActivationArena, ProgramWeights, capture::error, nvfp4_gemm::Workspace};
use super::MlpPlan;
use crate::resident::buffers::RowMajor;
use infer_core::Result;
use infer_cuda_kernels::contract::{Buffer, Recording};
use infer_ir::{TensorNode, TensorOp};

/// A norm the prompt plan records fused, with its resolved operands.
#[derive(Clone, Copy)]
pub(in crate::resident) struct FusedNorm<'a> {
    fusion: Fusion,
    norm: Norm,
    /// The consuming projection's node index.
    next: usize,
    input: Buffer<'a, f32>,
    weight: Buffer<'a, f32>,
    output: RowMajor<'a, f32>,
}

impl MlpPlan {
    /// The fused lowering of node `index` when it is a norm whose only consumer is the next
    /// projection, its fusion applies, and the pool's operands and quantizer buffers fit it.
    pub(in crate::resident) fn fused_norm<'a>(
        &self,
        (index, nodes): (usize, &[TensorNode]),
        arena: &'a ActivationArena,
        weights: &'a ProgramWeights,
        workspace: &Workspace,
    ) -> Result<Option<FusedNorm<'a>>> {
        let Some([node, linear]) = nodes.get(index..index + 2) else {
            return Ok(None);
        };
        let TensorOp::Norm {
            head_dim,
            epsilon,
            offset,
        } = node.op
        else {
            return Ok(None);
        };
        let feeds = node.inputs.len() == 2
            && node.outputs.len() == 1
            && node.states.is_empty()
            && linear.states.is_empty()
            && linear.op == TensorOp::Linear
            && linear.inputs.len() == 2
            && linear.outputs.len() == 1
            && linear.inputs[0] == node.outputs[0];
        let Some(projection) = weights.projections.get(&linear.inputs[1]) else {
            return Ok(None);
        };
        if !feeds {
            return Ok(None);
        }
        let norm = Norm {
            dimension: head_dim,
            epsilon,
            offset,
        };
        let consumer = Consumer {
            fuses: weights.attention_profile.is_bf16() && weights.fusion.is_none(),
            format: Format::of(projection),
            activation: weights.activation_quantization(linear.inputs[1]),
            fp8_input: weights.fp8_inputs.contains(&linear.inputs[1]),
            planned_fp4: self.fused.contains_key(&(index + 1)),
        };
        let Some(fusion) = fusion(norm, consumer) else {
            return Ok(None);
        };
        if separate(fusion) || weights.constants.contains_key(&node.inputs[0]) {
            return Ok(None);
        }
        let Some(weight) = weights.constants.get(&node.inputs[1]) else {
            return Ok(None);
        };
        let input = arena.activation(node.inputs[0]).map_err(error)?;
        let output = arena.activation(node.outputs[0]).map_err(error)?;
        let rows = arena.lanes();
        let fits = rows != 0
            && Some(output.len()) == rows.checked_mul(head_dim)
            && input.len() == output.len()
            && weight.len() == head_dim
            && input.disjoint(output)
            && weight.buffer().disjoint(output)
            && match fusion {
                Fusion::Fp4(_) => workspace.norm_fp4_ready(rows, head_dim),
                Fusion::Maximum => workspace.norm_maximum_ready(rows, head_dim),
            };
        if !fits {
            return Ok(None);
        }
        Ok(Some(FusedNorm {
            fusion,
            norm,
            next: index + 1,
            input,
            weight: weight.buffer(),
            output: RowMajor::new(output, rows, head_dim)?,
        }))
    }
}

/// Record `fused`; its quantized output stays valid for exactly the next projection.
/// # Errors
/// A capture failure.
pub(in crate::resident) fn record(
    scope: &Recording<'_>,
    fused: FusedNorm<'_>,
    weights: &ProgramWeights,
    workspace: &mut Workspace,
) -> Result<()> {
    let FusedNorm {
        fusion,
        norm,
        next,
        input,
        weight,
        output,
    } = fused;
    let kernels = (&*weights.kernels, scope);
    match fusion {
        Fusion::Fp4(scale) => {
            workspace.record_norm_fp4(
                kernels,
                next,
                input,
                weight,
                output,
                (norm.epsilon, norm.offset, scale),
            )?;
            #[cfg(test)]
            super::super::norm_fp4::record_capture();
        }
        Fusion::Maximum => {
            workspace.record_norm_maximum(
                kernels,
                next,
                input,
                weight,
                output,
                (norm.epsilon, norm.offset),
            )?;
            #[cfg(test)]
            super::super::norm_maximum::record_capture();
        }
    }
    Ok(())
}

/// Internal tests compare fused prompts against separate normalization.
#[cfg(test)]
fn separate(fusion: Fusion) -> bool {
    match fusion {
        Fusion::Fp4(_) => super::super::norm_fp4::separate(),
        Fusion::Maximum => super::super::norm_maximum::separate(),
    }
}
#[cfg(not(test))]
const fn separate(_: Fusion) -> bool {
    false
}
