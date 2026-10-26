//! RMS normalization primes only the immediately following fused NVFP4 gate/up pass.
use super::program::ActivationQuantization;
use super::{ActivationArena, ProgramWeights, capture::error, nvfp4_gemm::Workspace};
use crate::{
    constants::{F32_BYTES, NVFP4_QUANT_CODES_TILE},
    mlp::ProjectionWeight,
};
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};
#[cfg(test)]
use std::cell::Cell;

/// Full quantizer windows retain the canonical RMS reduction layout. Partial windows use
/// separate normalization: the fused compiler layout can otherwise change F32 rounding.
const QUANT_COLUMNS: usize = NVFP4_QUANT_CODES_TILE[1] * 2;

/// Bound the fused register tile to the measured normalization reduction range.
const MAX_PADDED_COLUMNS: usize = 8192;

pub const fn eligible_columns(columns: usize) -> bool {
    columns > super::capture::NORM_ROWS_MAX_DIMENSION
        && columns <= MAX_PADDED_COLUMNS
        && columns.is_multiple_of(QUANT_COLUMNS)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Binding {
    pub pointer: u64,
    pub rows: usize,
    pub columns: usize,
    pub scale_bits: u32,
    pub codes: u64,
    pub scales: u64,
}
#[derive(Clone, Copy)]
pub(super) struct Pending {
    pub next: usize,
    pub binding: Binding,
}

pub(super) fn record(
    scope: &Scope,
    (index, nodes): (usize, &[TensorNode]),
    arena: &mut ActivationArena,
    weights: &ProgramWeights,
    workspace: &mut Workspace,
) -> Result<bool, DeviceError> {
    #[cfg(test)]
    if DISABLED.get() {
        return Ok(false);
    }
    let Some([norm, linear]) = nodes.get(index..index + 2) else {
        return Ok(false);
    };
    let TensorOp::Norm {
        head_dim,
        epsilon,
        offset,
    } = norm.op
    else {
        return Ok(false);
    };
    // Preserve canonical normalization dispatch; fuse only in the resolved BF16 profile.
    if !weights.attention_profile.is_bf16()
        || weights.fusion.is_some()
        || norm.inputs.len() != 2
        || norm.outputs.len() != 1
        || !norm.states.is_empty()
        || !linear.states.is_empty()
        || linear.op != TensorOp::Linear
        || linear.inputs.len() != 2
        || linear.outputs.len() != 1
        || linear.inputs[0] != norm.outputs[0]
        || !eligible_columns(head_dim)
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || !offset.is_finite()
        || !matches!(
            weights.projections.get(&linear.inputs[1]),
            Some(ProjectionWeight::Fp4(..))
        )
    {
        return Ok(false);
    }
    let Some(ActivationQuantization::Fp4(scale)) =
        weights.activation_quantization(linear.inputs[1])
    else {
        return Ok(false);
    };
    if !scale.is_finite() || scale <= 0.0 || weights.constants.contains_key(&norm.inputs[0]) {
        return Ok(false);
    }
    let Some(weight) = weights.constants.get(&norm.inputs[1]) else {
        return Ok(false);
    };
    let input = arena.get(norm.inputs[0]).map_err(error)?;
    let output = arena.get(norm.outputs[0]).map_err(error)?;
    let rows = arena.lanes();
    if rows == 0
        || Some(output.size()) != rows.checked_mul(head_dim)
        || input.size() != output.size()
        || weight.size() != head_dim
        || !input.is_contiguous()
        || !output.is_contiguous()
        || !weight.is_contiguous()
        || !disjoint(input, output)
        || !disjoint(weight, output)
    {
        return Ok(false);
    }
    let slot = arena.slot(norm.outputs[0]).map_err(error)?;
    let flat = output.size();
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("norm FP4 output slot"))?;
    let mut output = output.reshape(&[rows, head_dim])?;
    let result = (|| {
        let input = arena
            .get(norm.inputs[0])
            .map_err(error)?
            .view(&[rows, head_dim])
            .map_err(error)?;
        workspace.record_norm_fp4(
            scope,
            index + 1,
            &input,
            weight,
            &mut output,
            (epsilon, offset, scale),
        )
    })();
    arena.buffers[slot] = Some(output.reshape(&[flat])?);
    #[cfg(test)]
    if matches!(result, Ok(true)) {
        CAPTURES.set(CAPTURES.get() + 1);
    }
    result
}

fn disjoint(a: &Tensor<f32>, b: &Tensor<f32>) -> bool {
    let span = |tensor: &Tensor<f32>| {
        let start = tensor.device_pointer().cu_deviceptr();
        let bytes = u64::try_from(tensor.size())
            .ok()?
            .checked_mul(u64::try_from(F32_BYTES).ok()?)?;
        Some((start, start.checked_add(bytes)?))
    };
    matches!((span(a), span(b)), (Some(a), Some(b)) if a.1 <= b.0 || b.1 <= a.0)
}

#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
    static CONSUMPTIONS: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.get()
}
#[cfg(test)]
pub fn consumptions() -> usize {
    CONSUMPTIONS.get()
}
#[cfg(test)]
pub(super) fn record_consumption(ready: bool) {
    if ready {
        CONSUMPTIONS.set(CONSUMPTIONS.get() + 1);
    }
}
#[cfg(test)]
pub fn with_separate<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let _restore = Restore(DISABLED.replace(true));
    run()
}
