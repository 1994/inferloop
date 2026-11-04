//! Prime only the immediately following token-FP8 projection's row maximum.
use super::{ActivationArena, ProgramWeights, capture::error, nvfp4_gemm::Workspace};
use crate::mlp::ProjectionWeight;
use crate::resident::buffers::RowMajor;
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};
#[cfg(test)]
use std::cell::Cell;

#[derive(Clone, Copy)]
pub struct Pending {
    pub next: usize,
    pub pointer: u64,
    pub rows: usize,
    pub columns: usize,
}
impl Pending {
    pub fn matches(self, pointer: Option<u64>, rows: usize, columns: usize) -> bool {
        pointer == Some(self.pointer) && rows == self.rows && columns == self.columns
    }
}

pub(super) fn record(
    scope: &Scope,
    index: usize,
    nodes: &[TensorNode],
    arena: &ActivationArena,
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
        || !super::fp8_quant_reduction::parallel_columns(head_dim)
        || !matches!(
            weights.projections.get(&linear.inputs[1]),
            Some(ProjectionWeight::Fp8(..))
        )
        || !weights.fp8_inputs.contains(&linear.inputs[1])
    {
        return Ok(false);
    }
    if weights.constants.contains_key(&norm.inputs[0]) {
        return Ok(false);
    }
    let Some(weight) = weights.constants.get(&norm.inputs[1]) else {
        return Ok(false);
    };
    let input = arena.activation(norm.inputs[0]).map_err(error)?;
    let output = arena.activation(norm.outputs[0]).map_err(error)?;
    let rows = arena.lanes();
    if rows == 0
        || Some(output.len()) != rows.checked_mul(head_dim)
        || input.len() != output.len()
        || weight.len() != head_dim
        || !input.disjoint(output)
        || !weight.buffer().disjoint(output)
    {
        return Ok(false);
    }
    let output = RowMajor::new(output, rows, head_dim)?;
    let result = workspace.record_norm_maximum(
        (&weights.kernels, scope),
        index + 1,
        input,
        weight.buffer(),
        output,
        (epsilon, offset),
    );
    #[cfg(test)]
    if matches!(result, Ok(true)) {
        CAPTURES.set(CAPTURES.get() + 1);
    }
    result
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
