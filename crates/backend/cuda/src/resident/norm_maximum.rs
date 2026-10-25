//! Prime only the immediately following token-FP8 projection's row maximum.
use super::{ActivationArena, ProgramWeights, capture::error, nvfp4_gemm::Workspace};
use crate::{constants::F32_BYTES, mlp::ProjectionWeight};
use cutile::prelude::*;
use infer_core::TensorId;
use infer_ir::{TensorNode, TensorOp};
#[cfg(test)]
use std::cell::Cell;

/// Add, Norm and the following FP8 linear.
const ADD_NORM_NODES: usize = 3;

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

/// A residual Add is fusible only when its sole consumer is the immediately following Norm,
/// and it is not a graph-hidden/logit output. Extra readers must keep the separate Add.
pub(super) fn add_norm_prefix(
    nodes: &[TensorNode],
    index: usize,
    protected: [Option<TensorId>; 2],
) -> bool {
    let Some([add, norm, _linear]) = nodes.get(index..index + ADD_NORM_NODES) else {
        return false;
    };
    add.op == TensorOp::Add
        && add.inputs.len() == 2
        && add.outputs.len() == 1
        && add.states.is_empty()
        && !protected.contains(&Some(add.outputs[0]))
        && nodes
            .iter()
            .filter(|node| node.inputs.contains(&add.outputs[0]))
            .count()
            == 1
        && matches!(norm.op, TensorOp::Norm { .. })
        && norm.inputs.first() == Some(&add.outputs[0])
}

/// Three outcomes of attempting a prompt norm fusion. `Handled` must still skip the
/// current Norm node; only `NotHandled` falls through to ordinary recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PromptRecord {
    NotHandled,
    Handled,
    HandledSkip(usize),
}

/// What the capture loop must do after a norm-fusion attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PromptControl {
    Ordinary,
    Continue,
    Skip(usize),
}

const fn prompt_record(add: bool, recorded: bool, index: usize) -> PromptRecord {
    match (add, recorded) {
        (_, false) => PromptRecord::NotHandled,
        (true, true) => PromptRecord::HandledSkip(index + 1),
        (false, true) => PromptRecord::Handled,
    }
}

pub(super) const fn apply_control(control: PromptControl, skip: &mut Option<usize>) -> bool {
    match control {
        PromptControl::Ordinary => false,
        PromptControl::Continue => true,
        PromptControl::Skip(next) => {
            *skip = Some(next);
            true
        }
    }
}

pub(super) const fn prompt_control(record: PromptRecord) -> PromptControl {
    match record {
        PromptRecord::NotHandled => PromptControl::Ordinary,
        PromptRecord::Handled => PromptControl::Continue,
        PromptRecord::HandledSkip(next) => PromptControl::Skip(next),
    }
}

/// Record add+norm, norm+max, or nothing. `HandledSkip(next)` means the caller also
/// consumes the following Norm node; `Handled` means only the current Norm is consumed.
pub(super) fn record_prompt(
    scope: &Scope,
    index: usize,
    nodes: &[TensorNode],
    arena: &mut ActivationArena,
    weights: &ProgramWeights,
    workspace: &mut Workspace,
    protected: [Option<TensorId>; 2],
) -> Result<PromptRecord, DeviceError> {
    let add = add_norm_prefix(nodes, index, protected);
    let recorded = record(scope, index, nodes, arena, weights, workspace, protected)?;
    Ok(prompt_record(add, recorded, index))
}

fn prefix(
    nodes: &[TensorNode],
    index: usize,
    protected: [Option<TensorId>; 2],
) -> Option<(usize, &TensorNode, Option<&TensorNode>)> {
    if nodes[index].op == TensorOp::Add {
        if !add_norm_prefix(nodes, index, protected) {
            return None;
        }
        let [add, _norm, linear] = &nodes[index..index + ADD_NORM_NODES] else {
            return None;
        };
        Some((index + 1, linear, Some(add)))
    } else {
        let [_norm, linear] = nodes.get(index..index + 2)? else {
            return None;
        };
        Some((index, linear, None))
    }
}

fn supported(
    norm: &TensorNode,
    linear: &TensorNode,
    weights: &ProgramWeights,
    head_dim: usize,
) -> bool {
    weights.attention_profile.is_bf16()
        && weights.fusion.is_none()
        && norm.inputs.len() == 2
        && norm.outputs.len() == 1
        && norm.states.is_empty()
        && linear.states.is_empty()
        && linear.op == TensorOp::Linear
        && linear.inputs.len() == 2
        && linear.outputs.len() == 1
        && linear.inputs[0] == norm.outputs[0]
        && super::fp8_quant_reduction::parallel_columns(head_dim)
        && matches!(
            weights.projections.get(&linear.inputs[1]),
            Some(ProjectionWeight::Fp8(..))
        )
        && weights.fp8_inputs.contains(&linear.inputs[1])
}

pub(super) fn record(
    scope: &Scope,
    index: usize,
    nodes: &[TensorNode],
    arena: &mut ActivationArena,
    weights: &ProgramWeights,
    workspace: &mut Workspace,
    protected: [Option<TensorId>; 2],
) -> Result<bool, DeviceError> {
    #[cfg(test)]
    if DISABLED.get() {
        return Ok(false);
    }
    let Some((norm_index, linear, add)) = prefix(nodes, index, protected) else {
        return Ok(false);
    };
    let norm = &nodes[norm_index];
    let TensorOp::Norm {
        head_dim,
        epsilon,
        offset,
    } = norm.op
    else {
        return Ok(false);
    };
    // Preserve canonical normalization dispatch; fuse only in the resolved BF16 profile.
    if !supported(norm, linear, weights, head_dim) {
        return Ok(false);
    }
    if weights.constants.contains_key(&norm.inputs[0]) {
        return Ok(false);
    }
    let Some(weight) = weights.constants.get(&norm.inputs[1]) else {
        return Ok(false);
    };
    let (input_id, residual_id) = add.map_or((norm.inputs[0], None), |add| {
        (add.inputs[0], Some(add.inputs[1]))
    });
    let input = arena.get(input_id).map_err(error)?;
    let residual = residual_id
        .map(|id| arena.get(id).map_err(error))
        .transpose()?;
    let output = arena.get(norm.outputs[0]).map_err(error)?;
    let rows = arena.lanes();
    if rows == 0
        || Some(output.size()) != rows.checked_mul(head_dim)
        || input.size() != output.size()
        || residual.is_some_and(|residual| residual.size() != output.size())
        || weight.size() != head_dim
        || !input.is_contiguous()
        || residual.is_some_and(|residual| !residual.is_contiguous())
        || !output.is_contiguous()
        || !weight.is_contiguous()
        || !disjoint(input, output)
        || residual.is_some_and(|residual| !disjoint(residual, output))
        || !disjoint(weight, output)
    {
        return Ok(false);
    }
    let slot = arena.slot(norm.outputs[0]).map_err(error)?;
    let flat = output.size();
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("norm maximum output slot"))?;
    let mut output = output.reshape(&[rows, head_dim])?;
    let result = (|| {
        let input = arena
            .get(input_id)
            .map_err(error)?
            .view(&[rows, head_dim])
            .map_err(error)?;
        if let Some(residual_id) = residual_id {
            let residual = arena
                .get(residual_id)
                .map_err(error)?
                .view(&[rows, head_dim])
                .map_err(error)?;
            workspace.record_add_norm_maximum(
                scope,
                norm_index + 1,
                &input,
                &residual,
                weight,
                &mut output,
                (epsilon, offset),
            )
        } else {
            workspace.record_norm_maximum(
                scope,
                norm_index + 1,
                &input,
                weight,
                &mut output,
                (epsilon, offset),
            )
        }
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

#[cfg(test)]
#[path = "../../tests/unit/resident_norm_maximum_fusion.rs"]
mod fusion_tests;
