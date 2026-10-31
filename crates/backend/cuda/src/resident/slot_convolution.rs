//! Read-only three-candidate convolution verification across the pool's slot axis.
use super::{
    ActivationArena, ProgramWeights,
    capture::error,
    slot_recurrent::{LANES, covers},
    slot_states::VerifyBatch,
};
use crate::constants::{CONV_KERNEL_SIZE, CONV_KERNEL_TILE, METADATA_FIELDS};
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};
#[cfg(test)]
use std::cell::Cell;
use std::result::Result as StdResult;

/// Preserve the original four-tap expression and leave accepted-prefix mutation to commit.
pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    batch: &VerifyBatch<'_>,
    weights: &ProgramWeights,
) -> StdResult<bool, DeviceError> {
    #[cfg(test)]
    if DISABLED.get() {
        return Ok(false);
    }
    let TensorOp::Conv { channels, .. } = node.op else {
        return Ok(false);
    };
    if !covers(node) || weights.constants.contains_key(&node.inputs[0]) {
        return Ok(false);
    }
    let Some(weight) = weights.constants.get(&node.inputs[1]) else {
        return Ok(false);
    };
    let state = batch
        .roots
        .get(&node.states[0])
        .ok_or_else(|| error("pooled convolution owner"))?;
    let [oldest, middle, newest] = state.as_slice() else {
        return Err(error("pooled convolution state layout"));
    };
    let lanes = batch.lanes;
    let slots = arena.lanes() / lanes;
    let index = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[index]
        .take()
        .ok_or_else(|| error("pooled convolution output"))?;
    let restore = output
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).map_err(error))
        .collect::<StdResult<Vec<_>, _>>()?;
    let mut output = output.reshape(&[slots, lanes, channels])?;
    let recorded = (|| {
        let input = arena
            .get(node.inputs[0])
            .map_err(error)?
            .view(&[slots, lanes, channels])?;
        let (h0, h1, h2) = (
            oldest.view(&[slots, channels])?,
            middle.view(&[slots, channels])?,
            newest.view(&[slots, channels])?,
        );
        let weight = weight.view(&[channels, CONV_KERNEL_SIZE])?;
        if lanes == LANES {
            scope.record(
                kernels::verify(
                    (&mut output).partition([1, LANES, CONV_KERNEL_TILE]),
                    &input,
                    &h0,
                    &h1,
                    &h2,
                    &weight,
                    batch.metadata,
                )
                .generics(
                    [channels, METADATA_FIELDS]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )?;
        } else {
            scope.record(
                kernels::verify_lanes(
                    (&mut output).partition([1, lanes, CONV_KERNEL_TILE]),
                    &input,
                    &h0,
                    &h1,
                    &h2,
                    &weight,
                    batch.metadata,
                )
                .generics(
                    [channels, METADATA_FIELDS, lanes]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )?;
        }
        #[cfg(test)]
        BUILDS.set(BUILDS.get() + 1);
        Ok(true)
    })();
    arena.buffers[index] = Some(output.reshape(&restore)?);
    recorded
}

pub use infer_cuda_kernels::tile::recurrent::slot_convolution as kernels;

#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static BUILDS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub fn builds() -> usize {
    BUILDS.get()
}

/// Compare with the original per-slot captures without adding a production configuration.
#[cfg(test)]
pub fn with_row_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let restore = Restore(DISABLED.replace(true));
    let value = run();
    drop(restore);
    value
}

#[cfg(test)]
#[path = "../../tests/unit/resident_slot_convolution.rs"]
mod tests;
