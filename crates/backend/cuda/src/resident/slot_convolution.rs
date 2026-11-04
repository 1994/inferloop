//! Read-only three-candidate convolution verification across the pool's slot axis.
use super::{
    ActivationArena, ProgramWeights,
    buffers::buffer,
    capture::error,
    slot_recurrent::{LANES, covers},
    slot_states::VerifyBatch,
};
use crate::constants::METADATA_FIELDS;
use cutile::prelude::*;
use infer_cuda_kernels::contract::{ConvBuffers, ConvKernel, Recording};
use infer_ir::{TensorNode, TensorOp};
#[cfg(test)]
use std::cell::Cell;
use std::result::Result as StdResult;

/// Preserve the original four-tap expression and leave accepted-prefix mutation to commit.
pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    arena: &ActivationArena,
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
    let output = arena
        .buffer(index)
        .and_then(|output| output.require("pooled convolution output", slots * lanes * channels))
        .map_err(error)?;
    weights
        .kernels
        .recurrent
        .conv(
            &Recording::scope(scope),
            ConvKernel::Verify {
                slots,
                lanes,
                fields: METADATA_FIELDS,
                fused: lanes == LANES,
            },
            channels,
            ConvBuffers {
                output,
                history: [buffer(oldest)?, buffer(middle)?, buffer(newest)?],
                input: buffer(arena.get(node.inputs[0]).map_err(error)?)?,
                weight: buffer(weight)?,
                metadata: buffer(batch.metadata)?,
            },
        )
        .map_err(error)?;
    #[cfg(test)]
    BUILDS.set(BUILDS.get() + 1);
    Ok(true)
}

#[cfg(test)]
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
