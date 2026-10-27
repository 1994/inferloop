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

#[cutile::module]
pub(super) mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, Tensor_1, Tensor_2, Tensor_3, Tile_1,
        exp, ge_tile, get_tile_block_id, select,
    };

    fn active<const MF: i32>(metadata: &Tensor<i32, { [-1] }>, row: i32) -> Tile<bool, { [128] }> {
        let position = metadata.partition(shape![1]).load([row * MF + 2i32]);
        ge_tile(position, 0i32.broadcast(shape![1])).broadcast(shape![128])
    }

    /// [`verify`] for `L` candidate lanes per slot, sliding the window over the lanes.
    #[cutile::entry()]
    fn verify_lanes<const C: i32, const MF: i32, const L: i32>(
        out: &mut Tensor<f32, { [1, L, 128] }>,
        input: &Tensor<f32, { [-1, L, C] }>,
        h0: &Tensor<f32, { [-1, C] }>,
        h1: &Tensor<f32, { [-1, C] }>,
        h2: &Tensor<f32, { [-1, C] }>,
        weight: &Tensor<f32, { [-1, 4] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let mut oldest = h0
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let mut middle = h1
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let mut newest = h2
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let inputs = input.partition(shape![1, 1, 128]);
        let weights = weight.partition(shape![128, 1]);
        let wa = weights.load([pid.2, 0i32]).reshape(shape![128]);
        let wb = weights.load([pid.2, 1i32]).reshape(shape![128]);
        let wc = weights.load([pid.2, 2i32]).reshape(shape![128]);
        let wd = weights.load([pid.2, 3i32]).reshape(shape![128]);
        let one = 1.0f32.broadcast(shape![128]);
        let zero = 0.0f32.broadcast(shape![128]);
        for lane in 0i32..L {
            let current = inputs.load([pid.0, lane, pid.2]).reshape(shape![128]);
            let value = oldest * wa + middle * wb + newest * wc + current * wd;
            let silu = value / (one + exp(zero - value));
            out.partition_mut(shape![1, 1, 128]).store(
                select(active::<MF>(metadata, pid.0 * L + lane), silu, zero)
                    .reshape(shape![1, 1, 128]),
                [0i32, lane, 0i32],
            );
            oldest = middle;
            middle = newest;
            newest = current;
        }
    }

    #[cutile::entry()]
    fn verify<const C: i32, const MF: i32>(
        out: &mut Tensor<f32, { [1, 3, 128] }>,
        input: &Tensor<f32, { [-1, 3, C] }>,
        h0: &Tensor<f32, { [-1, C] }>,
        h1: &Tensor<f32, { [-1, C] }>,
        h2: &Tensor<f32, { [-1, C] }>,
        weight: &Tensor<f32, { [-1, 4] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let oldest = h0
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let middle = h1
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let newest = h2
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let inputs = input.partition(shape![1, 1, 128]);
        let first = inputs.load([pid.0, 0i32, pid.2]).reshape(shape![128]);
        let second = inputs.load([pid.0, 1i32, pid.2]).reshape(shape![128]);
        let third = inputs.load([pid.0, 2i32, pid.2]).reshape(shape![128]);
        let weights = weight.partition(shape![128, 1]);
        let wa = weights.load([pid.2, 0i32]).reshape(shape![128]);
        let wb = weights.load([pid.2, 1i32]).reshape(shape![128]);
        let wc = weights.load([pid.2, 2i32]).reshape(shape![128]);
        let wd = weights.load([pid.2, 3i32]).reshape(shape![128]);
        let one = 1.0f32.broadcast(shape![128]);
        let zero = 0.0f32.broadcast(shape![128]);
        let value0 = oldest * wa + middle * wb + newest * wc + first * wd;
        let value1 = middle * wa + newest * wb + first * wc + second * wd;
        let value2 = newest * wa + first * wb + second * wc + third * wd;
        let silu0 = value0 / (one + exp(zero - value0));
        let silu1 = value1 / (one + exp(zero - value1));
        let silu2 = value2 / (one + exp(zero - value2));
        out.partition_mut(shape![1, 1, 128]).store(
            select(active::<MF>(metadata, pid.0 * 3i32), silu0, zero).reshape(shape![1, 1, 128]),
            [0i32, 0i32, 0i32],
        );
        out.partition_mut(shape![1, 1, 128]).store(
            select(active::<MF>(metadata, pid.0 * 3i32 + 1i32), silu1, zero)
                .reshape(shape![1, 1, 128]),
            [0i32, 1i32, 0i32],
        );
        out.partition_mut(shape![1, 1, 128]).store(
            select(active::<MF>(metadata, pid.0 * 3i32 + 2i32), silu2, zero)
                .reshape(shape![1, 1, 128]),
            [0i32, 2i32, 0i32],
        );
    }
}

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
