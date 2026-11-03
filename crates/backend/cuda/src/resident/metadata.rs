//! Scalar launch arguments update resident metadata without temporary device allocations.
use crate::constants::KV_BLOCK_TOKENS;
use crate::device::CudaDevice;
use crate::device::device_error;
use cutile::prelude::*;
use infer_core::Result;

#[cfg(test)]
#[path = "../../tests/unit/resident_metadata_check.rs"]
mod check;

pub(super) use infer_cuda_kernels::tile::state::{update, update_rotary};

/// Identity block table for one sequence's own KV cache: logical block `b` is physical block `b`.
/// Step 1 of the KV refactor keeps the arena untouched, so this is the mapping the kernels consult
/// while every byte stays where it is today; step 2 replaces it with a real allocator's output.
/// One identity table per lane in a single `[lanes, blocks]` tensor.
pub(super) fn identity_tables(
    device: &CudaDevice,
    capacity: usize,
    lanes: usize,
) -> Result<Tensor<i32>> {
    let blocks = capacity.div_ceil(KV_BLOCK_TOKENS).max(1);
    let values: Vec<i32> = (0..lanes)
        .flat_map(|_| 0..blocks)
        .map(|block| i32::try_from(block).unwrap_or(i32::MAX))
        .collect();
    let tables = device
        .upload(values, &[lanes, blocks])
        .map_err(device_error)?;
    Arc::try_unwrap(tables).map_err(|_| infer_core::Error::invariant("unique block tables"))
}

pub(super) fn identity_table(device: &CudaDevice, capacity: usize) -> Result<Tensor<i32>> {
    let blocks = capacity.div_ceil(KV_BLOCK_TOKENS);
    let values: Vec<i32> = (0..blocks)
        .map(|block| i32::try_from(block).unwrap_or(i32::MAX))
        .collect();
    let table = device
        .upload(values, &[blocks.max(1)])
        .map_err(device_error)?;
    Arc::try_unwrap(table).map_err(|_| infer_core::Error::invariant("unique block table"))
}
