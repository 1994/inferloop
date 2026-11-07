//! Scalar launch arguments update resident metadata without temporary device allocations.
use crate::constants::KV_BLOCK_TOKENS;
use crate::device::CudaDevice;
use infer_core::Result;
use infer_cuda_kernels::runtime::Allocation;

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
) -> Result<Allocation<i32>> {
    let blocks = capacity.div_ceil(KV_BLOCK_TOKENS).max(1);
    let values: Vec<i32> = (0..lanes)
        .flat_map(|_| 0..blocks)
        .map(|block| i32::try_from(block).unwrap_or(i32::MAX))
        .collect();
    device.runtime().upload(values)
}

pub(super) fn identity_table(device: &CudaDevice, capacity: usize) -> Result<Allocation<i32>> {
    let blocks = capacity.div_ceil(KV_BLOCK_TOKENS);
    let values: Vec<i32> = (0..blocks)
        .map(|block| i32::try_from(block).unwrap_or(i32::MAX))
        .collect();
    if values.is_empty() {
        return Err(infer_core::Error::invalid("a block table needs capacity"));
    }
    device.runtime().upload(values)
}
