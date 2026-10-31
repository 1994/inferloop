//! Scalar launch arguments update resident metadata without temporary device allocations.
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::METADATA_FIELDS;
use crate::device::CudaDevice;
use crate::device::GraphUpdate;
use crate::device::device_error;
use cutile::prelude::*;
use infer_core::Result;
use infer_models::mrope::ROTARY_AXES;

#[cfg(test)]
#[path = "../../tests/unit/resident_metadata_check.rs"]
mod check;

pub(super) fn update(
    graph: &CudaGraph<()>,
    output: &mut Tensor<i32>,
    values: [i32; METADATA_FIELDS],
) -> Result<()> {
    // CudaGraph::update retains the output lease until replay completes. The replay
    // must use graph.stream(): same-stream ordering replaces the former upload sync.
    // Pass opaque bit patterns through float arguments: cuTile 0.4 automatically
    // specializes integer scalars by DivHint, causing runtime JIT as tokens change.
    // No floating-point arithmetic or conversion is performed on these arguments.
    graph
        .update(GraphUpdate(kernels::write(
            output.partition([METADATA_FIELDS]),
            f32::from_ne_bytes(values[0].to_ne_bytes()),
            f32::from_ne_bytes(values[1].to_ne_bytes()),
            f32::from_ne_bytes(values[2].to_ne_bytes()),
            f32::from_ne_bytes(values[3].to_ne_bytes()),
        )))
        .map_err(device_error)
}

/// Update scalar control metadata and independent rotary coordinates on the same stream.
pub(super) fn update_rotary(
    graph: &CudaGraph<()>,
    output: &mut Tensor<i32>,
    values: [i32; METADATA_FIELDS],
    position: [usize; ROTARY_AXES],
) -> Result<()> {
    let bits = |value: i32| f32::from_ne_bytes(value.to_ne_bytes());
    let height = i32::try_from(position[1]).map_err(device_error)?;
    let width = i32::try_from(position[2]).map_err(device_error)?;
    graph
        .update(GraphUpdate(kernels::write_rotary(
            output.partition([METADATA_FIELDS * 2]),
            bits(values[0]),
            bits(values[1]),
            bits(values[2]),
            bits(values[3]),
            bits(height),
            bits(width),
        )))
        .map_err(device_error)
}

pub use infer_cuda_kernels::tile::state::metadata as kernels;

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
