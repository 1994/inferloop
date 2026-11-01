use super::buffers::{buffer, region};
#[cfg(test)]
use super::linear_batch::batched;
use super::{ProgramWeights, batch::Lane, capture::error};
#[cfg(test)]
use crate::constants::CB_DECODE_SLOTS;
use crate::constants::DEFAULT_TILE_COLUMNS;
use crate::constants::DEFAULT_TILE_ROWS;
use crate::constants::F32_BYTES;
use crate::constants::FUSED_VERIFY_LANES;
use crate::constants::GEMV_WIDE_OUTPUT_RATIO;
use crate::constants::NVFP4_GROUP_SIZE;
#[cfg(test)]
use crate::constants::SMALL_GEMM_TILE;
use crate::constants::SMALL_GEMV_MAX_ROWS;
use crate::strategy::LinearTiling;
use crate::tuning::pool_dense;
use crate::{
    device::{CudaDevice, device_error},
    mlp::ProjectionWeight,
};
#[cfg(test)]
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, OpId, Result};
use infer_cuda_kernels::contract::{Buffer, Kernels, Recording, RowsKernel, RowsPlan};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::result::Result as StdResult;

/// Rows of one verify tile: one row group of the fused GEMV. The kernel walks row groups
/// on grid dim 0, so any row count works; a partial group is predicated out.
const VERIFY_TILE_ROWS: usize = FUSED_VERIFY_LANES + 1;
/// Output columns per block. Measured on the slot decode hot path (3 lanes, n3 sweep):
/// 8 → 39.3 ms/step of linear, 16 → 28.6 ms; 32 crosses the register spill cliff of the
/// `[rows, BN, depth]` accumulator (16k f32 entries) and collapses to 88.9 ms.
const VERIFY_TILE_COLUMNS: usize = 16;
/// Inner depth of the verify GEMV k-loop.
const VERIFY_TILE_DEPTH: usize = 256;
/// Packed FP4 bytes per inner step: two nibble values share one byte.
const VERIFY_PACKED_DEPTH: usize = VERIFY_TILE_DEPTH / 2;
/// Scale groups per inner step: one E4M3 scale per NVFP4 group of 16 values.
const VERIFY_SCALE_DEPTH: usize = VERIFY_TILE_DEPTH / NVFP4_GROUP_SIZE;
/// Output tile of the batched verification GEMV.
const VERIFY_OUTPUT_TILE: [usize; 2] = [VERIFY_TILE_ROWS, VERIFY_TILE_COLUMNS];

pub(super) struct Workspace {
    nodes: BTreeMap<OpId, (usize, usize)>,
    buffers: BTreeMap<(usize, usize), (Tensor<f32>, Tensor<f32>)>,
}

pub(super) fn allocate(
    device: &CudaDevice,
    graph: &DataflowGraph,
    width: usize,
) -> Result<Workspace> {
    let mut buffers = Workspace {
        nodes: BTreeMap::new(),
        buffers: BTreeMap::new(),
    };
    if width < 2 {
        return Ok(buffers);
    }
    let sizes: BTreeMap<_, _> = graph
        .tensors
        .iter()
        .map(|t| Ok((t.id, t.elements()?)))
        .collect::<Result<_>>()?;
    let mut bytes = 0usize;
    for node in &graph.nodes {
        if node.op != TensorOp::Linear {
            continue;
        }
        let columns = sizes[&node.inputs[0]];
        let rows = sizes[&node.outputs[0]];
        let key = (rows, columns);
        buffers.nodes.insert(node.id, key);
        if buffers.buffers.contains_key(&key) {
            continue;
        }
        bytes = rows
            .checked_add(columns)
            .and_then(|n| n.checked_mul(width * F32_BYTES))
            .and_then(|n| bytes.checked_add(n))
            .ok_or_else(|| Error::invalid("batched projection workspace overflow"))?;
        if u64::try_from(bytes).map_err(device_error)? > device.profile()?.arena_budget_bytes() {
            return Err(Error::invalid("batched projection workspace budget"));
        }
        buffers.buffers.insert(
            key,
            (
                api::zeros::<f32>(&[width, columns])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
                api::zeros::<f32>(&[width, rows])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ),
        );
    }
    Ok(buffers)
}

pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    lanes: &mut [Lane],
    weights: &ProgramWeights,
    buffers: &mut Workspace,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
) -> StdResult<(), DeviceError> {
    let key = buffers
        .nodes
        .get(&node.id)
        .ok_or_else(|| error("batch workspace key"))?;
    let (input, output) = buffers
        .buffers
        .get_mut(key)
        .ok_or_else(|| error("batch projection workspace"))?;
    let width = lanes.len();
    let columns = input.size() / width;
    let kernels = &weights.kernels.projection;
    let recording = Recording::scope(scope);
    if width == FUSED_VERIFY_LANES {
        kernels
            .pack_three(
                &recording,
                columns,
                buffer(&*input)?,
                [
                    lane_input(weights, node, &lanes[0])?,
                    lane_input(weights, node, &lanes[1])?,
                    lane_input(weights, node, &lanes[2])?,
                ],
            )
            .map_err(error)?;
    } else {
        for (row, lane) in lanes.iter().enumerate() {
            kernels
                .pack_row(
                    &recording,
                    (width, columns),
                    buffer(&*input)?,
                    lane_input(weights, node, lane)?,
                    row,
                )
                .map_err(error)?;
        }
    }
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("batch projection weight"))?;
    let packed = input.view(&[width, columns]).map_err(error)?;
    record_projection(
        (&weights.kernels, scope),
        nvfp4,
        weight,
        weights.activation_quantization(node.inputs[1]),
        &packed,
        input.device_pointer().cu_deviceptr(),
        output,
        (columns, width),
    )?;
    let mut outputs = Vec::new();
    let mut slots = Vec::new();
    for lane in lanes.iter_mut() {
        let slot = lane.arena.slot(node.outputs[0]).map_err(error)?;
        outputs.push(
            lane.arena.buffers[slot]
                .take()
                .ok_or_else(|| error("batch output"))?,
        );
        slots.push(slot);
    }
    let packed_columns = output.size() / width;
    if let [first, second, third] = outputs.as_slice() {
        kernels
            .unpack_three(
                &recording,
                packed_columns,
                [buffer(first)?, buffer(second)?, buffer(third)?],
                buffer(&*output)?,
            )
            .map_err(error)?;
    } else {
        for (row, target) in outputs.iter().enumerate() {
            kernels
                .unpack_row(
                    &recording,
                    (width, packed_columns),
                    buffer(target)?,
                    buffer(&*output)?,
                    row,
                )
                .map_err(error)?;
        }
    }
    for ((lane, slot), output) in lanes.iter_mut().zip(slots).zip(outputs) {
        lane.arena.buffers[slot] = Some(output);
    }
    Ok(())
}

/// Slot-decode twin of the verify [`record`]: the batched arena slot already holds every
/// lane, so the fused GEMV reads the `[lanes, K]` slot in place and stores straight into
/// the reshaped `[lanes, N]` output slot — the pack and unpack copies disappear.
pub(super) fn record_slots(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut super::ActivationArena,
    weights: &ProgramWeights,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
    arithmetic_rows: Option<usize>,
) -> StdResult<(), DeviceError> {
    if weights.constants.contains_key(&node.inputs[0]) {
        return Err(error("slot decode constant linear input"));
    }
    let lanes = arena.lanes();
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("slot decode output"))?;
    let input = arena.get(node.inputs[0]).map_err(error)?;
    let input_pointer = input.device_pointer().cu_deviceptr();
    let columns = input.size() / lanes;
    let input = input.view(&[lanes, columns]).map_err(error)?;
    let flat = output.size();
    let mut output = output.reshape(&[lanes, flat / lanes]).map_err(error)?;
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("slot projection weight"))?;
    #[cfg(test)]
    if record_pool_small_f32(scope, node, weights, weight, &input, &mut output, columns)? {
        arena.buffers[slot] = Some(output.reshape(&[flat]).map_err(error)?);
        return Ok(());
    }
    // Verify graphs pin their tensor-core arithmetic through `arithmetic_rows`.
    if weights.activation_quantization(node.inputs[1]).is_none()
        && (arithmetic_rows.is_none() || pool_dense::supported([lanes, flat / lanes, columns]))
    {
        let tile =
            weights.tiling.get(&node.inputs[1]).copied().unwrap_or(
                LinearTiling::new(DEFAULT_TILE_ROWS, DEFAULT_TILE_COLUMNS).map_err(error)?,
            );
        let choice = nvfp4.pool_dense_choice(weight, [lanes, flat / lanes, columns], tile);
        if super::pool_gemv::record(
            &weights.kernels,
            scope,
            weight,
            &input,
            input_pointer,
            &mut output,
            columns,
            tile,
            choice,
        )? {
            nvfp4.clear_quant_reuse();
            arena.buffers[slot] = Some(output.reshape(&[flat]).map_err(error)?);
            return Ok(());
        }
    }
    if !super::nvfp4_gemm::record_with_input_pointer(
        scope,
        nvfp4,
        weight,
        weights.activation_quantization(node.inputs[1]),
        &input,
        &mut output,
        columns,
        input_pointer,
    )? {
        record_projection(
            (&weights.kernels, scope),
            nvfp4,
            weight,
            weights.activation_quantization(node.inputs[1]),
            &input,
            input_pointer,
            &mut output,
            (columns, arithmetic_rows.unwrap_or(lanes)),
        )?;
    }
    arena.buffers[slot] = Some(output.reshape(&[flat]).map_err(error)?);
    Ok(())
}

/// Project an already gathered `[rows, columns]` matrix.
#[expect(
    clippy::too_many_arguments,
    reason = "The view does not expose its address, so it is passed alongside"
)]
pub(super) fn record_rows(
    (kernels, scope): (&Kernels, &Scope),
    nvfp4: &mut super::nvfp4_gemm::Workspace,
    weight: &ProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: &TensorView<'_, f32>,
    input_pointer: u64,
    output: &mut Tensor<f32>,
    geometry: (usize, usize),
) -> StdResult<(), DeviceError> {
    record_projection(
        (kernels, scope),
        nvfp4,
        weight,
        input_scale,
        input,
        input_pointer,
        output,
        geometry,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "The view does not expose its address, so it is passed alongside"
)]
fn record_projection(
    (kernels, scope): (&Kernels, &Scope),
    nvfp4: &mut super::nvfp4_gemm::Workspace,
    weight: &ProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    x: &TensorView<'_, f32>,
    x_pointer: u64,
    output: &mut Tensor<f32>,
    (columns, arithmetic_rows): (usize, usize),
) -> StdResult<(), DeviceError> {
    if super::nvfp4_gemm::record(scope, nvfp4, weight, input_scale, x, output, columns)? {
        return Ok(());
    }
    // Tensor cores amortize weights across candidate rows. Keep the measured
    // GEMV path for narrow batches with a very wide output (e.g. a vocabulary head).
    let rows = usize::try_from(x.shape()[0]).map_err(error)?;
    let outputs = output.size() / rows;
    let small = arithmetic_rows > SMALL_GEMV_MAX_ROWS
        || outputs <= columns.saturating_mul(GEMV_WIDE_OUTPUT_RATIO);
    #[cfg(test)]
    if small
        && matches!(weight, ProjectionWeight::Dense(_))
        && super::small_gemm::precise_dense_enabled()
        && outputs <= columns.saturating_mul(GEMV_WIDE_OUTPUT_RATIO)
    {
        let ProjectionWeight::Dense(w) = weight else {
            return Err(error("precise dense weight"));
        };
        scope.record(
            super::small_gemm::precise_kernels::dense(output.partition(SMALL_GEMM_TILE), x, w)
                .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
        )?;
        return Ok(());
    }
    let kernel = if small
        && matches!(
            weight,
            ProjectionWeight::Dense(_) | ProjectionWeight::Fp8(..)
        ) {
        RowsKernel::Small
    } else {
        RowsKernel::Tiled {
            output_tile: VERIFY_OUTPUT_TILE,
            tile_output: VERIFY_TILE_COLUMNS,
            tile_columns: VERIFY_TILE_DEPTH,
            fp4_depths: [VERIFY_PACKED_DEPTH, VERIFY_SCALE_DEPTH],
        }
    };
    kernels
        .projection
        .rows(
            &Recording::scope(scope),
            &RowsPlan {
                kernel,
                rows,
                outputs,
                columns,
            },
            buffer(&*output)?,
            region(x_pointer, rows * columns),
            weight.weight_ref()?,
        )
        .map_err(error)
}

fn lane_input<'a>(
    weights: &'a ProgramWeights,
    node: &TensorNode,
    lane: &'a Lane,
) -> StdResult<Buffer<'a, f32>, DeviceError> {
    buffer(weights.constants.get(&node.inputs[0]).map_or_else(
        || lane.arena.get(node.inputs[0]).map_err(error),
        |value| Ok(value.as_ref()),
    )?)
}

/// The four-row kernel consumes the existing slot matrix without additional copies.
/// Reuse the private GEMV tile to keep its F32 accumulation and reduction order.
#[cfg(test)]
fn record_pool_small_f32(
    scope: &Scope,
    node: &TensorNode,
    weights: &ProgramWeights,
    weight: &ProjectionWeight,
    input: &TensorView<'_, f32>,
    output: &mut Tensor<f32>,
    columns: usize,
) -> StdResult<bool, DeviceError> {
    let rows = usize::try_from(input.shape()[0]).map_err(error)?;
    if rows != CB_DECODE_SLOTS
        || !super::small_gemm::pool_f32_enabled()
        || weights.activation_quantization(node.inputs[1]).is_some()
        || columns == 0
        || output.size() / rows > columns.saturating_mul(GEMV_WIDE_OUTPUT_RATIO)
    {
        return Ok(false);
    }
    let ProjectionWeight::Dense(weight) = weight else {
        return Ok(false);
    };
    let tile = weights
        .tiling
        .get(&node.inputs[1])
        .copied()
        .unwrap_or(LinearTiling::new(DEFAULT_TILE_ROWS, DEFAULT_TILE_COLUMNS).map_err(error)?);
    scope.record(
        batched::dense(
            output.partition([CB_DECODE_SLOTS, tile.rows()]),
            input,
            weight,
        )
        .generics(vec![
            bf16::DTYPE.as_str().into(),
            tile.rows().to_string(),
            tile.columns().to_string(),
            columns.to_string(),
        ]),
    )?;
    #[cfg(test)]
    super::small_gemm::note_pool_record();
    Ok(true)
}
