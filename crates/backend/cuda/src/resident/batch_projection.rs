use super::buffers::RowMajor;
#[cfg(test)]
use super::buffers::buffer;
use super::{ProgramWeights, batch::Lane, capture::error};
#[cfg(test)]
use crate::constants::CB_DECODE_SLOTS;
use crate::constants::DEFAULT_TILE_COLUMNS;
use crate::constants::DEFAULT_TILE_ROWS;
use crate::constants::F32_BYTES;
use crate::constants::FUSED_VERIFY_LANES;
use crate::constants::GEMV_WIDE_OUTPUT_RATIO;
use crate::constants::NVFP4_GROUP_SIZE;
use crate::constants::SMALL_GEMV_MAX_ROWS;
use crate::strategy::LinearTiling;
use crate::tuning::pool_dense;
use crate::{
    device::{CudaDevice, device_error},
    mlp::ProjectionWeight,
};
use infer_core::{Error, OpId, Result};
use infer_cuda_kernels::contract::{Buffer, Kernels, Recording, RowsKernel, RowsPlan};
#[cfg(test)]
use infer_cuda_kernels::contract::{DenseKernel, DenseRowsPlan, LinearRowsPlan};
use infer_cuda_kernels::tile::runtime::Allocation;
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
    buffers: BTreeMap<(usize, usize), (Allocation<f32>, Allocation<f32>)>,
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
                device.runtime().zeros(width * columns)?,
                device.runtime().zeros(width * rows)?,
            ),
        );
    }
    Ok(buffers)
}

pub(super) fn record(
    scope: &Recording<'_>,
    node: &TensorNode,
    lanes: &[Lane],
    weights: &ProgramWeights,
    buffers: &Workspace,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
) -> Result<()> {
    let key = buffers
        .nodes
        .get(&node.id)
        .ok_or_else(|| error("batch workspace key"))?;
    let (input, output) = buffers
        .buffers
        .get(key)
        .ok_or_else(|| error("batch projection workspace"))?;
    let width = lanes.len();
    let columns = input.len() / width;
    let kernels = &weights.kernels.projection;
    let recording = scope;
    if width == FUSED_VERIFY_LANES {
        kernels
            .pack_three(
                recording,
                columns,
                input.operand(),
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
                    recording,
                    (width, columns),
                    input.operand(),
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
    record_rows(
        (&weights.kernels, scope),
        nvfp4,
        weight,
        weights.activation_quantization(node.inputs[1]),
        RowMajor::new(input.operand(), width, columns)?,
        RowMajor::new(output.operand(), width, output.len() / width)?,
        width,
    )?;
    let outputs = lanes
        .iter()
        .map(|lane| {
            let slot = lane.arena.slot(node.outputs[0]).map_err(error)?;
            lane.arena.buffer(slot).map_err(error)
        })
        .collect::<StdResult<Vec<_>, _>>()?;
    let packed_columns = output.len() / width;
    if let &[first, second, third] = outputs.as_slice() {
        kernels
            .unpack_three(
                recording,
                packed_columns,
                [first, second, third],
                output.operand(),
            )
            .map_err(error)?;
    } else {
        for (row, &target) in outputs.iter().enumerate() {
            kernels
                .unpack_row(
                    recording,
                    (width, packed_columns),
                    target,
                    output.operand(),
                    row,
                )
                .map_err(error)?;
        }
    }
    Ok(())
}

/// Slot-decode twin of the verify [`record`]: the batched arena slot already holds every
/// lane, so the fused GEMV reads the `[lanes, K]` slot in place and stores straight into
/// the reshaped `[lanes, N]` output slot — the pack and unpack copies disappear.
pub(super) fn record_slots(
    scope: &Recording<'_>,
    node: &TensorNode,
    arena: &super::ActivationArena,
    weights: &ProgramWeights,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
    arithmetic_rows: Option<usize>,
) -> Result<()> {
    if weights.constants.contains_key(&node.inputs[0]) {
        return Err(error("slot decode constant linear input"));
    }
    let lanes = arena.lanes();
    let output = arena.activation(node.outputs[0]).map_err(error)?;
    let flat = output.len();
    let output = RowMajor::new(output, lanes, flat / lanes)?;
    let input = arena.activation(node.inputs[0]).map_err(error)?;
    let columns = input.len() / lanes;
    let input = RowMajor::new(input, lanes, columns)?;
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("slot projection weight"))?;
    #[cfg(test)]
    if record_pool_small_f32(
        scope,
        node,
        weights,
        weight,
        input.buffer,
        output.buffer,
        columns,
    )? {
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
            (&weights.kernels, scope),
            weight,
            input.buffer,
            output,
            columns,
            tile,
            choice,
        )? {
            nvfp4.clear_quant_reuse();
            return Ok(());
        }
    }
    if !super::nvfp4_gemm::record_certified(
        (&weights.kernels, scope),
        nvfp4,
        weight,
        weights.activation_quantization(node.inputs[1]),
        input.buffer,
        output,
        columns,
    )? {
        record_rows(
            (&weights.kernels, scope),
            nvfp4,
            weight,
            weights.activation_quantization(node.inputs[1]),
            input,
            output,
            arithmetic_rows.unwrap_or(lanes),
        )?;
    }
    Ok(())
}

/// Project an already gathered `[rows, columns]` matrix into `output`; `arithmetic_rows` is
/// the row count the kernel choice prices.
pub(super) fn record_rows(
    (kernels, scope): (&Kernels, &Recording<'_>),
    nvfp4: &mut super::nvfp4_gemm::Workspace,
    weight: &ProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: RowMajor<'_, f32>,
    output: RowMajor<'_, f32>,
    arithmetic_rows: usize,
) -> Result<()> {
    let (rows, columns) = (input.rows, input.columns);
    if super::nvfp4_gemm::record(
        (kernels, scope),
        nvfp4,
        weight,
        input_scale,
        input.buffer,
        output,
        columns,
    )? {
        return Ok(());
    }
    // Tensor cores amortize weights across candidate rows. Keep the measured
    // GEMV path for narrow batches with a very wide output (e.g. a vocabulary head).
    let outputs = output.columns;
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
        return kernels
            .projection
            .dense(
                scope,
                &DenseRowsPlan {
                    kernel: DenseKernel::Compensated,
                    rows,
                    outputs,
                    columns,
                },
                output.buffer,
                input.buffer,
                buffer(w)?,
            )
            .map_err(error);
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
            scope,
            &RowsPlan {
                kernel,
                rows,
                outputs,
                columns,
            },
            output.buffer,
            input.buffer,
            weight.weight_ref()?,
        )
        .map_err(error)
}

fn lane_input<'a>(
    weights: &'a ProgramWeights,
    node: &TensorNode,
    lane: &'a Lane,
) -> Result<Buffer<'a, f32>> {
    weights.constants.get(&node.inputs[0]).map_or_else(
        || lane.arena.activation(node.inputs[0]).map_err(error),
        |value| Ok(value.buffer()),
    )
}

/// The four-row kernel consumes the existing slot matrix without additional copies.
/// Reuse the private GEMV tile to keep its F32 accumulation and reduction order.
#[cfg(test)]
fn record_pool_small_f32(
    scope: &Recording<'_>,
    node: &TensorNode,
    weights: &ProgramWeights,
    weight: &ProjectionWeight,
    input: Buffer<'_, f32>,
    output: Buffer<'_, f32>,
    columns: usize,
) -> Result<bool> {
    let rows = input.len() / columns.max(1);
    if rows != CB_DECODE_SLOTS
        || !super::small_gemm::pool_f32_enabled()
        || weights.activation_quantization(node.inputs[1]).is_some()
        || columns == 0
        || output.len() / rows > columns.saturating_mul(GEMV_WIDE_OUTPUT_RATIO)
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
    weights
        .kernels
        .projection
        .linear_rows(
            scope,
            &LinearRowsPlan {
                rows,
                row_tile: CB_DECODE_SLOTS,
                output_tile: tile.rows(),
                depth: tile.columns(),
                columns,
            },
            output,
            input,
            buffer(weight)?,
        )
        .map_err(error)?;
    #[cfg(test)]
    super::small_gemm::note_pool_record();
    Ok(true)
}
