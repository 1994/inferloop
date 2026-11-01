//! Exact prompt fusion: original row normalization, then shared-weight F32 projection.
use super::buffers::buffer;
use super::{
    ActivationArena, ProgramWeights,
    batch::BatchBuilder,
    capture::{Capture, CaptureMode, error},
};
use crate::constants::AUX_KERNEL_TILE;
use crate::constants::CB_DECODE_SLOTS;
use crate::constants::METADATA_FIELDS;
use crate::tuning::prefill::fused_draft;
use crate::{
    device::{CudaDevice, DeviceProfile, device_error},
    mlp::ProjectionWeight,
    strategy::LinearTiling,
};
use cutile::{half::bf16, prelude::*};
use infer_core::Result;
use infer_cuda_kernels::contract::{FusionNormBuffers, FusionNormPlan, FusionShape, Recording};
use infer_ir::{TensorNode, TensorOp};
#[cfg(test)]
use std::cell::Cell;
use std::result::Result as StdResult;

const ROWS: usize = 128;
const HIDDEN: usize = 5120;
const COLUMNS: usize = HIDDEN * 2;
const WEIGHT_SHAPE: [i32; 2] = [5120, 10240];
const TILE: [usize; 2] = [4, 16];
const DEPTH: usize = 256;
#[cfg(test)]
const SCRATCH_BYTES: u64 = (5 * 1024 * 1024) + (ROWS as u64 * HIDDEN as u64 * 4);

/// Narrowest prompt width that takes the fully batched path: one gather, one norm launch over
/// every row, one vendor projection.
const WIDE_ROWS: usize = 256;

/// Pool captures use the same exact four-row projection as the 128-row prompt path.
const fn pooled(width: usize) -> bool {
    width >= 2 && width <= CB_DECODE_SLOTS
}

const fn wide(width: usize) -> bool {
    width >= WIDE_ROWS && width.is_multiple_of(WIDE_ROWS)
}

fn layout_bytes(
    architecture: &str,
    width: usize,
    shape: Option<&[i32]>,
    tiling: LinearTiling,
) -> u64 {
    if !fused_draft(architecture) || shape != Some(WEIGHT_SHAPE.as_slice()) {
        return 0;
    }
    if (width == ROWS || pooled(width)) && tiling.rows() == TILE[1] && tiling.columns() == DEPTH {
        (width * (COLUMNS + HIDDEN) * size_of::<f32>()) as u64
    } else if wide(width) {
        // F32 projection input and gathered embeddings, plus the BF16 copy the vendor GEMM reads.
        (width * (COLUMNS + HIDDEN) * size_of::<f32>() + width * COLUMNS * size_of::<bf16>()) as u64
    } else {
        0
    }
}

/// Whether prompt chunks of `width` lanes take the fully batched fusion, which needs no
/// per-lane fusion workspace.
pub fn batched(weights: &ProgramWeights, width: usize, profile: &DeviceProfile) -> bool {
    wide(width) && required_bytes(weights, width, profile) > 0
}

/// The allocation and sequence admission use the same bounded capability.
pub fn required_bytes(weights: &ProgramWeights, width: usize, profile: &DeviceProfile) -> u64 {
    #[cfg(test)]
    if DISABLED.get() {
        return 0;
    }
    let Some(fusion) = &weights.fusion else {
        return 0;
    };
    let ProjectionWeight::Dense(weight) = &fusion.projection else {
        return 0;
    };
    layout_bytes(
        &profile.architecture,
        width,
        Some(weight.shape()),
        fusion.tiling,
    )
}

pub(super) struct Workspace {
    input: Option<Tensor<f32>>,
    /// Batched embedding rows for one prompt chunk; one gather kernel fills all lanes.
    gathered: Option<Tensor<f32>>,
}
impl Workspace {
    pub(super) fn allocate(
        device: &CudaDevice,
        weights: &ProgramWeights,
        width: usize,
    ) -> Result<Option<Self>> {
        if required_bytes(weights, width, device.profile()?) == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            input: Some(
                api::zeros::<f32>(&[width, COLUMNS])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ),
            gathered: Some(
                api::zeros::<f32>(&[width, HIDDEN])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ),
        }))
    }

    /// A workspace over rows a narrower prompt rung borrows from a wider one.
    pub(super) const fn from_rows(input: Tensor<f32>, gathered: Tensor<f32>) -> Self {
        Self {
            input: Some(input),
            gathered: Some(gathered),
        }
    }

    /// The projection input and gathered embedding rows, when no capture holds them.
    pub(super) const fn rows(&self) -> Option<(&Tensor<f32>, &Tensor<f32>)> {
        match (&self.input, &self.gathered) {
            (Some(input), Some(gathered)) => Some((input, gathered)),
            _ => None,
        }
    }

    /// Columns of the projection input and the gathered embedding rows.
    pub(super) const fn columns() -> (usize, usize) {
        (COLUMNS, HIDDEN)
    }

    #[cfg(test)]
    pub(super) fn bytes(&self) -> usize {
        self.input.as_ref().map_or(0, Tensor::num_bytes)
            + self.gathered.as_ref().map_or(0, Tensor::num_bytes)
    }

    fn copy_row(
        &mut self,
        scope: &Scope,
        lane: usize,
        normalized: &Tensor<f32>,
    ) -> StdResult<(), DeviceError> {
        let matrix = self
            .input
            .take()
            .ok_or_else(|| error("fusion batch input"))?;
        let rows = usize::try_from(matrix.shape()[0]).map_err(error)?;
        let (mut row, owner) = super::arena::row_tensor(matrix, rows, lane).map_err(error)?;
        scope.record(api::memcpy(&mut row, normalized))?;
        self.input =
            Some(Arc::try_unwrap(owner).map_err(|_| error("fusion batch input still aliased"))?);
        Ok(())
    }
}

/// Returns false before recording when this embedding is outside the validated capability.
pub(super) fn record(
    builder: &mut BatchBuilder<'_>,
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    metadata: &[Tensor<i32>],
    metadata_batch: &Tensor<i32>,
    workspace: &mut Workspace,
) -> StdResult<bool, DeviceError> {
    #[cfg(test)]
    if DISABLED.get() {
        return Ok(false);
    }
    let rows = arena.lanes();
    if node.op == TensorOp::Embedding && node.outputs.len() == 1 && wide(arena.lanes()) {
        return record_wide(builder, scope, node, arena, metadata_batch, workspace);
    }
    if node.op != TensorOp::Embedding || node.outputs.len() != 1 || !(rows == ROWS || pooled(rows))
    {
        return Ok(false);
    }
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    if arena.get(node.outputs[0]).map_err(error)?.size() != rows * HIDDEN {
        return Ok(false);
    }
    if metadata.len() < rows || builder.lane_fusion.len() < rows {
        return Err(error("fusion batch lanes"));
    }
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("fusion batch output"))?;
    let restore = output
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).map_err(error))
        .collect::<StdResult<Vec<_>, _>>()?;
    let mut output = output.reshape(&[rows, HIDDEN])?;
    let Some(super::program::FusionWeights {
        projection: ProjectionWeight::Dense(weight),
        ..
    }) = &builder.weights.fusion
    else {
        return Err(error("fusion batch dense capability"));
    };
    let embedding_weight = builder
        .weights
        .embeddings
        .get(&node.inputs[0])
        .ok_or_else(|| error("fusion batch embedding"))?;
    let mut gathered = workspace
        .gathered
        .take()
        .ok_or_else(|| error("fusion batch gather"))?;
    scope.record(
        super::attention::attention::embedding_batch(
            (&mut gathered).partition([1, AUX_KERNEL_TILE]),
            embedding_weight,
            metadata_batch,
        )
        .generics(vec![HIDDEN.to_string(), METADATA_FIELDS.to_string()]),
    )?;
    for (lane, meta) in metadata.iter().enumerate().take(rows) {
        // The owned row tensor keeps the gather allocation anchored until it is put back;
        // `row_tensor` bounds the lane inside the slot and the return re-anchors ownership.
        let (embedding, owner) = super::arena::row_tensor(gathered, rows, lane).map_err(error)?;
        let normalized = Capture {
            scope,
            arena: &mut *arena,
            weights: builder.weights,
            nvfp4: builder.nvfp4,
            attention: builder.attention,
            states: builder.states,
            fp8_states: builder.fp8_states,
            metadata: meta,
            table: &builder.tables[0],
            external: &builder.lane_external[lane],
            capacity: builder.capacity,
            fusion: &mut builder.lane_fusion[lane],
            mode: CaptureMode::Row(lane),
        }
        .fusion_norm(HIDDEN, &embedding)?;
        gathered =
            Arc::try_unwrap(owner).map_err(|_| error("fusion batch gather still aliased"))?;
        workspace.copy_row(scope, lane, &normalized)?;
        builder.lane_fusion[lane]
            .as_mut()
            .ok_or_else(|| error("fusion row owner"))?
            .normalized = Some(normalized);
    }
    workspace.gathered = Some(gathered);
    scope.record(
        super::linear_batch::batched::dense(
            (&mut output).partition(TILE),
            workspace
                .input
                .as_ref()
                .ok_or_else(|| error("fusion batch owner"))?,
            weight,
        )
        .generics(vec![
            bf16::DTYPE.as_str().into(),
            TILE[1].to_string(),
            DEPTH.to_string(),
            COLUMNS.to_string(),
        ]),
    )?;
    arena.buffers[slot] = Some(output.reshape(&restore)?);
    #[cfg(test)]
    CAPTURES.set(CAPTURES.get() + 1);
    Ok(true)
}

/// Wide prompt chunk: gather every embedding row, normalize every embedding and external hidden
/// row in one launch, then project the `[lanes, 2 * hidden]` input with one vendor GEMM.
fn record_wide(
    builder: &mut BatchBuilder<'_>,
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    metadata_batch: &Tensor<i32>,
    workspace: &mut Workspace,
) -> StdResult<bool, DeviceError> {
    let lanes = arena.lanes();
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    if arena.get(node.outputs[0]).map_err(error)?.size() != lanes * HIDDEN {
        return Ok(false);
    }
    let Some(external) = builder.lane_external_owner else {
        return Ok(false);
    };
    if external.size() < lanes * HIDDEN {
        return Err(error("fusion batch external rows"));
    }
    // The owner holds the widest prompt's rows; a narrower rung's grid reads its leading lanes.
    let Some(super::program::FusionWeights {
        projection: ProjectionWeight::Dense(weight),
        norms,
        epsilon,
        offset,
        ..
    }) = &builder.weights.fusion
    else {
        return Err(error("fusion batch dense capability"));
    };
    let Some(delegated) = builder.delegated.as_deref_mut() else {
        return Ok(false);
    };
    let embedding_weight = builder
        .weights
        .embeddings
        .get(&node.inputs[0])
        .ok_or_else(|| error("fusion batch embedding"))?;
    let mut gathered = workspace
        .gathered
        .take()
        .ok_or_else(|| error("fusion batch gather"))?;
    scope.record(
        super::attention::attention::embedding_batch(
            (&mut gathered).partition([1, AUX_KERNEL_TILE]),
            embedding_weight,
            metadata_batch,
        )
        .generics(vec![HIDDEN.to_string(), METADATA_FIELDS.to_string()]),
    )?;
    let input = workspace
        .input
        .take()
        .ok_or_else(|| error("fusion batch input"))?;
    let rows = input.reshape(&[2 * lanes, HIDDEN])?;
    builder
        .weights
        .kernels
        .norm
        .fusion(
            &Recording::scope(scope),
            &FusionNormPlan {
                shape: FusionShape::Lanes { lanes },
                dimension: HIDDEN,
                epsilon: *epsilon,
                offset: *offset,
            },
            FusionNormBuffers {
                out: buffer(&rows)?,
                embedding: buffer(&gathered)?,
                hidden: buffer(external)?,
                norms: buffer(norms.as_ref())?,
            },
        )
        .map_err(error)?;
    workspace.gathered = Some(gathered);
    let input = rows.reshape(&[lanes * COLUMNS])?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("fusion batch output"))?;
    let recorded = delegated.record_dense(scope, &input, weight, &output, (lanes, HIDDEN, COLUMNS));
    workspace.input = Some(input.reshape(&[lanes, COLUMNS])?);
    arena.buffers[slot] = Some(output);
    if recorded? {
        Ok(true)
    } else {
        Err(error("fusion batch vendor projection"))
    }
}

#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.get()
}
/// Builds the unchanged row reference on this thread; no production configuration exists.
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
#[path = "../../tests/unit/resident_fusion_batch.rs"]
mod tests;

/// Captured nodes per fused chunk after the batched gather: one gather, one projection,
/// and one norm + one memcpy per lane. The IR graph node count is unchanged.
#[cfg(test)]
pub(super) const fn captured_nodes(width: usize) -> usize {
    2 + 2 * width
}

#[cfg(test)]
#[path = "../../tests/unit/resident_pooled_fusion.rs"]
mod pooled_tests;
