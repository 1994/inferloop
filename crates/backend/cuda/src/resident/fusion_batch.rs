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
    device::{CudaDevice, DeviceProfile},
    mlp::ProjectionWeight,
    strategy::LinearTiling,
};
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    Buffer, EmbeddingBatchPlan, FusionNormBuffers, FusionNormPlan, FusionShape, LinearRowsPlan,
    Recording,
};
use infer_cuda_kernels::runtime::Allocation;
use infer_cuda_kernels::runtime::bf16;
use infer_ir::{TensorNode, TensorOp};
#[cfg(test)]
use std::cell::Cell;
use std::ops::Range;

const ROWS: usize = 128;
const HIDDEN: usize = 5120;
const COLUMNS: usize = HIDDEN * 2;
const WEIGHT_SHAPE: [usize; 2] = [5120, 10240];
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
    shape: Option<[usize; 2]>,
    tiling: LinearTiling,
) -> u64 {
    if !fused_draft(architecture) || shape != Some(WEIGHT_SHAPE) {
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

/// Prompt fusion rows: the `[rows, COLUMNS]` projection input and the `[rows, HIDDEN]` gathered
/// embeddings one gather kernel fills for every lane. Captures write through views cut from the
/// allocations, so a narrower prompt rung shares a wider rung's memory by viewing its leading rows.
pub(super) struct Workspace {
    input: Allocation<f32>,
    gathered: Allocation<f32>,
    rows: usize,
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
        let zeros = |columns: usize| -> Result<Allocation<f32>> {
            device.tile().zeros::<f32>(width * columns)
        };
        Ok(Some(Self {
            input: zeros(COLUMNS)?,
            gathered: zeros(HIDDEN)?,
            rows: width,
        }))
    }

    /// The leading `rows` rows, for a narrower prompt rung; the memory is shared.
    /// # Errors
    /// Rejects more rows than this workspace holds.
    pub(super) fn narrowed(&self, rows: usize) -> Result<Self> {
        if rows > self.rows {
            return Err(Error::invalid("narrowed fusion rows"));
        }
        Ok(Self {
            input: self.input.clone(),
            gathered: self.gathered.clone(),
            rows,
        })
    }

    #[cfg(test)]
    pub(super) const fn bytes(&self) -> usize {
        self.rows * (COLUMNS + HIDDEN) * size_of::<f32>()
    }

    /// Projection input rows `lanes`.
    fn input_rows(&self, lanes: Range<usize>) -> Result<Buffer<'_, f32>> {
        self.rows_of(&self.input, COLUMNS, lanes)
    }

    /// Gathered embedding rows `lanes`.
    fn gathered_rows(&self, lanes: Range<usize>) -> Result<Buffer<'_, f32>> {
        self.rows_of(&self.gathered, HIDDEN, lanes)
    }

    fn rows_of<'a>(
        &self,
        memory: &'a Allocation<f32>,
        columns: usize,
        lanes: Range<usize>,
    ) -> Result<Buffer<'a, f32>> {
        if lanes.end > self.rows {
            return Err(error("fusion batch rows"));
        }
        memory
            .operand()
            .slice(lanes.start * columns..lanes.end * columns)
            .map_err(error)
    }

    fn copy_row(
        &self,
        scope: &Recording<'_>,
        lane: usize,
        normalized: Buffer<'_, f32>,
    ) -> Result<()> {
        scope
            .copy(self.input_rows(lane..lane + 1)?, normalized)
            .map_err(error)
    }
}

/// Returns false before recording when this embedding is outside the validated capability.
/// Gather every lane's embedding row in one launch.
fn gather(
    builder: &BatchBuilder<'_>,
    scope: &Recording<'_>,
    node: &TensorNode,
    gathered: Buffer<'_, f32>,
    metadata_batch: Buffer<'_, i32>,
) -> Result<()> {
    let weight = builder
        .weights
        .embeddings
        .get(&node.inputs[0])
        .ok_or_else(|| error("fusion batch embedding"))?;
    builder
        .weights
        .kernels
        .projection
        .embedding_batch(
            scope,
            &EmbeddingBatchPlan {
                dimension: HIDDEN,
                fields: METADATA_FIELDS,
                tile: AUX_KERNEL_TILE,
            },
            gathered,
            buffer(weight)?,
            None,
            metadata_batch,
        )
        .map_err(error)
}

pub(super) fn record(
    builder: &mut BatchBuilder<'_>,
    scope: &Recording<'_>,
    node: &TensorNode,
    arena: &mut ActivationArena,
    metadata: &[Allocation<i32>],
    metadata_batch: &Allocation<i32>,
    workspace: &Workspace,
) -> Result<bool> {
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
    if arena.get(node.outputs[0]).map_err(error)?.len() != rows * HIDDEN {
        return Ok(false);
    }
    if metadata.len() < rows || builder.lane_fusion.len() < rows {
        return Err(error("fusion batch lanes"));
    }
    let Some(super::program::FusionWeights {
        projection: ProjectionWeight::Dense(weight),
        ..
    }) = &builder.weights.fusion
    else {
        return Err(error("fusion batch dense capability"));
    };
    let gathered = workspace.gathered_rows(0..rows)?;
    gather(builder, scope, node, gathered, buffer(metadata_batch)?)?;
    let input = workspace.input_rows(0..rows)?;
    for (lane, meta) in metadata.iter().enumerate().take(rows) {
        let embedding = workspace.gathered_rows(lane..lane + 1)?;
        Capture {
            scope,
            arena: &mut *arena,
            weights: builder.weights,
            nvfp4: builder.nvfp4,
            attention: builder.attention,
            states: builder.states,
            fp8_states: builder.fp8_states,
            metadata: buffer(meta)?,
            table: buffer(&builder.tables[0])?,
            external: buffer(&builder.lane_external[lane])?,
            capacity: builder.capacity,
            fusion: &mut builder.lane_fusion[lane],
            mode: CaptureMode::Row(lane),
        }
        .fusion_norm(HIDDEN, embedding)?;
        let normalized = builder.lane_fusion[lane]
            .as_ref()
            .ok_or_else(|| error("fusion row owner"))?
            .normalized
            .operand();
        workspace.copy_row(scope, lane, normalized)?;
    }
    let output = arena.buffer(slot).map_err(error)?;
    builder
        .weights
        .kernels
        .projection
        .linear_rows(
            scope,
            &LinearRowsPlan {
                rows,
                row_tile: TILE[0],
                output_tile: TILE[1],
                depth: DEPTH,
                columns: COLUMNS,
            },
            output,
            input,
            buffer(weight)?,
        )
        .map_err(error)?;
    #[cfg(test)]
    CAPTURES.set(CAPTURES.get() + 1);
    Ok(true)
}

/// Wide prompt chunk: gather every embedding row, normalize every embedding and external hidden
/// row in one launch, then project the `[lanes, 2 * hidden]` input with one vendor GEMM.
fn record_wide(
    builder: &mut BatchBuilder<'_>,
    scope: &Recording<'_>,
    node: &TensorNode,
    arena: &ActivationArena,
    metadata_batch: &Allocation<i32>,
    workspace: &Workspace,
) -> Result<bool> {
    let lanes = arena.lanes();
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    if arena.get(node.outputs[0]).map_err(error)?.len() != lanes * HIDDEN {
        return Ok(false);
    }
    let Some(external) = builder.lane_external_owner else {
        return Ok(false);
    };
    if external.len() < lanes * HIDDEN {
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
    if builder.delegated.is_none() {
        return Ok(false);
    }
    let gathered = workspace.gathered_rows(0..lanes)?;
    gather(builder, scope, node, gathered, buffer(metadata_batch)?)?;
    let Some(delegated) = builder.delegated.as_deref_mut() else {
        return Ok(false);
    };
    // Each lane's `COLUMNS` input row is its normalized embedding then its normalized hidden row.
    let rows = workspace.input_rows(0..lanes)?;
    builder
        .weights
        .kernels
        .norm
        .fusion(
            scope,
            &FusionNormPlan {
                shape: FusionShape::Lanes { lanes },
                dimension: HIDDEN,
                epsilon: *epsilon,
                offset: *offset,
            },
            FusionNormBuffers {
                out: rows,
                embedding: gathered,
                hidden: buffer(external)?,
                norms: buffer(norms)?,
            },
        )
        .map_err(error)?;
    let output = arena.buffer(slot).map_err(error)?;
    let recorded = delegated.record_dense(
        (&builder.weights.kernels, scope),
        rows,
        weight,
        output,
        (lanes, HIDDEN, COLUMNS),
    )?;
    if recorded {
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
