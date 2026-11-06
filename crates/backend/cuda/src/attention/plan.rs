//! Dense SDPA graph recording with caller-owned buffers and stream.
use crate::device::device_error;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{Attention, DenseBuffers, DensePlan, Recording};
use infer_ir::DType as Precision;
use infer_kernel_api::attention::{AttentionDescriptor, AttentionMask};

/// Default query/key tile, independent of GPU model and model architecture.
const TILE: usize = 32;
/// Larger query/key tile amortizes the inner loop for long sequences.
const WIDE_TILE: usize = 64;
/// Shape-based crossover shared by dense attention and the vision caller.
const TILE_CROSSOVER: usize = 512;
/// Current compiled provider width limit; the semantic descriptor has no such limit.
const MAX_WIDTH: usize = 256;
/// Encoded mask kinds consumed by the CUDA kernel.
const WINDOW_MASK: i32 = 2;
/// Equal-sized segment mask discriminant.
const SEGMENT_MASK: i32 = 3;
/// Q, K, V and output buffers.
const BUFFER_COUNT: usize = 4;
/// Mask kind, segment length, query offset, left and right window extents.
const MASK_FIELDS: usize = 5;

/// Prepared dense attention. No allocations, host copies or synchronization occur in `record`.
/// All buffers use `[padded_tokens, heads * padded_head_width]`; padding lanes must be zero.
pub struct DenseAttentionPlan {
    descriptor: AttentionDescriptor,
    shapes: [[usize; 2]; BUFFER_COUNT],
    width: usize,
    value_width: usize,
    tile: usize,
    mask: [i32; MASK_FIELDS],
    queries: i32,
    keys: i32,
}

impl DenseAttentionPlan {
    /// Prepare a compensated-F32 provider without selecting by model or device name.
    /// # Errors
    /// Rejects invalid semantics, unsupported dtype/width or device index overflow.
    pub fn new(descriptor: AttentionDescriptor) -> Result<Self> {
        descriptor.validate()?;
        if descriptor.dtype != Precision::F32 {
            return Err(Error::unsupported(
                "dense CUDA attention provider requires F32",
            ));
        }
        let width = padded_width(descriptor.qk_dim)?;
        let value_width = padded_width(descriptor.value_dim)?;
        let tile = tile_rows(descriptor.queries);
        let qrows = descriptor
            .queries
            .checked_next_multiple_of(tile)
            .ok_or_else(|| Error::invalid("attention row overflow"))?;
        let qrows = if descriptor.queries == 1 { 1 } else { qrows };
        let krows = descriptor
            .keys
            .checked_next_multiple_of(tile)
            .ok_or_else(|| Error::invalid("attention row overflow"))?;
        let columns = |heads: usize, dim| {
            heads
                .checked_mul(dim)
                .ok_or_else(|| Error::invalid("attention columns overflow"))
        };
        let shapes = [
            [qrows, columns(descriptor.query_heads, width)?],
            [krows, columns(descriptor.kv_heads, width)?],
            [krows, columns(descriptor.kv_heads, value_width)?],
            [qrows, columns(descriptor.query_heads, value_width)?],
        ];
        for shape in shapes {
            for extent in shape {
                let _extent = index(extent)?;
            }
            shape[0]
                .checked_mul(shape[1])
                .ok_or_else(|| Error::invalid("attention allocation overflow"))?;
        }
        let mask = mask_arguments(descriptor.mask)?;
        let low = i64::from(mask[2]) - i64::from(mask[3]);
        let high = i64::from(index(qrows)?) + i64::from(mask[2]) + i64::from(mask[4]);
        if low < i64::from(i32::MIN) || high > i64::from(i32::MAX) {
            return Err(Error::unsupported(
                "attention mask exceeds device index range",
            ));
        }
        Ok(Self {
            descriptor,
            shapes,
            width,
            value_width,
            tile,
            mask,
            queries: index(descriptor.queries)?,
            keys: index(descriptor.keys)?,
        })
    }

    /// Required Q, K, V and output allocation shapes, respectively.
    #[must_use]
    pub const fn shapes(&self) -> [[usize; 2]; BUFFER_COUNT] {
        self.shapes
    }

    /// Tile selected from sequence geometry, without model or device-name rules.
    #[must_use]
    pub const fn tile(&self) -> usize {
        self.tile
    }

    /// Record into the caller's graph.
    /// # Errors
    /// Rejects incorrect physical shapes or failed kernel submission.
    pub fn record(
        &self,
        attention: &dyn Attention,
        recording: &Recording<'_>,
        buffers: DenseBuffers<'_>,
    ) -> Result<()> {
        let operands = [buffers.query, buffers.key, buffers.value, buffers.out];
        if operands
            .iter()
            .zip(self.shapes)
            .any(|(buffer, [rows, columns])| buffer.len() != rows * columns)
        {
            return Err(device_error("attention buffer shape mismatch"));
        }
        let plan = DensePlan {
            head_width: self.width,
            online: true,
            tiles: [self.tile, self.tile, 0],
            group: self.descriptor.query_heads / self.descriptor.kv_heads,
            value_width: self.value_width,
            mask: self.mask[0],
            query_rows: self.shapes[0][0],
            key_rows: self.shapes[1][0],
            tokens: self.keys,
            scale: self.descriptor.scale,
            frame_tokens: self.mask[1],
            queries: self.queries,
            query_start: self.mask[2],
            left: self.mask[3],
            right: self.mask[4],
        };
        if self.descriptor.queries == 1 {
            attention.dense_decode(recording, plan, buffers)
        } else {
            attention.dense(recording, plan, buffers)
        }
        .map_err(|error| device_error(error.to_string()))
    }
}

pub const fn tile_rows(queries: usize) -> usize {
    if queries >= TILE_CROSSOVER {
        WIDE_TILE
    } else {
        TILE
    }
}

fn index(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::unsupported("attention device index overflow"))
}

fn padded_width(width: usize) -> Result<usize> {
    let width = width
        .checked_next_power_of_two()
        .ok_or_else(|| Error::invalid("attention width overflow"))?
        .max(TILE);
    if width > MAX_WIDTH {
        return Err(Error::unsupported("attention provider head width limit"));
    }
    Ok(width)
}

fn mask_arguments(mask: AttentionMask) -> Result<[i32; MASK_FIELDS]> {
    let offset = |value| {
        i32::try_from(value).map_err(|_| Error::unsupported("attention causal offset overflow"))
    };
    Ok(match mask {
        AttentionMask::None => [0, 1, 0, 0, 0],
        AttentionMask::Causal { query_start } => [1, 1, offset(query_start)?, 0, 0],
        AttentionMask::Window {
            query_start,
            left,
            right,
        } => [
            WINDOW_MASK,
            1,
            offset(query_start)?,
            index(left)?,
            index(right)?,
        ],
        AttentionMask::Segments { tokens } => [SEGMENT_MASK, index(tokens)?, 0, 0, 0],
    })
}
