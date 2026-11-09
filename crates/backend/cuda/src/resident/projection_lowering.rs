//! One lowering for every projection. The weight's format, its declared activation format and
//! the capture mode decide what records before anything does, so the modes that capture a
//! projection (one lane, gathered verify rows, slot rows in place, prompt tiles, the prompt
//! logits row) cannot drift apart: each passes its mode and records the decision.
use super::capture::error;
use super::program::ActivationQuantization;
use crate::constants::{
    FUSED_VERIFY_LANES, GEMV_WIDE_OUTPUT_RATIO, NVFP4_GROUP_SIZE, SMALL_GEMV_MAX_ROWS,
};
use crate::mlp::ProjectionWeight;
use crate::tuning::pool_dense::{self, Choice as PoolDense};
use infer_core::Result;
use infer_cuda_kernels::contract::RowsKernel;

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

/// How a capture mode presents a projection's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    /// One lane's row, as single-lane decode records it.
    Single,
    /// Every lane's row gathered into one matrix; `arithmetic_rows` is the row count the kernel
    /// choice prices.
    Rows { arithmetic_rows: usize },
    /// Slot rows read in place. `pool` is the tuned pooled dense choice for the slot geometry,
    /// and `priced` says whether the caller prices fewer rows than it holds.
    Slots {
        arithmetic_rows: usize,
        priced: bool,
        pool: PoolDense,
    },
    /// Prompt tiles of the batched prefill.
    Prompt,
    /// The single prompt-logits row, which only a quantized head reaches.
    Logits,
}

/// The encoding of a projection weight, which is all the lowering reads of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Format {
    Dense,
    Fp8,
    Fp8Block,
    Fp4,
}

impl Format {
    pub(super) const fn of(weight: &ProjectionWeight) -> Self {
        match weight {
            ProjectionWeight::Dense(_) => Self::Dense,
            ProjectionWeight::Fp8(..) => Self::Fp8,
            ProjectionWeight::Fp8Block(..) => Self::Fp8Block,
            ProjectionWeight::Fp4(..) => Self::Fp4,
        }
    }
}

/// Activations quantized before the matching low-precision GEMM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Quantized {
    /// Block-16 E4M3 scales over packed FP4 weights, with the model's global scale.
    Fp4(f32),
    /// Per-token FP8 over FP8 weights.
    Fp8Token,
}

/// What records one projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Lowering {
    /// Quantize the activations, then the matching W4A4 or FP8 GEMM. Every mode but gathered
    /// rows lets a sibling projection of the same input reuse the quantized activations.
    Quantized(Quantized),
    /// The tuned pooled dense kernel over slot rows.
    PoolDense(PoolDense),
    /// The weight's own matrix-vector kernel with its tuned tiling.
    Matvec,
    /// One rows kernel over the weight's format.
    Rows(RowsKernel),
    /// The compensated dense diagnostic, which only internal tests enable.
    #[cfg(test)]
    Compensated,
    /// The prompt GEMM of a dense weight.
    PromptDense,
    /// The prompt GEMM of a channel FP8 weight.
    PromptFp8,
    /// The split-K prompt GEMM of a packed FP4 weight without activation scales.
    PromptFp4,
}

/// The quantized path `weight` takes under `activation`, when the two agree.
pub(super) const fn quantized(
    weight: Format,
    activation: Option<ActivationQuantization>,
) -> Option<Quantized> {
    match (weight, activation) {
        (Format::Fp4, Some(ActivationQuantization::Fp4(scale))) => Some(Quantized::Fp4(scale)),
        (Format::Fp8 | Format::Fp8Block, Some(ActivationQuantization::Fp8Token)) => {
            Some(Quantized::Fp8Token)
        }
        _ => None,
    }
}

/// Lower a `[rows, outputs, columns]` projection of `weight` captured in `mode`.
/// # Errors
/// A single lane or the logits row whose activation format does not match the weight, and a
/// block FP8 weight in a prompt without its resident FP8 GEMM.
pub(super) fn lower(
    weight: Format,
    activation: Option<ActivationQuantization>,
    mode: Mode,
    [rows, outputs, columns]: [usize; 3],
) -> Result<Lowering> {
    let quantized = quantized(weight, activation);
    let unquantized = |arithmetic_rows: usize| {
        #[cfg(test)]
        if weight == Format::Dense
            && super::small_gemm::precise_dense_enabled()
            && outputs <= columns.saturating_mul(GEMV_WIDE_OUTPUT_RATIO)
        {
            return Lowering::Compensated;
        }
        rows_lowering(weight, [outputs, columns], arithmetic_rows)
    };
    match mode {
        Mode::Single => {
            if activation.is_some() {
                return quantized
                    .map(Lowering::Quantized)
                    .ok_or_else(|| error("activation quantization and weight encoding disagree"));
            }
            Ok(Lowering::Matvec)
        }
        Mode::Logits => quantized
            .map(Lowering::Quantized)
            .ok_or_else(|| error("prompt logits quantized projection")),
        Mode::Prompt => Ok(match (quantized, weight) {
            (Some(quantized), _) => Lowering::Quantized(quantized),
            (None, Format::Fp8Block) => {
                return Err(error("block FP8 requires the resident FP8 GEMM"));
            }
            (None, Format::Dense) => Lowering::PromptDense,
            (None, Format::Fp8) => Lowering::PromptFp8,
            (None, Format::Fp4) => Lowering::PromptFp4,
        }),
        Mode::Slots {
            arithmetic_rows,
            priced,
            pool,
        } => {
            let shape = [rows, outputs, columns];
            if activation.is_none()
                && (!priced || pool_dense::supported(shape))
                && let Some(pool) = super::pool_gemv::lowered(weight, shape, pool)
            {
                return Ok(Lowering::PoolDense(pool));
            }
            Ok(quantized.map_or_else(|| unquantized(arithmetic_rows), Lowering::Quantized))
        }
        Mode::Rows { arithmetic_rows } => {
            Ok(quantized.map_or_else(|| unquantized(arithmetic_rows), Lowering::Quantized))
        }
    }
}

/// The rows kernel for unquantized activations. Tensor cores amortize weights across candidate
/// rows; the measured GEMV path stays for narrow batches with a very wide output, such as a
/// vocabulary head.
const fn rows_lowering(
    weight: Format,
    [outputs, columns]: [usize; 2],
    arithmetic_rows: usize,
) -> Lowering {
    let small = arithmetic_rows > SMALL_GEMV_MAX_ROWS
        || outputs <= columns.saturating_mul(GEMV_WIDE_OUTPUT_RATIO);
    if small && matches!(weight, Format::Dense | Format::Fp8) {
        Lowering::Rows(RowsKernel::Small)
    } else {
        Lowering::Rows(RowsKernel::Tiled {
            output_tile: VERIFY_OUTPUT_TILE,
            tile_output: VERIFY_TILE_COLUMNS,
            tile_columns: VERIFY_TILE_DEPTH,
            fp4_depths: [VERIFY_PACKED_DEPTH, VERIFY_SCALE_DEPTH],
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/projection_lowering.rs"]
mod tests;
