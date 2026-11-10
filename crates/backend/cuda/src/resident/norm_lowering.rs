//! One lowering for every normalization. The row layout of an ordinary norm and whether a
//! prompt norm fuses into its low-precision consumer are decided here, before anything records,
//! so the capture modes and the prompt plan cannot drift apart.
use super::fp8_quant_reduction::parallel_columns;
use super::norm_fp4::eligible_columns as fp4_columns;
use super::program::ActivationQuantization;
use super::projection_lowering::Format;
use infer_cuda_kernels::contract::NormLayout;

/// Rows per block of the row-tiled norm kernel, fixed in the kernel.
pub(super) const NORM_ROWS: usize = 16;
/// The widest dimension the row-tiled kernel holds in registers.
pub(super) const NORM_ROWS_MAX_DIMENSION: usize = 256;

/// The row layout of a norm over `heads` rows of `dimension` values. `heads` counts logical
/// serving rows, so physical funding does not change a pooled target's arithmetic; the row
/// kernel masks its final partial tile.
pub(super) const fn layout(dimension: usize, heads: usize) -> NormLayout {
    if dimension.is_power_of_two()
        && dimension <= NORM_ROWS_MAX_DIMENSION
        && heads >= NORM_ROWS * NORM_ROWS
        && heads.is_multiple_of(NORM_ROWS)
    {
        NormLayout::Rows { tile: NORM_ROWS }
    } else {
        NormLayout::Block
    }
}

/// A prompt RMS norm whose output feeds exactly one projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Norm {
    pub dimension: usize,
    pub epsilon: f32,
    pub offset: f32,
}

/// The projection consuming a prompt norm, as the fusion reads it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Consumer {
    /// The resolved BF16 profile without MTP fusion keeps the canonical dispatch otherwise.
    pub fuses: bool,
    pub format: Format,
    pub activation: Option<ActivationQuantization>,
    /// The consumer quantizes its input per token to FP8.
    pub fp8_input: bool,
    /// The prompt plan records the consumer's FP4 GEMM.
    pub planned_fp4: bool,
}

/// How a prompt norm fuses into its consumer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Fusion {
    /// Write the dense output and its FP4 codes and scales, with the model's global scale.
    Fp4(f32),
    /// Write the dense output and its row maxima for the token-FP8 quantizer.
    Maximum,
}

/// The fusion of `norm` into `consumer`, if any.
pub(super) fn fusion(norm: Norm, consumer: Consumer) -> Option<Fusion> {
    if !consumer.fuses {
        return None;
    }
    match (consumer.format, consumer.activation) {
        (Format::Fp4, Some(ActivationQuantization::Fp4(scale)))
            if consumer.planned_fp4
                && fp4_columns(norm.dimension)
                && norm.epsilon.is_finite()
                && norm.epsilon > 0.0
                && norm.offset.is_finite()
                && scale.is_finite()
                && scale > 0.0 =>
        {
            Some(Fusion::Fp4(scale))
        }
        (Format::Fp8, _) if consumer.fp8_input && parallel_columns(norm.dimension) => {
            Some(Fusion::Maximum)
        }
        _ => None,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/norm_lowering.rs"]
mod tests;
