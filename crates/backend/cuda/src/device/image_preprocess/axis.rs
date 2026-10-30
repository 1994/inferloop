//! Checked fixed-point bicubic coefficients for device image resampling.
use infer_core::{Error, ErrorCode, Result};

/// Radius of the bicubic reconstruction filter before antialiasing.
const FILTER_RADIUS: f64 = 2.0;
/// Pixel-center offset used by the reference image processor.
const HALF_PIXEL: f64 = 0.5;
/// Catmull–Rom cubic parameter.
const CUBIC_A: f64 = -0.5;
/// Linear coefficient of the near-span cubic.
const CUBIC_LINEAR: f64 = 3.0;
/// Pivot in the far-span cubic.
const CUBIC_PIVOT: f64 = 5.0;
/// Offset in the far-span cubic.
const CUBIC_OFFSET: f64 = 8.0;
/// Tail coefficient in the far-span cubic.
const CUBIC_TAIL: f64 = 4.0;
/// Largest signed coefficient scale accepted by the reference integer resampler.
const COEFFICIENT_LIMIT: f64 = 32_768.0;
/// Maximum fractional precision used by the reference resampler.
const MAX_PRECISION: usize = 22;
/// Maximum value of an input RGB8 channel.
const BYTE_MAX: i64 = 255;

/// One axis's row-major input indices and signed fixed-point weights.
pub(super) struct AxisTable {
    pub indices: Vec<i32>,
    pub weights: Vec<i32>,
    pub taps: usize,
    pub denominator: i32,
}

impl AxisTable {
    /// Construct the exact reference coefficients without imposing a fixed image-size limit.
    pub fn new(source: usize, target: usize) -> Result<Self> {
        let source_i32 = i32::try_from(source)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::invalid("image source axis is out of range"))?;
        let target_i32 = i32::try_from(target)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::invalid("image target axis is out of range"))?;
        let scale = f64::from(source_i32) / f64::from(target_i32);
        let filter_scale = scale.max(1.0);
        let bounds = bounds(source, target, scale, filter_scale)?;
        let taps = bounds
            .iter()
            .map(|(first, last)| last - first)
            .max()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::invalid("image filter has no taps"))?;
        let elements = target
            .checked_mul(taps)
            .ok_or_else(|| Error::invalid("image filter table size overflow"))?;
        let mut indices = zeroed(elements)?;
        let mut coefficients = zeroed(elements)?;
        let maximum = fill_coefficients(
            &bounds,
            taps,
            scale,
            filter_scale,
            &mut indices,
            &mut coefficients,
        )?;
        let denominator = denominator(maximum)?;
        let weights = quantize(&coefficients, taps, denominator)?;
        Ok(Self {
            indices,
            weights,
            taps,
            denominator,
        })
    }
}

fn zeroed<T: Clone + Default>(elements: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|error| {
        Error::new(
            ErrorCode::Capacity,
            format!("image filter allocation failed: {error}"),
        )
    })?;
    values.resize(elements, T::default());
    Ok(values)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "The finite tap bounds are clamped to the checked positive i32 source dimension"
)]
fn bounds(
    source: usize,
    target: usize,
    scale: f64,
    filter_scale: f64,
) -> Result<Vec<(usize, usize)>> {
    let mut result = Vec::new();
    result.try_reserve_exact(target).map_err(|error| {
        Error::new(
            ErrorCode::Capacity,
            format!("image filter bounds allocation failed: {error}"),
        )
    })?;
    let source_f64 = f64::from(
        i32::try_from(source).map_err(|_| Error::invalid("image source axis is out of range"))?,
    );
    for position in 0..target {
        let position = i32::try_from(position)
            .map_err(|_| Error::invalid("image filter position is out of range"))?;
        let center = (f64::from(position) + HALF_PIXEL) * scale;
        let support = FILTER_RADIUS * filter_scale;
        let first = (center - support + HALF_PIXEL)
            .trunc()
            .clamp(0.0, source_f64) as usize;
        let last = (center + support + HALF_PIXEL)
            .trunc()
            .clamp(0.0, source_f64) as usize;
        if first >= last {
            return Err(Error::invalid("image filter has empty bounds"));
        }
        result.push((first, last));
    }
    Ok(result)
}

fn fill_coefficients(
    bounds: &[(usize, usize)],
    taps: usize,
    scale: f64,
    filter_scale: f64,
    indices: &mut [i32],
    coefficients: &mut [f64],
) -> Result<f64> {
    let mut maximum = 0.0_f64;
    for (position, &(first, last)) in bounds.iter().enumerate() {
        let position_i32 = i32::try_from(position)
            .map_err(|_| Error::invalid("image filter position is out of range"))?;
        let center = (f64::from(position_i32) + HALF_PIXEL) * scale;
        let start = position * taps;
        let mut total = 0.0;
        for (slot, tap) in (first..last).enumerate() {
            let tap_i32 = i32::try_from(tap)
                .map_err(|_| Error::invalid("image filter index is out of range"))?;
            let weight = cubic((f64::from(tap_i32) + HALF_PIXEL - center) / filter_scale);
            indices[start + slot] = tap_i32;
            coefficients[start + slot] = weight;
            total += weight;
        }
        if total != 0.0 {
            for weight in &mut coefficients[start..start + last - first] {
                *weight /= total;
                maximum = maximum.max(*weight);
            }
        }
    }
    Ok(maximum)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Coefficient precision follows the reference operation order before integer rounding"
)]
fn denominator(maximum: f64) -> Result<i32> {
    let mut denominator = 1_i32;
    for _ in 0..MAX_PRECISION {
        if (maximum * f64::from(denominator) * 2.0 + HALF_PIXEL).trunc() >= COEFFICIENT_LIMIT {
            break;
        }
        denominator = denominator
            .checked_mul(2)
            .ok_or_else(|| Error::invalid("image filter precision overflow"))?;
    }
    Ok(denominator)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Rounded coefficients are checked against the i32 range before conversion"
)]
fn quantize(coefficients: &[f64], taps: usize, denominator: i32) -> Result<Vec<i32>> {
    let mut weights = zeroed(coefficients.len())?;
    for (weight, coefficient) in weights.iter_mut().zip(coefficients) {
        let value = (*coefficient * f64::from(denominator)).round();
        if !value.is_finite() || value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
            return Err(Error::invalid("image filter coefficient is out of range"));
        }
        *weight = value as i32;
    }
    for row in weights.chunks_exact(taps) {
        let _bound = row
            .iter()
            .try_fold(0_i64, |total, value| {
                total.checked_add(i64::from(*value).abs())
            })
            .and_then(|total| total.checked_mul(BYTE_MAX))
            .and_then(|total| total.checked_add(i64::from(denominator / 2)))
            .ok_or_else(|| Error::invalid("image filter accumulator overflow"))?;
    }
    Ok(weights)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Keep the independent processor's operation order before coefficient rounding"
)]
fn cubic(value: f64) -> f64 {
    let value = value.abs();
    if value <= 1.0 {
        ((CUBIC_A + 2.0) * value - (CUBIC_A + CUBIC_LINEAR)) * value * value + 1.0
    } else if value < 2.0 {
        (((value - CUBIC_PIVOT) * value + CUBIC_OFFSET) * value - CUBIC_TAIL) * CUBIC_A
    } else {
        0.0
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/vision_image_axis.rs"]
mod tests;
