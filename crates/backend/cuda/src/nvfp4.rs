//! Bounded CPU conversion for targets without a native FP4 element type.
//! This expands stored NVFP4 weights; it cannot recover the pre-quantization model.
use crate::constants::NVFP4_GROUP_SIZE;
use infer_core::{Error, Result};

/// E4M3 block scales above this code are non-finite.
const E4M3_MAX_FINITE_CODE: u8 = 126;
/// Bytes holding one packed group of FP4 nibbles.
const NVFP4_PACKED_GROUP_BYTES: usize = 8;
/// Nibble extraction within one packed byte.
const NIBBLE_LOW_MASK: u8 = 15;
/// Nibble extraction within one packed byte.
const NIBBLE_HIGH_SHIFT: u8 = 4;
/// BF16 round-to-nearest-even bias and exponent field.
const BF16_ROUND_BIAS: u32 = 0x7fff;
/// BF16 round-to-nearest-even bias and exponent field.
const BF16_MANTISSA_SHIFT: u32 = 16;
/// BF16 round-to-nearest-even bias and exponent field.
const BF16_EXPONENT_MASK: u16 = 0x7f80;
/// E2M1 value bits select from this magnitude table.
const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
/// E2M1 field masks.
const E2M1_VALUE_MASK: u8 = 7;
/// E2M1 field masks.
const E2M1_SIGN_MASK: u8 = 8;
/// E4M3 field layout: exponent shift, fraction mask and exponent bias.
const E4M3_EXPONENT_SHIFT: u8 = 3;
/// E4M3 field layout: exponent shift, fraction mask and exponent bias.
const E4M3_FRACTION_MASK: u8 = 7;
/// E4M3 field layout: exponent shift, fraction mask and exponent bias.
const E4M3_EXPONENT_BIAS: u8 = 7;
/// E4M3 subnormal and normal fraction divisors.
const E4M3_SUBNORMAL_DIVISOR: f32 = 512.0;
/// E4M3 subnormal and normal fraction divisors.
const E4M3_FRACTION_DIVISOR: f32 = 8.0;

/// Decode row-major E2M1 nibbles with E4M3 block-16 scales into BF16 bit patterns.
/// `global` is the checkpoint's inverse weight scale (as used by the native kernels).
///
/// # Errors
/// Rejects invalid geometry, byte budgets, scale encodings, and nonfinite BF16 results.
pub fn to_bf16(
    packed: &[u8],
    scales: &[u8],
    global: f32,
    rows: usize,
    columns: usize,
    byte_budget: usize,
) -> Result<Vec<u16>> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::invalid("NVFP4 decoded shape overflow"))?;
    if rows == 0
        || columns == 0
        || !columns.is_multiple_of(NVFP4_GROUP_SIZE)
        || elements.checked_mul(2).is_none_or(|n| n > byte_budget)
        || packed.len() != elements / 2
        || scales.len() != elements / NVFP4_GROUP_SIZE
        || !global.is_finite()
        || global <= 0.0
    {
        return Err(Error::invalid(
            "NVFP4 conversion shape, budget or global scale",
        ));
    }
    if scales.iter().any(|s| *s > E4M3_MAX_FINITE_CODE) {
        return Err(Error::invalid(
            "NVFP4 block scales must be finite nonnegative E4M3",
        ));
    }
    let mut result = Vec::with_capacity(elements);
    for (group, bytes) in packed
        .as_chunks::<NVFP4_PACKED_GROUP_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let factor = e4m3(scales[group]) / global;
        for byte in bytes {
            for code in [byte & NIBBLE_LOW_MASK, byte >> NIBBLE_HIGH_SHIFT] {
                let value = e2m1(code) * factor;
                if !value.is_finite() {
                    return Err(Error::invalid("NVFP4 conversion overflow"));
                }
                let bits = value.to_bits();
                let rounded = bits
                    .wrapping_add(BF16_ROUND_BIAS + ((bits >> BF16_MANTISSA_SHIFT) & 1))
                    >> BF16_MANTISSA_SHIFT;
                let rounded = u16::try_from(rounded).map_err(|e| Error::invalid(e.to_string()))?;
                if rounded & BF16_EXPONENT_MASK == BF16_EXPONENT_MASK {
                    return Err(Error::invalid("NVFP4 conversion exceeds BF16 range"));
                }
                result.push(rounded);
            }
        }
    }
    Ok(result)
}

fn e2m1(code: u8) -> f32 {
    let magnitude = E2M1_MAGNITUDES[usize::from(code & E2M1_VALUE_MASK)];
    if code & E2M1_SIGN_MASK == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn e4m3(code: u8) -> f32 {
    let exponent = code >> E4M3_EXPONENT_SHIFT;
    let fraction = f32::from(code & E4M3_FRACTION_MASK);
    if exponent == 0 {
        fraction / E4M3_SUBNORMAL_DIVISOR
    } else {
        (1.0 + fraction / E4M3_FRACTION_DIVISOR)
            * 2.0f32.powi(i32::from(exponent) - i32::from(E4M3_EXPONENT_BIAS))
    }
}

#[cfg(test)]
mod tests;
