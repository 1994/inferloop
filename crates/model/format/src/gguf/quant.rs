//! Reference decoding for the GGUF scalar and grouped-int tensor types.
//!
//! These routines define the numerical contract the device kernels must match. They are not the
//! production weight path: inference keeps the stored blocks and reads them in place. Grouped
//! types decode through the same [`super::grouped`] decoder the device packers use.
use super::grouped::dequantize_grouped;
use super::tensor::GgufType;
use infer_core::{Error, Result};

/// Bytes per F32 value.
const F32_BYTES: usize = 4;
/// Bytes per 16-bit value.
const F16_BYTES: usize = 2;

/// Decode `bytes` into `out`, which must hold exactly the stored element count.
///
/// Supported layouts are F32/F16/BF16 and every [`infer_ir::GroupedIntFormat`]. Other stored types
/// fail explicitly.
///
/// # Errors
/// Returns an unsupported error for a type this crate cannot decode, and an invalid-input error
/// when the byte or output length disagrees with the type's block geometry.
pub fn dequantize(ty: GgufType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let expected = ty.byte_len(out.len() as u64)?;
    if expected != bytes.len() as u64 {
        return Err(Error::invalid(format!(
            "GGUF {} payload is {} bytes, expected {expected} for {} values",
            ty.name(),
            bytes.len(),
            out.len()
        )));
    }
    match ty {
        GgufType::F32 => {
            scalar_f32(bytes, out);
            Ok(())
        }
        GgufType::F16 => {
            scalar_f16(bytes, out);
            Ok(())
        }
        GgufType::Bf16 => {
            scalar_bf16(bytes, out);
            Ok(())
        }
        other => match other.grouped_int() {
            Some(format) => dequantize_grouped(format, bytes, out),
            None => Err(Error::unsupported(format!(
                "GGUF {} decoding is not implemented",
                other.name()
            ))),
        },
    }
}

fn scalar_f32(bytes: &[u8], out: &mut [f32]) {
    for (value, chunk) in out.iter_mut().zip(bytes.as_chunks::<F32_BYTES>().0) {
        *value = f32::from_le_bytes(*chunk);
    }
}

fn scalar_f16(bytes: &[u8], out: &mut [f32]) {
    for (value, chunk) in out.iter_mut().zip(bytes.as_chunks::<F16_BYTES>().0) {
        *value = half_to_f32(u16::from_le_bytes(*chunk));
    }
}

fn scalar_bf16(bytes: &[u8], out: &mut [f32]) {
    for (value, chunk) in out.iter_mut().zip(bytes.as_chunks::<F16_BYTES>().0) {
        *value = f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << HIGH_HALF_SHIFT);
    }
}

/// IEEE 754 binary16 to binary32, preserving subnormals and infinities.
#[must_use]
pub fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & F16_SIGN_MASK) << HIGH_HALF_SHIFT;
    let exponent = (bits >> F16_MANTISSA_BITS) & F16_EXPONENT_MASK;
    let mantissa = u32::from(bits & F16_MANTISSA_MASK);
    let output = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let shift = mantissa.leading_zeros() - F16_SUBNORMAL_SHIFT;
            sign | ((F16_SUBNORMAL_EXPONENT - shift) << F32_MANTISSA_BITS)
                | ((mantissa << shift & u32::from(F16_MANTISSA_MASK)) << F16_MANTISSA_TO_F32_SHIFT)
        }
        F16_MAX_EXPONENT => sign | F32_EXPONENT_FIELD | (mantissa << F16_MANTISSA_TO_F32_SHIFT),
        _ => {
            sign | ((u32::from(exponent) + F16_TO_F32_DELTA) << F32_MANTISSA_BITS)
                | (mantissa << F16_MANTISSA_TO_F32_SHIFT)
        }
    };
    f32::from_bits(output)
}

/// Sign bit of a binary16 value.
const F16_SIGN_MASK: u16 = 0x8000;
/// Exponent field of a binary16 value.
const F16_EXPONENT_MASK: u16 = 0x1F;
/// Fraction field of a binary16 value.
const F16_MANTISSA_MASK: u16 = 0x3FF;
/// Biased exponent marking infinity or NaN in binary16.
const F16_MAX_EXPONENT: u16 = 31;
/// Fraction bits in a binary16 value.
const F16_MANTISSA_BITS: u32 = 10;
/// Exponent bias of binary16.
const F16_EXPONENT_BIAS: u32 = 15;
/// Fraction bits in binary32.
const F32_MANTISSA_BITS: u32 = 23;
/// Exponent bias of binary32.
const F32_EXPONENT_BIAS: u32 = 127;
/// In-place exponent field of binary32.
const F32_EXPONENT_FIELD: u32 = 0x7F80_0000;
/// Shift moving a binary16 bit pattern into the high half of binary32.
const HIGH_HALF_SHIFT: u32 = 16;
/// Leading-zero count of the largest binary16 subnormal.
const F16_SUBNORMAL_SHIFT: u32 = u32::BITS - 1 - F16_MANTISSA_BITS;
/// Exponent delta from binary16 to binary32.
const F16_TO_F32_DELTA: u32 = F32_EXPONENT_BIAS - F16_EXPONENT_BIAS;
/// binary32 exponent of the largest binary16 subnormal.
const F16_SUBNORMAL_EXPONENT: u32 = F16_TO_F32_DELTA + 1;
/// Shift aligning a binary16 fraction with the binary32 fraction field.
const F16_MANTISSA_TO_F32_SHIFT: u32 = F32_MANTISSA_BITS - F16_MANTISSA_BITS;
