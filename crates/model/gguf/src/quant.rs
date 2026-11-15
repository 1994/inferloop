//! Reference decoding for the GGUF scalar and k-quant tensor types.
//!
//! These routines define the numerical contract the device kernels must match. They are not the
//! production weight path: inference keeps the stored blocks and reads them in place.
use crate::tensor::GgufType;
use infer_core::{Error, Result};

/// Elements in one k-quant super-block.
const QK_K: usize = 256;
/// Elements in one legacy quant block.
const QK_LEGACY: usize = 32;
/// Stored bytes of one `Q4_0` block.
const Q4_0_BLOCK: usize = 18;
/// Stored bytes of one `Q4_K` block.
const Q4_K_BLOCK: usize = 144;
/// Stored bytes of one `Q5_K` block.
const Q5_K_BLOCK: usize = 176;
/// Stored bytes of one `Q6_K` block.
const Q6_K_BLOCK: usize = 210;
/// Bytes of the packed k-quant scale/min table.
const K_SCALE_BYTES: usize = 12;
/// Six bits of each packed k-quant scale/min entry.
const K_SCALE_MASK: u8 = 63;
/// Shift selecting the two high bits packed into a scale byte.
const K_SCALE_HIGH_SHIFT: u32 = 6;
/// Bytes of a k-quant block's `d` and `dmin` fields.
const K_BLOCK_HEADER: usize = 4;
/// Scale/min pairs consumed by each 64-value group.
const K_QUANT_GROUPS: usize = 4;
/// Values in one k-quant group.
const K_QUANT_GROUP: usize = 64;
/// Values in one `Q6_K` half-super-block.
const K_QUANT_HALF: usize = 128;
/// Lanes in one quant group or `Q6_K` sub-block.
const K_QUANT_LANES: usize = 32;
/// Low nibble of a packed byte.
const NIBBLE_MASK: u8 = 0xF;
/// Width of one nibble.
const NIBBLE_BITS: u32 = 4;
/// Legacy `Q4_0` code bias.
const Q4_0_ZERO_POINT: i8 = 8;
/// Offset of a `Q4_K` block's packed quants.
const Q4_K_QS_OFFSET: usize = K_BLOCK_HEADER + K_SCALE_BYTES;
/// Offset of a `Q5_K` block's high bits.
const Q5_K_QH_OFFSET: usize = Q4_K_QS_OFFSET;
/// Bytes of a `Q5_K` block's high-bit plane.
const Q5_K_QH_BYTES: usize = K_QUANT_LANES;
/// Offset of a `Q5_K` block's packed quants.
const Q5_K_QS_OFFSET: usize = Q5_K_QH_OFFSET + Q5_K_QH_BYTES;
/// Lowest high-bit mask for the first `Q5_K` group.
const Q5_LOW_BIT: u8 = 1;
/// Second high-bit mask for the first `Q5_K` group.
const Q5_HIGH_BIT: u8 = 2;
/// Bits advanced by each `Q5_K` group.
const Q5_BIT_SHIFT: usize = 2;
/// Value added by a set `Q5_K` high bit.
const Q5_HIGH_RANGE: u8 = 16;
/// Offset of a `Q6_K` block's high-bit plane.
const Q6_QH_OFFSET: usize = K_QUANT_HALF;
/// Offset of a `Q6_K` block's scales.
const Q6_SCALES_OFFSET: usize = 192;
/// Offset of a `Q6_K` block's final scale.
const Q6_D_OFFSET: usize = 208;
/// Bytes of `Q6_K` low nibbles per half-super-block.
const Q6_HALF_QL: usize = 64;
/// Bytes of `Q6_K` high bits per half-super-block.
const Q6_HALF_QH: usize = K_QUANT_LANES;
/// Bytes of `Q6_K` scales per half-super-block.
const Q6_HALF_SCALES: usize = 8;
/// Lanes sharing one `Q6_K` scale pair.
const Q6_LANES_PER_SCALE: usize = 16;
/// Two bits of `Q6_K` high bits per lane.
const Q6_QH_MASK: u8 = 3;
/// `Q6_K` code zero point.
const Q6_ZERO_POINT: i32 = 32;
/// Scale indices for the four `Q6_K` outputs of one lane.
const Q6_SCALE_OFFSETS: [usize; K_QUANT_GROUPS] = [0, 2, 4, 6];
/// `Q6_K` output offsets within one half-super-block.
const Q6_OUTPUT_OFFSETS: [usize; K_QUANT_GROUPS] = [0, K_QUANT_LANES, K_QUANT_GROUP, 96];
/// Bytes per F32 value.
const F32_BYTES: usize = 4;
/// Bytes per 16-bit value.
const F16_BYTES: usize = 2;

/// Decode `bytes` into `out`, which must hold exactly the stored element count.
///
/// Supported layouts are F32/F16/BF16, `Q4_0`, and the `Q4_K`, `Q5_K` and `Q6_K` super-block types used
/// by `Q4_K_M` checkpoints. Other stored types fail explicitly.
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
        GgufType::Q4K => {
            q4_k(bytes, out);
            Ok(())
        }
        GgufType::Q5K => {
            q5_k(bytes, out);
            Ok(())
        }
        GgufType::Q6K => {
            q6_k(bytes, out);
            Ok(())
        }
        GgufType::Q4_0 => {
            q4_0(bytes, out);
            Ok(())
        }
        other => Err(Error::unsupported(format!(
            "GGUF {} decoding is not implemented",
            other.name()
        ))),
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

fn q4_0(bytes: &[u8], out: &mut [f32]) {
    for (block, values) in bytes
        .as_chunks::<Q4_0_BLOCK>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_LEGACY>().0.iter_mut())
    {
        let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (index, byte) in block[2..].iter().enumerate() {
            values[index * 2] = d * f32::from((byte & NIBBLE_MASK).cast_signed() - Q4_0_ZERO_POINT);
            values[index * 2 + 1] =
                d * f32::from((byte >> NIBBLE_BITS).cast_signed() - Q4_0_ZERO_POINT);
        }
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "the reference decoder rounds the multiply and subtract separately"
)]
fn q4_k(bytes: &[u8], out: &mut [f32]) {
    for (block, values) in bytes
        .as_chunks::<Q4_K_BLOCK>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_K>().0.iter_mut())
    {
        let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[K_BLOCK_HEADER..K_BLOCK_HEADER + K_SCALE_BYTES];
        let qs = &block[Q4_K_QS_OFFSET..];
        for group in 0..K_QUANT_GROUPS {
            let (sc1, m1) = scale_min_k4(group * 2, scales);
            let (sc2, m2) = scale_min_k4(group * 2 + 1, scales);
            let (d1, m1) = (d * f32::from(sc1), min * f32::from(m1));
            let (d2, m2) = (d * f32::from(sc2), min * f32::from(m2));
            let quants = &qs[group * K_QUANT_LANES..][..K_QUANT_LANES];
            for lane in 0..K_QUANT_LANES {
                values[group * K_QUANT_GROUP + lane] =
                    d1 * f32::from(quants[lane] & NIBBLE_MASK) - m1;
                values[group * K_QUANT_GROUP + K_QUANT_LANES + lane] =
                    d2 * f32::from(quants[lane] >> NIBBLE_BITS) - m2;
            }
        }
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "the reference decoder rounds the multiply and subtract separately"
)]
fn q5_k(bytes: &[u8], out: &mut [f32]) {
    for (block, values) in bytes
        .as_chunks::<Q5_K_BLOCK>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_K>().0.iter_mut())
    {
        let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[K_BLOCK_HEADER..K_BLOCK_HEADER + K_SCALE_BYTES];
        let qh = &block[Q5_K_QH_OFFSET..Q5_K_QH_OFFSET + Q5_K_QH_BYTES];
        let qs = &block[Q5_K_QS_OFFSET..];
        for group in 0..K_QUANT_GROUPS {
            let (sc1, m1) = scale_min_k4(group * 2, scales);
            let (sc2, m2) = scale_min_k4(group * 2 + 1, scales);
            let (d1, m1) = (d * f32::from(sc1), min * f32::from(m1));
            let (d2, m2) = (d * f32::from(sc2), min * f32::from(m2));
            let low_bit = Q5_LOW_BIT << (Q5_BIT_SHIFT * group);
            let high_bit = Q5_HIGH_BIT << (Q5_BIT_SHIFT * group);
            let quants = &qs[group * K_QUANT_LANES..][..K_QUANT_LANES];
            for lane in 0..K_QUANT_LANES {
                let low = f32::from(
                    (quants[lane] & NIBBLE_MASK)
                        + if qh[lane] & low_bit != 0 {
                            Q5_HIGH_RANGE
                        } else {
                            0
                        },
                );
                let high = f32::from(
                    (quants[lane] >> NIBBLE_BITS)
                        + if qh[lane] & high_bit != 0 {
                            Q5_HIGH_RANGE
                        } else {
                            0
                        },
                );
                values[group * K_QUANT_GROUP + lane] = d1 * low - m1;
                values[group * K_QUANT_GROUP + K_QUANT_LANES + lane] = d2 * high - m2;
            }
        }
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Q6_K codes are small integers exactly representable in F32"
)]
fn q6_k(bytes: &[u8], out: &mut [f32]) {
    for (block, values) in bytes
        .as_chunks::<Q6_K_BLOCK>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_K>().0.iter_mut())
    {
        let d = half_to_f32(u16::from_le_bytes([
            block[Q6_D_OFFSET],
            block[Q6_D_OFFSET + 1],
        ]));
        for half in 0..2 {
            let ql = &block[half * Q6_HALF_QL..][..Q6_HALF_QL];
            let qh = &block[Q6_QH_OFFSET + half * Q6_HALF_QH..][..Q6_HALF_QH];
            let scales = &block[Q6_SCALES_OFFSET + half * Q6_HALF_SCALES..][..Q6_HALF_SCALES];
            let base = half * K_QUANT_HALF;
            for lane in 0..K_QUANT_LANES {
                let is = lane / Q6_LANES_PER_SCALE;
                let q1 = (i32::from(ql[lane] & NIBBLE_MASK)
                    | (i32::from(qh[lane] & Q6_QH_MASK) << NIBBLE_BITS))
                    - Q6_ZERO_POINT;
                let q2 = (i32::from(ql[lane + K_QUANT_LANES] & NIBBLE_MASK)
                    | (i32::from((qh[lane] >> 2) & Q6_QH_MASK) << NIBBLE_BITS))
                    - Q6_ZERO_POINT;
                let q3 = (i32::from(ql[lane] >> NIBBLE_BITS)
                    | (i32::from((qh[lane] >> NIBBLE_BITS) & Q6_QH_MASK) << NIBBLE_BITS))
                    - Q6_ZERO_POINT;
                let q4 = (i32::from(ql[lane + K_QUANT_LANES] >> NIBBLE_BITS)
                    | (i32::from((qh[lane] >> K_SCALE_HIGH_SHIFT) & Q6_QH_MASK) << NIBBLE_BITS))
                    - Q6_ZERO_POINT;
                for index in 0..K_QUANT_GROUPS {
                    let value = match index {
                        0 => q1,
                        1 => q2,
                        2 => q3,
                        _ => q4,
                    };
                    values[base + lane + Q6_OUTPUT_OFFSETS[index]] =
                        d * scale(scales, is + Q6_SCALE_OFFSETS[index]) * value as f32;
                }
            }
        }
    }
}

/// One signed `Q6_K` sub-block scale.
fn scale(scales: &[u8], index: usize) -> f32 {
    f32::from(scales[index].cast_signed())
}

/// Decode the 6-bit scale and minimum pair at `index` from a k-quant scale table.
const fn scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < K_BLOCK_HEADER {
        (
            scales[index] & K_SCALE_MASK,
            scales[index + K_BLOCK_HEADER] & K_SCALE_MASK,
        )
    } else {
        (
            (scales[index + K_BLOCK_HEADER] & NIBBLE_MASK)
                | ((scales[index - K_BLOCK_HEADER] >> K_SCALE_HIGH_SHIFT) << NIBBLE_BITS),
            (scales[index + K_BLOCK_HEADER] >> NIBBLE_BITS)
                | ((scales[index] >> K_SCALE_HIGH_SHIFT) << NIBBLE_BITS),
        )
    }
}

/// IEEE 754 binary16 to binary32, preserving subnormals and infinities.
fn half_to_f32(bits: u16) -> f32 {
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
