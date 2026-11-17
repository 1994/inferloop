//! GGML block layouts decoded to grouped codes, factors and biases.
//!
//! This is the only place that knows where each format keeps its bits. Every format produces one
//! unsigned code per value and one F32 factor (plus bias, when the format has a minimum) per
//! [`GroupedIntFormat::group_values`] values, so the reference decoder and every device packer
//! share one decode and differ only in how they lay the result out.
use super::quant::half_to_f32;
use super::tensor::GgufType;
use infer_core::{Error, Result};
use infer_ir::{GROUPED_INT_MAX_GROUPS, GROUPED_INT_SPAN, GroupedIntFormat};

/// Values in one legacy block.
const LEGACY_VALUES: usize = 32;
/// Packed code bytes of one legacy 4/5-bit block.
const LEGACY_NIBBLES: usize = LEGACY_VALUES / 2;
/// Bytes of one F16 field.
const HALF_BYTES: usize = 2;
/// Bytes of a legacy 5-bit block's high-bit word.
const LEGACY_HIGH_BYTES: usize = 4;
/// Low nibble of a packed byte.
const NIBBLE_MASK: u8 = 0xF;
/// Width of one nibble.
const NIBBLE_BITS: u32 = 4;
/// Value a set fifth bit adds to a code.
const FIFTH_BIT: u8 = 1 << NIBBLE_BITS;
/// Flipping the sign bit adds 128 to a two's-complement byte.
const SIGN_BIT: u8 = 0x80;
/// Bytes of a k-quant block's `d` and `dmin` fields.
const K_HEADER: usize = 4;
/// Bytes of the packed k-quant scale/min table.
const K_SCALE_BYTES: usize = 12;
/// Six bits of each packed k-quant scale/min entry.
const K_SCALE_MASK: u8 = 63;
/// Shift selecting the two high bits packed into a scale byte.
const K_SCALE_HIGH_SHIFT: u32 = 6;
/// Offset of a `Q4_K`/`Q5_K` block's first payload after the scale table.
const K_PAYLOAD: usize = K_HEADER + K_SCALE_BYTES;
/// 64-value groups in one super-block; each holds two scale/min groups.
const K_PAIRS: usize = 4;
/// Lanes in one 32-value group.
const K_LANES: usize = 32;
/// Bytes of a `Q5_K` block's high-bit plane.
const Q5_K_HIGH_BYTES: usize = K_LANES;
/// Offset of a `Q6_K` block's high-bit plane.
const Q6_HIGH_OFFSET: usize = 128;
/// Offset of a `Q6_K` block's signed scales.
const Q6_SCALES_OFFSET: usize = 192;
/// Offset of a `Q6_K` block's `d`.
const Q6_D_OFFSET: usize = 208;
/// Values in one `Q6_K` half-super-block.
const Q6_HALF: usize = 128;
/// Low-nibble bytes per `Q6_K` half.
const Q6_HALF_LOW: usize = 64;
/// Two high bits per `Q6_K` code.
const Q6_HIGH_MASK: u8 = 3;
/// Shifts of the high-bit pairs for the four codes one `Q6_K` lane produces, 32 values apart.
const Q6_HIGH_SHIFTS: [u32; 4] = [0, 2, 4, 6];

/// One [`GROUPED_INT_SPAN`] of decoded codes and per-group terms.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupedSpan {
    /// One unsigned code per value.
    pub codes: [u8; GROUPED_INT_SPAN],
    /// Per-group factors; the first `format.span_groups()` entries are meaningful.
    pub factors: [f32; GROUPED_INT_MAX_GROUPS],
    /// Per-group biases; zero for formats without a minimum.
    pub biases: [f32; GROUPED_INT_MAX_GROUPS],
}

impl Default for GroupedSpan {
    fn default() -> Self {
        Self {
            codes: [0; GROUPED_INT_SPAN],
            factors: [0.0; GROUPED_INT_MAX_GROUPS],
            biases: [0.0; GROUPED_INT_MAX_GROUPS],
        }
    }
}

impl GgufType {
    /// The grouped-int family this stored type decodes as, if any.
    #[must_use]
    pub const fn grouped_int(self) -> Option<GroupedIntFormat> {
        Some(match self {
            Self::Q4_0 => GroupedIntFormat::Q4_0,
            Self::Q4_1 => GroupedIntFormat::Q4_1,
            Self::Q5_0 => GroupedIntFormat::Q5_0,
            Self::Q5_1 => GroupedIntFormat::Q5_1,
            Self::Q8_0 => GroupedIntFormat::Q8_0,
            Self::Q4K => GroupedIntFormat::Q4K,
            Self::Q5K => GroupedIntFormat::Q5K,
            Self::Q6K => GroupedIntFormat::Q6K,
            _ => return None,
        })
    }
}

/// The stored GGUF type of `format`.
#[must_use]
pub const fn stored_type(format: GroupedIntFormat) -> GgufType {
    match format {
        GroupedIntFormat::Q4_0 => GgufType::Q4_0,
        GroupedIntFormat::Q4_1 => GgufType::Q4_1,
        GroupedIntFormat::Q5_0 => GgufType::Q5_0,
        GroupedIntFormat::Q5_1 => GgufType::Q5_1,
        GroupedIntFormat::Q8_0 => GgufType::Q8_0,
        GroupedIntFormat::Q4K => GgufType::Q4K,
        GroupedIntFormat::Q5K => GgufType::Q5K,
        GroupedIntFormat::Q6K => GgufType::Q6K,
    }
}

/// Stored bytes of one [`GROUPED_INT_SPAN`] of `format`.
/// # Errors
/// Fails only if the type table and the span disagree, which the table forbids.
pub fn span_bytes(format: GroupedIntFormat) -> Result<usize> {
    let bytes = stored_type(format).byte_len(GROUPED_INT_SPAN as u64)?;
    usize::try_from(bytes).map_err(|_| Error::invariant("GGUF span bytes exceed usize"))
}

/// Decode one stored span of `format` into `span`.
/// # Errors
/// Rejects a payload that is not exactly [`span_bytes`] long.
pub fn decode_span(format: GroupedIntFormat, bytes: &[u8], span: &mut GroupedSpan) -> Result<()> {
    let expected = span_bytes(format)?;
    if bytes.len() != expected {
        return Err(Error::invalid(format!(
            "{} span is {} bytes, expected {expected}",
            format.label(),
            bytes.len(),
        )));
    }
    span.biases = [0.0; GROUPED_INT_MAX_GROUPS];
    match format {
        GroupedIntFormat::Q4_0 | GroupedIntFormat::Q4_1 => legacy_nibbles(format, bytes, span),
        GroupedIntFormat::Q5_0 | GroupedIntFormat::Q5_1 => legacy_five(format, bytes, span),
        GroupedIntFormat::Q8_0 => legacy_bytes(bytes, span),
        GroupedIntFormat::Q4K | GroupedIntFormat::Q5K => k_quant(format, bytes, span),
        GroupedIntFormat::Q6K => q6_k(bytes, span),
    }
    Ok(())
}

/// Decode a whole tensor to F32 values through [`GroupedIntFormat::value`].
///
/// A legacy tensor may end in a partial span; its stored blocks are decoded from a zero-padded
/// copy and only the stored values are written.
/// # Errors
/// Rejects a payload whose length disagrees with `out`.
pub(super) fn dequantize_grouped(
    format: GroupedIntFormat,
    bytes: &[u8],
    out: &mut [f32],
) -> Result<()> {
    let stride = span_bytes(format)?;
    let mut span = GroupedSpan::default();
    let mut padded = Vec::new();
    for (stored, values) in bytes.chunks(stride).zip(out.chunks_mut(GROUPED_INT_SPAN)) {
        let stored = if stored.len() == stride {
            stored
        } else {
            padded.clear();
            padded.extend_from_slice(stored);
            padded.resize(stride, 0);
            padded.as_slice()
        };
        decode_span(format, stored, &mut span)?;
        write_values(format, &span, values);
    }
    Ok(())
}

fn write_values(format: GroupedIntFormat, span: &GroupedSpan, values: &mut [f32]) {
    let group = format.group_values();
    for (index, value) in values.iter_mut().enumerate() {
        *value = format.value(
            span.codes[index],
            span.factors[index / group],
            span.biases[index / group],
        );
    }
}

fn half(bytes: &[u8], offset: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

/// Header fields of one legacy block: the scale, then the minimum when the format has one.
fn legacy_header(format: GroupedIntFormat, block: &[u8], span: &mut GroupedSpan, index: usize) {
    span.factors[index] = half(block, 0);
    if format.has_biases() {
        // The stored minimum is added; the shared form subtracts the bias.
        span.biases[index] = -half(block, HALF_BYTES);
    }
}

const fn legacy_header_bytes(format: GroupedIntFormat) -> usize {
    if format.has_biases() {
        2 * HALF_BYTES
    } else {
        HALF_BYTES
    }
}

fn legacy_nibbles(format: GroupedIntFormat, bytes: &[u8], span: &mut GroupedSpan) {
    let header = legacy_header_bytes(format);
    let stride = header + LEGACY_NIBBLES;
    for (index, block) in bytes.chunks_exact(stride).enumerate() {
        legacy_header(format, block, span, index);
        let codes = &mut span.codes[index * LEGACY_VALUES..][..LEGACY_VALUES];
        for (lane, byte) in block[header..].iter().enumerate() {
            codes[lane] = byte & NIBBLE_MASK;
            codes[lane + LEGACY_NIBBLES] = byte >> NIBBLE_BITS;
        }
    }
}

fn legacy_five(format: GroupedIntFormat, bytes: &[u8], span: &mut GroupedSpan) {
    let header = legacy_header_bytes(format);
    let stride = header + LEGACY_HIGH_BYTES + LEGACY_NIBBLES;
    for (index, block) in bytes.chunks_exact(stride).enumerate() {
        legacy_header(format, block, span, index);
        let mut word = [0u8; LEGACY_HIGH_BYTES];
        word.copy_from_slice(&block[header..header + LEGACY_HIGH_BYTES]);
        let high = u32::from_le_bytes(word);
        let codes = &mut span.codes[index * LEGACY_VALUES..][..LEGACY_VALUES];
        for (lane, byte) in block[header + LEGACY_HIGH_BYTES..].iter().enumerate() {
            let low_bit = (high >> lane) & 1 != 0;
            let high_bit = (high >> (lane + LEGACY_NIBBLES)) & 1 != 0;
            codes[lane] = (byte & NIBBLE_MASK) | if low_bit { FIFTH_BIT } else { 0 };
            codes[lane + LEGACY_NIBBLES] =
                (byte >> NIBBLE_BITS) | if high_bit { FIFTH_BIT } else { 0 };
        }
    }
}

fn legacy_bytes(bytes: &[u8], span: &mut GroupedSpan) {
    let stride = HALF_BYTES + LEGACY_VALUES;
    for (index, block) in bytes.chunks_exact(stride).enumerate() {
        span.factors[index] = half(block, 0);
        let codes = &mut span.codes[index * LEGACY_VALUES..][..LEGACY_VALUES];
        for (code, byte) in codes.iter_mut().zip(&block[HALF_BYTES..]) {
            *code = byte ^ SIGN_BIT;
        }
    }
}

fn k_quant(format: GroupedIntFormat, block: &[u8], span: &mut GroupedSpan) {
    let d = half(block, 0);
    let dmin = half(block, HALF_BYTES);
    let scales = &block[K_HEADER..K_PAYLOAD];
    let (high, low) = if format == GroupedIntFormat::Q5K {
        (
            Some(&block[K_PAYLOAD..K_PAYLOAD + Q5_K_HIGH_BYTES]),
            &block[K_PAYLOAD + Q5_K_HIGH_BYTES..],
        )
    } else {
        (None, &block[K_PAYLOAD..])
    };
    for pair in 0..K_PAIRS {
        let quants = &low[pair * K_LANES..][..K_LANES];
        for part in 0..2 {
            let group = pair * 2 + part;
            let (scale, min) = scale_min(group, scales);
            span.factors[group] = d * f32::from(scale);
            span.biases[group] = dmin * f32::from(min);
            let bit = 1u8 << group;
            for lane in 0..K_LANES {
                let nibble = if part == 0 {
                    quants[lane] & NIBBLE_MASK
                } else {
                    quants[lane] >> NIBBLE_BITS
                };
                let fifth = high.is_some_and(|plane| plane[lane] & bit != 0);
                span.codes[group * K_LANES + lane] = nibble | if fifth { FIFTH_BIT } else { 0 };
            }
        }
    }
}

fn q6_k(block: &[u8], span: &mut GroupedSpan) {
    let d = half(block, Q6_D_OFFSET);
    for (factor, scale) in span
        .factors
        .iter_mut()
        .zip(&block[Q6_SCALES_OFFSET..Q6_D_OFFSET])
    {
        *factor = d * f32::from(scale.cast_signed());
    }
    for half_index in 0..2 {
        let low = &block[half_index * Q6_HALF_LOW..][..Q6_HALF_LOW];
        let high = &block[Q6_HIGH_OFFSET + half_index * K_LANES..][..K_LANES];
        let codes = &mut span.codes[half_index * Q6_HALF..][..Q6_HALF];
        for lane in 0..K_LANES {
            let nibbles = [
                low[lane] & NIBBLE_MASK,
                low[lane + K_LANES] & NIBBLE_MASK,
                low[lane] >> NIBBLE_BITS,
                low[lane + K_LANES] >> NIBBLE_BITS,
            ];
            for (index, (nibble, shift)) in nibbles.into_iter().zip(Q6_HIGH_SHIFTS).enumerate() {
                let top = (high[lane] >> shift) & Q6_HIGH_MASK;
                codes[lane + index * K_LANES] = nibble | (top << NIBBLE_BITS);
            }
        }
    }
}

/// Decode the 6-bit scale and minimum pair at `index` from a k-quant scale table.
const fn scale_min(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < K_HEADER {
        (
            scales[index] & K_SCALE_MASK,
            scales[index + K_HEADER] & K_SCALE_MASK,
        )
    } else {
        (
            (scales[index + K_HEADER] & NIBBLE_MASK)
                | ((scales[index - K_HEADER] >> K_SCALE_HIGH_SHIFT) << NIBBLE_BITS),
            (scales[index + K_HEADER] >> NIBBLE_BITS)
                | ((scales[index] >> K_SCALE_HIGH_SHIFT) << NIBBLE_BITS),
        )
    }
}
