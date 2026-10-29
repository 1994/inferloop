//! Borrowed JPEG frame-header inspection for the encoded-input experiment.
//!
//! This finds candidate dimensions without copying the compressed input or parsing metadata.
//! It is not a JPEG decoder or a validity check for the remainder of the bitstream. The chosen
//! decoder must validate the image and agree on dimensions before allocating its output.
use super::{CpuContext, Error, JPEG_SIGNATURE, MEDIA_CHUNK_CANCELLATION_INTERVAL, Result};

/// Bytes in the start-of-image marker.
const SOI_BYTES: usize = 2;
/// Marker prefix and standalone markers relevant before the first frame.
const MARKER_PREFIX: u8 = 0xff;
const MARKER_TEM: u8 = 0x01;
const MARKER_STUFFED: u8 = 0x00;
const MARKER_SOI: u8 = 0xd8;
const MARKER_EOI: u8 = 0xd9;
const MARKER_SOS: u8 = 0xda;
const MARKER_RESTART_FIRST: u8 = 0xd0;
const MARKER_RESTART_LAST: u8 = 0xd7;
/// All JPEG start-of-frame marker codes; actual codec support is decided by the decoder.
const SOF_MARKERS: [u8; 13] = [
    0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
];
/// A segment length includes its own two bytes; a frame has six fixed payload bytes and
/// three additional bytes per component.
const SEGMENT_LENGTH_BYTES: usize = 2;
const FRAME_FIXED_BYTES: usize = 6;
const FRAME_COMPONENT_BYTES: usize = 3;
/// Check cancellation between chunks while skipping a long run of marker-fill bytes.
const FILL_SCAN_CHUNK_BYTES: usize = 4096;

/// Return untrusted frame dimensions without allocating or decoding image data.
pub(super) fn frame_dimensions(bytes: &[u8], context: Option<&CpuContext>) -> Result<(u32, u32)> {
    if !bytes.starts_with(&JPEG_SIGNATURE) {
        return Err(Error::invalid("invalid JPEG signature"));
    }
    let mut offset = SOI_BYTES;
    let mut segments = 0usize;
    loop {
        if segments.is_multiple_of(MEDIA_CHUNK_CANCELLATION_INTERVAL)
            && let Some(context) = context
        {
            context.check()?;
        }
        let (marker, next) = marker(bytes, offset, context)?;
        if marker == MARKER_TEM {
            offset = next;
            segments = segments
                .checked_add(1)
                .ok_or_else(|| Error::invalid("too many JPEG segments"))?;
            continue;
        }
        if matches!(
            marker,
            MARKER_STUFFED | MARKER_SOI | MARKER_EOI | MARKER_SOS
        ) || (MARKER_RESTART_FIRST..=MARKER_RESTART_LAST).contains(&marker)
        {
            return Err(Error::invalid("JPEG has no valid frame header"));
        }
        let (payload, end) = segment(bytes, next)?;
        if SOF_MARKERS.contains(&marker) {
            return frame(payload);
        }
        offset = end;
        segments = segments
            .checked_add(1)
            .ok_or_else(|| Error::invalid("too many JPEG segments"))?;
    }
}

fn marker(bytes: &[u8], offset: usize, context: Option<&CpuContext>) -> Result<(u8, usize)> {
    if bytes.get(offset) != Some(&MARKER_PREFIX) {
        return Err(Error::invalid("invalid JPEG marker"));
    }
    let start = offset
        .checked_add(1)
        .ok_or_else(|| Error::invalid("invalid JPEG marker"))?;
    let remaining = bytes
        .get(start..)
        .ok_or_else(|| Error::invalid("truncated JPEG marker"))?;
    let mut scanned = 0usize;
    for chunk in remaining.chunks(FILL_SCAN_CHUNK_BYTES) {
        if scanned != 0
            && let Some(context) = context
        {
            context.check()?;
        }
        if let Some(index) = chunk.iter().position(|byte| *byte != MARKER_PREFIX) {
            let next = start
                .checked_add(scanned)
                .and_then(|value| value.checked_add(index))
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| Error::invalid("invalid JPEG marker"))?;
            return Ok((chunk[index], next));
        }
        scanned = scanned
            .checked_add(chunk.len())
            .ok_or_else(|| Error::invalid("invalid JPEG marker"))?;
    }
    Err(Error::invalid("truncated JPEG marker"))
}

fn segment(bytes: &[u8], offset: usize) -> Result<(&[u8], usize)> {
    let length_end = offset
        .checked_add(SEGMENT_LENGTH_BYTES)
        .ok_or_else(|| Error::invalid("invalid JPEG segment"))?;
    let length: [u8; SEGMENT_LENGTH_BYTES] = bytes
        .get(offset..length_end)
        .ok_or_else(|| Error::invalid("truncated JPEG segment"))?
        .try_into()
        .map_err(|_| Error::invalid("invalid JPEG segment"))?;
    let length = usize::from(u16::from_be_bytes(length));
    if length < SEGMENT_LENGTH_BYTES {
        return Err(Error::invalid("invalid JPEG segment length"));
    }
    let end = offset
        .checked_add(length)
        .ok_or_else(|| Error::invalid("invalid JPEG segment length"))?;
    let payload = bytes
        .get(length_end..end)
        .ok_or_else(|| Error::invalid("truncated JPEG segment"))?;
    Ok((payload, end))
}

fn frame(payload: &[u8]) -> Result<(u32, u32)> {
    let [_precision, h1, h2, w1, w2, components]: [u8; FRAME_FIXED_BYTES] = payload
        .get(..FRAME_FIXED_BYTES)
        .ok_or_else(|| Error::invalid("truncated JPEG frame header"))?
        .try_into()
        .map_err(|_| Error::invalid("invalid JPEG frame header"))?;
    let expected = usize::from(components)
        .checked_mul(FRAME_COMPONENT_BYTES)
        .and_then(|bytes| bytes.checked_add(FRAME_FIXED_BYTES))
        .ok_or_else(|| Error::invalid("invalid JPEG frame components"))?;
    if components == 0 || payload.len() != expected {
        return Err(Error::invalid("invalid JPEG frame components"));
    }
    let height = u32::from(u16::from_be_bytes([h1, h2]));
    let width = u32::from(u16::from_be_bytes([w1, w2]));
    if width == 0 || height == 0 {
        return Err(Error::invalid("invalid JPEG frame dimensions"));
    }
    Ok((width, height))
}

#[cfg(test)]
#[path = "../../../tests/unit/encoded_jpeg_header.rs"]
mod tests;
