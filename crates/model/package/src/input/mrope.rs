//! Multimodal rotary geometry, independent of the execution backend.
use crate::{PreparedPrompt, VisualPlacement};
use infer_core::{Error, Result};
use infer_ir::PositionSpec;
use std::collections::BTreeMap;
use std::ops::Range;

/// Number of temporal and spatial rotary axes.
pub const ROTARY_AXES: usize = 3;

/// Axis (time, height, width) used by each rotary frequency.
/// # Errors
/// Rejects sections that do not cover the rotary half-head.
pub fn axes(spec: &PositionSpec, half: usize) -> Result<Vec<i32>> {
    if spec.multimodal_sections.is_empty() {
        return Ok(vec![0; half]);
    }
    let [time, height, width] = spec.multimodal_sections.as_slice() else {
        return Err(Error::invalid("MRoPE requires three sections"));
    };
    if time
        .checked_add(*height)
        .and_then(|n| n.checked_add(*width))
        != Some(half)
    {
        return Err(Error::invalid(
            "MRoPE sections must cover the rotary half-head",
        ));
    }
    let mut result = vec![0; half];
    if spec.interleaved {
        for (axis, count) in [(1usize, *height), (2usize, *width)] {
            for index in 0..count {
                let lane = index
                    .checked_mul(ROTARY_AXES)
                    .and_then(|n| n.checked_add(axis))
                    .filter(|n| *n < half)
                    .ok_or_else(|| Error::invalid("interleaved MRoPE section overflow"))?;
                result[lane] = if axis == 1 { 1 } else { 2 };
            }
        }
    } else {
        result[*time..time + height].fill(1);
        result[time + height..].fill(2);
    }
    Ok(result)
}

/// Three-axis positions for merged image patches, including intervening text.
/// # Errors
/// Rejects malformed, overlapping or out-of-range media spans and grids.
pub fn image_positions(prompt: &PreparedPrompt, merge: usize) -> Result<Vec<[usize; ROTARY_AXES]>> {
    if prompt.spans.len() != prompt.images.len() {
        return Err(Error::invalid("MRoPE media geometry"));
    }
    let grids = prompt
        .spans
        .iter()
        .zip(&prompt.images)
        .map(|(span, image)| (span, image.grid));
    positions_from_grids(prompt.tokens.len(), grids, merge)
}

pub(super) fn positions_from_grids<'a>(
    tokens: usize,
    grids: impl Iterator<Item = (&'a Range<usize>, (usize, usize, usize))>,
    merge: usize,
) -> Result<Vec<[usize; ROTARY_AXES]>> {
    if merge == 0 {
        return Err(Error::invalid("MRoPE media geometry"));
    }
    let mut result = Vec::with_capacity(tokens);
    let mut cursor = 0usize;
    for (span, (t, h, w)) in grids {
        if t == 0
            || h == 0
            || w == 0
            || !h.is_multiple_of(merge)
            || !w.is_multiple_of(merge)
            || span.start < result.len()
            || span.end > tokens
            || span.end <= span.start
        {
            return Err(Error::invalid("MRoPE grid/span mismatch"));
        }
        let (h, w) = (h / merge, w / merge);
        if t.checked_mul(h).and_then(|n| n.checked_mul(w)) != Some(span.len()) {
            return Err(Error::invalid("MRoPE span must contain every merged patch"));
        }
        while result.len() < span.start {
            result.push([cursor; ROTARY_AXES]);
            cursor = cursor
                .checked_add(1)
                .ok_or_else(|| Error::invalid("MRoPE position overflow"))?;
        }
        let limit = cursor
            .checked_add(t.max(h).max(w))
            .ok_or_else(|| Error::invalid("MRoPE position overflow"))?;
        for time in 0..t {
            for row in 0..h {
                for col in 0..w {
                    result.push([cursor + time, cursor + row, cursor + col]);
                }
            }
        }
        cursor = limit;
    }
    while result.len() < tokens {
        result.push([cursor; ROTARY_AXES]);
        cursor = cursor
            .checked_add(1)
            .ok_or_else(|| Error::invalid("MRoPE position overflow"))?;
    }
    Ok(result)
}

/// Reconstruct text positions from placed media and return the next decode coordinate.
/// # Errors
/// Rejects duplicate/out-of-range placements and overflowing coordinates.
pub fn placed_positions(
    tokens: usize,
    media: &[VisualPlacement],
) -> Result<(Vec<[usize; ROTARY_AXES]>, usize)> {
    let mut placed = BTreeMap::new();
    for item in media {
        if item.position >= tokens || placed.insert(item.position, item.rotary_position).is_some() {
            return Err(Error::invalid(
                "MRoPE placement outside prompt or duplicated",
            ));
        }
    }
    let mut next = 0usize;
    let mut result = Vec::with_capacity(tokens);
    for index in 0..tokens {
        let coordinates = placed
            .get(&index)
            .copied()
            .flatten()
            .unwrap_or([next; ROTARY_AXES]);
        let end = coordinates
            .iter()
            .max()
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| Error::invalid("MRoPE position overflow"))?;
        next = next.max(end);
        result.push(coordinates);
    }
    Ok((result, next))
}

#[cfg(test)]
#[path = "../../tests/unit/input_mrope.rs"]
mod tests;
