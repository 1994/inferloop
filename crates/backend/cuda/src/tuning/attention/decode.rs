//! Decode launch geometry derived from hardware occupancy and the attention window.

use crate::constants::ATTENTION_SPLIT_MAX_PARTS;
use crate::constants::ATTENTION_SPLIT_THRESHOLD;
/// Padded query rows required by the grouped MMA kernel.
pub const GROUPED_ROWS: usize = 16;
/// Compiler occupancy hint of the grouped decode kernels. Without it the compiler spends the whole
/// register file on one CTA per SM, so the 1.5-wave split grid runs in two waves.
pub const GROUPED_OCCUPANCY: i32 = 2;
/// Upper bound shared with admission's scratch reservation.
pub const MAX_GROUPED_PARTS: usize = 64;
const WAVE_NUMERATOR: usize = 3;
const WAVE_DENOMINATOR: usize = 2;

/// Match the existing F32 reduction tree using hardware/window facts, never KV storage capacity.
pub fn partitions(heads: usize, window: Option<usize>, multiprocessors: usize) -> usize {
    if window.is_some_and(|tokens| tokens < ATTENTION_SPLIT_THRESHOLD) {
        return 1;
    }
    let limit = multiprocessors
        .div_ceil(heads)
        .clamp(1, ATTENTION_SPLIT_MAX_PARTS);
    1 << limit.ilog2()
}

/// Target about 1.5 waves of KV-head CTAs, keeping the scratch bound independent of hardware.
/// On 170 SMs and four KV heads this retains the measured 64-way split.
pub fn grouped_parts(heads: usize, kv_heads: usize, multiprocessors: usize) -> Option<usize> {
    grouped_parts_for(heads, kv_heads, multiprocessors, 1)
}

/// [`grouped_parts`] for a launch covering `slots` sequences: their KV-head CTAs already share
/// the wave target, so each slot splits its context fewer times. Over-splitting short contexts
/// writes and merges more partial rows than the KV it reads.
pub fn grouped_parts_for(
    heads: usize,
    kv_heads: usize,
    multiprocessors: usize,
    slots: usize,
) -> Option<usize> {
    if slots == 0
        || heads == 0
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || heads / kv_heads > GROUPED_ROWS
    {
        return None;
    }
    let blocks = multiprocessors
        .saturating_mul(WAVE_NUMERATOR)
        .div_ceil(WAVE_DENOMINATOR);
    let limit = blocks
        .div_ceil(kv_heads.saturating_mul(slots))
        .clamp(1, MAX_GROUPED_PARTS);
    Some(limit.next_power_of_two())
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_attention_decode_workspace.rs"]
mod tests;
