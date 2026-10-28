//! Device tensors retained by the draft engine, before any graph is captured.
use super::weights::Geometry;
use super::{KEY_TILE, PROMPT_ROWS, ROW_FIELDS, ROW_TILE, SLOT_FIELDS, TOP_BLOCK, split_plan};
use infer_core::{Error, Result};
use std::collections::BTreeSet;

/// K, V, normalized K and rotated K retained by ingestion and proposal.
const KV_BUFFERS: usize = 4;
/// Two residual streams, normalized input, convolution and projected output.
const HIDDEN_BUFFERS: usize = 5;
/// Query, normalized query, rotated query and attention output.
const QUERY_BUFFERS: usize = 4;
/// Two convolution sides, each with two tap coefficient vectors.
const COEFFICIENT_BUFFERS: usize = 4;
/// Gate, up and activated MLP outputs.
const MLP_BUFFERS: usize = 3;

/// Checked byte total of F32/I32 buffers, expressed as element-count factors.
fn buffers(shapes: impl IntoIterator<Item = Vec<usize>>) -> Result<u64> {
    shapes.into_iter().try_fold(0u64, |total, shape| {
        let bytes = shape
            .into_iter()
            .try_fold(size_of::<f32>() as u64, |n, dim| {
                n.checked_mul(dim as u64)
                    .ok_or_else(|| Error::invalid("DFlash2 buffer size overflow"))
            })?;
        total
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid("DFlash2 buffer total overflow"))
    })
}

/// Prompt/verified ingestion: metadata, normalized features and four KV projection buffers.
pub(super) fn ingest_bytes(g: &Geometry, rows: usize) -> Result<u64> {
    let kv = g
        .kv_heads
        .checked_mul(g.head_dim)
        .ok_or_else(|| Error::invalid("DFlash2 KV width overflow"))?;
    buffers([
        vec![2, rows],
        vec![rows, g.hidden],
        vec![KV_BUFFERS, rows, kv],
    ])
}

/// Rings plus prompt ingestion, charged once before request admission.
pub fn fixed_bytes(g: &Geometry, layers: usize, rings: usize) -> Result<u64> {
    let ring = g
        .window
        .checked_add(g.block)
        .and_then(|n| n.checked_add(KEY_TILE - 1))
        .map(|n| n / KEY_TILE * KEY_TILE)
        .ok_or_else(|| Error::invalid("DFlash2 ring length overflow"))?;
    // K and V are BF16: together they occupy one F32 word per element.
    let bytes = buffers([
        vec![layers, rings, ring, g.kv_heads, g.head_dim],
        vec![PROMPT_ROWS, g.hidden],
    ])?;
    bytes
        .checked_add(ingest_bytes(g, PROMPT_ROWS)?)
        .ok_or_else(|| Error::invalid("DFlash2 engine size overflow"))
}

/// Every proposal tensor, including narrow split-K partials. F32 and I32 are both four bytes.
pub(super) fn propose_bytes(g: &Geometry, slots: usize, multiprocessors: usize) -> Result<u64> {
    let rows = slots
        .checked_mul(g.block)
        .ok_or_else(|| Error::invalid("DFlash2 proposal rows overflow"))?;
    let query = g
        .heads
        .checked_mul(g.head_dim)
        .ok_or_else(|| Error::invalid("DFlash2 query width overflow"))?;
    let kv = g
        .kv_heads
        .checked_mul(g.head_dim)
        .ok_or_else(|| Error::invalid("DFlash2 KV width overflow"))?;
    let candidates = g
        .vocabulary
        .div_ceil(TOP_BLOCK)
        .checked_mul(g.top_k)
        .ok_or_else(|| Error::invalid("DFlash2 candidate size overflow"))?;
    let mut bytes = buffers([
        vec![ROW_FIELDS, rows],
        vec![SLOT_FIELDS, slots],
        vec![HIDDEN_BUFFERS, rows, g.hidden],
        vec![QUERY_BUFFERS, rows, query],
        vec![KV_BUFFERS, rows, kv],
        vec![COEFFICIENT_BUFFERS, rows, g.groups],
        vec![MLP_BUFFERS, rows, g.intermediate],
        vec![rows, g.vocabulary],
        vec![2, rows, candidates],
        vec![2, rows, g.top_k],
        vec![rows, g.rank],
        vec![rows],
    ])?;
    let shapes: BTreeSet<_> = [
        (query, g.hidden),
        (kv, g.hidden),
        (g.hidden, query),
        (g.hidden, g.intermediate),
    ]
    .into_iter()
    .collect();
    for (n, k) in shapes {
        if let Some((splits, _)) = split_plan(rows, n, k, multiprocessors) {
            bytes = bytes
                .checked_add(buffers([vec![splits, ROW_TILE, n]])?)
                .ok_or_else(|| Error::invalid("DFlash2 partial size overflow"))?;
        }
    }
    Ok(bytes)
}

/// One proposal and verified ingestion pair at a captured slot width.
pub fn pooled_bytes(g: &Geometry, slots: usize, multiprocessors: usize) -> Result<u64> {
    let rows = slots
        .checked_mul(g.block)
        .ok_or_else(|| Error::invalid("DFlash2 ingestion rows overflow"))?;
    propose_bytes(g, slots, multiprocessors)?
        .checked_add(ingest_bytes(g, rows)?)
        .ok_or_else(|| Error::invalid("DFlash2 pool size overflow"))
}
