//! One launch of decode attention across a pool's lanes instead of three per lane.
use super::{
    ActivationArena,
    batch::BatchBuilder,
    buffers::{buffer, kv_cache},
    capture::error,
};
use crate::constants::{ATTENTION_SPLIT_THRESHOLD, KV_BLOCK_TOKENS};
use crate::device::{CudaDevice, DeviceProfile, device_error};
use crate::tuning::attention::decode::{GROUPED_ROWS, grouped_parts_for, partitions};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    AppendBuffers, DecodeBuffers, DecodePlan, KvCache, LanesAppendPlan, LanesDecodePlan, Partials,
    Recording,
};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
#[cfg(test)]
use std::cell::Cell;
use std::collections::BTreeMap;
use std::result::Result as StdResult;

/// Elements of one split's `numerator`, `maxima` and `sums` rows beyond the head dimension.
const PARTIAL_ROW_SCALARS: usize = 2;

/// Scratch of one geometry's lane-batched splits.
pub struct Buffers {
    parts: usize,
    numerator: Allocation<f32>,
    maxima: Allocation<f32>,
    sums: Allocation<f32>,
}

/// Lane-batched scratch per `(heads, dim)`.
#[derive(Default)]
pub struct Workspace(BTreeMap<(usize, usize), Buffers>);

/// `(heads, kv_heads, dim, parts)` of each covered geometry.
fn geometries(
    graph: &DataflowGraph,
    lanes: usize,
    multiprocessors: usize,
) -> BTreeMap<(usize, usize), (usize, usize)> {
    let mut plans = BTreeMap::new();
    for node in &graph.nodes {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            continue;
        };
        if partitions(query_heads, window, multiprocessors) == 1 {
            continue;
        }
        if let Some(parts) = grouped_parts_for(query_heads, kv_heads, multiprocessors, lanes) {
            plans.insert((query_heads, head_dim), (kv_heads, parts));
        }
    }
    plans
}

/// Device bytes of [`Workspace::new`].
/// # Errors
/// Rejects an unrepresentable multiprocessor count.
pub fn required_bytes(graph: &DataflowGraph, lanes: usize, profile: &DeviceProfile) -> Result<u64> {
    let multiprocessors = usize::try_from(profile.multiprocessors).map_err(device_error)?;
    let bytes: usize = geometries(graph, lanes, multiprocessors)
        .into_iter()
        .map(|((_, dim), (kv_heads, parts))| {
            lanes * kv_heads * parts * GROUPED_ROWS * (dim + PARTIAL_ROW_SCALARS) * size_of::<f32>()
        })
        .sum();
    u64::try_from(bytes).map_err(device_error)
}

impl Workspace {
    /// # Errors
    /// Reports CUDA allocation failures.
    pub fn new(device: &CudaDevice, graph: &DataflowGraph, lanes: usize) -> Result<Self> {
        let multiprocessors =
            usize::try_from(device.profile()?.multiprocessors).map_err(device_error)?;
        let runtime = device.runtime();
        let zeros = |shape: &[usize]| runtime.zeros::<f32>(shape.iter().product());
        let mut buffers = BTreeMap::new();
        for ((heads, dim), (kv_heads, parts)) in geometries(graph, lanes, multiprocessors) {
            let blocks = lanes
                .checked_mul(kv_heads * parts)
                .ok_or_else(|| Error::invalid("lane attention overflow"))?;
            buffers.insert(
                (heads, dim),
                Buffers {
                    parts,
                    numerator: zeros(&[blocks, GROUPED_ROWS, dim])?,
                    maxima: zeros(&[blocks, GROUPED_ROWS])?,
                    sums: zeros(&[blocks, GROUPED_ROWS])?,
                },
            );
        }
        Ok(Self(buffers))
    }
}

#[cfg(test)]
std::thread_local! {
    static RECORDED: Cell<usize> = const { Cell::new(0) };
}
/// Lane-batched attention nodes this thread has recorded.
#[cfg(test)]
pub fn recorded() -> usize {
    RECORDED.get()
}

impl BatchBuilder<'_> {
    /// False leaves the node to one launch per lane.
    pub(super) fn record_lanes_attention(
        &self,
        scope: &Scope,
        node: &TensorNode,
        arena: &ActivationArena,
        caches: &[super::fp8_cache::Fp8Caches],
        metadata: &Allocation<i32>,
    ) -> StdResult<bool, DeviceError> {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return Ok(false);
        };
        let Some(&state) = node.states.first() else {
            return Ok(false);
        };
        if self.table_root.is_none()
            || window.is_some_and(|w| w < ATTENTION_SPLIT_THRESHOLD)
            || node
                .inputs
                .iter()
                .any(|id| self.weights.constants.contains_key(id))
            || !self
                .lanes_attention
                .as_ref()
                .is_some_and(|work| work.0.contains_key(&(query_heads, head_dim)))
        {
            return Ok(false);
        }
        if let Some(scales) = self.weights.kv_scales.get(&state).copied() {
            // Only paged slots share one cache; separate slot caches keep their own launches.
            let Some((first, rest)) = caches.split_first() else {
                return Ok(false);
            };
            let Some(shared) = first.get(&state) else {
                return Ok(false);
            };
            if rest.iter().any(|slot| {
                !slot
                    .get(&state)
                    .is_some_and(|cache| cache.same_memory(shared))
            }) {
                return Ok(false);
            }
            self.record_lanes(scope, node, arena, (shared.cache(scales), false), metadata)?;
            return Ok(true);
        }
        let owned = self
            .states
            .get(&state)
            .ok_or_else(|| error("pooled attention state"))?;
        let [keys, values] = owned.as_slice() else {
            return Err(error("pooled attention state layout"));
        };
        // A pooled root holds one row of every head per lane.
        let groups = arena.lanes() * kv_heads;
        let cache = kv_cache((keys, values), groups, head_dim, [1.0, 1.0])?;
        self.record_lanes(scope, node, arena, (cache, true), metadata)?;
        Ok(true)
    }

    /// Record the lane-batched append and decode over `cache`; `per_lane` when the cache holds
    /// one row per lane of a pooled root rather than pages every lane shares.
    fn record_lanes(
        &self,
        scope: &Scope,
        node: &TensorNode,
        arena: &ActivationArena,
        (cache, per_lane): (KvCache<'_>, bool),
        metadata: &Allocation<i32>,
    ) -> StdResult<(), DeviceError> {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        let lanes = arena.lanes();
        let table = self
            .table_root
            .ok_or_else(|| error("pooled attention tables"))?;
        if cache.rows().is_none() {
            return Err(error("pooled attention cache layout"));
        }
        let work = self
            .lanes_attention
            .as_deref()
            .and_then(|work| work.0.get(&(query_heads, head_dim)))
            .ok_or_else(|| error("pooled attention scratch"))?;
        let attention = &self.weights.kernels.attention;
        let recording = Recording::scope(scope);
        let table_rows = table.rows;
        attention
            .append_lanes(
                &recording,
                LanesAppendPlan {
                    kv_heads,
                    lanes,
                    per_lane,
                    table_rows,
                },
                AppendBuffers {
                    cache,
                    keys: arena.activation(node.inputs[1]).map_err(error)?,
                    values: arena.activation(node.inputs[2]).map_err(error)?,
                    metadata: buffer(metadata)?,
                    table: table.buffer,
                },
            )
            .map_err(error)?;
        attention
            .decode_lanes(
                &recording,
                LanesDecodePlan {
                    decode: DecodePlan {
                        heads: query_heads,
                        kv_heads,
                        dim: head_dim,
                        window: i32::try_from(window.unwrap_or(0)).map_err(error)?,
                        block: KV_BLOCK_TOKENS,
                    },
                    parts: work.parts,
                    lanes,
                    per_lane,
                    table_rows,
                },
                Partials {
                    numerator: buffer(&work.numerator)?,
                    maxima: buffer(&work.maxima)?,
                    sums: buffer(&work.sums)?,
                },
                DecodeBuffers {
                    output: arena.activation(node.outputs[0]).map_err(error)?,
                    query: arena.activation(node.inputs[0]).map_err(error)?,
                    cache,
                    metadata: buffer(metadata)?,
                    table: table.buffer,
                },
            )
            .map_err(error)?;
        #[cfg(test)]
        RECORDED.set(RECORDED.get() + 1);
        Ok(())
    }
}
