//! One KV read serves the three verification candidates of each pool slot.
use super::{
    ActivationArena,
    arena::row_tensor,
    attention_decode::{
        VERIFY_LANES,
        pooled::{Append, Attend},
    },
    batch::{BatchBuilder, States},
    capture::{Capture, CaptureMode, error},
    fp8_cache::Fp8Caches,
};
use crate::constants::ATTENTION_SPLIT_THRESHOLD;
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};
use std::result::Result as StdResult;
use std::sync::Arc;

impl BatchBuilder<'_> {
    /// Append all candidates before attending; per-lane metadata still enforces causality.
    pub(super) fn record_slot_verify_attention(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        (states, caches): (&mut [States], &mut [Fp8Caches]),
        (verify, metadata, metadata_owner): (usize, &[Tensor<i32>], &Tensor<i32>),
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
        if verify != VERIFY_LANES
            || window.is_some_and(|w| w < ATTENTION_SPLIT_THRESHOLD)
            || !self.attention.supports_verify(query_heads, head_dim)
            || self.weights.constants.contains_key(&node.inputs[0])
            || !self.weights.kv_scales.contains_key(&state)
            || caches.iter().any(|cache| !cache.contains_key(&state))
        {
            return Ok(false);
        }
        let prefix = self.append_pooled(scope, node, arena, caches, metadata_owner, state)?;
        self.append_slot_candidates(
            scope,
            node,
            arena,
            (states, caches),
            metadata,
            prefix * VERIFY_LANES,
        )?;
        let slot = arena.slot(node.outputs[0]).map_err(error)?;
        let output = arena.buffers[slot]
            .take()
            .ok_or_else(|| error("slot attention output"))?;
        let restore = output
            .shape()
            .iter()
            .map(|&dim| usize::try_from(dim).map_err(error))
            .collect::<StdResult<Vec<_>, _>>()?;
        let mut output = output.reshape(&[metadata.len() * query_heads, head_dim])?;
        if prefix > 0 {
            let (keys, values) = &caches[0][&state];
            self.attention.record_pooled_attention(
                scope,
                Attend {
                    node: node.id,
                    output: &mut output,
                    queries: arena.get(node.inputs[0]).map_err(error)?,
                    keys,
                    values,
                },
            )?;
        }
        let output = output.reshape(&restore)?;
        if prefix == caches.len() {
            arena.buffers[slot] = Some(output);
            return Ok(true);
        }
        let (mut rows, output) =
            attention_output_rows(output, metadata.len(), query_heads, head_dim)?;
        for (index, [first, second, third]) in rows
            .as_chunks_mut::<VERIFY_LANES>()
            .0
            .iter_mut()
            .enumerate()
            .skip(prefix)
        {
            let base = index * VERIFY_LANES;
            let queries = [
                arena.row(node.inputs[0], base).map_err(error)?,
                arena.row(node.inputs[0], base + 1).map_err(error)?,
                arena.row(node.inputs[0], base + 2).map_err(error)?,
            ];
            let (keys, values) = &caches[index][&state];
            self.attention.record_verify(
                scope,
                [first, second, third],
                [
                    &queries[0].view(&[query_heads, head_dim])?,
                    &queries[1].view(&[query_heads, head_dim])?,
                    &queries[2].view(&[query_heads, head_dim])?,
                ],
                [&metadata[base], &metadata[base + 1], &metadata[base + 2]],
                (keys, values, &self.tables[index]),
                (query_heads, kv_heads, head_dim),
                i32::try_from(window.unwrap_or(0)).map_err(error)?,
                self.weights.kv_scales[&state],
            )?;
        }
        arena.buffers[slot] = Some(output);
        Ok(true)
    }

    fn append_pooled(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &ActivationArena,
        caches: &mut [Fp8Caches],
        metadata_owner: &Tensor<i32>,
        state: infer_core::TensorId,
    ) -> StdResult<usize, DeviceError> {
        let shared = caches.first().is_some_and(|first| {
            let (keys, values) = &first[&state];
            caches.iter().all(|cache| {
                let (k, v) = &cache[&state];
                k.shape() == keys.shape()
                    && v.shape() == values.shape()
                    && k.device_pointer().cu_deviceptr() == keys.device_pointer().cu_deviceptr()
                    && v.device_pointer().cu_deviceptr() == values.device_pointer().cu_deviceptr()
            })
        });
        let prefix = if shared {
            self.attention.pooled_prefix(node.id, caches.len())
        } else {
            0
        };
        if prefix > 0 {
            let TensorOp::Attention {
                kv_heads, head_dim, ..
            } = node.op
            else {
                return Err(error("pooled append operation"));
            };
            // The original append capture leaves every slot's cache head-major. Homes and
            // prompt workers discover their copy geometry from these tensor shapes.
            for cache in caches.iter_mut() {
                let (keys, values) = cache.remove(&state).ok_or_else(|| error("pooled cache"))?;
                let rows = keys.size() / (kv_heads * head_dim);
                cache.insert(
                    state,
                    (
                        keys.reshape(&[kv_heads, rows, head_dim])?,
                        values.reshape(&[kv_heads, rows, head_dim])?,
                    ),
                );
            }
            let (keys, values) = caches[0]
                .get_mut(&state)
                .ok_or_else(|| error("pooled cache"))?;
            self.attention.record_pooled_append(
                scope,
                Append {
                    node: node.id,
                    keys,
                    values,
                    inputs: [
                        arena.get(node.inputs[1]).map_err(error)?,
                        arena.get(node.inputs[2]).map_err(error)?,
                    ],
                    metadata: metadata_owner,
                    tables: self.tables,
                },
            )?;
        }
        Ok(prefix)
    }

    fn append_slot_candidates(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        (states, caches): (&mut [States], &mut [Fp8Caches]),
        metadata: &[Tensor<i32>],
        first_lane: usize,
    ) -> StdResult<(), DeviceError> {
        for (lane, metadata) in metadata.iter().enumerate().skip(first_lane) {
            let slot = lane / VERIFY_LANES;
            Capture {
                scope,
                arena,
                weights: self.weights,
                nvfp4: self.nvfp4,
                attention: self.attention,
                states: &mut states[slot],
                fp8_states: &mut caches[slot],
                metadata,
                table: &self.tables[slot],
                external: &self.lane_external[lane],
                capacity: self.capacity,
                fusion: &mut self.lane_fusion[lane],
                mode: CaptureMode::Row(lane),
            }
            .record_attention_append(node)?;
        }
        Ok(())
    }
}

/// Preserve the arena owner's exclusive storage while capture binds borrowed row outputs.
/// Owned shared aliases would prevent a later node from reusing this arena buffer mutably.
pub(super) fn attention_output_rows(
    mut output: Tensor<f32>,
    lanes: usize,
    heads: usize,
    dim: usize,
) -> StdResult<(Vec<Tensor<f32>>, Tensor<f32>), DeviceError> {
    let mut rows = Vec::with_capacity(lanes);
    for lane in 0..lanes {
        let (row, owner) = row_tensor(output, lanes, lane).map_err(error)?;
        output = Arc::try_unwrap(owner)
            .map_err(|_| error("slot attention arena owner still aliased"))?;
        rows.push(row.reshape(&[heads, dim])?);
    }
    Ok((rows, output))
}
