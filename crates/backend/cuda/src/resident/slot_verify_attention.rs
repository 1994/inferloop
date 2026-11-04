//! One KV read serves the three verification candidates of each pool slot.
use super::{
    ActivationArena,
    attention_decode::{
        VERIFY_LANES,
        pooled::{Append, Attend},
    },
    batch::{BatchBuilder, States},
    buffers::{buffer, kv_cache},
    capture::{Capture, CaptureMode, error},
    fp8_cache::Fp8Caches,
    slot_recurrent::row_buffer,
};
use crate::constants::{ATTENTION_SPLIT_THRESHOLD, KV_BLOCK_TOKENS};
use cutile::prelude::*;
use infer_cuda_kernels::contract::{DecodePlan, Recording, VerifyDecodeBuffers};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_ir::{TensorNode, TensorOp};
use std::result::Result as StdResult;

impl BatchBuilder<'_> {
    /// Append all candidates before attending; per-lane metadata still enforces causality.
    pub(super) fn record_slot_verify_attention(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        (states, caches): (&mut [States], &mut [Fp8Caches]),
        (verify, metadata, metadata_owner): (usize, &[Allocation<i32>], &Allocation<i32>),
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
        let attention = &*self.weights.kernels.attention;
        let recording = Recording::scope(scope);
        let output = arena.get(node.outputs[0]).map_err(error)?;
        let queries = arena.get(node.inputs[0]).map_err(error)?;
        if prefix > 0 {
            let (keys, values) = &caches[0][&state];
            self.attention.record_pooled_attention(
                attention,
                &recording,
                &Attend {
                    node: node.id,
                    output,
                    queries,
                    keys,
                    values,
                },
            )?;
        }
        let lanes = metadata.len();
        let outputs = buffer(output)?;
        let queries = buffer(queries)?;
        for (index, cache) in caches.iter().enumerate().skip(prefix) {
            let base = index * VERIFY_LANES;
            let (keys, values) = &cache[&state];
            let row = |lane: usize| row_buffer(queries, lanes, base + lane);
            let out = |lane: usize| row_buffer(outputs, lanes, base + lane);
            self.attention.record_verify(
                attention,
                &recording,
                DecodePlan {
                    heads: query_heads,
                    kv_heads,
                    dim: head_dim,
                    window: i32::try_from(window.unwrap_or(0)).map_err(error)?,
                    block: KV_BLOCK_TOKENS,
                },
                &VerifyDecodeBuffers {
                    outputs: [out(0)?, out(1)?, out(2)?],
                    queries: [row(0)?, row(1)?, row(2)?],
                    metadata: [
                        buffer(&metadata[base])?,
                        buffer(&metadata[base + 1])?,
                        buffer(&metadata[base + 2])?,
                    ],
                    cache: kv_cache(
                        (keys, values),
                        kv_heads,
                        head_dim,
                        self.weights.kv_scales[&state],
                    )?,
                    table: buffer(&self.tables[index])?,
                },
            )?;
        }
        Ok(true)
    }

    fn append_pooled(
        &self,
        scope: &Scope,
        node: &TensorNode,
        arena: &ActivationArena,
        caches: &mut [Fp8Caches],
        metadata_owner: &Allocation<i32>,
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
            let (keys, values) = caches[0].get(&state).ok_or_else(|| error("pooled cache"))?;
            self.attention.record_pooled_append(
                &*self.weights.kernels.attention,
                &Recording::scope(scope),
                &Append {
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
        metadata: &[Allocation<i32>],
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
                metadata: buffer(metadata)?,
                table: buffer(&self.tables[slot])?,
                external: buffer(&self.lane_external[lane])?,
                capacity: self.capacity,
                fusion: &mut self.lane_fusion[lane],
                mode: CaptureMode::Row(lane),
            }
            .record_attention_append(node)?;
        }
        Ok(())
    }
}
