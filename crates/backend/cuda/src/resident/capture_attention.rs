use super::{
    arena::SlotMemory,
    buffers::buffer,
    capture::{Capture, CaptureMode, error},
};
use crate::constants::ATTENTION_SPLIT_THRESHOLD;
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::WIDE_PREFILL_LANES;
use cutile::prelude::*;
use infer_cuda_kernels::contract::{
    AppendBuffers, Buffer, DecodeBuffers, DecodePlan, KvBuffers, KvCache, PrefillAppendPlan,
    PrefillDecodePlan,
};
use infer_ir::{TensorNode, TensorOp};

impl Capture<'_> {
    /// Append this lane's key and value row to the shared FP8 cache without attending; a
    /// verification step then attends all its lanes in one grouped launch.
    pub fn record_attention_append(&self, node: &TensorNode) -> Result<(), DeviceError> {
        let id = *node
            .states
            .first()
            .ok_or_else(|| error("missing state binding"))?;
        let scales = self.weights.kv_scales[&id];
        let state = self
            .fp8_states
            .get(&id)
            .ok_or_else(|| error("missing FP8 attention state"))?;
        self.record_append(node, state.cache(scales))
    }

    /// Record one attention node over `cache`, whose groups and dimension are the node's.
    pub fn record_attention(
        &mut self,
        node: &TensorNode,
        output: &SlotMemory,
        cache: KvCache<'_>,
    ) -> Result<(), DeviceError> {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) || !head_dim.is_power_of_two() {
            return Err(error("resident attention dimensions"));
        }
        if (cache.groups, cache.dim) != (kv_heads, head_dim) || cache.rows().is_none() {
            return Err(error("attention cache layout"));
        }
        if self.mode == CaptureMode::Prefill {
            return self.record_prefill_attention(node, output, cache);
        }
        self.record_append(node, cache)?;
        let plan = DecodePlan {
            heads: query_heads,
            kv_heads,
            dim: head_dim,
            window: i32::try_from(window.unwrap_or(0)).map_err(error)?,
            block: KV_BLOCK_TOKENS,
        };
        let buffers = DecodeBuffers {
            output: buffer(output)?,
            query: self.input_buffer(node.inputs[0])?,
            cache,
            metadata: self.metadata,
            table: self.table,
        };
        let recording = self.scope;
        if self.attention.supports(query_heads, head_dim)
            && window.is_none_or(|w| w >= ATTENTION_SPLIT_THRESHOLD)
        {
            self.attention
                .record(&*self.weights.kernels.attention, recording, plan, buffers)?;
        } else {
            self.weights
                .kernels
                .attention
                .decode(recording, plan, buffers)
                .map_err(error)?;
        }
        Ok(())
    }

    /// Append one lane's key and value row.
    fn record_append(&self, node: &TensorNode, cache: KvCache<'_>) -> Result<(), DeviceError> {
        self.weights
            .kernels
            .attention
            .append(
                self.scope,
                AppendBuffers {
                    cache,
                    keys: self.input_buffer(node.inputs[1])?,
                    values: self.input_buffer(node.inputs[2])?,
                    metadata: self.metadata,
                    table: self.table,
                },
            )
            .map_err(error)
    }

    fn record_prefill_attention(
        &mut self,
        node: &TensorNode,
        output: &SlotMemory,
        cache: KvCache<'_>,
    ) -> Result<(), DeviceError> {
        let lanes = self.arena.lanes();
        let rows = cache
            .rows()
            .ok_or_else(|| error("attention cache layout"))?;
        let fp8 = matches!(cache.buffers, KvBuffers::Fp8 { .. });
        // A CTA owns one physical page, including its unmodified tail. Narrow
        // graphs retain the whole-head loop rather than launching idle pages.
        let append_rows =
            if fp8 && lanes >= WIDE_PREFILL_LANES && rows.is_multiple_of(KV_BLOCK_TOKENS) {
                KV_BLOCK_TOKENS
            } else {
                rows
            };
        self.weights
            .kernels
            .attention
            .append_prefill(
                self.scope,
                PrefillAppendPlan {
                    lanes,
                    rows: append_rows,
                    block: KV_BLOCK_TOKENS,
                },
                AppendBuffers {
                    cache,
                    keys: self.input_buffer(node.inputs[1])?,
                    values: self.input_buffer(node.inputs[2])?,
                    metadata: self.metadata,
                    table: self.table,
                },
            )
            .map_err(error)?;
        self.record_prefill_decode(node, output, cache)
    }

    fn record_prefill_decode(
        &mut self,
        node: &TensorNode,
        output: &SlotMemory,
        cache: KvCache<'_>,
    ) -> Result<(), DeviceError> {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        let lanes = self.arena.lanes();
        // The choice was measured for a slot's layout; a pool's operands may not fit it. That is
        // decided here, in the open, and a selected kernel then records or fails the capture.
        let geometry = (query_heads, kv_heads, head_dim, lanes, self.capacity);
        let window = i32::try_from(window.unwrap_or(0)).map_err(error)?;
        let buffers = DecodeBuffers {
            output: buffer(output)?,
            query: self.input_buffer(node.inputs[0])?,
            cache,
            metadata: self.metadata,
            table: self.table,
        };
        let recording = self.scope;
        if let Some((mma, variant)) = self.attention.prefill_mma(node.outputs[0], lanes) {
            if mma.accepts(&buffers, geometry, window) {
                mma.record(recording, buffers, geometry, window, variant)?;
                return Ok(());
            }
            tracing::info!(target: "infer::capture", lanes, "selected MMA does not fit this pool");
        }
        if let Some(oxide) = self.attention.prefill_oxide(node.outputs[0], lanes) {
            if oxide.accepts(&buffers, geometry) {
                oxide.record(recording, buffers, geometry, window)?;
                return Ok(());
            }
            tracing::info!(target: "infer::capture", lanes, "selected SIMT does not fit this pool");
        }
        self.record_prefill_legacy(node, buffer(output)?, cache)
    }
    fn record_prefill_legacy(
        &self,
        node: &TensorNode,
        output: Buffer<'_, f32>,
        cache: KvCache<'_>,
    ) -> Result<(), DeviceError> {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        let lanes = self.arena.lanes();
        let plan = PrefillDecodePlan {
            decode: DecodePlan {
                heads: query_heads,
                kv_heads,
                dim: head_dim,
                window: i32::try_from(window.unwrap_or(0)).map_err(error)?,
                block: KV_BLOCK_TOKENS,
            },
            lanes,
        };
        let buffers = DecodeBuffers {
            output,
            query: self.input_buffer(node.inputs[0])?,
            cache,
            metadata: self.metadata,
            table: self.table,
        };
        let fp8 = matches!(cache.buffers, KvBuffers::Fp8 { .. });
        let recording = self.scope;
        let attention = &self.weights.kernels.attention;
        // Query/head axes remain separate so each CTA addresses its own KV head.
        // Selection was measured and independently checked before graph capture.
        self.attention
            .prefill_query_tile(node.outputs[0], lanes, fp8)
            .map_or_else(
                || attention.decode_prefill(recording, plan, buffers),
                |tile| attention.decode_prefill_tiled(recording, plan, tile, buffers),
            )
            .map_err(error)?;
        Ok(())
    }
}
