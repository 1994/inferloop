use super::{
    buffers::{buffer, kv_cache},
    capture::{Capture, CaptureMode, error},
};
use crate::constants::ATTENTION_SPLIT_THRESHOLD;
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::WIDE_PREFILL_LANES;
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_cuda_kernels::contract::{
    AppendBuffers, DecodeBuffers, DecodePlan, KvCache, KvElement, PrefillAppendPlan,
    PrefillDecodePlan, Recording,
};
use infer_cuda_kernels::tile::runtime::Element;
use infer_ir::{TensorNode, TensorOp};

type AttentionResult<E> = Result<(Tensor<f32>, Tensor<E>, Tensor<E>), DeviceError>;

impl Capture<'_> {
    /// Append this lane's key and value row to the shared FP8 cache without attending; a
    /// verification step then attends all its lanes in one grouped launch.
    pub fn record_attention_append(&mut self, node: &TensorNode) -> Result<(), DeviceError> {
        let TensorOp::Attention {
            kv_heads, head_dim, ..
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        let id = *node
            .states
            .first()
            .ok_or_else(|| error("missing state binding"))?;
        let scales = self.weights.kv_scales[&id];
        let (keys, values) = self
            .fp8_states
            .remove(&id)
            .ok_or_else(|| error("missing FP8 attention state"))?;
        // A paged slot's cache holds the pool's physical rows, not one sequence's capacity.
        let rows = keys.size() / (kv_heads * head_dim);
        let keys = keys.reshape(&[kv_heads, rows, head_dim])?;
        let values = values.reshape(&[kv_heads, rows, head_dim])?;
        let result = kv_cache((&keys, &values), kv_heads, head_dim, scales)
            .and_then(|cache| self.record_append(node, cache));
        self.fp8_states.insert(id, (keys, values));
        result
    }

    pub fn record_attention<E: DType + Element<Marker: KvElement>>(
        &mut self,
        node: &TensorNode,
        output: Tensor<f32>,
        keys: Tensor<E>,
        values: Tensor<E>,
        scales: [f32; 2],
    ) -> AttentionResult<E> {
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
        if self.mode == CaptureMode::Prefill {
            return self.record_prefill_attention(node, output, keys, values, scales);
        }
        // A paged slot's cache holds the pool's physical rows, not one sequence's capacity.
        let rows = keys.size() / (kv_heads * head_dim);
        let keys = keys.reshape(&[kv_heads, rows, head_dim])?;
        let values = values.reshape(&[kv_heads, rows, head_dim])?;
        let cache = kv_cache((&keys, &values), kv_heads, head_dim, scales)?;
        self.record_append(node, cache)?;
        let output = output.reshape(&[query_heads, head_dim])?;
        let plan = DecodePlan {
            heads: query_heads,
            kv_heads,
            dim: head_dim,
            window: i32::try_from(window.unwrap_or(0)).map_err(error)?,
            block: KV_BLOCK_TOKENS,
        };
        let buffers = DecodeBuffers {
            output: buffer(&output)?,
            query: self.input_buffer(node.inputs[0])?,
            cache,
            metadata: buffer(self.metadata)?,
            table: buffer(self.table)?,
        };
        let recording = Recording::scope(self.scope);
        if self.attention.supports(query_heads, head_dim)
            && window.is_none_or(|w| w >= ATTENTION_SPLIT_THRESHOLD)
        {
            self.attention
                .record(&*self.weights.kernels.attention, &recording, plan, buffers)?;
        } else {
            self.weights
                .kernels
                .attention
                .decode(&recording, plan, buffers)
                .map_err(error)?;
        }
        Ok((output, keys, values))
    }

    /// Append one lane's key and value row.
    fn record_append(&self, node: &TensorNode, cache: KvCache<'_>) -> Result<(), DeviceError> {
        self.weights
            .kernels
            .attention
            .append(
                &Recording::scope(self.scope),
                AppendBuffers {
                    cache,
                    keys: self.input_buffer(node.inputs[1])?,
                    values: self.input_buffer(node.inputs[2])?,
                    metadata: buffer(self.metadata)?,
                    table: buffer(self.table)?,
                },
            )
            .map_err(error)
    }

    fn record_prefill_attention<E: DType + Element<Marker: KvElement>>(
        &mut self,
        node: &TensorNode,
        output: Tensor<f32>,
        keys: Tensor<E>,
        values: Tensor<E>,
        scales: [f32; 2],
    ) -> AttentionResult<E> {
        let TensorOp::Attention {
            kv_heads, head_dim, ..
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        let lanes = self.arena.lanes();
        // A paged slot's cache holds the pool's physical rows, not one sequence's capacity.
        let rows = keys.size() / (kv_heads * head_dim);
        let keys = keys.reshape(&[kv_heads, rows, head_dim])?;
        let values = values.reshape(&[kv_heads, rows, head_dim])?;
        // A CTA owns one physical page, including its unmodified tail. Narrow
        // graphs retain the whole-head loop rather than launching idle pages.
        let append_rows = if E::DTYPE == f8e4m3fn::DTYPE
            && lanes >= WIDE_PREFILL_LANES
            && rows.is_multiple_of(KV_BLOCK_TOKENS)
        {
            KV_BLOCK_TOKENS
        } else {
            rows
        };
        self.weights
            .kernels
            .attention
            .append_prefill(
                &Recording::scope(self.scope),
                PrefillAppendPlan {
                    lanes,
                    rows: append_rows,
                    block: KV_BLOCK_TOKENS,
                },
                AppendBuffers {
                    cache: kv_cache((&keys, &values), kv_heads, head_dim, scales)?,
                    keys: self.input_buffer(node.inputs[1])?,
                    values: self.input_buffer(node.inputs[2])?,
                    metadata: buffer(self.metadata)?,
                    table: buffer(self.table)?,
                },
            )
            .map_err(error)?;
        self.record_prefill_decode(node, output, keys, values, scales)
    }

    fn record_prefill_decode<E: DType + Element<Marker: KvElement>>(
        &mut self,
        node: &TensorNode,
        output: Tensor<f32>,
        keys: Tensor<E>,
        values: Tensor<E>,
        scales: [f32; 2],
    ) -> AttentionResult<E> {
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
        let output = output.reshape(&[lanes * query_heads, head_dim])?;
        // The choice was measured for a slot's layout; a pool's operands may not fit it. That is
        // decided here, in the open, and a selected kernel then records or fails the capture.
        let geometry = (query_heads, kv_heads, head_dim, lanes, self.capacity);
        let window = i32::try_from(window.unwrap_or(0)).map_err(error)?;
        let buffers = DecodeBuffers {
            output: buffer(&output)?,
            query: self.input_buffer(node.inputs[0])?,
            cache: kv_cache((&keys, &values), kv_heads, head_dim, scales)?,
            metadata: buffer(self.metadata)?,
            table: buffer(self.table)?,
        };
        let recording = Recording::scope(self.scope);
        if let Some((mma, variant)) = self.attention.prefill_mma(node.outputs[0], lanes) {
            if mma.accepts(&buffers, geometry, window) {
                mma.record(&recording, buffers, geometry, window, variant)?;
                return Ok((output, keys, values));
            }
            tracing::info!(target: "infer::capture", lanes, "selected MMA does not fit this pool");
        }
        if let Some(oxide) = self.attention.prefill_oxide(node.outputs[0], lanes) {
            if oxide.accepts(&buffers, geometry) {
                oxide.record(&recording, buffers, geometry, window)?;
                return Ok((output, keys, values));
            }
            tracing::info!(target: "infer::capture", lanes, "selected SIMT does not fit this pool");
        }
        self.record_prefill_legacy(node, output, keys, values, scales)
    }
    fn record_prefill_legacy<E: DType + Element<Marker: KvElement>>(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
        keys: Tensor<E>,
        values: Tensor<E>,
        scales: [f32; 2],
    ) -> AttentionResult<E> {
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
            output: buffer(&output)?,
            query: self.input_buffer(node.inputs[0])?,
            cache: kv_cache((&keys, &values), kv_heads, head_dim, scales)?,
            metadata: buffer(self.metadata)?,
            table: buffer(self.table)?,
        };
        let recording = Recording::scope(self.scope);
        let attention = &self.weights.kernels.attention;
        // Query/head axes remain separate so each CTA addresses its own KV head.
        // Selection was measured and independently checked before graph capture.
        self.attention
            .prefill_query_tile(node.outputs[0], lanes, E::DTYPE == f8e4m3fn::DTYPE)
            .map_or_else(
                || attention.decode_prefill(&recording, plan, buffers),
                |tile| attention.decode_prefill_tiled(&recording, plan, tile, buffers),
            )
            .map_err(error)?;
        Ok((output, keys, values))
    }
}
