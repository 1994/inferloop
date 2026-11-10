use super::{
    arena::SlotMemory,
    attention_lowering::{Lowering, Mode, Prepared, lower},
    buffers::buffer,
    capture::{Capture, CaptureMode, error},
};
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::WIDE_PREFILL_LANES;
use crate::tuning::attention::Choice;
use infer_core::Result;
use infer_cuda_kernels::contract::{
    AppendBuffers, Buffer, DecodeBuffers, DecodePlan, KvBuffers, KvCache, PrefillAppendPlan,
    PrefillDecodePlan,
};
use infer_ir::{TensorNode, TensorOp};

impl Capture<'_> {
    /// Append this lane's key and value row to the shared FP8 cache without attending; a
    /// verification step then attends all its lanes in one grouped launch.
    pub fn record_attention_append(&self, node: &TensorNode) -> Result<()> {
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
    ) -> Result<()> {
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
        let prepared = self.attention.prepared(query_heads, head_dim);
        if lower(Mode::Decode, window, prepared) == Lowering::Split {
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
    fn record_append(&self, node: &TensorNode, cache: KvCache<'_>) -> Result<()> {
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
    ) -> Result<()> {
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
    ) -> Result<()> {
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
        let selected = self.attention.selected_prefill(node.outputs[0], lanes);
        let fits = match selected {
            Some(Choice::Mma { .. } | Choice::MmaF16) => self
                .attention
                .prefill_mma(node.outputs[0], lanes)
                .is_some_and(|(mma, _)| mma.accepts(&buffers, geometry, window)),
            Some(choice) if choice.uses_oxide() => self
                .attention
                .prefill_oxide(node.outputs[0], lanes)
                .is_some_and(|oxide| oxide.accepts(&buffers, geometry)),
            _ => true,
        };
        if !fits {
            tracing::info!(target: "infer::capture", lanes, ?selected, "selected prefill attention does not fit this pool");
        }
        let fp8 = matches!(cache.buffers, KvBuffers::Fp8 { .. });
        let mode = Mode::Prefill {
            selected,
            fp8,
            fits,
        };
        let recording = self.scope;
        match lower(mode, None, Prepared::default()) {
            Lowering::Mma => {
                let (mma, variant) = self
                    .attention
                    .prefill_mma(node.outputs[0], lanes)
                    .ok_or_else(|| error("selected MMA attention"))?;
                mma.record(recording, buffers, geometry, window, variant)
            }
            Lowering::Simt => self
                .attention
                .prefill_oxide(node.outputs[0], lanes)
                .ok_or_else(|| error("selected SIMT attention"))?
                .record(recording, buffers, geometry, window),
            Lowering::Tiled(tile) => {
                self.record_prefill_legacy(node, buffers.output, cache, Some(tile))
            }
            _ => self.record_prefill_legacy(node, buffers.output, cache, None),
        }
    }
    fn record_prefill_legacy(
        &self,
        node: &TensorNode,
        output: Buffer<'_, f32>,
        cache: KvCache<'_>,
        tile: Option<usize>,
    ) -> Result<()> {
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
        let recording = self.scope;
        let attention = &self.weights.kernels.attention;
        // Query/head axes remain separate so each CTA addresses its own KV head.
        // Selection was measured and independently checked before graph capture.
        tile.map_or_else(
            || attention.decode_prefill(recording, plan, buffers),
            |tile| attention.decode_prefill_tiled(recording, plan, tile, buffers),
        )
        .map_err(error)?;
        Ok(())
    }
}
