use super::{
    attention::attention,
    capture::{Capture, CaptureMode, error},
};
use crate::constants::ATTENTION_SPLIT_THRESHOLD;
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::WIDE_PREFILL_LANES;
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
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
        let mut keys = keys.reshape(&[kv_heads, rows, head_dim])?;
        let mut values = values.reshape(&[kv_heads, rows, head_dim])?;
        let result = self
            .scope
            .record(
                attention::append(
                    (&mut keys).partition([1, rows, head_dim]),
                    (&mut values).partition([1, rows, head_dim]),
                    &self.input(node.inputs[1])?.view(&[kv_heads, head_dim])?,
                    &self.input(node.inputs[2])?.view(&[kv_heads, head_dim])?,
                    self.metadata,
                    self.table,
                    scales[0],
                    scales[1],
                )
                .generics(vec![
                    f8e4m3fn::DTYPE.as_str().into(),
                    head_dim.to_string(),
                    rows.to_string(),
                    "1".into(),
                ]),
            )
            .map(|_| ());
        self.fp8_states.insert(id, (keys, values));
        result
    }

    pub fn record_attention<E: DType>(
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
        let mut keys = keys.reshape(&[kv_heads, rows, head_dim])?;
        let mut values = values.reshape(&[kv_heads, rows, head_dim])?;
        self.scope.record(
            attention::append(
                (&mut keys).partition([1, rows, head_dim]),
                (&mut values).partition([1, rows, head_dim]),
                &self.input(node.inputs[1])?.view(&[kv_heads, head_dim])?,
                &self.input(node.inputs[2])?.view(&[kv_heads, head_dim])?,
                self.metadata,
                self.table,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                head_dim.to_string(),
                rows.to_string(),
                i32::from(E::DTYPE != f32::DTYPE).to_string(),
            ]),
        )?;
        let mut output = output.reshape(&[query_heads, head_dim])?;
        if self.attention.supports(query_heads, head_dim)
            && window.is_none_or(|w| w >= ATTENTION_SPLIT_THRESHOLD)
        {
            let query = if let Some(query) = self.weights.constants.get(&node.inputs[0]) {
                query.view(&[query_heads * head_dim])?
            } else {
                match self.mode {
                    CaptureMode::Row(lane) => {
                        self.arena.row(node.inputs[0], lane).map_err(error)?
                    }
                    _ => self
                        .arena
                        .get(node.inputs[0])
                        .map_err(error)?
                        .view(&[query_heads * head_dim])?,
                }
            };
            self.attention.record(
                self.scope,
                &mut output,
                &query.view(&[query_heads, head_dim])?,
                &keys,
                &values,
                self.metadata,
                self.table,
                query_heads,
                kv_heads,
                head_dim,
                i32::try_from(window.unwrap_or(0)).map_err(error)?,
                scales,
            )?;
            return Ok((output, keys, values));
        }
        self.scope.record(
            attention::decode(
                (&mut output).partition([1, head_dim]),
                &self.input(node.inputs[0])?.view(&[query_heads, head_dim])?,
                &keys,
                &values,
                self.metadata,
                self.table,
                i32::try_from(window.unwrap_or(0)).map_err(error)?,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                head_dim.to_string(),
                (query_heads / kv_heads).to_string(),
                KV_BLOCK_TOKENS.to_string(),
            ]),
        )?;
        Ok((output, keys, values))
    }
    fn record_prefill_attention<E: DType>(
        &mut self,
        node: &TensorNode,
        output: Tensor<f32>,
        keys: Tensor<E>,
        values: Tensor<E>,
        scales: [f32; 2],
    ) -> AttentionResult<E> {
        use super::attention_prefill::kernels;
        let TensorOp::Attention {
            kv_heads, head_dim, ..
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        let lanes = self.arena.lanes();
        // A paged slot's cache holds the pool's physical rows, not one sequence's capacity.
        let rows = keys.size() / (kv_heads * head_dim);
        let mut keys = keys.reshape(&[kv_heads, rows, head_dim])?;
        let mut values = values.reshape(&[kv_heads, rows, head_dim])?;
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
        let lookup = lanes.div_ceil(KV_BLOCK_TOKENS).next_power_of_two();
        self.scope.record(
            kernels::append(
                (&mut keys).partition([1, append_rows, head_dim]),
                (&mut values).partition([1, append_rows, head_dim]),
                &self
                    .input(node.inputs[1])?
                    .view(&[lanes, kv_heads, head_dim])?,
                &self
                    .input(node.inputs[2])?
                    .view(&[lanes, kv_heads, head_dim])?,
                self.metadata,
                self.table,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                head_dim.to_string(),
                append_rows.to_string(),
                i32::from(E::DTYPE != f32::DTYPE).to_string(),
                KV_BLOCK_TOKENS.to_string(),
                lookup.to_string(),
            ]),
        )?;
        self.record_prefill_decode(node, output, keys, values, scales)
    }

    fn record_prefill_decode<E: DType>(
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
        let mut output = output.reshape(&[lanes * query_heads, head_dim])?;
        let query = if let Some(value) = self.weights.constants.get(&node.inputs[0]) {
            value
        } else {
            self.arena.get(node.inputs[0]).map_err(error)?
        };
        // The choice was measured for a slot's layout; a pool's operands may not fit it. That is
        // decided here, in the open, and a selected kernel then records or fails the capture.
        let geometry = (query_heads, kv_heads, head_dim, lanes, self.capacity);
        let window = i32::try_from(window.unwrap_or(0)).map_err(error)?;
        if let Some((mma, variant)) = self.attention.prefill_mma(node.outputs[0], lanes) {
            let bindings = super::attention_mma::Bindings {
                output: &mut output,
                query,
                keys: &keys,
                values: &values,
                metadata: self.metadata,
                table: self.table,
            };
            if mma.accepts(&bindings, geometry, window, scales)? {
                mma.record(self.scope, bindings, geometry, window, scales, variant)?;
                return Ok((output, keys, values));
            }
            tracing::info!(target: "infer::capture", lanes, "selected MMA does not fit this pool");
        }
        if let Some(oxide) = self.attention.prefill_oxide(node.outputs[0], lanes) {
            let bindings = super::attention_oxide::Bindings {
                output: &mut output,
                query,
                keys: &keys,
                values: &values,
                metadata: self.metadata,
                table: self.table,
            };
            if oxide.accepts(&bindings, geometry, scales)? {
                oxide.record(self.scope, bindings, geometry, window, scales)?;
                return Ok((output, keys, values));
            }
            tracing::info!(target: "infer::capture", lanes, "selected SIMT does not fit this pool");
        }
        self.record_prefill_legacy(node, output, keys, values, scales)
    }
    fn record_prefill_legacy<E: DType>(
        &self,
        node: &TensorNode,
        mut output: Tensor<f32>,
        keys: Tensor<E>,
        values: Tensor<E>,
        scales: [f32; 2],
    ) -> AttentionResult<E> {
        use super::attention_prefill::kernels;
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
        // Query/head axes remain separate so each CTA addresses its own KV head.
        // Selection was measured and independently checked before graph capture.
        if let Some(query_tile) =
            self.attention
                .prefill_query_tile(node.outputs[0], lanes, E::DTYPE == f8e4m3fn::DTYPE)
        {
            let scale = 1.0f32 / f32::from(u16::try_from(head_dim).map_err(error)?).sqrt();
            let query = self.input(node.inputs[0])?;
            let query = query.view(&[lanes, query_heads * head_dim])?;
            let mut flat = output.reshape(&[lanes, query_heads * head_dim])?;
            self.scope.record(
                kernels::decode_tiled(
                    (&mut flat).partition([query_tile, head_dim]),
                    &query,
                    &keys,
                    &values,
                    self.metadata,
                    self.table,
                    i32::try_from(window.unwrap_or(0)).map_err(error)?,
                    scales[0],
                    scales[1],
                    scale,
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    head_dim.to_string(),
                    (query_heads / kv_heads).to_string(),
                    query_tile.to_string(),
                    KV_BLOCK_TOKENS.to_string(),
                    KV_BLOCK_TOKENS.to_string(),
                    "1".into(),
                ]),
            )?;
            return Ok((
                flat.reshape(&[lanes * query_heads, head_dim])?,
                keys,
                values,
            ));
        }
        self.scope.record(
            kernels::decode(
                (&mut output).partition([1, head_dim]),
                &self
                    .input(node.inputs[0])?
                    .view(&[lanes * query_heads, head_dim])?,
                &keys,
                &values,
                self.metadata,
                self.table,
                i32::try_from(window.unwrap_or(0)).map_err(error)?,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                head_dim.to_string(),
                (query_heads / kv_heads).to_string(),
                query_heads.to_string(),
                KV_BLOCK_TOKENS.to_string(),
            ]),
        )?;
        Ok((output, keys, values))
    }
}
