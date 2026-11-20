use infer_core::{Error, IdAllocator, Result, TensorId};
use infer_ir::{
    BackboneKind, DType, DataflowGraph, FeedForward, Mixer, ModelIr, StateKind, TensorNode,
    TensorOp, TensorSpec, TensorStorage,
};

pub(crate) struct Builder {
    ids: IdAllocator,
    graph: DataflowGraph,
    layer: Option<usize>,
}
impl Builder {
    fn tensor(&mut self, shape: Vec<usize>, storage: TensorStorage) -> Result<TensorId> {
        let id = self.ids.allocate()?;
        self.graph.tensors.push(TensorSpec {
            id,
            shape,
            dtype: DType::F32,
            storage,
        });
        Ok(id)
    }
    pub(crate) fn weight(&mut self, slot: String, shape: Vec<usize>) -> Result<TensorId> {
        self.tensor(shape, TensorStorage::Weight { slot })
    }
    fn node(
        &mut self,
        op: TensorOp,
        inputs: Vec<TensorId>,
        sizes: &[usize],
        states: Vec<TensorId>,
    ) -> Result<Vec<TensorId>> {
        let outputs = sizes
            .iter()
            .map(|n| self.tensor(vec![*n], TensorStorage::Activation))
            .collect::<Result<Vec<_>>>()?;
        self.graph.nodes.push(TensorNode {
            id: self.ids.allocate()?,
            layer: self.layer,
            op,
            inputs,
            outputs: outputs.clone(),
            states,
        });
        Ok(outputs)
    }
    pub(crate) fn one(
        &mut self,
        op: TensorOp,
        inputs: Vec<TensorId>,
        size: usize,
    ) -> Result<TensorId> {
        Ok(self.node(op, inputs, &[size], vec![])?[0])
    }
    pub(crate) fn linear(
        &mut self,
        x: TensorId,
        slot: String,
        input: usize,
        output: usize,
    ) -> Result<TensorId> {
        let weight = self.weight(slot, vec![output, input])?;
        self.one(TensorOp::Linear, vec![x, weight], output)
    }
    pub(crate) fn norm(
        &mut self,
        x: TensorId,
        slot: String,
        size: usize,
        dim: usize,
        model: &ModelIr,
    ) -> Result<TensorId> {
        let weight = self.weight(slot, vec![dim])?;
        self.one(
            TensorOp::Norm {
                epsilon: model.norm_epsilon,
                offset: model.norm_weight_offset,
                head_dim: dim,
            },
            vec![x, weight],
            size,
        )
    }
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::invalid("lowering dimension overflow"))
}
/// Explicit per-token decoder dataflow; host/CUDA consume the same weight/state bindings.
///
/// # Errors
/// Returns an invalid-input or unsupported error for an invalid model graph or unsupported model operations.
pub fn lower(model: &ModelIr) -> Result<DataflowGraph> {
    model.validate()?;
    if !matches!(model.backbone, BackboneKind::Decoder | BackboneKind::Hybrid) {
        return Err(Error::unsupported("non-decoder dataflow provider required"));
    }
    let hidden_size = model.hidden_size;
    let mut b = Builder {
        ids: IdAllocator::default(),
        graph: DataflowGraph::default(),
        layer: None,
    };
    let embedding = b.weight(
        "embed_tokens.weight".into(),
        vec![model.vocab_size, hidden_size],
    )?;
    let mut x = b.one(TensorOp::Embedding, vec![embedding], hidden_size)?;
    for (layer, mixer) in model.mixers.iter().enumerate() {
        b.layer = Some(layer);
        let prefix = format!("layers.{layer}");
        let norm = b.norm(
            x,
            format!("{prefix}.input_layernorm.weight"),
            hidden_size,
            hidden_size,
            model,
        )?;
        let projected = b.mixer(norm, mixer, &prefix, model, layer)?;
        x = b.one(TensorOp::Add, vec![x, projected], hidden_size)?;
        x = match &model.feed_forward {
            FeedForward::Dense { intermediate } => {
                b.feed_forward(x, &prefix, *intermediate, model)?
            }
            FeedForward::Moe {
                dense_layers,
                dense_intermediate,
                ..
            } if dense_layers.contains(&layer) => {
                b.feed_forward(x, &prefix, *dense_intermediate, model)?
            }
            FeedForward::Moe { .. } => b.moe(x, &prefix, model)?,
        };
        if layer < model.media_features {
            x = b.one(
                TensorOp::MediaFeature { index: layer },
                vec![x],
                hidden_size,
            )?;
        }
    }
    b.layer = None;
    let hidden = b.norm(x, "norm.weight".into(), hidden_size, hidden_size, model)?;
    let head = if model.tied_embeddings {
        embedding
    } else {
        b.weight("lm_head.weight".into(), vec![model.vocab_size, hidden_size])?
    };
    let logits = b.one(TensorOp::Linear, vec![hidden, head], model.vocab_size)?;
    b.graph.hidden = Some(hidden);
    b.graph.logits = Some(logits);
    b.graph.plan_lifetimes()?;
    Ok(b.graph)
}

#[derive(Clone, Copy)]
struct AttentionLayout {
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    output_gate: bool,
    qk_norm: bool,
}
#[derive(Clone, Copy)]
struct DeltaLayout {
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
}
impl Builder {
    fn mixer(
        &mut self,
        norm: TensorId,
        mixer: &Mixer,
        prefix: &str,
        model: &ModelIr,
        layer: usize,
    ) -> Result<TensorId> {
        match *mixer {
            Mixer::Attention {
                query_heads,
                kv_heads,
                head_dim,
                sliding_window,
                output_gate,
                qk_norm,
            } => self.attention(
                norm,
                prefix,
                model,
                layer,
                AttentionLayout {
                    query_heads,
                    kv_heads,
                    head_dim,
                    sliding_window,
                    output_gate,
                    qk_norm,
                },
            ),
            Mixer::LinearAttention {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                conv_kernel,
            } => self.delta(
                norm,
                prefix,
                model,
                layer,
                DeltaLayout {
                    key_heads,
                    value_heads,
                    key_dim,
                    value_dim,
                    conv_kernel,
                },
            ),
            Mixer::Recurrent { .. } => Err(Error::unsupported("recurrent graph provider required")),
        }
    }
    fn attention_projection(
        &mut self,
        norm: TensorId,
        prefix: &str,
        model: &ModelIr,
        layout: &AttentionLayout,
    ) -> Result<(TensorId, TensorId, TensorId, Option<TensorId>)> {
        let AttentionLayout {
            query_heads,
            kv_heads,
            head_dim,
            output_gate,
            qk_norm,
            ..
        } = *layout;
        let qsize = mul(query_heads, head_dim)?;
        let ksize = mul(kv_heads, head_dim)?;
        let qproject = self.linear(
            norm,
            format!("{prefix}.self_attn.q_proj.weight"),
            model.hidden_size,
            mul(qsize, if output_gate { 2 } else { 1 })?,
        )?;
        let (mut q, gate) = if output_gate {
            let split = self.node(
                TensorOp::Split {
                    widths: vec![head_dim, head_dim],
                    heads: query_heads,
                },
                vec![qproject],
                &[qsize, qsize],
                vec![],
            )?;
            (split[0], Some(split[1]))
        } else {
            (qproject, None)
        };
        let mut k = self.linear(
            norm,
            format!("{prefix}.self_attn.k_proj.weight"),
            model.hidden_size,
            ksize,
        )?;
        let vocab_size = self.linear(
            norm,
            format!("{prefix}.self_attn.v_proj.weight"),
            model.hidden_size,
            ksize,
        )?;
        if qk_norm {
            q = self.norm(
                q,
                format!("{prefix}.self_attn.q_norm.weight"),
                qsize,
                head_dim,
                model,
            )?;
            k = self.norm(
                k,
                format!("{prefix}.self_attn.k_norm.weight"),
                ksize,
                head_dim,
                model,
            )?;
        }
        Ok((q, k, vocab_size, gate))
    }
    fn attention(
        &mut self,
        norm: TensorId,
        prefix: &str,
        model: &ModelIr,
        layer: usize,
        layout: AttentionLayout,
    ) -> Result<TensorId> {
        let AttentionLayout {
            query_heads,
            kv_heads,
            head_dim,
            sliding_window,
            ..
        } = layout;
        let qsize = mul(query_heads, head_dim)?;
        let ksize = mul(kv_heads, head_dim)?;
        let (mut q, mut k, vocab_size, gate) =
            self.attention_projection(norm, prefix, model, &layout)?;
        let rotary_dim = rotary_dimension(head_dim, model)?;
        q = self.one(
            TensorOp::Rope {
                heads: query_heads,
                head_dim,
                rotary_dim,
                theta: model.position.rope_theta,
            },
            vec![q],
            qsize,
        )?;
        k = self.one(
            TensorOp::Rope {
                heads: kv_heads,
                head_dim,
                rotary_dim,
                theta: model.position.rope_theta,
            },
            vec![k],
            ksize,
        )?;
        let state = self.tensor(
            vec![2, ksize],
            TensorStorage::State {
                layer,
                kind: StateKind::AttentionKv,
            },
        )?;
        let mut attention = self.node(
            TensorOp::Attention {
                query_heads,
                kv_heads,
                head_dim,
                window: sliding_window,
            },
            vec![q, k, vocab_size],
            &[qsize],
            vec![state],
        )?[0];
        if let Some(gate) = gate {
            let gate = self.one(TensorOp::Sigmoid, vec![gate], qsize)?;
            attention = self.one(TensorOp::Multiply, vec![attention, gate], qsize)?;
        }
        self.linear(
            attention,
            format!("{prefix}.self_attn.o_proj.weight"),
            qsize,
            model.hidden_size,
        )
    }
    fn delta(
        &mut self,
        norm: TensorId,
        prefix: &str,
        model: &ModelIr,
        layer: usize,
        layout: DeltaLayout,
    ) -> Result<TensorId> {
        let DeltaLayout {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
        } = layout;
        let ksize = mul(key_heads, key_dim)?;
        let vsize = mul(value_heads, value_dim)?;
        let channels = mul(2, ksize)?
            .checked_add(vsize)
            .ok_or_else(|| Error::invalid("linear channels overflow"))?;
        let qkv = self.linear(
            norm,
            format!("{prefix}.linear_attn.in_proj_qkv.weight"),
            model.hidden_size,
            channels,
        )?;
        let z = self.linear(
            norm,
            format!("{prefix}.linear_attn.in_proj_z.weight"),
            model.hidden_size,
            vsize,
        )?;
        let beta = self.linear(
            norm,
            format!("{prefix}.linear_attn.in_proj_b.weight"),
            model.hidden_size,
            value_heads,
        )?;
        let a = self.linear(
            norm,
            format!("{prefix}.linear_attn.in_proj_a.weight"),
            model.hidden_size,
            value_heads,
        )?;
        let conv = self.weight(
            format!("{prefix}.linear_attn.conv1d.weight"),
            vec![channels, 1, conv_kernel],
        )?;
        let conv_state = self.tensor(
            vec![channels, conv_kernel],
            TensorStorage::State {
                layer,
                kind: StateKind::Conv,
            },
        )?;
        let mixed = self.node(
            TensorOp::Conv {
                channels,
                kernel: conv_kernel,
            },
            vec![qkv, conv],
            &[channels],
            vec![conv_state],
        )?[0];
        let a_log = self.weight(format!("{prefix}.linear_attn.A_log"), vec![value_heads])?;
        let dt_bias = self.weight(format!("{prefix}.linear_attn.dt_bias"), vec![value_heads])?;
        let state = self.tensor(
            vec![value_heads, key_dim, value_dim],
            TensorStorage::State {
                layer,
                kind: StateKind::LinearAttention,
            },
        )?;
        let delta = self.node(
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            },
            vec![mixed, beta, a, a_log, dt_bias],
            &[vsize],
            vec![state],
        )?[0];
        let weight = self.weight(format!("{prefix}.linear_attn.norm.weight"), vec![value_dim])?;
        let gated = self.one(
            TensorOp::GatedNorm {
                head_dim: value_dim,
                epsilon: model.norm_epsilon,
            },
            vec![delta, z, weight],
            vsize,
        )?;
        self.linear(
            gated,
            format!("{prefix}.linear_attn.out_proj.weight"),
            vsize,
            model.hidden_size,
        )
    }
    fn feed_forward(
        &mut self,
        x: TensorId,
        prefix: &str,
        intermediate: usize,
        model: &ModelIr,
    ) -> Result<TensorId> {
        let normalized = self.norm(
            x,
            format!("{prefix}.post_attention_layernorm.weight"),
            model.hidden_size,
            model.hidden_size,
            model,
        )?;
        let gate = self.linear(
            normalized,
            format!("{prefix}.mlp.gate_proj.weight"),
            model.hidden_size,
            intermediate,
        )?;
        let up = self.linear(
            normalized,
            format!("{prefix}.mlp.up_proj.weight"),
            model.hidden_size,
            intermediate,
        )?;
        let active = self.one(TensorOp::Silu, vec![gate], intermediate)?;
        let product = self.one(TensorOp::Multiply, vec![active, up], intermediate)?;
        let down = self.linear(
            product,
            format!("{prefix}.mlp.down_proj.weight"),
            intermediate,
            model.hidden_size,
        )?;
        self.one(TensorOp::Add, vec![x, down], model.hidden_size)
    }
}
#[expect(
    clippy::cast_possible_truncation,
    reason = "Partial RoPE intentionally truncates a validated fraction of the head dimension, matching the model configuration contract"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
#[expect(
    clippy::cast_sign_loss,
    reason = "Partial RoPE intentionally truncates a validated fraction of the head dimension, matching the model configuration contract"
)]
fn rotary_dimension(head_dim: usize, model: &ModelIr) -> Result<usize> {
    let dim = (head_dim as f64 * model.position.rotary_fraction) as usize;
    if dim > head_dim || !dim.is_multiple_of(2) {
        return Err(Error::invalid("invalid partial rotary dimension"));
    }
    Ok(dim)
}
