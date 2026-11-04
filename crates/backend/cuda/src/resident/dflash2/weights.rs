//! Device weights of an attached `DFlash2` draft, with the target tensors it shares.
use crate::device::CudaDevice;
use crate::loading::floats;
use crate::mlp::ProjectionWeight;
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_ir::{DataflowGraph, TensorOp};
use infer_models::{DFlash2Package, DeviceWeight, QuantizedPackage, TensorDtype};

use super::super::ProgramWeights;

/// Read budget for one BF16 selector codebook.
const CODEBOOK_READ_BYTES: u64 = 1 << 30;

/// Static draft dimensions, taken from its configuration.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocabulary: usize,
    pub block: usize,
    pub window: usize,
    pub rank: usize,
    pub top_k: usize,
    pub groups: usize,
    pub group_size: usize,
    pub mask_token: u32,
    pub epsilon: f32,
}

/// One dynamic two-tap convolution: the base kernel `[side * tap, channels]` and the
/// projection producing its per-row group coefficients.
pub struct Convolution {
    pub base: Allocation<f32>,
    pub projection: ProjectionWeight,
}

pub struct Layer {
    pub input_norm: Allocation<f32>,
    pub post_norm: Allocation<f32>,
    pub q_norm: Allocation<f32>,
    pub k_norm: Allocation<f32>,
    pub attention_conv: Convolution,
    pub mlp_conv: Convolution,
    pub q: ProjectionWeight,
    pub k: ProjectionWeight,
    pub v: ProjectionWeight,
    pub o: ProjectionWeight,
    pub gate: ProjectionWeight,
    pub up: ProjectionWeight,
    pub down: ProjectionWeight,
}

pub struct DraftWeights {
    pub geometry: Geometry,
    pub layers: Vec<Layer>,
    pub hidden_norm: Allocation<f32>,
    pub norm: Allocation<f32>,
    pub selector: ProjectionWeight,
    pub predecessor: Allocation<bf16>,
    pub successor: Allocation<bf16>,
    /// The target's input embedding and output head: the checkpoint ships neither.
    pub embedding: Allocation<bf16>,
    pub head: ProjectionWeight,
    pub frequencies: Allocation<f32>,
}

impl DraftWeights {
    /// Upload the attached draft and bind the target's embedding and head to it.
    /// # Errors
    /// Rejects a package without an attached draft, unsupported encodings, a target graph
    /// without a bound embedding or head, and CUDA allocation failures.
    pub fn load(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        target: &ProgramWeights,
    ) -> Result<Self> {
        let draft = package
            .dflash2
            .take()
            .ok_or_else(|| Error::invalid("no DFlash2 draft is attached"))?;
        let result = Self::bind(device, package, &draft, target);
        package.dflash2 = Some(draft);
        result
    }

    fn bind(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        draft: &DFlash2Package,
        target: &ProgramWeights,
    ) -> Result<Self> {
        let geometry = Geometry::of(draft)?;
        let mut source = Source {
            device,
            package,
            draft,
        };
        let layers = (0..draft.config.num_hidden_layers)
            .map(|layer| source.layer(layer))
            .collect::<Result<_>>()?;
        let half = geometry.head_dim / 2;
        let theta = draft.config.rope_theta();
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            reason = "rotary frequencies are computed in F64 and stored in F32 like the target's"
        )]
        let frequencies = (0..half)
            .map(|i| theta.powf(-((2 * i) as f64) / geometry.head_dim as f64) as f32)
            .collect();
        let (embedding, head) = shared_target(&source.package.graph, target)?;
        Ok(Self {
            geometry,
            layers,
            hidden_norm: source.vector("hidden_norm.weight")?,
            norm: source.vector("norm.weight")?,
            selector: source.projection("candidate_selector.hidden_projection.weight")?,
            predecessor: source.codebook("candidate_selector.predecessor_codebook")?,
            successor: source.codebook("candidate_selector.successor_codebook")?,
            embedding,
            head,
            frequencies: device.runtime().upload(frequencies)?,
        })
    }

    /// Device bytes the draft's own weights occupy, excluding the shared target tensors.
    pub fn bytes(&self) -> u64 {
        let projection = |weight: &ProjectionWeight| weight.bytes() as u64;
        let mut total = bytes(&self.hidden_norm)
            + bytes(&self.norm)
            + projection(&self.selector)
            + bytes(&self.predecessor)
            + bytes(&self.successor)
            + bytes(&self.frequencies);
        for layer in &self.layers {
            total += [
                &layer.input_norm,
                &layer.post_norm,
                &layer.q_norm,
                &layer.k_norm,
                &layer.attention_conv.base,
                &layer.mlp_conv.base,
            ]
            .into_iter()
            .map(bytes)
            .sum::<u64>();
            total += [
                &layer.attention_conv.projection,
                &layer.mlp_conv.projection,
                &layer.q,
                &layer.k,
                &layer.v,
                &layer.o,
                &layer.gate,
                &layer.up,
                &layer.down,
            ]
            .into_iter()
            .map(projection)
            .sum::<u64>();
        }
        total
    }
}

impl Geometry {
    fn of(draft: &DFlash2Package) -> Result<Self> {
        let config = &draft.config;
        let mask_token = config
            .dflash_config
            .mask_token_id
            .ok_or_else(|| Error::invalid("draft has no mask token"))?;
        Ok(Self {
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            vocabulary: config.vocab_size,
            block: config.dflash_config.block_size,
            window: config.sliding_window,
            rank: config.dflash_config.selector_rank,
            top_k: config.dflash_config.selector_top_k,
            groups: config.hidden_size / config.dflash_config.conv_group_size,
            group_size: config.dflash_config.conv_group_size,
            mask_token: u32::try_from(mask_token).map_err(|e| Error::invalid(e.to_string()))?,
            // The norms run in F32; the configured epsilon is far above F32 resolution.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "an RMSNorm epsilon is representable in F32"
            )]
            epsilon: config.rms_norm_eps as f32,
        })
    }
}

/// Uploads the attached draft's tensors by checkpoint name.
struct Source<'a> {
    device: &'a CudaDevice,
    package: &'a mut QuantizedPackage,
    draft: &'a DFlash2Package,
}

impl Source<'_> {
    fn weight(&self, name: &str) -> Result<&DeviceWeight> {
        self.draft
            .weights
            .get(name)
            .ok_or_else(|| Error::invalid(format!("draft is missing {name}")))
    }

    fn vector(&mut self, name: &str) -> Result<Allocation<f32>> {
        let source = self.weight(name)?.data.clone();
        let values = floats(self.package, &source)?;
        self.device.runtime().upload(values)
    }

    fn projection(&mut self, name: &str) -> Result<ProjectionWeight> {
        let weight = self.weight(name)?.clone();
        ProjectionWeight::load(self.device, self.package, &weight)
    }

    fn codebook(&mut self, name: &str) -> Result<Allocation<bf16>> {
        let weight = self.weight(name)?.clone();
        if weight.data.dtype != TensorDtype::BF16 {
            return Err(Error::unsupported(
                "DFlash2 selector codebooks must be BF16",
            ));
        }
        let bytes = self.package.read(&weight.data, CODEBOOK_READ_BYTES)?;
        let values = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
            .collect();
        self.device.runtime().upload(values)
    }

    fn layer(&mut self, layer: usize) -> Result<Layer> {
        let name = |suffix: &str| format!("layers.{layer}.{suffix}");
        Ok(Layer {
            input_norm: self.vector(&name("input_layernorm.weight"))?,
            post_norm: self.vector(&name("post_attention_layernorm.weight"))?,
            q_norm: self.vector(&name("self_attn.q_norm.weight"))?,
            k_norm: self.vector(&name("self_attn.k_norm.weight"))?,
            attention_conv: Convolution {
                base: self.vector(&name("attention_conv.base_kernel"))?,
                projection: self.projection(&name("attention_conv.kernel_projection.weight"))?,
            },
            mlp_conv: Convolution {
                base: self.vector(&name("mlp_conv.base_kernel"))?,
                projection: self.projection(&name("mlp_conv.kernel_projection.weight"))?,
            },
            q: self.projection(&name("self_attn.q_proj.weight"))?,
            k: self.projection(&name("self_attn.k_proj.weight"))?,
            v: self.projection(&name("self_attn.v_proj.weight"))?,
            o: self.projection(&name("self_attn.o_proj.weight"))?,
            gate: self.projection(&name("mlp.gate_proj.weight"))?,
            up: self.projection(&name("mlp.up_proj.weight"))?,
            down: self.projection(&name("mlp.down_proj.weight"))?,
        })
    }
}

fn bytes<T: DType>(memory: &Allocation<T>) -> u64 {
    (memory.len() * size_of::<T>()) as u64
}

/// The target graph's token embedding and language-model head.
fn shared_target(
    graph: &DataflowGraph,
    target: &ProgramWeights,
) -> Result<(Allocation<bf16>, ProjectionWeight)> {
    let embedding = graph
        .nodes
        .iter()
        .find(|node| node.op == TensorOp::Embedding)
        .and_then(|node| target.embeddings.get(&node.inputs[0]))
        .ok_or_else(|| Error::invalid("target has no bound token embedding"))?;
    let logits = graph
        .logits
        .ok_or_else(|| Error::invalid("target graph has no logits"))?;
    let head = graph
        .nodes
        .iter()
        .find(|node| node.outputs.contains(&logits) && node.op == TensorOp::Linear)
        .and_then(|node| target.projections.get(&node.inputs[1]))
        .ok_or_else(|| Error::invalid("target has no bound language-model head"))?;
    Ok((embedding.clone(), head.clone()))
}
