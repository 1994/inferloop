//! Model extension contract.
use super::{GenerationPolicy, ProviderMetadata};
use infer_core::{ModelId, Result};
use infer_ir::{CapabilityRequirements, DType, Modality, ModelIr, PrecisionPlan};
use serde::{Deserialize, Serialize};

/// Everything one model family customizes, produced once from its configuration.
///
/// This is the single place a family is described: canonical IR, device requirements,
/// precision policy, speculative draft head and extra modalities. Engines consume it
/// declaratively, so no core path branches on a model name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedModel {
    /// Backbone description every backend consumes.
    pub model: ModelIr,
    /// Draft layers the configuration declares; zero disables speculation.
    pub draft_layers: usize,
    /// Device capabilities this family requires; validated before any weight is bound.
    #[serde(default)]
    pub requirements: CapabilityRequirements,
    /// Storage/compute policy, resolved against the device that will run the model.
    #[serde(default)]
    pub precision: PrecisionPolicy,
    /// Speculative draft head, when the family ships one.
    #[serde(default)]
    pub speculation: Option<SpeculationPlan>,
    /// Non-text modalities the family can encode.
    #[serde(default)]
    pub modalities: Vec<ModalityPlan>,
}

impl ImportedModel {
    /// Family description with only the canonical model and draft depth set.
    #[must_use]
    pub fn new(model: ModelIr, draft_layers: usize) -> Self {
        Self {
            model,
            draft_layers,
            requirements: CapabilityRequirements::default(),
            precision: PrecisionPolicy::default(),
            speculation: None,
            modalities: Vec::new(),
        }
    }
}

/// Storage/compute policy of a family, selected against device capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionPolicy {
    /// One plan for every device.
    Fixed(PrecisionPlan),
    /// First plan the device supports; the last entry is the fallback. This is where a family
    /// expresses "native FP4 when available, otherwise decode", without naming an architecture.
    Preferred(Vec<PrecisionPlan>),
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::Fixed(PrecisionPlan::f32())
    }
}

impl PrecisionPolicy {
    /// Plan to run on a device whose `supports` predicate answers for native compute dtypes.
    #[must_use]
    pub fn resolve(&self, supports: impl Fn(DType) -> bool) -> PrecisionPlan {
        match self {
            Self::Fixed(plan) => plan.clone(),
            Self::Preferred(plans) => plans
                .iter()
                .find(|plan| supports(plan.compute))
                .or_else(|| plans.last())
                .cloned()
                .unwrap_or_else(PrecisionPlan::f32),
        }
    }
}

/// Speculative draft head: the package prefix and the slots it fuses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculationPlan {
    /// Package prefix holding draft tensors, for example `mtp.`.
    pub prefix: String,
    /// Draft layers declared by the head.
    pub layers: usize,
    /// Fusion projection the head applies to embedding and hidden states.
    #[serde(default)]
    pub fusion: Option<FusionPlan>,
}

/// Fusion projection of a speculative draft head.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionPlan {
    /// Canonical slot of the fused projection, relative to [`SpeculationPlan::prefix`].
    pub projection: String,
    /// Canonical slots of the embedding and hidden normalization pair.
    pub norms: [String; 2],
}

/// One non-text modality a family can encode, with its media placeholder tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalityPlan {
    /// Modality the family accepts.
    pub modality: Modality,
    /// Token ids that stand in for media in the text stream.
    pub placeholder_tokens: Vec<u32>,
    /// Encoder that turns raw media into text-aligned embeddings.
    #[serde(default)]
    pub encoder: Option<ModalityEncoder>,
}

/// Encoder geometry of one modality, declared by the model provider and consumed by backends.
///
/// Nothing here names a model: a backend builds its kernels from this description, so a new
/// encoder family (a different `ViT`, or a vision-language variant) is a provider change, not a
/// backend change. Checkpoint tensor names are derived from [`Self::prefix`] by
/// `infer_models::vision::slots`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalityEncoder {
    /// Package prefix holding the encoder tensors, for example `model.visual.`.
    pub prefix: String,
    /// Transformer blocks.
    pub depth: usize,
    /// Hidden width of every block.
    pub hidden_size: usize,
    /// MLP intermediate width.
    pub intermediate_size: usize,
    /// Attention heads; `hidden_size / heads` is the head dimension.
    pub heads: usize,
    /// Input channels of one patch.
    pub in_channels: usize,
    /// Spatial patch edge; also the conv stride.
    pub patch_size: usize,
    /// Temporal patch depth; a still image repeats frames to fill it.
    pub temporal_patch_size: usize,
    /// Spatial merge edge applied before the projector (`2` means 2×2 tokens per output).
    pub spatial_merge_size: usize,
    /// Width the encoder emits to the language model.
    pub out_hidden_size: usize,
    /// Learned absolute position table size; its square root is the table grid.
    pub position_embeddings: usize,
    /// Axial `RoPE` theta applied per spatial axis.
    pub rope_theta: f64,
    /// `LayerNorm` epsilon of blocks and merger.
    pub norm_epsilon: f32,
    /// Activation of the block MLP. The merger uses exact erf GELU regardless.
    pub hidden_activation: HiddenActivation,
}

/// Activation functions an encoder may declare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HiddenActivation {
    /// `gelu_pytorch_tanh`.
    GeluTanh,
    /// Exact `gelu` (erf).
    GeluErf,
    /// `silu`.
    Silu,
}

/// One model architecture family, addressed by name rather than by code path.
///
/// Adding a model means implementing this trait and registering the implementation with
/// `infer_models::ModelRegistry`; no core dispatch changes. Everything the family customizes is
/// defined by [`ModelProvider::import`], [`ModelProvider::generation_policy`], [`ModelProvider::graph`],
/// [`ModelProvider::draft_graph`] and the weight-naming hooks below, so a new family — a
/// decoder, a hybrid, or a full omni model with several modalities — is one implementation in one
/// place. Providers are shared across loading threads.
pub trait ModelProvider: Send + Sync {
    /// Identity, name and SPI version of this provider.
    fn metadata(&self) -> ProviderMetadata;
    /// Hugging Face `architectures` or `model_type` values this provider claims. A configuration
    /// is routed to the provider that claims one of its hints; the first registered match wins.
    fn architectures(&self) -> &'static [&'static str];
    /// Provide family defaults from the model configuration and optional bounded model card.
    /// Resolution applies package defaults, then mode presets, then request overrides.
    /// # Errors
    /// Rejects invalid model metadata needed to select a generation policy.
    fn generation_policy(
        &self,
        _config: &[u8],
        _model_card: Option<&str>,
    ) -> Result<GenerationPolicy> {
        Ok(GenerationPolicy::default())
    }
    /// Import one configuration into the canonical family description.
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error for malformed or unsupported configuration.
    fn import(&self, id: ModelId, config: &[u8]) -> Result<ImportedModel>;
    /// Define execution topology, independently of the device implementation.
    /// # Errors
    /// Rejects a model for which this provider has no execution recipe.
    fn graph(&self, _model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        Err(infer_core::Error::unsupported(
            "model provider has no execution recipe",
        ))
    }
    /// Define the speculative draft topology and its model geometry.
    /// # Errors
    /// Rejects providers without a supported draft recipe.
    fn draft_graph(&self, _model: &ModelIr) -> Result<(ModelIr, infer_ir::DataflowGraph)> {
        Err(infer_core::Error::unsupported(
            "model provider has no draft recipe",
        ))
    }
    /// Candidate Hugging Face prefixes for canonical weight slots, most specific first. The
    /// prefix whose [`Self::anchor_slot`] tensor exists in the package is the one used.
    fn weight_prefixes(&self) -> &'static [&'static str] {
        &["model.", ""]
    }
    /// Canonical slot whose presence selects one entry of [`Self::weight_prefixes`].
    fn anchor_slot(&self) -> &'static str {
        "embed_tokens.weight"
    }
    /// Hugging Face tensor name for one canonical weight slot under the resolved prefix.
    fn weight_source(&self, slot: &str, prefix: &str) -> String {
        format!("{prefix}{slot}")
    }
}
