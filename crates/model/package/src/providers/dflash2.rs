//! `DFlash2` draft geometry and the compatibility contract a target has to satisfy.
//!
//! D1's first step: the geometry the official draft declares, the checks that decide whether it can
//! be bound to a target at all, and the part of the resource quote that geometry alone determines.
//! Nothing here touches a device - the draft graph, its CUDA module and the proposal SPI come
//! later, and the SPI that carries proposals belongs to I1.
use infer_core::{Error, Result};
use infer_ir::{BackboneKind, ModelIr, Operation};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

/// The architecture string the official draft declares in its configuration.
pub const ARCHITECTURE: &str = "DFlash2DraftModel";

/// Operations the draft's fused modules require from a backend.
///
/// The mapping from the plan's module list - feature fusion, double-tap dynamic convolution,
/// block-internal non-causal attention, candidate selector - onto IR operations is provisional
/// until the draft graph lands. What D1 accepts is the check below: whatever the loader lowers to
/// must be a subset of what the backend reports.
pub const REQUIRED_OPERATIONS: [Operation; 5] = [
    Operation::Convolution,
    Operation::Attention,
    Operation::MatMul,
    Operation::RmsNorm,
    Operation::Residual,
];

/// Checks that a configuration declares this draft architecture.
///
/// An explicit `DFlash2` request must never be served silently by another algorithm, so the loader
/// calls this before it maps any field.
///
/// # Errors
/// Rejects a configuration whose declared architectures do not include [`ARCHITECTURE`].
pub fn check_declared_architecture(declared: &[String]) -> Result<()> {
    if declared.iter().any(|name| name == ARCHITECTURE) {
        return Ok(());
    }
    Err(Error::invalid(format!(
        "configuration does not declare {ARCHITECTURE}"
    )))
}

/// The draft's own `dflash_config` block.
#[derive(Debug, Clone, Deserialize)]
pub struct DFlash2DraftConfig {
    pub block_size: usize,
    pub conv_group_size: usize,
    pub conv_kernel_size: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    /// Token the draft embeds in the slots it proposes into; absent disables the replacement.
    pub mask_token_id: Option<usize>,
    /// Target layers whose hidden states the draft consumes.
    pub target_layer_ids: Vec<usize>,
}

/// `z-lab/Qwen3.8-27B-DFlash2`'s `config.json` as the model declares it.
///
/// Only the fields this contract reads are listed; the loader maps the rest.
#[derive(Debug, Clone, Deserialize)]
pub struct DFlash2Config {
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    /// Target layers the checkpoint was trained against.
    pub num_target_layers: usize,
    /// Blocks attend inside themselves, so the draft graph is deliberately not causal.
    pub is_causal: bool,
    pub tie_word_embeddings: bool,
    /// Vocabulary boundaries the draft shares with its target.
    pub eos_token_id: Option<usize>,
    pub pad_token_id: Option<usize>,
    /// Epsilon of every draft `RMSNorm`.
    #[serde(default = "default_norm_epsilon")]
    pub rms_norm_eps: f64,
    /// Rotary parameters of the draft's own queries and keys.
    #[serde(default)]
    pub rope_parameters: Option<DFlash2Rope>,
    pub dflash_config: DFlash2DraftConfig,
}

/// The draft's `rope_parameters` block; only the default rotary type is declared.
#[derive(Debug, Clone, Deserialize)]
pub struct DFlash2Rope {
    pub rope_theta: f64,
}

/// Epsilon the reference applies when a configuration omits `rms_norm_eps`.
const DEFAULT_NORM_EPSILON: f64 = 1e-6;
/// Rotary base the reference applies when a configuration omits `rope_theta`.
const DEFAULT_ROPE_THETA: f64 = 1_000_000.0;

const fn default_norm_epsilon() -> f64 {
    DEFAULT_NORM_EPSILON
}

impl DFlash2Config {
    /// # Errors
    /// Rejects bytes that are not this draft's configuration, or that declare another architecture.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let config: Self = serde_json::from_slice(bytes)
            .map_err(|error| Error::invalid(format!("draft configuration is invalid: {error}")))?;
        check_declared_architecture(&config.architectures)?;
        Ok(config)
    }

    /// Rotary base of the draft attention; the reference's default applies when it is omitted.
    #[must_use]
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map_or(DEFAULT_ROPE_THETA, |rope| rope.rope_theta)
    }

    #[must_use]
    pub fn geometry(&self) -> DraftGeometry {
        DraftGeometry {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            layers: self.num_hidden_layers,
            target_layers: self.num_target_layers,
            attention_heads: self.num_attention_heads,
            key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            block_size: self.dflash_config.block_size,
            target_taps: self.dflash_config.target_layer_ids.clone(),
            sliding_window: self.sliding_window,
            selector_rank: self.dflash_config.selector_rank,
            selector_top_k: self.dflash_config.selector_top_k,
        }
    }

    /// Checks the draft's special tokens against its own vocabulary.
    ///
    /// # Errors
    /// Rejects a packed draft whose mask token is missing or outside the vocabulary, or whose
    /// end-of-sequence and padding tokens disagree with themselves.
    pub fn check_special_tokens(&self) -> Result<()> {
        match self.dflash_config.mask_token_id {
            None => {
                return Err(Error::invalid(
                    "draft must declare the token it embeds in proposed slots",
                ));
            }
            Some(mask) if mask >= self.vocab_size => {
                return Err(Error::invalid(format!(
                    "draft mask token {mask} is outside its vocabulary of {}",
                    self.vocab_size
                )));
            }
            Some(_) => {}
        }
        for (name, token) in [
            ("eos_token_id", self.eos_token_id),
            ("pad_token_id", self.pad_token_id),
        ] {
            if let Some(token) = token
                && token >= self.vocab_size
            {
                return Err(Error::invalid(format!(
                    "draft {name} {token} is outside its vocabulary of {}",
                    self.vocab_size
                )));
            }
        }
        Ok(())
    }

    /// Checks that a target's tokenizer boundaries are the ones this draft was packed against.
    ///
    /// # Errors
    /// Rejects a pairing whose end-of-sequence or padding tokens differ, which would mean the draft
    /// was packed against a different vocabulary than the target it is being attached to.
    pub fn check_target_tokens(
        &self,
        eos_token_id: usize,
        pad_token_id: Option<usize>,
    ) -> Result<()> {
        for (name, mine, theirs) in [
            ("eos_token_id", self.eos_token_id, Some(eos_token_id)),
            ("pad_token_id", self.pad_token_id, pad_token_id),
        ] {
            // A target that declares no padding token has nothing to compare against.
            let Some(theirs) = theirs else { continue };
            if let Some(mine) = mine
                && mine != theirs
            {
                return Err(Error::invalid(format!(
                    "draft {name} {mine} does not match the target's {theirs}"
                )));
            }
        }
        Ok(())
    }

    /// Checks the draft's own block semantics.
    ///
    /// # Errors
    /// Rejects a configuration that declares a causal draft: the score block attends inside itself
    /// and only the target verifies causally.
    pub fn check_block_semantics(&self) -> Result<()> {
        if self.is_causal {
            return Err(Error::invalid(
                "draft blocks attend inside themselves and cannot be causal",
            ));
        }
        if self.tie_word_embeddings {
            return Err(Error::invalid(
                "draft embeddings are not tied to the target vocabulary",
            ));
        }
        Ok(())
    }

    /// The tensors the draft's package must provide and the shape geometry fixes for each.
    ///
    /// The double-tap convolution's shapes come from the reference implementation the plan cites:
    /// `base_kernel` is `[side, tap, channel]` and `kernel_projection` is
    /// `[side * taps * groups, hidden]`, with the group count taken from `conv_group_size`.
    #[must_use]
    pub fn expected_weight_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        let hidden = self.hidden_size;
        let intermediate = self.intermediate_size;
        let kv = self.num_key_value_heads * self.head_dim;
        let q = self.num_attention_heads * self.head_dim;
        // The fusion concatenates one feature per target tap; the convolution slides its own
        // kernel, and the two counts are unrelated even though both are "taps" in the plan's prose.
        let feature_taps = self.dflash_config.target_layer_ids.len();
        let rank = self.dflash_config.selector_rank;
        let groups = self.hidden_size / self.dflash_config.conv_group_size;
        let conv_taps = self.dflash_config.conv_kernel_size;
        let conv_projection = vec![2 * conv_taps * groups, hidden];
        let base_kernel = vec![2, conv_taps, hidden];
        let mut expected: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (name, shape) in [
            (
                "candidate_selector.hidden_projection.weight",
                vec![rank, hidden],
            ),
            (
                "candidate_selector.predecessor_codebook",
                vec![self.vocab_size, rank],
            ),
            (
                "candidate_selector.successor_codebook",
                vec![self.vocab_size, rank],
            ),
            ("fc.weight", vec![hidden, feature_taps * hidden]),
            ("hidden_norm.weight", vec![hidden]),
            ("norm.weight", vec![hidden]),
        ] {
            expected.insert(name.to_owned(), shape);
        }
        for layer in 0..self.num_hidden_layers {
            for (suffix, shape) in [
                ("input_layernorm.weight", vec![hidden]),
                ("post_attention_layernorm.weight", vec![hidden]),
                ("self_attn.q_proj.weight", vec![q, hidden]),
                ("self_attn.k_proj.weight", vec![kv, hidden]),
                ("self_attn.v_proj.weight", vec![kv, hidden]),
                ("self_attn.o_proj.weight", vec![hidden, q]),
                ("self_attn.q_norm.weight", vec![self.head_dim]),
                ("self_attn.k_norm.weight", vec![self.head_dim]),
                ("mlp.gate_proj.weight", vec![intermediate, hidden]),
                ("mlp.up_proj.weight", vec![intermediate, hidden]),
                ("mlp.down_proj.weight", vec![hidden, intermediate]),
                (
                    "attention_conv.kernel_projection.weight",
                    conv_projection.clone(),
                ),
                ("attention_conv.base_kernel", base_kernel.clone()),
                ("mlp_conv.kernel_projection.weight", conv_projection.clone()),
                ("mlp_conv.base_kernel", base_kernel.clone()),
            ] {
                expected.insert(format!("layers.{layer}.{suffix}"), shape);
            }
        }
        expected
    }

    /// The draft's own footprint, for the parts its geometry fixes.
    ///
    /// The plan's acceptance is that the complete quote is explainable; this covers the weights, the
    /// draft's K/V and its window, the convolution history both dynamic convolutions keep, the
    /// target features it retains, and what a rejected round leaves to discard. The arena, the
    /// captured graph and the device's own accounting still need device facts and are not quoted
    /// here.
    #[must_use]
    pub fn quote(&self, dtype_bytes: usize) -> DraftQuote {
        let attention = self.num_hidden_layers * self.num_key_value_heads * self.head_dim;
        let kv_bytes_per_token = attention * dtype_bytes * 2;
        // One window per convolution per layer: it is `kernel - 1` wide over the hidden channels,
        // and the configuration's group size is a grouping rather than a width.
        let conv_history = self.num_hidden_layers
            * 2
            * self.hidden_size
            * (self.dflash_config.conv_kernel_size - 1);
        DraftQuote {
            weights_bytes: self.weights_bytes(dtype_bytes),
            kv_bytes_per_token,
            window_kv_bytes: kv_bytes_per_token * self.sliding_window,
            conv_history_bytes_per_sequence: conv_history * dtype_bytes,
            feature_bytes_per_token: self.dflash_config.target_layer_ids.len()
                * self.hidden_size
                * dtype_bytes,
            rollback_bytes_per_round: kv_bytes_per_token * (self.dflash_config.block_size - 1),
        }
    }
    /// Every weight the draft loads, in bytes at `dtype_bytes` each.
    ///
    /// The resource quote and the inventory have to agree, so this is the same sum
    /// [`Self::quote`] reports as `weights_bytes`; the arena and the captured graph still need
    /// device facts and are not quoted here.
    #[must_use]
    pub fn weights_bytes(&self, dtype_bytes: usize) -> usize {
        self.expected_weight_shapes()
            .values()
            .map(|shape| shape.iter().product::<usize>() * dtype_bytes)
            .sum()
    }

    /// Checks a package's tensor inventory against [`Self::expected_weight_shapes`].
    ///
    /// # Errors
    /// Reports a missing tensor, a tensor the draft does not define, or a shape that disagrees with
    /// the geometry.
    pub fn check_weight_inventory(&self, actual: &BTreeMap<String, Vec<usize>>) -> Result<()> {
        let expected = self.expected_weight_shapes();
        if let Some(missing) = expected.keys().find(|name| !actual.contains_key(*name)) {
            return Err(Error::invalid(format!(
                "draft package is missing {missing}"
            )));
        }
        if let Some(stray) = actual.keys().find(|name| !expected.contains_key(*name)) {
            return Err(Error::invalid(format!(
                "draft package carries an unknown tensor: {stray}"
            )));
        }
        for (name, shape) in expected {
            let found = actual
                .get(&name)
                .ok_or_else(|| Error::invalid(format!("draft package is missing {name}")))?;
            if *found != shape {
                return Err(Error::invalid(format!(
                    "draft tensor {name} has shape {found:?}, geometry requires {shape:?}"
                )));
            }
        }
        Ok(())
    }
}

/// Geometry declared by the draft's `config.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftGeometry {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub layers: usize,
    /// Target layers the draft was trained against.
    pub target_layers: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    /// Target layers whose hidden states the draft consumes, in ascending order.
    pub target_taps: Vec<usize>,
    pub sliding_window: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
}

/// What the draft costs before weights, arena, graph and rollback are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftQuote {
    /// Every weight the draft loads.
    pub weights_bytes: usize,
    /// K and V for one token across every draft layer.
    pub kv_bytes_per_token: usize,
    /// The window the draft attention keeps per sequence.
    pub window_kv_bytes: usize,
    /// The history both dynamic convolutions keep per layer, per sequence.
    pub conv_history_bytes_per_sequence: usize,
    /// One retained target feature vector per tap, per token.
    pub feature_bytes_per_token: usize,
    /// Draft K/V a rejected round leaves to discard, in the round where only the first candidate is
    /// accepted.
    pub rollback_bytes_per_round: usize,
}

/// Which part of a draft's cost a budget cannot cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftCost {
    /// The weights the draft loads at all.
    Weights,
    /// The attention window one sequence keeps.
    Window,
    /// Key/value and retained target features for the tokens in flight.
    Cache,
    /// The draft key/value a rejected round discards.
    Rollback,
    /// The activation arena, which needs device facts.
    Arena,
    /// The captured graph, which needs device facts.
    Graph,
}

/// Whether a device can hold a draft, and - when it cannot - which parts are uncovered.
///
/// The stage-1 acceptance asks for an explainable admit-or-reject result without executing the
/// model, so this is arithmetic over the quote rather than a measurement: components are charged
/// in the order weights, window, cache, rollback, and the ones the budget does not reach are
/// named. A caller that needs a different order can read the quote directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftAdmission {
    pub budget_bytes: usize,
    /// The bytes the requested shape needs in total.
    pub required_bytes: usize,
    /// The components the budget does not cover, in the order they are charged.
    pub shortfall: Vec<DraftCost>,
    /// The costs this quote cannot state: the plan's resource row asks for the arena and the
    /// captured graph too, and both need device facts the package layer does not have. Naming them
    /// here keeps an admission from reading as a complete answer when it is a partial one.
    pub deferred: Vec<DraftCost>,
}

impl DraftAdmission {
    /// Whether the whole request fits; `shortfall` says what does not when it is false.
    #[must_use]
    pub const fn admitted(&self) -> bool {
        self.shortfall.is_empty()
    }
}

impl DraftQuote {
    /// The costs the package layer cannot quote without device facts.
    #[must_use]
    pub fn deferred() -> Vec<DraftCost> {
        vec![DraftCost::Arena, DraftCost::Graph]
    }

    /// The bytes this quote needs at its peak: weights, the windows, the cache, and the buffer a
    /// rejected round discards. The target's own state and cache are not part of it, because the
    /// target runs whether or not a draft does; the draft's arena and graph are deferred, see
    /// [`Self::deferred`]. The rollback buffer is charged even though it is released after the
    /// round, because the round cannot run without room for it.
    #[must_use]
    pub const fn required_bytes(&self, sequences: usize, tokens: usize) -> usize {
        self.weights_bytes
            + sequences * (self.window_kv_bytes + self.conv_history_bytes_per_sequence)
            + tokens * (self.kv_bytes_per_token + self.feature_bytes_per_token)
            + self.rollback_bytes_per_round
    }

    /// Admit or reject this quote against a device budget, naming what does not fit.
    #[must_use]
    pub fn admit(&self, budget_bytes: usize, sequences: usize, tokens: usize) -> DraftAdmission {
        let components = [
            (DraftCost::Weights, self.weights_bytes),
            (
                DraftCost::Window,
                sequences * (self.window_kv_bytes + self.conv_history_bytes_per_sequence),
            ),
            (
                DraftCost::Cache,
                tokens * (self.kv_bytes_per_token + self.feature_bytes_per_token),
            ),
            (DraftCost::Rollback, self.rollback_bytes_per_round),
        ];
        let mut remaining = budget_bytes;
        let mut shortfall = Vec::new();
        for (cost, bytes) in components {
            if bytes > remaining {
                shortfall.push(cost);
                remaining = 0;
            } else {
                remaining -= bytes;
            }
        }
        DraftAdmission {
            budget_bytes,
            required_bytes: self.required_bytes(sequences, tokens),
            shortfall,
            deferred: Self::deferred(),
        }
    }
}

/// The values `z-lab/Qwen3.8-27B-DFlash2` declares, named so the contract holds no bare numbers.
mod released {
    pub const HIDDEN_SIZE: usize = 5_120;
    pub const INTERMEDIATE_SIZE: usize = 17_408;
    pub const VOCAB_SIZE: usize = 248_320;
    pub const LAYERS: usize = 5;
    /// Layers of the target the checkpoint was trained against.
    pub const TARGET_LAYERS: usize = 64;
    pub const ATTENTION_HEADS: usize = 32;
    pub const KEY_VALUE_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 128;
    /// The score block the draft proposes in one step.
    pub const BLOCK_SIZE: usize = 8;
    /// Target layers whose hidden states the draft consumes.
    pub const TARGET_TAPS: [usize; 5] = [5, 19, 33, 47, 61];
    pub const SLIDING_WINDOW: usize = 2_048;
    pub const SELECTOR_RANK: usize = 256;
    pub const SELECTOR_TOP_K: usize = 16;
}

impl DraftGeometry {
    /// Whether a `query` position may attend a `key` position in the draft's score block.
    ///
    /// The released configuration is non-causal and every layer is a sliding layer, and the plan's
    /// contract is narrower than "not causal": a query sees its own block in both directions, sees
    /// the blocks before it, and never reaches further back than the sliding window. The reference
    /// resolves the same pair per layer - `causal = false`, window from the configuration - and
    /// leaves the mask itself to the attention backend, so this is this project's statement of the
    /// rule rather than a transcription.
    #[must_use]
    pub const fn attends(&self, query: usize, key: usize) -> bool {
        let same_block = query / self.block_size == key / self.block_size;
        same_block || (key <= query && query - key < self.sliding_window)
    }

    /// The block a position belongs to.
    #[must_use]
    pub const fn block_of(&self, position: usize) -> usize {
        position / self.block_size
    }

    /// The registered fixture in `examples/qwen3.8-27b-dflash2` is the same configuration, and a
    /// test asserts this constructor and the parsed file agree.
    #[must_use]
    pub fn official() -> Self {
        Self {
            hidden_size: released::HIDDEN_SIZE,
            intermediate_size: released::INTERMEDIATE_SIZE,
            vocab_size: released::VOCAB_SIZE,
            layers: released::LAYERS,
            target_layers: released::TARGET_LAYERS,
            attention_heads: released::ATTENTION_HEADS,
            key_value_heads: released::KEY_VALUE_HEADS,
            head_dim: released::HEAD_DIM,
            block_size: released::BLOCK_SIZE,
            target_taps: released::TARGET_TAPS.to_vec(),
            sliding_window: released::SLIDING_WINDOW,
            selector_rank: released::SELECTOR_RANK,
            selector_top_k: released::SELECTOR_TOP_K,
        }
    }

    /// # Errors
    /// Rejects a geometry that cannot describe a draft: an empty dimension, KV heads that do not
    /// divide the attention heads, an empty or unordered tap list, a selector that takes more
    /// candidates than it scores, or a block wider than the window it slides in.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.layers == 0
            || self.target_layers == 0
            || self.attention_heads == 0
            || self.key_value_heads == 0
            || self.head_dim == 0
            || self.block_size == 0
            || self.sliding_window == 0
            || self.selector_rank == 0
        {
            return Err(Error::invalid("draft geometry has an empty dimension"));
        }
        if !self.attention_heads.is_multiple_of(self.key_value_heads) {
            return Err(Error::invalid(
                "draft key/value heads do not divide the attention heads",
            ));
        }
        if self.selector_top_k == 0 || self.selector_top_k > self.selector_rank {
            return Err(Error::invalid(
                "draft selector takes more candidates than it scores",
            ));
        }
        if self.block_size > self.sliding_window {
            return Err(Error::invalid(
                "draft block is wider than the sliding window",
            ));
        }
        if self.target_taps.is_empty() || self.target_taps.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::invalid(
                "draft target taps must be a non-empty ascending list",
            ));
        }
        Ok(())
    }

    /// Checks the target-side acceptance conditions: a causal decoder (dense or hybrid) whose hidden
    /// geometry the taps can feed, with every tapped layer present.
    ///
    /// A tap is the residual stream after the tapped layer, whatever that layer's mixer is: the
    /// reference reads `hidden + residual` once the layer returns, so a recurrent layer feeds a tap
    /// exactly like an attention layer does.
    ///
    /// Target identity, tokenizer correspondence and embedding sharing need the package's declared
    /// metadata and a token map, so they stay with the loader that resolves them.
    ///
    /// # Errors
    /// Reports the first condition the target fails, naming the tap or the geometry that differs.
    pub fn check_target(&self, target: &ModelIr) -> Result<()> {
        if !matches!(
            target.backbone,
            BackboneKind::Decoder | BackboneKind::Hybrid
        ) {
            return Err(Error::invalid("draft requires a causal decoder target"));
        }
        if target.hidden_size != self.hidden_size {
            return Err(Error::invalid(format!(
                "draft hidden {} does not match target hidden {}",
                self.hidden_size, target.hidden_size
            )));
        }
        let layers = target.mixers.len();
        if layers != self.target_layers {
            return Err(Error::invalid(format!(
                "draft was trained against {} target layers, target has {layers}",
                self.target_layers
            )));
        }
        if let Some(tap) = self.target_taps.iter().find(|tap| **tap >= layers) {
            return Err(Error::invalid(format!(
                "draft tap {tap} has no target layer (target has {layers})"
            )));
        }
        Ok(())
    }

    /// # Errors
    /// Names the first required operation the backend does not report.
    pub fn check_operations(&self, supported: &BTreeSet<Operation>) -> Result<()> {
        for operation in REQUIRED_OPERATIONS {
            if !supported.contains(&operation) {
                return Err(Error::invalid(format!(
                    "draft requires an unsupported operation: {operation:?}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/providers_dflash2.rs"]
mod tests;
