//! Vision-modality execution: provider-declared geometry, bound weights, captured programs.
#[cfg(test)]
#[path = "../../tests/unit/vision_attention_bench_check.rs"]
pub(crate) mod attention_bench;
#[cfg(test)]
#[path = "../../tests/unit/vision_attention_check.rs"]
mod attention_check;
mod block_mlp;
mod device_merger;
mod device_tower;
mod footprint;
pub(crate) mod kernels;
#[cfg(test)]
#[path = "../../tests/unit/vision_padding.rs"]
mod padding_check;
pub(crate) mod program;
mod session;
mod weights;
pub(crate) use footprint::footprint;
pub use program::{
    AttentionMode, add, attention, attention_with, block, gelu, layernorm, merger, mlp_width,
    patch_embed, patch_width, project, project_with, tower,
};
pub use session::{
    MediaPrefill, argmax, decode, generate, media_positions, placeholder_tokens, prefill,
    require_placements,
};
pub use weights::{VisionWeights, matrix};
