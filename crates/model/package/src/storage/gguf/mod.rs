//! The GGUF platform: what every architecture shares, so one adapter is all a family adds.
//!
//! The platform owns the container, the tokenizer, the llama.cpp names every architecture shares,
//! the split of trailing `nextn` blocks into a speculative draft head, value transforms, the
//! mmproj patch slices and the rule that every stored tensor maps somewhere. An adapter
//! ([`GgufArchitecture`], [`GgufProjector`]) declares only what its architecture adds: the HF
//! configuration it synthesizes, its own tensor names and the inverse of what its converter
//! changed. Adapters are registered with the [`ModelRegistry`](crate::ModelRegistry), never named
//! here. The result is an ordinary HF configuration and canonical tensor names, so providers,
//! recipes and backends are the same for every checkpoint format.
mod arch;
mod package;
mod requirements;
mod tensor;
mod tokenizer;
mod vision;

pub use arch::{
    Block, GgufArchitecture, GgufProjector, LayerPlan, bool_array, common_block_field, count,
    token_id, trim_trailing_zeros, u64_array,
};
pub use package::{GgufPackage, GgufSlot};
pub use requirements::{PackageRequirements, encoding_for, gguf_requirements};
pub use tensor::{
    ExpectedTensors, MappedTensor, TensorSpec, ValueTransform, VisionTensor, is_float,
};
pub use tokenizer::{tokenizer_config, tokenizer_json};
pub use vision::{GgufVisionPackage, VisionSlot};
