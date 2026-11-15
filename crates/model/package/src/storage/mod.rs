//! Package metadata, weight bindings and bounded loading.
pub mod index;
pub mod loader;
pub mod memory;
pub mod package;
pub mod quantized;
pub mod safetensors;
#[cfg(test)]
#[path = "../../tests/unit/storage_shards_tests.rs"]
pub mod shards_tests;
#[cfg(feature = "gguf")]
pub mod grouped_int;
pub mod source;
