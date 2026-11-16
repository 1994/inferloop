//! Package metadata, weight bindings and bounded loading.
#[cfg(feature = "gguf")]
pub mod gguf_package;
#[cfg(feature = "gguf")]
pub mod gguf_requirements;
#[cfg(feature = "gguf")]
pub mod gguf_vision;
#[cfg(feature = "gguf")]
pub mod grouped_int;
#[cfg(feature = "gguf")]
pub mod grouped_int6;
#[cfg(feature = "gguf")]
pub mod grouped_int_tiled;
pub mod index;
pub mod loader;
pub mod memory;
pub mod package;
pub mod quantized;
pub mod safetensors;
#[cfg(test)]
#[path = "../../tests/unit/storage_shards_tests.rs"]
pub mod shards_tests;
pub mod source;
