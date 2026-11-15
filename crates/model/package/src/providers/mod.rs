//! Architecture registration and model-specific execution customization.
pub mod dflash2;
#[cfg(feature = "gguf")]
pub mod gguf;
pub mod llama;
pub mod qwen;
pub mod registry;
