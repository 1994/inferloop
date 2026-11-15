//! GGUF package adaptation.
//!
//! The module is feature-gated: `infer-gguf` parses the container and this layer maps model
//! metadata to the existing HF-shaped provider configuration.
pub mod qwen3vl;
pub mod tokenizer;
pub mod validate;
