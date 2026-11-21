//! The GGUF platform and its adapters, through the public API only.
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "fixture helpers build synthetic metadata and files that must be well formed"
)]

mod dflash2;
mod package;
mod platform;
mod qwen35;
mod qwen3vl;
mod support;
mod tokenizer;
