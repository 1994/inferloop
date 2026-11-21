//! Qwen GGUF adapters: the `qwen3vl` and `qwen35` text architectures and the `qwen3vl_merger`
//! mmproj projector.
mod qwen35;
mod qwen3vl;
mod vision;

pub use qwen3vl::QWEN3VL;
pub use qwen35::QWEN35;
pub use vision::QWEN3VL_MERGER;
