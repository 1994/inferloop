//! Bounded GGUF container parsing and k-quant reference decoding.
//!
//! This crate owns the GGUF format; it does not know about model graphs, schedulers or device
//! kernels. A caller reads typed metadata, tensor geometry and stored bytes, then decides how to
//! present them (for example as a virtual HF package). Unknown or unsupported tensor types fail
//! explicitly instead of being silently reinterpreted.
mod container;
mod device;
mod metadata;
mod quant;
mod tensor;

pub use container::{GgufFile, GgufLimits, TensorInfo};
pub use device::{Q4K_TILE_ROWS, Q4KDevice, Q4KTiled};
pub use metadata::{Metadata, MetadataValue};
pub use quant::dequantize;
pub use tensor::GgufType;
