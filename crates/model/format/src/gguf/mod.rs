//! Bounded GGUF container parsing and k-quant reference decoding.
//!
//! This module owns the GGUF format; it does not know about model graphs, schedulers or device
//! kernels. A caller reads typed metadata, tensor geometry and stored bytes, then decides how to
//! present them (for example as a virtual HF package). Unknown or unsupported tensor types fail
//! explicitly instead of being silently reinterpreted.
pub mod container;
pub mod grouped;
pub mod metadata;
pub mod quant;
pub mod tensor;

pub use container::{GgufFile, GgufLimits, TensorInfo};
pub use grouped::{GroupedSpan, decode_span, span_bytes, stored_type};
pub use metadata::{Metadata, MetadataValue};
pub use quant::{dequantize, half_to_f32};
pub use tensor::GgufType;
