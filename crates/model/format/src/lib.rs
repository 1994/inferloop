//! Model container formats.
//!
//! Each format is a module; callers use the typed root exports for the format they opened.
pub mod gguf;

pub use gguf::{
    GgufFile, GgufLimits, GgufType, GroupedSpan, Metadata, MetadataValue, TensorInfo, decode_span,
    dequantize, half_to_f32, span_bytes, stored_type,
};
