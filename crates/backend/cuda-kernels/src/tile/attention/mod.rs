//! SDPA, decode and prefill attention.
mod attention_kernels;
mod attention_model_kernels;
mod attn_decode_kernels;
mod pooled_device_kernels;
mod resident_attention_decode_kernels;
mod resident_attention_prefill_kernels;

pub use attention_kernels::sdpa;
pub use attention_model_kernels::attention;
pub use attn_decode_kernels::kernel as attn_decode;
pub use pooled_device_kernels::kernels as pooled_device;
pub use resident_attention_decode_kernels::kernels as resident_attention_decode;
pub use resident_attention_prefill_kernels::kernels as resident_attention_prefill;
