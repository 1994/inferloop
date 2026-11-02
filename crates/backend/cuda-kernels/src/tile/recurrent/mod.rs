//! Gated delta, convolution and their pooled verification.
mod contract;
mod conv_prefill_kernels;
mod recurrent_checkpoint_kernels;
mod recurrent_chunk_kernels;
mod recurrent_kernels;
mod recurrent_prefill_kernels;
mod slot_convolution_kernels;
mod slot_recurrent_kernels;

pub use contract::{SCAN_COLUMNS, TileRecurrent, WIDE_SCAN_COLUMNS, WIDE_SCAN_VALUE_HEADS};
pub use conv_prefill_kernels::kernels as conv_prefill;
pub use recurrent_checkpoint_kernels::checkpoint as recurrent_checkpoint;
pub use recurrent_chunk_kernels::kernels as recurrent_chunk;
pub use recurrent_kernels::recurrent;
pub use recurrent_prefill_kernels::kernels as recurrent_prefill;
pub use slot_convolution_kernels::kernels as slot_convolution;
pub use slot_recurrent_kernels::fused as slot_recurrent;
