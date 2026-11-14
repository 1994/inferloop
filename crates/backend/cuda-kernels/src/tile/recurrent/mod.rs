//! Gated delta, convolution and their pooled verification.

/// Bind a delta state at its storage type and run `$body` once per type the contract admits, with
/// `$tensor` the bound tensor and `dtype` the kernel's leading element generic. `$bind` binds the
/// state buffer `$buffer` and is expanded at each type.
macro_rules! with_state {
    ($state:expr, |$buffer:ident| $bind:expr, |$tensor:pat_param, $dtype:ident| $body:block) => {
        match $state {
            $crate::contract::kernels::StateBuffer::F32($buffer) => {
                let $tensor = $bind?;
                let $dtype = String::from(<f32 as cuda_core::DType>::DTYPE.as_str());
                $body
            }
            $crate::contract::kernels::StateBuffer::Bf16(buffer) => {
                let $buffer = $crate::tile::projection::retyped_bf16(buffer);
                let $tensor = $bind?;
                let $dtype = String::from(<cutile::half::bf16 as cuda_core::DType>::DTYPE.as_str());
                $body
            }
        }
    };
}

mod contract;
mod conv_prefill_kernels;
mod pool;
mod recurrent_checkpoint_kernels;
mod recurrent_chunk_kernels;
mod recurrent_kernels;
mod recurrent_prefill_kernels;
mod simt;
mod slot_convolution_kernels;
mod slot_recurrent_kernels;

pub use contract::{SCAN_COLUMNS, TileRecurrent, WIDE_SCAN_COLUMNS, WIDE_SCAN_VALUE_HEADS};
pub use conv_prefill_kernels::kernels as conv_prefill;
pub use pool::{TileRecurrentPool, pack_parameters};
pub use recurrent_checkpoint_kernels::checkpoint as recurrent_checkpoint;
pub use recurrent_chunk_kernels::kernels as recurrent_chunk;
pub use recurrent_kernels::recurrent;
pub use recurrent_prefill_kernels::kernels as recurrent_prefill;
pub use slot_convolution_kernels::kernels as slot_convolution;
pub use slot_recurrent_kernels::fused as slot_recurrent;
