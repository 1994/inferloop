//! Gated and quantized norm reductions.
mod contract;
mod norm_fp4_kernels;
mod norm_maximum_kernels;

pub use norm_fp4_kernels::kernels as norm_fp4;
pub use norm_maximum_kernels::kernels as norm_maximum;

pub use contract::TileNorm;
