//! cuTile kernels now live in `infer-cuda-kernels`.
//!
//! The module keeps the host-side path (`crate::kernels::linear`) stable while the kernel bodies
//! move out of this crate.
pub use infer_cuda_kernels::tile::linear;
