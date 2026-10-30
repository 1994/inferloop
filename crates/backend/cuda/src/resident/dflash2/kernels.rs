//! `DFlash2` draft kernels: everything the draft does that the target graph does not.

pub use infer_cuda_kernels::tile::dflash2 as draft;

#[cfg(test)]
#[path = "../../../tests/unit/resident_dflash2_kernels.rs"]
mod tests;
