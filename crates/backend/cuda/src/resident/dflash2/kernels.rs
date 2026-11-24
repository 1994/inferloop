//! `DFlash2` draft kernels: everything the draft does that the target graph does not.

pub use infer_cuda_kernels::tile::speculation::dflash2 as draft;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests;
