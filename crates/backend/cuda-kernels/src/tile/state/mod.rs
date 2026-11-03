//! Auxiliary state updates and resident metadata.
mod aux_kernels;
mod contract;
mod metadata_kernels;

pub use aux_kernels::aux;
pub use metadata_kernels::{kernels as metadata, load_rotary};

pub use contract::TileState;
