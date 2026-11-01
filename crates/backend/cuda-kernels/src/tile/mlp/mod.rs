//! Fused MLP and PDL edges.
mod contract;
mod mlp_kernels;
mod pdl_consumers_kernels;
mod pdl_kernels;

pub use mlp_kernels::fused;
pub use pdl_consumers_kernels::consumers as pdl_consumers;
pub use pdl_kernels::producer as pdl;

pub use contract::TileMlp;
