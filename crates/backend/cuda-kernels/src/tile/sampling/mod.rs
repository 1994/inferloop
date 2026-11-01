//! Device-side token selection.
mod contract;
mod greedy_kernels;

pub use greedy_kernels::kernels as greedy;

pub use contract::TileSampling;
