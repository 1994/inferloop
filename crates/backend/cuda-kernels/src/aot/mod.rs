//! AOT source: a driver-only runtime for precompiled SASS.
//!
//! The source builds without a CUDA toolkit: every symbol is resolved from `libcuda.so.1` at
//! runtime, so the crate has no headers, no `cudart` and no `tileiras` dependency. Artifacts are
//! loaded as cubins and launched through the driver.
pub mod artifact;
mod driver;
mod projection;

pub use artifact::{Entry, Manifest};
pub use driver::{Buffer, Device, Function, Graph, Module};
pub use projection::AotProjection;
