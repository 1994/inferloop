//! AOT profile: a driver-only runtime for precompiled SASS.
//!
//! The profile builds without a CUDA toolkit: every symbol is resolved from `libcuda.so.1` at
//! runtime, so the crate has no headers, no `cudart` and no `tileiras` dependency. Artifacts are
//! loaded as cubins and launched through the driver.
pub mod artifact;
mod driver;

pub use artifact::{Entry, Manifest};
pub use driver::{Buffer, Device, Function, Module};
