//! CUDA kernel implementations for `infer-backend-cuda`.
//!
//! The host owns device allocation, streams, graphs, KV and tuning; this crate owns kernel bodies.
//! Profiles are mutually exclusive and selected at build time:
//!
//! - `tile`: cuTile kernels compiled by the CUDA 13.x toolchain;
//! - `aot`: precompiled SASS loaded through the driver, with no CUDA toolkit at build time.
//!
//! The contract types (`Buffer`, `Device`, `Plan`) move here next. This first slice only moves the
//! linear projection kernels so the crate boundary can be reviewed on its own.

#[cfg(all(feature = "tile", feature = "aot"))]
compile_error!("tile and aot are mutually exclusive kernel profiles");

#[cfg(feature = "tile")]
pub mod tile;
