//! CUDA kernel implementations for `infer-backend-cuda`.
//!
//! The host owns device allocation, streams, graphs, KV and tuning; this crate owns kernel bodies.
//! `contract` holds what the host may see, one trait per operator family. Kernels come from two
//! sources that can be built together:
//!
//! - `tile`: cuTile kernels compiled by the CUDA 13.x toolchain; cuTile only exists on Linux, so
//!   the gate mirrors the host's `cuda_backend` and `--features cuda` keeps building elsewhere;
//! - `aot`: precompiled SASS loaded through the driver, with no CUDA toolkit at build time.
//!
//! Both use the same family names, so a new device implements a family by following its
//! counterpart.

pub mod contract;

#[cfg(feature = "aot")]
pub mod aot;

#[cfg(all(feature = "tile", target_os = "linux"))]
pub mod tile;
