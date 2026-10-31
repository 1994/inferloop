//! CUDA kernel implementations for `infer-backend-cuda`.
//!
//! The host owns device allocation, streams, graphs, KV and tuning; this crate owns kernel bodies.
//! Profiles are mutually exclusive and selected at build time:
//!
//! - `tile`: cuTile kernels compiled by the CUDA 13.x toolchain;
//! - `aot`: precompiled SASS loaded through the driver, with no CUDA toolkit at build time.
//!
//! `contract` holds what the host may see. Both profiles use the same operator-family names, so a
//! new device implements a family by following its counterpart. cuTile only exists on Linux; the
//! gate mirrors the host's `cuda_backend` so `--features cuda` keeps building elsewhere.

#[cfg(all(feature = "tile", feature = "aot"))]
compile_error!("tile and aot are mutually exclusive kernel profiles");

pub mod contract;

#[cfg(feature = "aot")]
pub mod aot;

#[cfg(all(feature = "tile", target_os = "linux"))]
pub mod tile;
