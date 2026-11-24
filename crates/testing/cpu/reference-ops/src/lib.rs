//! Scalar reference implementations of the physical operations.
//!
//! A device kernel is compared against a computation that shares nothing with it but the operation
//! and the state layout. This crate is that reference: it executes no model, and the CUDA checks
//! compare against it rather than against a backend.

mod kernels;

pub use kernels::{execute as reference_operation, sigmoid};

mod moe;
