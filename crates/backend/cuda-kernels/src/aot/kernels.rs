//! The AOT kernel families: every method reports it is not provided until its family ships a
//! precompiled artifact, which the contract's defaults do.
use crate::contract::{
    Attention, Draft, Kernels, Mlp, Moe, Norm, Projection, Recurrent, RecurrentPool, Rope,
    Sampling, State, Vision,
};

/// A family whose methods all report they are not provided.
struct Unavailable;

impl Vision for Unavailable {}
impl Draft for Unavailable {}
impl Attention for Unavailable {}
impl Projection for Unavailable {}
impl Rope for Unavailable {}
impl Norm for Unavailable {}
impl Mlp for Unavailable {}
impl State for Unavailable {}
impl Moe for Unavailable {}
impl Sampling for Unavailable {}
impl Recurrent for Unavailable {}
impl RecurrentPool for Unavailable {}

/// The AOT kernels of one device.
#[must_use]
pub fn kernels(_ordinal: usize) -> Kernels {
    Kernels {
        vision: Box::new(Unavailable),
        draft: Box::new(Unavailable),
        attention: Box::new(Unavailable),
        projection: Box::new(Unavailable),
        rope: Box::new(Unavailable),
        norm: Box::new(Unavailable),
        mlp: Box::new(Unavailable),
        state: Box::new(Unavailable),
        recurrent: Box::new(Unavailable),
        recurrent_pool: Box::new(Unavailable),
        moe: Box::new(Unavailable),
        sampling: Box::new(Unavailable),
    }
}
