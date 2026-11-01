//! cuTile kernels, one directory per operator family.
//!
//! The `aot` source mirrors these family names so an implementation can be compared side by side.
use crate::contract::Kernels;
use crate::contract::buffer::Buffer;
use cuda_core::DType;
pub use cutile::prelude::Scope;
use cutile::prelude::Tensor;
use infer_core::{Error, ErrorCode, Result};
use projection::TileProjection;
use rope::TileRope;
use std::fmt::Display;

pub mod attention;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod projection;
pub mod recurrent;
pub mod rope;
pub mod sampling;
pub mod speculation;
pub mod state;
pub mod vision;

/// The cuTile kernels of one device.
#[must_use]
pub fn kernels(ordinal: usize) -> Kernels {
    Kernels {
        projection: Box::new(TileProjection::new(ordinal)),
        rope: Box::new(TileRope::new(ordinal)),
    }
}

/// Wrap a host buffer in a borrowed tensor for one recording; the graph keeps only the address.
#[expect(
    unsafe_code,
    reason = "Buffer::new already holds the host to what a borrowed tensor needs"
)]
fn borrowed<T: DType>(buffer: Buffer<'_, T>, ordinal: usize, shape: &[usize]) -> Result<Tensor<T>> {
    let dimension = |value: &usize| {
        i32::try_from(*value).map_err(|_| Error::invalid("kernel operand exceeds i32"))
    };
    let shape: Vec<i32> = shape.iter().map(dimension).collect::<Result<_>>()?;
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    // SAFETY: `Buffer::new` guarantees a contiguous allocation outliving every replay, and the
    // caller checked the plan's shape covers exactly its elements. The tensor frees nothing.
    Ok(unsafe { Tensor::<T>::borrow_raw_parts(buffer.address(), ordinal, shape, strides) })
}

fn backend_error(error: impl Display) -> Error {
    Error::new(ErrorCode::Backend, error.to_string())
}
