//! Contract buffers over tensors a resident program owns.
#![expect(
    unsafe_code,
    reason = "The one place host tensors become contract buffers, so kernel calls stay safe"
)]
use super::arena::SlotMemory;
use super::capture::error;
use cutile::prelude::*;
use infer_cuda_kernels::contract::{Buffer, KvCache, KvElement};
use infer_cuda_kernels::tile::runtime::{Allocation, Element};
use std::sync::Arc;

/// Device memory a launch can take as an operand, while the host still holds two kinds of it.
pub trait Operand {
    /// The contract's element type.
    type Marker;

    /// The memory as a contract buffer.
    /// # Errors
    /// Rejects storage a buffer cannot describe.
    fn operand(&self) -> Result<Buffer<'_, Self::Marker>, DeviceError>;
}

/// For tests that launch over cuTile tensors; production memory is an [`Allocation`], whose
/// buffer the recording graph retains. A buffer over a tensor does not make the graph hold its
/// memory, so the test must own the tensor for as long as the graph lives, which is what
/// [`Buffer::new`] requires. It stays here rather than in the test support modules because the
/// policy keeps `unsafe` out of test files.
#[cfg(test)]
impl<T: DType + Element> Operand for Tensor<T> {
    type Marker = T::Marker;

    fn operand(&self) -> Result<Buffer<'_, T::Marker>, DeviceError> {
        if !self.is_contiguous() {
            return Err(error("kernel operand must be contiguous"));
        }
        // SAFETY: see above; the arena hands distinct slots to a launch's inputs and outputs.
        Ok(unsafe { Buffer::new(self.device_pointer().cu_deviceptr(), self.size()) })
    }
}

impl<T: DType + Element> Operand for Allocation<T> {
    type Marker = T::Marker;

    fn operand(&self) -> Result<Buffer<'_, T::Marker>, DeviceError> {
        Ok(Self::operand(self))
    }
}

impl Operand for SlotMemory {
    type Marker = f32;

    fn operand(&self) -> Result<Buffer<'_, f32>, DeviceError> {
        self.buffer().map_err(error)
    }
}

impl<O: Operand> Operand for Arc<O> {
    type Marker = O::Marker;

    fn operand(&self) -> Result<Buffer<'_, O::Marker>, DeviceError> {
        O::operand(self)
    }
}

/// A buffer over `memory`; see [`Operand`] for who keeps it alive.
/// # Errors
/// Rejects storage a buffer cannot describe.
pub fn buffer<O: Operand + ?Sized>(memory: &O) -> Result<Buffer<'_, O::Marker>, DeviceError> {
    memory.operand()
}

/// A row-major `[rows, columns]` operand: a buffer and the matrix shape a launch reads it as.
#[derive(Clone, Copy)]
pub struct RowMajor<'a, T> {
    pub buffer: Buffer<'a, T>,
    pub rows: usize,
    pub columns: usize,
}

impl<'a, T> RowMajor<'a, T> {
    /// `buffer` read as `rows` rows of `columns`.
    /// # Errors
    /// Rejects a shape that does not cover the buffer exactly.
    pub fn new(buffer: Buffer<'a, T>, rows: usize, columns: usize) -> Result<Self, DeviceError> {
        if rows.checked_mul(columns) != Some(buffer.len()) {
            return Err(error("operand rows do not cover the buffer"));
        }
        Ok(Self {
            buffer,
            rows,
            columns,
        })
    }
}

/// A KV pair as a launch addresses it, in `groups` of `[rows, dim]`.
pub fn kv_cache<'a, O: Operand<Marker: KvElement> + ?Sized>(
    (keys, values): (&'a O, &'a O),
    groups: usize,
    dim: usize,
    scales: [f32; 2],
) -> Result<KvCache<'a>, DeviceError> {
    Ok(KvCache {
        buffers: O::Marker::kv(buffer(keys)?, buffer(values)?),
        groups,
        dim,
        scales,
    })
}

/// A buffer over part of a program-owned allocation, which a view does not expose by address.
pub(super) const fn region<T>(address: u64, len: usize) -> Buffer<'static, T> {
    // SAFETY: as for [`buffer`]; callers derive both from a contiguous program-owned tensor.
    unsafe { Buffer::new(address, len) }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_runtime_replay.rs"]
mod runtime_tests;

#[cfg(test)]
#[path = "../../tests/unit/buffer_tensors.rs"]
mod tensors;
#[cfg(test)]
pub use tensors::tensor_rows;
