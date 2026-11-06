//! Device memory with the dimensions host code reads it as.
use super::buffers::Operand;
use crate::device::CudaDevice;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{Buffer, Width};
use infer_cuda_kernels::tile::runtime::{Allocation, Element, HeadMajor};

/// An allocation and its logical dimensions. State copies and checks read the dimensions; kernels
/// see only the buffer. A clone shares the memory, and so does a reshape.
#[derive(Clone)]
pub struct Shaped<T: DType> {
    memory: Allocation<T>,
    dims: Vec<usize>,
}

impl<T: DType> Shaped<T> {
    /// `memory` read as `dims`.
    /// # Errors
    /// Rejects dimensions that do not cover the memory exactly.
    pub fn new(memory: Allocation<T>, dims: Vec<usize>) -> Result<Self> {
        if dims
            .iter()
            .try_fold(1usize, |all, &dim| all.checked_mul(dim))
            != Some(memory.len())
        {
            return Err(Error::invalid(format!(
                "dimensions {dims:?} do not cover {} elements",
                memory.len()
            )));
        }
        Ok(Self { memory, dims })
    }

    /// Zeroed memory of `dims`.
    /// # Errors
    /// Reports a size overflow or a failed allocation.
    pub fn zeros(device: &CudaDevice, dims: Vec<usize>) -> Result<Self> {
        let len = dims
            .iter()
            .try_fold(1usize, |all, &dim| all.checked_mul(dim))
            .ok_or_else(|| Error::invalid("state size overflow"))?;
        Self::new(device.runtime().zeros(len)?, dims)
    }

    /// The same memory read as `dims`.
    /// # Errors
    /// Rejects dimensions that do not cover the memory exactly.
    pub fn reshaped(&self, dims: Vec<usize>) -> Result<Self> {
        Self::new(self.memory.clone(), dims)
    }

    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// The memory behind these dimensions.
    #[must_use]
    pub const fn memory(&self) -> &Allocation<T> {
        &self.memory
    }

    /// Whether `other` starts at the same device address.
    #[must_use]
    pub fn same_memory(&self, other: &Self) -> bool {
        self.memory.buffer().address() == other.memory.buffer().address()
    }
}

impl<T: DType + Element<Marker: Width>> Shaped<T> {
    /// Copy all of `source`, of the same dimensions, on the device stream. Copying memory onto
    /// itself moves nothing.
    /// # Errors
    /// Rejects other dimensions and a copy the driver does not enqueue.
    pub fn copy_from(&self, device: &CudaDevice, source: &Self) -> Result<()> {
        if self.dims != source.dims {
            return Err(Error::invariant("state copy shape"));
        }
        if self.same_memory(source) {
            return Ok(());
        }
        device.copy_d2d_at(self, source, 0, 0, source.len())
    }

    /// Copy the first `rows` positions of every head of `source`; both are head-major
    /// `[heads, positions, dim]` and may hold different position counts. Copying memory onto
    /// itself moves nothing.
    /// # Errors
    /// Rejects another rank, head count or dimension, a prefix past either side, and a copy the
    /// driver does not enqueue.
    pub fn copy_head_prefix(&self, device: &CudaDevice, source: &Self, rows: usize) -> Result<()> {
        let (&[heads, to, dim], &[from_heads, from, from_dim]) = (&self.dims[..], &source.dims[..])
        else {
            return Err(Error::invariant("head prefix copy rank"));
        };
        if heads != from_heads || dim != from_dim || rows > to || rows > from {
            return Err(Error::invariant("head prefix copy layout or range"));
        }
        if self.same_memory(source) {
            return Ok(());
        }
        let dst = HeadMajor::new(self.memory.operand(), [heads, to, dim])?;
        let src = HeadMajor::new(source.memory.operand(), [heads, from, dim])?;
        device
            .runtime()
            .enqueue(|transfers| transfers.copy_heads(dst, src, rows))
    }
}

impl<T: DType + Element> Operand for Shaped<T> {
    type Marker = T::Marker;

    fn operand(&self) -> Result<Buffer<'_, T::Marker>> {
        Ok(self.memory.operand())
    }
}
