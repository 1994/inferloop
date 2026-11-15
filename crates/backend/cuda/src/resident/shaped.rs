//! Device memory with the dimensions host code reads it as.
use super::buffers::Operand;
use crate::RecurrentProfile;
use crate::device::CudaDevice;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{Buffer, Recording, StateBuffer, Width};
use infer_cuda_kernels::runtime::{Allocation, DType, Element, HeadMajor, Transfers, bf16};
use std::ops::Range;

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

/// One state tensor at its storage type. Gated-delta states follow the program's
/// [`RecurrentProfile`]; convolution and KV states are always F32. A clone shares the memory.
#[derive(Clone)]
pub enum StateTensor {
    F32(Shaped<f32>),
    Bf16(Shaped<bf16>),
}

impl StateTensor {
    /// Zeroed memory of `dims` stored as `profile` selects.
    /// # Errors
    /// Reports a size overflow or a failed allocation.
    pub fn zeros(device: &CudaDevice, dims: Vec<usize>, profile: RecurrentProfile) -> Result<Self> {
        Ok(match profile {
            RecurrentProfile::F32 => Self::F32(Shaped::zeros(device, dims)?),
            RecurrentProfile::Bf16 => Self::Bf16(Shaped::zeros(device, dims)?),
        })
    }

    /// Zeroed memory of the same dimensions and storage.
    /// # Errors
    /// Reports a failed allocation.
    pub fn zeros_like(&self, device: &CudaDevice) -> Result<Self> {
        Self::zeros(device, self.dims().to_vec(), self.profile())
    }

    /// The storage this tensor holds.
    #[must_use]
    pub const fn profile(&self) -> RecurrentProfile {
        match self {
            Self::F32(_) => RecurrentProfile::F32,
            Self::Bf16(_) => RecurrentProfile::Bf16,
        }
    }

    #[must_use]
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::F32(tensor) => tensor.dims(),
            Self::Bf16(tensor) => tensor.dims(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(tensor) => tensor.len(),
            Self::Bf16(tensor) => tensor.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device bytes of the stored elements.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        (self.len() * self.profile().element_bytes()) as u64
    }

    /// The tensor as F32, which every state but a gated-delta one is.
    /// # Errors
    /// Rejects a BF16 state, which only the recurrence kernels read.
    pub fn f32(&self) -> Result<&Shaped<f32>> {
        match self {
            Self::F32(tensor) => Ok(tensor),
            Self::Bf16(_) => Err(Error::invariant("an F32 state is stored in BF16")),
        }
    }

    /// The buffer a recurrence binds, at its storage type.
    #[must_use]
    pub fn buffer(&self) -> StateBuffer<'_> {
        match self {
            Self::F32(tensor) => tensor.memory().operand().into(),
            Self::Bf16(tensor) => tensor.memory().operand().into(),
        }
    }

    /// The same memory read as `dims`.
    /// # Errors
    /// Rejects dimensions that do not cover the memory exactly.
    pub fn reshaped(&self, dims: Vec<usize>) -> Result<Self> {
        Ok(match self {
            Self::F32(tensor) => Self::F32(tensor.reshaped(dims)?),
            Self::Bf16(tensor) => Self::Bf16(tensor.reshaped(dims)?),
        })
    }

    /// The memory read as F32 words, for work that only moves or clears bits; a BF16 state
    /// reads two elements per word.
    /// # Errors
    /// Rejects a BF16 state of odd length.
    pub fn words(&self) -> Result<Allocation<f32>> {
        match self {
            Self::F32(tensor) => Ok(tensor.memory().clone()),
            Self::Bf16(tensor) => tensor.memory().reinterpret(),
        }
    }

    /// Elements `range` of the memory read as `dims`; the view holds the allocation.
    /// # Errors
    /// Rejects a range past the memory or dimensions that do not cover it exactly.
    pub fn view(&self, range: Range<usize>, dims: Vec<usize>) -> Result<Self> {
        Ok(match self {
            Self::F32(tensor) => Self::F32(Shaped::new(tensor.memory().slice(range)?, dims)?),
            Self::Bf16(tensor) => Self::Bf16(Shaped::new(tensor.memory().slice(range)?, dims)?),
        })
    }

    /// Whether `other` starts at the same device address.
    #[must_use]
    pub fn same_memory(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F32(tensor), Self::F32(other)) => tensor.same_memory(other),
            (Self::Bf16(tensor), Self::Bf16(other)) => tensor.same_memory(other),
            _ => false,
        }
    }

    /// Copy all of `source`, of the same dimensions and storage, on the device stream.
    /// # Errors
    /// Rejects other dimensions or storage and a copy the driver does not enqueue.
    pub fn copy_from(&self, device: &CudaDevice, source: &Self) -> Result<()> {
        match (self, source) {
            (Self::F32(tensor), Self::F32(source)) => tensor.copy_from(device, source),
            (Self::Bf16(tensor), Self::Bf16(source)) => tensor.copy_from(device, source),
            _ => Err(Error::invariant("state copy storage")),
        }
    }
}

impl StateTensor {
    /// Record a copy of `source`, of the same storage and length, into this tensor.
    /// # Errors
    /// Rejects another storage or length and a copy the recording rejects.
    pub fn record_copy(&self, recording: &Recording<'_>, source: &Self) -> Result<()> {
        match (self, source) {
            (Self::F32(tensor), Self::F32(source)) => {
                recording.copy(tensor.memory().operand(), source.memory().operand())
            }
            (Self::Bf16(tensor), Self::Bf16(source)) => {
                recording.copy(tensor.memory().operand(), source.memory().operand())
            }
            _ => Err(Error::invariant("state copy storage")),
        }
    }

    /// Enqueue a copy of `source`, of the same storage and length, into this tensor.
    /// # Errors
    /// Rejects another storage or length and a copy the driver does not enqueue.
    pub fn copy_in(&self, transfers: &Transfers<'_>, source: &Self) -> Result<()> {
        match (self, source) {
            (Self::F32(tensor), Self::F32(source)) => {
                transfers.copy(tensor.memory().operand(), source.memory().operand())
            }
            (Self::Bf16(tensor), Self::Bf16(source)) => {
                transfers.copy(tensor.memory().operand(), source.memory().operand())
            }
            _ => Err(Error::invariant("state copy storage")),
        }
    }
}

impl From<Shaped<f32>> for StateTensor {
    fn from(tensor: Shaped<f32>) -> Self {
        Self::F32(tensor)
    }
}

/// The F32 view every state but a gated-delta one has; a BF16 state has none, so binding one
/// where F32 is expected fails instead of reading its memory as F32.
impl Operand for StateTensor {
    type Marker = f32;

    fn operand(&self) -> Result<Buffer<'_, f32>> {
        self.f32().map(|tensor| tensor.memory().operand())
    }
}
