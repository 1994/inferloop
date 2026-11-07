//! Storage strategy shared by graph capture and model-specific binding adapters.
use crate::constants::FP8_BLOCK_COLUMNS;
use crate::constants::NVFP4_GROUP_SIZE;
use crate::device::CudaDevice;
use crate::device::device_error;
use crate::resident::buffers::Operand;
use crate::resident::tensor_buffer as buffer;
use crate::strategy::LinearTiling;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{Buffer, DownPlan, Kernels, Owner, Recording, WeightRef};
use infer_cuda_kernels::runtime::DType;
use infer_cuda_kernels::runtime::bf16;
use infer_cuda_kernels::runtime::{Allocation, Element, Watch};
use infer_cuda_kernels::runtime::{f4e2m1fnx2, f8e4m3fn};
use std::sync::Arc;

/// A row-major `[rows, columns]` matrix in device memory; a graph that records it keeps it alive.
pub struct Matrix<T: DType> {
    memory: Allocation<T>,
    rows: usize,
    columns: usize,
}

impl<T: DType> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
            rows: self.rows,
            columns: self.columns,
        }
    }
}

impl<T: DType> Matrix<T> {
    /// A copy of the row-major `values`, complete when this returns.
    /// # Errors
    /// Rejects a `shape` that is not a matrix covering `values`, or a failed allocation or
    /// transfer.
    pub fn upload(device: &CudaDevice, values: Vec<T>, shape: &[usize]) -> Result<Self> {
        let &[rows, columns] = shape else {
            return Err(Error::invalid("weight matrix rank"));
        };
        if rows.checked_mul(columns) != Some(values.len()) {
            return Err(Error::invalid("weight matrix shape"));
        }
        Ok(Self {
            memory: device.runtime().upload(values)?,
            rows,
            columns,
        })
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    #[must_use]
    pub const fn shape(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    /// Elements, which for packed FP4 count byte pairs of codes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    #[must_use]
    pub const fn allocation(&self) -> &Allocation<T> {
        &self.memory
    }

    /// Identifies the weight's memory while it lives; see [`Allocation::identity`].
    #[must_use]
    pub fn identity(&self) -> usize {
        self.memory.identity()
    }

    /// A handle on the weight's memory that does not keep it alive.
    #[must_use]
    pub fn watch(&self) -> Watch {
        self.memory.watch()
    }
}

impl<T: DType + Element> Operand for Matrix<T> {
    type Marker = T::Marker;

    fn operand(&self) -> Result<Buffer<'_, T::Marker>> {
        Ok(self.memory.operand())
    }
}

/// A projection weight in the encoding its checkpoint stores, with the scales it needs.
#[derive(Clone)]
pub enum ProjectionWeight {
    Dense(Matrix<bf16>),
    /// One scale per output channel.
    Fp8(Matrix<f8e4m3fn>, Allocation<f32>),
    /// Block-scaled FP8: the scales hold one entry per output channel per 128-column input
    /// block, already expanded along rows so the GEMM can index them like per-channel ones.
    Fp8Block(Matrix<f8e4m3fn>, Matrix<f32>),
    /// Packed code pairs, one E4M3 scale per 16-column group, and the global scale.
    Fp4(Matrix<f4e2m1fnx2>, Matrix<f8e4m3fn>, f32),
}

impl ProjectionWeight {
    /// What keeps this weight's storage alive; the FP4 and FP8 forms also hold their scales.
    pub(crate) fn owner(&self) -> Owner {
        Arc::new(self.clone())
    }

    /// Bytes of device memory the weight and its scales occupy.
    pub(crate) fn bytes(&self) -> usize {
        fn bytes<T: DType>(memory: &Allocation<T>) -> usize {
            memory.len() * size_of::<T>()
        }
        match self {
            Self::Dense(w) => bytes(w.allocation()),
            Self::Fp8(w, s) => bytes(w.allocation()) + bytes(s),
            Self::Fp8Block(w, s) => bytes(w.allocation()) + bytes(s.allocation()),
            Self::Fp4(w, s, _) => bytes(w.allocation()) + bytes(s.allocation()),
        }
    }

    /// The stored `[rows, columns]` of the weight matrix; FP4 columns count packed pairs.
    pub(crate) const fn shape(&self) -> [usize; 2] {
        match self {
            Self::Dense(w) => w.shape(),
            Self::Fp8(w, _) | Self::Fp8Block(w, _) => w.shape(),
            Self::Fp4(w, _, _) => w.shape(),
        }
    }

    pub(crate) fn validate(&self, rows: usize, columns: usize) -> Result<()> {
        let valid = match self {
            Self::Dense(w) => w.shape() == [rows, columns],
            Self::Fp8(w, s) => w.shape() == [rows, columns] && s.len() == rows,
            Self::Fp8Block(w, s) => {
                columns.is_multiple_of(FP8_BLOCK_COLUMNS)
                    && w.shape() == [rows, columns]
                    && s.shape() == [rows, columns / FP8_BLOCK_COLUMNS]
            }
            Self::Fp4(w, s, g) => {
                columns.is_multiple_of(NVFP4_GROUP_SIZE)
                    && w.shape() == [rows, columns / 2]
                    && s.shape() == [rows, columns / NVFP4_GROUP_SIZE]
                    && g.is_finite()
                    && *g > 0.0
            }
        };
        if valid {
            Ok(())
        } else {
            Err(Error::invalid("MLP projection dimensions/scales"))
        }
    }

    /// The weight in the form the projection contract takes.
    pub(crate) fn weight_ref(&self) -> Result<WeightRef<'_>> {
        match self {
            Self::Dense(w) => Ok(WeightRef::Dense(buffer(w)?)),
            Self::Fp8(w, s) => Ok(WeightRef::Fp8 {
                weight: buffer(w)?,
                scales: buffer(s)?,
            }),
            Self::Fp8Block(..) => Err(device_error(
                "block FP8 requires the resident FP8 GEMM".to_string(),
            )),
            Self::Fp4(w, s, global) => Ok(WeightRef::Fp4 {
                weight: buffer(w)?,
                scales: buffer(s)?,
                inverse_global: global.recip(),
            }),
        }
    }

    pub(crate) fn record(
        &self,
        kernels: &Kernels,
        scope: &Recording<'_>,
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
        columns: usize,
        tile: LinearTiling,
    ) -> Result<()> {
        kernels
            .projection
            .row(
                scope,
                &DownPlan {
                    tile_rows: tile.rows(),
                    tile_columns: tile.columns(),
                    columns,
                },
                out,
                input,
                self.weight_ref()?,
            )
            .map_err(|error| device_error(error.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/projection_tensors.rs"]
mod tensors;
