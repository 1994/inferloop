//! Matrices over cuTile tensors, for tests that build weights or launch reference kernels on them.
use super::Matrix;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Allocation;
use std::sync::Arc;

impl<T: DType> Matrix<T> {
    /// Adopt a rank-2 tensor.
    /// # Errors
    /// Rejects tensors that are not contiguous matrices.
    pub fn adopt(tensor: Arc<Tensor<T>>) -> Result<Self> {
        let [rows, columns] = tensor.shape() else {
            return Err(Error::invalid("weight matrix rank"));
        };
        let dimension = |value: &i32| {
            usize::try_from(*value).map_err(|_| Error::invalid("weight matrix shape"))
        };
        let (rows, columns) = (dimension(rows)?, dimension(columns)?);
        Ok(Self {
            memory: Allocation::adopt(tensor)?,
            rows,
            columns,
        })
    }

    /// The tensor behind this matrix, which shares its memory.
    #[must_use]
    pub const fn tensor(&self) -> &Arc<Tensor<T>> {
        self.memory.tensor()
    }

    /// The matrix as a `[rows, columns]` tensor sharing its memory.
    /// # Errors
    /// Rejects a shape cuTile cannot describe.
    pub fn shaped(&self) -> Result<Tensor<T>> {
        self.memory
            .view(0..self.memory.len(), &[self.rows, self.columns])
    }
}
