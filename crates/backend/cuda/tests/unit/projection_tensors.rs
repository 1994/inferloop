//! Matrices over cuTile tensors, for tests that build weights or launch reference kernels on them.
use super::Matrix;
use crate::device::device_error;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::runtime::Allocation;
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

    /// The matrix as a `[rows, columns]` tensor sharing its memory, as reference kernels take it.
    /// # Errors
    /// Rejects a shape cuTile cannot describe.
    pub fn tensor(&self) -> Result<Arc<Tensor<T>>> {
        self.memory
            .view(0..self.memory.len(), &[self.rows, self.columns])
            .map(Arc::new)
            .map_err(|error| device_error(error.to_string()))
    }
}
