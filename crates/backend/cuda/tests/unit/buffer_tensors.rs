//! Rank-2 cuTile tensors as row-major operands, for tests that still launch over tensors.
use super::{RowMajor, buffer, error};
use crate::device::device_error;
use cutile::prelude::*;
use infer_core::Result;
use infer_cuda_kernels::contract::Recording;
use infer_cuda_kernels::tile::runtime::Element;

/// A rank-2 tensor as the [`RowMajor`] it holds.
/// # Errors
/// Rejects tensors that are not contiguous matrices.
pub fn tensor_rows<E: DType + Element>(tensor: &Tensor<E>) -> Result<RowMajor<'_, E::Marker>> {
    let [rows, columns] = tensor.shape() else {
        return Err(error("operand is not a matrix"));
    };
    let dimension = |value: &i32| usize::try_from(*value).map_err(error);
    RowMajor::new(buffer(tensor)?, dimension(rows)?, dimension(columns)?)
}

/// The cuTile capture behind `recording`, for tests that record reference kernels directly.
/// # Errors
/// Rejects a recording that is not a cuTile capture.
pub fn cutile<'a>(recording: &Recording<'a>) -> Result<Reference<'a>> {
    recording
        .cutile_scope()
        .map(Reference)
        .ok_or_else(|| device_error("reference kernels need a cuTile capture"))
}

/// A cuTile capture that reports failures as the project's errors.
pub struct Reference<'a>(&'a Scope);

impl Reference<'_> {
    /// Record `operation` into the capture.
    /// # Errors
    /// Returns the capture's refusal.
    pub fn record<O: GraphNode>(&self, operation: O) -> Result<<O as DeviceOp>::Output> {
        self.0.record(operation).map_err(device_error)
    }
}
