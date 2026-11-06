//! Rank-2 cuTile tensors as row-major operands, for tests that still launch over tensors.
use super::{RowMajor, buffer, error};
use cutile::prelude::*;
use infer_cuda_kernels::contract::Recording;
use infer_cuda_kernels::tile::runtime::Element;

/// A rank-2 tensor as the [`RowMajor`] it holds.
/// # Errors
/// Rejects tensors that are not contiguous matrices.
pub fn tensor_rows<E: DType + Element>(
    tensor: &Tensor<E>,
) -> Result<RowMajor<'_, E::Marker>, DeviceError> {
    let [rows, columns] = tensor.shape() else {
        return Err(error("operand is not a matrix"));
    };
    let dimension = |value: &i32| usize::try_from(*value).map_err(error);
    RowMajor::new(buffer(tensor)?, dimension(rows)?, dimension(columns)?)
}

/// The cuTile scope behind `recording`, for tests that record reference kernels directly.
/// # Errors
/// Rejects a recording that is not a cuTile capture.
pub fn cutile<'a>(recording: &Recording<'a>) -> Result<&'a Scope, DeviceError> {
    recording
        .cutile_scope()
        .ok_or_else(|| DeviceError::Internal("reference kernels need a cuTile capture".into()))
}
