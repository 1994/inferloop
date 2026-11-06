//! Rank-2 cuTile tensors as row-major operands, for tests that still launch over tensors.
use super::{RowMajor, buffer, error};
use cutile::prelude::*;
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
