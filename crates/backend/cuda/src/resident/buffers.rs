//! Contract buffers over tensors a resident program owns.
#![expect(
    unsafe_code,
    reason = "The one place host tensors become contract buffers, so kernel calls stay safe"
)]
use super::capture::error;
use cutile::prelude::*;
use infer_cuda_kernels::contract::Buffer;
use infer_cuda_kernels::tile::runtime::Element;

/// A buffer over one tensor.
///
/// A resident program owns its tensors until its captured graphs are destroyed, which is what
/// [`Buffer::new`] requires of every caller here.
pub fn buffer<T: DType + Element>(
    tensor: &Tensor<T>,
) -> Result<Buffer<'_, T::Marker>, DeviceError> {
    if !tensor.is_contiguous() {
        return Err(error("kernel operand must be contiguous"));
    }
    // SAFETY: see above; the arena hands distinct slots to a launch's inputs and outputs.
    Ok(unsafe { Buffer::new(tensor.device_pointer().cu_deviceptr(), tensor.size()) })
}

/// A buffer over part of a program-owned allocation, which a view does not expose by address.
pub(super) const fn region(address: u64, len: usize) -> Buffer<'static, f32> {
    // SAFETY: as for [`buffer`]; callers derive both from a contiguous program-owned tensor.
    unsafe { Buffer::new(address, len) }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_runtime_replay.rs"]
mod runtime_tests;
