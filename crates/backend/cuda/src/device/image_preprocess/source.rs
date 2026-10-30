//! Device input operations retained through their submission fence.
use super::{nvjpeg::Lease, staging::Staged};
use crate::device::CudaDevice;
use cutile::prelude::{DeviceError, ExecutionContext, Tensor};
use std::sync::Arc;

pub(super) enum Source {
    Rgb(Arc<Staged<u8>>),
    Jpeg {
        lease: Arc<Lease>,
        bytes: Arc<Vec<u8>>,
        width: usize,
    },
}

impl Source {
    pub fn retain(&self, context: &ExecutionContext) -> Result<(), DeviceError> {
        match self {
            Self::Rgb(staged) => context.retain(Arc::clone(staged)),
            Self::Jpeg { lease, bytes, .. } => {
                context.retain(Arc::clone(lease))?;
                context.retain(Arc::clone(bytes))
            }
        }
    }

    pub fn enqueue(
        &self,
        context: &ExecutionContext,
        device: &CudaDevice,
        output: &mut Tensor<u8>,
    ) -> Result<(), DeviceError> {
        match self {
            Self::Rgb(staged) => device
                .copy_h2d_pinned(output, staged.buffer()?, staged.len())
                .map_err(|error| DeviceError::Internal(error.to_string())),
            Self::Jpeg {
                lease,
                bytes,
                width,
            } => lease
                .decode(
                    Arc::clone(bytes),
                    output.device_pointer().cu_deviceptr(),
                    *width,
                    context.get_cuda_stream().cu_stream().cast(),
                )
                .map_err(|error| DeviceError::Internal(error.to_string())),
        }
    }
}
