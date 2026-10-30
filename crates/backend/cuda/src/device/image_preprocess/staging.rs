//! Reusable pinned uploads returned to their pool only after device completion.
use crate::device::device_error;
use cuda_core::{CudaContext, DType, DeviceCopy, PinnedHostBuffer};
use cutile::prelude::DeviceError;
use infer_core::{Error, Result};
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};

type Available<T> = Arc<Mutex<Vec<PinnedHostBuffer<T>>>>;

pub(super) struct UploadPool<T: DType + DeviceCopy> {
    context: Arc<CudaContext>,
    available: Available<T>,
}

impl<T: DType + DeviceCopy> UploadPool<T> {
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self {
            context,
            available: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn stage(&self, values: &[T]) -> Result<Arc<Staged<T>>> {
        let reusable = {
            let mut available = self
                .available
                .lock()
                .map_err(|_| Error::invariant("image upload pool poisoned"))?;
            let index = available
                .iter()
                .enumerate()
                .filter(|(_, buffer)| buffer.len() >= values.len())
                .min_by_key(|(_, buffer)| buffer.len())
                .map(|(index, _)| index);
            let buffer = index.map(|index| available.swap_remove(index));
            drop(available);
            buffer
        };
        let buffer = if let Some(mut buffer) = reusable {
            buffer.as_mut_slice()[..values.len()].copy_from_slice(values);
            buffer
        } else {
            PinnedHostBuffer::from_slice(&self.context, values).map_err(device_error)?
        };
        Ok(Arc::new(Staged {
            buffer: Some(buffer),
            len: values.len(),
            available: Arc::clone(&self.available),
        }))
    }
}

pub(super) struct Staged<T: DType + DeviceCopy> {
    buffer: Option<PinnedHostBuffer<T>>,
    len: usize,
    available: Available<T>,
}

impl<T: DType + DeviceCopy> Staged<T> {
    pub fn buffer(&self) -> StdResult<&PinnedHostBuffer<T>, DeviceError> {
        self.buffer
            .as_ref()
            .ok_or_else(|| DeviceError::Internal("image upload buffer missing".into()))
    }

    pub const fn len(&self) -> usize {
        self.len
    }
}

impl<T: DType + DeviceCopy> Drop for Staged<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            if let Ok(mut available) = self.available.lock() {
                available.push(buffer);
            } else {
                tracing::error!("image upload pool poisoned; releasing pinned buffer");
            }
        }
    }
}
