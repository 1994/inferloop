//! Clearing every recurrent and rank-1 state in place, as one replayable graph.
use super::batch;
use super::buffers::buffer;
use crate::device::{CudaDevice, device_error};
use infer_core::Result;
use infer_cuda_kernels::runtime::Graph;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests;

/// Record a clear of every state tensor; the program owns them for as long as the graph lives.
///
/// # Errors
/// Reports a state the device cannot address or a rejected capture.
pub(super) fn capture(device: &CudaDevice, states: &batch::States) -> Result<Graph> {
    let kernels = device.kernels();
    device.tile().capture(|scope| {
        let recording = scope;
        // Zero is the same bits at every storage, so a BF16 state clears as F32 words.
        for tensor in states.values().flatten() {
            let words = tensor.words()?;
            kernels
                .state
                .clear(recording, buffer(&words)?)
                .map_err(|error| device_error(error.to_string()))?;
        }
        Ok(())
    })
}
