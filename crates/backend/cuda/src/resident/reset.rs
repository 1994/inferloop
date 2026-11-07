//! Clearing every recurrent and rank-1 state in place, as one replayable graph.
use super::buffers::buffer;
use crate::device::{CudaDevice, device_error};
use infer_core::Result;
use infer_cuda_kernels::runtime::Graph;

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_reset.rs"]
mod tests;

/// Record a clear of every state tensor; the program owns them for as long as the graph lives.
///
/// # Errors
/// Reports a state the device cannot address or a rejected capture.
pub(super) fn capture(device: &CudaDevice, states: &super::batch::States) -> Result<Graph> {
    let kernels = device.kernels();
    device.tile().capture(|scope| {
        let recording = scope;
        for tensor in states.values().flatten() {
            kernels
                .state
                .clear(recording, buffer(tensor)?)
                .map_err(|error| device_error(error.to_string()))?;
        }
        Ok(())
    })
}
