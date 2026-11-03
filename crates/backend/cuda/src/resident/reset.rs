//! Clearing every recurrent and rank-1 state in place, as one replayable graph.
use super::buffers::buffer;
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::{Result, TensorId};
use infer_cuda_kernels::contract::Recording;
use infer_cuda_kernels::tile::runtime::Graph;
use std::collections::BTreeMap;

#[cfg(test)]
#[path = "../../tests/unit/resident_reset.rs"]
mod tests;

/// Record a clear of every state tensor; the program owns them for as long as the graph lives.
///
/// # Errors
/// Reports a state the device cannot address or a rejected capture.
pub(super) fn capture(
    device: &CudaDevice,
    states: &BTreeMap<TensorId, Vec<Tensor<f32>>>,
) -> Result<Graph> {
    let kernels = device.kernels();
    Graph::scope(&device.stream, |scope| {
        let recording = Recording::scope(scope);
        for tensor in states.values().flatten() {
            kernels
                .state
                .clear(&recording, buffer(tensor)?)
                .map_err(|error| DeviceError::Launch(error.to_string()))?;
        }
        Ok(())
    })
    .map_err(device_error)
}
