//! Where one vision capture's launches go.
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::Result;
use infer_cuda_kernels::contract::{
    Attention, Buffer, DenseBuffers, DensePlan, Gelu, Pack, PositionTaps, Recording, Vision,
    VisionDense, VisionNorm, VisionRope,
};
use infer_cuda_kernels::tile::kernels;
use infer_cuda_kernels::tile::runtime::Graph;
use std::result::Result as StdResult;

/// Capture one vision graph; it keeps every allocation it records alive.
///
/// # Errors
/// Reports what `record` reports, or a failed capture.
pub(super) fn capture(
    device: &CudaDevice,
    record: impl FnOnce(&Launches<'_>) -> StdResult<(), DeviceError>,
) -> Result<Graph> {
    let families = kernels(device.ordinal());
    device.runtime().capture(|recording| {
        record(&Launches {
            vision: &*families.vision,
            attention: &*families.attention,
            recording,
        })
        .map_err(device_error)
    })
}

/// One capture's destination.
pub(super) struct Launches<'a> {
    vision: &'a dyn Vision,
    attention: &'a dyn Attention,
    recording: &'a Recording<'a>,
}

fn launched(result: Result<()>) -> StdResult<(), DeviceError> {
    result.map_err(|error| DeviceError::Launch(error.to_string()))
}

impl Launches<'_> {
    pub(super) fn dense_bias(&self, dense: VisionDense<'_>) -> StdResult<(), DeviceError> {
        launched(self.vision.dense_bias(self.recording, dense))
    }

    pub(super) fn add(
        &self,
        columns: usize,
        out: Buffer<'_, f32>,
        left: Buffer<'_, f32>,
        right: Buffer<'_, f32>,
    ) -> StdResult<(), DeviceError> {
        launched(self.vision.add(self.recording, columns, out, left, right))
    }

    pub(super) fn gelu(
        &self,
        form: Gelu,
        columns: usize,
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
    ) -> StdResult<(), DeviceError> {
        launched(self.vision.gelu(self.recording, form, columns, out, input))
    }

    pub(super) fn layernorm(&self, norm: VisionNorm<'_>) -> StdResult<(), DeviceError> {
        launched(self.vision.layernorm(self.recording, norm))
    }

    pub(super) fn rope(&self, rope: VisionRope<'_>) -> StdResult<(), DeviceError> {
        launched(self.vision.rope(self.recording, rope))
    }

    pub(super) fn pack(
        &self,
        direction: Pack,
        rows: usize,
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
    ) -> StdResult<(), DeviceError> {
        launched(
            self.vision
                .pack(self.recording, direction, rows, out, input),
        )
    }

    pub(super) fn add_positions(
        &self,
        out: Buffer<'_, f32>,
        embedded: Buffer<'_, f32>,
        taps: PositionTaps<'_>,
    ) -> StdResult<(), DeviceError> {
        launched(
            self.vision
                .add_positions(self.recording, out, embedded, taps),
        )
    }

    pub(super) fn dense(
        &self,
        plan: DensePlan,
        buffers: DenseBuffers<'_>,
    ) -> StdResult<(), DeviceError> {
        launched(self.attention.dense(self.recording, plan, buffers))
    }
}
