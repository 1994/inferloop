//! Complete prepared recurrence captures, including original transpose and state reset.
use super::super::reference::{Fixture, maximum_error};
use crate::device::{CudaDevice, device_error};
use crate::resident::tensor_buffer as buffer;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{GateBuffers, PreparedDeltaBuffers, RecurrentShape};
use infer_cuda_kernels::tile::runtime::{Allocation, Graph};
pub(super) const KH: usize = 16;
pub(super) const VH: usize = 48;
pub(super) const DIM: usize = 128;
pub(super) const NATIVE_VALUE_COLUMNS: usize = 16;
fn zeros(device: &CudaDevice, shape: &[usize]) -> Result<Allocation<f32>> {
    device.runtime().zeros(shape.iter().product())
}
pub(super) struct Captured {
    qkv: Allocation<f32>,
    beta: Allocation<f32>,
    alpha: Allocation<f32>,
    a_log: Allocation<f32>,
    bias: Allocation<f32>,
    metadata: Allocation<i32>,
    initial: Allocation<f32>,
    normalized: Allocation<f32>,
    betas: Allocation<f32>,
    decays: Allocation<f32>,
    state: Allocation<f32>,
    scratch: Allocation<f32>,
    output: Allocation<f32>,
    width: usize,
}
impl Captured {
    pub fn new(device: &CudaDevice, f: &Fixture) -> Result<Self> {
        let width = f.shape[3];
        Ok(Self {
            qkv: device.runtime().upload(f.qkv.clone())?,
            beta: device.runtime().upload(f.beta.clone())?,
            alpha: device.runtime().upload(f.alpha.clone())?,
            a_log: device.runtime().upload(f.a_log.clone())?,
            bias: device.runtime().upload(f.bias.clone())?,
            metadata: device.runtime().upload(f.metadata.to_vec())?,
            initial: device.runtime().upload(f.initial.clone())?,
            normalized: zeros(device, &[width, 2 * KH, DIM])?,
            betas: zeros(device, &[width, VH])?,
            decays: zeros(device, &[width, VH])?,
            state: zeros(device, &[VH, DIM, DIM])?,
            scratch: zeros(device, &[VH, width, DIM])?,
            output: zeros(device, &[width, VH, DIM])?,
            width,
        })
    }
    /// The pipeline with the SIMT recurrence when `simt`, otherwise the tile one.
    pub fn graph(&self, device: &CudaDevice, simt: bool) -> Result<Graph> {
        let width = self.width;
        let contract = device.kernels();
        let shape = RecurrentShape {
            key_heads: KH,
            value_heads: VH,
            dim: DIM,
            lanes: width,
        };
        if simt && !contract.recurrent.prepare_simt_delta(shape)? {
            return Err(Error::invalid("no SIMT recurrence build covers this shape"));
        }
        Graph::scope(&device.stream, |scope| {
            let recording = scope;
            recording
                .copy(self.state.operand(), buffer(&self.initial)?)
                .map_err(|error| device_error(error.to_string()))?;
            contract
                .recurrent
                .normalize(
                    recording,
                    shape,
                    buffer(&self.normalized)?,
                    buffer(&self.qkv)?,
                )
                .map_err(|error| device_error(error.to_string()))?;
            contract
                .recurrent
                .gates(
                    recording,
                    shape,
                    GateBuffers {
                        betas: buffer(&self.betas)?,
                        decays: buffer(&self.decays)?,
                        beta: buffer(&self.beta)?,
                        alpha: buffer(&self.alpha)?,
                        decay: buffer(&self.a_log)?,
                        bias: buffer(&self.bias)?,
                    },
                )
                .map_err(|error| device_error(error.to_string()))?;
            let buffers = PreparedDeltaBuffers {
                state: buffer(&self.state)?,
                output: buffer(&self.scratch)?,
                qkv: buffer(&self.qkv)?,
                normalized: buffer(&self.normalized)?,
                betas: buffer(&self.betas)?,
                decays: buffer(&self.decays)?,
                metadata: buffer(&self.metadata)?,
            };
            if simt {
                contract
                    .recurrent
                    .delta_prepared_simt(recording, shape, buffers)
                    .map_err(|error| device_error(error.to_string()))?;
            } else {
                contract
                    .recurrent
                    .delta_prepared(recording, shape, NATIVE_VALUE_COLUMNS, buffers)
                    .map_err(|error| device_error(error.to_string()))?;
            }
            contract
                .recurrent
                .transpose(
                    recording,
                    shape,
                    buffer(&self.output)?,
                    buffer(&self.scratch)?,
                )
                .map_err(|error| device_error(error.to_string()))
        })
        .map_err(device_error)
    }
    pub fn run(&self, device: &CudaDevice, graph: &Graph) -> Result<[Vec<f32>; 2]> {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        Ok([
            device.runtime().read(&self.state)?,
            device.runtime().read(&self.output)?,
        ])
    }
    pub fn verify(
        &self,
        device: &CudaDevice,
        native: &Graph,
        rust: &Graph,
        fixture: &Fixture,
    ) -> Result<f64> {
        let expected = self.run(device, native)?;
        let actual = self.run(device, rust)?;
        for (a, b) in actual.iter().zip(&expected) {
            if a.len() != b.len()
                || !a
                    .iter()
                    .zip(b)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits())
            {
                return Err(Error::invalid(
                    "exact GDN complete state/output bits differ",
                ));
            }
        }
        let state_error = maximum_error(&actual[0], &fixture.state)?;
        let output_error = maximum_error(&actual[1], &fixture.output)?;
        Ok(state_error.max(output_error))
    }
}
