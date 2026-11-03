//! Complete prepared recurrence captures, including original transpose and state reset.
use super::super::reference::{Fixture, maximum_error};
use crate::constants::METADATA_FIELDS;
use crate::device::{CudaDevice, device_error};
use crate::resident::recurrent_prefill::oxide;
use crate::resident::tensor_buffer as buffer;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{GateBuffers, PreparedDeltaBuffers, Recording, RecurrentShape};
use infer_cuda_kernels::tile::runtime::Graph;
pub(super) const KH: usize = 16;
pub(super) const VH: usize = 48;
pub(super) const DIM: usize = 128;
pub(super) const NATIVE_VALUE_COLUMNS: usize = 16;
fn zeros(device: &CudaDevice, shape: &[usize]) -> Result<Tensor<f32>> {
    api::zeros::<f32>(shape)
        .sync_on(&device.stream)
        .map_err(device_error)
}
pub(super) struct Captured {
    qkv: Arc<Tensor<f32>>,
    beta: Arc<Tensor<f32>>,
    alpha: Arc<Tensor<f32>>,
    a_log: Arc<Tensor<f32>>,
    bias: Arc<Tensor<f32>>,
    metadata: Arc<Tensor<i32>>,
    initial: Arc<Tensor<f32>>,
    normalized: Tensor<f32>,
    betas: Tensor<f32>,
    decays: Tensor<f32>,
    state: Tensor<f32>,
    scratch: Tensor<f32>,
    output: Tensor<f32>,
    width: usize,
}
impl Captured {
    pub fn new(device: &CudaDevice, f: &Fixture) -> Result<Self> {
        let width = f.shape[3];
        Ok(Self {
            qkv: device.upload(f.qkv.clone(), &[width, 2 * KH + VH, DIM])?,
            beta: device.upload(f.beta.clone(), &[width, VH])?,
            alpha: device.upload(f.alpha.clone(), &[width, VH])?,
            a_log: device.upload(f.a_log.clone(), &[VH])?,
            bias: device.upload(f.bias.clone(), &[VH])?,
            metadata: device.upload(f.metadata.to_vec(), &[METADATA_FIELDS])?,
            initial: device.upload(f.initial.clone(), &[VH, DIM, DIM])?,
            normalized: zeros(device, &[width, 2 * KH, DIM])?,
            betas: zeros(device, &[width, VH])?,
            decays: zeros(device, &[width, VH])?,
            state: zeros(device, &[VH, DIM, DIM])?,
            scratch: zeros(device, &[VH, width, DIM])?,
            output: zeros(device, &[width, VH, DIM])?,
            width,
        })
    }
    pub fn graph(&mut self, device: &CudaDevice, rust: Option<&oxide::Support>) -> Result<Graph> {
        let width = self.width;
        let contract = device.kernels();
        let shape = RecurrentShape {
            key_heads: KH,
            value_heads: VH,
            dim: DIM,
            lanes: width,
        };
        Graph::scope(&device.stream, |scope| {
            scope.record(api::memcpy(&mut self.state, &self.initial))?;
            let recording = Recording::scope(scope);
            contract
                .recurrent
                .normalize(
                    &recording,
                    shape,
                    buffer(&self.normalized)?,
                    buffer(self.qkv.as_ref())?,
                )
                .map_err(|error| DeviceError::Internal(error.to_string()))?;
            contract
                .recurrent
                .gates(
                    &recording,
                    shape,
                    GateBuffers {
                        betas: buffer(&self.betas)?,
                        decays: buffer(&self.decays)?,
                        beta: buffer(self.beta.as_ref())?,
                        alpha: buffer(self.alpha.as_ref())?,
                        decay: buffer(self.a_log.as_ref())?,
                        bias: buffer(self.bias.as_ref())?,
                    },
                )
                .map_err(|error| DeviceError::Internal(error.to_string()))?;
            if let Some(function) = rust {
                function.record(
                    scope,
                    oxide::Bindings {
                        state: &mut self.state,
                        output: &mut self.scratch,
                        qkv: &self.qkv,
                        normalized: &self.normalized,
                        beta: &self.betas,
                        decay: &self.decays,
                        metadata: &self.metadata,
                    },
                    width,
                )?;
            } else {
                contract
                    .recurrent
                    .delta_prepared(
                        &recording,
                        shape,
                        NATIVE_VALUE_COLUMNS,
                        PreparedDeltaBuffers {
                            state: buffer(&self.state)?,
                            output: buffer(&self.scratch)?,
                            qkv: buffer(self.qkv.as_ref())?,
                            normalized: buffer(&self.normalized)?,
                            betas: buffer(&self.betas)?,
                            decays: buffer(&self.decays)?,
                            metadata: buffer(self.metadata.as_ref())?,
                        },
                    )
                    .map_err(|error| DeviceError::Internal(error.to_string()))?;
            }
            contract
                .recurrent
                .transpose(
                    &recording,
                    shape,
                    buffer(&self.output)?,
                    buffer(&self.scratch)?,
                )
                .map_err(|error| DeviceError::Internal(error.to_string()))
        })
        .map_err(device_error)
    }
    pub fn run(&self, device: &CudaDevice, graph: &Graph) -> Result<[Vec<f32>; 2]> {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        Ok([
            device.read_borrowed(&self.state)?,
            device.read_borrowed(&self.output)?,
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
