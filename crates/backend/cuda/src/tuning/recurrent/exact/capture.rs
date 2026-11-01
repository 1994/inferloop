//! Complete prepared recurrence captures, including original transpose and state reset.
use super::super::reference::{Fixture, maximum_error};
use crate::constants::METADATA_FIELDS;
use crate::{
    device::{CudaDevice, device_error},
    resident::recurrent_prefill::{kernels, oxide},
};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Graph;
const KH: usize = 16;
const VH: usize = 48;
const DIM: usize = 128;
const NATIVE_VALUE_COLUMNS: usize = 16;
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
        Graph::scope(&device.stream, |scope| {
            scope.record(api::memcpy(&mut self.state, &self.initial))?;
            scope.record(
                kernels::normalize_shared((&mut self.normalized).partition([1, 1, DIM]), &self.qkv)
                    .generics(vec![KH.to_string(), DIM.to_string()]),
            )?;
            scope.record(kernels::gates_shared(
                (&mut self.betas).partition([1, 1]),
                (&mut self.decays).partition([1, 1]),
                &self.beta,
                &self.alpha,
                &self.a_log,
                &self.bias,
            ))?;
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
                scope.record(
                    kernels::delta_precomputed(
                        (&mut self.state).partition([1, DIM, NATIVE_VALUE_COLUMNS]),
                        (&mut self.scratch).partition([1, width, NATIVE_VALUE_COLUMNS]),
                        &self.qkv,
                        &self.normalized,
                        &self.betas,
                        &self.decays,
                        &self.metadata,
                    )
                    .generics(vec![
                        KH.to_string(),
                        VH.to_string(),
                        DIM.to_string(),
                        width.to_string(),
                        "16".into(),
                    ]),
                )?;
            }
            scope.record(
                kernels::transpose((&mut self.output).partition([1, 1, DIM]), &self.scratch)
                    .generics(vec![DIM.to_string()]),
            )?;
            Ok(())
        })
        .map_err(device_error)
    }
    pub fn identities(&mut self, device: &CudaDevice) -> Result<Vec<String>> {
        let width = self.width;
        let keys = vec![
            kernels::normalize_shared((&mut self.normalized).partition([1, 1, DIM]), &self.qkv)
                .generics(vec![KH.to_string(), DIM.to_string()])
                .l2_cache_key_on(&device.stream)
                .map_err(device_error)?,
            kernels::gates_shared(
                (&mut self.betas).partition([1, 1]),
                (&mut self.decays).partition([1, 1]),
                &self.beta,
                &self.alpha,
                &self.a_log,
                &self.bias,
            )
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?,
            kernels::delta_precomputed(
                (&mut self.state).partition([1, DIM, NATIVE_VALUE_COLUMNS]),
                (&mut self.scratch).partition([1, width, NATIVE_VALUE_COLUMNS]),
                &self.qkv,
                &self.normalized,
                &self.betas,
                &self.decays,
                &self.metadata,
            )
            .generics(vec![
                KH.to_string(),
                VH.to_string(),
                DIM.to_string(),
                width.to_string(),
                "16".into(),
            ])
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?,
            kernels::transpose((&mut self.output).partition([1, 1, DIM]), &self.scratch)
                .generics(vec![DIM.to_string()])
                .l2_cache_key_on(&device.stream)
                .map_err(device_error)?,
        ];
        if keys.iter().any(String::is_empty) {
            return Err(Error::invalid("exact GDN native kernel identity"));
        }
        Ok(keys)
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
