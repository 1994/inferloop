//! Allocation-free MLP graph replay with resident intermediate activations.
mod kernels;
mod pdl;
mod pdl_consumers;
mod projection;
use crate::constants::AUX_KERNEL_TILE;
use crate::constants::MAX_HIDDEN_SIZE;
use crate::{
    device::{CudaDevice, device_error},
    strategy::LinearTiling,
};
use cutile::prelude::*;
use infer_core::{Error, Result};
use kernels::fused;
pub use projection::ProjectionWeight;

/// Intermediate size bound for resident MLP graphs.
const MAX_INTERMEDIATE_SIZE: usize = 131_072;

pub struct MlpConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub epsilon: f32,
    pub norm_offset: f32,
    pub tiling: LinearTiling,
    /// Experimental PDL edge; disabled unless explicitly requested.
    pub pdl: bool,
}

pub struct MlpWeights {
    pub norm: Arc<Tensor<f32>>,
    pub gate: ProjectionWeight,
    pub up: ProjectionWeight,
    pub down: ProjectionWeight,
}

/// Owns stable input/output buffers; the graph retains all intermediate storage.
pub struct MlpGraph {
    graph: CudaGraph<()>,
    input: Tensor<f32>,
    output: Arc<Tensor<f32>>,
    device: CudaDevice,
    hidden: usize,
}

impl MlpGraph {
    /// # Errors
    /// Rejects invalid weights, unsupported precision or failed CUDA capture.
    pub fn new(device: &CudaDevice, config: &MlpConfig, weights: &MlpWeights) -> Result<Self> {
        let d = config.hidden;
        if config.pdl {
            device.validate_programmatic_dependent_launch()?;
        }
        let n = config.intermediate;
        if d == 0
            || n == 0
            || d > MAX_HIDDEN_SIZE
            || n > MAX_INTERMEDIATE_SIZE
            || !config.epsilon.is_finite()
            || config.epsilon <= 0.0
            || !config.norm_offset.is_finite()
            || weights.norm.shape() != [i32::try_from(d).map_err(device_error)?]
        {
            return Err(Error::invalid("MLP configuration"));
        }
        weights.gate.validate(n, d)?;
        weights.up.validate(n, d)?;
        weights.down.validate(d, n)?;
        for weight in [&weights.gate, &weights.up, &weights.down] {
            if matches!(weight, ProjectionWeight::Fp4(..)) {
                device.target().require_native_nvfp4()?;
            }
        }
        let allocate = |size| {
            api::zeros::<f32>(&[size])
                .sync_on(&device.stream)
                .map_err(device_error)
        };
        let input = allocate(d)?;
        let mut normalized = allocate(d)?;
        let mut gate = allocate(n)?;
        let mut up = allocate(n)?;
        let mut activated = allocate(n)?;
        let mut down = allocate(d)?;
        let mut output = allocate(d)?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                fused::norm(
                    (&mut normalized).partition([d.next_power_of_two()]),
                    &input,
                    &weights.norm,
                    config.epsilon,
                    config.norm_offset,
                )
                .generics(vec![d.to_string(), d.next_power_of_two().to_string()]),
            )?;
            weights
                .gate
                .record(scope, &mut gate, &normalized, d, config.tiling)?;
            weights
                .up
                .record(scope, &mut up, &normalized, d, config.tiling)?;
            if config.pdl {
                pdl::record_product(scope, &mut activated, &gate, &up)?;
                pdl::record_down(
                    scope,
                    &weights.down,
                    &mut down,
                    &activated,
                    n,
                    config.tiling,
                )?;
            } else {
                scope.record(
                    fused::silu_mul((&mut activated).partition([AUX_KERNEL_TILE]), &gate, &up)
                        .generics(vec![AUX_KERNEL_TILE.to_string()]),
                )?;
                weights
                    .down
                    .record(scope, &mut down, &activated, n, config.tiling)?;
            }
            scope.record(
                fused::residual((&mut output).partition([AUX_KERNEL_TILE]), &down, &input)
                    .generics(vec![AUX_KERNEL_TILE.to_string()]),
            )?;
            Ok(())
        })
        .map_err(device_error)?;
        Ok(Self {
            graph,
            input,
            output: Arc::new(output),
            device: device.clone(),
            hidden: d,
        })
    }

    /// Replays all six kernels with no device allocation or intermediate host copies.
    /// # Errors
    /// Rejects invalid input dimensions or CUDA execution failures.
    pub fn replay(&mut self, input: &Tensor<f32>) -> Result<()> {
        if input.shape() != self.input.shape() {
            return Err(Error::invalid("MLP input dimensions"));
        }
        self.graph
            .update(api::memcpy(&mut self.input, input))
            .map_err(device_error)?;
        self.graph
            .launch()
            .sync_on(&self.device.stream)
            .map_err(device_error)
    }

    /// Diagnostic host boundary; production callers should keep activations on device.
    /// # Errors
    /// Returns dimension or CUDA transfer errors.
    pub fn apply(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.hidden {
            return Err(Error::invalid("MLP input length"));
        }
        let uploaded = self.device.upload(input.to_vec(), &[self.hidden])?;
        self.replay(&uploaded)?;
        (&self.output)
            .to_host_vec()
            .sync_on(&self.device.stream)
            .map_err(device_error)
    }
}
