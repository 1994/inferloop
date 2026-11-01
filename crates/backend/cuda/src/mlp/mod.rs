//! Allocation-free MLP graph replay with resident intermediate activations.
mod projection;
use crate::constants::MAX_HIDDEN_SIZE;
use crate::resident::tensor_buffer as buffer;
use crate::{
    device::{CudaDevice, device_error},
    strategy::LinearTiling,
};
use cutile::prelude::*;
use cutile::tile_kernel::validate_programmatic_dependent_launch;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    DownPlan, DownWeight, Kernels, ProductBuffers, Recording, RowNormPlan,
};
use infer_cuda_kernels::tile::runtime::Graph;
pub use projection::ProjectionWeight;
use std::fmt::Display;
use std::result::Result as StdResult;

/// Normalized, gate, up, activated and down rows.
const ACTIVATIONS: usize = 5;
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

#[derive(Clone)]
pub struct MlpWeights {
    pub norm: Arc<Tensor<f32>>,
    pub gate: ProjectionWeight,
    pub up: ProjectionWeight,
    pub down: ProjectionWeight,
}

/// Owns stable input/output buffers; the graph retains all intermediate storage.
pub struct MlpGraph {
    /// Activations the graph addresses by device pointer; it does not hold them itself.
    _storage: [Tensor<f32>; ACTIVATIONS],
    /// Weights the graph addresses by device pointer.
    _weights: MlpWeights,
    graph: Graph,
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
            validate_programmatic_dependent_launch(device.stream.device().ordinal())
                .map_err(device_error)?;
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
        let normalized = allocate(d)?;
        let mut gate = allocate(n)?;
        let mut up = allocate(n)?;
        let activated = allocate(n)?;
        let mut down = allocate(d)?;
        let output = allocate(d)?;
        let kernels = device.kernels();
        let graph = Graph::scope(&device.stream, |scope| {
            let recording = Recording::scope(scope);
            kernels
                .mlp
                .row_norm(
                    &recording,
                    &RowNormPlan {
                        dimension: d,
                        epsilon: config.epsilon,
                        offset: config.norm_offset,
                    },
                    buffer(&normalized)?,
                    buffer(&input)?,
                    buffer(weights.norm.as_ref())?,
                )
                .map_err(failed)?;
            weights
                .gate
                .record(scope, &mut gate, &normalized, d, config.tiling)?;
            weights
                .up
                .record(scope, &mut up, &normalized, d, config.tiling)?;
            let parts = Activation {
                gate: &gate,
                up: &up,
                activated: &activated,
            };
            record_down(&kernels, scope, weights, config, &parts, &mut down, n)?;
            kernels
                .mlp
                .residual(
                    &recording,
                    buffer(&output)?,
                    buffer(&down)?,
                    buffer(&input)?,
                )
                .map_err(failed)?;
            Ok(())
        })
        .map_err(device_error)?;
        Ok(Self {
            graph,
            input,
            output: Arc::new(output),
            device: device.clone(),
            hidden: d,
            _storage: [normalized, gate, up, activated, down],
            _weights: weights.clone(),
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

fn failed(error: impl Display) -> DeviceError {
    DeviceError::Internal(error.to_string())
}

/// The activation product's operands.
struct Activation<'a> {
    gate: &'a Tensor<f32>,
    up: &'a Tensor<f32>,
    activated: &'a Tensor<f32>,
}

/// The activation product and the down projection it feeds, waiting on it under PDL.
fn record_down(
    kernels: &Kernels,
    scope: &Scope,
    weights: &MlpWeights,
    config: &MlpConfig,
    parts: &Activation<'_>,
    down: &mut Tensor<f32>,
    columns: usize,
) -> StdResult<(), DeviceError> {
    let recording = Recording::scope(scope);
    let product = ProductBuffers {
        out: buffer(parts.activated)?,
        gate: buffer(parts.gate)?,
        up: buffer(parts.up)?,
    };
    if config.pdl {
        kernels
            .mlp
            .signalling_product(&recording, product)
            .map_err(failed)?;
        return kernels
            .mlp
            .down_after_product(
                &recording,
                &DownPlan {
                    tile_rows: config.tiling.rows(),
                    tile_columns: config.tiling.columns(),
                    columns,
                },
                buffer(&*down)?,
                buffer(parts.activated)?,
                down_weight(&weights.down)?,
            )
            .map_err(failed);
    }
    kernels.mlp.product(&recording, product).map_err(failed)?;
    weights
        .down
        .record(scope, down, parts.activated, columns, config.tiling)
}

/// The weight in the form the contract's down projection takes.
fn down_weight(weight: &ProjectionWeight) -> StdResult<DownWeight<'_>, DeviceError> {
    match weight {
        ProjectionWeight::Dense(w) => Ok(DownWeight::Dense(buffer(w.as_ref())?)),
        ProjectionWeight::Fp8(w, s) => Ok(DownWeight::Fp8 {
            weight: buffer(w.as_ref())?,
            scales: buffer(s.as_ref())?,
        }),
        ProjectionWeight::Fp8Block(..) => Err(DeviceError::Launch(
            "block FP8 requires the resident FP8 GEMM".to_string(),
        )),
        ProjectionWeight::Fp4(w, s, global) => Ok(DownWeight::Fp4 {
            weight: buffer(w.as_ref())?,
            scales: buffer(s.as_ref())?,
            inverse_global: global.recip(),
        }),
    }
}
