//! Allocation-free MLP graph replay with resident intermediate activations.
mod projection;
use crate::constants::MAX_HIDDEN_SIZE;
use crate::{
    device::{CudaDevice, device_error},
    strategy::LinearTiling,
};
use cutile::tile_kernel::validate_programmatic_dependent_launch;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{DownPlan, Kernels, ProductBuffers, Recording, RowNormPlan};
use infer_cuda_kernels::tile::runtime::{Allocation, Graph};
pub use projection::{Matrix, ProjectionWeight};
use std::fmt::Display;

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
    pub norm: Allocation<f32>,
    pub gate: ProjectionWeight,
    pub up: ProjectionWeight,
    pub down: ProjectionWeight,
}

/// Owns stable input/output buffers; the graph retains all intermediate storage.
pub struct MlpGraph {
    /// Activations the graph addresses by device pointer; it does not hold them itself.
    _storage: [Allocation<f32>; ACTIVATIONS],
    /// Weights the graph addresses by device pointer.
    _weights: MlpWeights,
    graph: Graph,
    input: Allocation<f32>,
    output: Allocation<f32>,
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
            || weights.norm.len() != d
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
        let runtime = device.runtime();
        let allocate = |size| runtime.zeros::<f32>(size);
        let input = allocate(d)?;
        let normalized = allocate(d)?;
        let gate = allocate(n)?;
        let up = allocate(n)?;
        let activated = allocate(n)?;
        let down = allocate(d)?;
        let output = allocate(d)?;
        let kernels = device.kernels();
        let graph = device.tile().capture(|scope| {
            let recording = scope;
            kernels
                .mlp
                .row_norm(
                    recording,
                    &RowNormPlan {
                        dimension: d,
                        epsilon: config.epsilon,
                        offset: config.norm_offset,
                    },
                    normalized.operand(),
                    input.operand(),
                    weights.norm.operand(),
                )
                .map_err(failed)?;
            weights.gate.record(
                &kernels,
                scope,
                gate.operand(),
                normalized.operand(),
                d,
                config.tiling,
            )?;
            weights.up.record(
                &kernels,
                scope,
                up.operand(),
                normalized.operand(),
                d,
                config.tiling,
            )?;
            let parts = Activation {
                gate: &gate,
                up: &up,
                activated: &activated,
            };
            record_down(&kernels, scope, weights, config, &parts, &down, n)?;
            kernels
                .mlp
                .residual(recording, output.operand(), down.operand(), input.operand())
                .map_err(failed)?;
            Ok(())
        })?;
        Ok(Self {
            graph,
            input,
            output,
            device: device.clone(),
            hidden: d,
            _storage: [normalized, gate, up, activated, down],
            _weights: weights.clone(),
        })
    }

    /// Replays all six kernels with no device allocation or intermediate host copies.
    /// # Errors
    /// Rejects invalid input dimensions or CUDA execution failures.
    pub fn replay(&mut self, input: &Allocation<f32>) -> Result<()> {
        if input.len() != self.input.len() {
            return Err(Error::invalid("MLP input dimensions"));
        }
        self.device
            .enqueue(|transfers| transfers.copy(self.input.operand(), input.operand()))?;
        self.device.tile().sync(self.graph.launch())
    }

    /// Diagnostic host boundary; production callers should keep activations on device.
    /// # Errors
    /// Returns dimension or CUDA transfer errors.
    pub fn apply(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.hidden {
            return Err(Error::invalid("MLP input length"));
        }
        let uploaded = self.device.runtime().upload(input.to_vec())?;
        self.replay(&uploaded)?;
        self.device.runtime().read(&self.output)
    }
}

fn failed(error: impl Display) -> Error {
    device_error(error.to_string())
}

/// The activation product's operands.
struct Activation<'a> {
    gate: &'a Allocation<f32>,
    up: &'a Allocation<f32>,
    activated: &'a Allocation<f32>,
}

/// The activation product and the down projection it feeds, waiting on it under PDL.
fn record_down(
    kernels: &Kernels,
    scope: &Recording<'_>,
    weights: &MlpWeights,
    config: &MlpConfig,
    parts: &Activation<'_>,
    down: &Allocation<f32>,
    columns: usize,
) -> Result<()> {
    let recording = scope;
    let product = ProductBuffers {
        out: parts.activated.operand(),
        gate: parts.gate.operand(),
        up: parts.up.operand(),
    };
    if config.pdl {
        kernels
            .mlp
            .signalling_product(recording, product)
            .map_err(failed)?;
        return kernels
            .mlp
            .down_after_product(
                recording,
                &DownPlan {
                    tile_rows: config.tiling.rows(),
                    tile_columns: config.tiling.columns(),
                    columns,
                },
                down.operand(),
                parts.activated.operand(),
                weights.down.weight_ref()?,
            )
            .map_err(failed);
    }
    kernels.mlp.product(recording, product).map_err(failed)?;
    weights.down.record(
        kernels,
        scope,
        down.operand(),
        parts.activated.operand(),
        columns,
        config.tiling,
    )
}
