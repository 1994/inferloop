use super::{
    CANDIDATES, SCHEMA, compile_options,
    evidence::{Key, Shape, Times, Trial},
};
use crate::device::{CudaDevice, device_error};
use crate::tuning::preparation::driver_build;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{NormBuffers, NormLayout, NormPlan};
use infer_cuda_kernels::tile::identity;
use infer_cuda_kernels::tile::runtime::{Allocation, Graph};
use std::iter::once;
use std::time::Duration;
pub(super) const JIT_OPT_LEVEL: u8 = 3;
const INPUT_PERIOD: u16 = 71;
const INPUT_CENTER: f32 = 35.0;
const INPUT_SCALE: f32 = 0.013;
const GATE_PERIOD: u16 = 67;
const GATE_CENTER: f32 = 33.0;
const GATE_SCALE: f32 = 0.031;
const WEIGHT_PERIOD: u16 = 11;
const WEIGHT_SCALE: f32 = 0.021;
const WEIGHT_BASE: f32 = 0.7;

pub struct Buffers {
    shape: Shape,
    x: Allocation<f32>,
    gate: Allocation<f32>,
    weight: Allocation<f32>,
    output: Allocation<f32>,
}
impl Buffers {
    pub fn new(device: &CudaDevice, shape: Shape) -> Result<Self> {
        let n = shape
            .rows
            .checked_mul(shape.dimension)
            .ok_or_else(|| Error::invalid("norm shape overflow"))?;
        let x = (0u16..INPUT_PERIOD)
            .cycle()
            .take(n)
            .map(|i| (f32::from(i) - INPUT_CENTER) * INPUT_SCALE)
            .collect();
        let gate = (0u16..GATE_PERIOD)
            .cycle()
            .take(n)
            .map(|i| (f32::from(i) - GATE_CENTER) * GATE_SCALE)
            .collect();
        let weight = (0u16..WEIGHT_PERIOD)
            .cycle()
            .take(shape.dimension)
            .map(|i| f32::from(i).mul_add(WEIGHT_SCALE, WEIGHT_BASE))
            .collect();
        let runtime = device.runtime();
        Ok(Self {
            shape,
            x: runtime.upload(x)?,
            gate: runtime.upload(gate)?,
            weight: runtime.upload(weight)?,
            output: runtime.zeros(n)?,
        })
    }
    pub fn key(device: &CudaDevice, shape: Shape) -> Result<Key> {
        use cutile::cutile_compiler::cuda_tile_runtime_utils as compiler;
        let occupancies: Vec<_> = once(0)
            .chain(CANDIDATES)
            .map(|occupancy| compile_options(occupancy).occupancy)
            .collect();
        let ids = identity::gated_norm_candidates(
            &device.stream,
            [shape.rows, shape.dimension],
            f32::from_bits(shape.epsilon_bits),
            &occupancies,
        )?;
        Ok(Key {
            hardware: serde_json::to_string(device.profile()?).map_err(device_error)?,
            driver_version: CudaDevice::driver_version()?,
            driver_build: driver_build(),
            shape,
            kernel_fingerprint: format!(
                "{SCHEMA}:{}:{:?}:{:?}:{}",
                compiler::tileiras_fingerprint(),
                compiler::bytecode_override(),
                compile_options(0),
                serde_json::to_string(&ids).map_err(device_error)?
            ),
        })
    }
    pub fn capture(&self, device: &CudaDevice, occupancy: i32) -> Result<Graph> {
        let s = self.shape;
        let kernels = device.kernels();
        let plan = NormPlan {
            layout: NormLayout::Block,
            rows: s.rows,
            dimension: s.dimension,
            gated: true,
            epsilon: f32::from_bits(s.epsilon_bits),
            offset: 0.0,
            occupancy: compile_options(occupancy).occupancy,
        };
        device.runtime().capture(|recording| {
            kernels.norm.norm(
                recording,
                &plan,
                NormBuffers {
                    out: self.output.operand(),
                    x: self.x.operand(),
                    weight: self.weight.operand(),
                    gate: self.gate.operand(),
                },
            )
        })
    }
    pub fn read(&self, device: &CudaDevice, graph: &Graph) -> Result<Vec<f32>> {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        device.runtime().read(&self.output)
    }
}
pub fn bits(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits())
    {
        return Err(Error::invariant(
            "gated norm candidate changed native output bits",
        ));
    }
    Ok(())
}
pub fn measure(
    device: &CudaDevice,
    baseline: &Graph,
    candidate: &Graph,
    occupancy: i32,
    reverse: bool,
    elements: usize,
) -> Result<Trial> {
    let (left, right) = if reverse {
        (candidate, baseline)
    } else {
        (baseline, candidate)
    };
    let measure = |cold| -> Result<Times> {
        let opts = BenchOptions {
            warmup: Duration::from_millis(super::super::WARMUP_MS),
            rep: Duration::from_millis(super::super::REP_MS),
            min_reps: super::super::MIN_REPS,
            max_reps: super::super::MAX_REPS,
            clear_l2: cold,
        };
        let run = |g: &Graph| {
            g.launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        };
        let (a, b) = do_bench_paired(&device.stream, &opts, |_| run(left), |_| run(right))
            .map_err(device_error)?;
        let (a, b) = if reverse { (b, a) } else { (a, b) };
        Ok(Times {
            baseline_ms: a.times_ms().to_vec(),
            candidate_ms: b.times_ms().to_vec(),
        })
    };
    Ok(Trial {
        occupancy,
        candidate_first: reverse,
        verified_elements: elements,
        cold: measure(true)?,
        warm: measure(false)?,
    })
}
