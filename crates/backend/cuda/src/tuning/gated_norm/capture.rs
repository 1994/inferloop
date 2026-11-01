use super::{
    CANDIDATES, SCHEMA, compile_options,
    evidence::{Key, Shape, Times, Trial},
};
use crate::device::{CudaDevice, device_error};
use crate::resident::kernels::aux::norm;
use crate::tuning::preparation::driver_build;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Graph;
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
    x: Tensor<f32>,
    gate: Tensor<f32>,
    weight: Tensor<f32>,
    output: Tensor<f32>,
}
impl Buffers {
    pub fn metadata(device: &CudaDevice, shape: Shape) -> Result<Self> {
        let matrix = [shape.rows, shape.dimension];
        let meta = || {
            api::meta::<f32>(&matrix)
                .sync_on(&device.stream)
                .map_err(device_error)
        };
        Ok(Self {
            shape,
            x: meta()?,
            gate: meta()?,
            output: meta()?,
            weight: api::meta::<f32>(&[shape.dimension])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        })
    }
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
        let matrix = [shape.rows, shape.dimension];
        let unique =
            |value| Arc::try_unwrap(value).map_err(|_| Error::invariant("unique norm fixture"));
        Ok(Self {
            shape,
            x: unique(device.upload(x, &matrix)?)?,
            gate: unique(device.upload(gate, &matrix)?)?,
            weight: unique(device.upload(weight, &[shape.dimension])?)?,
            output: api::zeros::<f32>(&matrix)
                .sync_on(&device.stream)
                .map_err(device_error)?,
        })
    }
    pub fn key(&self, device: &CudaDevice, shape: Shape) -> Result<Key> {
        use cutile::cutile_compiler::cuda_tile_runtime_utils as compiler;
        let x = self
            .x
            .view(&[shape.rows, shape.dimension])
            .map_err(device_error)?;
        let gate = self
            .gate
            .view(&[shape.rows, shape.dimension])
            .map_err(device_error)?;
        let mut output = api::meta::<f32>(&[shape.rows, shape.dimension])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut ids = Vec::new();
        for occupancy in once(0).chain(CANDIDATES) {
            ids.push(
                norm(
                    (&mut output).partition([1, shape.dimension]),
                    &x,
                    &self.weight,
                    &gate,
                    f32::from_bits(shape.epsilon_bits),
                    0.0f32,
                )
                .generics(vec![
                    shape.dimension.to_string(),
                    shape.dimension.to_string(),
                    "1".into(),
                ])
                .compile_options(compile_options(occupancy))
                .l2_cache_key_on(&device.stream)
                .map_err(device_error)?,
            );
        }
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
    pub fn capture(&mut self, device: &CudaDevice, occupancy: i32) -> Result<Graph> {
        let s = self.shape;
        let x = self.x.view(&[s.rows, s.dimension]).map_err(device_error)?;
        let gate = self
            .gate
            .view(&[s.rows, s.dimension])
            .map_err(device_error)?;
        Graph::scope(&device.stream, |scope| {
            scope.record(
                norm(
                    (&mut self.output).partition([1, s.dimension]),
                    &x,
                    &self.weight,
                    &gate,
                    f32::from_bits(s.epsilon_bits),
                    0.0f32,
                )
                .generics(vec![
                    s.dimension.to_string(),
                    s.dimension.to_string(),
                    "1".into(),
                ])
                .compile_options(compile_options(occupancy)),
            )?;
            Ok(())
        })
        .map_err(device_error)
    }
    pub fn read(&self, device: &CudaDevice, graph: &Graph) -> Result<Vec<f32>> {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        device.read_borrowed(&self.output)
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
