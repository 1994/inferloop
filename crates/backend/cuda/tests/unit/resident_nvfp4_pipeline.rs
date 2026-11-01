//! Paired graph replay measurements against the pre-optimization W4A4 main loop.
use super::kernels;
use crate::constants::QUANT_GEMM_TILE;
use crate::device::CudaDevice;
use crate::tuning::resident_gemm::native_tile;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::bench::BenchOptions;
use cutile::bench::do_bench_paired;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error;
use std::time::Duration;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct Inputs {
    shape: [usize; 3],
    activation: Arc<Tensor<f32>>,
    weight: Arc<Tensor<f4e2m1fnx2>>,
    weight_scales: Arc<Tensor<f8e4m3fn>>,
    packed: Tensor<f4e2m1fnx2>,
    activation_scales: Tensor<f8e4m3fn>,
}

impl Inputs {
    fn new(device: &CudaDevice, shape: [usize; 3]) -> TestResult<Self> {
        let [rows, outputs, columns] = shape;
        let activation = (0..rows * columns)
            .map(|i| (f32::from(u16::try_from(i * 17 % 131).unwrap()) - 65.0) / 37.0)
            .collect();
        let weight = (0..outputs * columns / 2)
            .map(|i| f4e2m1fnx2::from_bits(u8::try_from((i * 13 + i / 9) % 256).unwrap()))
            .collect();
        let scales = (0..outputs * columns / 16)
            .map(|i| f8e4m3fn(48 + u8::try_from(i % 16).unwrap()))
            .collect();
        Ok(Self {
            shape,
            activation: device.upload(activation, &[rows, columns])?,
            weight: device.upload(weight, &[outputs, columns / 2])?,
            weight_scales: device.upload(scales, &[outputs, columns / 16])?,
            packed: api::zeros::<f4e2m1fnx2>(&[rows, columns / 2]).sync_on(&device.stream)?,
            activation_scales: api::zeros::<f8e4m3fn>(&[rows, columns / 16])
                .sync_on(&device.stream)?,
        })
    }

    fn record(
        &mut self,
        scope: &Scope,
        output: &mut Tensor<f32>,
        tile: [usize; 2],
        optimized: bool,
    ) -> Result<(), DeviceError> {
        let [rows, _, columns] = self.shape;
        scope.record(
            kernels::quantize(
                (&mut self.packed).partition([1, 256]),
                (&mut self.activation_scales).partition([1, 32]),
                &self.activation,
                1.0,
            )
            .generics(vec![columns.to_string()]),
        )?;
        let generics = vec![
            columns.to_string(),
            tile[0].to_string(),
            tile[1].to_string(),
        ];
        if optimized && rows > QUANT_GEMM_TILE[0] && tile[1] > QUANT_GEMM_TILE[1] {
            scope.record(
                kernels::packed_prefill(
                    output.partition(tile),
                    &self.packed,
                    &self.activation_scales,
                    &self.weight,
                    &self.weight_scales,
                    1.0,
                )
                .generics(generics),
            )?;
        } else {
            scope.record(
                baseline::packed(
                    output.partition(tile),
                    &self.packed,
                    &self.activation_scales,
                    &self.weight,
                    &self.weight_scales,
                    1.0,
                )
                .generics(generics),
            )?;
        }
        Ok(())
    }
}

#[test]
#[ignore = "requires CUDA hardware; run release inside safe-run"]
fn nvfp4_prefill_performance_comparison() -> TestResult {
    if cfg!(debug_assertions) {
        return Err("performance measurements require cargo test --release".into());
    }
    let compile = CompileOptions::default();
    if compile.device_debug || compile.opt_level.is_some_and(|level| level != 3) {
        return Err("performance measurements require CUDA JIT O3 without device debug".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    println!(
        "{}",
        serde_json::json!({
            "device": device.profile()?, "build_profile": "release", "cuda_jit_opt_level": 3,
            "device_debug": false, "scope": "preallocated graph replay: quantization plus GEMM"
        })
    );
    for round in 0..3 {
        for clear_l2 in [true, false] {
            let options = BenchOptions {
                warmup: Duration::from_millis(50),
                rep: Duration::from_millis(100),
                min_reps: 30,
                max_reps: 100,
                clear_l2,
            };
            for rows in [1, 4, 16, 32, 64, 128, 256] {
                for (outputs, columns) in [(17408, 5120), (5120, 5120), (5120, 17408)] {
                    compare_case(&device, &options, [rows, outputs, columns], round)?;
                }
            }
        }
    }
    Ok(())
}

fn compare_case(
    device: &CudaDevice,
    options: &BenchOptions,
    shape: [usize; 3],
    round: usize,
) -> TestResult {
    let [rows, outputs, _] = shape;
    let mut inputs = Inputs::new(device, shape)?;
    let mut original_output = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let mut optimized_output = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let original_tile = native_tile(rows);
    // Sweep the supported wider-column candidate; production choice requires measured proof.
    let optimized_tile = [original_tile[0], original_tile[1] * 2];
    let original = Graph::scope(&device.stream, |scope| {
        inputs.record(scope, &mut original_output, original_tile, false)
    })?;
    let optimized = Graph::scope(&device.stream, |scope| {
        inputs.record(scope, &mut optimized_output, optimized_tile, true)
    })?;
    original.launch().sync_on(&device.stream)?;
    optimized.launch().sync_on(&device.stream)?;
    device.reclaim_barrier()?;
    // Downloads borrow these owners; graph replays continue writing the same live allocations.
    let original_output = Arc::new(original_output);
    let optimized_output = Arc::new(optimized_output);
    let expected = (&original_output).to_host_vec().sync_on(&device.stream)?;
    let actual = (&optimized_output).to_host_vec().sync_on(&device.stream)?;
    for (index, (&value, &reference)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            value.is_finite() && (value - reference).abs() <= 5e-4 * reference.abs().max(1.0),
            "{shape:?} at {index}: {value} != {reference}"
        );
    }
    let max_absolute_error = actual
        .iter()
        .zip(&expected)
        .map(|(value, reference)| (value - reference).abs())
        .fold(0.0_f32, f32::max);
    let all_outputs_identical = actual
        .iter()
        .zip(&expected)
        .all(|(value, reference)| value.to_bits() == reference.to_bits());
    let (base, candidate) = do_bench_paired(
        &device.stream,
        options,
        |_| {
            original
                .launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
        |_| {
            optimized
                .launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
    )?;
    println!(
        "{}",
        serde_json::json!({
            "shape": shape, "round": round, "original_tile": original_tile,
            "optimized_tile": optimized_tile, "clear_l2": options.clear_l2,
            "baseline_ms": base.median_ms(), "candidate_ms": candidate.median_ms(),
            "speedup": base.median_ms() / candidate.median_ms(),
        "max_absolute_error": max_absolute_error, "all_outputs_identical": all_outputs_identical,
            "baseline_p95_ms": base.quantile_ms(0.95), "candidate_p95_ms": candidate.quantile_ms(0.95),
            "baseline_samples_ms": base.times_ms(), "candidate_samples_ms": candidate.times_ms(),
        })
    );
    Ok(())
}
#[cutile::module]
mod baseline {
    use cutile::core::{
        BroadcastScalar, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_2, UnpackF4e2m1fnx2Tile,
        constant, f4e2m1fnx2, f8e4m3fn, get_tile_block_id, mmaf_scaled,
    };
    #[cutile::entry()]
    pub fn packed<const K: i32, const M: i32, const N: i32>(
        out: &mut Tensor<f32, { [M, N] }>,
        input: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        input_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        weight: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![M, 128]);
        let xs = input_scales.partition(shape![M, 16]);
        let wp = weight.partition(shape![N, 128]);
        let sp = scales.partition(shape![N, 16]);
        let mut acc: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        for k in 0i32..((K + 255) / 256) {
            let x = xp.load([pid.0, k]).unpack(shape![M, 256]);
            let w = wp.load([pid.1, k]).unpack(shape![N, 256]).transpose();
            acc = mmaf_scaled(
                x,
                w,
                acc,
                xs.load([pid.0, k]),
                sp.load([pid.1, k]).transpose(),
            );
        }
        out.store(acc * alpha.broadcast(shape![M, N]));
    }
}
