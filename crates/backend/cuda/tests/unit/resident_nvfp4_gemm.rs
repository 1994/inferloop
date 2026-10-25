use super::kernels;
use crate::device::CudaDevice;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::bench::BenchOptions as CutileBenchOptions;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::prelude::*;
use std::error::Error as StdError;
use std::sync::Arc;

/// Tile these tests construct their output partition with; the GEMM's `M` and `N` generics and
/// the partition shape must agree. They stay on the narrow tile the correctness cases were
/// written against; the shipped tile is exercised end to end by the serving benchmark.
const QUANT_TILE_ROWS: usize = 16;
const QUANT_TILE_COLUMNS: usize = 64;

fn fp8(code: u8) -> f32 {
    let exponent = code >> 3;
    let mantissa = f32::from(code & 7);
    if exponent == 0 {
        mantissa / 512.0
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    }
}
fn nearest(value: f32, table: &[f32]) -> usize {
    (0..table.len())
        .min_by(|&a, &b| {
            (table[a] - value)
                .abs()
                .total_cmp(&(table[b] - value).abs())
                .then((a % 2).cmp(&(b % 2)))
        })
        .expect("nonempty codebook")
}

#[test]
#[ignore = "requires a CUDA device; run inside safe-run"]
fn nvfp4_mma_matches_independent_block_quantization() -> Result<(), Box<dyn StdError>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let f4 = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let f8: Vec<_> = (0..=126).map(fp8).collect();
    for (m, n, k, global) in [
        (1, 65, 144, 1.0_f32),
        (3, 64, 256, 16.0),
        (17, 129, 128, 0.25),
        (32, 64, 512, 1024.0),
        (65, 129, 784, 1.0),
    ] {
        let input: Vec<_> = (0..m * k)
            .map(|i| {
                if i < 16 {
                    0.0
                } else {
                    (f32::from(u8::try_from(i * 17 % 131).expect("bounded value")) - 65.0) / 37.0
                }
            })
            .collect();
        let codes: Vec<u8> = (0..n * k)
            .map(|i| u8::try_from((i * 7 + i / 9) % 16).expect("FP4 code"))
            .collect();
        let packed: Vec<_> = codes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f4e2m1fnx2::from_bits(c[0] | c[1] << 4))
            .collect();
        let scale_codes: Vec<u8> = (0..n * k / 16)
            .map(|i| u8::try_from(48 + i % 16).expect("FP8 code"))
            .collect();
        let scales: Vec<_> = scale_codes.iter().copied().map(f8e4m3fn).collect();
        let x = device.upload(input.clone(), &[m, k])?;
        let w = device.upload(packed, &[n, k / 2])?;
        let s = device.upload(scales, &[n, k / 16])?;
        let mut quantized_input = api::zeros::<f4e2m1fnx2>(&[m, k / 2]).sync_on(&device.stream)?;
        let mut input_scales = api::zeros::<f8e4m3fn>(&[m, k / 16]).sync_on(&device.stream)?;
        kernels::quantize(
            (&mut quantized_input).partition([1, 256]),
            (&mut input_scales).partition([1, 32]),
            &x,
            global,
        )
        .generics(vec![k.to_string()])
        .sync_on(&device.stream)?;
        let mut quantized = input.clone();
        for values in quantized.as_chunks_mut::<16>().0 {
            let maximum = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let scale = f8[nearest((maximum * global / 6.0).min(448.0), &f8)];
            for value in values {
                *value = if scale == 0.0 {
                    0.0
                } else {
                    value.signum()
                        * f4[nearest((value.abs() * global / scale).min(6.0), &f4)]
                        * scale
                        / global
                };
            }
        }
        for tile in [[16, 64], [64, 64], [64, 128]] {
            let generics = vec![k.to_string(), tile[0].to_string(), tile[1].to_string()];
            let output = if tile == [16, 64] {
                kernels::packed(
                    api::zeros::<f32>(&[m, n]).partition(tile),
                    &quantized_input,
                    &input_scales,
                    &w,
                    &s,
                    1.0 / global,
                )
                .generics(generics)
                .first()
                .unpartition()
                .sync_on(&device.stream)?
            } else {
                kernels::packed_prefill(
                    api::zeros::<f32>(&[m, n]).partition(tile),
                    &quantized_input,
                    &input_scales,
                    &w,
                    &s,
                    1.0 / global,
                )
                .generics(generics)
                .first()
                .unpartition()
                .sync_on(&device.stream)?
            };
            let output = output.to_host_vec().sync_on(&device.stream)?;
            check_product((m, n, k), &codes, &scale_codes, &quantized, &output);
        }
    }
    Ok(())
}

/// Fixture rows, input columns, MLP hidden columns and prompt tile rows.
const SWIGLU_FIXTURE: (usize, usize, usize, usize) = (128, 768, 256, 64);
/// Activation, gate, up and down global scales in the checkpoint's range.
const SWIGLU_GLOBALS: (f32, f32, f32, f32) = (836.0, 6400.0, 5120.0, 161.0);
type Fp4Operand = (Tensor<f4e2m1fnx2>, Tensor<f8e4m3fn>);
type Fp4Weight = (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>);

fn swiglu_weight(device: &CudaDevice, seed: usize) -> Result<Fp4Weight, Box<dyn StdError>> {
    let (_, columns, hidden, _) = SWIGLU_FIXTURE;
    let packed: Vec<_> = (0..hidden * columns / 2)
        .map(|i| {
            let low = u8::try_from((i * 7 + seed) % 16).expect("FP4 code");
            let high = u8::try_from((i * 11 + seed * 3 + i / 5) % 16).expect("FP4 code");
            f4e2m1fnx2::from_bits(low | high << 4)
        })
        .collect();
    let scales: Vec<_> = (0..hidden * columns / 16)
        .map(|i| f8e4m3fn(u8::try_from(40 + (i + seed) % 24).expect("FP8 code")))
        .collect();
    Ok((
        device.upload(packed, &[hidden, columns / 2])?,
        device.upload(scales, &[hidden, columns / 16])?,
    ))
}

/// Shipped unfused path: gate and up F32 projections, then `swiglu_quantize`.
fn separate_swiglu(
    device: &CudaDevice,
    (q, qs): (&Tensor<f4e2m1fnx2>, &Tensor<f8e4m3fn>),
    gate: &Fp4Weight,
    up: &Fp4Weight,
) -> Result<Fp4Operand, Box<dyn StdError>> {
    let (rows, columns, hidden, _) = SWIGLU_FIXTURE;
    let (input_global, gate_global, up_global, down_global) = SWIGLU_GLOBALS;
    let project = |(codes, scales): &Fp4Weight, global: f32| {
        kernels::packed_prefill(
            api::zeros::<f32>(&[rows, hidden]).partition([128, 128]),
            q,
            qs,
            codes,
            scales,
            1.0 / (global * input_global),
        )
        .generics(vec![columns.to_string(), "128".into(), "128".into()])
        .first()
        .unpartition()
        .sync_on(&device.stream)
    };
    let gate = project(gate, gate_global)?;
    let up = project(up, up_global)?;
    let mut codes = api::zeros::<f4e2m1fnx2>(&[rows, hidden / 2]).sync_on(&device.stream)?;
    let mut scales = api::zeros::<f8e4m3fn>(&[rows, hidden / 16]).sync_on(&device.stream)?;
    kernels::swiglu_quantize(
        (&mut codes).partition([16, 256]),
        (&mut scales).partition([16, 32]),
        &gate,
        &up,
        down_global,
    )
    .generics(vec![hidden.to_string()])
    .sync_on(&device.stream)?;
    Ok((codes, scales))
}

fn operand_bytes(
    device: &CudaDevice,
    (codes, scales): Fp4Operand,
) -> Result<(Vec<u8>, Vec<u8>), Box<dyn StdError>> {
    let codes = codes.to_host_vec().sync_on(&device.stream)?;
    let scales = scales.to_host_vec().sync_on(&device.stream)?;
    Ok((
        codes.into_iter().map(f4e2m1fnx2::to_bits).collect(),
        scales.into_iter().map(|code| code.0).collect(),
    ))
}

#[test]
#[ignore = "requires a CUDA device; run inside safe-run"]
fn fused_gate_up_swiglu_matches_separate_projections() -> Result<(), Box<dyn StdError>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (rows, columns, hidden, tile_rows) = SWIGLU_FIXTURE;
    let (input_global, gate_global, up_global, down_global) = SWIGLU_GLOBALS;
    let input: Vec<_> = (0..rows * columns)
        .map(|i| (f32::from(u8::try_from(i * 29 % 251).expect("bounded")) - 125.0) / 97.0)
        .collect();
    let gate = swiglu_weight(&device, 1)?;
    let up = swiglu_weight(&device, 5)?;
    let x = device.upload(input, &[rows, columns])?;
    let mut q = api::zeros::<f4e2m1fnx2>(&[rows, columns / 2]).sync_on(&device.stream)?;
    let mut qs = api::zeros::<f8e4m3fn>(&[rows, columns / 16]).sync_on(&device.stream)?;
    kernels::quantize_rows(
        (&mut q).partition([16, 256]),
        (&mut qs).partition([16, 32]),
        &x,
        input_global,
    )
    .generics(vec![columns.to_string()])
    .sync_on(&device.stream)?;
    let expected = operand_bytes(&device, separate_swiglu(&device, (&q, &qs), &gate, &up)?)?;
    let mut codes = api::zeros::<f4e2m1fnx2>(&[rows, hidden / 2]).sync_on(&device.stream)?;
    let mut scales = api::zeros::<f8e4m3fn>(&[rows, hidden / 16]).sync_on(&device.stream)?;
    kernels::packed_swiglu(
        (&mut codes).partition([tile_rows, 32]),
        (&mut scales).partition([tile_rows, 4]),
        &q,
        &qs,
        &gate.0,
        &gate.1,
        &up.0,
        &up.1,
        1.0 / (gate_global * input_global),
        1.0 / (up_global * input_global),
        down_global,
    )
    .generics(
        [columns, tile_rows, 64, 32, 4]
            .iter()
            .map(ToString::to_string)
            .collect(),
    )
    .sync_on(&device.stream)?;
    assert!(
        expected.0.iter().any(|&code| code != 0),
        "fixture must produce nonzero codes"
    );
    assert_eq!(operand_bytes(&device, (codes, scales))?, expected);
    Ok(())
}

fn check_product(
    (m, n, k): (usize, usize, usize),
    codes: &[u8],
    scale_codes: &[u8],
    quantized: &[f32],
    output: &[f32],
) {
    let f4 = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for row in 0..m {
        for col in 0..n {
            let expected: f64 = (0..k)
                .map(|i| {
                    let code = codes[col * k + i];
                    let weight = f4[usize::from(code & 7)] * if code & 8 == 0 { 1.0 } else { -1.0 };
                    f64::from(quantized[row * k + i])
                        * f64::from(weight * fp8(scale_codes[col * k / 16 + i / 16]))
                })
                .sum();
            let actual = f64::from(output[row * n + col]);
            assert!(
                (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "{m}x{n}x{k} [{row},{col}]: {actual} != {expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA device; run inside safe-run"]
fn nvfp4_quantization_split_performance() -> Result<(), Box<dyn StdError>> {
    use cutile::bench::{BenchOptions, do_bench_paired};
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err("performance measurements require cargo test --release".into());
    }
    let compile = CompileOptions::default();
    if compile.device_debug || compile.opt_level.is_some_and(|level| level != 3) {
        return Err("performance measurements require CUDA JIT O3 without device debug".into());
    }
    println!("build_profile=release cuda_jit_opt_level=3 device_debug=false");
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(100),
        min_reps: 10,
        max_reps: 100,
        clear_l2: true,
    };
    for (n, k) in [(14336, 4096), (4096, 14336)] {
        let weights = device.upload(vec![f4e2m1fnx2::from_bits(0x21); n * k / 2], &[n, k / 2])?;
        let scales = device.upload(vec![f8e4m3fn(56); n * k / 16], &[n, k / 16])?;
        for m in [1, 3, 32] {
            let input = device.upload(vec![0.25_f32; m * k], &[m, k])?;
            let mut quantized = api::zeros::<f4e2m1fnx2>(&[m, k / 2]).sync_on(&device.stream)?;
            let mut input_scales = api::zeros::<f8e4m3fn>(&[m, k / 16]).sync_on(&device.stream)?;
            let mut output = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
            let fused = CudaGraph::scope(&device.stream, |scope| {
                scope.record(
                    kernels::matmul(
                        (&mut output).partition([16, 64]),
                        &input,
                        &weights,
                        &scales,
                        1.0,
                        1.0,
                    )
                    .generics(vec![k.to_string()]),
                )?;
                Ok(())
            })?;
            let separated = CudaGraph::scope(&device.stream, |scope| {
                scope.record(
                    kernels::quantize(
                        (&mut quantized).partition([1, 256]),
                        (&mut input_scales).partition([1, 32]),
                        &input,
                        1.0,
                    )
                    .generics(vec![k.to_string()]),
                )?;
                scope.record(
                    kernels::packed(
                        (&mut output).partition([16, 64]),
                        &quantized,
                        &input_scales,
                        &weights,
                        &scales,
                        1.0,
                    )
                    .generics(vec![
                        k.to_string(),
                        QUANT_TILE_ROWS.to_string(),
                        QUANT_TILE_COLUMNS.to_string(),
                    ]),
                )?;
                Ok(())
            })?;
            let (baseline, candidate) = do_bench_paired(
                &device.stream,
                &options,
                |_| {
                    fused
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
                |_| {
                    separated
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
            )?;
            println!(
                "{m}x{n}x{k}: fused_ms={} separated_ms={} speedup={}",
                baseline.median_ms(),
                candidate.median_ms(),
                baseline.median_ms() / candidate.median_ms()
            );
        }
    }
    Ok(())
}

/// Rows every sweep variant is benchmarked against, so all tiles read one allocation size.
const SWEEP_PAD_ROWS: usize = 64;
/// Production projection shapes the sweep covers: the 27B MLP up/gate and down projections at
/// single-token, verification and prompt widths.
const SWEEP_SHAPES: [(usize, usize, usize); 8] = [
    (1, 17408, 5120),
    (1, 5120, 5120),
    (1, 5120, 17408),
    (4, 17408, 5120),
    (4, 5120, 17408),
    (12, 17408, 5120),
    (12, 5120, 17408),
    (64, 17408, 5120),
];
/// Output tiles the sweep compares against the shipped `[16, 64]`.
const SWEEP_TILES: [(usize, usize); 4] = [(16, 64), (16, 128), (64, 128), (64, 64)];

/// Tile sweep for the production W4A4 kernel: does a wider output column tile or a taller row
/// tile lift the weight bandwidth the prompt and slot-verify replays are bound by?
///
/// Measured per shape against the shipped `[16, 64]` tile with paired alternation, so a
/// throttled host cannot flip the ranking. Informational: it prints, it does not assert.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn nvfp4_packed_tile_sweep() -> Result<(), Box<dyn StdError>> {
    use cutile::bench::BenchOptions;
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err("performance measurements require cargo test --release".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(100),
        min_reps: 10,
        max_reps: 100,
        clear_l2: true,
    };
    for (m, n, k) in SWEEP_SHAPES {
        sweep_shape(&device, &options, (m, n, k))?;
    }
    Ok(())
}

/// One shape's tile comparison, alternating baseline and candidate under `do_bench_paired`.
fn sweep_shape(
    device: &CudaDevice,
    options: &CutileBenchOptions,
    (m, n, k): (usize, usize, usize),
) -> Result<(), Box<dyn StdError>> {
    use cutile::bench::do_bench_paired;
    assert!(m <= SWEEP_PAD_ROWS);
    let input = device.upload(
        vec![f4e2m1fnx2::from_bits(0x21); SWEEP_PAD_ROWS * k / 2],
        &[SWEEP_PAD_ROWS, k / 2],
    )?;
    let input_scales = device.upload(
        vec![f8e4m3fn(48); SWEEP_PAD_ROWS * k / 16],
        &[SWEEP_PAD_ROWS, k / 16],
    )?;
    let weight = device.upload(vec![f4e2m1fnx2::from_bits(0x21); n * k / 2], &[n, k / 2])?;
    let scales = device.upload(vec![f8e4m3fn(48); n * k / 16], &[n, k / 16])?;
    let mut output = api::zeros::<f32>(&[SWEEP_PAD_ROWS, n]).sync_on(&device.stream)?;
    let mut tile_graph = |rows: usize, columns: usize| -> Result<CudaGraph<()>, Error> {
        Ok(CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                kernels::packed(
                    (&mut output).partition([rows, columns]),
                    &input,
                    &input_scales,
                    &weight,
                    &scales,
                    1.0,
                )
                .generics(vec![
                    k.to_string(),
                    rows.to_string(),
                    columns.to_string(),
                ]),
            )?;
            Ok(())
        })?)
    };
    let baseline = tile_graph(QUANT_TILE_ROWS, QUANT_TILE_COLUMNS)?;
    for (rows, columns) in SWEEP_TILES {
        let candidate = tile_graph(rows, columns)?;
        let (base, cand) = do_bench_paired(
            &device.stream,
            options,
            |_| {
                baseline
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
            |_| {
                candidate
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
        )?;
        println!(
            "m{m} n{n} k{k} tile[{rows},{columns}]: base_ms={} cand_ms={} ratio={}",
            base.median_ms(),
            cand.median_ms(),
            cand.median_ms() / base.median_ms()
        );
    }
    Ok(())
}

/// Prompt-width gated MLP shape of the 27B checkpoint: rows, input columns, hidden columns.
const SWIGLU_PERF_SHAPE: (usize, usize, usize) = (1024, 5120, 17408);

/// Gate and up projection outputs plus the down projection's codes and scales.
struct SwigluBuffers {
    gate: Tensor<f32>,
    up: Tensor<f32>,
    codes: Tensor<f4e2m1fnx2>,
    scales: Tensor<f8e4m3fn>,
}

fn separated_swiglu_graph(
    device: &CudaDevice,
    (q, qs): (&Tensor<f4e2m1fnx2>, &Tensor<f8e4m3fn>),
    (gate, up): (&Fp4Weight, &Fp4Weight),
    buffers: &mut SwigluBuffers,
) -> Result<CudaGraph<()>, Box<dyn StdError>> {
    let (_, columns, hidden) = SWIGLU_PERF_SHAPE;
    Ok(CudaGraph::scope(&device.stream, |scope| {
        for (out, (w, s)) in [(&mut buffers.gate, gate), (&mut buffers.up, up)] {
            scope.record(
                kernels::packed_prefill(out.partition([128, 128]), q, qs, w, s, 1.0e-6)
                    .generics(vec![columns.to_string(), "128".into(), "128".into()]),
            )?;
        }
        scope.record(
            kernels::swiglu_quantize(
                (&mut buffers.codes).partition([16, 256]),
                (&mut buffers.scales).partition([16, 32]),
                &buffers.gate,
                &buffers.up,
                161.0,
            )
            .generics(vec![hidden.to_string()]),
        )?;
        Ok(())
    })?)
}

fn fused_swiglu_graph(
    device: &CudaDevice,
    (q, qs): (&Tensor<f4e2m1fnx2>, &Tensor<f8e4m3fn>),
    (gate, up): (&Fp4Weight, &Fp4Weight),
    buffers: &mut SwigluBuffers,
    [tile_rows, tile_columns]: [usize; 2],
) -> Result<CudaGraph<()>, Box<dyn StdError>> {
    let (_, columns, _) = SWIGLU_PERF_SHAPE;
    Ok(CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::packed_swiglu(
                (&mut buffers.codes).partition([tile_rows, tile_columns / 2]),
                (&mut buffers.scales).partition([tile_rows, tile_columns / 16]),
                q,
                qs,
                &gate.0,
                &gate.1,
                &up.0,
                &up.1,
                1.0e-6,
                1.0e-6,
                161.0,
            )
            .generics(
                [
                    columns,
                    tile_rows,
                    tile_columns,
                    tile_columns / 2,
                    tile_columns / 16,
                ]
                .iter()
                .map(ToString::to_string)
                .collect(),
            ),
        )?;
        Ok(())
    })?)
}

#[test]
#[ignore = "requires a CUDA device; run inside safe-run"]
fn fused_gate_up_swiglu_performance() -> Result<(), Box<dyn StdError>> {
    use cutile::bench::{BenchOptions, do_bench_paired};
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err("performance measurements require cargo test --release".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(200),
        min_reps: 10,
        max_reps: 100,
        clear_l2: true,
    };
    let (rows, columns, hidden) = SWIGLU_PERF_SHAPE;
    let weight = |seed: u8| -> Result<Fp4Weight, Box<dyn StdError>> {
        Ok((
            device.upload(
                vec![f4e2m1fnx2::from_bits(0x21 ^ seed); hidden * columns / 2],
                &[hidden, columns / 2],
            )?,
            device.upload(
                vec![f8e4m3fn(56); hidden * columns / 16],
                &[hidden, columns / 16],
            )?,
        ))
    };
    let weights = (weight(0)?, weight(3)?);
    let q = device.upload(
        vec![f4e2m1fnx2::from_bits(0x13); rows * columns / 2],
        &[rows, columns / 2],
    )?;
    let qs = device.upload(
        vec![f8e4m3fn(56); rows * columns / 16],
        &[rows, columns / 16],
    )?;
    let mut buffers = SwigluBuffers {
        gate: api::zeros::<f32>(&[rows, hidden]).sync_on(&device.stream)?,
        up: api::zeros::<f32>(&[rows, hidden]).sync_on(&device.stream)?,
        codes: api::zeros::<f4e2m1fnx2>(&[rows, hidden / 2]).sync_on(&device.stream)?,
        scales: api::zeros::<f8e4m3fn>(&[rows, hidden / 16]).sync_on(&device.stream)?,
    };
    let inputs = (&*q, &*qs);
    let separated =
        separated_swiglu_graph(&device, inputs, (&weights.0, &weights.1), &mut buffers)?;
    for tile in [[128, 64], [64, 64], [128, 32], [64, 32]] {
        let fused = fused_swiglu_graph(
            &device,
            inputs,
            (&weights.0, &weights.1),
            &mut buffers,
            tile,
        )?;
        let (fused_time, separated_time) = do_bench_paired(
            &device.stream,
            &options,
            |_| {
                fused
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
            |_| {
                separated
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
        )?;
        println!(
            "tile={}x{}: fused_ms={} separated_ms={}",
            tile[0],
            tile[1],
            fused_time.median_ms(),
            separated_time.median_ms()
        );
    }
    Ok(())
}

/// Narrow verification shapes of the 27B MLP: rows (16 slots x 3 lanes), outputs, reduction,
/// and the production tile.
const SPLIT_SHAPES: [(usize, usize, usize, [usize; 2]); 2] =
    [(48, 17408, 5120, [64, 128]), (48, 5120, 17408, [64, 64])];

fn split_graph(
    device: &CudaDevice,
    operands: (&Tensor<f4e2m1fnx2>, &Tensor<f8e4m3fn>, &Fp4Weight),
    (out, partials): (&mut Tensor<f32>, &mut Tensor<f32>),
    (reduction, tile, splits): (usize, [usize; 2], usize),
) -> Result<CudaGraph<()>, Box<dyn StdError>> {
    let (codes, code_scales, (weight, weight_scales)) = operands;
    let rows = usize::try_from(out.shape()[0])?;
    let columns = usize::try_from(out.shape()[1])?;
    let padded = rows.div_ceil(tile[0]) * tile[0];
    let grid = |value: usize| u32::try_from(value).map_err(|e| tensor_error(&e.to_string()));
    Ok(CudaGraph::scope(&device.stream, |scope| {
        if splits == 1 {
            scope.record(
                kernels::packed(
                    out.partition(tile),
                    codes,
                    code_scales,
                    weight,
                    weight_scales,
                    1.0e-6,
                )
                .generics(vec![
                    reduction.to_string(),
                    tile[0].to_string(),
                    tile[1].to_string(),
                ]),
            )?;
            return Ok(());
        }
        scope.record(
            kernels::packed_split(
                (&mut *partials).partition([splits, padded, columns]),
                codes,
                code_scales,
                weight,
                weight_scales,
            )
            .generics(
                [reduction, tile[0], tile[1], splits, padded, columns]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .grid((
                grid(padded / tile[0])?,
                grid(columns / tile[1])?,
                grid(splits)?,
            )),
        )?;
        scope.record(
            kernels::sum_splits(out.partition(tile), &*partials, 1.0e-6).generics(
                [tile[0], tile[1], splits]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            ),
        )?;
        Ok(())
    })?)
}

#[test]
#[ignore = "requires a CUDA device; run inside safe-run"]
fn narrow_split_k_performance() -> Result<(), Box<dyn StdError>> {
    use cutile::bench::{BenchOptions, do_bench_paired};
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err("performance measurements require cargo test --release".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(200),
        min_reps: 10,
        max_reps: 200,
        clear_l2: true,
    };
    for (rows, columns, reduction, tile) in SPLIT_SHAPES {
        let weight: Fp4Weight = (
            device.upload(
                vec![f4e2m1fnx2::from_bits(0x21); columns * reduction / 2],
                &[columns, reduction / 2],
            )?,
            device.upload(
                vec![f8e4m3fn(56); columns * reduction / 16],
                &[columns, reduction / 16],
            )?,
        );
        let codes = device.upload(
            vec![f4e2m1fnx2::from_bits(0x13); rows * reduction / 2],
            &[rows, reduction / 2],
        )?;
        let code_scales = device.upload(
            vec![f8e4m3fn(56); rows * reduction / 16],
            &[rows, reduction / 16],
        )?;
        let operands = (&*codes, &*code_scales, &weight);
        let padded = rows.div_ceil(tile[0]) * tile[0];
        let mut base_out = api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?;
        let mut split_out = api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?;
        let mut unused = api::zeros::<f32>(&[1]).sync_on(&device.stream)?;
        let base = split_graph(
            &device,
            operands,
            (&mut base_out, &mut unused),
            (reduction, tile, 1),
        )?;
        for splits in [2, 4, 8] {
            if !(reduction / 256).is_multiple_of(splits) {
                continue;
            }
            let mut partials =
                api::zeros::<f32>(&[splits, padded, columns]).sync_on(&device.stream)?;
            let split = split_graph(
                &device,
                operands,
                (&mut split_out, &mut partials),
                (reduction, tile, splits),
            )?;
            let (split_time, base_time) = do_bench_paired(
                &device.stream,
                &options,
                |_| {
                    split
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
                |_| {
                    base.launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
            )?;
            let bytes = f64::from(u32::try_from(columns * reduction)?) * 0.5625;
            println!(
                "{rows}x{columns}x{reduction} tile={tile:?} splits={splits}: split_ms={} ({:.2} TB/s) base_ms={} ({:.2} TB/s)",
                split_time.median_ms(),
                bytes / (f64::from(split_time.median_ms()) * 1e9),
                base_time.median_ms(),
                bytes / (f64::from(base_time.median_ms()) * 1e9),
            );
        }
    }
    Ok(())
}
