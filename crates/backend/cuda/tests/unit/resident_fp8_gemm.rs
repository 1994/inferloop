use super::kernels;
use crate::device::CudaDevice;
use cuda_core::f8e4m3fn;
use cutile::error::tensor_error;
use cutile::prelude::*;
use std::error::Error;

/// Tile these tests construct their output partition with; the GEMM's `M` and `N` generics and
/// the partition shape must agree.
const QUANT_TILE_ROWS: usize = 16;
const QUANT_TILE_COLUMNS: usize = 64;

/// Generics of the narrow tile above, so each call site stays one line.
fn tile_generics(k: usize) -> Vec<String> {
    vec![
        k.to_string(),
        QUANT_TILE_ROWS.to_string(),
        QUANT_TILE_COLUMNS.to_string(),
    ]
}

fn decode(code: u8) -> f32 {
    let magnitude = code & 127;
    let exponent = magnitude >> 3;
    let mantissa = f32::from(magnitude & 7);
    let value = if exponent == 0 {
        mantissa / 512.0
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if code & 128 == 0 { value } else { -value }
}
fn quantize(value: f32) -> u8 {
    let value_abs = value.abs().min(448.0);
    let magnitude = (0..=126u8)
        .min_by(|&a, &b| {
            (decode(a) - value_abs)
                .abs()
                .total_cmp(&(decode(b) - value_abs).abs())
                .then((a % 2).cmp(&(b % 2)))
        })
        .unwrap();
    magnitude | if value.is_sign_negative() { 128 } else { 0 }
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn fp8_token_mma_matches_independent_quantization() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (m, n, k) in [
        (1, 65, 144),
        (3, 64, 256),
        (12, 129, 512),
        (32, 64, 1024),
        (1, 65, 17408),
        (3, 64, 5120),
    ] {
        let input: Vec<f32> = (0..m * k)
            .map(|i| {
                if m > 1 && i < k {
                    0.0
                } else {
                    (f32::from(u16::try_from(i * 17 % 131).unwrap()) - 65.0) / 37.0
                }
            })
            .collect();
        let codes: Vec<u8> = (0..n * k)
            .map(|i| 48 + u8::try_from(i % 16).unwrap() + if i % 3 == 0 { 128 } else { 0 })
            .collect();
        let scales: Vec<f32> = (0..n)
            .map(|i| f32::from(u16::try_from(i % 13 + 1).unwrap()) / 11.0)
            .collect();
        let x = device.upload(input.clone(), &[m, k])?;
        let w = device.upload(codes.iter().copied().map(f8e4m3fn).collect(), &[n, k])?;
        let ws = device.upload(scales.clone(), &[n])?;
        let mut q = api::zeros::<f8e4m3fn>(&[m, k]).sync_on(&device.stream)?;
        let mut qs = api::zeros::<f32>(&[m, 1]).sync_on(&device.stream)?;
        kernels::quantize(
            (&mut q).partition([1, k.next_power_of_two()]),
            (&mut qs).partition([1, 1]),
            &x,
        )
        .generics(vec![k.to_string(), k.next_power_of_two().to_string()])
        .sync_on(&device.stream)?;
        let output = kernels::matmul(
            api::zeros::<f32>(&[m, n]).partition([16, 64]),
            &q,
            &w,
            &qs,
            &ws,
        )
        .generics(tile_generics(k))
        .first()
        .unpartition()
        .sync_on(&device.stream)?;
        let actual = output.to_host_vec().sync_on(&device.stream)?;
        let encoded = q.to_host_vec().sync_on(&device.stream)?;
        for row in 0..m {
            let scale = (input[row * k..(row + 1) * k]
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0, f32::max)
                / 448.0)
                .max(1e-12);
            let quantized: Vec<_> = input[row * k..(row + 1) * k]
                .iter()
                .map(|x| quantize(*x / scale))
                .collect();
            for col in 0..k {
                assert_eq!(
                    encoded[row * k + col].0,
                    quantized[col],
                    "row={row} k={col}"
                );
            }
            for col in 0..n {
                let expected = (0..k)
                    .map(|i| {
                        f64::from(decode(quantized[i])) * f64::from(decode(codes[col * k + i]))
                    })
                    .sum::<f64>()
                    * f64::from(scale)
                    * f64::from(scales[col]);
                assert!(
                    (f64::from(actual[row * n + col]) - expected).abs()
                        < expected.abs().mul_add(0.00002, 0.0002),
                    "{m}x{n}x{k} [{row},{col}] {} != {expected}",
                    actual[row * n + col]
                );
            }
        }
    }
    Ok(())
}

/// Block-scaled FP8 must reproduce an independent per-128-block quantization and scaling:
/// the activation scale is indexed per input block and the weight scale per (channel block,
/// input block), unlike the per-channel path the other tests cover.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn fp8_block_mma_matches_independent_block_quantization() -> Result<(), Box<dyn Error>> {
    const BLOCK: usize = 128;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (m, n, k) in [(12, 65, 512), (64, 129, 1024), (12, 256, 5120)] {
        let input: Vec<f32> = (0..m * k)
            .map(|i| (f32::from(u16::try_from(i * 17 % 131).unwrap()) - 65.0) / 37.0)
            .collect();
        let codes: Vec<u8> = (0..n * k)
            .map(|i| 48 + u8::try_from(i % 16).unwrap() + if i % 3 == 0 { 128 } else { 0 })
            .collect();
        let blocks_k = k / BLOCK;
        let blocks_n = n.div_ceil(BLOCK);
        let blocks: Vec<f32> = (0..blocks_n * blocks_k)
            .map(|i| f32::from(u16::try_from(i % 13 + 1).unwrap()) / 11.0)
            .collect();
        // The kernel reads one scale per output channel: every row of a channel block shares it.
        let mut expanded = Vec::with_capacity(n * blocks_k);
        for row in 0..n {
            let block = (row / BLOCK).min(blocks_n - 1);
            expanded.extend_from_slice(&blocks[block * blocks_k..(block + 1) * blocks_k]);
        }
        let x = device.upload(input.clone(), &[m, k])?;
        let w = device.upload(codes.iter().copied().map(f8e4m3fn).collect(), &[n, k])?;
        let ws = device.upload(expanded, &[n, blocks_k])?;
        let mut q = api::zeros::<f8e4m3fn>(&[m, k]).sync_on(&device.stream)?;
        let mut qs = api::zeros::<f32>(&[m, blocks_k]).sync_on(&device.stream)?;
        kernels::quantize_block(
            (&mut q).partition([1, BLOCK]),
            (&mut qs).partition([1, 1]),
            &x,
        )
        .generics(vec![k.to_string(), BLOCK.to_string()])
        .sync_on(&device.stream)?;
        let output = kernels::matmul_block(
            api::zeros::<f32>(&[m, n]).partition([QUANT_TILE_ROWS, QUANT_TILE_COLUMNS]),
            &q,
            &w,
            &qs,
            &ws,
        )
        .generics(tile_generics(k))
        .first()
        .unpartition()
        .sync_on(&device.stream)?;
        let actual = output.to_host_vec().sync_on(&device.stream)?;
        let encoded = q.to_host_vec().sync_on(&device.stream)?;
        let scales = qs.to_host_vec().sync_on(&device.stream)?;
        for row in 0..m {
            for block in 0..blocks_k {
                let span = &input[row * k + block * BLOCK..row * k + (block + 1) * BLOCK];
                let scale =
                    (span.iter().copied().map(f32::abs).fold(0.0, f32::max) / 448.0).max(1e-12);
                assert!(
                    (scales[row * blocks_k + block] - scale).abs() <= scale * 1e-6,
                    "row={row} block={block} scale {} != {scale}",
                    scales[row * blocks_k + block]
                );
                for offset in 0..BLOCK {
                    let expected = quantize(span[offset] / scale);
                    assert_eq!(
                        encoded[row * k + block * BLOCK + offset].0,
                        expected,
                        "row={row} k={}",
                        block * BLOCK + offset
                    );
                }
            }
            for col in 0..n {
                let channel = col / BLOCK;
                let expected = (0..k)
                    .map(|i| {
                        let block = i / BLOCK;
                        f64::from(decode(encoded[row * k + i].0))
                            * f64::from(decode(codes[col * k + i]))
                            * f64::from(scales[row * blocks_k + block])
                            * f64::from(blocks[channel * blocks_k + block])
                    })
                    .sum::<f64>();
                assert!(
                    (f64::from(actual[row * n + col]) - expected).abs()
                        < expected.abs().mul_add(0.0002, 0.002),
                    "{m}x{n}x{k} [{row},{col}] {} != {expected}",
                    actual[row * n + col]
                );
            }
        }
    }
    Ok(())
}

/// Split-K over the K axis must reproduce the unsplit kernel exactly, and must be faster
/// on the narrow-row geometries the verification path actually runs.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn fp8_small_m_split_k_preserves_values_and_measures_better() -> Result<(), Box<dyn Error>> {
    use cutile::bench::{BenchOptions, do_bench_paired};
    use std::time::Duration;
    const SPLITS: i32 = 4;
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
    for (n, k) in [(17408, 5120), (18432, 5120)] {
        let codes: Vec<f8e4m3fn> = (0..k)
            .map(|i| f8e4m3fn(u8::try_from(1 + (i % 40)).unwrap_or(1)))
            .collect();
        // The kernels tile rows by 16, so the activation buffer carries 12 real rows plus
        // padding; a 12-row buffer would make the tiled load read past the allocation.
        let mut input_codes: Vec<f8e4m3fn> = Vec::with_capacity(16 * k);
        for _ in 0..12 {
            input_codes.extend_from_slice(&codes);
        }
        input_codes.resize(16 * k, f8e4m3fn(0));
        let mut weight_codes: Vec<f8e4m3fn> = Vec::with_capacity(n * k);
        for _ in 0..n {
            weight_codes.extend_from_slice(&codes);
        }
        let input = device.upload(input_codes, &[16, k])?;
        let weight = device.upload(weight_codes, &[n, k])?;
        let input_scale = device.upload(vec![0.5_f32; 16], &[16, 1])?;
        let weight_scale = device.upload(vec![0.25_f32; n], &[n])?;
        let mut output = api::zeros::<f32>(&[16, n]).sync_on(&device.stream)?;
        let mut split_out = api::zeros::<f32>(&[16, n]).sync_on(&device.stream)?;
        let splits = usize::try_from(SPLITS).unwrap();
        let mut partials = api::zeros::<f32>(&[splits, 16, n]).sync_on(&device.stream)?;
        let windows = k.div_ceil(128);
        let k_tiles = i32::try_from(windows.div_ceil(splits)).unwrap();
        let baseline = CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                kernels::matmul(
                    (&mut output).partition([16, 64]),
                    &input,
                    &weight,
                    &input_scale,
                    &weight_scale,
                )
                .generics(tile_generics(k)),
            )?;
            Ok(())
        })?;
        let candidate = CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                kernels::matmul_split(
                    (&mut partials).partition([1, 16, 64]),
                    &input,
                    &weight,
                    k_tiles,
                )
                .generics(vec![k.to_string()]),
            )?;
            scope.record(kernels::reduce_split(
                (&mut split_out).partition([16, 64]),
                &partials,
                &input_scale,
                &weight_scale,
                SPLITS,
            ))?;
            Ok(())
        })?;
        baseline.launch().sync_on(&device.stream)?;
        candidate.launch().sync_on(&device.stream)?;
        let direct = output.to_host_vec().sync_on(&device.stream)?;
        let split = split_out.to_host_vec().sync_on(&device.stream)?;
        let worst = relative_gap(&direct[..12 * n], &split[..12 * n]);
        let (slow, fast) = do_bench_paired(
            &device.stream,
            &options,
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
            "12x{n}x{k}: direct_ms={} split_ms={} speedup={} worst_rel={worst:e}",
            slow.median_ms(),
            fast.median_ms(),
            slow.median_ms() / fast.median_ms()
        );
    }
    Ok(())
}

/// Largest relative difference between two equal-length result rows.
fn relative_gap(direct: &[f32], split: &[f32]) -> f32 {
    direct
        .iter()
        .zip(split)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0_f32, f32::max)
}
