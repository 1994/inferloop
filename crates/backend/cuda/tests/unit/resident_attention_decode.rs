use super::kernels;
use crate::constants::KV_BLOCK_TOKENS;
use crate::device::CudaDevice;
use cuda_core::f8e4m3fn;
use cutile::bench::BenchOptions as CutileBenchOptions;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error;

/// Identity block table for a KV arena of `capacity` tokens.
fn block_table(device: &CudaDevice, capacity: usize) -> Result<Arc<Tensor<i32>>, Box<dyn Error>> {
    let blocks = capacity.div_ceil(KV_BLOCK_TOKENS);
    let values: Vec<i32> = (0..blocks)
        .map(|b| i32::try_from(b).unwrap_or(i32::MAX))
        .collect();
    Ok(device.upload(values, &[blocks])?)
}

fn decoded(code: u8) -> f32 {
    let magnitude = code & 127;
    let x = (1.0 + f32::from(magnitude & 7) / 8.0) * 2.0_f32.powi(i32::from(magnitude >> 3) - 7);
    if code & 128 == 0 { x } else { -x }
}
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn split_kv_matches_independent_causal_window_reference() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (heads, kv_heads, dim) in [(4, 2, 32), (8, 8, 64), (16, 8, 128), (24, 4, 256)] {
        check_geometry(&device, heads, kv_heads, dim)?;
    }
    Ok(())
}

fn check_geometry(
    device: &CudaDevice,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) -> Result<(), Box<dyn Error>> {
    let capacity = 1024;
    let scale = f64::from(u32::try_from(dim)?).sqrt();
    let query: Vec<f32> = (0..heads * dim)
        .map(|i| (f32::from(u16::try_from(i % 15).unwrap()) - 7.0) / 8.0)
        .collect();
    let codes: Vec<u8> = (0..kv_heads * capacity * dim)
        .map(|i| 32 + u8::try_from(i % 31).unwrap() + if i % 3 == 0 { 128 } else { 0 })
        .collect();
    let value_codes: Vec<u8> = codes.iter().copied().rev().collect();
    let keys: Vec<f32> = codes.iter().copied().map(decoded).collect();
    let values: Vec<f32> = value_codes.iter().copied().map(decoded).collect();
    let q = device.upload(query.clone(), &[heads, dim])?;
    let k = device.upload(keys.clone(), &[kv_heads, capacity, dim])?;
    let v = device.upload(values.clone(), &[kv_heads, capacity, dim])?;
    let k8 = device.upload(
        codes.into_iter().map(f8e4m3fn).collect(),
        &[kv_heads, capacity, dim],
    )?;
    let v8 = device.upload(
        value_codes.into_iter().map(f8e4m3fn).collect(),
        &[kv_heads, capacity, dim],
    )?;
    for (position, window) in [
        (-1, 0),
        (0, 0),
        (31, 0),
        (32, 17),
        (511, 0),
        (1023, 0),
        (1023, 17),
    ] {
        let mut expected = vec![0.0_f64; heads * dim];
        if position >= 0 {
            let end = usize::try_from(position + 1)?;
            let start = if window == 0 {
                0
            } else {
                end.saturating_sub(usize::try_from(window)?)
            };
            for head in 0..heads {
                let offset = head / (heads / kv_heads) * capacity * dim;
                let mut scores: Vec<f64> = (start..end)
                    .map(|row| {
                        (0..dim)
                            .map(|d| {
                                f64::from(query[head * dim + d])
                                    * f64::from(keys[offset + row * dim + d])
                                    * 0.5
                            })
                            .sum::<f64>()
                            / scale
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for score in &mut scores {
                    *score = (*score - maximum).exp();
                }
                let total: f64 = scores.iter().sum();
                for d in 0..dim {
                    expected[head * dim + d] = scores
                        .iter()
                        .enumerate()
                        .map(|(i, s)| s * f64::from(values[offset + (start + i) * dim + d]) * 2.0)
                        .sum::<f64>()
                        / total;
                }
            }
        }
        for parts in [1, 2, 8, 16] {
            for actual in [
                run(device, &q, &k, &v, position, window, parts)?,
                run(device, &q, &k8, &v8, position, window, parts)?,
            ] {
                for (a, b) in actual.iter().zip(&expected) {
                    assert!(
                        (f64::from(*a) - b).abs() < 2e-5,
                        "position={position} window={window} parts={parts}: {a} != {b}"
                    );
                }
            }
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn stacked_verification_matches_split_reference() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (heads, kv_heads, dim, capacity, parts) = (24, 4, 256, 1024, 8);
    let codes: Vec<u8> = (0..kv_heads * capacity * dim)
        .map(|i| 32 + u8::try_from(i % 31).unwrap() + if i % 3 == 0 { 128 } else { 0 })
        .collect();
    let v8 = device.upload(
        codes.iter().copied().rev().map(f8e4m3fn).collect(),
        &[kv_heads, capacity, dim],
    )?;
    let k8 = device.upload(
        codes.into_iter().map(f8e4m3fn).collect(),
        &[kv_heads, capacity, dim],
    )?;
    let table = block_table(&device, capacity)?;
    let queries = (0..3)
        .map(|lane| {
            let values: Vec<f32> = (0..heads * dim)
                .map(|i| (f32::from(u16::try_from((i + 5 * lane) % 15).unwrap()) - 7.0) / 8.0)
                .collect();
            device.upload(values, &[heads, dim])
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The three lanes straddle a KV block boundary, start the context, or end it under a window.
    for (first, window) in [(0, 0), (30, 0), (510, 0), (1021, 0), (40, 17)] {
        let positions = [first, first + 1, first + 2];
        let metadata = positions
            .iter()
            .map(|&p| device.upload(vec![p, 0, p, 0], &[4]))
            .collect::<Result<Vec<_>, _>>()?;
        let blocks = kv_heads * parts;
        let mut numerator = api::zeros::<f32>(&[blocks, 32, dim]).sync_on(&device.stream)?;
        let mut maxima = api::zeros::<f32>(&[blocks, 32]).sync_on(&device.stream)?;
        let mut sums = api::zeros::<f32>(&[blocks, 32]).sync_on(&device.stream)?;
        let views = queries
            .iter()
            .map(|q| q.view(&[kv_heads, heads / kv_heads, dim]))
            .collect::<Result<Vec<_>, _>>()?;
        kernels::partial_gqa3(
            (&mut numerator).partition([1, 32, dim]),
            (&mut maxima).partition([1, 32]),
            (&mut sums).partition([1, 32]),
            &views[0],
            &views[1],
            &views[2],
            &metadata[0],
            &metadata[1],
            &metadata[2],
            &k8,
            &v8,
            &table,
            window,
            0.5,
            2.0,
        )
        .generics(vec![
            f8e4m3fn::DTYPE.as_str().into(),
            dim.to_string(),
            parts.to_string(),
            KV_BLOCK_TOKENS.to_string(),
        ])
        .sync_on(&device.stream)?;
        for (lane, q) in queries.iter().enumerate() {
            let actual = kernels::merge_gqa(
                api::zeros::<f32>(&[heads, dim]).partition([1, dim]),
                &numerator.view(&[kv_heads, parts, 32, dim])?,
                &maxima.view(&[kv_heads, parts, 32])?,
                &sums.view(&[kv_heads, parts, 32])?,
                i32::try_from(lane * 8)?,
            )
            .generics(vec![
                dim.to_string(),
                parts.to_string(),
                (heads / kv_heads).to_string(),
                "32".into(),
            ])
            .first()
            .unpartition()
            .sync_on(&device.stream)?
            .to_host_vec()
            .sync_on(&device.stream)?;
            let expected = run(&device, q, &k8, &v8, positions[lane], window, 1)?;
            for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                // BF16 query and probability products against the F32 per-head reference.
                assert!(
                    (a - b).abs() <= 2e-2 * b.abs().max(1.0),
                    "first={first} window={window} lane={lane} index={index}: {a} != {b}"
                );
            }
        }
    }
    Ok(())
}

fn run<E: DType>(
    device: &CudaDevice,
    q: &Tensor<f32>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    position: i32,
    window: i32,
    parts: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let heads = usize::try_from(q.shape()[0])?;
    let dim = usize::try_from(q.shape()[1])?;
    let kv_heads = usize::try_from(k.shape()[0])?;
    let meta = device.upload(vec![position, 0, position, 0], &[4])?;
    let mut numerator = api::zeros::<f32>(&[heads * parts, dim]).sync_on(&device.stream)?;
    let mut maxima = api::zeros::<f32>(&[heads * parts, 1]).sync_on(&device.stream)?;
    let mut sums = api::zeros::<f32>(&[heads * parts, 1]).sync_on(&device.stream)?;
    // Step 1 of the KV refactor: every KV read resolves its block through this table.
    let table = block_table(device, usize::try_from(k.shape()[1])?)?;
    kernels::partial(
        (&mut numerator).partition([1, dim]),
        (&mut maxima).partition([1, 1]),
        (&mut sums).partition([1, 1]),
        q,
        k,
        v,
        &meta,
        &table,
        window,
        0.5,
        2.0,
    )
    .generics(vec![
        E::DTYPE.as_str().into(),
        dim.to_string(),
        (heads / kv_heads).to_string(),
        parts.to_string(),
        KV_BLOCK_TOKENS.to_string(),
    ])
    .sync_on(&device.stream)?;
    let output = kernels::merge(
        api::zeros::<f32>(&[heads, dim]).partition([1, dim]),
        &numerator.view(&[heads, parts, dim])?,
        &maxima.view(&[heads, parts])?,
        &sums.view(&[heads, parts])?,
    )
    .generics(vec![dim.to_string(), parts.to_string()])
    .first()
    .unpartition()
    .sync_on(&device.stream)?;
    Ok(output.to_host_vec().sync_on(&device.stream)?)
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn split_kv_performance() -> Result<(), Box<dyn Error>> {
    use cutile::bench::BenchOptions;
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err("performance requires release".into());
    }
    let compile = CompileOptions::default();
    if compile.device_debug || compile.opt_level.is_some_and(|x| x != 3) {
        return Err("performance requires CUDA O3".into());
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
    for (heads, kv_heads, dim) in [(16, 8, 128), (32, 4, 128), (8, 2, 64)] {
        bench_case(&device, &options, heads, kv_heads, dim)?;
    }
    Ok(())
}

fn bench_case(
    device: &CudaDevice,
    options: &CutileBenchOptions,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) -> Result<(), Box<dyn Error>> {
    use cutile::bench::do_bench_paired;
    for capacity in [128, 512, 4096, 16384] {
        let q = device.upload(vec![0.125_f32; heads * dim], &[heads, dim])?;
        let k = device.upload(
            vec![0.25_f32; kv_heads * capacity * dim],
            &[kv_heads, capacity, dim],
        )?;
        let v = device.upload(
            vec![0.375_f32; kv_heads * capacity * dim],
            &[kv_heads, capacity, dim],
        )?;
        let position = i32::try_from(capacity - 1)?;
        let meta = device.upload(vec![position, 0, position, 0], &[4])?;
        let mut output = api::zeros::<f32>(&[heads, dim]).sync_on(&device.stream)?;
        let table = block_table(device, capacity)?;
        let baseline = Graph::scope(&device.stream, |scope| {
            scope.record(
                super::super::attention::attention::decode(
                    (&mut output).partition([1, dim]),
                    &q,
                    &k,
                    &v,
                    &meta,
                    &table,
                    0,
                    1.0,
                    1.0,
                )
                .generics(vec![
                    f32::DTYPE.as_str().into(),
                    dim.to_string(),
                    (heads / kv_heads).to_string(),
                    KV_BLOCK_TOKENS.to_string(),
                ]),
            )?;
            Ok(())
        })?;
        for parts in [2, 8, 16] {
            let mut numerator = api::zeros::<f32>(&[heads * parts, dim]).sync_on(&device.stream)?;
            let mut maxima = api::zeros::<f32>(&[heads * parts, 1]).sync_on(&device.stream)?;
            let mut sums = api::zeros::<f32>(&[heads * parts, 1]).sync_on(&device.stream)?;
            let candidate = Graph::scope(&device.stream, |scope| {
                scope.record(
                    kernels::partial(
                        (&mut numerator).partition([1, dim]),
                        (&mut maxima).partition([1, 1]),
                        (&mut sums).partition([1, 1]),
                        &q,
                        &k,
                        &v,
                        &meta,
                        &table,
                        0,
                        1.0,
                        1.0,
                    )
                    .generics(vec![
                        f32::DTYPE.as_str().into(),
                        dim.to_string(),
                        (heads / kv_heads).to_string(),
                        parts.to_string(),
                        KV_BLOCK_TOKENS.to_string(),
                    ]),
                )?;
                scope.record(
                    kernels::merge(
                        (&mut output).partition([1, dim]),
                        &numerator.view(&[heads, parts, dim])?,
                        &maxima.view(&[heads, parts])?,
                        &sums.view(&[heads, parts])?,
                    )
                    .generics(vec![dim.to_string(), parts.to_string()]),
                )?;
                Ok(())
            })?;
            let (old, new) = do_bench_paired(
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
                "heads={heads} kv_heads={kv_heads} dim={dim} tokens={capacity} parts={parts} old_ms={} new_ms={} speedup={}",
                old.median_ms(),
                new.median_ms(),
                old.median_ms() / new.median_ms()
            );
        }
    }
    Ok(())
}

// Rust compilation does not specialize the cuTile DSL bodies. Compile Tile IR on
// header-only CI as well, so missing operand types fail before a device serves a request.
#[test]
fn grouped_decode_helpers_compile_without_a_device() -> Result<(), Box<dyn Error>> {
    use cutile::cutile_compiler::compile_api::KernelCompiler;
    let single: Vec<(&str, &[i32])> = vec![
        ("out0", &[4096, 256, 1]),
        ("maxima0", &[16, 1]),
        ("sums0", &[16, 1]),
        ("q0", &[1536, 256, 1]),
        ("metadata0", &[1]),
    ];
    let stacked: Vec<(&str, &[i32])> = vec![
        ("out", &[8192, 256, 1]),
        ("maxima", &[32, 1]),
        ("sums", &[32, 1]),
        ("q0", &[1536, 256, 1]),
        ("q1", &[1536, 256, 1]),
        ("q2", &[1536, 256, 1]),
        ("metadata0", &[1]),
        ("metadata1", &[1]),
        ("metadata2", &[1]),
    ];
    for (entry, lanes) in [("partial_gqa", single), ("partial_gqa3", stacked)] {
        let mut strides: Vec<(&str, &[i32])> = vec![
            ("keys", &[32768, 256, 1]),
            ("values", &[32768, 256, 1]),
            ("table", &[1]),
        ];
        strides.extend(lanes);
        for dtype in [f32::DTYPE.as_str(), f8e4m3fn::DTYPE.as_str()] {
            let compiled = KernelCompiler::new(kernels::__module_ast_self, "kernels", entry)
                .target("sm_120")
                .generics(vec![dtype.into(), "256".into(), "64".into(), "32".into()])
                .strides(&strides)
                .compile()?;
            assert!(compiled.ir_text().contains("mma"));
        }
    }
    for rows in ["16", "32"] {
        let row_stride: i32 = rows.parse()?;
        let numerator = [64 * row_stride * 256, row_stride * 256, 256, 1];
        let partials = [64 * row_stride, row_stride, 1];
        let strides: Vec<(&str, &[i32])> = vec![
            ("out", &[256, 1]),
            ("numerator", &numerator),
            ("maxima", &partials),
            ("sums", &partials),
        ];
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "merge_gqa")
            .target("sm_120")
            .generics(vec!["256".into(), "64".into(), "6".into(), rows.into()])
            .strides(&strides)
            .compile()?;
    }
    Ok(())
}
