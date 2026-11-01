use super::kernels;
use crate::device::CudaDevice;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error;

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (f32::from(u16::try_from((i * 17 + salt) % 131).unwrap()) - 65.0) / 91.0)
        .collect()
}
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn parallel_conv_matches_causal_reference_and_partial_state() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (lanes, channels) in [(32, 17), (32, 128), (128, 10240)] {
        let weights = values(channels * 4, 3);
        let w = device.upload(weights.clone(), &[channels, 4])?;
        let mut expected: Vec<_> = (0..3).map(|i| values(channels, i)).collect();
        let mut h0 = api::zeros::<f32>(&[channels]).sync_on(&device.stream)?;
        let mut h1 = api::zeros::<f32>(&[channels]).sync_on(&device.stream)?;
        let mut h2 = api::zeros::<f32>(&[channels]).sync_on(&device.stream)?;
        for (target, initial) in [&mut h0, &mut h1, &mut h2].into_iter().zip(&expected) {
            let uploaded = device.upload(initial.clone(), &[channels])?;
            api::memcpy(target, &uploaded).sync_on(&device.stream)?;
        }
        for (base, count, offset) in [
            (0, 0, 0),
            (0, 1, -1),
            (0, 2, -1),
            (9, 1, 0),
            (10, 2, 0),
            (12, 3, 0),
            (15, lanes - 3, 0),
            (128, lanes, 0),
        ] {
            let input = values(lanes * channels, count + 19);
            let x = device.upload(input.clone(), &[lanes, channels])?;
            let meta = device.upload(vec![base, i32::try_from(count)?, offset, 0], &[4])?;
            let mut out = api::zeros::<f32>(&[lanes, channels]).sync_on(&device.stream)?;
            kernels::forward((&mut out).partition([1, 128]), &h0, &h1, &h2, &x, &w, &meta)
                .sync_on(&device.stream)?;
            kernels::commit(
                (&mut h0).partition([128]),
                (&mut h1).partition([128]),
                (&mut h2).partition([128]),
                &x,
                &meta,
            )
            .sync_on(&device.stream)?;
            let mut expected_output = vec![0.0_f64; lanes * channels];
            for row in 0..count {
                if base + i32::try_from(row)? + offset < 0 {
                    continue;
                }
                for col in 0..channels {
                    let v = f64::from(input[row * channels + col]).mul_add(
                        f64::from(weights[col * 4 + 3]),
                        (0..3)
                            .map(|i| f64::from(expected[i][col]) * f64::from(weights[col * 4 + i]))
                            .sum::<f64>(),
                    );
                    expected_output[row * channels + col] = v / (1.0 + (-v).exp());
                }
                expected.rotate_left(1);
                expected[2].copy_from_slice(&input[row * channels..][..channels]);
            }
            for (a, b) in out
                .to_host_vec()
                .sync_on(&device.stream)?
                .iter()
                .zip(expected_output)
            {
                assert!((f64::from(*a) - b).abs() < 0.000_002, "{a} != {b}");
            }
            for (actual, expected) in [&h0, &h1, &h2].into_iter().zip(&expected) {
                let mut snapshot = api::zeros::<f32>(&[channels]).sync_on(&device.stream)?;
                api::memcpy(&mut snapshot, actual).sync_on(&device.stream)?;
                assert_eq!(&snapshot.to_host_vec().sync_on(&device.stream)?, expected);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run in release"]
fn parallel_conv_performance() -> Result<(), Box<dyn Error>> {
    use cutile::bench::{BenchOptions, do_bench_paired};
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err("performance gate requires release".into());
    }
    let compile = CompileOptions::default();
    assert!(!compile.device_debug && compile.opt_level.is_none_or(|level| level == 3));
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(100),
        min_reps: 10,
        max_reps: 100,
        clear_l2: true,
    };
    for (lanes, channels) in [(32, 128), (32, 10240), (128, 10240)] {
        let x = device.upload(values(lanes * channels, 0), &[lanes, channels])?;
        let w = device.upload(values(channels * 4, 3), &[channels, 4])?;
        let mut out = api::zeros::<f32>(&[lanes, channels]).sync_on(&device.stream)?;
        let mut h0 = api::zeros::<f32>(&[channels]).sync_on(&device.stream)?;
        let mut h1 = api::zeros::<f32>(&[channels]).sync_on(&device.stream)?;
        let mut h2 = api::zeros::<f32>(&[channels]).sync_on(&device.stream)?;
        let metadata = device.upload(vec![0, i32::try_from(lanes)?, 0, 0], &[4])?;
        let row_meta = device.upload(vec![0, 0, 1, 0], &[4])?;
        let candidate = Graph::scope(&device.stream, |scope| {
            scope.record(kernels::forward(
                (&mut out).partition([1, 128]),
                &h0,
                &h1,
                &h2,
                &x,
                &w,
                &metadata,
            ))?;
            scope.record(kernels::commit(
                (&mut h0).partition([128]),
                (&mut h1).partition([128]),
                (&mut h2).partition([128]),
                &x,
                &metadata,
            ))?;
            Ok(())
        })?;
        let host = values(lanes * channels, 0);
        let inputs = host
            .chunks(channels)
            .map(|row| device.upload(row.to_vec(), &[channels]))
            .collect::<infer_core::Result<Vec<_>>>()?;
        let mut outputs = (0..lanes)
            .map(|_| api::zeros::<f32>(&[channels]).sync_on(&device.stream))
            .collect::<Result<Vec<_>, _>>()?;
        let baseline = Graph::scope(&device.stream, |scope| {
            for (output, input) in outputs.iter_mut().zip(&inputs) {
                scope.record(super::super::recurrent::recurrent::conv4(
                    output.partition([128]),
                    (&mut h0).partition([128]),
                    (&mut h1).partition([128]),
                    (&mut h2).partition([128]),
                    input,
                    &w,
                    &row_meta,
                ))?;
            }
            Ok(())
        })?;
        let (a, b) = do_bench_paired(
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
            "lanes={lanes} channels={channels} serial_ms={} parallel_ms={} speedup={}",
            a.median_ms(),
            b.median_ms(),
            a.median_ms() / b.median_ms()
        );
        assert!(b.median_ms() <= 1.05 * a.median_ms());
    }
    Ok(())
}
