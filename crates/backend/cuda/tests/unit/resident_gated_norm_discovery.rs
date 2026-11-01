//! Same-address gated-norm compiler occupancy discovery; no production policy.
use crate::device::{CudaDevice, device_error};
use crate::resident::kernels::aux::norm;
use cutile::bench::BenchOptions;
use cutile::bench::do_bench_paired;
use cutile::error::tensor_error;
use cutile::{prelude::*, tile_kernel::CompileOptions};
use infer_core::Result;
use infer_cuda_kernels::tile::runtime::Graph;
use std::cell::Cell;
use std::time::Duration;

#[test]
#[ignore = "requires CUDA/safe-run; bit equality before same-address AB/BA cold/warm discovery"]
fn gated_norm_occupancy_bits_and_cost() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for rows in [48, 3072, 12288] {
        let x: Vec<f32> = (0u16..71)
            .cycle()
            .take(rows * 128)
            .map(|i| (f32::from(i) - 35.0) * 0.013)
            .collect();
        let z: Vec<f32> = (0u16..67)
            .cycle()
            .take(rows * 128)
            .map(|i| (f32::from(i) - 33.0) * 0.031)
            .collect();
        let w: Vec<f32> = (0u16..128)
            .map(|i| f32::from(i % 11).mul_add(0.021, 0.7))
            .collect();
        let x = device.upload(x, &[rows, 128])?;
        let z = device.upload(z, &[rows, 128])?;
        let w = device.upload(w, &[128])?;
        let mut out = api::zeros::<f32>(&[rows, 128])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let baseline = capture(&device, &x, &z, &w, &mut out, CompileOptions::default())?;
        let run = |g: &Graph| g.launch().sync_on(&device.stream).map_err(device_error);
        run(&baseline)?;
        let expected = device.read_borrowed(&out)?;
        assert!(expected.iter().all(|v| v.is_finite()));
        for occupancy in [2, 4, 6] {
            let candidate = capture(
                &device,
                &x,
                &z,
                &w,
                &mut out,
                CompileOptions::default().occupancy(occupancy),
            )?;
            run(&candidate)?;
            compare(&device.read_borrowed(&out)?, &expected);
            cost(&device, rows, occupancy, &baseline, &candidate)?;
            run(&candidate)?;
            compare(&device.read_borrowed(&out)?, &expected);
            println!(
                "{}",
                serde_json::json!({"kind":"gated_norm_bits","rows":rows,"D":128,
                "occupancy":occupancy,"elements":expected.len(),"bit_differences":0,"same_addresses":true})
            );
        }
    }
    Ok(())
}

fn compare(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite());
        assert_eq!(a.to_bits(), b.to_bits(), "norm element {i}");
    }
}

fn capture(
    device: &CudaDevice,
    x: &Tensor<f32>,
    z: &Tensor<f32>,
    w: &Tensor<f32>,
    out: &mut Tensor<f32>,
    options: CompileOptions,
) -> Result<Graph> {
    let xv = x.view(&[x.size() / 128, 128]).map_err(device_error)?;
    let zv = z.view(&[z.size() / 128, 128]).map_err(device_error)?;
    Graph::scope(&device.stream, |scope| {
        scope.record(
            norm(out.partition([1, 128]), &xv, w, &zv, 1e-6f32, 0.0f32)
                .generics(vec!["128".into(), "128".into(), "1".into()])
                .compile_options(options),
        )?;
        Ok(())
    })
    .map_err(device_error)
}

fn cost(
    device: &CudaDevice,
    rows: usize,
    occupancy: i32,
    baseline: &Graph,
    candidate: &Graph,
) -> Result<()> {
    for reverse in [false, true] {
        for cold in [true, false] {
            let options = BenchOptions {
                warmup: Duration::from_millis(30),
                rep: Duration::from_millis(60),
                min_reps: 10,
                max_reps: 30,
                clear_l2: cold,
            };
            let (a, b) = if reverse {
                (candidate, baseline)
            } else {
                (baseline, candidate)
            };
            let run = |g: &Graph| {
                g.launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            };
            let (a, b) = do_bench_paired(&device.stream, &options, |_| run(a), |_| run(b))
                .map_err(device_error)?;
            let (a, b) = if reverse { (b, a) } else { (a, b) };
            println!(
                "{}",
                serde_json::json!({"kind":"gated_norm_cost","rows":rows,"D":128,
                "occupancy":occupancy,"candidate_first":reverse,"clear_l2":cold,
                "same_addresses":true,"baseline_ms":a.times_ms(),"candidate_ms":b.times_ms()})
            );
        }
    }
    Ok(())
}

thread_local! {
    static MODEL_BASELINE: Cell<bool> = const { Cell::new(false) };
    static MODEL_SITES: Cell<usize> = const { Cell::new(0) };
}
pub fn model_options(
    dimension: usize,
    gated: bool,
    rows: usize,
    prepared: CompileOptions,
) -> CompileOptions {
    if MODEL_BASELINE.with(Cell::get) {
        return CompileOptions::default();
    }
    if dimension == 128 && gated && [3072, 12288].contains(&rows) && prepared.occupancy.is_some() {
        MODEL_SITES.with(|n| n.set(n.get() + 1));
    }
    prepared
}
pub fn model_sites() -> usize {
    MODEL_SITES.with(Cell::get)
}
pub fn with_model_baseline<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            MODEL_BASELINE.with(|n| n.set(self.0));
        }
    }
    let _restore = Restore(MODEL_BASELINE.with(|n| n.replace(true)));
    MODEL_SITES.with(|n| n.set(0));
    run()
}
