//! Exhaustive decode and complete append/attention output gates before paired timing.
use super::super::{
    Choice, Shape,
    capture::Captured,
    reference::{self, FIXTURES, Fixture},
};
use crate::{
    device::{CudaDevice, device_error},
    resident::attention_oxide::Support,
};
use cuda_core::f8e4m3fn;
use cutile::bench::BenchOptions;
use cutile::bench::do_bench_paired;
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Graph;
use std::env::var;
use std::fs::read_to_string;
use std::ptr::from_mut;
use std::time::Duration;
#[expect(
    unsafe_code,
    reason = "test binds retained 256-element buffers on the owning stream and synchronizes before release"
)]
fn conversion_gate(device: &CudaDevice, ptx: &str) -> Result<()> {
    let input = device.upload((u8::MIN..=u8::MAX).collect(), &[256])?;
    let mut output = api::zeros::<f32>(&[256])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let module = device
        .stream
        .device()
        .load_module_from_ptx_src(ptx)
        .map_err(device_error)?;
    let function = module
        .load_function("fp8_conversion_all_codes")
        .map_err(device_error)?;
    let ctx = ExecutionContext::new(device.stream.clone());
    KernelInputStored::retain(&&*input, &ctx).map_err(device_error)?;
    KernelOutputStored::retain(&(&mut output).partition([256]), &ctx).map_err(device_error)?;
    let mut a = input.device_pointer().cu_deviceptr();
    let mut b = output.device_pointer().cu_deviceptr();
    let mut args = [from_mut(&mut a).cast(), from_mut(&mut b).cast()];
    // SAFETY: the owning module and function remain live through the synchronous test.
    let raw = unsafe { function.cu_function() };
    // SAFETY: the complete buffers/module are live, leases exclude other streams; one thread writes one cell.
    unsafe {
        cuda_core::launch_kernel(
            raw,
            (1, 1, 1),
            (256, 1, 1),
            0,
            device.stream.cu_stream(),
            &mut args,
        )
    }
    .map_err(device_error)?;
    // SAFETY: retained allocations/module stay live until completion on their owning stream.
    unsafe { device.stream.synchronize() }.map_err(device_error)?;
    drop(ctx);
    let actual = device.read_borrowed(&output)?;
    for (i, value) in actual.iter().enumerate() {
        let expected = reference::decode(f8e4m3fn(u8::try_from(i).map_err(device_error)?));
        assert_eq!(value.to_bits(), expected.to_bits(), "FP8 code {i}");
    }
    println!("FP8_ALL_CODES bits_identical=256 signed_zero=true reserved_codes_preserved=true");
    Ok(())
}
fn ptx() -> Result<String> {
    let path = var("INFER_TEST_ATTENTION_PTX").map_err(|e| Error::invalid(e.to_string()))?;
    read_to_string(path).map_err(|e| Error::invalid(e.to_string()))
}
#[test]
#[ignore = "requires CUDA/safe-run; exhaustive codes and full bits/F64 before any timing"]
fn fp8_hardware_complete_bits_gate() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let ptx = ptx()?;
    conversion_gate(&device, &ptx)?;
    for lanes in [64, 256] {
        let shape = Shape {
            heads: 24,
            kv_heads: 4,
            dim: 256,
            lanes,
            window: 0,
        };
        for index in 0..FIXTURES {
            let fixture = Fixture::new(shape, index)?;
            let baseline = Captured::new(&device, &fixture, Choice::RustSimt)?;
            let candidate =
                Captured::candidate(&device, &fixture, Support::load_test_ptx(&device, &ptx)?)?;
            let bits = candidate.exact_match(&device, &baseline)?;
            let error = candidate.verify(&device, &fixture)?;
            println!(
                "{}",
                serde_json::json!({"kind":"fp8_hardware_bits","lanes":lanes,"fixture":index,"elements":bits,"bit_differences":0,"f64_error":error})
            );
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires CUDA/safe-run; complete pipeline cold/warm AB/BA with pre/post bits"]
fn fp8_hardware_complete_pipeline_cost() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let ptx = ptx()?;
    conversion_gate(&device, &ptx)?;
    for lanes in [64, 256] {
        let shape = Shape {
            heads: 24,
            kv_heads: 4,
            dim: 256,
            lanes,
            window: 0,
        };
        for index in [0, 1] {
            let fixture = Fixture::new(shape, index)?;
            let mut baseline = Captured::new(&device, &fixture, Choice::RustSimt)?;
            let support = Support::load_test_ptx(&device, &ptx)?;
            let candidate = baseline.graph_variant(&device, &fixture, &support)?;
            baseline.verify_variant(&device, &fixture, &candidate)?;
            for reverse in [false, true] {
                for cold in [true, false] {
                    let options = BenchOptions {
                        warmup: Duration::from_millis(50),
                        rep: Duration::from_millis(100),
                        min_reps: 10,
                        max_reps: 30,
                        clear_l2: cold,
                    };
                    let (left, right) = if reverse {
                        (&candidate, &baseline.graph)
                    } else {
                        (&baseline.graph, &candidate)
                    };
                    let run = |graph: &Graph| {
                        graph
                            .launch()
                            .sync_on(&device.stream)
                            .map_err(|e| tensor_error(&e.to_string()))
                    };
                    let (a, b) =
                        do_bench_paired(&device.stream, &options, |_| run(left), |_| run(right))
                            .map_err(device_error)?;
                    let (a, b) = if reverse { (b, a) } else { (a, b) };
                    println!(
                        "{}",
                        serde_json::json!({"kind":"fp8_hardware_cost","lanes":lanes,
                        "fixture":index,"candidate_first":reverse,"clear_l2":cold,"same_addresses":true,
                        "baseline_ms":a.times_ms(),"candidate_ms":b.times_ms()})
                    );
                }
            }
            baseline.verify_variant(&device, &fixture, &candidate)?;
        }
    }
    Ok(())
}
