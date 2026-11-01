use super::kernels;
use crate::{device::CudaDevice, resident::fp8_gemm};
use cuda_core::f8e4m3fn;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error;
use std::time::Duration;
type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct Case {
    rows: usize,
    k: usize,
    input: Tensor<f32>,
    old: Tensor<f8e4m3fn>,
    new: Tensor<f8e4m3fn>,
    old_scale: Tensor<f32>,
    new_scale: Tensor<f32>,
}
impl Case {
    fn new(device: &CudaDevice, rows: usize, k: usize) -> TestResult<Self> {
        Ok(Self {
            rows,
            k,
            input: api::zeros::<f32>(&[rows, k]).sync_on(&device.stream)?,
            old: api::zeros::<f8e4m3fn>(&[rows, k]).sync_on(&device.stream)?,
            new: api::zeros::<f8e4m3fn>(&[rows, k]).sync_on(&device.stream)?,
            old_scale: api::zeros::<f32>(&[rows, 1]).sync_on(&device.stream)?,
            new_scale: api::zeros::<f32>(&[rows, 1]).sync_on(&device.stream)?,
        })
    }
    fn graphs(&mut self, device: &CudaDevice, count: usize) -> TestResult<(Graph, Graph)> {
        let pad = self.k.next_power_of_two();
        let old = Graph::scope(&device.stream, |scope| {
            for _ in 0..count {
                scope.record(
                    fp8_gemm::kernels::quantize(
                        (&mut self.old).partition([1, pad]),
                        (&mut self.old_scale).partition([1, 1]),
                        &self.input,
                    )
                    .generics(vec![self.k.to_string(), pad.to_string()]),
                )?;
            }
            Ok(())
        })?;
        let new = Graph::scope(&device.stream, |scope| {
            for _ in 0..count {
                scope.record(
                    kernels::maximum((&mut self.new_scale).partition([1, 1]), &self.input)
                        .generics(vec![self.k.to_string(), pad.to_string()]),
                )?;
                scope.record(
                    kernels::quantize(
                        (&mut self.new).partition([1, 512]),
                        &self.new_scale,
                        &self.input,
                    )
                    .generics(vec![self.k.to_string()]),
                )?;
            }
            Ok(())
        })?;
        Ok((old, new))
    }
    fn input(&mut self, device: &CudaDevice, fixture: usize) -> TestResult {
        let input = (0..self.rows * self.k)
            .map(|i| {
                let v = (f32::from(u16::try_from(i * 17 % 131).unwrap_or_default()) - 65.0) / 37.0;
                match fixture {
                    1 => 0.0,
                    2 => -0.0,
                    3 => v * 1.0e-30,
                    4 => v * 1.0e20,
                    5 => {
                        if i % 19 == 0 {
                            v
                        } else {
                            0.0
                        }
                    }
                    6 => {
                        if i % 2 == 0 {
                            v.abs()
                        } else {
                            -v.abs()
                        }
                    }
                    7 => v + 0.03125,
                    8 => {
                        if i % self.k == self.k - 1 {
                            448.0
                        } else {
                            v
                        }
                    }
                    9 => {
                        if i % self.k == 0 {
                            0.000_448
                        } else {
                            v * 1.0e-7
                        }
                    }
                    _ => v,
                }
            })
            .collect();
        let uploaded = device.upload(input, &[self.rows, self.k])?;
        api::memcpy(&mut self.input, &uploaded).sync_on(&device.stream)?;
        Ok(())
    }
    fn equal(&self, device: &CudaDevice) -> TestResult {
        let a = device.read_borrowed(&self.old)?;
        let b = device.read_borrowed(&self.new)?;
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.0 == y.0), "FP8 codes");
        let a = device.read_borrowed(&self.old_scale)?;
        let b = device.read_borrowed(&self.new_scale)?;
        assert!(a.iter().chain(&b).all(|x| x.is_finite()));
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "FP8 scale"
        );
        Ok(())
    }
}
#[test]
#[ignore = "requires CUDA/safe-run; exact finite row max and quantizer cost"]
fn bits_and_cost() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for rows in [1, 4, 64, 256] {
        for k in [144, 1024, 4096, 4097, 5120, 6144, 8192, 16384, 17408, 32768] {
            let mut case = Case::new(&device, rows, k)?;
            let (old, new) = case.graphs(&device, 1)?;
            for fixture in 0..10 {
                case.input(&device, fixture)?;
                old.launch().sync_on(&device.stream)?;
                new.launch().sync_on(&device.stream)?;
                case.equal(&device)?;
            }
            println!(
                "{}",
                serde_json::json!({"rows":rows,"k":k,"fixtures":10,"differences":0})
            );
            case.input(&device, 0)?;
            drop((old, new));
            let (old, new) = case.graphs(&device, 73)?;
            for reverse in [false, true] {
                for cold in [false, true] {
                    let options = BenchOptions {
                        warmup: Duration::from_millis(20),
                        rep: Duration::from_millis(100),
                        min_reps: 32,
                        max_reps: 96,
                        clear_l2: cold,
                    };
                    let run = |g: &Graph| {
                        g.launch()
                            .sync_on(&device.stream)
                            .map_err(|e| tensor_error(&e.to_string()))
                    };
                    let (first, second) = if reverse { (&new, &old) } else { (&old, &new) };
                    let (a, b) =
                        do_bench_paired(&device.stream, &options, |_| run(first), |_| run(second))?;
                    let (a, b) = if reverse { (b, a) } else { (a, b) };
                    println!(
                        "{}",
                        serde_json::json!({"rows":rows,"k":k,"count":73,
                        "reverse":reverse,"cold":cold,"old_ms":a.median_ms(),"new_ms":b.median_ms(),
                        "speedup":a.median_ms()/b.median_ms(),"old_samples":a.times_ms(),
                        "new_samples":b.times_ms(),"service_admission":false})
                    );
                }
            }
        }
    }
    Ok(())
}
