use super::kernels;
use crate::{constants::KV_BLOCK_TOKENS, device::CudaDevice};
use cuda_core::f8e4m3fn;
use cutile::{cutile_compiler::compile_api::KernelCompiler, prelude::*};
use infer_cuda_kernels::tile::runtime::Graph;
use std::{collections::BTreeSet, error::Error};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[test]
fn physical_page_append_headers_compile_without_a_device() -> TestResult {
    for (dimension, capacity, lookup) in [(32, 128, 4), (256, 71200, 32), (256, 32768, 64)] {
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "append")
            .target("sm_120")
            .generics(vec![
                "f8e4m3fn".into(),
                dimension.to_string(),
                "32".into(),
                "1".into(),
                "32".into(),
                lookup.to_string(),
            ])
            .strides(&[
                ("keys", &[capacity * dimension, dimension, 1]),
                ("values", &[capacity * dimension, dimension, 1]),
                ("k", &[4 * dimension, dimension, 1]),
                ("v", &[4 * dimension, dimension, 1]),
                ("metadata", &[1]),
                ("table", &[1]),
            ])
            .compile()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA; run inside safe-run"]
fn physical_page_append_matches_head_loop_after_reassignment() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (capacity, dimension, lanes, base, count, offset) in [
        (128, 32, 128, 0, 128, 0),
        (128, 32, 128, 3, 117, 0),
        (128, 32, 128, 0, 117, -1),
        (128, 32, 128, 32, 0, 0),
        (32768, 256, 1024, 31744, 1024, 0),
        (32768, 256, 2048, 2047, 2048, 0),
        (32768, 256, 2048, 30719, 2048, 0),
        (32768, 256, 2048, 30719, 2048, -1),
        (71200, 256, 1024, 0, 1024, 0),
        (71200, 256, 1024, 1536, 1024, 0),
        (71200, 256, 1024, 993, 1024, 0),
        (71200, 256, 1024, 69600, 1024, 0),
        (71200, 256, 1024, 70175, 1024, -1),
        (71200, 256, 1024, 0, 1024, -1),
        (71200, 256, 1024, 4096, 1001, -1),
    ] {
        Fixture::new(&device, capacity, dimension, lanes)?
            .check(&device, [base, count, offset, 0])?;
    }
    Ok(())
}

struct Fixture {
    capacity: usize,
    dimension: usize,
    lanes: usize,
    key: Arc<Tensor<f32>>,
    value: Arc<Tensor<f32>>,
    metadata: Tensor<i32>,
    table: Tensor<i32>,
    reference: [Tensor<f8e4m3fn>; 2],
    candidate: [Tensor<f8e4m3fn>; 2],
}

impl Fixture {
    fn new(
        device: &CudaDevice,
        capacity: usize,
        dimension: usize,
        lanes: usize,
    ) -> TestResult<Self> {
        let data = |seed: usize| {
            (0..lanes * 4 * dimension)
                .map(|i| match i % 113 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 14.0,
                    3 => -14.0,
                    4 => 1e5,
                    5 => -1e5,
                    _ => (f32::from(u16::try_from((i + seed) % 1021).unwrap()) - 510.0) / 73.0,
                })
                .collect()
        };
        let cache = || -> TestResult<Tensor<f8e4m3fn>> {
            let values: Vec<_> = (0..4 * capacity * dimension)
                .map(|i| f8e4m3fn(u8::try_from(i % 120).unwrap()))
                .collect();
            Ok(api::copy_host_vec_to_device(&Arc::new(values))
                .sync_on(&device.stream)?
                .reshape(&[4, capacity, dimension])?)
        };
        Ok(Self {
            capacity,
            dimension,
            lanes,
            key: device.upload(data(7), &[lanes, 4, dimension])?,
            value: device.upload(data(19), &[lanes, 4, dimension])?,
            metadata: api::zeros::<i32>(&[4]).sync_on(&device.stream)?,
            table: api::zeros::<i32>(&[capacity / KV_BLOCK_TOKENS]).sync_on(&device.stream)?,
            reference: [cache()?, cache()?],
            candidate: [cache()?, cache()?],
        })
    }

    fn graph(&mut self, device: &CudaDevice, optimized: bool) -> TestResult<Graph> {
        let rows = if optimized {
            KV_BLOCK_TOKENS
        } else {
            self.capacity
        };
        let lookup = self.lanes.div_ceil(KV_BLOCK_TOKENS).next_power_of_two();
        let [keys, values] = if optimized {
            &mut self.candidate
        } else {
            &mut self.reference
        };
        Ok(Graph::scope(&device.stream, |scope| {
            scope.record(
                kernels::append(
                    keys.partition([1, rows, self.dimension]),
                    values.partition([1, rows, self.dimension]),
                    &self.key,
                    &self.value,
                    &self.metadata,
                    &self.table,
                    0.03125,
                    3.0,
                )
                .generics(vec![
                    "f8e4m3fn".into(),
                    self.dimension.to_string(),
                    rows.to_string(),
                    "1".into(),
                    KV_BLOCK_TOKENS.to_string(),
                    lookup.to_string(),
                ]),
            )?;
            Ok(())
        })?)
    }

    fn check(&mut self, device: &CudaDevice, metadata: [i32; 4]) -> TestResult {
        device.copy_host_slice(&mut self.metadata, &metadata)?;
        let reference = self.graph(device, false)?;
        let candidate = self.graph(device, true)?;
        let blocks = self.capacity / KV_BLOCK_TOKENS;
        // Update the captured table, so a capture-time inverse or cached assignment fails.
        for phase in [1, 13] {
            let mut assignment: Vec<_> = (0..blocks)
                .map(|i| i32::try_from((i * 37 + phase) % blocks).unwrap())
                .collect();
            let end = (metadata[0] + metadata[1] + metadata[2]).max(1);
            let used = usize::try_from(end)?.div_ceil(KV_BLOCK_TOKENS);
            assert_eq!(
                assignment[..used].iter().collect::<BTreeSet<_>>().len(),
                used
            );
            // SlotPool::assign pads unreachable entries with the first leased page.
            let padding = assignment[0];
            assignment[used..].fill(padding);
            device.copy_host_slice(&mut self.table, &assignment)?;
            reference.launch().sync_on(&device.stream)?;
            candidate.launch().sync_on(&device.stream)?;
            for (expected, actual) in self.reference.iter().zip(&self.candidate) {
                let expected = device.read_borrowed(expected)?;
                let actual = device.read_borrowed(actual)?;
                assert!(
                    expected.iter().zip(actual).all(|(a, b)| a.0 == b.0),
                    "capacity {} metadata {metadata:?} phase {phase}",
                    self.capacity
                );
            }
        }
        Ok(())
    }
}

#[test]
#[ignore = "release CUDA performance diagnostic; run inside safe-run"]
fn physical_page_append_paired_identity_and_fragmented_cost() -> TestResult {
    use cutile::{
        bench::{BenchOptions, do_bench_paired},
        cutile_compiler::hints::CompileOptions,
        error::tensor_error,
    };
    use std::time::Duration;
    if cfg!(debug_assertions) || CompileOptions::default().device_debug {
        return Err("release O3 performance diagnostic required".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let mut fixture = Fixture::new(&device, 71200, 256, 1024)?;
    device.copy_host_slice(&mut fixture.metadata, &[0, 1024, 0, 0])?;
    let baseline = fixture.graph(&device, false)?;
    let candidate = fixture.graph(&device, true)?;
    let blocks = fixture.capacity / KV_BLOCK_TOKENS;
    for fragmented in [false, true] {
        let table: Vec<_> = (0..blocks)
            .map(|i| {
                let page = if fragmented { (i * 37 + 1) % blocks } else { i };
                i32::try_from(page).unwrap()
            })
            .collect();
        device.copy_host_slice(&mut fixture.table, &table)?;
        for clear_l2 in [true, false] {
            let options = BenchOptions {
                warmup: Duration::from_millis(30),
                rep: Duration::from_millis(150),
                min_reps: 30,
                max_reps: 1000,
                clear_l2,
            };
            for reverse in [false, true] {
                let run = |graph: &Graph| {
                    graph
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                };
                let (a, b) = if reverse {
                    let (b, a) = do_bench_paired(
                        &device.stream,
                        &options,
                        |_| run(&candidate),
                        |_| run(&baseline),
                    )?;
                    (a, b)
                } else {
                    do_bench_paired(
                        &device.stream,
                        &options,
                        |_| run(&baseline),
                        |_| run(&candidate),
                    )?
                };
                println!(
                    "{}",
                    serde_json::json!({
                        "diagnostic_only":true, "fragmented":fragmented, "clear_l2":clear_l2,
                        "reverse":reverse, "baseline_ms":a.times_ms(), "candidate_ms":b.times_ms(),
                        "baseline_median_us":a.median_ms()*1000.0,
                        "candidate_median_us":b.median_ms()*1000.0,
                    })
                );
            }
        }
    }
    Ok(())
}
