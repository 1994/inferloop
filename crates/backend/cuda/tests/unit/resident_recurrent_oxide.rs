use super::{
    kernels,
    oxide::{Bindings, Support},
    tests::reference_fixture,
};
use crate::device::{CudaDevice, device_error};
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Graph;
const KH: usize = 16;
const VH: usize = 48;
const DIM: usize = 128;
#[test]
#[ignore = "requires CUDA/safe-run; full original F32 state/output and independent F64"]
fn rust_gdn_complete_state_output_gate() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let function = Support::load(&device)?;
    for width in [64, 256] {
        for (count, base, offset) in [
            (0, 0, 0),
            (1, 0, 0),
            (51, 0, 0),
            (52, 17, -18),
            (width - 1, 0, 0),
            (width, 0, -1),
            (width, 128, 0),
            (width, 0, 0),
        ] {
            check(&device, &function, width, count, base, offset)?;
        }
    }
    Ok(())
}
fn check(
    device: &CudaDevice,
    function: &Support,
    lanes: usize,
    count: usize,
    base: i32,
    offset: i32,
) -> Result<()> {
    let f = reference_fixture((KH, VH, DIM, lanes), count, base + offset)
        .map_err(|e| Error::invalid(e.to_string()))?;
    let qkv = device.upload(f.qkv, &[lanes, 2 * KH + VH, DIM])?;
    let beta = device.upload(f.beta, &[lanes, VH])?;
    let alpha = device.upload(f.alpha, &[lanes, VH])?;
    let a_log = device.upload(f.a_log, &[VH])?;
    let bias = device.upload(f.bias, &[VH])?;
    let metadata = device.upload(
        vec![base, i32::try_from(count).map_err(device_error)?, offset, 0],
        &[4],
    )?;
    let mut normalized = zeros(device, &[lanes, 2 * KH, DIM])?;
    let mut betas = zeros(device, &[lanes, VH])?;
    let mut decays = zeros(device, &[lanes, VH])?;
    kernels::normalize_shared((&mut normalized).partition([1, 1, DIM]), &qkv)
        .generics(vec![KH.to_string(), DIM.to_string()])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    kernels::gates_shared(
        (&mut betas).partition([1, 1]),
        (&mut decays).partition([1, 1]),
        &beta,
        &alpha,
        &a_log,
        &bias,
    )
    .sync_on(&device.stream)
    .map_err(device_error)?;
    let initial = device.upload(f.initial, &[VH, DIM, DIM])?;
    let mut native_state = zeros(device, &[VH, DIM, DIM])?;
    let mut rust_state = zeros(device, &[VH, DIM, DIM])?;
    api::memcpy(&mut native_state, &initial)
        .sync_on(&device.stream)
        .map_err(device_error)?;
    api::memcpy(&mut rust_state, &initial)
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut native_output = zeros(device, &[VH, lanes, DIM])?;
    let mut rust_output = zeros(device, &[VH, lanes, DIM])?;
    kernels::delta_precomputed(
        (&mut native_state).partition([1, DIM, 16]),
        (&mut native_output).partition([1, lanes, 16]),
        &qkv,
        &normalized,
        &betas,
        &decays,
        &metadata,
    )
    .generics(vec![
        KH.to_string(),
        VH.to_string(),
        DIM.to_string(),
        lanes.to_string(),
        "16".into(),
    ])
    .sync_on(&device.stream)
    .map_err(device_error)?;
    let graph = Graph::scope(&device.stream, |scope| {
        function.record(
            scope,
            Bindings {
                state: &mut rust_state,
                output: &mut rust_output,
                qkv: &qkv,
                normalized: &normalized,
                beta: &betas,
                decay: &decays,
                metadata: &metadata,
            },
            lanes,
        )
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    compare(
        device,
        (lanes, count, base, offset),
        [
            ("state", &rust_state, &native_state, &f.expected),
            ("output", &rust_output, &native_output, &f.outputs),
        ],
    )
}

type ResultPair<'a> = (&'a str, &'a Tensor<f32>, &'a Tensor<f32>, &'a [f64]);
fn compare(
    device: &CudaDevice,
    (lanes, count, base, offset): (usize, usize, i32, i32),
    pairs: [ResultPair<'_>; 2],
) -> Result<()> {
    let mut elements = 0;
    let mut maximum = 0.0_f64;
    for (name, rust, native, reference) in pairs {
        let a = device.read_borrowed(rust)?;
        let b = device.read_borrowed(native)?;
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), reference.len());
        for (index, ((&a, &b), &reference)) in a.iter().zip(&b).zip(reference).enumerate() {
            assert!(a.is_finite() && b.is_finite() && reference.is_finite());
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "Rust GDN {name} lanes={lanes} count={count} offset={offset} at {index}"
            );
            let error = (f64::from(a) - reference).abs();
            maximum = maximum.max(error);
            assert!(error < 2e-5, "GDN F64 {name} at {index}: {error}");
        }
        elements += a.len();
    }
    println!(
        "{}",
        serde_json::json!({"rust_gdn_numeric":true,"lanes":lanes,"count":count,"base":base,
  "offset":offset,"elements":elements,"different_bits":0,"max_F64_error":maximum})
    );
    Ok(())
}

fn zeros(device: &CudaDevice, shape: &[usize]) -> Result<Tensor<f32>> {
    api::zeros::<f32>(shape)
        .sync_on(&device.stream)
        .map_err(device_error)
}

struct Cost {
    qkv: Arc<Tensor<f32>>,
    beta: Arc<Tensor<f32>>,
    alpha: Arc<Tensor<f32>>,
    a_log: Arc<Tensor<f32>>,
    bias: Arc<Tensor<f32>>,
    metadata: Arc<Tensor<i32>>,
    initial: Arc<Tensor<f32>>,
    normalized: Tensor<f32>,
    betas: Tensor<f32>,
    decays: Tensor<f32>,
    state: Tensor<f32>,
    scratch: Tensor<f32>,
    output: Tensor<f32>,
    width: usize,
}
impl Cost {
    fn new(device: &CudaDevice, width: usize, count: usize) -> Result<Self> {
        let f = reference_fixture((KH, VH, DIM, width), count, 0)
            .map_err(|e| Error::invalid(e.to_string()))?;
        Ok(Self {
            qkv: device.upload(f.qkv, &[width, 2 * KH + VH, DIM])?,
            beta: device.upload(f.beta, &[width, VH])?,
            alpha: device.upload(f.alpha, &[width, VH])?,
            a_log: device.upload(f.a_log, &[VH])?,
            bias: device.upload(f.bias, &[VH])?,
            metadata: device.upload(
                vec![0, i32::try_from(count).map_err(device_error)?, 0, 0],
                &[4],
            )?,
            initial: device.upload(f.initial, &[VH, DIM, DIM])?,
            normalized: zeros(device, &[width, 2 * KH, DIM])?,
            betas: zeros(device, &[width, VH])?,
            decays: zeros(device, &[width, VH])?,
            state: zeros(device, &[VH, DIM, DIM])?,
            scratch: zeros(device, &[VH, width, DIM])?,
            output: zeros(device, &[width, VH, DIM])?,
            width,
        })
    }
    fn graph(&mut self, device: &CudaDevice, rust: Option<&Support>) -> Result<Graph> {
        let width = self.width;
        Graph::scope(&device.stream, |scope| {
            scope.record(api::memcpy(&mut self.state, &self.initial))?;
            scope.record(
                kernels::normalize_shared((&mut self.normalized).partition([1, 1, DIM]), &self.qkv)
                    .generics(vec![KH.to_string(), DIM.to_string()]),
            )?;
            scope.record(kernels::gates_shared(
                (&mut self.betas).partition([1, 1]),
                (&mut self.decays).partition([1, 1]),
                &self.beta,
                &self.alpha,
                &self.a_log,
                &self.bias,
            ))?;
            if let Some(function) = rust {
                function.record(
                    scope,
                    Bindings {
                        state: &mut self.state,
                        output: &mut self.scratch,
                        qkv: &self.qkv,
                        normalized: &self.normalized,
                        beta: &self.betas,
                        decay: &self.decays,
                        metadata: &self.metadata,
                    },
                    width,
                )?;
            } else {
                scope.record(
                    kernels::delta_precomputed(
                        (&mut self.state).partition([1, DIM, 16]),
                        (&mut self.scratch).partition([1, width, 16]),
                        &self.qkv,
                        &self.normalized,
                        &self.betas,
                        &self.decays,
                        &self.metadata,
                    )
                    .generics(vec![
                        KH.to_string(),
                        VH.to_string(),
                        DIM.to_string(),
                        width.to_string(),
                        "16".into(),
                    ]),
                )?;
            }
            scope.record(
                kernels::transpose((&mut self.output).partition([1, 1, DIM]), &self.scratch)
                    .generics(vec![DIM.to_string()]),
            )?;
            Ok(())
        })
        .map_err(device_error)
    }
    fn verify(&self, device: &CudaDevice, native: &Graph, rust: &Graph) -> Result<()> {
        native
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let expected = [
            device.read_borrowed(&self.state)?,
            device.read_borrowed(&self.output)?,
        ];
        rust.launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let actual = [
            device.read_borrowed(&self.state)?,
            device.read_borrowed(&self.output)?,
        ];
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter()
                    .zip(b)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits()),
                "complete timed pipeline bits"
            );
        }
        Ok(())
    }
}
#[test]
#[ignore = "requires CUDA and safe-run; gated complete GDN pipeline paired timing"]
fn rust_gdn_complete_pipeline_cost() -> Result<()> {
    use cutile::bench::{BenchOptions, do_bench_paired};
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err(Error::invalid("GDN benchmark requires release"));
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let function = Support::load(&device)?;
    for (width, count) in [(64, 51), (64, 64), (256, 51), (256, 255), (256, 256)] {
        let mut data = Cost::new(&device, width, count)?;
        let native = data.graph(&device, None)?;
        let rust = data.graph(&device, Some(&function))?;
        data.verify(&device, &native, &rust)?;
        for round in 0..2 {
            for clear_l2 in [true, false] {
                let reverse = round == 1;
                let (left, right) = if reverse {
                    (&rust, &native)
                } else {
                    (&native, &rust)
                };
                let options = BenchOptions {
                    warmup: Duration::from_millis(50),
                    rep: Duration::from_millis(100),
                    min_reps: 10,
                    max_reps: 30,
                    clear_l2,
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
                let (baseline, candidate) = if reverse {
                    (b.times_ms(), a.times_ms())
                } else {
                    (a.times_ms(), b.times_ms())
                };
                println!(
                    "{}",
                    serde_json::json!({"rust_gdn_cost":true,"width":width,"count":count,
                    "round":round,"candidate_first":reverse,"clear_l2":clear_l2,
                    "baseline_ms":baseline,"candidate_ms":candidate})
                );
            }
        }
        data.verify(&device, &native, &rust)?;
    }
    Ok(())
}
