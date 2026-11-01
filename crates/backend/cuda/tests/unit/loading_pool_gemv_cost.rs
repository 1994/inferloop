use super::{LoadOptions, LoadedModel};
use crate::constants::SMALL_GEMM_TILE;
use crate::kernels::linear::dense;
use crate::strategy::LinearTiling;
use crate::{
    device::CudaDevice,
    mlp::ProjectionWeight,
    resident::{pool_gemv, small_gemm_kernels},
};
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    half::bf16,
    prelude::*,
};
use infer_core::{Error, ModelId, Result};
use infer_cuda_kernels::tile::runtime::Graph;
use infer_ir::TensorStorage;
use std::env::var;
use std::error::Error as StdError;
use std::result::Result as StdResult;
use std::time::Duration;
type TestResult<T = ()> = StdResult<T, Box<dyn StdError>>;

struct Case {
    name: String,
    weight: Arc<Tensor<bf16>>,
    tile: LinearTiling,
    joined: Arc<Tensor<f32>>,
    rows: Vec<Arc<Tensor<f32>>>,
    reference: Vec<Tensor<f32>>,
    original: Tensor<f32>,
    candidate: Tensor<f32>,
    k: usize,
    n: usize,
}
impl Case {
    fn new(
        device: &CudaDevice,
        name: String,
        weight: Arc<Tensor<bf16>>,
        tile: LinearTiling,
        rows: usize,
    ) -> TestResult<Self> {
        let n = usize::try_from(weight.shape()[0])?;
        let k = usize::try_from(weight.shape()[1])?;
        Ok(Self {
            name,
            weight,
            tile,
            k,
            n,
            joined: device.upload(vec![0.0; rows * k], &[rows, k])?,
            rows: (0..rows)
                .map(|_| device.upload(vec![0.0; k], &[k]))
                .collect::<Result<_>>()?,
            reference: (0..rows)
                .map(|_| api::zeros::<f32>(&[n]).sync_on(&device.stream))
                .collect::<StdResult<_, _>>()?,
            original: api::zeros::<f32>(&[rows, n]).sync_on(&device.stream)?,
            candidate: api::zeros::<f32>(&[rows, n]).sync_on(&device.stream)?,
        })
    }
    fn set(&mut self, device: &CudaDevice, fixture: usize) -> TestResult {
        let rows = self.rows.len();
        let k = self.k;
        let data: Vec<_> = (0..rows * k)
            .map(|i| {
                let v = (f32::from(u16::try_from(i * 17 % 131).unwrap_or_default()) - 65.0) / 37.0;
                match fixture {
                    1 => 0.0,
                    2 => -0.0,
                    3 => v * 1.0e-30,
                    4 => v * 80.0,
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
                    _ => v,
                }
            })
            .collect();
        let upload = device.upload(data.clone(), &[rows, k])?;
        let joined = Arc::get_mut(&mut self.joined).ok_or("joined alias during upload")?;
        api::memcpy(joined, &upload).sync_on(&device.stream)?;
        for (row, input) in self.rows.iter_mut().enumerate() {
            let upload = device.upload(data[row * k..(row + 1) * k].to_vec(), &[k])?;
            let input = Arc::get_mut(input).ok_or("row alias during upload")?;
            api::memcpy(input, &upload).sync_on(&device.stream)?;
        }
        Ok(())
    }
    fn reference(&mut self, scope: &Scope) -> StdResult<(), DeviceError> {
        for (input, output) in self.rows.iter().zip(&mut self.reference) {
            scope.record(
                dense(output.partition([self.tile.rows()]), input, &self.weight).generics(vec![
                    bf16::DTYPE.as_str().into(),
                    self.tile.rows().to_string(),
                    self.tile.columns().to_string(),
                    self.k.to_string(),
                ]),
            )?;
        }
        Ok(())
    }
    fn original(&mut self, scope: &Scope) -> StdResult<(), DeviceError> {
        scope.record(
            small_gemm_kernels::dense(
                (&mut self.original).partition(SMALL_GEMM_TILE),
                &self.joined,
                &self.weight,
            )
            .generics(vec![bf16::DTYPE.as_str().into(), self.k.to_string()]),
        )?;
        Ok(())
    }
    fn shared(&mut self, scope: &Scope) -> StdResult<(), DeviceError> {
        use crate::resident::linear_batch::batched;
        scope.record(
            batched::dense(
                (&mut self.candidate).partition([4, self.tile.rows()]),
                &self.joined,
                &self.weight,
            )
            .generics(vec![
                bf16::DTYPE.as_str().into(),
                self.tile.rows().to_string(),
                self.tile.columns().to_string(),
                self.k.to_string(),
            ]),
        )?;
        Ok(())
    }
    fn candidate(&mut self, scope: &Scope) -> StdResult<(), DeviceError> {
        scope.record(
            pool_gemv::kernels::dense(
                (&mut self.candidate).partition([1, self.tile.rows()]),
                &self.joined,
                &self.weight,
            )
            .generics(vec![
                bf16::DTYPE.as_str().into(),
                self.tile.rows().to_string(),
                self.tile.columns().to_string(),
                self.k.to_string(),
            ]),
        )?;
        Ok(())
    }
    fn scalar_reference(&self, device: &CudaDevice) -> TestResult {
        let input = device.read(&self.joined)?;
        let weights = device.read(&self.weight)?;
        let actual = device.read_borrowed(&self.candidate)?;
        assert_eq!(actual.len(), self.rows.len() * self.n);
        for (row, input) in input.chunks_exact(self.k).enumerate() {
            for (column, weight) in weights.chunks_exact(self.k).enumerate() {
                let expected: f64 = input
                    .iter()
                    .zip(weight)
                    .map(|(a, b)| f64::from(*a) * f64::from(b.to_f32()))
                    .sum();
                let value = f64::from(actual[row * self.n + column]);
                assert!(value.is_finite() && expected.is_finite());
                assert!(
                    (value - expected).abs() <= expected.abs().mul_add(2.0e-5, 5.0e-4),
                    "{} row={row} column={column}: {value} != scalar {expected}",
                    self.name
                );
            }
        }
        Ok(())
    }
    fn equal(&self, device: &CudaDevice) -> TestResult {
        let actual = device.read_borrowed(&self.candidate)?;
        for (row, output) in self.reference.iter().enumerate() {
            let expected = device.read_borrowed(output)?;
            assert!(
                actual[row * self.n..(row + 1) * self.n]
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits()),
                "private GEMV bits {} row{row}",
                self.name
            );
        }
        Ok(())
    }
}
fn cases(loaded: &LoadedModel, rows: usize) -> TestResult<Vec<Case>> {
    let cases = limited_cases(loaded, rows, usize::MAX)?;
    assert_eq!(cases.len(), 96);
    Ok(cases)
}
fn limited_cases(loaded: &LoadedModel, rows: usize, limit: usize) -> TestResult<Vec<Case>> {
    let mut cases = Vec::new();
    for (id, projection) in &loaded.weights.projections {
        if cases.len() >= limit {
            break;
        }
        let ProjectionWeight::Dense(weight) = projection else {
            continue;
        };
        let n = usize::try_from(weight.shape()[0])?;
        let k = usize::try_from(weight.shape()[1])?;
        if n > 128 || k < 4096 {
            continue;
        }
        let spec = loaded
            .graph
            .tensors
            .iter()
            .find(|t| t.id == *id)
            .ok_or("weight spec")?;
        let TensorStorage::Weight { slot } = &spec.storage else {
            return Err("weight name".into());
        };
        let tile = loaded
            .weights
            .tiling
            .get(id)
            .copied()
            .ok_or("private tiling")?;
        cases.push(Case::new(
            loaded.device(),
            slot.clone(),
            Arc::clone(weight),
            tile,
            rows,
        )?);
    }
    Ok(cases)
}
fn run_group(loaded: &LoadedModel, rows: usize) -> TestResult {
    let device = loaded.device();
    let mut cases = cases(loaded, rows)?;
    let reference = Graph::scope(&device.stream, |scope| {
        for case in &mut cases {
            case.reference(scope)?;
        }
        Ok(())
    })?;
    let old = Graph::scope(&device.stream, |scope| {
        for case in &mut cases {
            case.original(scope)?;
        }
        Ok(())
    })?;
    let new = Graph::scope(&device.stream, |scope| {
        for case in &mut cases {
            case.candidate(scope)?;
        }
        Ok(())
    })?;
    for fixture in 0..8 {
        for case in &mut cases {
            case.set(device, fixture)?;
        }
        reference.launch().sync_on(&device.stream)?;
        new.launch().sync_on(&device.stream)?;
        for case in &cases {
            case.equal(device)?;
        }
    }
    println!(
        "{}",
        serde_json::json!({"rows":rows,"weights":96,"fixtures":8,
        "private_bit_differences":0,"tiles":cases.iter().map(|c|[c.tile.rows(),c.tile.columns()]).collect::<Vec<_>>()})
    );
    for case in &mut cases {
        case.set(device, 0)?;
    }
    for reverse in [false, true] {
        for cold in [false, true] {
            let opts = BenchOptions {
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
            let (a, b) = do_bench_paired(&device.stream, &opts, |_| run(first), |_| run(second))?;
            let (a, b) = if reverse { (b, a) } else { (a, b) };
            println!(
                "{}",
                serde_json::json!({"rows":rows,"weights":96,"reverse":reverse,"cold":cold,
                "old_ms":a.median_ms(),"new_ms":b.median_ms(),"speedup":a.median_ms()/b.median_ms(),
                "old_samples":a.times_ms(),"new_samples":b.times_ms(),"serving_admission":false})
            );
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires CUDA/checkpoint/safe-run; all96 real small weights, original private GEMV bits"]
fn checkpoint_row_gemv_bits_and_cost() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            autotune: false,
            mtp_depth: 0,
            prefill_width: 64,
            ..LoadOptions::default()
        },
    )?;
    for rows in [1, 2, 3, 4, 8, 16, 18, 36, 48, 144] {
        run_group(&loaded, rows)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA/checkpoint/safe-run; bounded exact row-tiled crossover probe"]
fn bounded_pool_dense_crossover_probe() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL")?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            autotune: false,
            mtp_depth: 0,
            prefill_width: 64,
            ..LoadOptions::default()
        },
    )?;
    for rows in [1, 4, 16, 17, 24, 45, 64] {
        probe_rows(&loaded, rows)?;
    }
    Ok(())
}
fn probe_rows(loaded: &LoadedModel, rows: usize) -> TestResult {
    let device = loaded.device();
    let mut cases = limited_cases(loaded, rows, 2)?;
    assert_eq!(cases.len(), 2);
    let reference = Graph::scope(&device.stream, |scope| {
        for case in &mut cases {
            case.reference(scope)?;
        }
        Ok(())
    })?;
    let gemv = Graph::scope(&device.stream, |scope| {
        for case in &mut cases {
            case.candidate(scope)?;
        }
        Ok(())
    })?;
    let shared = Graph::scope(&device.stream, |scope| {
        for case in &mut cases {
            case.shared(scope)?;
        }
        Ok(())
    })?;
    let native = Graph::scope(&device.stream, |scope| {
        for case in &mut cases {
            case.original(scope)?;
        }
        Ok(())
    })?;
    for fixture in 0..8 {
        for case in &mut cases {
            case.set(device, fixture)?;
        }
        reference.launch().sync_on(&device.stream)?;
        gemv.launch().sync_on(&device.stream)?;
        for case in &cases {
            case.equal(device)?;
            case.scalar_reference(device)?;
        }
        shared.launch().sync_on(&device.stream)?;
        for case in &cases {
            case.equal(device)?;
            case.scalar_reference(device)?;
        }
    }
    for case in &mut cases {
        case.set(device, 0)?;
    }
    let baseline = if rows <= 16 { &gemv } else { &native };
    for (name, candidate) in [("gemv", &gemv), ("shared4", &shared)] {
        for reverse in [false, true] {
            for cold in [true, false] {
                let options = BenchOptions {
                    warmup: Duration::from_millis(50),
                    rep: Duration::from_millis(100),
                    min_reps: 10,
                    max_reps: 30,
                    clear_l2: cold,
                };
                let run = |graph: &Graph| {
                    graph
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                };
                let (first, second) = if reverse {
                    (candidate, baseline)
                } else {
                    (baseline, candidate)
                };
                let (a, b) =
                    do_bench_paired(&device.stream, &options, |_| run(first), |_| run(second))?;
                let (a, b) = if reverse { (b, a) } else { (a, b) };
                println!(
                    "{}",
                    serde_json::json!({"rows":rows,"weights":2,"candidate":name,
                    "reverse":reverse,"cold":cold,"baseline_ms":a.times_ms(),
                    "candidate_ms":b.times_ms(),"candidate_ratio":b.median_ms()/a.median_ms(),
                    "complete_private_bits_equal":true,"serving_admission":false})
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA/checkpoint/private tuning cache; actual geometry selection and complete outputs"]
fn checkpoint_selector_measures_and_reuses_complete_choices() -> TestResult {
    use crate::tuning::{
        pool_dense, resident_gemm::ResidentTuningSession, resolve_table, table_path,
    };
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            autotune: false,
            mtp_depth: 0,
            prefill_width: 64,
            ..LoadOptions::default()
        },
    )?;
    let session = ResidentTuningSession::default();
    for rows in [4, 17, 24, 45, 64] {
        let mut cases = limited_cases(&loaded, rows, 1)?;
        assert_eq!(cases.len(), 1);
        let case = &mut cases[0];
        let shape = [rows, case.n, case.k];
        case.set(loaded.device(), 0)?;
        session.plan_projection(
            pool_dense::kind(case.tile),
            shape,
            pool_dense::search_trials(shape),
        )?;
        let selected = pool_dense::select(
            loaded.device(),
            &case.weight,
            shape,
            case.tile,
            true,
            &session,
        );
        let table = resolve_table(table_path().as_deref())?.ok_or("private tuning table absent")?;
        let entry = table
            .resident_pool_dense
            .iter()
            .find(|entry| {
                [entry.key.native.m, entry.key.native.n, entry.key.native.k] == shape
                    && entry.key.tile == [case.tile.rows(), case.tile.columns()]
            })
            .ok_or("complete crossover evidence absent")?;
        assert!(entry.valid());
        assert_eq!(entry.selected, selected);
        let reference = Graph::scope(&loaded.device().stream, |scope| case.reference(scope))?;
        let graph = Graph::scope(&loaded.device().stream, |scope| {
            let input = case.joined.view(&[rows, case.k])?;
            selected.record(
                scope,
                &case.weight,
                &input,
                &mut case.candidate,
                case.k,
                case.tile,
            )
        })?;
        reference.launch().sync_on(&loaded.device().stream)?;
        graph.launch().sync_on(&loaded.device().stream)?;
        if selected != pool_dense::Choice::Native {
            case.equal(loaded.device())?;
        }
        case.scalar_reference(loaded.device())?;
        assert_eq!(
            pool_dense::select(
                loaded.device(),
                &case.weight,
                shape,
                case.tile,
                false,
                &session
            ),
            selected
        );
        println!(
            "{}",
            serde_json::json!({"pool_dense_selection":shape, "tile":entry.key.tile,
            "selected":selected,"cold_speedup":entry.confirmation.cold.speedup,
            "warm_speedup":entry.confirmation.warm.speedup,"complete_output_verified":true})
        );
    }
    Ok(())
}
