use super::{LoadOptions, LoadedModel};
use crate::constants::SMALL_GEMM_TILE;
use crate::kernels::linear::dense;
use crate::strategy::LinearTiling;
use crate::tuning::pool_gemv::select as select_pool_gemv;
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
    let mut cases = Vec::new();
    for (id, projection) in &loaded.weights.projections {
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
    assert_eq!(cases.len(), 96);
    Ok(cases)
}
fn run_group(loaded: &LoadedModel, rows: usize) -> TestResult {
    let device = loaded.device();
    let mut cases = cases(loaded, rows)?;
    let reference = CudaGraph::scope(&device.stream, |scope| {
        for case in &mut cases {
            case.reference(scope)?;
        }
        Ok(())
    })?;
    let old = CudaGraph::scope(&device.stream, |scope| {
        for case in &mut cases {
            case.original(scope)?;
        }
        Ok(())
    })?;
    let new = CudaGraph::scope(&device.stream, |scope| {
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
            let run = |g: &CudaGraph<()>| {
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
#[ignore = "requires CUDA/checkpoint/safe-run; actual Dense weights drive the production crossover tuner"]
fn checkpoint_pool_gemv_tuner_uses_actual_weights() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            autotune: true,
            mtp_depth: 0,
            prefill_width: 64,
            ..LoadOptions::default()
        },
    )?;
    let mut seen = 0usize;
    for rows in [1, 3, 16, 18] {
        for (id, projection) in &loaded.weights.projections {
            let ProjectionWeight::Dense(weight) = projection else {
                continue;
            };
            let n = usize::try_from(weight.shape()[0])?;
            let k = usize::try_from(weight.shape()[1])?;
            if n > 128 || k < 4096 {
                continue;
            }
            let tile = loaded
                .weights
                .tiling
                .get(id)
                .copied()
                .ok_or("private tiling")?;
            let choice = select_pool_gemv(
                loaded.device(),
                projection,
                (rows, k, n),
                tile,
                true,
                &loaded.weights.resident_tuning,
            )?;
            eprintln!("POOL_REAL rows={rows} n={n} k={k} choice={choice:?}");
            seen += 1;
        }
    }
    assert!(
        seen > 0,
        "real checkpoint exposed no eligible Dense projections"
    );
    Ok(())
}
