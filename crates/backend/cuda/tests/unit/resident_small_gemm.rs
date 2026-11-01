use super::kernels;
use crate::device::CudaDevice;
use crate::kernels::linear::dense;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::{half::bf16, prelude::*};
use infer_cuda_kernels::tile::runtime::Graph;
use std::env::var;
use std::error::Error;

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn small_mma_matches_f64_oracle() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (m, n, k) in [(2, 65, 144), (4, 32, 256), (12, 129, 512), (17, 33, 128)] {
        let input: Vec<f32> = (0..m * k)
            .map(|i| (f32::from(u16::try_from(i * 17 % 131).unwrap()) - 65.0) / 37.0)
            .collect();
        let weights: Vec<bf16> = (0..n * k)
            .map(|i| bf16::from_f32((f32::from(u16::try_from(i * 7 % 127).unwrap()) - 63.0) / 41.0))
            .collect();
        let x = device.upload(input.clone(), &[m, k])?;
        let w = device.upload(weights.clone(), &[n, k])?;
        let output = kernels::dense(api::zeros::<f32>(&[m, n]).partition([16, 32]), &x, &w)
            .generics(vec![bf16::DTYPE.as_str().into(), k.to_string()])
            .first()
            .unpartition()
            .sync_on(&device.stream)?;
        let actual = output.to_host_vec().sync_on(&device.stream)?;
        for row in 0..m {
            for col in 0..n {
                let expected: f64 = (0..k)
                    .map(|i| {
                        f64::from(input[row * k + i]) * f64::from(weights[col * k + i].to_f32())
                    })
                    .sum();
                let error = (f64::from(actual[row * n + col]) - expected).abs();
                assert!(
                    error < expected.abs().mul_add(0.00002, 0.0005),
                    "{m}x{n}x{k} [{row},{col}] {} != {expected}",
                    actual[row * n + col]
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn small_mma_performance() -> Result<(), Box<dyn Error>> {
    use cutile::bench::{BenchOptions, do_bench_paired};
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
    for (n, k) in [(4096, 4096), (14336, 4096), (4096, 14336), (151_936, 2048)] {
        let weights = device.upload(vec![bf16::from_f32(0.25); n * k], &[n, k])?;
        for m in [2, 4, 12] {
            let input = device.upload(vec![0.253_f32; m * k], &[m, k])?;
            let mut output = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
            let old = Graph::scope(&device.stream, |scope| {
                scope.record(
                    super::super::linear_batch::batched::dense(
                        (&mut output).partition([4, 16]),
                        &input,
                        &weights,
                    )
                    .generics(vec![
                        bf16::DTYPE.as_str().into(),
                        "16".into(),
                        "256".into(),
                        k.to_string(),
                    ]),
                )?;
                Ok(())
            })?;
            let new = Graph::scope(&device.stream, |scope| {
                scope.record(
                    kernels::dense((&mut output).partition([16, 32]), &input, &weights)
                        .generics(vec![bf16::DTYPE.as_str().into(), k.to_string()]),
                )?;
                Ok(())
            })?;
            let (baseline, candidate) = do_bench_paired(
                &device.stream,
                &options,
                |_| {
                    old.launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
                |_| {
                    new.launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
            )?;
            println!(
                "{m}x{n}x{k} old_ms={} new_ms={} speedup={}",
                baseline.median_ms(),
                candidate.median_ms(),
                baseline.median_ms() / candidate.median_ms()
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn small_scaled_mma_matches_f64_oracle() -> Result<(), Box<dyn Error>> {
    use cuda_core::f8e4m3fn;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (m, n, k) in [(2, 65, 144), (12, 129, 256)] {
        let input: Vec<f32> = (0..m * k)
            .map(|i| (f32::from(u16::try_from(i * 17 % 131).unwrap()) - 65.0) / 37.0)
            .collect();
        let codes: Vec<u8> = (0..n * k)
            .map(|i| 48 + u8::try_from(i % 16).unwrap() + if i % 3 == 0 { 128 } else { 0 })
            .collect();
        let decoded: Vec<f32> = codes
            .iter()
            .map(|c| {
                let magnitude = c & 127;
                let sign = if c & 128 == 0 { 1.0 } else { -1.0 };
                sign * (1.0 + f32::from(magnitude & 7) / 8.0)
                    * 2.0_f32.powi(i32::from(magnitude >> 3) - 7)
            })
            .collect();
        let scales: Vec<f32> = (0..n)
            .map(|i| f32::from(u16::try_from(i % 13 + 1).unwrap()) / 11.0)
            .collect();
        let x = device.upload(input.clone(), &[m, k])?;
        let w = device.upload(codes.into_iter().map(f8e4m3fn).collect(), &[n, k])?;
        let scale = device.upload(scales.clone(), &[n])?;
        let output = kernels::scaled(
            api::zeros::<f32>(&[m, n]).partition([16, 32]),
            &x,
            &w,
            &scale,
        )
        .generics(vec![f8e4m3fn::DTYPE.as_str().into(), k.to_string()])
        .first()
        .unpartition()
        .sync_on(&device.stream)?;
        let actual = output.to_host_vec().sync_on(&device.stream)?;
        for row in 0..m {
            for col in 0..n {
                let expected: f64 = (0..k)
                    .map(|i| f64::from(input[row * k + i]) * f64::from(decoded[col * k + i]))
                    .sum::<f64>()
                    * f64::from(scales[col]);
                assert!(
                    (f64::from(actual[row * n + col]) - expected).abs()
                        < expected.abs().mul_add(0.00002, 0.0005)
                );
            }
        }
    }
    Ok(())
}

fn checkpoint_f64_census(
    actual: &[f32],
    input: &[f32],
    weights: &[f32],
    rows: usize,
    outputs: usize,
    columns: usize,
) -> (usize, f64, serde_json::Value) {
    assert_eq!(actual.len(), rows * outputs);
    let mut failures = 0;
    let mut maximum = 0.0_f64;
    let mut first = serde_json::Value::Null;
    for row in 0..rows {
        for output in 0..outputs {
            let expected = (0..columns)
                .map(|column| {
                    f64::from(input[row * columns + column])
                        * f64::from(weights[output * columns + column])
                })
                .sum::<f64>();
            let observed = actual[row * outputs + output];
            let limit = expected.abs().mul_add(0.00002, 0.0005);
            let normalized = (f64::from(observed) - expected).abs() / limit;
            maximum = maximum.max(normalized);
            if !observed.is_finite() || normalized >= 1.0 {
                failures += 1;
                if first.is_null() {
                    first = serde_json::json!({"row":row,"output":output,"observed":observed,"expected":expected,"limit":limit,"normalized":normalized});
                }
            }
        }
    }
    (failures, maximum, first)
}

fn checkpoint_small_input(columns: usize, fixture: &str) -> Vec<f32> {
    let input: Vec<f32> = (0..4 * columns)
        .map(|i| {
            let signed = f32::from(u16::try_from((i * 17 + i / 11) % 131).unwrap()) - 65.0;
            match fixture {
                "zero" => 0.0,
                "ordinary" => signed / 37.0,
                "sparse" => {
                    if i % 97 == 0 {
                        signed / 31.0
                    } else {
                        0.0
                    }
                }
                "wide_exponent" => {
                    signed / 63.0 * 2.0_f32.powi(i32::try_from((i * 7 + i / 13) % 17).unwrap() - 8)
                }
                _ => unreachable!(),
            }
        })
        .collect();
    input
}

struct CheckpointDenseOutputs {
    multi: Vec<f32>,
    single: Vec<f32>,
    gemv: Vec<f32>,
}

fn checkpoint_small_outputs(
    device: &CudaDevice,
    weight: &Tensor<bf16>,
    input: &[f32],
    outputs: usize,
    columns: usize,
) -> Result<CheckpointDenseOutputs, Box<dyn Error>> {
    let matrix = device.upload(input.to_vec(), &[4, columns])?;
    let single = device.upload(input[..columns].to_vec(), &[1, columns])?;
    let vector = device.upload(input[..columns].to_vec(), &[columns])?;
    let multi = super::precise_kernels::dense(
        api::zeros::<f32>(&[4, outputs]).partition([16, 32]),
        &matrix,
        weight,
    )
    .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()])
    .first()
    .unpartition()
    .sync_on(&device.stream)?;
    let one = super::precise_kernels::dense(
        api::zeros::<f32>(&[1, outputs]).partition([16, 32]),
        &single,
        weight,
    )
    .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()])
    .first()
    .unpartition()
    .sync_on(&device.stream)?;
    let gemv = dense(
        api::zeros::<f32>(&[outputs]).partition([16]),
        &vector,
        weight,
    )
    .generics(vec![
        bf16::DTYPE.as_str().into(),
        "16".into(),
        "256".into(),
        columns.to_string(),
    ])
    .first()
    .unpartition()
    .sync_on(&device.stream)?;
    let multi = device.read_borrowed(&multi)?;
    let single = device.read_borrowed(&one)?;
    let gemv = device.read_borrowed(&gemv)?;
    Ok(CheckpointDenseOutputs {
        multi,
        single,
        gemv,
    })
}

fn checkpoint_small_outputs_f32(
    device: &CudaDevice,
    weight: &Tensor<bf16>,
    input: &[f32],
    outputs: usize,
    columns: usize,
) -> Result<CheckpointDenseOutputs, Box<dyn Error>> {
    let matrix = device.upload(input.to_vec(), &[4, columns])?;
    let single = device.upload(input[..columns].to_vec(), &[1, columns])?;
    let vector = device.upload(input[..columns].to_vec(), &[columns])?;
    let multi = super::super::linear_batch::batched::dense(
        api::zeros::<f32>(&[4, outputs]).partition([4, 16]),
        &matrix,
        weight,
    )
    .generics(vec![
        bf16::DTYPE.as_str().into(),
        "16".into(),
        "256".into(),
        columns.to_string(),
    ])
    .first()
    .unpartition()
    .sync_on(&device.stream)?;
    let one = super::super::linear_batch::batched::dense(
        api::zeros::<f32>(&[1, outputs]).partition([4, 16]),
        &single,
        weight,
    )
    .generics(vec![
        bf16::DTYPE.as_str().into(),
        "16".into(),
        "256".into(),
        columns.to_string(),
    ])
    .first()
    .unpartition()
    .sync_on(&device.stream)?;
    let gemv = dense(
        api::zeros::<f32>(&[outputs]).partition([16]),
        &vector,
        weight,
    )
    .generics(vec![
        bf16::DTYPE.as_str().into(),
        "16".into(),
        "256".into(),
        columns.to_string(),
    ])
    .first()
    .unpartition()
    .sync_on(&device.stream)?;
    let multi = device.read_borrowed(&multi)?;
    let single = device.read_borrowed(&one)?;
    let gemv = device.read_borrowed(&gemv)?;
    Ok(CheckpointDenseOutputs {
        multi,
        single,
        gemv,
    })
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL, safe-run; all 96 real GDN small weights, full unchanged F64 gate"]
fn checkpoint_small_dense_full_f64_and_row_equivalence() -> Result<(), Box<dyn Error>> {
    checkpoint_small_reference(false)
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL, safe-run; existing native F32 GEMV, all real weights and complete outputs"]
fn checkpoint_small_f32_full_f64_and_row_equivalence() -> Result<(), Box<dyn Error>> {
    checkpoint_small_reference(true)
}

struct CheckpointDenseWeight {
    outputs: usize,
    columns: usize,
    values: Vec<f32>,
    device: Arc<Tensor<bf16>>,
}
impl CheckpointDenseWeight {
    fn load(
        device: &CudaDevice,
        package: &infer_models::QuantizedPackage,
        source: &infer_models::DeviceWeight,
    ) -> Result<Self, Box<dyn Error>> {
        use infer_models::{TensorDtype, WeightEncoding};
        assert_eq!(source.encoding, WeightEncoding::Float);
        assert_eq!(source.data.dtype, TensorDtype::BF16);
        let [outputs, columns] = source.shape.as_slice() else {
            return Err("small checkpoint rank".into());
        };
        let (outputs, columns) = (*outputs, *columns);
        let bytes = package.read_view(&source.data, 2 * 1024 * 1024 * 1024)?;
        assert_eq!(bytes.len(), outputs * columns * 2);
        let (chunks, tail) = bytes.as_chunks::<2>();
        assert_eq!(tail.len(), 0);
        let codes: Vec<_> = chunks
            .iter()
            .map(|chunk| u16::from_le_bytes(*chunk))
            .collect();
        let weights: Vec<_> = codes
            .iter()
            .map(|bits| f32::from_bits(u32::from(*bits) << 16))
            .collect();
        assert!(weights.iter().all(|value| value.is_finite()));
        let weight = device.upload(
            codes.into_iter().map(bf16::from_bits).collect(),
            &[outputs, columns],
        )?;
        Ok(Self {
            outputs,
            columns,
            values: weights,
            device: weight,
        })
    }
}

fn checkpoint_small_reference(native: bool) -> Result<(), Box<dyn Error>> {
    use infer_models::QuantizedPackage;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let package = QuantizedPackage::open(var("INFER_TEST_MODEL")?, infer_core::ModelId::ONE)?;
    let selected: Vec<_> = package
        .weights
        .values()
        .filter(|source| {
            source.data.name.contains(".linear_attn.in_proj_a.weight")
                || source.data.name.contains(".linear_attn.in_proj_b.weight")
        })
        .collect();
    assert_eq!(selected.len(), 96, "all checkpoint small GDN projections");
    let mut reference_failures = 0;
    let mut row_differences = 0;
    let mut total_outputs = 0;
    let mut maximum = 0.0_f64;
    for source in selected {
        let CheckpointDenseWeight {
            outputs,
            columns,
            values: weights,
            device: weight,
        } = CheckpointDenseWeight::load(&device, &package, source)?;
        for fixture in ["zero", "ordinary", "sparse", "wide_exponent"] {
            let input = checkpoint_small_input(columns, fixture);
            let CheckpointDenseOutputs {
                multi,
                single,
                gemv,
            } = if native {
                checkpoint_small_outputs_f32(&device, &weight, &input, outputs, columns)?
            } else {
                checkpoint_small_outputs(&device, &weight, &input, outputs, columns)?
            };
            let multi_reference =
                checkpoint_f64_census(&multi, &input, &weights, 4, outputs, columns);
            let one_reference =
                checkpoint_f64_census(&single, &input[..columns], &weights, 1, outputs, columns);
            let gemv_reference =
                checkpoint_f64_census(&gemv, &input[..columns], &weights, 1, outputs, columns);
            if native {
                let original_bits = single
                    .iter()
                    .zip(&gemv)
                    .filter(|(left, right)| left.to_bits() != right.to_bits())
                    .count();
                println!(
                    "{}",
                    serde_json::json!({"small_f32_original_private_bit_census":true,"weight":source.data.name,"fixture":fixture,"different":original_bits,"elements":outputs})
                );
            }
            let changed = multi[..outputs]
                .iter()
                .zip(&single)
                .filter(|(multi, single)| multi.to_bits() != single.to_bits())
                .count();
            row_differences += changed;
            reference_failures += multi_reference.0 + one_reference.0 + gemv_reference.0;
            maximum = maximum
                .max(multi_reference.1)
                .max(one_reference.1)
                .max(gemv_reference.1);
            total_outputs += 6 * outputs;
            println!(
                "{}",
                serde_json::json!({"small_dense_checkpoint_reference":true,"weight":source.data.name,"shape":[outputs,columns],"fixture":fixture,"multi":multi_reference,"single":one_reference,"original_gemv":gemv_reference,"row_bit_differences":changed,"outputs":6*outputs,"oracle":"raw checkpoint BF16 bits shifted to F32, independent F64 dot","original_gate":"abs(expected)*2e-5+5e-4"})
            );
        }
    }
    println!(
        "{}",
        serde_json::json!({"small_dense_checkpoint_reference_summary":true,"native_f32":native,"weights":96,"fixtures":4,"outputs":total_outputs,"reference_failures":reference_failures,"row_bit_differences":row_differences,"max_normalized_error":maximum})
    );
    assert_eq!(reference_failures, 0, "original full F64 gate");
    assert_eq!(row_differences, 0, "same input independent of row count");
    Ok(())
}

fn checkpoint_pool_outputs(
    device: &CudaDevice,
    weight: &Tensor<bf16>,
    input: &[f32],
    outputs: usize,
    columns: usize,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let vectors = input
        .chunks_exact(columns)
        .map(|row| device.upload(row.to_vec(), &[columns]))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(vectors.len(), 4);
    let packed = device.upload(input.to_vec(), &[4, columns])?;
    let mut result = api::zeros::<f32>(&[4, outputs]).sync_on(&device.stream)?;
    let graph = Graph::scope(&device.stream, |scope| {
        scope.record(
            super::super::linear_batch::batched::dense(
                (&mut result).partition([4, 16]),
                &packed,
                weight,
            )
            .generics(vec![
                bf16::DTYPE.as_str().into(),
                "16".into(),
                "256".into(),
                columns.to_string(),
            ]),
        )?;
        Ok(())
    })?;
    graph.launch().sync_on(&device.stream)?;
    let fused = device.read_borrowed(&result)?;
    let mut private = Vec::new();
    for vector in &vectors {
        let reference = dense(
            api::zeros::<f32>(&[outputs]).partition([16]),
            vector,
            weight,
        )
        .generics(vec![
            bf16::DTYPE.as_str().into(),
            "16".into(),
            "256".into(),
            columns.to_string(),
        ])
        .first()
        .unpartition()
        .sync_on(&device.stream)?;
        private.extend(device.read_borrowed(&reference)?);
    }
    Ok((fused, private))
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL, safe-run; four slot matrix rows, all 96 weights and complete original F64/bit gates"]
fn checkpoint_pool_four_dense_full_f64_and_private_bits() -> Result<(), Box<dyn Error>> {
    use infer_models::QuantizedPackage;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let package = QuantizedPackage::open(var("INFER_TEST_MODEL")?, infer_core::ModelId::ONE)?;
    let selected: Vec<_> = package
        .weights
        .values()
        .filter(|source| {
            source.data.name.contains(".linear_attn.in_proj_a.weight")
                || source.data.name.contains(".linear_attn.in_proj_b.weight")
        })
        .collect();
    assert_eq!(selected.len(), 96);
    let mut failures = 0;
    let mut changed = 0;
    let mut count = 0;
    let mut maximum = 0.0_f64;
    for source in selected {
        let weight = CheckpointDenseWeight::load(&device, &package, source)?;
        for fixture in ["zero", "ordinary", "sparse", "wide_exponent"] {
            let input = checkpoint_small_input(weight.columns, fixture);
            let (fused, private) = checkpoint_pool_outputs(
                &device,
                &weight.device,
                &input,
                weight.outputs,
                weight.columns,
            )?;
            let census = checkpoint_f64_census(
                &fused,
                &input,
                &weight.values,
                4,
                weight.outputs,
                weight.columns,
            );
            let bits = fused
                .iter()
                .zip(&private)
                .filter(|(left, right)| left.to_bits() != right.to_bits())
                .count();
            failures += census.0;
            changed += bits;
            count += fused.len();
            maximum = maximum.max(census.1);
            println!(
                "{}",
                serde_json::json!({"pool_four_reference":true,"weight":source.data.name,"fixture":fixture,"shape":[4,weight.outputs,weight.columns],"full_F64":census,"original_private_bit_differences":bits,"outputs":fused.len()})
            );
        }
    }
    println!(
        "{}",
        serde_json::json!({"pool_four_reference_summary":true,"weights":96,"fixtures":4,"outputs":count,"F64_failures":failures,"original_private_bit_differences":changed,"max_normalized_error":maximum})
    );
    assert_eq!(failures, 0, "original independent full F64 gate");
    assert_eq!(
        changed, 0,
        "all four lanes must match original private GEMV bits"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA/checkpoint/release/safe-run; CI first, original frozen tile; projection-only cost diagnostic"]
fn checkpoint_pool_small_f32_projection_cost() -> Result<(), Box<dyn Error>> {
    use cutile::bench::{BenchOptions, do_bench_paired};
    use infer_models::QuantizedPackage;
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err("cost diagnostic requires release".into());
    }
    let compile = CompileOptions::default();
    if compile.device_debug || compile.opt_level.is_some_and(|level| level != 3) {
        return Err("cost diagnostic requires CUDA O3".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let package = QuantizedPackage::open(var("INFER_TEST_MODEL")?, infer_core::ModelId::ONE)?;
    let weights = package
        .weights
        .values()
        .filter(|source| {
            source.data.name.contains(".linear_attn.in_proj_a.weight")
                || source.data.name.contains(".linear_attn.in_proj_b.weight")
        })
        .map(|source| CheckpointDenseWeight::load(&device, &package, source))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(weights.len(), 96);
    let first = &weights[0];
    assert!(
        weights
            .iter()
            .all(|weight| weight.outputs == first.outputs && weight.columns == first.columns)
    );
    let input = device.upload(
        checkpoint_small_input(first.columns, "ordinary"),
        &[4, first.columns],
    )?;
    let mut output = api::zeros::<f32>(&[4, first.outputs]).sync_on(&device.stream)?;
    let baseline = Graph::scope(&device.stream, |scope| {
        for weight in &weights {
            scope.record(
                kernels::dense((&mut output).partition([16, 32]), &input, &weight.device).generics(
                    vec![bf16::DTYPE.as_str().into(), weight.columns.to_string()],
                ),
            )?;
        }
        Ok(())
    })?;
    let candidate = Graph::scope(&device.stream, |scope| {
        for weight in &weights {
            scope.record(
                super::super::linear_batch::batched::dense(
                    (&mut output).partition([4, 16]),
                    &input,
                    &weight.device,
                )
                .generics(vec![
                    bf16::DTYPE.as_str().into(),
                    "16".into(),
                    "256".into(),
                    weight.columns.to_string(),
                ]),
            )?;
        }
        Ok(())
    })?;
    for cold in [true, false] {
        let options = BenchOptions {
            warmup: Duration::from_millis(50),
            rep: Duration::from_millis(150),
            min_reps: 20,
            max_reps: 100,
            clear_l2: cold,
        };
        let (old, new) = do_bench_paired(
            &device.stream,
            &options,
            |_| {
                baseline
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|error| tensor_error(&error.to_string()))
            },
            |_| {
                candidate
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|error| tensor_error(&error.to_string()))
            },
        )?;
        println!(
            "{}",
            serde_json::json!({"pool_small_projection_cost":true,"scope":"96 real small BF16 projections in two complete captured graphs; actual slot entry has no pack/unpack; service work outside these projections excluded","weights":96,"shape":[4,first.outputs,first.columns],"cold_l2":cold,"baseline_ms":old.median_ms(),"candidate_ms":new.median_ms(),"speedup":old.median_ms()/new.median_ms(),"service_performance_acceptance":false})
        );
    }
    Ok(())
}
