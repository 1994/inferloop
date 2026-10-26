//! Actual-checkpoint RMS normalization followed by the existing FP4 quantizer.
use crate::device::CudaDevice;
use crate::loading::{LoadOptions, LoadedModel};
use crate::mlp::ProjectionWeight;
use crate::resident::kernels::aux::norm;
use crate::resident::nvfp4_gemm::kernels::quantize_rows;
use crate::resident::{norm_fp4_columns, norm_fp4_kernel_entry as norm_fp4};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use infer_core::{Error, ModelId};
use infer_ir::TensorOp;
use std::env::var;
use std::error::Error as StdError;
use std::slice::from_ref;
use std::time::Duration;
const ROWS: usize = 1024;
const D: usize = 5120;
const B: usize = 8192;
type TestResult<T = ()> = Result<T, Box<dyn StdError>>;
type Quantized = (Tensor<f4e2m1fnx2>, Tensor<f8e4m3fn>);
type Owners = (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>);
struct Layer {
    weight: Arc<Tensor<f32>>,
    epsilon: f32,
    offset: f32,
    scale: f32,
}
fn layers(loaded: &LoadedModel) -> TestResult<Vec<Layer>> {
    let mut result = Vec::new();
    for pair in loaded.graph().nodes.windows(2) {
        let [node, next] = pair else { continue };
        let TensorOp::Norm {
            head_dim,
            epsilon,
            offset,
        } = node.op
        else {
            continue;
        };
        if head_dim != D
            || next.op != TensorOp::Linear
            || next.inputs[0] != node.outputs[0]
            || !matches!(
                loaded.weights.projections.get(&next.inputs[1]),
                Some(ProjectionWeight::Fp4(..))
            )
        {
            continue;
        }
        let weight = loaded
            .weights
            .constants
            .get(&node.inputs[1])
            .ok_or_else(|| Error::invalid("norm weight missing"))?;
        let scale = *loaded
            .weights
            .input_scales
            .get(&next.inputs[1])
            .ok_or_else(|| Error::invalid("FP4 activation scale missing"))?;
        result.push(Layer {
            weight: weight.clone(),
            epsilon,
            offset,
            scale,
        });
    }
    if result.len() != 56 {
        return Err(format!("expected 56 target norm/FP4 pairs, found {}", result.len()).into());
    }
    Ok(result)
}
fn capture(
    device: &CudaDevice,
    layers: &[Layer],
    input: &Tensor<f32>,
    temporary: &mut Tensor<f32>,
    outputs: &mut [Quantized],
    fused: bool,
) -> TestResult<CudaGraph<()>> {
    Ok(CudaGraph::scope(&device.stream, |scope| {
        for (layer, (codes, scales)) in layers.iter().zip(outputs) {
            if fused {
                scope.record(
                    norm_fp4(
                        (&mut *temporary).partition([1, B]),
                        codes.partition([1, B / 2]),
                        scales.partition([1, B / 16]),
                        input,
                        &layer.weight,
                        layer.epsilon,
                        layer.offset,
                        layer.scale,
                    )
                    .generics(
                        [D, B, B / 16, B / 2]
                            .iter()
                            .map(ToString::to_string)
                            .collect(),
                    ),
                )?;
            } else {
                scope.record(
                    norm(
                        (&mut *temporary).partition([1, B]),
                        input,
                        &layer.weight,
                        input,
                        layer.epsilon,
                        layer.offset,
                    )
                    .generics([D, B, 0].iter().map(ToString::to_string).collect()),
                )?;
                scope.record(
                    quantize_rows(
                        codes.partition([16, 256]),
                        scales.partition([16, 32]),
                        &*temporary,
                        layer.scale,
                    )
                    .generics(vec![D.to_string()]),
                )?;
            }
        }
        Ok(())
    })?)
}
fn compare(device: &CudaDevice, a: &[Owners], b: &[Owners]) -> TestResult<usize> {
    let mut elements = 0;
    for (layer, ((ac, asf), (bc, bsf))) in a.iter().zip(b).enumerate() {
        let ac = ac.to_host_vec().sync_on(&device.stream)?;
        let bc = bc.to_host_vec().sync_on(&device.stream)?;
        assert_eq!(ac.len(), ROWS * D / 2);
        assert_eq!(ac.len(), bc.len());
        for (index, (x, y)) in ac.iter().zip(&bc).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "code layer {layer} at {index}");
        }
        let asf = asf.to_host_vec().sync_on(&device.stream)?;
        let bsf = bsf.to_host_vec().sync_on(&device.stream)?;
        assert_eq!(asf.len(), ROWS * D / 16);
        assert_eq!(asf.len(), bsf.len());
        for (index, (x, y)) in asf.iter().zip(&bsf).enumerate() {
            assert!(
                x.0 & 127 != 127 && y.0 & 127 != 127,
                "nonfinite scale layer {layer} at {index}"
            );
            assert_eq!(x.0, y.0, "scale layer {layer} at {index}");
        }
        elements += ac.len() + asf.len();
    }
    Ok(elements)
}

fn compare_dense(
    device: &CudaDevice,
    layers: &[Layer],
    input: &Tensor<f32>,
    temporary: &mut Tensor<f32>,
) -> TestResult<usize> {
    let mut scratch = vec![(
        api::zeros::<f4e2m1fnx2>(&[ROWS, D / 2]).sync_on(&device.stream)?,
        api::zeros::<f8e4m3fn>(&[ROWS, D / 16]).sync_on(&device.stream)?,
    )];
    let mut elements = 0;
    for (layer_index, layer) in layers.iter().enumerate() {
        let base = capture(
            device,
            from_ref(layer),
            input,
            temporary,
            &mut scratch,
            false,
        )?;
        let candidate = capture(
            device,
            from_ref(layer),
            input,
            temporary,
            &mut scratch,
            true,
        )?;
        base.launch().sync_on(&device.stream)?;
        let expected = device.read_borrowed(temporary)?;
        candidate.launch().sync_on(&device.stream)?;
        let actual = device.read_borrowed(temporary)?;
        assert_eq!(expected.len(), ROWS * D);
        assert_eq!(actual.len(), expected.len());
        for (index, (x, y)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                x.is_finite() && y.is_finite(),
                "nonfinite layer {layer_index} at {index}"
            );
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "norm layer {layer_index} at {index}"
            );
        }
        elements += actual.len();
    }
    Ok(elements)
}

fn measure(device: &CudaDevice, a: &CudaGraph<()>, b: &CudaGraph<()>) -> TestResult {
    for cold in [true, false] {
        for reverse in [false, true] {
            let (first, second) = if reverse { (b, a) } else { (a, b) };
            let options = BenchOptions {
                warmup: Duration::from_millis(30),
                rep: Duration::from_millis(100),
                min_reps: 20,
                max_reps: 100,
                clear_l2: cold,
            };
            let (x, y) = do_bench_paired(
                &device.stream,
                &options,
                |_| {
                    first
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
                |_| {
                    second
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
            )?;
            let (base, candidate) = if reverse { (y, x) } else { (x, y) };
            println!(
                "{}",
                serde_json::json!({
                 "kind":"norm_fp4_quant_latency","layers":56,"rows":ROWS,"columns":D,
                 "cold":cold,"reverse":reverse,"baseline_ms":base.median_ms(),
                 "candidate_ms":candidate.median_ms(),"speedup":base.median_ms()/candidate.median_ms(),
                 "baseline_samples_ms":base.times_ms(),"candidate_samples_ms":candidate.times_ms(),
                 "scope":"complete norm/quantize layer set with original arithmetic and actual checkpoint norm weights"
                })
            );
        }
    }
    Ok(())
}
#[test]
#[ignore = "actual 27B checkpoint, CUDA release and protected runner"]
fn actual_norm_fp4_quant_screen() -> TestResult {
    let options = CompileOptions::default();
    if cfg!(debug_assertions) || options.device_debug || options.opt_level.is_some_and(|v| v != 3) {
        return Err("release and CUDA O3 without device debug required".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let loaded = LoadedModel::open(
        device.clone(),
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            autotune: false,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let layers = layers(&loaded)?;
    let input = device.upload(
        (0..ROWS * D)
            .map(|i| (f32::from(u16::try_from(i % 211).unwrap()) - 105.0) / 37.0)
            .collect::<Vec<_>>(),
        &[ROWS, D],
    )?;
    let mut temporary = api::zeros::<f32>(&[ROWS, D]).sync_on(&device.stream)?;
    let mut a = Vec::new();
    let mut b = Vec::new();
    for _ in &layers {
        for outputs in [&mut a, &mut b] {
            outputs.push((
                api::zeros::<f4e2m1fnx2>(&[ROWS, D / 2]).sync_on(&device.stream)?,
                api::zeros::<f8e4m3fn>(&[ROWS, D / 16]).sync_on(&device.stream)?,
            ));
        }
    }
    let ga = capture(&device, &layers, &input, &mut temporary, &mut a, false)?;
    let gb = capture(&device, &layers, &input, &mut temporary, &mut b, true)?;
    ga.launch().sync_on(&device.stream)?;
    gb.launch().sync_on(&device.stream)?;
    let a: Vec<_> = a
        .into_iter()
        .map(|(c, s)| (Arc::new(c), Arc::new(s)))
        .collect();
    let b: Vec<_> = b
        .into_iter()
        .map(|(c, s)| (Arc::new(c), Arc::new(s)))
        .collect();
    let elements = compare(&device, &a, &b)?;
    println!(
        "{}",
        serde_json::json!({
         "kind":"norm_fp4_quant_bits","layers":layers.len(),"packed_codes_and_scales":elements,
         "all_bitwise_equal":true,"new_product_scratch_bytes":0
        })
    );
    let dense_elements = compare_dense(&device, &layers, &input, &mut temporary)?;
    println!(
        "{}",
        serde_json::json!({"kind":"norm_fp4_dense_bits","full_finite_elements":dense_elements,
        "all_bitwise_equal":true})
    );
    measure(&device, &ga, &gb)?;
    Ok(())
}

fn shape_graph(
    device: &CudaDevice,
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    (dense, (codes, scales)): &mut (Tensor<f32>, Quantized),
    (columns, offset, scale): (usize, f32, f32),
    fused: bool,
) -> TestResult<CudaGraph<()>> {
    let block = columns.next_power_of_two();
    let fused = fused && norm_fp4_columns(columns);
    Ok(CudaGraph::scope(&device.stream, |scope| {
        if fused {
            scope.record(
                norm_fp4(
                    dense.partition([1, block]),
                    codes.partition([1, block / 2]),
                    scales.partition([1, block / 16]),
                    input,
                    weight,
                    1e-5,
                    offset,
                    scale,
                )
                .generics(
                    [columns, block, block / 16, block / 2]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )?;
        } else {
            scope.record(
                norm(
                    dense.partition([1, block]),
                    input,
                    weight,
                    input,
                    1e-5,
                    offset,
                )
                .generics(
                    [columns, block, 0]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )?;
            scope.record(
                quantize_rows(
                    codes.partition([16, 256]),
                    scales.partition([16, 32]),
                    &*dense,
                    scale,
                )
                .generics(vec![columns.to_string()]),
            )?;
        }
        Ok(())
    })?)
}

fn independent_rms(input: &[f32], weights: &[f32], actual: &[f32], offset: f32) {
    for (row, (input, actual)) in input
        .chunks(weights.len())
        .zip(actual.chunks(weights.len()))
        .enumerate()
    {
        let square = input.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        let inverse = (square / f64::from(u32::try_from(weights.len()).unwrap()) + 1e-5)
            .sqrt()
            .recip();
        for (index, ((&x, &w), &y)) in input.iter().zip(weights).zip(actual).enumerate() {
            let expected = f64::from(x) * inverse * (f64::from(w) + f64::from(offset));
            assert!(y.is_finite());
            assert!(
                (f64::from(y) - expected).abs() <= 2e-5 * (1.0 + expected.abs()),
                "rms row {row}, column {index}"
            );
        }
    }
}

fn shape_case(
    device: &CudaDevice,
    rows: usize,
    columns: usize,
    offset: f32,
    scale: f32,
) -> TestResult<bool> {
    let values: Vec<_> = (0..rows * columns)
        .map(|i| match i / columns {
            0 => 0.0,
            1 => 1e-20,
            _ => (f32::from(u16::try_from(i % 211).unwrap()) - 105.0) / 37.0,
        })
        .collect();
    let weights: Vec<_> = (0..columns)
        .map(|i| f32::from(u8::try_from(i % 31).unwrap()) / 16.0 - 0.5)
        .collect();
    let input = device.upload(values.clone(), &[rows, columns])?;
    let weight = device.upload(weights.clone(), &[columns])?;
    let mut output = (
        api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?,
        (
            api::zeros::<f4e2m1fnx2>(&[rows, columns / 2]).sync_on(&device.stream)?,
            api::zeros::<f8e4m3fn>(&[rows, columns / 16]).sync_on(&device.stream)?,
        ),
    );
    let baseline = shape_graph(
        device,
        &input,
        &weight,
        &mut output,
        (columns, offset, scale),
        false,
    )?;
    let fused = shape_graph(
        device,
        &input,
        &weight,
        &mut output,
        (columns, offset, scale),
        true,
    )?;
    baseline.launch().sync_on(&device.stream)?;
    let expected_dense = device.read_borrowed(&output.0)?;
    let expected_codes = device.read_borrowed(&output.1.0)?;
    let expected_scales = device.read_borrowed(&output.1.1)?;
    fused.launch().sync_on(&device.stream)?;
    let dense = device.read_borrowed(&output.0)?;
    let codes = device.read_borrowed(&output.1.0)?;
    let scales = device.read_borrowed(&output.1.1)?;
    assert_eq!(dense.len(), rows * columns);
    assert_eq!(codes.len(), rows * columns / 2);
    assert_eq!(scales.len(), rows * columns / 16);
    let dense_differences = dense
        .iter()
        .zip(&expected_dense)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let code_differences = codes
        .iter()
        .zip(&expected_codes)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let scale_differences = scales
        .iter()
        .zip(&expected_scales)
        .filter(|(a, b)| a.0 != b.0)
        .count();
    for s in &scales {
        assert!(s.0 & 127 != 127);
    }
    let all_equal = dense_differences + code_differences + scale_differences == 0;
    println!(
        "{}",
        serde_json::json!({"kind":"norm_fp4_shape_bits","rows":rows,"columns":columns,
        "dense_differences":dense_differences,"code_differences":code_differences,"scale_differences":scale_differences})
    );
    independent_rms(&values, &weights, &dense, offset);
    shape_latency(
        device,
        &baseline,
        &fused,
        (rows, columns, offset, scale),
        all_equal,
    )?;

    Ok(all_equal)
}

fn shape_latency(
    device: &CudaDevice,
    baseline: &CudaGraph<()>,
    fused: &CudaGraph<()>,
    (rows, columns, offset, scale): (usize, usize, f32, f32),
    all_equal: bool,
) -> TestResult {
    let options = BenchOptions {
        warmup: Duration::from_millis(10),
        rep: Duration::from_millis(30),
        min_reps: 10,
        max_reps: 50,
        clear_l2: true,
    };
    for reverse in [false, true] {
        let (first, second) = if reverse {
            (fused, baseline)
        } else {
            (baseline, fused)
        };
        let (a, b) = do_bench_paired(
            &device.stream,
            &options,
            |_| {
                first
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
            |_| {
                second
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
        )?;
        let (a, b) = if reverse { (b, a) } else { (a, b) };
        println!(
            "{}",
            serde_json::json!({"kind":"norm_fp4_shape", "rows":rows, "columns":columns,
            "offset":offset,"scale":scale,"reverse":reverse,"baseline_ms":a.median_ms(),
            "candidate_ms":b.median_ms(),"full_bits_equal":all_equal,"independent_rms":true,"fused_selected":norm_fp4_columns(columns)})
        );
    }
    Ok(())
}

#[test]
#[ignore = "CUDA release and protected runner"]
fn norm_fp4_shapes_preserve_bits_and_independent_rms() -> TestResult {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let mut bitwise = true;
    for (rows, columns, offset, scale) in [
        (3, 272, 1.0, 1.0),
        (17, 528, -0.5, 0.25),
        (64, 512, 0.0, 16.0),
        (64, 528, -0.5, 0.25),
        (64, 1024, 1.0, 1.0),
        (64, 1536, -0.5, 1.0),
        (64, 2048, 0.0, 1.0),
        (64, 2560, 1.0, 1.0),
        (64, 3072, -0.5, 1.0),
        (64, 3584, 0.0, 1.0),
        (64, 4608, 1.0, 1.0),
        (64, 5632, -0.5, 1.0),
        (64, 6144, 0.0, 1.0),
        (64, 6656, 1.0, 1.0),
        (64, 7168, -0.5, 1.0),
        (64, 7680, 0.0, 1.0),
        (64, 5120, 1.0, 1024.0),
        (256, 4096, 0.0, 0.25),
        (1024, 5120, 1.0, 1.0),
        (1024, 8192, -0.5, 1.0),
    ] {
        bitwise &= shape_case(&device, rows, columns, offset, scale)?;
    }
    if !bitwise {
        return Err("fused normalization changed output bits".into());
    }
    Ok(())
}
