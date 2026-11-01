//! Exact activation-quantization CSE on real checkpoint gate/up pairs.
//! No observer kernels enter either arm; snapshots and input updates stay outside replay.
use super::{QuantPairPlan, Workspace, record_with_input_pointer};
use crate::tuning::resident_gemm::ResidentTuningSession;
use crate::{
    device::CudaDevice,
    loading::{Projection, floats},
    mlp::ProjectionWeight,
    resident::{ProgramWeights, PromptScratchRegistry},
};
use cutile::bench::BenchOptions;
use cutile::bench::do_bench_paired;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_core::{ModelId, OpId, TensorId};
use infer_cuda_kernels::tile::kernels as tile_kernels;
use infer_cuda_kernels::tile::runtime::Graph;
use infer_ir::{DType as IrDType, DataflowGraph, TensorNode, TensorOp, TensorSpec, TensorStorage};
use infer_models::{DeviceWeight, QuantizedPackage, WeightEncoding};
use std::env::var;
use std::error::Error;
use std::process::Command;
use std::thread::sleep;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::{Duration, Instant},
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
const ROWS: [usize; 4] = [1, 4, 64, 256];
const FIXTURES: [&str; 8] = [
    "ordinary",
    "zero",
    "tiny",
    "sparse",
    "sign",
    "ordinary_shift",
    "zero",
    "ordinary",
];

struct Pair {
    name: String,
    encoding: WeightEncoding,
    columns: usize,
    outputs: usize,
    graph: DataflowGraph,
    weights: ProgramWeights,
}

impl Pair {
    fn new(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        name: &str,
        gate: &DeviceWeight,
    ) -> TestResult<Self> {
        let up_name = gate.data.name.replace("mlp.gate_proj.", "mlp.up_proj.");
        let (up_key, up) = package
            .weights
            .iter()
            .find(|(_, value)| value.data.name == up_name)
            .map(|(key, value)| (key.clone(), value.clone()))
            .ok_or("checkpoint up projection missing")?;
        let [outputs, columns] = gate.shape.as_slice() else {
            return Err("checkpoint gate matrix rank".into());
        };
        if up.shape != gate.shape || up.encoding != gate.encoding {
            return Err("checkpoint gate/up quantization or geometry differs".into());
        }
        if gate.encoding == WeightEncoding::Fp8Channel
            && (!gate.fp8_token_input || !up.fp8_token_input)
        {
            return Err("checkpoint pair does not declare token FP8 input".into());
        }
        let mut weights = fixture_weights();
        for (id, source) in [(10, gate), (11, &up)] {
            let id = TensorId::new(id)?;
            let projection = Projection::load(device, package, source)?.resident();
            projection.validate(*outputs, *columns)?;
            weights.projections.insert(id, projection);
            if source.encoding == WeightEncoding::Nvfp4 {
                let scale = floats(
                    package,
                    source
                        .input_global_scale
                        .as_ref()
                        .ok_or("NVFP4 activation scale missing")?,
                )?;
                let [scale] = scale.as_slice() else {
                    return Err("NVFP4 activation scale rank".into());
                };
                if !scale.is_finite() || *scale <= 0.0 {
                    return Err("invalid checkpoint activation scale".into());
                }
                weights.input_scales.insert(id, *scale);
            } else {
                weights.fp8_inputs.insert(id);
            }
        }
        if weights
            .input_scales
            .get(&TensorId::new(10)?)
            .map(|value| value.to_bits())
            != weights
                .input_scales
                .get(&TensorId::new(11)?)
                .map(|value| value.to_bits())
        {
            return Err("checkpoint gate/up activation scales differ".into());
        }
        let graph = pair_graph(*columns, *outputs)?;
        let plan = QuantPairPlan::new(&graph, &weights);
        if plan.role(0).is_none() || plan.role(1).is_none() {
            return Err("real checkpoint pair not certified by SwiGLU planner".into());
        }
        println!(
            "{}",
            serde_json::json!({
                "kind":"quant_reuse_checkpoint_pair", "gate":name, "up":up_key,
                "gate_source":gate, "up_source":up, "shape":[outputs, columns],
                "encoding":format!("{:?}", gate.encoding),
                "weight_requantized":false, "autotune":false,
            })
        );
        Ok(Self {
            name: name.to_owned(),
            encoding: gate.encoding,
            columns: *columns,
            outputs: *outputs,
            graph,
            weights,
        })
    }
}

fn fixture_weights() -> ProgramWeights {
    ProgramWeights {
        kernels: Arc::new(tile_kernels(0)),
        mrope: false,
        prompt_scratch: PromptScratchRegistry::default(),
        autotune: false,
        resident_tuning: Arc::new(ResidentTuningSession::default()),
        batch_width: 4,
        prefill_width: 256,
        narrow_prefill_width: 64,
        attention_profile: crate::AttentionProfile::Canonical,
        chunked_recurrent: false,
        fusion: None,
        projections: BTreeMap::new(),
        input_scales: BTreeMap::new(),
        fp8_inputs: BTreeSet::new(),
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        embeddings: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
    }
}

fn pair_graph(columns: usize, outputs: usize) -> TestResult<DataflowGraph> {
    let mut graph = DataflowGraph::default();
    for (id, size) in [
        (1, columns),
        (2, outputs),
        (3, outputs),
        (4, outputs),
        (5, outputs),
    ] {
        graph.tensors.push(TensorSpec {
            id: TensorId::new(id)?,
            shape: vec![size],
            dtype: IrDType::F32,
            storage: TensorStorage::Activation,
        });
    }
    for (id, op, inputs, output) in [
        (1, TensorOp::Linear, vec![1, 10], 2),
        (2, TensorOp::Linear, vec![1, 11], 3),
        (3, TensorOp::Silu, vec![2], 4),
        (4, TensorOp::Multiply, vec![4, 3], 5),
    ] {
        graph.nodes.push(TensorNode {
            id: OpId::new(id)?,
            layer: None,
            op,
            inputs: inputs
                .into_iter()
                .map(TensorId::new)
                .collect::<infer_core::Result<_>>()?,
            outputs: vec![TensorId::new(output)?],
            states: vec![],
        });
    }
    Ok(graph)
}

fn checkpoint_pairs(package: &QuantizedPackage) -> TestResult<Vec<(String, DeviceWeight)>> {
    let mut selected = Vec::new();
    for encoding in [WeightEncoding::Nvfp4, WeightEncoding::Fp8Channel] {
        let mut gates = package
            .weights
            .iter()
            .filter(|(_, weight)| {
                weight.encoding == encoding && weight.data.name.contains(".mlp.gate_proj.")
            })
            .map(|(name, weight)| {
                let layer = weight
                    .data
                    .name
                    .split(".layers.")
                    .nth(1)
                    .and_then(|tail| tail.split('.').next())
                    .ok_or("checkpoint layer coordinate missing")?
                    .parse::<usize>()?;
                Ok((layer, name.clone(), weight.clone()))
            })
            .collect::<TestResult<Vec<_>>>()?;
        gates.sort_by_key(|(layer, _, _)| *layer);
        let first = gates
            .first()
            .ok_or("checkpoint quantization family has no gates")?;
        let last = gates
            .last()
            .ok_or("checkpoint quantization family has no gates")?;
        if first.0 == last.0 {
            return Err("checkpoint family needs distinct first/last layers".into());
        }
        selected.push((first.1.clone(), first.2.clone()));
        selected.push((last.1.clone(), last.2.clone()));
    }
    Ok(selected)
}

struct Case {
    // Graphs drop before every allocation and workspace they reference.
    separate: Graph,
    shared: Graph,
    workspace: Workspace,
    input: Tensor<f32>,
    gate: Tensor<f32>,
    up: Tensor<f32>,
    pair: Arc<Pair>,
    rows: usize,
    pointers: serde_json::Value,
    strategies: serde_json::Value,
}

impl Case {
    fn new(device: &CudaDevice, pair: Arc<Pair>, rows: usize) -> TestResult<Self> {
        let mut workspace = Workspace::new(device, &pair.graph, &pair.weights, &[rows])?;
        let input = api::zeros::<f32>(&[rows, pair.columns]).sync_on(&device.stream)?;
        let mut gate = api::zeros::<f32>(&[rows, pair.outputs]).sync_on(&device.stream)?;
        let mut up = api::zeros::<f32>(&[rows, pair.outputs]).sync_on(&device.stream)?;
        let plan = QuantPairPlan::new(&pair.graph, &pair.weights);
        workspace.set_quant_reuse_enabled(false);
        let separate = capture_pair(
            device,
            &pair,
            &plan,
            &mut workspace,
            &input,
            &mut gate,
            &mut up,
        )?;
        if workspace.quant_reuse_stats() != (2, 0) {
            return Err(
                format!("Separate record count {:?}", workspace.quant_reuse_stats()).into(),
            );
        }
        workspace.set_quant_reuse_enabled(true);
        let shared = capture_pair(
            device,
            &pair,
            &plan,
            &mut workspace,
            &input,
            &mut gate,
            &mut up,
        )?;
        if workspace.quant_reuse_stats() != (1, 1) {
            return Err(format!("Shared record count {:?}", workspace.quant_reuse_stats()).into());
        }
        let snapshot_pointers =
            if let Some((codes, scales)) = workspace.quantized_fp4(rows, pair.columns) {
                [pointer(codes), pointer(scales)]
            } else if let Some((codes, scales)) = workspace.quantized_fp8(rows, pair.columns) {
                [pointer(codes), pointer(scales)]
            } else {
                return Err("quantization buffer missing".into());
            };
        let pointers = serde_json::json!({
            "input":pointer(&input), "gate_output":pointer(&gate), "up_output":pointer(&up),
            "quantized_input":snapshot_pointers[0], "activation_scales":snapshot_pointers[1],
            "weights":projection_pointers(&pair.weights),
            "same_physical_allocations_in_both_arms":true,
        });
        let strategies = workspace.prepared_configs();
        Ok(Self {
            separate,
            shared,
            workspace,
            input,
            gate,
            up,
            pair,
            rows,
            pointers,
            strategies,
        })
    }

    fn update_input(&mut self, device: &CudaDevice, fixture: &str) -> TestResult<Vec<f32>> {
        let values = fixture_values(self.rows, self.pair.columns, fixture)?;
        let uploaded = device.upload(values.clone(), &[self.rows, self.pair.columns])?;
        api::memcpy(&mut self.input, &uploaded).sync_on(&device.stream)?;
        Ok(values)
    }

    fn snapshot(&self, device: &CudaDevice) -> TestResult<Snapshot> {
        let (codes, scales) = if let Some((codes, scales)) =
            self.workspace.quantized_fp4(self.rows, self.pair.columns)
        {
            (
                device
                    .read_borrowed(codes)?
                    .into_iter()
                    .map(cuda_core::f4e2m1fnx2::to_bits)
                    .collect(),
                device
                    .read_borrowed(scales)?
                    .into_iter()
                    .map(|x| u32::from(x.0))
                    .collect(),
            )
        } else if let Some((codes, scales)) =
            self.workspace.quantized_fp8(self.rows, self.pair.columns)
        {
            (
                device
                    .read_borrowed(codes)?
                    .into_iter()
                    .map(|x| x.0)
                    .collect(),
                device
                    .read_borrowed(scales)?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect(),
            )
        } else {
            return Err("quantization snapshot missing".into());
        };
        Ok(Snapshot {
            gate: device.read_borrowed(&self.gate)?,
            up: device.read_borrowed(&self.up)?,
            codes,
            scales,
        })
    }

    fn validate(&mut self, device: &CudaDevice) -> TestResult {
        for (replay, fixture) in FIXTURES.iter().enumerate() {
            let values = self.update_input(device, fixture)?;
            let reference = quantization_reference(&self.pair, &values, self.rows)?;
            for reverse in [false, true] {
                let (first, second) = if reverse {
                    (&self.shared, &self.separate)
                } else {
                    (&self.separate, &self.shared)
                };
                first.launch().sync_on(&device.stream)?;
                let expected = self.snapshot(device)?;
                check_quantization(&expected, &reference)?;
                second.launch().sync_on(&device.stream)?;
                let actual = self.snapshot(device)?;
                check_quantization(&actual, &reference)?;
                check_outputs(&expected.gate, &actual.gate, "gate")?;
                check_outputs(&expected.up, &actual.up, "up")?;
                if expected.codes != actual.codes || expected.scales != actual.scales {
                    return Err("Separate/Shared quantizer codes or scales changed".into());
                }
                println!(
                    "{}",
                    serde_json::json!({
                        "kind":"quant_reuse_full_numeric", "weight":self.pair.name,
                        "shape":[self.rows,self.pair.outputs,self.pair.columns],
                        "fixture":fixture, "replay":replay, "order":if reverse {"BA"} else {"AB"},
                        "complete_output_elements":expected.gate.len()+expected.up.len(),
                        "quantized_code_bytes":expected.codes.len(), "scale_elements":expected.scales.len(),
                        "outputs_bitwise_identical":true, "outputs_finite":true,
                        "quantized_codes_bitwise_identical":true, "scales_bitwise_identical":true,
                        "independent_cpu_quantizer_full_match":true,
                        "separate_recorded_quantizers":2, "shared_recorded_quantizers":1,
                        "pointers":self.pointers, "strategies":self.strategies,
                        "performance_accepted":false,
                    })
                );
            }
        }
        Ok(())
    }
}

fn capture_pair(
    device: &CudaDevice,
    pair: &Pair,
    plan: &QuantPairPlan,
    workspace: &mut Workspace,
    input: &Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
) -> TestResult<Graph> {
    let input_view = input.view(&[usize::try_from(input.shape()[0])?, pair.columns])?;
    let input_pointer = pointer(input);
    workspace.begin_quant_capture();
    let graph = Graph::scope(&device.stream, |scope| {
        for (index, output) in [(0, &mut *gate), (1, &mut *up)] {
            workspace.enter_quant_node(index, plan.role(index));
            let id = pair.graph.nodes[index].inputs[1];
            let weight = pair
                .weights
                .projections
                .get(&id)
                .ok_or_else(|| DeviceError::Launch("pair weight missing".into()))?;
            if !record_with_input_pointer(
                scope,
                workspace,
                weight,
                pair.weights.activation_quantization(id),
                &input_view,
                output,
                pair.columns,
                input_pointer,
            )? {
                return Err(DeviceError::Launch(
                    "quantized pair path not recorded".into(),
                ));
            }
        }
        Ok(())
    });
    workspace.end_quant_capture();
    Ok(graph?)
}

fn pointer<T: DType>(tensor: &Tensor<T>) -> u64 {
    tensor.device_pointer().cu_deviceptr()
}

fn projection_pointers(weights: &ProgramWeights) -> serde_json::Value {
    serde_json::json!(
        weights
            .projections
            .iter()
            .map(|(id, weight)| {
                let allocations = match weight {
                    ProjectionWeight::Fp4(codes, scales, _) => [pointer(codes), pointer(scales)],
                    ProjectionWeight::Fp8(codes, scales)
                    | ProjectionWeight::Fp8Block(codes, scales) => {
                        [pointer(codes), pointer(scales)]
                    }
                    ProjectionWeight::Dense(codes) => [pointer(codes), 0],
                };
                serde_json::json!({"id":id,"allocations":allocations})
            })
            .collect::<Vec<_>>()
    )
}

fn fixture_values(rows: usize, columns: usize, fixture: &str) -> TestResult<Vec<f32>> {
    (0..rows * columns)
        .map(|i| {
            let row = i / columns;
            let ordinary = (f32::from(u16::try_from((i * 17 + row * 23) % 131)?) - 65.0) / 37.0;
            Ok(match fixture {
                "ordinary" => ordinary,
                "ordinary_shift" => (-ordinary).mul_add(0.75, 0.125),
                "zero" => 0.0,
                "tiny" => ordinary * 1e-10,
                "sparse" => {
                    if (i + row * 7) % 47 < 45 {
                        0.0
                    } else {
                        ordinary
                    }
                }
                "sign" => {
                    if (i + row).is_multiple_of(2) {
                        1.75
                    } else {
                        -0.375
                    }
                }
                _ => return Err("unknown fixture".into()),
            })
        })
        .collect()
}

struct Snapshot {
    gate: Vec<f32>,
    up: Vec<f32>,
    codes: Vec<u8>,
    scales: Vec<u32>,
}
struct QuantizationReference {
    codes: Vec<u8>,
    scales: Vec<u32>,
}

fn decode_fp8(code: u8) -> f32 {
    let magnitude = code & 127;
    let exponent = magnitude >> 3;
    let mantissa = f32::from(magnitude & 7);
    let value = if exponent == 0 {
        mantissa / 512.0
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if code & 128 == 0 { value } else { -value }
}

fn nearest(value: f32, table: &[f32]) -> TestResult<u8> {
    let next = table.partition_point(|candidate| *candidate < value);
    let index = if next == 0 {
        0
    } else if next == table.len() {
        next - 1
    } else {
        let lower = (table[next - 1] - value).abs();
        let upper = (table[next] - value).abs();
        if lower < upper || (lower.to_bits() == upper.to_bits() && (next - 1) % 2 == 0) {
            next - 1
        } else {
            next
        }
    };
    Ok(u8::try_from(index)?)
}

fn quantization_reference(
    pair: &Pair,
    values: &[f32],
    rows: usize,
) -> TestResult<QuantizationReference> {
    let table: Vec<f32> = (0..=126).map(decode_fp8).collect();
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    if pair.encoding == WeightEncoding::Fp8Channel {
        for row in 0..rows {
            let span = &values[row * pair.columns..(row + 1) * pair.columns];
            let scale = (span.iter().copied().map(f32::abs).fold(0.0, f32::max) / 448.0).max(1e-12);
            scales.push(scale.to_bits());
            for &value in span {
                let normalized = value / scale;
                codes.push(
                    nearest(normalized.abs().min(448.0), &table)?
                        | if normalized.is_sign_negative() {
                            128
                        } else {
                            0
                        },
                );
            }
        }
    } else {
        let global = pair
            .weights
            .input_scales
            .get(&TensorId::new(10)?)
            .ok_or("checkpoint NVFP4 activation global missing")?;
        let fp4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for span in values.as_chunks::<16>().0 {
            let maximum = span.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let encoded = nearest((maximum * (*global / 6.0)).min(448.0), &table)?;
            scales.push(u32::from(encoded));
            let denominator = if encoded == 0 {
                1.0
            } else {
                table[usize::from(encoded)]
            };
            let multiplier = *global / denominator;
            let mut packed = [0_u8; 16];
            for (i, &value) in span.iter().enumerate() {
                let normalized = (value * multiplier).clamp(-6.0, 6.0);
                packed[i] = nearest(normalized.abs(), &fp4)?
                    | if normalized.is_sign_negative() { 8 } else { 0 };
            }
            codes.extend(packed.as_chunks::<2>().0.iter().map(|x| x[0] | x[1] << 4));
        }
    }
    Ok(QuantizationReference { codes, scales })
}

fn check_quantization(snapshot: &Snapshot, reference: &QuantizationReference) -> TestResult {
    if snapshot.codes != reference.codes || snapshot.scales != reference.scales {
        let code = snapshot
            .codes
            .iter()
            .zip(&reference.codes)
            .position(|(actual, expected)| actual != expected);
        let scale = snapshot
            .scales
            .iter()
            .zip(&reference.scales)
            .position(|(actual, expected)| actual != expected);
        return Err(format!(
            "independent quantizer mismatch code={code:?} scale={scale:?}, lengths={:?}/{:?}",
            (snapshot.codes.len(), snapshot.scales.len()),
            (reference.codes.len(), reference.scales.len())
        )
        .into());
    }
    Ok(())
}

fn check_outputs(expected: &[f32], actual: &[f32], projection: &str) -> TestResult {
    if expected.len() != actual.len() {
        return Err("pair output extent changed".into());
    }
    if let Some((index, (left, right))) =
        expected
            .iter()
            .zip(actual)
            .enumerate()
            .find(|(_, (left, right))| {
                !left.is_finite() || !right.is_finite() || left.to_bits() != right.to_bits()
            })
    {
        return Err(
            format!("{projection} full output mismatch index={index} {left} != {right}").into(),
        );
    }
    Ok(())
}

fn await_cost_idle() -> TestResult {
    let started = Instant::now();
    let mut consecutive = 0;
    while started.elapsed() < Duration::from_secs(30) {
        let query = Command::new("nvidia-smi")
            .args([
                "--id=0",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()?;
        let stdout = String::from_utf8(query.stdout)?;
        println!(
            "{}",
            serde_json::json!({
                "kind":"quant_reuse_cost_idle_sample", "elapsed_seconds":started.elapsed().as_secs_f64(),
                "query_success":query.status.success(), "utilization_raw":stdout,
                "stderr":String::from_utf8(query.stderr)?,
            })
        );
        if !query.status.success() {
            return Err("cost idle query failed".into());
        }
        let utilization: u32 = stdout.trim().parse()?;
        consecutive = if utilization <= 2 { consecutive + 1 } else { 0 };
        if consecutive == 3 {
            return Ok(());
        }
        sleep(Duration::from_secs(1));
    }
    Err("cost not started: three consecutive <=2% samples absent within 30s; numerical evidence retained".into())
}

fn measure_case(
    device: &CudaDevice,
    case: &Case,
    round: usize,
    clear_l2: bool,
    reverse: bool,
) -> TestResult {
    let options = BenchOptions {
        warmup: Duration::from_millis(5),
        rep: Duration::from_millis(10),
        min_reps: 16,
        max_reps: 32,
        clear_l2,
    };
    let (first, second) = if reverse {
        (&case.shared, &case.separate)
    } else {
        (&case.separate, &case.shared)
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
    let (separate, shared) = if reverse { (b, a) } else { (a, b) };
    println!(
        "{}",
        serde_json::json!({
            "kind":"quant_reuse_pair_cost", "weight":case.pair.name,
            "shape":[case.rows,case.pair.outputs,case.pair.columns],
            "round":round, "clear_l2":clear_l2, "order":if reverse {"BA"}else{"AB"},
            "separate_samples_ms":separate.times_ms(), "shared_samples_ms":shared.times_ms(),
            "separate_median_ms":separate.median_ms(), "shared_median_ms":shared.median_ms(),
            "separate_p95_ms":separate.quantile_ms(0.95), "shared_p95_ms":shared.quantile_ms(0.95),
            "speedup":separate.median_ms()/shared.median_ms(),
            "pointers":case.pointers, "strategies":case.strategies,
            "timed_scope":"complete quantization + gate GEMM + up GEMM graph",
            "all_prepared_case_numerics_pass":true, "performance_accepted":false,
        })
    );
    Ok(())
}

fn run(performance: bool) -> TestResult {
    if performance {
        let compile = CompileOptions::default();
        if cfg!(debug_assertions)
            || compile.device_debug
            || compile.opt_level.is_some_and(|x| x != 3)
        {
            return Err("cost requires release and CUDA JIT O3 without device debug".into());
        }
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    println!(
        "{}",
        serde_json::json!({
            "kind":"quant_reuse_device", "hardware":device.profile()?,
            "release":!cfg!(debug_assertions), "functional_idle_gate":false,
            "cost_after_all_numeric_cases":true,
        })
    );
    let mut package = QuantizedPackage::open(var("INFER_TEST_MODEL")?, ModelId::ONE)?;
    let selected = checkpoint_pairs(&package)?;
    let mut cases = Vec::new();
    for (name, weight) in selected {
        let pair = Arc::new(Pair::new(&device, &mut package, &name, &weight)?);
        for rows in ROWS {
            cases.push(Case::new(&device, Arc::clone(&pair), rows)?);
        }
    }
    for case in &mut cases {
        case.validate(&device)?;
    }
    println!(
        "{}",
        serde_json::json!({
            "kind":"quant_reuse_all_numerics_pass", "pairs":4, "cases":cases.len(),
            "fixtures_per_case":FIXTURES.len(), "both_orders_per_fixture":true,
            "full_outputs_finite_and_bitwise_identical":true,
            "full_quantizer_matches_independent_cpu":true, "timing_started":false,
        })
    );
    if performance {
        for case in &mut cases {
            let _values = case.update_input(&device, "ordinary")?;
        }
        device.reclaim_barrier()?;
        await_cost_idle()?;
        for round in 0..3 {
            for clear_l2 in [true, false] {
                for reverse in [false, true] {
                    for case in &cases {
                        measure_case(&device, case, round, clear_l2, reverse)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires real checkpoint and CUDA; run serially inside safe-run"]
fn quant_reuse_checkpoint_full_bitwise() -> TestResult {
    run(false)
}

#[test]
#[ignore = "requires real checkpoint and CUDA; run release serially inside safe-run"]
fn quant_reuse_checkpoint_pair_cost() -> TestResult {
    run(true)
}
