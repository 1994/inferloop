//! Actual graph-role recorders compared with original Dense MMA and an independent scalar dot.
use super::{fixture, id};
use crate::AttentionProfile;
use crate::resident::capture::error as capture_error;
use crate::resident::prefill_projection::DensePlan;
use crate::{
    device::CudaDevice,
    mlp::ProjectionWeight,
    resident::{
        ActivationArena, ProgramWeights, PromptScratchRegistry, prefill_gemm::gemm,
        prefill_projection,
    },
    tuning::{median, resident_gemm::ResidentTuningSession},
};
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    error::tensor_error,
    half::bf16,
    prelude::*,
};
use infer_core::OpId;
use infer_cuda_kernels::tile::kernels as tile_kernels;
use infer_ir::{DType as IrDType, TensorNode, TensorOp, TensorSpec, TensorStorage};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    time::Duration,
};
type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|i| f32::from(u16::try_from((i * salt + 7) % 103).unwrap()) / 103.0 - 0.5)
        .collect()
}

struct Fixture {
    graph: infer_ir::DataflowGraph,
    weights: ProgramWeights,
    arena: ActivationArena,
    input: Vec<f32>,
    original_weights: [Vec<bf16>; 2],
    shape: [usize; 3],
}

impl Fixture {
    fn new(device: &CudaDevice, [lanes, rows, columns]: [usize; 3]) -> TestResult<Self> {
        let mut graph = fixture(rows, columns);
        graph
            .tensors
            .iter_mut()
            .find(|s| s.id == id(6))
            .unwrap()
            .storage = TensorStorage::Weight {
            slot: "unused-qkv".into(),
        };
        graph.tensors.push(TensorSpec {
            id: id(11),
            shape: vec![1, columns],
            dtype: IrDType::F32,
            storage: TensorStorage::Weight {
                slot: "embedding".into(),
            },
        });
        graph.nodes.insert(
            0,
            TensorNode {
                id: OpId::new(4)?,
                layer: Some(3),
                op: TensorOp::Embedding,
                inputs: vec![id(11)],
                outputs: vec![id(1)],
                states: Vec::new(),
            },
        );
        graph.hidden = Some(id(3));
        graph.logits = Some(id(5));
        let mut arena = ActivationArena::new_batched(device, &graph, lanes, 16 * 1024 * 1024)?;
        let input = values(lanes * columns, 17);
        let input_slot = arena.slot(id(1))?;
        device.copy_host_slice(
            arena.buffers[input_slot]
                .as_mut()
                .ok_or("missing fixture input")?,
            &input,
        )?;
        let original_weights = [7, 13].map(|salt| {
            values(rows * columns, salt)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>()
        });
        let projections = BTreeMap::from([
            (
                id(2),
                ProjectionWeight::Dense(
                    device.upload(original_weights[0].clone(), &[rows, columns])?,
                ),
            ),
            (
                id(4),
                ProjectionWeight::Dense(
                    device.upload(original_weights[1].clone(), &[rows, columns])?,
                ),
            ),
        ]);
        let weights = ProgramWeights {
            kernels: Arc::new(tile_kernels(0)),
            autotune: false,
            resident_tuning: Arc::new(ResidentTuningSession::default()),
            batch_width: 3,
            prefill_width: lanes,
            narrow_prefill_width: 0,
            kv_scales: BTreeMap::new(),
            tiling: BTreeMap::new(),
            fusion: None,
            attention_profile: AttentionProfile::Canonical,
            projections,
            input_scales: BTreeMap::new(),
            fp8_inputs: BTreeSet::new(),
            chunked_recurrent: false,
            expert_banks: BTreeMap::new(),
            constants: BTreeMap::new(),
            embeddings: BTreeMap::new(),
            rope_axes: BTreeMap::new(),
            rope_frequencies: BTreeMap::new(),
            mrope: false,
            prompt_scratch: PromptScratchRegistry::default(),
        };
        Ok(Self {
            graph,
            weights,
            arena,
            input,
            original_weights,
            shape: [lanes, rows, columns],
        })
    }

    fn original(&mut self, device: &CudaDevice) -> TestResult<CudaGraph<()>> {
        Ok(CudaGraph::scope(&device.stream, |scope| {
            for index in [1, 2] {
                let node = &self.graph.nodes[index];
                let output_slot = self.arena.slot(node.outputs[0]).map_err(capture_error)?;
                let [lanes, rows, columns] = self.shape;
                let output = self.arena.buffers[output_slot].take().unwrap();
                let mut output = output.reshape(&[lanes, rows])?;
                let input = self
                    .arena
                    .get(id(1))
                    .map_err(capture_error)?
                    .view(&[lanes, columns])?;
                let ProjectionWeight::Dense(weight) = &self.weights.projections[&node.inputs[1]]
                else {
                    unreachable!()
                };
                scope.record(
                    gemm::dense((&mut output).partition([32, 64]), &input, weight)
                        .generics(vec![columns.to_string()]),
                )?;
                self.arena.buffers[output_slot] = Some(output.reshape(&[lanes * rows])?);
            }
            Ok(())
        })?)
    }

    fn candidate(&mut self, device: &CudaDevice) -> TestResult<CudaGraph<()>> {
        let plan = DensePlan::new(&self.graph, None);
        assert_eq!(plan.pair(1), Some(2));
        Ok(CudaGraph::scope(&device.stream, |scope| {
            let profile = device.profile().map_err(capture_error)?;
            if !prefill_projection::record_pair(
                scope,
                &self.graph.nodes[1],
                &self.graph.nodes[2],
                &mut self.arena,
                (&self.weights, profile),
            )? {
                for index in [1, 2] {
                    assert!(plan.narrow(index));
                    assert!(
                        prefill_projection::record_narrow(
                            scope,
                            &self.graph.nodes[index],
                            &mut self.arena,
                            (&self.weights, profile)
                        )?,
                        "selected graph control must use narrow recorder"
                    );
                }
            }
            Ok(())
        })?)
    }

    fn outputs(&self, device: &CudaDevice) -> TestResult<Vec<Vec<f32>>> {
        [3, 5]
            .into_iter()
            .map(|output| Ok(device.read_borrowed(self.arena.get(id(output))?)?))
            .collect()
    }

    fn scalar(&self, output: usize) -> Vec<f64> {
        let [lanes, rows, columns] = self.shape;
        (0..lanes * rows)
            .map(|index| {
                let lane = index / rows;
                let row = index % rows;
                (0..columns)
                    .map(|column| {
                        f64::from(bf16::from_f32(self.input[lane * columns + column]).to_f32())
                            * f64::from(
                                self.original_weights[output][row * columns + column].to_f32(),
                            )
                    })
                    .sum()
            })
            .collect()
    }
}

fn pair(
    device: &CudaDevice,
    baseline: &CudaGraph<()>,
    candidate: &CudaGraph<()>,
    options: &BenchOptions,
    reverse: bool,
) -> TestResult<(Vec<f32>, Vec<f32>)> {
    let launch = |graph: &CudaGraph<()>| {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|error| tensor_error(&error.to_string()))
    };
    let (first, second) = if reverse {
        (candidate, baseline)
    } else {
        (baseline, candidate)
    };
    let (before, after) = do_bench_paired(
        &device.stream,
        options,
        |_| launch(first),
        |_| launch(second),
    )?;
    let result = (before.times_ms().to_vec(), after.times_ms().to_vec());
    Ok(if reverse {
        (result.1, result.0)
    } else {
        result
    })
}

fn performance(
    device: &CudaDevice,
    baseline: &CudaGraph<()>,
    candidate: &CudaGraph<()>,
    shape: [usize; 3],
) -> TestResult {
    if shape[1..] == [48, 5120] {
        println!(
            "{}",
            serde_json::json!({"kind":"existing_model_control","shape":shape})
        );
        return Ok(());
    }
    let mut options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::ZERO,
        min_reps: 3,
        max_reps: 3,
        clear_l2: true,
    };
    let (before, after) = pair(device, baseline, candidate, &options, false)?;
    println!(
        "{}",
        serde_json::json!({"kind":"gdn_geometry_canary","shape":shape,"baseline_ms":before,"candidate_ms":after})
    );
    if median(&after) > median(&before) * 1.10 {
        return Err("GDN geometry gross performance regression".into());
    }
    options.rep = Duration::from_millis(100);
    options.min_reps = 10;
    options.max_reps = 30;
    for reverse in [false, true] {
        for cold in [true, false] {
            options.clear_l2 = cold;
            let (before, after) = pair(device, baseline, candidate, &options, reverse)?;
            println!(
                "{}",
                serde_json::json!({"kind":"gdn_geometry_timing","shape":shape,"reverse":reverse,"cold":cold,"baseline_ms":before,"candidate_ms":after})
            );
            if median(&after) > median(&before) * 1.02 {
                return Err("GDN geometry paired median regression exceeds two percent".into());
            }
        }
    }
    Ok(())
}

fn case(device: &CudaDevice, shape: [usize; 3]) -> TestResult {
    let mut fixture = Fixture::new(device, shape)?;
    let original = fixture.original(device)?;
    let before_captures = prefill_projection::narrow_dense_captures();
    let candidate = fixture.candidate(device)?;
    let narrow = prefill_projection::narrow_dense_captures() - before_captures;
    if shape[0] == 64 && shape[1] <= 64 {
        assert!(narrow > 0);
    } else {
        assert_eq!(narrow, 0);
    }
    original.launch().sync_on(&device.stream)?;
    let expected = fixture.outputs(device)?;
    candidate.launch().sync_on(&device.stream)?;
    let actual = fixture.outputs(device)?;
    for (output, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let scalar = fixture.scalar(output);
        assert_eq!(actual.len(), scalar.len());
        for ((&actual, &expected), reference) in actual.iter().zip(expected).zip(scalar) {
            assert!(actual.is_finite() && expected.is_finite() && reference.is_finite());
            assert_eq!(actual.to_bits(), expected.to_bits(), "shape {shape:?}");
            assert!((f64::from(actual) - reference).abs() <= 1e-5 * reference.abs().max(1.0));
        }
    }
    assert_eq!(
        device.read_borrowed(fixture.arena.get(id(1))?)?,
        fixture.input
    );
    println!(
        "{}",
        serde_json::json!({"kind":"gdn_geometry_numerical","shape":shape,"narrow_captures":narrow,"full_bits_equal":true,"scalar_reference_passed":true})
    );
    performance(device, &original, &candidate, shape)
}

#[test]
#[ignore = "CUDA release; actual graph-role capture, full Dense/scalar comparison, bounded operator timing"]
fn graph_derived_gdn_recorders_match_reference_and_performance() -> TestResult {
    if cfg!(debug_assertions) {
        return Err("release required".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for shape in [
        [32, 16, 96],
        [64, 16, 96],
        [64, 48, 5120],
        [128, 48, 5120],
        [64, 13, 127],
        [128, 128, 192],
        [64, 128, 96],
    ] {
        case(&device, shape)?;
    }
    Ok(())
}
