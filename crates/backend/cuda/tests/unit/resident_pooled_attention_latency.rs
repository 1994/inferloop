//! Compare the actual historical four-slot graph with all-slot recording.
use super::super::{
    Append, Attend, Blueprint, Buffers, CONTROL_TILE, Geometry, Header, Layout, METADATA_FIELDS,
    VERIFY_LANES, Workspace, grouped_baseline, grouped_parts_for,
};
use super::{CAPACITY, DIM, Fixture, HEADS, KV, SCALES, TestResult};
use crate::device::CudaDevice;
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_core::OpId;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

enum RecordedWork {
    Historical(grouped_baseline::Workspace),
    All(Workspace),
}

impl RecordedWork {
    fn append(
        &mut self,
        scope: &Scope,
        fixture: &mut Fixture,
        metadata: &Tensor<i32>,
    ) -> Result<(), DeviceError> {
        let [keys, values] = &mut fixture.candidate;
        match self {
            Self::Historical(work) => work.append(
                scope,
                grouped_baseline::Append {
                    node: OpId::ONE,
                    keys,
                    values,
                    inputs: [&fixture.inputs[0], &fixture.inputs[1]],
                    metadata,
                    tables: &fixture.tables,
                },
            ),
            Self::All(work) => work.append(
                scope,
                Append {
                    node: OpId::ONE,
                    keys,
                    values,
                    inputs: [&fixture.inputs[0], &fixture.inputs[1]],
                    metadata,
                    tables: &fixture.tables,
                    table_batch: &fixture.table_batch,
                },
            ),
        }
    }
    fn attend(
        &mut self,
        scope: &Scope,
        fixture: &Fixture,
        output: &mut Tensor<f32>,
    ) -> Result<(), DeviceError> {
        match self {
            Self::Historical(work) => work.attend(
                scope,
                grouped_baseline::Attend {
                    node: OpId::ONE,
                    output,
                    queries: &fixture.queries,
                    keys: &fixture.candidate[0],
                    values: &fixture.candidate[1],
                },
            ),
            Self::All(work) => work.attend(
                scope,
                Attend {
                    node: OpId::ONE,
                    output,
                    queries: &fixture.queries,
                    keys: &fixture.candidate[0],
                    values: &fixture.candidate[1],
                },
            ),
        }
    }
}

struct ComparedAttention {
    graph: CudaGraph<()>,
    _work: RecordedWork,
    fixture: Fixture,
    output: Tensor<f32>,
    _metadata: Arc<Tensor<i32>>,
}

fn capture(
    device: &CudaDevice,
    slots: usize,
    parts: usize,
    historical: bool,
) -> TestResult<ComparedAttention> {
    let geometry = Geometry {
        heads: HEADS,
        kv_heads: KV,
        dim: DIM,
    };
    let mut header: Header = [0; CONTROL_TILE];
    header[0] = i32::from_ne_bytes(SCALES[0].to_ne_bytes());
    header[1] = i32::from_ne_bytes(SCALES[1].to_ne_bytes());
    let mut work = if historical {
        RecordedWork::Historical(grouped_baseline::Workspace::fixture(
            device,
            geometry,
            (slots, CAPACITY, parts),
            header,
        )?)
    } else {
        RecordedWork::All(Workspace {
            buffers: BTreeMap::from([(
                geometry,
                Buffers::new(
                    device,
                    Blueprint {
                        layout: Layout::new(geometry, slots, CAPACITY, parts)?,
                        parameters: BTreeMap::from([(OpId::ONE, header)]),
                    },
                )?,
            )]),
            nodes: BTreeMap::from([(OpId::ONE, geometry)]),
        })
    };
    let lanes = slots * VERIFY_LANES;
    let words: Vec<i32> = (0..lanes)
        .flat_map(|lane| {
            let position = 4093 + i32::try_from(lane % VERIFY_LANES).unwrap();
            [position + 17, 0, position, 0]
        })
        .collect();
    assert_eq!(words.len(), lanes * METADATA_FIELDS);
    let metadata = device.upload(words, &[lanes * METADATA_FIELDS])?;
    let mut fixture = Fixture::new(device, slots)?;
    let mut output = api::zeros::<f32>(&[lanes * HEADS, DIM]).sync_on(&device.stream)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        work.append(scope, &mut fixture, &metadata)?;
        work.attend(scope, &fixture, &mut output)?;
        Ok(())
    })?;
    Ok(ComparedAttention {
        graph,
        _work: work,
        fixture,
        output,
        _metadata: metadata,
    })
}

fn compare(
    device: &CudaDevice,
    baseline: &ComparedAttention,
    candidate: &ComparedAttention,
    exact: bool,
) -> TestResult<(usize, f64)> {
    let expected = device.read_borrowed(&baseline.output)?;
    let actual = device.read_borrowed(&candidate.output)?;
    assert_eq!(actual.len(), expected.len());
    let mut differences = 0;
    let mut maximum = 0.0f64;
    for (a, b) in actual.iter().zip(&expected) {
        assert!(a.is_finite() && b.is_finite());
        differences += usize::from(a.to_bits() != b.to_bits());
        maximum = maximum.max((f64::from(*a) - f64::from(*b)).abs());
    }
    if exact {
        assert_eq!(
            differences, 0,
            "all-slot fusion changed arithmetic at identical splits"
        );
    }
    for (a, b) in candidate
        .fixture
        .candidate
        .iter()
        .zip(&baseline.fixture.candidate)
    {
        let a = device.read_borrowed(a)?;
        let b = device.read_borrowed(b)?;
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(&b).all(|(a, b)| a.0 == b.0),
            "complete cache bytes differ"
        );
    }
    Ok((differences, maximum))
}

#[test]
#[ignore = "requires CUDA; original four-slot vs all-slot numerical and latency diagnostic"]
fn original_four_slot_vs_all_slot_attention() -> TestResult {
    if cfg!(debug_assertions) {
        return Err("latency requires release".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let sms = usize::try_from(device.profile()?.multiprocessors)?;
    let slots = 16;
    let original_parts = grouped_parts_for(HEADS, KV, sms, 4).ok_or("old split geometry")?;
    let all_parts = grouped_parts_for(HEADS, KV, sms, slots).ok_or("all-slot split geometry")?;
    let baseline = capture(&device, slots, original_parts, true)?;
    let identical_parts = capture(&device, slots, original_parts, false)?;
    let actual_parts = capture(&device, slots, all_parts, false)?;
    for work in [&baseline, &identical_parts, &actual_parts] {
        work.graph.launch().sync_on(&device.stream)?;
    }
    let exact = compare(&device, &baseline, &identical_parts, true)?;
    let changed = compare(&device, &baseline, &actual_parts, false)?;
    eprintln!(
        "{}",
        serde_json::json!({
            "attention_grouping_comparison":true,"slots":slots,"heads":HEADS,"kv_heads":KV,"dim":DIM,
            "cache_capacity":CAPACITY,"prefix_position":4093,"multiprocessors":sms,
            "old_parts":original_parts,"all_slot_parts":all_parts,
            "same_parts_bit_differences":exact.0,"default_parts_bit_differences":changed.0,
            "default_parts_max_abs":changed.1,"full_output_elements":slots*VERIFY_LANES*HEADS*DIM,
            "complete_cache_bytes_equal":true,"changed_parts_quality_pending":original_parts!=all_parts
        })
    );
    for (label, candidate) in [
        ("same_parts", &identical_parts),
        ("actual_parts", &actual_parts),
    ] {
        for cold in [true, false] {
            let options = BenchOptions {
                warmup: Duration::from_millis(50),
                rep: Duration::from_millis(180),
                min_reps: 20,
                max_reps: 100,
                clear_l2: cold,
            };
            for reverse in [false, true] {
                let (first, second) = if reverse {
                    (candidate, &baseline)
                } else {
                    (&baseline, candidate)
                };
                let (a, b) = do_bench_paired(
                    &device.stream,
                    &options,
                    |_| {
                        first
                            .graph
                            .launch()
                            .sync_on(&device.stream)
                            .map_err(|e| tensor_error(&e.to_string()))
                    },
                    |_| {
                        second
                            .graph
                            .launch()
                            .sync_on(&device.stream)
                            .map_err(|e| tensor_error(&e.to_string()))
                    },
                )?;
                let (old, new) = if reverse { (b, a) } else { (a, b) };
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "attention_all_slots_latency":true,"candidate":label,"cold_l2":cold,"reverse":reverse,
                        "baseline_ms":old.median_ms(),"candidate_ms":new.median_ms(),
                        "speedup":old.median_ms()/new.median_ms(),"baseline_raw_ms":old.times_ms(),
                        "candidate_raw_ms":new.times_ms(),"service_acceptance":false
                    })
                );
            }
        }
    }
    Ok(())
}
