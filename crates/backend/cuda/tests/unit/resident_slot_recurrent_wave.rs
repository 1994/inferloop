//! Wave-selected fused launches against the independent scalar recurrence.
use super::super::{BenchOptions, Duration, assert_close, upload, wave_perf_pair};
use super::{D, Fixture, KH, QKV, TestResult, VH};
use crate::{
    device::CudaDevice,
    resident::slot_recurrent::{ValueDevice, fused, value_split},
};
use cutile::prelude::*;
use infer_workloads::DeltaGeometry;

struct Reference {
    output: Vec<f32>,
    states: Vec<Vec<Vec<f32>>>,
}

fn reference(
    fixture: &Fixture,
    device: &CudaDevice,
    slots: usize,
    padded: bool,
) -> TestResult<Reference> {
    let qkv = device.read_borrowed(&fixture.qkv)?;
    let beta = device.read_borrowed(&fixture.beta)?;
    let alpha = device.read_borrowed(&fixture.alpha)?;
    let rate = device.read_borrowed(&fixture.rate)?;
    let bias = device.read_borrowed(&fixture.bias)?;
    let initial = device.read_borrowed(&fixture.state)?;
    let geometry = DeltaGeometry {
        key_heads: KH,
        value_heads: VH,
        key_dim: D,
        value_dim: D,
        norm_epsilon: 1e-6,
    };
    let mut output = vec![0.0; slots * 3 * VH * D];
    let mut states = Vec::new();
    for slot in 0..slots {
        let mut state = initial[slot * VH * D * D..(slot + 1) * VH * D * D].to_vec();
        let mut history = vec![state.clone()];
        for lane in 0..3 {
            let row = slot * 3 + lane;
            if !padded || lane < slot % 4 {
                let inputs = [
                    &qkv[row * QKV..(row + 1) * QKV],
                    &beta[row * VH..(row + 1) * VH],
                    &alpha[row * VH..(row + 1) * VH],
                    &rate,
                    &bias,
                ];
                let result = geometry.step(&mut state, &inputs)?;
                output[row * VH * D..(row + 1) * VH * D].copy_from_slice(&result);
            }
            history.push(state.clone());
        }
        states.push(history);
    }
    Ok(Reference { output, states })
}

fn capture_verify(
    fixture: &mut Fixture,
    device: &CudaDevice,
    slots: usize,
    block: usize,
) -> TestResult<CudaGraph<()>> {
    Ok(CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            fused::delta_verify_batch(
                (&mut fixture.candidate).partition([1, 3, block]),
                &fixture.state.view(&[slots * VH, D, D])?,
                &fixture.qkv,
                &fixture.beta,
                &fixture.alpha,
                &fixture.parameters.view(&[2 * VH])?,
                &fixture.metadata,
            )
            .generics(
                [KH, VH, D, block, 4]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            ),
        )?;
        Ok(())
    })?)
}

fn maximum_gap(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| {
            assert!(a.is_finite() && b.is_finite());
            (a - b).abs()
        })
        .fold(0.0, f32::max)
}

fn check_commits(
    fixture: &Fixture,
    device: &CudaDevice,
    slots: usize,
    block: usize,
    expected: &Reference,
) -> TestResult {
    let initial = device.read_borrowed(&fixture.state)?;
    for count in 0..=3 {
        let counts = device.upload(vec![i32::try_from(count)?; slots], &[slots])?;
        let mut results = Vec::new();
        for selected in [32, block] {
            let mut state = upload(device, initial.clone(), &[slots, VH * D, D])?;
            let graph = CudaGraph::scope(&device.stream, |scope| {
                scope.record(
                    fused::delta_commit_batch(
                        (&mut state).partition([1, D, selected]),
                        &fixture.qkv,
                        &fixture.beta,
                        &fixture.alpha,
                        &fixture.parameters.view(&[2 * VH])?,
                        &fixture.metadata,
                        &counts,
                    )
                    .generics(
                        [KH, VH, D, selected, 4]
                            .iter()
                            .map(ToString::to_string)
                            .collect(),
                    ),
                )?;
                Ok(())
            })?;
            graph.launch().sync_on(&device.stream)?;
            let actual = device.read_borrowed(&state)?;
            for slot in 0..slots {
                assert_close(
                    "scalar commit",
                    &actual[slot * VH * D * D..(slot + 1) * VH * D * D],
                    &expected.states[slot][count],
                );
            }
            results.push(actual);
        }
        let gap = maximum_gap(&results[0], &results[1]);
        assert!(gap < 1e-6, "existing split state bound exceeded: {gap:e}");
    }
    Ok(())
}

fn median(samples: &[f32]) -> f32 {
    assert_ne!(samples.len(), 0);
    let mut sorted = samples.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted[sorted.len() / 2]
}

fn performance(
    device: &CudaDevice,
    baseline: &CudaGraph<()>,
    candidate: &CudaGraph<()>,
    slots: usize,
    block: usize,
) -> TestResult {
    if block == 32 {
        println!(
            "{}",
            serde_json::json!({"kind":"identical_kernel_control","slots":slots,"block":block})
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
    let (before, after) = wave_perf_pair(device, baseline, candidate, &options, false)?;
    println!(
        "{}",
        serde_json::json!({"kind":"global_wave_canary","slots":slots,"block":block,"baseline_ms":before,"candidate_ms":after})
    );
    if median(&after) > median(&before) * 1.10 {
        return Err("global wave gross operator regression".into());
    }
    options.warmup = Duration::from_millis(50);
    options.rep = Duration::from_millis(100);
    options.min_reps = 10;
    options.max_reps = 30;
    for reverse in [false, true] {
        for cold in [true, false] {
            options.clear_l2 = cold;
            let (before, after) = wave_perf_pair(device, baseline, candidate, &options, reverse)?;
            println!(
                "{}",
                serde_json::json!({"kind":"global_wave_timing","slots":slots,"block":block,"cold":cold,"reverse":reverse,"baseline_ms":before,"candidate_ms":after})
            );
            if median(&after) > median(&before) * 1.02 {
                return Err("global wave paired median regression exceeds two percent".into());
            }
        }
    }
    Ok(())
}

fn case(device: &CudaDevice, slots: usize) -> TestResult {
    let block = D / value_split(ValueDevice::new(device)?, VH, D, slots)?;
    let mut fixture = Fixture::new(device, slots)?;
    fixture.prepare_parameters(device)?;
    for padded in [true, false] {
        if !padded {
            fixture.metadata = device.upload(
                (0..slots * 3)
                    .flat_map(|lane| {
                        let position = i32::try_from(lane).unwrap();
                        [position, 0, position, 0]
                    })
                    .collect(),
                &[slots * 3 * 4],
            )?;
        }
        let expected = reference(&fixture, device, slots, padded)?;
        let baseline = capture_verify(&mut fixture, device, slots, 32)?;
        let candidate = capture_verify(&mut fixture, device, slots, block)?;
        baseline.launch().sync_on(&device.stream)?;
        let before = device.read_borrowed(&fixture.candidate)?;
        candidate.launch().sync_on(&device.stream)?;
        let after = device.read_borrowed(&fixture.candidate)?;
        assert_close("scalar reference before", &before, &expected.output);
        assert_close("scalar reference after", &after, &expected.output);
        let gap = maximum_gap(&before, &after);
        assert!(gap < 1e-7, "existing split output bound exceeded: {gap:e}");
        println!(
            "{}",
            serde_json::json!({"kind":"global_wave_numerical","slots":slots,"block":block,"padded":padded,"maximum_output_gap":gap})
        );
        check_commits(&fixture, device, slots, block, &expected)?;
        assert_eq!(
            device.read_borrowed(&fixture.state)?,
            super::values(slots * VH * D * D, 23)
        );
        if !padded {
            performance(device, &baseline, &candidate, slots, block)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "CUDA release; original scalar tolerance and split error bounds, then bounded performance screen"]
fn global_wave_matches_scalar_and_split_contract() -> TestResult {
    if cfg!(debug_assertions) {
        return Err("release profile required".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    if device.profile()?.multiprocessors != 170 {
        return Err("frozen four-split operator baseline requires 170 SMs".into());
    }
    for slots in [2, 4, 16] {
        case(&device, slots)?;
    }
    Ok(())
}
