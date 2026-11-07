use super::kernels;
use crate::device::CudaDevice;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_ir::TensorOp;
use std::error::Error;

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn routing_matches_independent_reference() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let mut cases = vec![
        vec![-4.0, 0.0, 1.0, 3.0],
        vec![0.0; 4],
        vec![1000.0, -1000.0, 999.0, 0.0],
        vec![-3.0, -2.0, -1.0],
        vec![-3.0e37, -2.0e37, -1.0e37],
        vec![f32::MIN; 3],
    ];
    cases.push(
        (0_u16..512)
            .map(|id| (f32::from(id) * 0.123).sin() * 20.0)
            .collect(),
    );
    for scores in cases {
        let active_counts = if scores.len() == 512 {
            vec![10]
        } else {
            vec![1, 2, scores.len()]
        };
        for active in active_counts {
            for normalize in [false, true] {
                let x = device.upload(scores.clone(), &[scores.len()])?;
                let output = kernels::route(
                    api::zeros::<f32>(&[2 * active]).partition([1]),
                    &x,
                    f32::NEG_INFINITY,
                )
                .generics(vec![
                    scores.len().to_string(),
                    scores.len().next_power_of_two().to_string(),
                    active.to_string(),
                    i32::from(normalize).to_string(),
                ])
                .first()
                .unpartition()
                .sync_on(device.stream())?;
                let actual = device.read_borrowed(&output)?;
                let expected = infer_reference_ops::reference_operation(
                    &TensorOp::MoeRoute {
                        experts: scores.len(),
                        active,
                        normalize,
                    },
                    &[&scores],
                    None,
                    0,
                    0,
                )?;
                for (a, b) in actual.iter().zip(&expected[0]) {
                    assert!((a - b).abs() < 0.000_002, "{actual:?} vs {expected:?}");
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn selected_dense_experts_match_reference() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let experts = 4;
    let columns = 16;
    let rows = 12;
    let active = 2;
    let values: Vec<f32> = (0..experts * rows * columns)
        .map(|i| f32::from(u16::try_from(i % 17).unwrap()) / 16.0 - 0.5)
        .collect();
    let w = device.upload(
        values.iter().copied().map(bf16::from_f32).collect(),
        &[experts, rows, columns],
    )?;
    let routes = vec![3.0, 1.0, 0.7, 0.3];
    let route = device.upload(routes.clone(), &[4])?;
    for routed in [false, true] {
        let count = if routed { active } else { 1 };
        let input: Vec<f32> = (0..count * columns)
            .map(|i| f32::from(u16::try_from(i).unwrap()) / 8.0 - 1.0)
            .collect();
        let x = device.upload(input.clone(), &[count, columns])?;
        let out = kernels::dense(
            api::zeros::<f32>(&[active, rows]).partition([1, 4]),
            &x,
            &route,
            &w,
        )
        .generics(vec![
            rows.to_string(),
            columns.to_string(),
            i32::from(routed).to_string(),
        ])
        .first()
        .unpartition()
        .sync_on(device.stream())?;
        let mut inputs: Vec<&[f32]> = vec![&input, &routes];
        inputs.extend(values.chunks_exact(rows * columns));
        let expected = infer_reference_ops::reference_operation(
            &TensorOp::ExpertLinear {
                experts,
                active,
                input: columns,
                output: rows,
                routed_input: routed,
            },
            &inputs,
            None,
            0,
            0,
        )?;
        let actual = device.read_borrowed(&out)?;
        for (a, b) in actual.iter().zip(&expected[0]) {
            assert!((a - b).abs() < 0.000_002, "{actual:?} vs {expected:?}");
        }
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct Fp8Golden {
    experts: usize,
    rows: usize,
    columns: usize,
    active: usize,
    routes: Vec<f32>,
    weights: Vec<u8>,
    cases: Vec<Fp8Case>,
}
#[derive(serde::Deserialize)]
struct Fp8Case {
    routed: bool,
    channel: bool,
    quantized: bool,
    input: Vec<f32>,
    scales: Vec<f32>,
    expected: Vec<f32>,
}
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn selected_fp8_experts_match_pytorch() -> Result<(), Box<dyn Error>> {
    let golden: Fp8Golden = serde_json::from_str(include_str!(
        "../../../../../examples/moe-operators/golden.json"
    ))?;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let w = device.upload(
        golden
            .weights
            .into_iter()
            .map(cuda_core::f8e4m3fn)
            .collect(),
        &[golden.experts, golden.rows, golden.columns],
    )?;
    let routes = device.upload(golden.routes, &[2 * golden.active])?;
    for case in golden.cases {
        let lanes = if case.routed { golden.active } else { 1 };
        let x = device.upload(case.input, &[lanes, golden.columns])?;
        let blocks = if case.channel {
            1
        } else {
            golden.columns / 128
        };
        let scales = device.upload(case.scales, &[golden.experts, golden.rows, blocks])?;
        let out = kernels::fp8(
            api::zeros::<f32>(&[golden.active, golden.rows]).partition([1, 4]),
            &x,
            &routes,
            &w,
            &scales,
        )
        .generics(vec![
            golden.rows.to_string(),
            golden.columns.to_string(),
            i32::from(case.routed).to_string(),
            i32::from(case.channel).to_string(),
            i32::from(case.quantized).to_string(),
            golden.columns.next_power_of_two().to_string(),
        ])
        .first()
        .unpartition()
        .sync_on(device.stream())?;
        let actual = device.read_borrowed(&out)?;
        for (a, b) in actual.iter().zip(case.expected) {
            assert!(
                (a - b).abs() < b.abs().mul_add(0.000_003, 0.000_03),
                "{a} vs {b}"
            );
        }
    }
    Ok(())
}
