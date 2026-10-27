use super::{ValueDevice, fused};
use crate::constants::CONV_KERNEL_TILE;
use crate::tuning::median;
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use infer_workloads::DeltaGeometry;
use std::error::Error;
use std::time::Duration;

#[test]
fn wave_split_uses_actual_head_grid_and_value_dimension() -> TestResult {
    for (sm, heads, dim, slots, expected) in [
        (32, 48, 128, 1, 4),
        (80, 48, 128, 1, 4),
        (170, 48, 128, 1, 4),
        (170, 48, 128, 2, 4),
        (170, 48, 128, 4, 4),
        (170, 48, 128, 16, 4),
        (170, 192, 128, 1, 4),
        (170, 16, 32, 1, 8),
        (170, 4, 32, 1, 32),
    ] {
        assert_eq!(
            super::value_split(
                ValueDevice {
                    multiprocessors: sm,
                    warp_size: 32
                },
                heads,
                dim,
                slots
            )?,
            expected
        );
    }
    Ok(())
}

#[test]
fn wave_split_minimizes_global_grid_distance() -> TestResult {
    for sm in [1, 32, 80, 170, 512] {
        for heads in [1, 4, 16, 48, 192] {
            for dim in [1, 32, 128] {
                for slots in [1, 2, 4, 16] {
                    let split = super::value_split(
                        ValueDevice {
                            multiprocessors: sm,
                            warp_size: 32,
                        },
                        heads,
                        dim,
                        slots,
                    )?;
                    assert!(split.is_power_of_two());
                    assert_eq!(dim % split, 0);
                    let target = usize::try_from(sm)?;
                    let distance = (slots * heads * split).abs_diff(target);
                    assert!(dim / split <= 32);
                    let mut alternative = (dim / 32).max(1);
                    while alternative <= dim {
                        assert!(distance <= (slots * heads * alternative).abs_diff(target));
                        alternative *= 2;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn wave_split_rejects_invalid_graph_or_device_geometry() {
    for (sm, heads, dim, slots) in [
        (0, 48, 128, 1),
        (170, 0, 128, 1),
        (170, 48, 0, 1),
        (170, 48, 3, 1),
        (170, 48, 128, 0),
        (170, usize::MAX, 128, 2),
    ] {
        assert!(
            super::value_split(
                ValueDevice {
                    multiprocessors: sm,
                    warp_size: 32
                },
                heads,
                dim,
                slots
            )
            .is_err()
        );
    }
    for warp_size in [0, 3] {
        assert!(
            super::value_split(
                ValueDevice {
                    multiprocessors: 170,
                    warp_size
                },
                48,
                128,
                1
            )
            .is_err()
        );
    }
}

// Tile IR is only specialized at launch; compile the snapshot-free kernels on header-only CI.
#[test]
fn snapshot_free_recurrent_kernels_compile_without_a_device() -> Result<(), Box<dyn Error>> {
    use cutile::cutile_compiler::compile_api::KernelCompiler;
    let lanes = |names: &[&'static str]| -> Vec<(&'static str, &'static [i32])> {
        names.iter().map(|&name| (name, &[1][..])).collect()
    };
    let rows = [
        "qkv0",
        "qkv1",
        "qkv2",
        "beta0",
        "beta1",
        "beta2",
        "alpha0",
        "alpha1",
        "alpha2",
        "a_log",
        "bias",
        "metadata0",
        "metadata1",
        "metadata2",
    ];
    let mut verify = lanes(&rows);
    verify.extend([
        ("out0", &[128, 128, 1][..]),
        ("out1", &[128, 128, 1][..]),
        ("out2", &[128, 128, 1][..]),
        ("state", &[16384, 128, 1][..]),
    ]);
    let mut commit = lanes(&rows);
    commit.push(("state", &[16384, 128, 1][..]));
    for generics in [
        vec!["16".into(), "48".into(), "128".into(), "32".into()],
        vec!["2".into(), "4".into(), "32".into(), "32".into()],
    ] {
        for (entry, strides) in [("delta_verify3", &verify), ("delta_commit3", &commit)] {
            KernelCompiler::new(fused::__module_ast_self, "fused", entry)
                .target("sm_120")
                .generics(generics.clone())
                .strides(strides)
                .compile()?;
        }
    }
    let mut conv_verify = lanes(&[
        "out0",
        "out1",
        "out2",
        "h0",
        "h1",
        "h2",
        "x0",
        "x1",
        "x2",
        "metadata0",
        "metadata1",
        "metadata2",
    ]);
    conv_verify.push(("w", &[4, 1][..]));
    let conv_commit = lanes(&[
        "h0",
        "h1",
        "h2",
        "x0",
        "x1",
        "x2",
        "metadata0",
        "metadata1",
        "metadata2",
    ]);
    for (entry, strides) in [
        ("conv_verify3", &conv_verify),
        ("conv_commit3", &conv_commit),
    ] {
        KernelCompiler::new(fused::__module_ast_self, "fused", entry)
            .target("sm_120")
            .strides(strides)
            .compile()?;
    }
    Ok(())
}

use crate::{device::CudaDevice, resident::recurrent::recurrent};
use cutile::prelude::*;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn upload(
    device: &CudaDevice,
    values: Vec<f32>,
    shape: &[usize],
) -> infer_core::Result<Tensor<f32>> {
    Arc::try_unwrap(device.upload(values, shape)?)
        .map_err(|_| infer_core::Error::invariant("slot recurrent test upload"))
}

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|i| f32::from(u16::try_from((i * salt + 7) % 89).unwrap()) / 97.0 - 0.43)
        .collect()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "{label}[{index}] nonfinite");
        assert!(
            (a - b).abs() <= 1e-5 * b.abs().max(1.0),
            "{label}[{index}]: {a} != {b}"
        );
    }
}

const PATTERNS: [[i32; 3]; 3] = [[4, 5, 6], [9, 10, -1], [-1, -1, -1]];

type Reference<S> = (Vec<Vec<f32>>, S);

fn conv_reference(
    device: &CudaDevice,
    inputs: &[Tensor<f32>],
    w: &Tensor<f32>,
    metadata: &[Arc<Tensor<i32>>],
    count: usize,
    channels: usize,
) -> Result<Reference<Vec<Vec<f32>>>, Box<dyn Error>> {
    let tile = CONV_KERNEL_TILE;
    // Reference: conv4 lane after lane, keeping the state after `count` active lanes.
    let mut state = (0..3)
        .map(|h| upload(device, values(channels, h + 3), &[channels]))
        .collect::<infer_core::Result<Vec<_>>>()?;
    let mut expected_out = Vec::new();
    let mut expected_state = None;
    for lane in 0..3 {
        let mut out = upload(device, vec![0.0; channels], &[channels])?;
        let [h0, h1, h2] = state.as_mut_slice() else {
            unreachable!()
        };
        recurrent::conv4(
            (&mut out).partition([tile]),
            h0.partition([tile]),
            h1.partition([tile]),
            h2.partition([tile]),
            &inputs[lane],
            w,
            &metadata[lane],
        )
        .sync_on(&device.stream)?;
        expected_out.push(device.read_borrowed(&out)?);
        if lane + 1 == count {
            expected_state = Some(
                state
                    .iter()
                    .map(|h| device.read_borrowed(h))
                    .collect::<infer_core::Result<Vec<_>>>()?,
            );
        }
    }
    Ok((expected_out, expected_state.ok_or("conv reference state")?))
}

type DeltaLane = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

fn delta_reference(
    device: &CudaDevice,
    lanes: &[DeltaLane],
    a_log: &Tensor<f32>,
    bias: &Tensor<f32>,
    metadata: &[Arc<Tensor<i32>>],
    count: usize,
    shape: [usize; 3],
) -> Result<Reference<Vec<f32>>, Box<dyn Error>> {
    let [key_heads, value_heads, dim] = shape;
    let state_len = value_heads * dim * dim;
    let generics = vec![
        key_heads.to_string(),
        value_heads.to_string(),
        dim.to_string(),
        dim.to_string(),
    ];
    let mut state = upload(device, values(state_len, 23), &[value_heads, dim, dim])?;
    let mut expected_out = Vec::new();
    let mut expected_state = None;
    for (lane, (qkv, beta, alpha)) in lanes.iter().enumerate() {
        let mut out = upload(device, vec![0.0; value_heads * dim], &[value_heads, 1, dim])?;
        recurrent::delta(
            (&mut out).partition([1, 1, dim]),
            (&mut state).partition([1, dim, dim]),
            qkv,
            beta,
            alpha,
            a_log,
            bias,
            &metadata[lane],
        )
        .generics(generics.clone())
        .sync_on(&device.stream)?;
        expected_out.push(device.read_borrowed(&out)?);
        if lane + 1 == count {
            expected_state = Some(device.read_borrowed(&state)?);
        }
    }
    Ok((expected_out, expected_state.ok_or("delta reference state")?))
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
fn conv_verify_and_commit_match_sequential_lanes() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let channels = 256;
    let tile = CONV_KERNEL_TILE;
    let w = upload(&device, values(channels * 4, 5), &[channels, 4])?;
    let inputs = (0..3)
        .map(|lane| upload(&device, values(channels, lane + 11), &[channels]))
        .collect::<infer_core::Result<Vec<_>>>()?;
    for positions in PATTERNS {
        let metadata = positions
            .iter()
            .map(|&p| device.upload(vec![p, 0, p, 0], &[4]))
            .collect::<infer_core::Result<Vec<_>>>()?;
        for count in 1..=3usize {
            let (expected_out, expected_state) =
                conv_reference(&device, &inputs, &w, &metadata, count, channels)?;
            let mut fused_state = (0..3)
                .map(|h| upload(&device, values(channels, h + 3), &[channels]))
                .collect::<infer_core::Result<Vec<_>>>()?;
            let mut outs = (0..3)
                .map(|_| upload(&device, vec![0.0; channels], &[channels]))
                .collect::<infer_core::Result<Vec<_>>>()?;
            {
                let [o0, o1, o2] = outs.as_mut_slice() else {
                    unreachable!()
                };
                let [h0, h1, h2] = fused_state.as_slice() else {
                    unreachable!()
                };
                fused::conv_verify3(
                    o0.partition([tile]),
                    o1.partition([tile]),
                    o2.partition([tile]),
                    h0,
                    h1,
                    h2,
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &w,
                    &metadata[0],
                    &metadata[1],
                    &metadata[2],
                )
                .sync_on(&device.stream)?;
            }
            for (lane, out) in outs.iter().enumerate() {
                assert_close(
                    &format!("{positions:?} conv out {lane}"),
                    &device.read_borrowed(out)?,
                    &expected_out[lane],
                );
            }
            {
                let [h0, h1, h2] = fused_state.as_mut_slice() else {
                    unreachable!()
                };
                fused::conv_commit3(
                    h0.partition([tile]),
                    h1.partition([tile]),
                    h2.partition([tile]),
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &metadata[0],
                    &metadata[1],
                    &metadata[2],
                    i32::try_from(count)?,
                )
                .sync_on(&device.stream)?;
            }
            for (h, expected) in fused_state.iter().zip(&expected_state) {
                assert_close(
                    &format!("{positions:?} count {count} conv state"),
                    &device.read_borrowed(h)?,
                    expected,
                );
            }
        }
    }
    Ok(())
}

struct DeltaFixture {
    lanes: Vec<DeltaLane>,
    a_log: Tensor<f32>,
    bias: Tensor<f32>,
    dim: usize,
    block: usize,
    generics: Vec<String>,
}

impl DeltaFixture {
    fn graph(
        &self,
        device: &CudaDevice,
        state: &Tensor<f32>,
        outs: &mut [Tensor<f32>],
        metadata: &[Arc<Tensor<i32>>],
        compile: &CompileOptions,
    ) -> TestResult<CudaGraph<()>> {
        let block = self.block;
        let [o0, o1, o2] = outs else { unreachable!() };
        Ok(CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                fused::delta_verify3(
                    o0.partition([1, 1, block]),
                    o1.partition([1, 1, block]),
                    o2.partition([1, 1, block]),
                    state,
                    &self.lanes[0].0,
                    &self.lanes[1].0,
                    &self.lanes[2].0,
                    &self.lanes[0].1,
                    &self.lanes[1].1,
                    &self.lanes[2].1,
                    &self.lanes[0].2,
                    &self.lanes[1].2,
                    &self.lanes[2].2,
                    &self.a_log,
                    &self.bias,
                    &metadata[0],
                    &metadata[1],
                    &metadata[2],
                )
                .generics(self.generics.clone())
                .compile_options(compile.clone()),
            )?;
            Ok(())
        })?)
    }

    fn verify(
        &self,
        device: &CudaDevice,
        state: &Tensor<f32>,
        outs: &mut [Tensor<f32>],
        metadata: &[Arc<Tensor<i32>>],
    ) -> TestResult {
        let block = self.block;
        let [o0, o1, o2] = outs else { unreachable!() };
        fused::delta_verify3(
            o0.partition([1, 1, block]),
            o1.partition([1, 1, block]),
            o2.partition([1, 1, block]),
            state,
            &self.lanes[0].0,
            &self.lanes[1].0,
            &self.lanes[2].0,
            &self.lanes[0].1,
            &self.lanes[1].1,
            &self.lanes[2].1,
            &self.lanes[0].2,
            &self.lanes[1].2,
            &self.lanes[2].2,
            &self.a_log,
            &self.bias,
            &metadata[0],
            &metadata[1],
            &metadata[2],
        )
        .generics(self.generics.clone())
        .sync_on(&device.stream)?;
        Ok(())
    }

    fn commit(
        &self,
        device: &CudaDevice,
        state: &mut Tensor<f32>,
        metadata: &[Arc<Tensor<i32>>],
        count: usize,
    ) -> TestResult {
        let dim = self.dim;
        let block = self.block;
        fused::delta_commit3(
            state.partition([1, dim, block]),
            &self.lanes[0].0,
            &self.lanes[1].0,
            &self.lanes[2].0,
            &self.lanes[0].1,
            &self.lanes[1].1,
            &self.lanes[2].1,
            &self.lanes[0].2,
            &self.lanes[1].2,
            &self.lanes[2].2,
            &self.a_log,
            &self.bias,
            &metadata[0],
            &metadata[1],
            &metadata[2],
            i32::try_from(count)?,
        )
        .generics(self.generics.clone())
        .sync_on(&device.stream)?;
        Ok(())
    }
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
fn delta_verify_and_commit_match_sequential_lanes() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    delta_reference_case(&device, [2, 4, 32], 32)
}

#[test]
#[ignore = "requires CUDA hardware; derived head grids, unchanged existing reference tolerance"]
fn wave_derived_delta_matches_existing_reference() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for geometry in [
        [2, 4, 32],
        [4, 16, 32],
        [4, 16, 128],
        [16, 48, 128],
        [16, 192, 128],
    ] {
        let [_, heads, dim] = geometry;
        let mut blocks = Vec::new();
        for slots in [1, 2, 4, 16] {
            let split = super::value_split(ValueDevice::new(&device)?, heads, dim, slots)?;
            let block = dim / split;
            if blocks.contains(&block) {
                continue;
            }
            blocks.push(block);
            delta_reference_case(&device, geometry, block)?;
        }
    }
    Ok(())
}

fn delta_scalar_reference(
    device: &CudaDevice,
    fixture: &DeltaFixture,
    positions: [i32; 3],
    count: usize,
    geometry: [usize; 3],
) -> TestResult<Reference<Vec<f32>>> {
    let [key_heads, value_heads, dim] = geometry;
    let geometry = DeltaGeometry {
        key_heads,
        value_heads,
        key_dim: dim,
        value_dim: dim,
        norm_epsilon: 1e-6,
    };
    let rate = device.read_borrowed(&fixture.a_log)?;
    let bias = device.read_borrowed(&fixture.bias)?;
    let mut state = values(value_heads * dim * dim, 23);
    let mut outputs = Vec::new();
    let mut accepted = None;
    for (lane, (qkv, beta, alpha)) in fixture.lanes.iter().enumerate() {
        let qkv = device.read_borrowed(qkv)?;
        let beta = device.read_borrowed(beta)?;
        let alpha = device.read_borrowed(alpha)?;
        let output = if positions[lane] >= 0 {
            geometry.step(&mut state, &[&qkv, &beta, &alpha, &rate, &bias])?
        } else {
            vec![0.0; value_heads * dim]
        };
        outputs.push(output);
        if lane + 1 == count {
            accepted = Some(state.clone());
        }
    }
    Ok((outputs, accepted.ok_or("scalar accepted state missing")?))
}

fn delta_reference_case(device: &CudaDevice, geometry: [usize; 3], block: usize) -> TestResult {
    let [key_heads, value_heads, dim] = geometry;
    let generics = vec![
        key_heads.to_string(),
        value_heads.to_string(),
        dim.to_string(),
        block.to_string(),
    ];
    let qkv_len = (2 * key_heads + value_heads) * dim;
    let state_len = value_heads * dim * dim;
    let lanes = (0..3)
        .map(|lane| {
            Ok((
                upload(device, values(qkv_len, lane + 17), &[qkv_len])?,
                upload(device, values(value_heads, lane + 13), &[value_heads])?,
                upload(device, values(value_heads, lane + 7), &[value_heads])?,
            ))
        })
        .collect::<infer_core::Result<Vec<_>>>()?;
    let a_log = upload(device, values(value_heads, 3), &[value_heads])?;
    let bias = upload(device, values(value_heads, 5), &[value_heads])?;
    let fixture = DeltaFixture {
        lanes,
        a_log,
        bias,
        dim,
        block,
        generics,
    };
    for positions in PATTERNS {
        let metadata = positions
            .iter()
            .map(|&p| device.upload(vec![p, 0, p, 0], &[4]))
            .collect::<infer_core::Result<Vec<_>>>()?;
        for count in 1..=3usize {
            let (expected_out, expected_state) = delta_reference(
                device,
                &fixture.lanes,
                &fixture.a_log,
                &fixture.bias,
                &metadata,
                count,
                [key_heads, value_heads, dim],
            )?;
            let (scalar_out, scalar_state) =
                delta_scalar_reference(device, &fixture, positions, count, geometry)?;
            let mut fused_state = upload(device, values(state_len, 23), &[value_heads, dim, dim])?;
            let mut outs = (0..3)
                .map(|_| upload(device, vec![0.0; value_heads * dim], &[value_heads, 1, dim]))
                .collect::<infer_core::Result<Vec<_>>>()?;
            fixture.verify(device, &fused_state, &mut outs, &metadata)?;
            for (lane, out) in outs.iter().enumerate() {
                let actual = device.read_borrowed(out)?;
                assert_close(
                    &format!("{positions:?} delta out {lane}"),
                    &actual,
                    &expected_out[lane],
                );
                assert_close("independent scalar output", &actual, &scalar_out[lane]);
            }
            fixture.commit(device, &mut fused_state, &metadata, count)?;
            assert_close(
                "independent scalar state",
                &device.read_borrowed(&fused_state)?,
                &scalar_state,
            );
            assert_close(
                &format!("{positions:?} count {count} delta state"),
                &device.read_borrowed(&fused_state)?,
                &expected_state,
            );
        }
    }
    Ok(())
}

/// Validate the non-power-of-two candidate axis and batched strides before hardware execution.
#[test]
fn batched_recurrent_kernels_compile_without_a_device() -> Result<(), Box<dyn Error>> {
    use cutile::cutile_compiler::compile_api::KernelCompiler;
    for (kh, vh, dim, block) in [(2, 4, 32, 32), (16, 48, 128, 32)] {
        let output_strides = [3 * vh * dim, vh * dim, 1];
        let state_strides = [dim * dim, dim, 1];
        let mut strides = vec![
            ("out", output_strides.as_slice()),
            ("state", state_strides.as_slice()),
        ];
        for name in ["qkv", "beta", "alpha", "parameters", "metadata"] {
            strides.push((name, &[1]));
        }
        KernelCompiler::new(fused::__module_ast_self, "fused", "delta_verify_batch")
            .target("sm_120")
            .generics(
                [kh, vh, dim, block, 4]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .strides(&strides)
            .compile()?;
        KernelCompiler::new(fused::__module_ast_self, "fused", "pack_parameters")
            .target("sm_120")
            .generics(vec![vh.to_string()])
            .strides(&[("out", &[vh, 1]), ("rate", &[1]), ("bias", &[1])])
            .compile()?;
    }
    Ok(())
}

#[path = "resident_slot_recurrent_batch.rs"]
mod batched;

#[test]
fn batched_commit_kernels_compile_without_a_device() -> Result<(), Box<dyn Error>> {
    use cutile::cutile_compiler::compile_api::KernelCompiler;
    for (kh, vh, dim, block) in [(2, 4, 32, 32), (16, 48, 128, 32)] {
        let state_strides = [vh * dim * dim, dim, 1];
        let mut strides = vec![("state", state_strides.as_slice())];
        for name in ["qkv", "beta", "alpha", "parameters", "metadata", "counts"] {
            strides.push((name, &[1]));
        }
        KernelCompiler::new(fused::__module_ast_self, "fused", "delta_commit_batch")
            .target("sm_120")
            .generics(
                [kh, vh, dim, block, 4]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .strides(&strides)
            .compile()?;
    }
    for channels in [32, 10240] {
        let row_strides = [channels, 1];
        let mut strides = vec![];
        for name in ["h0", "h1", "h2", "input"] {
            strides.push((name, row_strides.as_slice()));
        }
        strides.extend([("metadata", &[1][..]), ("counts", &[1][..])]);
        KernelCompiler::new(fused::__module_ast_self, "fused", "conv_commit_batch")
            .target("sm_120")
            .generics(vec!["4".into()])
            .strides(&strides)
            .compile()?;
    }
    Ok(())
}

fn wave_perf_fixture(
    device: &CudaDevice,
    geometry: [usize; 3],
    block: usize,
) -> TestResult<DeltaFixture> {
    let [key_heads, value_heads, dim] = geometry;
    let qkv_len = (2 * key_heads + value_heads) * dim;
    let lanes = (0..3)
        .map(|lane| {
            Ok((
                upload(device, values(qkv_len, lane + 17), &[qkv_len])?,
                upload(device, values(value_heads, lane + 13), &[value_heads])?,
                upload(device, values(value_heads, lane + 7), &[value_heads])?,
            ))
        })
        .collect::<infer_core::Result<Vec<_>>>()?;
    Ok(DeltaFixture {
        lanes,
        a_log: upload(device, values(value_heads, 3), &[value_heads])?,
        bias: upload(device, values(value_heads, 5), &[value_heads])?,
        dim,
        block,
        generics: geometry
            .into_iter()
            .chain([block])
            .map(|v| v.to_string())
            .collect(),
    })
}

fn wave_perf_pair(
    device: &CudaDevice,
    a: &CudaGraph<()>,
    b: &CudaGraph<()>,
    options: &BenchOptions,
    reverse: bool,
) -> TestResult<(Vec<f32>, Vec<f32>)> {
    let launch = |graph: &CudaGraph<()>| {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|e| tensor_error(&e.to_string()))
    };
    let (a, b) = if reverse { (b, a) } else { (a, b) };
    let (first, second) = do_bench_paired(&device.stream, options, |_| launch(a), |_| launch(b))?;
    let pair = (first.times_ms().to_vec(), second.times_ms().to_vec());
    Ok(if reverse { (pair.1, pair.0) } else { pair })
}

fn wave_perf_case(
    device: &CudaDevice,
    geometry: [usize; 3],
    split: usize,
    compile: &CompileOptions,
) -> TestResult {
    let [_, heads, dim] = geometry;
    // Frozen pre-change policy on the selected 170-SM device uses four splits.
    let old_block = dim / 4;
    let block = dim / split;
    if block == old_block {
        println!(
            "{}",
            serde_json::json!({"kind":"head_grid_control","shape":geometry,"split":split})
        );
        return Ok(());
    }
    let mut fixture = wave_perf_fixture(device, geometry, old_block)?;
    let state = upload(device, values(heads * dim * dim, 23), &[heads, dim, dim])?;
    let mut output = (0..3)
        .map(|_| upload(device, vec![0.0; heads * dim], &[heads, 1, dim]))
        .collect::<infer_core::Result<Vec<_>>>()?;
    let metadata = PATTERNS[0]
        .iter()
        .map(|&p| device.upload(vec![p, 0, p, 0], &[4]))
        .collect::<infer_core::Result<Vec<_>>>()?;
    let a = fixture.graph(
        device,
        &state,
        &mut output,
        &metadata,
        &CompileOptions::default(),
    )?;
    a.launch().sync_on(&device.stream)?;
    let expected = output
        .iter()
        .map(|t| device.read_borrowed(t))
        .collect::<infer_core::Result<Vec<_>>>()?;
    fixture.block = block;
    fixture.generics[3] = block.to_string();
    let b = fixture.graph(device, &state, &mut output, &metadata, compile)?;
    b.launch().sync_on(&device.stream)?;
    for (actual, expected) in output.iter().zip(&expected) {
        assert_close(
            "derived launch performance preflight",
            &device.read_borrowed(actual)?,
            expected,
        );
    }
    let mut options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::ZERO,
        min_reps: 3,
        max_reps: 3,
        clear_l2: true,
    };
    let (x, y) = wave_perf_pair(device, &a, &b, &options, false)?;
    println!(
        "{}",
        serde_json::json!({"kind":"head_grid_canary","shape":geometry,"old_split":4,"split":split,"baseline_ms":x,"candidate_ms":y})
    );
    if median(&y) > median(&x) * 1.10 {
        return Err(
            format!("head-aware launch gross regression at {geometry:?}, split={split}").into(),
        );
    }
    options.warmup = Duration::from_millis(50);
    options.rep = Duration::from_millis(100);
    options.min_reps = 10;
    options.max_reps = 30;
    for reverse in [false, true] {
        fixture.block = old_block;
        fixture.generics[3] = old_block.to_string();
        let fresh_a = fixture.graph(
            device,
            &state,
            &mut output,
            &metadata,
            &CompileOptions::default(),
        )?;
        fixture.block = block;
        fixture.generics[3] = block.to_string();
        let fresh_b = fixture.graph(device, &state, &mut output, &metadata, compile)?;
        for cold in [true, false] {
            options.clear_l2 = cold;
            let (x, y) = wave_perf_pair(device, &fresh_a, &fresh_b, &options, reverse)?;
            println!(
                "{}",
                serde_json::json!({"kind":"head_grid_timing","shape":geometry,"split":split,"cold":cold,"reverse":reverse,"baseline_ms":x,"candidate_ms":y})
            );
            if median(&y) > median(&x) * 1.02 {
                return Err("head-grid paired median regression exceeds two percent".into());
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA release, 170-SM baseline; no serving acceptance from this operator screen"]
fn wave_derived_head_grid_performance_screen() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    if cfg!(debug_assertions) {
        return Err("release profile required".into());
    }
    let device = CudaDevice::new(0)?;
    let sm = device.profile()?.multiprocessors;
    if sm != 170 {
        return Err("frozen operator reference requires the 170-SM device".into());
    }
    for geometry in [[2, 4, 32], [4, 16, 128], [16, 48, 128], [16, 192, 128]] {
        let [_, heads, dim] = geometry;
        wave_perf_case(
            &device,
            geometry,
            super::value_split(ValueDevice::new(&device)?, heads, dim, 1)?,
            &CompileOptions::default(),
        )?;
    }
    Ok(())
}
