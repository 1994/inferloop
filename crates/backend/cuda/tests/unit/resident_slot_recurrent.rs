use super::fused;
use crate::constants::CONV_KERNEL_TILE;
use std::error::Error;
use std::time::Instant;

#[test]
fn value_split_follows_sm_heads_slots_and_legal_value_divisors() {
    assert_eq!(super::value_split(32, 12, 4, 128), 1);
    assert_eq!(super::value_split(80, 12, 4, 128), 2);
    assert_eq!(super::value_split(170, 12, 4, 128), 4);
    // A tail value width only accepts legal divisors.
    assert_eq!(super::value_split(170, 12, 4, 96), 4);
    assert_eq!(super::value_split(170, 12, 4, 100), 4);
    assert_eq!(super::value_split(170, 12, 4, 72), 4);
    assert_eq!(super::value_split(170, 12, 4, 6), 2);
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

type TestResult = Result<(), Box<dyn Error>>;

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
    geometry: ([usize; 3], usize),
) -> Result<Reference<Vec<f32>>, Box<dyn Error>> {
    let ([key_heads, value_heads, dim], block) = geometry;
    let state_len = value_heads * dim * dim;
    let generics = vec![
        key_heads.to_string(),
        value_heads.to_string(),
        dim.to_string(),
        block.to_string(),
    ];
    let mut state = upload(device, values(state_len, 23), &[value_heads, dim, dim])?;
    let mut expected_out = Vec::new();
    let mut expected_state = None;
    for (lane, (qkv, beta, alpha)) in lanes.iter().enumerate() {
        let mut out = upload(device, vec![0.0; value_heads * dim], &[value_heads, 1, dim])?;
        recurrent::delta(
            (&mut out).partition([1, 1, block]),
            (&mut state).partition([1, dim, block]),
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
    fn verify(
        &self,
        device: &CudaDevice,
        state: &Tensor<f32>,
        outs: &mut [Tensor<f32>],
        metadata: &[Arc<Tensor<i32>>],
    ) -> TestResult {
        let [o0, o1, o2] = outs else { unreachable!() };
        fused::delta_verify3(
            o0.partition([1, 1, self.block]),
            o1.partition([1, 1, self.block]),
            o2.partition([1, 1, self.block]),
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
        fused::delta_commit3(
            state.partition([1, dim, self.block]),
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
    let (key_heads, value_heads, dim) = (2usize, 4usize, 32usize);
    let generics = vec![
        key_heads.to_string(),
        value_heads.to_string(),
        dim.to_string(),
        dim.to_string(),
    ];
    let qkv_len = (2 * key_heads + value_heads) * dim;
    let state_len = value_heads * dim * dim;
    let lanes = (0..3)
        .map(|lane| {
            Ok((
                upload(&device, values(qkv_len, lane + 17), &[qkv_len])?,
                upload(&device, values(value_heads, lane + 13), &[value_heads])?,
                upload(&device, values(value_heads, lane + 7), &[value_heads])?,
            ))
        })
        .collect::<infer_core::Result<Vec<_>>>()?;
    let a_log = upload(&device, values(value_heads, 3), &[value_heads])?;
    let bias = upload(&device, values(value_heads, 5), &[value_heads])?;
    let fixture = DeltaFixture {
        lanes,
        a_log,
        bias,
        dim,
        block: dim,
        generics,
    };
    for positions in PATTERNS {
        let metadata = positions
            .iter()
            .map(|&p| device.upload(vec![p, 0, p, 0], &[4]))
            .collect::<infer_core::Result<Vec<_>>>()?;
        for count in 1..=3usize {
            let (expected_out, expected_state) = delta_reference(
                &device,
                &fixture.lanes,
                &fixture.a_log,
                &fixture.bias,
                &metadata,
                count,
                ([key_heads, value_heads, dim], fixture.block),
            )?;
            let mut fused_state = upload(&device, values(state_len, 23), &[value_heads, dim, dim])?;
            let mut outs = (0..3)
                .map(|_| {
                    upload(
                        &device,
                        vec![0.0; value_heads * dim],
                        &[value_heads, 1, dim],
                    )
                })
                .collect::<infer_core::Result<Vec<_>>>()?;
            fixture.verify(&device, &fused_state, &mut outs, &metadata)?;
            for (lane, out) in outs.iter().enumerate() {
                assert_close(
                    &format!("{positions:?} delta out {lane}"),
                    &device.read_borrowed(out)?,
                    &expected_out[lane],
                );
            }
            fixture.commit(&device, &mut fused_state, &metadata, count)?;
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

#[test]
#[ignore = "requires CUDA hardware; actual 27B GDN geometry and wave-derived split, safe-run"]
fn wave_split_delta_matches_reference_at_27b_geometry() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (key_heads, value_heads, dim) = (16usize, 48usize, 128usize);
    let slots = 3usize;
    let split = super::value_split(170, key_heads, slots, dim);
    let block = super::fused_block(dim, split);
    assert_eq!(split, 4, "27B wave split");
    assert_eq!(block, 32, "27B fused block");
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
                upload(&device, values(qkv_len, lane + 17), &[qkv_len])?,
                upload(&device, values(value_heads, lane + 13), &[value_heads])?,
                upload(&device, values(value_heads, lane + 7), &[value_heads])?,
            ))
        })
        .collect::<infer_core::Result<Vec<_>>>()?;
    let fixture = DeltaFixture {
        lanes,
        a_log: upload(&device, values(value_heads, 3), &[value_heads])?,
        bias: upload(&device, values(value_heads, 5), &[value_heads])?,
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
                &device,
                &fixture.lanes,
                &fixture.a_log,
                &fixture.bias,
                &metadata,
                count,
                ([key_heads, value_heads, dim], fixture.block),
            )?;
            let mut fused_state = upload(&device, values(state_len, 23), &[value_heads, dim, dim])?;
            let mut outs = (0..3)
                .map(|_| {
                    upload(
                        &device,
                        vec![0.0; value_heads * dim],
                        &[value_heads, 1, dim],
                    )
                })
                .collect::<infer_core::Result<Vec<_>>>()?;
            let started = Instant::now();
            fixture.verify(&device, &fused_state, &mut outs, &metadata)?;
            let verify_us = started.elapsed().as_micros();
            for (lane, out) in outs.iter().enumerate() {
                assert_close(
                    &format!("{positions:?} split{split} delta out {lane}"),
                    &device.read_borrowed(out)?,
                    &expected_out[lane],
                );
            }
            let started = Instant::now();
            fixture.commit(&device, &mut fused_state, &metadata, count)?;
            let commit_us = started.elapsed().as_micros();
            assert_close(
                &format!("{positions:?} split{split} count {count} delta state"),
                &device.read_borrowed(&fused_state)?,
                &expected_state,
            );
            eprintln!(
                "VALUE_SPLIT_REAL sm=170 key_heads={key_heads} value_heads={value_heads} dim={dim} slots={slots} split={split} block={block} count={count} verify_us={verify_us} commit_us={commit_us}"
            );
        }
    }
    Ok(())
}
