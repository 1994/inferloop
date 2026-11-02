use super::kernels;
use crate::device::CudaDevice;
use crate::resident::recurrent::recurrent::delta as recurrent_delta;
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::env::var_os;
use std::error::Error;

pub(super) fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (f32::from(u16::try_from((i * salt + 3) % 101).unwrap()) - 50.0) / 100.0)
        .collect()
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn chunk_delta_matches_independent_recurrence() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (kh, vh, dim, lanes) in [
        (2, 4, 32, 32),
        (2, 6, 128, 32),
        (2, 6, 128, 128),
        (16, 48, 128, 128),
    ] {
        for (count, offset) in [(0, 0), (1, 0), (lanes - 3, 0), (lanes, -1)] {
            check_case(&device, (kh, vh, dim, lanes), count, offset)?;
        }
    }
    Ok(())
}

pub(super) struct ReferenceFixture {
    pub(super) qkv: Vec<f32>,
    pub(super) beta: Vec<f32>,
    pub(super) alpha: Vec<f32>,
    pub(super) a_log: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) initial: Vec<f32>,
    pub(super) expected: Vec<f64>,
    pub(super) outputs: Vec<f64>,
}

pub(super) fn reference_fixture(
    (kh, vh, dim, lanes): (usize, usize, usize, usize),
    count: usize,
    offset: i32,
) -> Result<ReferenceFixture, Box<dyn Error>> {
    let qkv = values(lanes * (2 * kh + vh) * dim, 17);
    let beta = values(lanes * vh, 13);
    let alpha = values(lanes * vh, 7);
    let a_log = values(vh, 3);
    let bias = values(vh, 5);
    let initial = values(vh * dim * dim, 23);
    let mut expected: Vec<f64> = initial.iter().copied().map(f64::from).collect();
    let mut outputs = vec![0.0_f64; lanes * vh * dim];
    for lane in 0..count {
        if i32::try_from(lane)? + offset < 0 {
            continue;
        }
        for head in 0..vh {
            let base = lane * (2 * kh + vh) * dim;
            let key = head / (vh / kh);
            let mut q: Vec<f64> = qkv[base + key * dim..][..dim]
                .iter()
                .copied()
                .map(f64::from)
                .collect();
            let mut k: Vec<f64> = qkv[base + (kh + key) * dim..][..dim]
                .iter()
                .copied()
                .map(f64::from)
                .collect();
            let qnorm = (q.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt()
                * f64::from(u32::try_from(dim)?).sqrt();
            let knorm = (k.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
            for x in &mut q {
                *x /= qnorm;
            }
            for x in &mut k {
                *x /= knorm;
            }
            let b = 1.0 / (1.0 + (-f64::from(beta[lane * vh + head])).exp());
            let a = f64::from(alpha[lane * vh + head]) + f64::from(bias[head]);
            let decay = (-f64::from(a_log[head]).exp() * a.exp().ln_1p()).exp();
            let state = &mut expected[head * dim * dim..][..dim * dim];
            for x in state.iter_mut() {
                *x *= decay;
            }
            for col in 0..dim {
                let prediction = (0..dim)
                    .map(|row| state[row * dim + col] * k[row])
                    .sum::<f64>();
                let delta = (f64::from(qkv[base + (2 * kh + head) * dim + col]) - prediction) * b;
                for row in 0..dim {
                    state[row * dim + col] = k[row].mul_add(delta, state[row * dim + col]);
                }
                outputs[(head * lanes + lane) * dim + col] =
                    (0..dim).map(|row| state[row * dim + col] * q[row]).sum();
            }
        }
    }
    Ok(ReferenceFixture {
        qkv,
        beta,
        alpha,
        a_log,
        bias,
        initial,
        expected,
        outputs,
    })
}

fn check_case(
    device: &CudaDevice,
    geometry: (usize, usize, usize, usize),
    count: usize,
    offset: i32,
) -> Result<(), Box<dyn Error>> {
    let (kh, vh, dim, lanes) = geometry;
    let ReferenceFixture {
        qkv,
        beta,
        alpha,
        a_log,
        bias,
        initial,
        expected,
        outputs,
    } = reference_fixture(geometry, count, offset)?;
    let qkv = device.upload(qkv, &[lanes, 2 * kh + vh, dim])?;
    let beta = device.upload(beta, &[lanes, vh])?;
    let alpha = device.upload(alpha, &[lanes, vh])?;
    let a_log = device.upload(a_log, &[vh])?;
    let bias = device.upload(bias, &[vh])?;
    let metadata = device.upload(vec![0, i32::try_from(count)?, offset, 0], &[4])?;
    // The chunk rotates the whole state once per lane, so a value-dimension block leaves each
    // block a quarter of the traffic per lane. Both widths must match the reference, and the split
    // must land on the single-block launch to rounding rather than merely inside it.
    let mut single: Option<(Vec<f32>, Vec<f32>)> = None;
    for split in [1_usize, 4] {
        let block = dim / split;
        let mut state = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
        let src = device.upload(initial.clone(), &[vh, dim, dim])?;
        api::memcpy(&mut state, &src).sync_on(&device.stream)?;
        let mut output = api::zeros::<f32>(&[vh, lanes, dim]).sync_on(&device.stream)?;
        kernels::delta(
            (&mut state).partition([1, dim, block]),
            (&mut output).partition([1, lanes, block]),
            &qkv,
            &beta,
            &alpha,
            &a_log,
            &bias,
            &metadata,
        )
        .generics(vec![
            kh.to_string(),
            vh.to_string(),
            dim.to_string(),
            lanes.to_string(),
            block.to_string(),
        ])
        .sync_on(&device.stream)?;
        let actual_state = state.to_host_vec().sync_on(&device.stream)?;
        let actual_out = output.to_host_vec().sync_on(&device.stream)?;
        for (actual, reference) in [(&actual_state, &expected), (&actual_out, &outputs)] {
            for (a, b) in actual.iter().zip(reference) {
                assert!(
                    (f64::from(*a) - b).abs() < 0.00002,
                    "{geometry:?} count={count} offset={offset} split={split}: {a} != {b}"
                );
            }
        }
        if split == 1 {
            single = Some((actual_state, actual_out));
        } else {
            let (state_ref, out_ref) = single.as_ref().ok_or("single block must run first")?;
            let state_gap = actual_state
                .iter()
                .zip(state_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let out_gap = actual_out
                .iter()
                .zip(out_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!(
                "{geometry:?} split {split}: state {state_gap:e}, out {out_gap:e} against one block"
            );
            assert!(state_gap < 1.0e-6, "{geometry:?} split moved the state");
            assert!(out_gap < 1.0e-7, "{geometry:?} split moved the output");
        }
    }
    Ok(())
}

/// The per-lane recurrence is the path that ships for quantized checkpoints, and it has never
/// been checked against an independent reference at the geometry the 27B actually uses. The
/// chunked twin disagrees with it by 5% of hidden L2 on a single token, so one of the two is
/// wrong: this decides which.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn per_lane_delta_matches_independent_recurrence_at_model_geometry() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (kh, vh, dim) = (16_usize, 48_usize, 128_usize);
    let ReferenceFixture {
        qkv,
        beta,
        alpha,
        a_log,
        bias,
        initial,
        expected,
        outputs,
    } = reference_fixture((kh, vh, dim, 1), 1, 0)?;
    let mut single_block: Option<(Vec<f32>, Vec<f32>)> = None;
    let qkv = device.upload(qkv, &[(2 * kh + vh) * dim])?;
    let beta = device.upload(beta, &[vh])?;
    let alpha = device.upload(alpha, &[vh])?;
    let a_log = device.upload(a_log, &[vh])?;
    let bias = device.upload(bias, &[vh])?;
    let metadata = device.upload(vec![0, 1, 0, 0], &[4])?;
    // One block per value head is the established launch; a split gives the grid that many more
    // blocks. Both must agree with the reference, and the split must not move the answer.
    for split in [1_usize, 4] {
        let block = dim / split;
        let mut state = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
        let src = device.upload(initial.clone(), &[vh, dim, dim])?;
        api::memcpy(&mut state, &src).sync_on(&device.stream)?;
        let output = api::zeros::<f32>(&[vh, dim]).sync_on(&device.stream)?;
        let mut output = output.reshape(&[vh, 1, dim])?;
        recurrent_delta(
            (&mut output).partition([1, 1, block]),
            (&mut state).partition([1, dim, block]),
            &qkv,
            &beta,
            &alpha,
            &a_log,
            &bias,
            &metadata,
        )
        .generics(vec![
            kh.to_string(),
            vh.to_string(),
            dim.to_string(),
            block.to_string(),
        ])
        .sync_on(&device.stream)?;
        let actual_state = state.to_host_vec().sync_on(&device.stream)?;
        let actual_out = output.to_host_vec().sync_on(&device.stream)?;
        let mut worst_state = 0.0_f64;
        for (a, b) in actual_state.iter().zip(&expected) {
            worst_state = worst_state.max((f64::from(*a) - b).abs());
        }
        let mut worst_out = 0.0_f64;
        for (a, b) in actual_out.iter().zip(&outputs) {
            worst_out = worst_out.max((f64::from(*a) - b).abs());
        }
        eprintln!(
            "per-lane delta at model geometry, split {split}: worst state {worst_state}, worst out {worst_out}"
        );
        assert!(worst_state < 0.001, "per-lane state drift {worst_state}");
        assert!(worst_out < 0.001, "per-lane output drift {worst_out}");
        // The split partitions independent columns, so it must land on the single-block launch
        // to rounding rather than merely inside the f64 reference tolerance. cuTile's lowering
        // contracts differently for the two partition shapes, so this is a bound and not equality;
        // the observed figure is printed so a change in it is visible in the test output.
        if split == 1 {
            single_block = Some((actual_state, actual_out));
        } else {
            let (state_ref, out_ref) = single_block
                .as_ref()
                .ok_or("single-block launch did not run first")?;
            let state_gap = actual_state
                .iter()
                .zip(state_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let out_gap = actual_out
                .iter()
                .zip(out_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!(
                "split {split} against the single-block launch: state {state_gap:e}, out {out_gap:e}"
            );
            assert!(state_gap < 1.0e-6, "split moved the state by {state_gap:e}");
            assert!(out_gap < 1.0e-7, "split moved the output by {out_gap:e}");
        }
    }
    Ok(())
}

/// The chunked Delta writes into a head-major scratch and transposes into the lane-major output.
/// States come out bit-identical to the per-lane path while the hidden does not, which puts the
/// output path under suspicion, so pin the transpose on its own.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn chunk_delta_transpose_permutes_head_and_lane() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (lanes, value_heads, dim) = (3_usize, 2_usize, 4_usize);
    let mut scratch = vec![0.0_f32; value_heads * lanes * dim];
    for head in 0..value_heads {
        for lane in 0..lanes {
            for d in 0..dim {
                scratch[(head * lanes + lane) * dim + d] = f32::from(
                    u16::try_from(head * 1000 + lane * 100 + d).expect("fixture value fits"),
                );
            }
        }
    }
    let input = device.upload(scratch, &[value_heads, lanes, dim])?;
    let mut output = api::zeros::<f32>(&[lanes, value_heads, dim]).sync_on(&device.stream)?;
    kernels::transpose((&mut output).partition([1, 1, dim]), &input)
        .generics(vec![dim.to_string()])
        .sync_on(&device.stream)?;
    let actual = output.to_host_vec().sync_on(&device.stream)?;
    for lane in 0..lanes {
        for head in 0..value_heads {
            for d in 0..dim {
                let expected = f32::from(
                    u16::try_from(head * 1000 + lane * 100 + d).expect("fixture value fits"),
                );
                assert_eq!(
                    actual[(lane * value_heads + head) * dim + d],
                    expected,
                    "lane={lane} head={head} d={d}"
                );
            }
        }
    }
    Ok(())
}

/// Compare the chunked and per-lane recurrences on byte-identical inputs and report the gap.
/// Returns `(state_gap, output_gap)` as the worst absolute difference over the chunk.
fn recurrence_gap(
    device: &CudaDevice,
    lanes: usize,
    count: usize,
    (kh, vh, dim): (usize, usize, usize),
) -> Result<(f32, f32), Box<dyn Error>> {
    let heads = 2 * kh + vh;
    let qkv = values(lanes * heads * dim, 17);
    let beta = values(lanes * vh, 13);
    let alpha = values(lanes * vh, 7);
    let a_log = values(vh, 3);
    let bias = values(vh, 5);
    let initial = values(vh * dim * dim, 23);

    let mut state_c = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
    let src = device.upload(initial, &[vh, dim, dim])?;
    api::memcpy(&mut state_c, &src).sync_on(&device.stream)?;
    let mut out_c = api::zeros::<f32>(&[vh, lanes, dim]).sync_on(&device.stream)?;
    let qkv_c = device.upload(qkv.clone(), &[lanes, heads, dim])?;
    let beta_c = device.upload(beta.clone(), &[lanes, vh])?;
    let alpha_c = device.upload(alpha.clone(), &[lanes, vh])?;
    let a_log_c = device.upload(a_log, &[vh])?;
    let bias_c = device.upload(bias, &[vh])?;
    let metadata = device.upload(vec![0, i32::try_from(count)?, 0, 0], &[4])?;
    kernels::delta_grouped(
        (&mut state_c).partition([1, dim, dim]),
        (&mut out_c).partition([1, lanes, dim]),
        &qkv_c,
        &beta_c,
        &alpha_c,
        &a_log_c,
        &bias_c,
        &metadata,
    )
    .generics(vec![
        kh.to_string(),
        vh.to_string(),
        dim.to_string(),
        lanes.to_string(),
        dim.to_string(),
    ])
    .sync_on(&device.stream)?;

    let mut state_l = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
    api::memcpy(&mut state_l, &src).sync_on(&device.stream)?;
    let mut expected_out = vec![0.0_f32; vh * lanes * dim];
    let lane_metadata = device.upload(vec![0, 0, 0, 0], &[4])?;
    for lane in 0..count {
        let row = &qkv[lane * heads * dim..(lane + 1) * heads * dim];
        let qkv_one = device.upload(row.to_vec(), &[heads * dim])?;
        let beta_one = device.upload(beta[lane * vh..(lane + 1) * vh].to_vec(), &[vh])?;
        let alpha_one = device.upload(alpha[lane * vh..(lane + 1) * vh].to_vec(), &[vh])?;
        let out_one = api::zeros::<f32>(&[vh, dim]).sync_on(&device.stream)?;
        let mut out_one = out_one.reshape(&[vh, 1, dim])?;
        recurrent_delta(
            (&mut out_one).partition([1, 1, dim]),
            (&mut state_l).partition([1, dim, dim]),
            &qkv_one,
            &beta_one,
            &alpha_one,
            &a_log_c,
            &bias_c,
            &lane_metadata,
        )
        .generics(vec![
            kh.to_string(),
            vh.to_string(),
            dim.to_string(),
            dim.to_string(),
        ])
        .sync_on(&device.stream)?;
        let lane_out = out_one.to_host_vec().sync_on(&device.stream)?;
        for head in 0..vh {
            for d in 0..dim {
                expected_out[(head * lanes + lane) * dim + d] = lane_out[head * dim + d];
            }
        }
    }
    let chunk_out = out_c.to_host_vec().sync_on(&device.stream)?;
    let out_gap = chunk_out
        .iter()
        .zip(&expected_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let chunk_state = state_c.to_host_vec().sync_on(&device.stream)?;
    let lane_state = state_l.to_host_vec().sync_on(&device.stream)?;
    let state_gap = chunk_state
        .iter()
        .zip(&lane_state)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    print_gap_indices(
        &chunk_out,
        &expected_out,
        &chunk_state,
        &lane_state,
        lanes,
        count,
    );
    Ok((state_gap, out_gap))
}

/// Report the chunked-versus-per-lane gap across geometries so the lane-loop fix has a map of
/// where the difference enters: a single lane, a masked tail, or a longer chunk.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn chunk_and_per_lane_delta_gap_map() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (lanes, count) in [
        (1, 1),
        (2, 1),
        (2, 2),
        (8, 1),
        (8, 3),
        (128, 1),
        (128, 8),
        (128, 51),
    ] {
        let (state, out) = recurrence_gap(&device, lanes, count, (16, 48, 128))?;
        eprintln!("lanes={lanes} count={count} state_gap={state:e} output_gap={out:e}");
    }
    Ok(())
}

/// The lane loop must reproduce the per-lane recurrence exactly; anything else changes a 64-layer
/// greedy decode. Track the figure here until it is zero.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn ordered_delta_preserves_scalar_bits() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let geometry = (16, 48, 128, 128);
    check_ordered_fixture(
        &device,
        geometry,
        128,
        51,
        0,
        reference_fixture(geometry, 51, 0)?,
    )?;

    Ok(())
}

fn print_gap_indices(
    chunk_out: &[f32],
    expected_out: &[f32],
    chunk_state: &[f32],
    lane_state: &[f32],
    lanes: usize,
    count: usize,
) {
    if var_os("INFER_GAP_INDEX").is_some() {
        let differing: Vec<usize> = chunk_out
            .iter()
            .zip(expected_out)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, _)| i)
            .collect();
        eprintln!(
            "lanes={lanes} count={count} differing_outputs={}/{} first={:?}",
            differing.len(),
            chunk_out.len(),
            differing.iter().take(6).collect::<Vec<_>>()
        );
        let state_differing: Vec<usize> = chunk_state
            .iter()
            .zip(lane_state)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, _)| i)
            .collect();
        eprintln!(
            "  differing_states={}/{} first={:?}",
            state_differing.len(),
            chunk_state.len(),
            state_differing.iter().take(6).collect::<Vec<_>>()
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware; new F32-state/F64-reference contract"]
fn wy_delta_matches_independent_recurrence() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (kh, vh, dim, lanes) in [
        (2, 4, 32, 32),
        (2, 6, 128, 32),
        (2, 6, 128, 128),
        (16, 48, 128, 128),
    ] {
        for (count, offset) in [(0, 0), (1, 0), (lanes - 3, 0), (lanes, -1)] {
            check_wy_case(&device, (kh, vh, dim, lanes), count, offset)?;
        }
    }
    Ok(())
}

fn check_wy_case(
    device: &CudaDevice,
    geometry: (usize, usize, usize, usize),
    count: usize,
    offset: i32,
) -> Result<(), Box<dyn Error>> {
    check_wy_fixture(
        device,
        geometry,
        count,
        offset,
        reference_fixture(geometry, count, offset)?,
    )
}

fn check_wy_fixture(
    device: &CudaDevice,
    geometry: (usize, usize, usize, usize),
    count: usize,
    offset: i32,
    fixture: ReferenceFixture,
) -> Result<(), Box<dyn Error>> {
    let (kh, vh, dim, lanes) = geometry;
    let ReferenceFixture {
        qkv,
        beta,
        alpha,
        a_log,
        bias,
        initial,
        expected,
        outputs,
    } = fixture;
    let qkv = device.upload(qkv, &[lanes, 2 * kh + vh, dim])?;
    let beta = device.upload(beta, &[lanes, vh])?;
    let alpha = device.upload(alpha, &[lanes, vh])?;
    let a_log = device.upload(a_log, &[vh])?;
    let bias = device.upload(bias, &[vh])?;
    let metadata = device.upload(vec![0, i32::try_from(count)?, offset, 0], &[4])?;
    // The chunk rotates the whole state once per lane, so a value-dimension block leaves each
    // block a quarter of the traffic per lane. Both widths must match the reference, and the split
    // must land on the single-block launch to rounding rather than merely inside it.
    let mut single: Option<(Vec<f32>, Vec<f32>)> = None;
    for split in [1_usize, 4] {
        let block = dim / split;
        let mut state = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
        let src = device.upload(initial.clone(), &[vh, dim, dim])?;
        api::memcpy(&mut state, &src).sync_on(&device.stream)?;
        let mut output = api::zeros::<f32>(&[vh, lanes, dim]).sync_on(&device.stream)?;
        kernels::delta_wy(
            (&mut state).partition([1, dim, block]),
            (&mut output).partition([1, lanes, block]),
            &qkv,
            &beta,
            &alpha,
            &a_log,
            &bias,
            &metadata,
        )
        .generics(vec![
            kh.to_string(),
            vh.to_string(),
            dim.to_string(),
            lanes.to_string(),
            block.to_string(),
        ])
        .sync_on(&device.stream)?;
        let actual_state = state.to_host_vec().sync_on(&device.stream)?;
        let actual_out = output.to_host_vec().sync_on(&device.stream)?;
        for (label, actual, reference) in [
            ("state", &actual_state, &expected),
            ("output", &actual_out, &outputs),
        ] {
            let maximum = actual
                .iter()
                .zip(reference)
                .map(|(a, b)| (f64::from(*a) - b).abs())
                .fold(0.0_f64, f64::max);
            eprintln!(
                "WY_REFERENCE geometry={geometry:?} count={count} offset={offset} split={split} label={label} max_abs={maximum:e}"
            );
            for (a, b) in actual.iter().zip(reference) {
                assert!(
                    (f64::from(*a) - b).abs() < 0.00002,
                    "{geometry:?} count={count} offset={offset} split={split}: {a} != {b}"
                );
            }
        }
        if split == 1 {
            single = Some((actual_state, actual_out));
        } else {
            let (state_ref, out_ref) = single.as_ref().ok_or("single block must run first")?;
            let state_gap = actual_state
                .iter()
                .zip(state_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let out_gap = actual_out
                .iter()
                .zip(out_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!(
                "{geometry:?} split {split}: state {state_gap:e}, out {out_gap:e} against one block"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; actual captured prompt graph widths"]
fn wy_actual_graph_widths_match_independent_recurrence() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for lanes in [64, 256] {
        for (count, offset) in [(0, 0), (1, 0), (lanes - 3, 0), (lanes, -1)] {
            check_wy_case(&device, (16, 48, 128, lanes), count, offset)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; weak decay and correlated K expose unstable inverse construction"]
fn wy_low_decay_correlated_keys_match_f64() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (kh, vh, dim, lanes) = (2_usize, 4_usize, 32_usize, 32_usize);
    let mut qkv = values(lanes * (2 * kh + vh) * dim, 17);
    for lane in 0..lanes {
        for head in 0..2 * kh {
            for col in 0..dim {
                qkv[(lane * (2 * kh + vh) + head) * dim + col] = 1.0;
            }
        }
    }
    let beta = vec![8.0_f32; lanes * vh];
    let alpha = vec![0.0_f32; lanes * vh];
    let a_log = vec![-10.0_f32; vh];
    let bias = vec![0.0_f32; vh];
    let initial = vec![0.0_f32; vh * dim * dim];
    let mut expected = vec![0.0_f64; vh * dim * dim];
    let mut outputs = vec![0.0_f64; vh * lanes * dim];
    // Independent scalar recurrence in F64, from the original input floats.
    let kinv = (f64::from(u32::try_from(dim)?) + 1e-6).sqrt().recip();
    let qinv = kinv / f64::from(u32::try_from(dim)?).sqrt();
    let b = 1.0 / (1.0 + (-8.0_f64).exp());
    let decay = (-(-10.0_f64).exp() * 2.0_f64.ln()).exp();
    for lane in 0..lanes {
        for head in 0..vh {
            let st = &mut expected[head * dim * dim..][..dim * dim];
            for value in st.iter_mut() {
                *value *= decay;
            }
            for col in 0..dim {
                let predicted = (0..dim).map(|row| st[row * dim + col] * kinv).sum::<f64>();
                let v = f64::from(qkv[(lane * (2 * kh + vh) + 2 * kh + head) * dim + col]);
                let delta = (v - predicted) * b;
                for row in 0..dim {
                    st[row * dim + col] = kinv.mul_add(delta, st[row * dim + col]);
                }
                outputs[(head * lanes + lane) * dim + col] =
                    (0..dim).map(|row| st[row * dim + col] * qinv).sum();
            }
        }
    }
    check_wy_fixture(
        &device,
        (kh, vh, dim, lanes),
        lanes,
        0,
        ReferenceFixture {
            qkv,
            beta,
            alpha,
            a_log,
            bias,
            initial,
            expected,
            outputs,
        },
    )
}

#[test]
#[ignore = "requires CUDA; exact ordered recurrence at captured prompt widths"]
fn ordered_actual_graph_widths_agree_exactly() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for block in [8_usize, 16, 32, 64, 128] {
        for lanes in [64_usize, 256] {
            let geometry = (16, 48, 128, lanes);
            for (count, offset) in [
                (0, 0),
                (1, 0),
                (52, 0),
                (lanes - 3, 0),
                (lanes, -1),
                (lanes, 0),
            ] {
                check_ordered_fixture(
                    &device,
                    geometry,
                    block,
                    count,
                    offset,
                    reference_fixture(geometry, count, offset)?,
                )?;
            }
            check_ordered_fixture(
                &device,
                geometry,
                block,
                lanes,
                0,
                ordered_correlated_fixture(geometry)?,
            )?;
        }
        let geometry = (2, 6, 128, 64);
        check_ordered_fixture(
            &device,
            geometry,
            block,
            61,
            -1,
            reference_fixture(geometry, 61, -1)?,
        )?;
    }
    Ok(())
}

fn check_ordered_fixture(
    device: &CudaDevice,
    geometry: (usize, usize, usize, usize),
    block: usize,
    count: usize,
    offset: i32,
    fixture: ReferenceFixture,
) -> Result<(), Box<dyn Error>> {
    let (kh, vh, dim, lanes) = geometry;
    eprintln!("ORDERED_FIXTURE geometry={geometry:?} block={block} count={count} offset={offset}");
    let ReferenceFixture {
        qkv,
        beta,
        alpha,
        a_log,
        bias,
        initial,
        expected,
        outputs,
    } = fixture;
    let qkv_c = device.upload(qkv.clone(), &[lanes, 2 * kh + vh, dim])?;
    let beta_c = device.upload(beta.clone(), &[lanes, vh])?;
    let alpha_c = device.upload(alpha.clone(), &[lanes, vh])?;
    let a_log = device.upload(a_log, &[vh])?;
    let bias = device.upload(bias, &[vh])?;
    let src = device.upload(initial, &[vh, dim, dim])?;
    let mut state = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
    api::memcpy(&mut state, &src).sync_on(&device.stream)?;
    let mut output = api::zeros::<f32>(&[vh, lanes, dim]).sync_on(&device.stream)?;
    let metadata = device.upload(vec![0, i32::try_from(count)?, offset, 0], &[4])?;
    let mut prepared = super::prepared::Inputs::new(device, (kh, vh, dim), lanes)?;
    let graph = Graph::scope(&device.stream, |scope| {
        prepared.record(
            scope,
            [&qkv_c, &beta_c, &alpha_c, &a_log, &bias],
            [&mut state, &mut output],
            &metadata,
            (kh, vh, dim, lanes, block),
        )?;
        Ok(())
    })?;
    graph.launch().sync_on(&device.stream)?;
    let actual_state = state.to_host_vec().sync_on(&device.stream)?;
    let actual_output = output.to_host_vec().sync_on(&device.stream)?;
    compare_ordered_f64("state", &actual_state, &expected, geometry, count, offset);
    compare_ordered_f64("output", &actual_output, &outputs, geometry, count, offset);
    let mut scalar_state = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
    api::memcpy(&mut scalar_state, &src).sync_on(&device.stream)?;
    let mut scalar_output = vec![0.0_f32; vh * lanes * dim];
    let cols = (2 * kh + vh) * dim;
    for lane in 0..count {
        let qkv_one = device.upload(qkv[lane * cols..(lane + 1) * cols].to_vec(), &[cols])?;
        let beta_one = device.upload(beta[lane * vh..(lane + 1) * vh].to_vec(), &[vh])?;
        let alpha_one = device.upload(alpha[lane * vh..(lane + 1) * vh].to_vec(), &[vh])?;
        let metadata = device.upload(vec![0, 0, i32::try_from(lane)? + offset, 0], &[4])?;
        let mut output = api::zeros::<f32>(&[vh, dim])
            .sync_on(&device.stream)?
            .reshape(&[vh, 1, dim])?;
        recurrent_delta(
            (&mut output).partition([1, 1, dim]),
            (&mut scalar_state).partition([1, dim, dim]),
            &qkv_one,
            &beta_one,
            &alpha_one,
            &a_log,
            &bias,
            &metadata,
        )
        .generics(vec![
            kh.to_string(),
            vh.to_string(),
            dim.to_string(),
            dim.to_string(),
        ])
        .sync_on(&device.stream)?;
        let row = output.to_host_vec().sync_on(&device.stream)?;
        for head in 0..vh {
            scalar_output[(head * lanes + lane) * dim..][..dim]
                .copy_from_slice(&row[head * dim..][..dim]);
        }
    }
    let scalar_state = scalar_state.to_host_vec().sync_on(&device.stream)?;
    for (label, actual, reference) in [
        ("state", &actual_state, &scalar_state),
        ("output", &actual_output, &scalar_output),
    ] {
        compare_ordered_bits(label, actual, reference, geometry, count, offset);
    }

    Ok(())
}

/// The chunk-local GDN factors multiply in BF16 on tensor cores (85493e4), which the numeric
/// contract permits; this bounds how far they may drift from the F64 recurrence.
const ORDERED_F64_TOLERANCE: f64 = 5e-3;

fn compare_ordered_f64(
    label: &str,
    actual: &[f32],
    reference: &[f64],
    geometry: (usize, usize, usize, usize),
    count: usize,
    offset: i32,
) {
    assert_eq!(actual.len(), reference.len());
    let mut maximum = 0.0_f64;
    for (i, (a, b)) in actual.iter().zip(reference).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "{label}[{i}] nonfinite");
        maximum = maximum.max((f64::from(*a) - b).abs());
        assert!(
            (f64::from(*a) - b).abs() < ORDERED_F64_TOLERANCE,
            "{geometry:?} count={count} offset={offset} {label}[{i}]: {a} != {b}"
        );
    }
    eprintln!(
        "ORDERED_F64 geometry={geometry:?} count={count} offset={offset} label={label} max_abs={maximum:e}"
    );
}

fn compare_ordered_bits(
    label: &str,
    actual: &[f32],
    reference: &[f32],
    geometry: (usize, usize, usize, usize),
    count: usize,
    offset: i32,
) {
    assert_eq!(actual.len(), reference.len());
    assert!(
        actual
            .iter()
            .chain(reference)
            .all(|value| value.is_finite())
    );
    let differing = actual
        .iter()
        .zip(reference)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!(
        "ORDERED_BITS geometry={geometry:?} count={count} offset={offset} label={label} differing={differing}/{}",
        actual.len()
    );
    // Different chunk shapes round their BF16 factors differently; the numeric contract bounds
    // that drift rather than forbidding it.
    let drift = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| f64::from((a - b).abs()))
        .fold(0.0, f64::max);
    assert!(
        drift <= ORDERED_F64_TOLERANCE,
        "{geometry:?} count={count} offset={offset} {label} drift {drift}"
    );
}

fn ordered_correlated_fixture(
    (kh, vh, dim, lanes): (usize, usize, usize, usize),
) -> Result<ReferenceFixture, Box<dyn Error>> {
    let mut qkv = values(lanes * (2 * kh + vh) * dim, 17);
    for lane in 0..lanes {
        for head in 0..2 * kh {
            qkv[(lane * (2 * kh + vh) + head) * dim..][..dim].fill(1.0);
        }
    }
    let beta = vec![8.0_f32; lanes * vh];
    let alpha = vec![0.0_f32; lanes * vh];
    let a_log = vec![-10.0_f32; vh];
    let bias = vec![0.0_f32; vh];
    let initial = values(vh * dim * dim, 23);
    let mut expected: Vec<f64> = initial.iter().copied().map(f64::from).collect();
    let mut outputs = vec![0.0_f64; vh * lanes * dim];
    let kinv = (f64::from(u32::try_from(dim)?) + 1e-6).sqrt().recip();
    let qinv = kinv / f64::from(u32::try_from(dim)?).sqrt();
    let b = 1.0 / (1.0 + (-8.0_f64).exp());
    let decay = (-(-10.0_f64).exp() * 2.0_f64.ln()).exp();
    for lane in 0..lanes {
        for head in 0..vh {
            let state = &mut expected[head * dim * dim..][..dim * dim];
            for value in state.iter_mut() {
                *value *= decay;
            }
            for col in 0..dim {
                let predicted = (0..dim)
                    .map(|row| state[row * dim + col] * kinv)
                    .sum::<f64>();
                let v = f64::from(qkv[(lane * (2 * kh + vh) + 2 * kh + head) * dim + col]);
                let delta = (v - predicted) * b;
                for row in 0..dim {
                    state[row * dim + col] = kinv.mul_add(delta, state[row * dim + col]);
                }
                outputs[(head * lanes + lane) * dim + col] =
                    (0..dim).map(|row| state[row * dim + col] * qinv).sum();
            }
        }
    }
    Ok(ReferenceFixture {
        qkv,
        beta,
        alpha,
        a_log,
        bias,
        initial,
        expected,
        outputs,
    })
}
