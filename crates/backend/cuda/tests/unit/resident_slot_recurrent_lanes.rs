//! Lane-generic recurrent kernels against the three-lane batch kernels on the first three
//! lanes of every slot of a wider verification. Convolution matches bit for bit; the delta
//! loop contracts its arithmetic differently from the unrolled kernel, so it is held to the
//! existing reference tolerance.
use super::{assert_close, upload, values};
use crate::test_support::f32_state;
use crate::{
    constants::CONV_KERNEL_TILE,
    device::CudaDevice,
    resident::{
        slot_convolution::kernels as convolution,
        slot_recurrent::{ValueDevice, fused, value_split},
    },
};
use cutile::half::bf16;
use cutile::prelude::*;
use std::error::Error;
use std::time::{Duration, Instant};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const KH: usize = 16;
const VH: usize = 48;
const D: usize = 128;
const QKV: usize = (2 * KH + VH) * D;
const CHANNELS: usize = 1024;
const WIDE: usize = 8;
const NARROW: usize = 3;
const SLOTS: usize = 3;
const FIELDS: usize = 4;
/// Launches averaged into one kernel timing.
const TIMED_LAUNCHES: usize = 20;

/// Rows `slot * WIDE + lane` for `lane < NARROW`, packed as `slot * NARROW + lane`.
fn narrow_rows<T: Clone>(wide: &[T], width: usize) -> Vec<T> {
    (0..SLOTS)
        .flat_map(|slot| {
            (0..NARROW).flat_map(move |lane| {
                let row = slot * WIDE + lane;
                row * width..(row + 1) * width
            })
        })
        .map(|index| wide[index].clone())
        .collect()
}

/// Every lane active except the tail of slot 1, so padding inside a slot is covered.
fn metadata() -> Vec<i32> {
    (0..SLOTS * WIDE)
        .flat_map(|row| {
            let (slot, lane) = (row / WIDE, row % WIDE);
            let position = if slot == 1 && lane >= 2 {
                -1
            } else {
                i32::try_from(row).unwrap()
            };
            [position, 0, position, 0]
        })
        .collect()
}

fn generics(values: &[usize]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

fn assert_bits(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{label}[{index}]: {a} != {b}");
    }
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn wide_delta_lanes_match_three_lane_reference() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let block = D / value_split(ValueDevice::new(&device)?, VH, D, SLOTS)?;
    let state = values(SLOTS * VH * D * D, 23);
    let qkv = values(SLOTS * WIDE * QKV, 17);
    let beta = values(SLOTS * WIDE * VH, 13);
    let alpha = values(SLOTS * WIDE * VH, 7);
    let meta = metadata();
    let mut parameters = values(VH, 3);
    parameters.extend(values(VH, 5));
    let parameters = upload(&device, parameters, &[2 * VH])?;
    let wide = (
        upload(&device, qkv.clone(), &[qkv.len()])?,
        upload(&device, beta.clone(), &[beta.len()])?,
        upload(&device, alpha.clone(), &[alpha.len()])?,
        device.upload(meta.clone(), &[meta.len()])?,
    );
    let narrow = (
        upload(&device, narrow_rows(&qkv, QKV), &[SLOTS * NARROW * QKV])?,
        upload(&device, narrow_rows(&beta, VH), &[SLOTS * NARROW * VH])?,
        upload(&device, narrow_rows(&alpha, VH), &[SLOTS * NARROW * VH])?,
        device.upload(narrow_rows(&meta, FIELDS), &[SLOTS * NARROW * FIELDS])?,
    );
    let states = upload(&device, state.clone(), &[SLOTS * VH, D, D])?;
    let mut wide_out = upload(
        &device,
        vec![0.0; SLOTS * WIDE * VH * D],
        &[SLOTS, WIDE, VH * D],
    )?;
    let mut narrow_out = upload(
        &device,
        vec![0.0; SLOTS * NARROW * VH * D],
        &[SLOTS, NARROW, VH * D],
    )?;
    fused::delta_verify_lanes(
        (&mut wide_out).partition([1, WIDE, block]),
        &states,
        &wide.0,
        &wide.1,
        &wide.2,
        &parameters,
        &wide.3,
    )
    .generics(f32_state(generics(&[KH, VH, D, block, FIELDS, WIDE])))
    .sync_on(device.stream())?;
    fused::delta_verify_batch(
        (&mut narrow_out).partition([1, NARROW, block]),
        &states,
        &narrow.0,
        &narrow.1,
        &narrow.2,
        &parameters,
        &narrow.3,
    )
    .generics(f32_state(generics(&[KH, VH, D, block, FIELDS])))
    .sync_on(device.stream())?;
    let row = VH * D;
    assert_close(
        "delta verify",
        &narrow_rows(&device.read_borrowed(&wide_out)?, row),
        &device.read_borrowed(&narrow_out)?,
    );
    for counts in [[1, 2, 3], [3, 1, 2], [0, 3, 1]] {
        let counts = device.upload(counts.to_vec(), &[SLOTS])?;
        let mut wide_state = upload(&device, state.clone(), &[SLOTS, VH * D, D])?;
        let mut narrow_state = upload(&device, state.clone(), &[SLOTS, VH * D, D])?;
        fused::delta_commit_lanes(
            (&mut wide_state).partition([1, D, block]),
            &wide.0,
            &wide.1,
            &wide.2,
            &parameters,
            &wide.3,
            &counts,
        )
        .generics(f32_state(generics(&[KH, VH, D, block, FIELDS, WIDE])))
        .sync_on(device.stream())?;
        fused::delta_commit_batch(
            (&mut narrow_state).partition([1, D, block]),
            &narrow.0,
            &narrow.1,
            &narrow.2,
            &parameters,
            &narrow.3,
            &counts,
        )
        .generics(f32_state(generics(&[KH, VH, D, block, FIELDS])))
        .sync_on(device.stream())?;
        assert_close(
            "delta commit",
            &device.read_borrowed(&wide_state)?,
            &device.read_borrowed(&narrow_state)?,
        );
    }
    Ok(())
}

/// The chunked delta verification against the serial lanes kernel it replaces: every candidate
/// output, including a slot whose tail lanes are padding, within the same tolerance, for both
/// checkpoint precisions and the value splits a three-slot and a full 27B pool take.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn chunked_delta_verify_matches_serial_lanes() -> TestResult {
    for block in [32, 64] {
        for bf16_state in [false, true] {
            chunked_matches_serial(block, bf16_state)?;
        }
    }
    Ok(())
}

/// Average device time of `launch` over repeated synchronized launches.
fn timed(mut launch: impl FnMut() -> TestResult) -> TestResult<Duration> {
    launch()?;
    let start = Instant::now();
    for _ in 0..TIMED_LAUNCHES {
        launch()?;
    }
    Ok(start.elapsed() / u32::try_from(TIMED_LAUNCHES)?)
}

fn chunked_matches_serial(block: usize, bf16_state: bool) -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let state = values(SLOTS * VH * D * D, 23);
    let qkv = values(SLOTS * WIDE * QKV, 17);
    let beta = values(SLOTS * WIDE * VH, 13);
    let alpha = values(SLOTS * WIDE * VH, 7);
    let meta = metadata();
    let mut parameters = values(VH, 3);
    parameters.extend(values(VH, 5));
    let parameters = upload(&device, parameters, &[2 * VH])?;
    let qkv = upload(&device, qkv.clone(), &[qkv.len()])?;
    let beta = upload(&device, beta.clone(), &[beta.len()])?;
    let alpha = upload(&device, alpha.clone(), &[alpha.len()])?;
    let meta = device.upload(meta.clone(), &[meta.len()])?;
    let shape = [SLOTS, WIDE, VH * D];
    let mut serial = upload(&device, vec![0.0; SLOTS * WIDE * VH * D], &shape)?;
    let mut chunked = upload(&device, vec![0.0; SLOTS * WIDE * VH * D], &shape)?;
    let mut serial_generics = f32_state(generics(&[KH, VH, D, block, FIELDS, WIDE]));
    let terms = if bf16_state { 1 } else { 3 };
    let mut chunked_generics = f32_state(generics(&[KH, VH, D, block, FIELDS, WIDE, terms]));
    if bf16_state {
        serial_generics[0] = String::from("bf16");
        chunked_generics[0] = String::from("bf16");
    }
    let label = format!("chunked delta verify, {block}-column block, bf16 state {bf16_state}");
    let state_f32 = upload(&device, state.clone(), &[SLOTS * VH, D, D])?;
    let state_bf16 = device.upload(
        state
            .iter()
            .map(|&value| bf16::from_f32(value))
            .collect::<Vec<_>>(),
        &[SLOTS * VH, D, D],
    )?;
    macro_rules! run {
        ($kernel:ident, $out:expr, $generics:expr, $state:expr) => {
            fused::$kernel(
                ($out).partition([1, WIDE, block]),
                $state,
                &qkv,
                &beta,
                &alpha,
                &parameters,
                &meta,
            )
            .generics($generics.clone())
            .sync_on(device.stream())
        };
    }
    let (serial_time, chunked_time) = if bf16_state {
        (
            timed(|| {
                Ok(run!(
                    delta_verify_lanes,
                    &mut serial,
                    serial_generics,
                    &state_bf16
                )
                .map(drop)?)
            })?,
            timed(|| {
                Ok(run!(
                    delta_verify_chunked,
                    &mut chunked,
                    chunked_generics,
                    &state_bf16
                )
                .map(drop)?)
            })?,
        )
    } else {
        (
            timed(|| {
                Ok(run!(delta_verify_lanes, &mut serial, serial_generics, &state_f32).map(drop)?)
            })?,
            timed(|| {
                Ok(run!(
                    delta_verify_chunked,
                    &mut chunked,
                    chunked_generics,
                    &state_f32
                )
                .map(drop)?)
            })?,
        )
    };
    eprintln!("{label}: serial {serial_time:?}, chunked {chunked_time:?}");
    assert_close(
        &label,
        &device.read_borrowed(&chunked)?,
        &device.read_borrowed(&serial)?,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn wide_convolution_lanes_match_three_lane_bits() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let input = values(SLOTS * WIDE * CHANNELS, 11);
    let meta = metadata();
    let history: Vec<_> = (0..NARROW)
        .map(|h| values(SLOTS * CHANNELS, h + 3))
        .collect();
    let weight = upload(&device, values(CHANNELS * 4, 19), &[CHANNELS, 4])?;
    let wide_input = upload(&device, input.clone(), &[SLOTS, WIDE, CHANNELS])?;
    let narrow_input = upload(
        &device,
        narrow_rows(&input, CHANNELS),
        &[SLOTS, NARROW, CHANNELS],
    )?;
    let wide_meta = device.upload(meta.clone(), &[meta.len()])?;
    let narrow_meta = device.upload(narrow_rows(&meta, FIELDS), &[SLOTS * NARROW * FIELDS])?;
    let h = history
        .iter()
        .map(|values| upload(&device, values.clone(), &[SLOTS, CHANNELS]))
        .collect::<infer_core::Result<Vec<_>>>()?;
    let mut wide_out = upload(
        &device,
        vec![0.0; SLOTS * WIDE * CHANNELS],
        &[SLOTS, WIDE, CHANNELS],
    )?;
    let mut narrow_out = upload(
        &device,
        vec![0.0; SLOTS * NARROW * CHANNELS],
        &[SLOTS, NARROW, CHANNELS],
    )?;
    convolution::verify_lanes(
        (&mut wide_out).partition([1, WIDE, CONV_KERNEL_TILE]),
        &wide_input,
        &h[0],
        &h[1],
        &h[2],
        &weight,
        &wide_meta,
    )
    .generics(generics(&[CHANNELS, FIELDS, WIDE]))
    .sync_on(device.stream())?;
    convolution::verify(
        (&mut narrow_out).partition([1, NARROW, CONV_KERNEL_TILE]),
        &narrow_input,
        &h[0],
        &h[1],
        &h[2],
        &weight,
        &narrow_meta,
    )
    .generics(generics(&[CHANNELS, FIELDS]))
    .sync_on(device.stream())?;
    assert_bits(
        "convolution verify",
        &narrow_rows(&device.read_borrowed(&wide_out)?, CHANNELS),
        &device.read_borrowed(&narrow_out)?,
    );
    convolution_commits_match(
        &device,
        &history,
        (&wide_input, &narrow_input),
        (&wide_meta, &narrow_meta),
    )
}

/// Commit every count pattern through both kernels and require identical histories.
fn convolution_commits_match(
    device: &CudaDevice,
    history: &[Vec<f32>],
    (wide_input, narrow_input): (&Tensor<f32>, &Tensor<f32>),
    (wide_meta, narrow_meta): (&Tensor<i32>, &Tensor<i32>),
) -> TestResult {
    for counts in [[1, 2, 3], [3, 1, 2], [0, 3, 1]] {
        let counts = device.upload(counts.to_vec(), &[SLOTS])?;
        let mut wide_h = history
            .iter()
            .map(|values| upload(device, values.clone(), &[SLOTS, CHANNELS]))
            .collect::<infer_core::Result<Vec<_>>>()?;
        let mut narrow_h = history
            .iter()
            .map(|values| upload(device, values.clone(), &[SLOTS, CHANNELS]))
            .collect::<infer_core::Result<Vec<_>>>()?;
        let [w0, w1, w2] = wide_h.as_mut_slice() else {
            unreachable!()
        };
        fused::conv_commit_lanes(
            w0.partition([1, CONV_KERNEL_TILE]),
            w1.partition([1, CONV_KERNEL_TILE]),
            w2.partition([1, CONV_KERNEL_TILE]),
            &wide_input.view(&[SLOTS * WIDE, CHANNELS])?,
            wide_meta,
            &counts,
        )
        .generics(generics(&[FIELDS, WIDE]))
        .sync_on(device.stream())?;
        let [n0, n1, n2] = narrow_h.as_mut_slice() else {
            unreachable!()
        };
        fused::conv_commit_batch(
            n0.partition([1, CONV_KERNEL_TILE]),
            n1.partition([1, CONV_KERNEL_TILE]),
            n2.partition([1, CONV_KERNEL_TILE]),
            &narrow_input.view(&[SLOTS * NARROW, CHANNELS])?,
            narrow_meta,
            &counts,
        )
        .generics(generics(&[FIELDS]))
        .sync_on(device.stream())?;
        for (wide, narrow) in wide_h.iter().zip(&narrow_h) {
            assert_bits(
                "convolution commit",
                &device.read_borrowed(wide)?,
                &device.read_borrowed(narrow)?,
            );
        }
    }
    Ok(())
}

/// Temporary occupancy sweep of the chunked kernel at the production geometry.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn chunked_occupancy_sweep() -> TestResult {
    use cutile::tile_kernel::CompileOptions;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let block = 32;
    let state = values(SLOTS * VH * D * D, 23);
    let qkv = upload(
        &device,
        values(SLOTS * WIDE * QKV, 17),
        &[SLOTS * WIDE * QKV],
    )?;
    let beta = upload(&device, values(SLOTS * WIDE * VH, 13), &[SLOTS * WIDE * VH])?;
    let alpha = upload(&device, values(SLOTS * WIDE * VH, 7), &[SLOTS * WIDE * VH])?;
    let meta = metadata();
    let meta = device.upload(meta.clone(), &[meta.len()])?;
    let mut parameters = values(VH, 3);
    parameters.extend(values(VH, 5));
    let parameters = upload(&device, parameters, &[2 * VH])?;
    let state = device.upload(
        state
            .iter()
            .map(|&value| bf16::from_f32(value))
            .collect::<Vec<_>>(),
        &[SLOTS * VH, D, D],
    )?;
    let mut out = upload(
        &device,
        vec![0.0; SLOTS * WIDE * VH * D],
        &[SLOTS, WIDE, VH * D],
    )?;
    let mut g = generics(&[KH, VH, D, block, FIELDS, WIDE, 1]);
    g.insert(0, String::from("bf16"));
    for occupancy in [None, Some(1), Some(2), Some(4), Some(8)] {
        let options = occupancy.map_or_else(CompileOptions::default, |o| {
            CompileOptions::default().occupancy(o)
        });
        let time = timed(|| {
            fused::delta_verify_chunked(
                (&mut out).partition([1, WIDE, block]),
                &state,
                &qkv,
                &beta,
                &alpha,
                &parameters,
                &meta,
            )
            .generics(g.clone())
            .compile_options(options.clone())
            .sync_on(device.stream())?;
            Ok(())
        })?;
        eprintln!("occupancy {occupancy:?}: chunked {time:?}");
    }
    Ok(())
}
