use super::checkpoint;
use crate::constants::CONV_KERNEL_TILE;
use crate::{device::CudaDevice, resident::recurrent::recurrent};
use cutile::prelude::*;
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

fn upload<T: DType>(
    device: &CudaDevice,
    values: Vec<T>,
    shape: &[usize],
) -> infer_core::Result<Tensor<T>> {
    Arc::try_unwrap(device.upload(values, shape)?)
        .map_err(|_| infer_core::Error::invariant("checkpoint test upload ownership"))
}

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|index| f32::from(u16::try_from((index * salt + 3) % 97).unwrap() + 1) / 193.0 - 0.251)
        .collect()
}

fn assert_bits(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label} index={index}"
        );
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} index={index}"
        );
    }
}

struct ConvBuffers {
    original: Vec<Tensor<f32>>,
    candidate: Vec<Tensor<f32>>,
    snapshots: Vec<Tensor<f32>>,
    original_output: Tensor<f32>,
    candidate_output: Tensor<f32>,
}

fn run_conv(
    device: &CudaDevice,
    buffers: &mut ConvBuffers,
    input: &Tensor<f32>,
    weights: &Tensor<f32>,
    metadata: &Tensor<i32>,
) -> TestResult {
    let [h0, h1, h2] = buffers.original.as_mut_slice() else {
        return Err("original convolution fixture layout".into());
    };
    recurrent::conv4(
        (&mut buffers.original_output).partition([CONV_KERNEL_TILE]),
        h0.partition([CONV_KERNEL_TILE]),
        h1.partition([CONV_KERNEL_TILE]),
        h2.partition([CONV_KERNEL_TILE]),
        input,
        weights,
        metadata,
    )
    .sync_on(device.stream())?;
    let [h0, h1, h2] = buffers.candidate.as_mut_slice() else {
        return Err("candidate convolution fixture layout".into());
    };
    let [c0, c1, c2] = buffers.snapshots.as_mut_slice() else {
        return Err("convolution checkpoint fixture layout".into());
    };
    checkpoint::conv4_checkpoint(
        (&mut buffers.candidate_output).partition([CONV_KERNEL_TILE]),
        h0.partition([CONV_KERNEL_TILE]),
        h1.partition([CONV_KERNEL_TILE]),
        h2.partition([CONV_KERNEL_TILE]),
        c0.partition([CONV_KERNEL_TILE]),
        c1.partition([CONV_KERNEL_TILE]),
        c2.partition([CONV_KERNEL_TILE]),
        input,
        weights,
        metadata,
    )
    .sync_on(device.stream())?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
fn fused_conv_checkpoints_preserve_state_output_and_inactive_tail() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for channels in [32, 128, 257] {
        let initial: Vec<Vec<f32>> = (0..3).map(|index| values(channels, index + 3)).collect();
        let upload_states = |host: &[Vec<f32>]| {
            host.iter()
                .map(|row| upload(&device, row.clone(), &[channels]))
                .collect::<infer_core::Result<Vec<_>>>()
        };
        let mut previous = vec![vec![17.0; channels]; 3];
        let mut buffers = ConvBuffers {
            original: upload_states(&initial)?,
            candidate: upload_states(&initial)?,
            snapshots: upload_states(&previous)?,
            original_output: upload(&device, vec![0.0; channels], &[channels])?,
            candidate_output: upload(&device, vec![0.0; channels], &[channels])?,
        };
        let weights = upload(&device, values(channels * 4, 11), &[channels, 4])?;
        // Three active tokens with inactive replays between them retain nonzero state.
        for (step, position) in [0, -1, 1, -1, 2].into_iter().enumerate() {
            let input = upload(&device, values(channels, step + 17), &[channels])?;
            let metadata = upload(&device, vec![0, 0, position, 0], &[4])?;
            run_conv(&device, &mut buffers, &input, &weights, &metadata)?;
            assert_bits(
                "conv output",
                &device.read_borrowed(&buffers.candidate_output)?,
                &device.read_borrowed(&buffers.original_output)?,
            );
            for (index, ((candidate, original), snapshot)) in buffers
                .candidate
                .iter()
                .zip(&buffers.original)
                .zip(&buffers.snapshots)
                .enumerate()
            {
                let candidate = device.read_borrowed(candidate)?;
                assert_bits("conv state", &candidate, &device.read_borrowed(original)?);
                if position >= 0 {
                    previous[index] = candidate;
                }
                assert_bits(
                    "conv checkpoint",
                    &device.read_borrowed(snapshot)?,
                    &previous[index],
                );
            }
        }
    }
    Ok(())
}

struct DeltaBuffers {
    original_state: Tensor<f32>,
    candidate_state: Tensor<f32>,
    checkpoint: Tensor<f32>,
    original_output: Tensor<f32>,
    candidate_output: Tensor<f32>,
}

struct DeltaInputs {
    qkv: Tensor<f32>,
    beta: Tensor<f32>,
    alpha: Tensor<f32>,
    a_log: Tensor<f32>,
    bias: Tensor<f32>,
    metadata: Tensor<i32>,
}

fn run_delta(
    device: &CudaDevice,
    buffers: &mut DeltaBuffers,
    inputs: &DeltaInputs,
    (key_heads, value_heads, dim, block): (usize, usize, usize, usize),
) -> TestResult {
    let generics = vec![
        key_heads.to_string(),
        value_heads.to_string(),
        dim.to_string(),
        block.to_string(),
    ];
    recurrent::delta(
        (&mut buffers.original_output).partition([1, 1, block]),
        (&mut buffers.original_state).partition([1, dim, block]),
        &inputs.qkv,
        &inputs.beta,
        &inputs.alpha,
        &inputs.a_log,
        &inputs.bias,
        &inputs.metadata,
    )
    .generics(generics.clone())
    .sync_on(device.stream())?;
    checkpoint::delta_checkpoint(
        (&mut buffers.candidate_output).partition([1, 1, block]),
        (&mut buffers.candidate_state).partition([1, dim, block]),
        (&mut buffers.checkpoint).partition([1, dim, block]),
        &inputs.qkv,
        &inputs.beta,
        &inputs.alpha,
        &inputs.a_log,
        &inputs.bias,
        &inputs.metadata,
    )
    .generics(generics)
    .sync_on(device.stream())?;
    Ok(())
}

fn check_delta(
    device: &CudaDevice,
    (key_heads, value_heads, dim): (usize, usize, usize),
    split: usize,
) -> TestResult {
    let state_size = value_heads * dim * dim;
    let initial = values(state_size, 23);
    let mut previous = vec![17.0; state_size];
    let mut buffers = DeltaBuffers {
        original_state: upload(device, initial.clone(), &[value_heads, dim, dim])?,
        candidate_state: upload(device, initial, &[value_heads, dim, dim])?,
        checkpoint: upload(device, previous.clone(), &[value_heads, dim, dim])?,
        original_output: upload(device, vec![0.0; value_heads * dim], &[value_heads, 1, dim])?,
        candidate_output: upload(device, vec![0.0; value_heads * dim], &[value_heads, 1, dim])?,
    };
    // The original and checkpoint kernels see identical operands, including inactive metadata.
    for (step, position) in [0, -1, 1, -1, 2].into_iter().enumerate() {
        let qkv_columns = (2 * key_heads + value_heads) * dim;
        let inputs = DeltaInputs {
            qkv: upload(device, values(qkv_columns, step + 17), &[qkv_columns])?,
            beta: upload(device, values(value_heads, step + 13), &[value_heads])?,
            alpha: upload(device, values(value_heads, step + 7), &[value_heads])?,
            a_log: upload(device, values(value_heads, 3), &[value_heads])?,
            bias: upload(device, values(value_heads, 5), &[value_heads])?,
            metadata: upload(device, vec![0, 0, position, 0], &[4])?,
        };
        run_delta(
            device,
            &mut buffers,
            &inputs,
            (key_heads, value_heads, dim, dim / split),
        )?;
        assert_bits(
            "delta output",
            &device.read_borrowed(&buffers.candidate_output)?,
            &device.read_borrowed(&buffers.original_output)?,
        );
        let candidate = device.read_borrowed(&buffers.candidate_state)?;
        assert_bits(
            "delta state",
            &candidate,
            &device.read_borrowed(&buffers.original_state)?,
        );
        if position >= 0 {
            previous = candidate;
        }
        assert_bits(
            "delta checkpoint",
            &device.read_borrowed(&buffers.checkpoint)?,
            &previous,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
fn fused_delta_checkpoints_preserve_nonzero_state_and_split_arithmetic() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for geometry in [(1, 2, 32), (16, 48, 128)] {
        for split in [1, 4] {
            check_delta(&device, geometry, split)?;
        }
    }
    Ok(())
}
