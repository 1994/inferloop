//! Device regression for the complete fused vision block MLP.
use super::{MlpLayer, run};
use crate::device::CudaDevice;
use crate::vision::program::{add, gelu, project_with};
use cutile::half::bf16;
use infer_core::Result;

/// Independent scalar GELU comparison, including GPU transcendental rounding.
const TOLERANCE: f32 = 1e-4;

fn weights(device: &CudaDevice, hidden: usize, width: usize) -> Result<MlpLayer> {
    let mut first = vec![bf16::ZERO; hidden * width];
    let mut second = vec![bf16::ZERO; hidden * width];
    for column in 0..width {
        first[column * hidden + column % hidden] = bf16::from_f32(0.5);
        first[column * hidden + (column + 1) % hidden] = bf16::from_f32(-0.25);
    }
    for column in 0..hidden {
        second[column * width + column % width] = bf16::from_f32(0.25);
        second[column * width + (column + 1) % width] = bf16::from_f32(-0.125);
    }
    Ok(MlpLayer {
        fc1: (
            device.upload(first, &[width, hidden])?,
            device.upload(vec![bf16::from_f32(0.125); width], &[width])?,
        ),
        fc2: (
            device.upload(second, &[hidden, width])?,
            device.upload(vec![bf16::from_f32(-0.0625); hidden], &[hidden])?,
        ),
    })
}

fn oracle(input: &[f32], residual: &[f32], hidden: usize, width: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(input.len());
    for (row, residual_row) in input
        .chunks_exact(hidden)
        .zip(residual.chunks_exact(hidden))
    {
        let activated: Vec<f32> = (0..width)
            .map(|column| {
                let x = row[column % hidden] * 0.5 - row[(column + 1) % hidden] * 0.25 + 0.125;
                0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
            })
            .collect();
        result.extend(residual_row.iter().enumerate().map(|(column, value)| {
            value + activated[column % width] * 0.25
                - activated[(column + 1) % width] * 0.125
                - 0.0625
        }));
    }
    result
}

#[test]
#[ignore = "requires a reserved CUDA device; compares full outputs with scalar and existing paths"]
fn fused_block_mlp_matches_scalar_and_unfused_paths() -> Result<()> {
    let device = CudaDevice::new(0)?;
    for (hidden, width) in [(64, 128), (128, 192)] {
        let weights = weights(&device, hidden, width)?;
        for rows in [1, 31, 32, 33, 65] {
            let values = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75];
            let input: Vec<f32> = (0..rows * hidden).map(|index| values[index % 7]).collect();
            let residual: Vec<f32> = (0..rows * hidden)
                .map(|index| values[(index + 3) % 7])
                .collect();
            let first = project_with(&device, &weights.fc1.0, &weights.fc1.1, &input, rows, width)?;
            let activated = gelu(&device, false, &first, rows, width)?;
            let second = project_with(
                &device,
                &weights.fc2.0,
                &weights.fc2.1,
                &activated,
                rows,
                hidden,
            )?;
            let prior = add(&device, &residual, &second, rows, hidden)?;
            let actual = run(&device, &weights, &input, &residual, rows, hidden, width)?;
            let expected = oracle(&input, &residual, hidden, width);
            assert_eq!(actual.len(), expected.len());
            for (index, ((actual, prior), expected)) in
                actual.iter().zip(&prior).zip(&expected).enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    prior.to_bits(),
                    "unfused mismatch at {index}, rows {rows}"
                );
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= TOLERANCE,
                    "scalar mismatch at {index}, rows {rows}: {actual} versus {expected}"
                );
            }
        }
    }
    Ok(())
}
