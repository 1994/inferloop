//! Compare the device-resident merger with the former path and a scalar reference.
use super::{MergerLayer, VisionWeights};
use crate::device::CudaDevice;
use crate::vision::{device_merger, program};
use cutile::half::bf16;
use infer_core::Result;
use infer_spi::{HiddenActivation, ModalityEncoder};
use std::collections::BTreeMap;
use std::f64::consts::PI;

const HIDDEN: usize = 64;
const UNIT: usize = 4;
const MERGED: usize = HIDDEN * UNIT;
const OUTPUT: usize = HIDDEN;
const EPSILON: f32 = 1e-6;
const TOLERANCE: f32 = 1e-4;

fn encoder() -> ModalityEncoder {
    ModalityEncoder {
        prefix: "model.visual.".into(),
        depth: 1,
        hidden_size: HIDDEN,
        intermediate_size: 128,
        heads: 1,
        in_channels: 3,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        out_hidden_size: OUTPUT,
        position_embeddings: 16,
        rope_theta: 10_000.0,
        norm_epsilon: EPSILON,
        hidden_activation: HiddenActivation::GeluTanh,
    }
}

fn weights(device: &CudaDevice) -> Result<VisionWeights> {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "merger.norm.weight".to_owned(),
        device.upload(vec![bf16::from_f32(0.5); HIDDEN], &[HIDDEN])?,
    );
    tensors.insert(
        "merger.norm.bias".to_owned(),
        device.upload(vec![bf16::from_f32(0.125); HIDDEN], &[HIDDEN])?,
    );
    let mut fc1 = vec![bf16::ZERO; MERGED * MERGED];
    let mut fc2 = vec![bf16::ZERO; OUTPUT * MERGED];
    for index in 0..MERGED {
        fc1[index * MERGED + index] = bf16::from_f32(0.5);
    }
    for index in 0..OUTPUT {
        for (patch, coefficient) in [0.5, -0.25, 0.125, 0.0625].into_iter().enumerate() {
            fc2[index * MERGED + patch * HIDDEN + index] = bf16::from_f32(coefficient);
        }
    }
    Ok(VisionWeights {
        tensors,
        attention: Vec::new(),
        mlp: Vec::new(),
        merger: Some(MergerLayer {
            fc1: (
                device.upload(fc1, &[MERGED, MERGED])?,
                device.upload(vec![bf16::from_f32(0.0625); MERGED], &[MERGED])?,
            ),
            fc2: (
                device.upload(fc2, &[OUTPUT, MERGED])?,
                device.upload(vec![bf16::from_f32(-0.125); OUTPUT], &[OUTPUT])?,
            ),
        }),
        positions: Vec::new(),
    })
}

fn prior(device: &CudaDevice, weights: &VisionWeights, input: &[f32]) -> Result<Vec<f32>> {
    let tokens = input.len() / HIDDEN;
    let rows = tokens / UNIT;
    let norm = program::layernorm(
        device,
        weights,
        &encoder(),
        "merger.norm.weight",
        "merger.norm.bias",
        input,
        tokens,
    )?;
    let layer = weights.merger()?;
    let fc1 = program::project_with(device, &layer.fc1.0, &layer.fc1.1, &norm, rows, MERGED)?;
    let active = program::gelu(device, true, &fc1, rows, MERGED)?;
    program::project_with(device, &layer.fc2.0, &layer.fc2.1, &active, rows, OUTPUT)
}

fn erf(value: f64) -> f64 {
    let mut term = value;
    let mut sum = term;
    for index in 1..20_u32 {
        term *= -(value * value) / f64::from(index);
        sum += term / f64::from(2 * index + 1);
    }
    2.0 * sum / PI.sqrt()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The bounded reference inputs and coefficients produce finite f32-range outputs"
)]
fn scalar(input: &[f32]) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(input.len());
    for row in input.as_chunks::<HIDDEN>().0 {
        let mean = row.iter().map(|value| f64::from(*value)).sum::<f64>() / 64.0;
        let variance = row
            .iter()
            .map(|value| (f64::from(*value) - mean).powi(2))
            .sum::<f64>()
            / 64.0;
        for value in row {
            let norm = (f64::from(*value) - mean) / (variance + f64::from(EPSILON)).sqrt();
            let projected = norm.mul_add(0.5, 0.125).mul_add(0.5, 0.0625);
            normalized.push(0.5 * projected * (1.0 + erf(projected / 2.0_f64.sqrt())));
        }
    }
    let mut output = Vec::with_capacity(input.len() / UNIT);
    for group in normalized.as_chunks::<MERGED>().0 {
        for channel in 0..OUTPUT {
            let value = group[HIDDEN + channel].mul_add(-0.25, group[channel] * 0.5);
            let value = group[2 * HIDDEN + channel].mul_add(0.125, value);
            let value = group[3 * HIDDEN + channel].mul_add(0.0625, value) - 0.125;
            output.push(value as f32);
        }
    }
    output
}

#[test]
fn resident_merger_requires_compatible_row_and_projection_tiles() -> Result<()> {
    let mut geometry = encoder();
    assert!(device_merger::supports(&geometry)?);
    geometry.hidden_size = 96;
    assert!(!device_merger::supports(&geometry)?);
    geometry.hidden_size = HIDDEN;
    geometry.out_hidden_size = 65;
    assert!(!device_merger::supports(&geometry)?);
    Ok(())
}

#[test]
#[ignore = "requires a reserved CUDA device"]
fn resident_merger_matches_prior_and_scalar_across_padding() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let weights = weights(&device)?;
    assert!(device_merger::run(&device, &weights, &encoder(), &[], 0).is_err());
    assert!(device_merger::run(&device, &weights, &encoder(), &[0.0; HIDDEN], 1).is_err());
    let values = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75];
    for tokens in [4, 124, 128, 132, 256] {
        let input: Vec<f32> = (0..tokens * HIDDEN)
            .map(|index| values[index % values.len()])
            .collect();
        let expected = prior(&device, &weights, &input)?;
        let reference = scalar(&input);
        let actual = device_merger::run(&device, &weights, &encoder(), &input, tokens)?;
        let routed = program::merger(&device, &weights, &encoder(), &input, tokens)?;
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, routed);
        for (index, ((actual, expected), reference)) in
            actual.iter().zip(&expected).zip(&reference).enumerate()
        {
            assert_eq!(actual.to_bits(), expected.to_bits(), "prior at {index}");
            assert!(
                actual.is_finite() && (actual - reference).abs() <= TOLERANCE,
                "scalar at {index}: {actual} versus {reference}"
            );
        }
    }
    Ok(())
}
