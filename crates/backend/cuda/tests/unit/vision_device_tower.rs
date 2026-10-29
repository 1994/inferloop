//! Complete tower comparison across padded rows and multi-frame inputs.
use super::{AttentionLayer, MlpLayer, Prepared, VisionWeights};
use crate::device::CudaDevice;
use crate::vision::{device_tower, program};
use cutile::half::bf16;
use infer_core::Result;
use infer_spi::{HiddenActivation, ModalityEncoder};
use std::collections::BTreeMap;

fn encoder() -> ModalityEncoder {
    ModalityEncoder {
        prefix: "model.visual.".into(),
        depth: 2,
        hidden_size: 64,
        intermediate_size: 128,
        heads: 1,
        in_channels: 3,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        out_hidden_size: 64,
        position_embeddings: 2304,
        rope_theta: 10_000.0,
        norm_epsilon: 1e-6,
        hidden_activation: HiddenActivation::GeluTanh,
    }
}

fn prepared(device: &CudaDevice, rows: usize, columns: usize, scale: f32) -> Result<Prepared> {
    let mut matrix = vec![bf16::ZERO; rows * columns];
    for row in 0..rows {
        matrix[row * columns + row % columns] = bf16::from_f32(scale);
        matrix[row * columns + (row + 1) % columns] = bf16::from_f32(-0.125);
    }
    Ok((
        device.upload(matrix, &[rows, columns])?,
        device.upload(vec![bf16::from_f32(0.0625); rows], &[rows])?,
    ))
}

fn weights(device: &CudaDevice, encoder: &ModalityEncoder) -> Result<VisionWeights> {
    let mut tensors = BTreeMap::new();
    let mut attention = Vec::new();
    let mut mlp = Vec::new();
    let hidden = encoder.hidden_size;
    for layer in 0..encoder.depth {
        for norm in ["norm1", "norm2"] {
            for (name, value) in [("weight", 0.5), ("bias", 0.125)] {
                let tensor = device.upload(vec![bf16::from_f32(value); hidden], &[hidden])?;
                tensors.insert(format!("blocks.{layer}.{norm}.{name}"), tensor);
            }
        }
        attention.push(AttentionLayer {
            q: prepared(device, hidden, hidden, 0.5)?,
            k: prepared(device, hidden, hidden, 0.25)?,
            v: prepared(device, hidden, hidden, 0.75)?,
            proj: prepared(device, hidden, hidden, 0.25)?,
        });
        mlp.push(MlpLayer {
            fc1: prepared(device, encoder.intermediate_size, hidden, 0.5)?,
            fc2: prepared(device, hidden, encoder.intermediate_size, 0.25)?,
        });
    }
    Ok(VisionWeights {
        tensors,
        attention,
        mlp,
        merger: None,
        positions: Vec::new(),
    })
}

#[test]
fn tower_selection_requires_unpadded_norm_and_noncompact_attention() -> Result<()> {
    let mut encoder = encoder();
    assert!(device_tower::supports(&encoder)?);
    encoder.hidden_size = 1152;
    encoder.heads = 16;
    assert!(!device_tower::supports(&encoder)?);
    encoder.depth = 0;
    assert!(!device_tower::supports(&encoder)?);
    Ok(())
}

#[test]
#[ignore = "requires a reserved CUDA device; compares every output with the prior tower"]
fn resident_tower_matches_prior_blocks() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let encoder = encoder();
    let weights = weights(&device, &encoder)?;
    for grid in [(1, 2, 2), (1, 4, 10), (1, 8, 8), (2, 4, 10), (1, 24, 24)] {
        let tokens = grid.0 * grid.1 * grid.2;
        let values = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75];
        let input: Vec<f32> = (0..tokens * encoder.hidden_size)
            .map(|index| values[index % values.len()])
            .collect();
        let mut expected = input.clone();
        for layer in 0..encoder.depth {
            expected = program::block(&device, &weights, &encoder, layer, &expected, grid, tokens)?;
        }
        let actual = device_tower::run(&device, &weights, &encoder, &input, grid, tokens)?;
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "grid {grid:?} at {index}"
            );
        }
    }
    Ok(())
}
