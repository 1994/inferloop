//! Complete tower comparison across padded rows and multi-frame inputs.
use super::{AttentionLayer, MergerLayer, MlpLayer, Prepared, VisionWeights};
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
        deepstack_layers: Vec::new(),
    }
}

fn prepared(device: &CudaDevice, rows: usize, columns: usize, scale: f32) -> Result<Prepared> {
    let mut matrix = vec![bf16::ZERO; rows * columns];
    for row in 0..rows {
        matrix[row * columns + row % columns] = bf16::from_f32(scale);
        matrix[row * columns + (row + 1) % columns] = bf16::from_f32(-0.125);
    }
    Ok((
        device.runtime().upload(matrix)?,
        device
            .runtime()
            .upload(vec![bf16::from_f32(0.0625); rows])?,
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
                let tensor = device
                    .runtime()
                    .upload(vec![bf16::from_f32(value); hidden])?;
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
        deepstack: Vec::new(),
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

fn merger(device: &CudaDevice, encoder: &ModalityEncoder, norm: usize) -> Result<MergerLayer> {
    let merged = encoder.hidden_size * encoder.spatial_merge_size * encoder.spatial_merge_size;
    Ok(MergerLayer {
        norm: (
            device.runtime().upload(vec![bf16::from_f32(0.75); norm])?,
            device.runtime().upload(vec![bf16::from_f32(-0.25); norm])?,
        ),
        fc1: prepared(device, merged, merged, 0.5)?,
        fc2: prepared(device, encoder.out_hidden_size, merged, 0.25)?,
    })
}

#[test]
#[ignore = "requires a reserved CUDA device; compares the one-graph encode with the host path"]
fn one_graph_encode_matches_blockwise_tower_and_host_mergers() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let mut encoder = encoder();
    // The last layer is tapped too: its state is also the final state.
    encoder.deepstack_layers = vec![0, 1];
    let merged = encoder.hidden_size * 4;
    let mut weights = weights(&device, &encoder)?;
    weights.merger = Some(merger(&device, &encoder, encoder.hidden_size)?);
    weights.deepstack = vec![
        merger(&device, &encoder, merged)?,
        merger(&device, &encoder, merged)?,
    ];
    assert!(device_tower::supports_mergers(&encoder)?);
    for grid in [(1, 2, 2), (1, 4, 10), (2, 4, 10), (1, 24, 24)] {
        let tokens = grid.0 * grid.1 * grid.2;
        let values = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75];
        let input: Vec<f32> = (0..tokens * encoder.hidden_size)
            .map(|index| values[index % values.len()])
            .collect();
        let mut state = input.clone();
        let mut taps = Vec::new();
        for layer in 0..encoder.depth {
            state = program::block(&device, &weights, &encoder, layer, &state, grid, tokens)?;
            taps.push(state.clone());
        }
        let expected = program::merger_with(&device, weights.merger()?, &encoder, &state, tokens)?;
        let features = taps
            .iter()
            .enumerate()
            .map(|(level, tap)| {
                program::deepstack_merger(
                    &device,
                    weights.deepstack_merger(level)?,
                    &encoder,
                    tap,
                    tokens,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let (actual, actual_features) =
            device_tower::encode(&device, &weights, &encoder, (&input, grid, tokens))?;
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&actual), bits(&expected), "grid {grid:?} merger");
        assert_eq!(actual_features.len(), features.len());
        for (level, (actual, expected)) in actual_features.iter().zip(&features).enumerate() {
            assert_eq!(bits(actual), bits(expected), "grid {grid:?} level {level}");
        }
    }
    Ok(())
}
