use super::{DeviceTaps, TOKEN_TILE};
use crate::device::CudaDevice;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::Result;
use infer_models::vision::PositionTaps;
use std::sync::Arc;

#[test]
#[ignore = "requires a reserved CUDA device"]
fn device_positions_match_scalar_interpolation_and_padding() -> Result<()> {
    const HIDDEN: usize = 64;
    const ROWS: usize = 16;
    let device = CudaDevice::new(0)?;
    let table: Vec<bf16> = (0..ROWS * HIDDEN)
        .map(|i| bf16::from_f32((i as f32 * 0.013).sin()))
        .collect();
    let device_table = device.upload(table.clone(), &[ROWS, HIDDEN])?;
    for tokens in [1_usize, 31, 32, 33, 96] {
        let padded = tokens.div_ceil(TOKEN_TILE) * TOKEN_TILE;
        let taps = PositionTaps {
            indices: (0..tokens)
                .map(|row| {
                    [
                        (row % ROWS) as u32,
                        ((row + 1) % ROWS) as u32,
                        ((row + 5) % ROWS) as u32,
                        ((row + 9) % ROWS) as u32,
                    ]
                })
                .collect(),
            weights: (0..tokens)
                .map(|row| {
                    let a = (row % 13) as f32 / 13.0;
                    [a, 0.3 - a, 0.2, 0.5]
                })
                .collect(),
        };
        let mut invalid = taps.clone();
        invalid.indices[0][3] = ROWS as u32;
        assert!(DeviceTaps::new(&device, &invalid, &device_table, ROWS, HIDDEN, padded).is_err());
        invalid = taps.clone();
        invalid.weights.clear();
        assert!(DeviceTaps::new(&device, &invalid, &device_table, ROWS, HIDDEN, padded).is_err());
        let mut input = vec![0.0_f32; padded * HIDDEN];
        for (i, value) in input.iter_mut().take(tokens * HIDDEN).enumerate() {
            *value = (i as f32 * 0.017).cos();
        }
        let uploaded = device.upload(input.clone(), &[padded, HIDDEN])?;
        let staged = DeviceTaps::new(&device, &taps, &device_table, ROWS, HIDDEN, padded)?;
        let mut result = api::zeros::<f32>(&[padded, HIDDEN])
            .sync_on(&device.stream)
            .map_err(crate::device::device_error)?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            staged.record(scope, &uploaded, &mut result)
        })
        .map_err(crate::device::device_error)?;
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(crate::device::device_error)?;
        let actual = device.read(&Arc::new(result))?;
        for patch in 0..tokens {
            for channel in 0..HIDDEN {
                let mut expected = 0.0_f32;
                for tap in 0..4 {
                    let index = taps.indices[patch][tap] as usize * HIDDEN + channel;
                    expected = taps.weights[patch][tap].mul_add(table[index].to_f32(), expected);
                }
                expected += input[patch * HIDDEN + channel];
                assert_eq!(
                    actual[patch * HIDDEN + channel].to_bits(),
                    expected.to_bits()
                );
            }
        }
    }
    Ok(())
}
