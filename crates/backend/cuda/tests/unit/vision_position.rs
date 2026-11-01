use super::{DeviceTaps, TOKEN_TILE};
use crate::device::{CudaDevice, device_error};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::Result;
use infer_cuda_kernels::tile::runtime::Graph;
use infer_models::vision::PositionTaps;
use std::sync::Arc;

const HIDDEN: usize = 64;
const ROWS: usize = 16;

fn bounded_float(value: usize) -> Result<f32> {
    Ok(f32::from(u16::try_from(value).map_err(device_error)?))
}

fn fixture_taps(tokens: usize) -> Result<PositionTaps> {
    let mut indices = Vec::with_capacity(tokens);
    let mut weights = Vec::with_capacity(tokens);
    for row in 0..tokens {
        let mut item = [0_u32; 4];
        for (index, offset) in item.iter_mut().zip([0, 1, 5, 9]) {
            *index = u32::try_from((row + offset) % ROWS).map_err(device_error)?;
        }
        indices.push(item);
        let a = bounded_float(row % 13)? / 13.0;
        weights.push([a, 0.3 - a, 0.2, 0.5]);
    }
    Ok(PositionTaps { indices, weights })
}

#[test]
#[ignore = "requires a reserved CUDA device"]
fn device_positions_match_scalar_interpolation_and_padding() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let table: Vec<bf16> = (0..ROWS * HIDDEN)
        .map(|i| bounded_float(i).map(|value| bf16::from_f32((value * 0.013).sin())))
        .collect::<Result<_>>()?;
    let device_table = device.upload(table.clone(), &[ROWS, HIDDEN])?;
    for tokens in [1_usize, 31, 32, 33, 96] {
        let padded = tokens.div_ceil(TOKEN_TILE) * TOKEN_TILE;
        let taps = fixture_taps(tokens)?;
        let mut invalid = taps.clone();
        invalid.indices[0][3] = u32::try_from(ROWS).map_err(device_error)?;
        assert!(
            DeviceTaps::new(
                &device,
                &invalid.indices,
                &invalid.weights,
                &device_table,
                ROWS,
                HIDDEN,
                padded,
            )
            .is_err()
        );
        invalid = taps.clone();
        invalid.weights.clear();
        assert!(
            DeviceTaps::new(
                &device,
                &invalid.indices,
                &invalid.weights,
                &device_table,
                ROWS,
                HIDDEN,
                padded,
            )
            .is_err()
        );
        let mut input = vec![0.0_f32; padded * HIDDEN];
        for (i, value) in input.iter_mut().take(tokens * HIDDEN).enumerate() {
            *value = (bounded_float(i)? * 0.017).cos();
        }
        let uploaded = device.upload(input.clone(), &[padded, HIDDEN])?;
        let staged = DeviceTaps::new(
            &device,
            &taps.indices,
            &taps.weights,
            &device_table,
            ROWS,
            HIDDEN,
            padded,
        )?;
        let mut result = api::zeros::<f32>(&[padded, HIDDEN])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let graph = Graph::scope(&device.stream, |scope| {
            staged.record(scope, &uploaded, &mut result)
        })
        .map_err(device_error)?;
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let actual = device.read(&Arc::new(result))?;
        for patch in 0..tokens {
            for channel in 0..HIDDEN {
                let mut expected = 0.0_f32;
                for tap in 0..4 {
                    let index = usize::try_from(taps.indices[patch][tap]).map_err(device_error)?
                        * HIDDEN
                        + channel;
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
