//! Batched media feature rows against an independent host sum.
use crate::constants::METADATA_FIELDS;
use crate::device::CudaDevice;
use cutile::prelude::*;
use infer_cuda_kernels::tile::state::aux;
use std::error::Error as StdError;

/// Hidden width (two 256-column tiles) and prompt lanes.
const SHAPE: (usize, usize) = (512, 8);

#[test]
#[ignore = "requires a CUDA device; run inside safe-run"]
fn media_lanes_add_their_feature_row_and_text_lanes_pass_through() -> Result<(), Box<dyn StdError>>
{
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (dimension, lanes) = SHAPE;
    let value = |i: usize, scale: f32| f32::from(u16::try_from(i % 997).expect("bounded")) * scale;
    let hidden: Vec<f32> = (0..lanes * dimension)
        .map(|i| value(i, 0.25) - 60.0)
        .collect();
    let feature: Vec<f32> = (0..lanes * dimension)
        .map(|i| value(i * 7, 0.125) + 0.5)
        .collect();
    let mut metadata = vec![0_i32; lanes * METADATA_FIELDS];
    let media = |lane: usize| lane % 3 == 1;
    for lane in 0..lanes {
        metadata[lane * METADATA_FIELDS + 1] = i32::try_from(lane)?;
        metadata[lane * METADATA_FIELDS + 3] = i32::from(media(lane));
    }
    let actual = aux::media_feature_rows(
        api::zeros::<f32>(&[lanes, dimension]).partition([1, 256]),
        &device.upload(hidden.clone(), &[lanes, dimension])?,
        &device.upload(feature.clone(), &[lanes, dimension])?,
        &device.upload(metadata, &[lanes * METADATA_FIELDS])?,
    )
    .generics(vec![dimension.to_string(), METADATA_FIELDS.to_string()])
    .first()
    .unpartition()
    .sync_on(device.stream())?
    .to_host_vec()
    .sync_on(device.stream())?;
    for lane in 0..lanes {
        for column in 0..dimension {
            let index = lane * dimension + column;
            let expected = if media(lane) {
                hidden[index] + feature[index]
            } else {
                hidden[index]
            };
            assert_eq!(
                actual[index].to_bits(),
                expected.to_bits(),
                "lane {lane} column {column}"
            );
        }
    }
    assert!(
        (0..lanes).any(media) && !(0..lanes).all(media),
        "fixture must mix lanes"
    );
    Ok(())
}
