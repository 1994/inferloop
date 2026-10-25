//! Batched target prompt embedding against the per-lane kernel it replaces.
use super::attention;
use crate::constants::METADATA_FIELDS;
use crate::device::CudaDevice;
use cutile::half::bf16;
use cutile::prelude::*;
use std::error::Error as StdError;

/// Vocabulary rows, hidden width (two 256-column tiles) and prompt lanes.
const SHAPE: (usize, usize, usize) = (64, 512, 8);

#[test]
#[ignore = "requires a CUDA device; run inside safe-run"]
fn batched_prompt_embedding_matches_per_lane_rows() -> Result<(), Box<dyn StdError>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (vocabulary, dimension, lanes) = SHAPE;
    let weight: Vec<_> = (0..vocabulary * dimension)
        .map(|i| {
            bf16::from_f32(
                (f32::from(u8::try_from(i * 37 % 101).expect("bounded fixture")) - 50.0) / 7.0,
            )
        })
        .collect();
    let external: Vec<_> = (0..lanes * dimension)
        .map(|i| f32::from(u16::try_from(i).expect("bounded fixture")) / 3.0 + 1000.0)
        .collect();
    let mut metadata = vec![0_i32; lanes * METADATA_FIELDS];
    for lane in 0..lanes {
        metadata[lane * METADATA_FIELDS + 1] = i32::try_from(lane * 7 % vocabulary)?;
        metadata[lane * METADATA_FIELDS + 3] = i32::from(lane % 3 == 1);
    }
    let weight = device.upload(weight, &[vocabulary, dimension])?;
    let all_external = device.upload(external.clone(), &[lanes, dimension])?;
    let all_metadata = device.upload(metadata.clone(), &[lanes * METADATA_FIELDS])?;
    let batched = attention::embedding_rows(
        api::zeros::<f32>(&[lanes, dimension]).partition([1, 256]),
        &weight,
        &all_external,
        &all_metadata,
    )
    .generics(vec![dimension.to_string(), METADATA_FIELDS.to_string()])
    .first()
    .unpartition()
    .sync_on(&device.stream)?
    .to_host_vec()
    .sync_on(&device.stream)?;
    for lane in 0..lanes {
        let row = &external[lane * dimension..(lane + 1) * dimension];
        let fields = &metadata[lane * METADATA_FIELDS..(lane + 1) * METADATA_FIELDS];
        let expected = attention::embedding(
            api::zeros::<f32>(&[dimension]).partition([256]),
            &weight,
            &device.upload(row.to_vec(), &[dimension])?,
            &device.upload(fields.to_vec(), &[METADATA_FIELDS])?,
        )
        .generics(vec![dimension.to_string(), "1".into()])
        .first()
        .unpartition()
        .sync_on(&device.stream)?
        .to_host_vec()
        .sync_on(&device.stream)?;
        let actual = &batched[lane * dimension..(lane + 1) * dimension];
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(actual), bits(&expected), "lane {lane}");
    }
    assert!(
        batched.iter().any(|&value| value >= 1000.0),
        "fixture must cover overrides"
    );
    Ok(())
}
