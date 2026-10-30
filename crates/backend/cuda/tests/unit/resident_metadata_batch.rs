use super::{PromptOverrides, prompt_rotary_values, stage_metadata_batch};
use crate::resident::arena::batch_rows;
use crate::{
    constants::{METADATA_FIELDS, PREFILL_LANES},
    device::{CudaDevice, PinnedUpload, device_error},
};
use cutile::prelude::*;
use infer_core::Result;

const CAPACITY: usize = 4096;
const STATE_OFFSET: i32 = -1;

#[test]
fn media_rotary_coordinates_remain_absolute_across_prompt_chunks() -> Result<()> {
    let coordinates = [[7, 8, 9], [10, 11, 12]];
    let actual = prompt_rotary_values(4, coordinates.len(), 128, Some(&coordinates))?;
    assert_eq!(actual, [7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0]);
    Ok(())
}

#[test]
fn text_rotary_coordinates_use_the_chunk_position_and_validate_bounds() -> Result<()> {
    assert_eq!(
        prompt_rotary_values(3, 2, 128, None)?,
        [128, 128, 128, 129, 129, 129, 0, 0, 0]
    );
    assert!(prompt_rotary_values(1, 2, 128, None).is_err());
    assert!(prompt_rotary_values(2, 2, usize::MAX, None).is_err());
    Ok(())
}

fn allocate_owner(device: &CudaDevice) -> Result<Tensor<i32>> {
    api::zeros::<i32>(&[PREFILL_LANES * METADATA_FIELDS])
        .sync_on(&device.stream)
        .map_err(device_error)
}

fn stage(
    device: &CudaDevice,
    owner: &mut Tensor<i32>,
    upload: &mut (Vec<i32>, PinnedUpload<i32>),
    tokens: &[u32],
    position: usize,
    flags: &[bool],
) -> Result<()> {
    stage_metadata_batch(
        device,
        owner,
        (&mut upload.0, &mut upload.1),
        PREFILL_LANES,
        CAPACITY,
        tokens,
        position,
        STATE_OFFSET,
        PromptOverrides {
            lanes: flags,
            rotary: None,
        },
    )
}

fn assert_metadata_rows(
    actual: &[i32],
    position: usize,
    tokens: &[u32],
    flags: &[bool],
) -> Result<()> {
    for (lane, row) in actual
        .as_chunks::<METADATA_FIELDS>()
        .0
        .iter()
        .take(PREFILL_LANES)
        .enumerate()
    {
        let base = i32::try_from(position).map_err(device_error)?
            + i32::try_from(lane).map_err(device_error)?;
        let expected = if let Some(token) = tokens.get(lane) {
            [
                base,
                i32::try_from(*token).map_err(device_error)?,
                base + STATE_OFFSET,
                i32::from(flags[lane]),
            ]
        } else {
            // Inactive tail lanes must overwrite the preceding full chunk instead of
            // retaining its state position, token or media flag.
            [base, 0, STATE_OFFSET, 0]
        };
        assert_eq!(row.as_slice(), expected.as_slice(), "metadata lane {lane}");
    }
    Ok(())
}

/// GPU transfer, not just host-row construction: stage real chunks into one owner and read
/// the owner back. The two phase-2 chunks have no readback or drain between them, matching
/// `run32`'s no-metadata-readback path.
#[test]
#[ignore = "requires an NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn metadata_batch_full_tail_and_media_flags_round_trip() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let mut owner = allocate_owner(&device)?;
    let mut upload = (Vec::new(), PinnedUpload::chain(&device, owner.size())?);

    // Full media chunk: sparse external-embedding lanes and every metadata field distinct.
    let media_tokens: Vec<u32> = (1000..1032).collect();
    let media_flags: Vec<bool> = (0..PREFILL_LANES)
        .map(|lane| matches!(lane, 0 | 7 | 31))
        .collect();
    stage(
        &device,
        &mut owner,
        &mut upload,
        &media_tokens,
        100,
        &media_flags,
    )?;
    assert_metadata_rows(
        &device.read_borrowed(&owner)?,
        100,
        &media_tokens,
        &media_flags,
    )?;

    // Phase 2: a full text chunk followed immediately by a 5-token tail. Neither transfer is
    // drained before the next one; distinct pinned ring cells retain both sources.
    let full_tokens: Vec<u32> = (2000..2032).collect();
    let no_flags = vec![false; PREFILL_LANES];
    stage(
        &device,
        &mut owner,
        &mut upload,
        &full_tokens,
        132,
        &no_flags,
    )?;
    let mut first = allocate_owner(&device)?;
    device.copy_d2d(&mut first, &owner, owner.size())?;
    let tail_tokens: Vec<u32> = (3000..3005).collect();
    let mut tail_flags = vec![false; PREFILL_LANES];
    tail_flags[2] = true;
    tail_flags[4] = true;
    stage(
        &device,
        &mut owner,
        &mut upload,
        &tail_tokens,
        164,
        &tail_flags,
    )?;
    assert_metadata_rows(&device.read_borrowed(&first)?, 132, &full_tokens, &no_flags)?;
    assert_metadata_rows(
        &device.read_borrowed(&owner)?,
        164,
        &tail_tokens,
        &tail_flags,
    )
}

/// The row tensors own the backing allocation, not just a raw pointer: dropping the outer
/// owner handle must leave every lane readable and writable.
#[test]
#[ignore = "requires an NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn batch_rows_outlive_the_owner_handle() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let owner = api::zeros::<i32>(&[8])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut rows = batch_rows(&owner, 2)?;
    drop(owner);

    device.copy_host_slice(&mut rows[1], &[7, 8, 9, 10])?;
    assert_eq!(device.read_borrowed(&rows[1])?, vec![7, 8, 9, 10]);
    assert_eq!(device.read_borrowed(&rows[0])?, vec![0; 4]);
    Ok(())
}

/// Zero lanes is a normal API error, not an unchecked unsafe precondition.
#[test]
#[ignore = "requires an NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn batch_rows_rejects_zero_lanes() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let owner = allocate_owner(&device)?;
    let rows = batch_rows(&owner, 0);
    assert!(rows.is_err(), "zero-lane batch must be rejected");
    Ok(())
}
