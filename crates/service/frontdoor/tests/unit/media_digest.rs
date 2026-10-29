use super::{BATCH_IMAGES, hashes};
use infer_models::PromptImage;

fn image(bits: &[u32], grid: (usize, usize, usize)) -> PromptImage {
    PromptImage {
        pixels: bits.iter().copied().map(f32::from_bits).collect(),
        grid,
    }
}

/// Independent byte-oriented FNV-1a reference for the historical digest contract.
fn reference(image: &PromptImage) -> u64 {
    let mut bytes = Vec::new();
    for value in &image.pixels {
        bytes.extend(value.to_bits().to_le_bytes());
    }
    for axis in <[usize; 3]>::from(image.grid) {
        bytes.extend(u64::try_from(axis).unwrap_or(u64::MAX).to_le_bytes());
    }
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn golden_digests_preserve_float_bits_grid_and_image_order() {
    let images = [
        image(&[0, 0x8000_0000, 0x3f80_0000], (1, 16, 16)),
        image(
            &[0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0x7fa1_2345, 1],
            (7, 1, 0),
        ),
        image(&[], (0, 0, 0)),
        image(&[0xffff_ffff, 0x3eaa_aaab, 0x1234_5678, 0], (65535, 7, 19)),
    ];
    // Generated independently from the specified little-endian byte stream.
    let expected = [
        0xd81f_9d28_d4bd_49d9,
        0x7949_5669_f314_3ebe,
        0x81d2_3fd7_003c_2305,
        0xc138_c60e_ce82_4ce6,
    ];
    for count in 0..=BATCH_IMAGES {
        let result = hashes(&images[..count]).expect("bounded batch");
        assert_eq!(&result[..count], &expected[..count]);
    }
}

#[test]
fn batches_match_independent_reference_for_unequal_lengths_and_tails() {
    for length in 0..40_u32 {
        let images: Vec<_> = (0..BATCH_IMAGES)
            .map(|lane| {
                let lane = u32::try_from(lane).expect("bounded lane");
                let bits: Vec<_> = (0..length + lane)
                    .map(|index| index.wrapping_mul(2_654_435_761) ^ lane)
                    .collect();
                image(&bits, (1, usize::try_from(length).unwrap(), 16))
            })
            .collect();
        for count in 1..=BATCH_IMAGES {
            let result = hashes(&images[..count]).expect("bounded batch");
            for (actual, image) in result.iter().zip(&images).take(count) {
                assert_eq!(*actual, reference(image));
            }
        }
    }
    let too_many = vec![image(&[1], (1, 1, 1)); BATCH_IMAGES + 1];
    assert!(hashes(&too_many).is_err());
}
