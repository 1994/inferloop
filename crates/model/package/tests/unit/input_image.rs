use super::*;
use std::env::var;
use std::fs::read;
use std::path::Path;

fn package_processor() -> Result<ImageProcessor> {
    let Ok(path) = var("INFER_VISION_PACKAGE") else {
        return Err(Error::invalid("package not configured"));
    };
    ImageProcessor::from_json(
        &read(Path::new(&path).join("preprocessor_config.json"))
            .map_err(|error| Error::invalid(error.to_string()))?,
    )
}

#[test]
fn smart_resize_matches_the_reference_grid() -> Result<()> {
    let processor = ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 65_536,
        max_pixels: 16_777_216,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: RESCALE,
    };
    // 60x100 is below the pixel budget, so it upscales to the closest 32-multiple grid.
    assert_eq!(processor.smart_resize(60, 100)?, (224, 352));
    // Already on the grid and inside the budget: unchanged.
    assert_eq!(processor.smart_resize(224, 352)?, (224, 352));
    assert!(processor.smart_resize(0, 10).is_err());
    Ok(())
}

#[test]
fn rgb8_resize_matches_torchvision_antialiased_bicubic() -> Result<()> {
    let source = [
        0, 30, 240, 60, 255, 90, 180, 10, 220, 40, 255, 75, 160, 0, 120,
    ];
    let image = RgbImage {
        height: 3,
        width: 5,
        pixels: source.into_iter().flat_map(|v| [v; RGB_CHANNELS]).collect(),
    };
    // Independent torchvision CPU uint8 fixtures, including boundary overshoot and downsampling.
    let up: [u8; 48] = [
        0, 0, 27, 211, 212, 48, 173, 255, 12, 25, 67, 166, 172, 96, 161, 210, 54, 101, 153, 71, 88,
        197, 136, 70, 128, 153, 163, 55, 64, 180, 106, 34, 224, 174, 96, 121, 105, 47, 75, 105,
        255, 184, 64, 152, 124, 0, 60, 138,
    ];
    let down: [u8; 6] = [50, 134, 156, 168, 97, 82];
    let vertical_up: [u8; 30] = [
        0, 17, 255, 46, 255, 13, 61, 193, 95, 212, 59, 154, 51, 200, 82, 133, 167, 27, 182, 43,
        226, 98, 126, 45, 100, 255, 66, 173, 0, 127,
    ];
    let horizontal_up: [u8; 24] = [
        0, 1, 39, 197, 200, 62, 169, 255, 84, 134, 176, 44, 61, 217, 122, 27, 255, 180, 73, 143,
        119, 3, 65, 129,
    ];
    let vertical_down: [u8; 10] = [25, 87, 154, 125, 177, 201, 118, 98, 83, 83];
    let horizontal_down: [u8; 9] = [12, 152, 175, 126, 100, 116, 183, 98, 68];
    for (height, width, expected) in [
        (6, 8, up.as_slice()),
        (2, 3, down.as_slice()),
        (6, 5, vertical_up.as_slice()),
        (3, 8, horizontal_up.as_slice()),
        (2, 5, vertical_down.as_slice()),
        (3, 3, horizontal_down.as_slice()),
    ] {
        let actual = ImageProcessor::resize(&image, height, width);
        let expected: Vec<f32> = expected
            .iter()
            .flat_map(|v| [f32::from(*v); RGB_CHANNELS])
            .collect();
        assert_eq!(actual, expected);
        let simd = image_resize::resize(&image.pixels, image.height, image.width, height, width)?;
        assert_eq!(
            simd.into_iter().map(f32::from).collect::<Vec<_>>(),
            expected
        );
    }
    Ok(())
}

#[test]
fn unchanged_rgb8_resize_preserves_every_value_bitwise() {
    for (height, width) in [(1, 256), (256, 1), (16, 16), (32, 64)] {
        let pixels: Vec<u8> = (0..height * width * RGB_CHANNELS)
            .map(|index| (index * 37 + 11).to_le_bytes()[0])
            .collect();
        let image = RgbImage {
            height,
            width,
            pixels,
        };
        let resized = ImageProcessor::resize(&image, height, width);
        assert_eq!(resized.len(), image.pixels.len());
        for (index, (&actual, &source)) in resized.iter().zip(&image.pixels).enumerate() {
            assert_eq!(
                actual.to_bits(),
                f32::from(source).to_bits(),
                "{height}x{width}, RGB element {index}"
            );
        }
    }
}

#[test]
fn horizontal_only_resize_keeps_unit_vertical_pass_zero_sign() {
    let image = RgbImage {
        height: 1,
        width: 5,
        pixels: [0, 1, 0, 0, 0]
            .into_iter()
            .flat_map(|value| [value; RGB_CHANNELS])
            .collect(),
    };
    let resized = ImageProcessor::resize(&image, 1, 8);
    for &value in &resized {
        if value == 0.0 {
            assert_eq!(value.to_bits(), 0.0f32.to_bits());
        }
    }
}

#[test]
fn simd_preprocessing_matches_scalar_patch_bits() -> Result<()> {
    let processor = ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 65_536,
        max_pixels: 16_777_216,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: RESCALE,
    };
    for (height, width) in [
        (60, 100),
        (63, 65),
        (257, 256),
        (256, 257),
        (333, 571),
        (320, 576),
    ] {
        let image = RgbImage {
            height,
            width,
            pixels: (0..height * width * RGB_CHANNELS)
                .map(|index| (index * 37 + 11).to_le_bytes()[0])
                .collect(),
        };
        let (target_height, target_width) = processor.smart_resize(height, width)?;
        let scalar = ImageProcessor::resize(&image, target_height, target_width);
        let frames = vec![scalar.as_slice(); processor.temporal_patch_size];
        let expected = patchify(
            &frames,
            PatchGrid {
                temporal: processor.temporal_patch_size,
                height: target_height / processor.patch_size,
                width: target_width / processor.patch_size,
                patch: processor.patch_size,
                merge: processor.merge_size,
                groups: 1,
            },
            &Normalization {
                mean: processor.mean,
                std: processor.std,
                rescale: processor.rescale,
            },
        )?;
        let actual = processor.preprocess(&image)?;
        assert_eq!(actual.pixels.len(), expected.len());
        for (&actual, &expected) in actual.pixels.iter().zip(&expected) {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{height}x{width}");
        }
    }
    Ok(())
}

#[test]
fn simd_resize_matches_scalar_across_random_shapes() -> Result<()> {
    fn next(seed: &mut u64) -> u8 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 32).to_le_bytes()[0]
    }
    let norm = Normalization {
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: RESCALE,
    };
    let mut seed = 456_321_u64;
    for _ in 0..100 {
        let height = 2 + usize::from(next(&mut seed) % 100);
        let width = 2 + usize::from(next(&mut seed) % 100);
        let target_height = 2 + usize::from(next(&mut seed) % 150);
        let target_width = 2 + usize::from(next(&mut seed) % 150);
        let image = RgbImage {
            height,
            width,
            pixels: (0..height * width * RGB_CHANNELS)
                .map(|_| next(&mut seed))
                .collect(),
        };
        let scalar = ImageProcessor::resize(&image, target_height, target_width);
        let simd = image_resize::resize(&image.pixels, height, width, target_height, target_width)?;
        assert_eq!(simd.len(), scalar.len());
        for (index, (&actual, &expected)) in simd.iter().zip(&scalar).enumerate() {
            let channel = index % RGB_CHANNELS;
            assert_eq!(
                norm.apply(f32::from(actual), channel).to_bits(),
                norm.apply(expected, channel).to_bits(),
                "{height}x{width} -> {target_height}x{target_width} element {index}"
            );
        }
    }
    Ok(())
}

#[test]
fn unchanged_images_preserve_patch_order_and_normalization() -> Result<()> {
    for (height, width, temporal) in [(32, 32, 1), (32, 64, 2), (64, 32, 3)] {
        for rescale in [RESCALE, 0.3] {
            let processor = ImageProcessor {
                patch_size: 16,
                temporal_patch_size: temporal,
                merge_size: 2,
                min_pixels: 1,
                max_pixels: height * width,
                mean: [0.0, 0.22, 0.5],
                std: [1.0, 0.25, 0.5],
                rescale,
            };
            let image = RgbImage {
                height,
                width,
                pixels: (0..height * width * RGB_CHANNELS)
                    .map(|index| (index * 37 + 11).to_le_bytes()[0])
                    .collect(),
            };
            let output = processor.preprocess(&image)?;
            assert_eq!(output.grid, (1, height / 16, width / 16));
            assert_eq!(
                output.pixels.len(),
                height * width * RGB_CHANNELS * temporal
            );
            assert_image_patch_values(&processor, &image, &output);
        }
    }
    Ok(())
}

#[test]
fn shared_patches_match_owned_processing_for_identity_and_resized_images() -> Result<()> {
    for (height, width, min_pixels, rescale, mean, std) in [
        (32, 64, 1, RESCALE, [0.0, 0.22, 0.5], [1.0, 0.25, 0.5]),
        (257, 256, 65_536, RESCALE, [0.5; 3], [0.5; 3]),
        (60, 100, 65_536, 0.3, [0.0, 0.22, 0.5], [1.0, 0.25, 0.5]),
    ] {
        let processor = ImageProcessor {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            min_pixels,
            max_pixels: 16_777_216,
            mean,
            std,
            rescale,
        };
        let image = RgbImage {
            height,
            width,
            pixels: (0..height * width * RGB_CHANNELS)
                .map(|index| (index * 37 + 11).to_le_bytes()[0])
                .collect(),
        };
        let owned = processor.preprocess(&image)?;
        let shared = processor.preprocess_shared(&image)?;
        assert_eq!(shared.grid, owned.grid);
        assert_eq!(shared.pixels.len(), owned.pixels.len());
        for (&actual, &expected) in shared.pixels.iter().zip(&owned.pixels) {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{height}x{width}");
        }
        let malformed = RgbImage {
            height,
            width,
            pixels: Vec::new(),
        };
        assert_eq!(
            processor.preprocess_shared(&malformed).unwrap_err().code,
            processor.preprocess(&malformed).unwrap_err().code
        );
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "The scalar reference preserves the normalization rounding order"
)]
fn assert_image_patch_values(
    processor: &ImageProcessor,
    image: &RgbImage,
    output: &PreprocessedImage,
) {
    let patch_area = processor.patch_size * processor.patch_size;
    let channel_width = processor.temporal_patch_size * patch_area;
    let patch_width = RGB_CHANNELS * channel_width;
    let merge_area = processor.merge_size * processor.merge_size;
    let blocks_per_row = output.grid.2 / processor.merge_size;
    for (index, &actual) in output.pixels.iter().enumerate() {
        let patch = index / patch_width;
        let channel = (index % patch_width) / channel_width;
        let spatial = index % patch_area;
        let block = patch / merge_area;
        let local = patch % merge_area;
        let row = (block / blocks_per_row * processor.merge_size + local / processor.merge_size)
            * processor.patch_size
            + spatial / processor.patch_size;
        let col = (block % blocks_per_row * processor.merge_size + local % processor.merge_size)
            * processor.patch_size
            + spatial % processor.patch_size;
        let value = f32::from(image.pixels[(row * image.width + col) * RGB_CHANNELS + channel]);
        let expected = if processor.rescale == RESCALE {
            (value - processor.mean[channel] * 255.0) / (processor.std[channel] * 255.0)
        } else {
            value.mul_add(processor.rescale, -processor.mean[channel]) / processor.std[channel]
        };
        assert_eq!(actual.to_bits(), expected.to_bits(), "patch value {index}");
    }
}

#[test]
fn configuration_comes_from_the_package() {
    let Ok(processor) = package_processor() else {
        return;
    };
    assert_eq!(processor.patch_size, 16);
    assert_eq!(processor.merge_size, 2);
    assert_eq!(processor.temporal_patch_size, 2);
    assert_eq!(processor.min_pixels, 65_536);
    assert_eq!(processor.mean, [0.5; 3]);
    assert_eq!(processor.patch_width(), 1536);
}

/// The pipeline must reproduce the reference processor's pixels.
///
/// Skipped unless both the package and the image golden are provided.
#[test]
fn preprocessing_matches_the_reference_processor() -> Result<()> {
    // RGB8 resize must match; only final F32 normalization rounding may differ.
    const RESAMPLE_TOLERANCE: f32 = 1e-6;
    let (Ok(package), Ok(golden)) = (
        var("INFER_VISION_PACKAGE"),
        var("INFER_VISION_IMAGE_GOLDEN"),
    ) else {
        return Ok(());
    };
    let processor = ImageProcessor::from_json(
        &read(Path::new(&package).join("preprocessor_config.json"))
            .map_err(|error| Error::invalid(error.to_string()))?,
    )?;
    let mut file = crate::SafetensorsFile::open(&golden)?;
    let expected = file.read_f32("pixel_values", 1 << 30)?;
    let grid = file.read_bytes("image_grid_thw", 1 << 20)?;
    let dimensions = file.read_bytes("image", 1 << 20)?;
    let pixels = file.read_bytes("image_pixels", 1 << 30)?;
    let grid: Vec<i64> = grid
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_le_bytes(*chunk))
        .collect();
    let dimensions: Vec<i64> = dimensions
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_le_bytes(*chunk))
        .collect();
    let image = RgbImage {
        height: usize::try_from(dimensions[0]).map_err(|_| Error::invalid("golden height"))?,
        width: usize::try_from(dimensions[1]).map_err(|_| Error::invalid("golden width"))?,
        pixels,
    };
    let processed = processor.preprocess(&image)?;
    let expected_grid = (
        usize::try_from(grid[0]).map_err(|_| Error::invalid("golden grid"))?,
        usize::try_from(grid[1]).map_err(|_| Error::invalid("golden grid"))?,
        usize::try_from(grid[2]).map_err(|_| Error::invalid("golden grid"))?,
    );
    assert_eq!(processed.grid, expected_grid);
    assert_eq!(processed.pixels.len(), expected.data.len());
    let mut max_abs = 0f32;
    let mut differing = 0usize;
    for (value, gold) in processed.pixels.iter().zip(expected.data.iter()) {
        let difference = (value - gold).abs();
        if difference > 1e-6 {
            differing += 1;
        }
        max_abs = max_abs.max(difference);
    }
    assert!(
        max_abs <= RESAMPLE_TOLERANCE,
        "preprocessing deviates by {max_abs} across {differing} values"
    );
    Ok(())
}

#[test]
fn fused_normalization_preserves_reference_bits() {
    let norm = Normalization {
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: RESCALE,
    };
    for (value, bits) in [
        (223.0f32, 0x3f3f_bfc0u32),
        (192.0, 0x3f01_8182u32),
        (255.0, 0x3f80_0000u32),
        (0.0, 0xbf80_0000u32),
    ] {
        assert_eq!(norm.apply(value, 0).to_bits(), bits, "value {value}");
    }
}

#[test]
fn custom_rescale_preserves_original_float_semantics() {
    for rescale in [0.3f32, 0.7, 0.0, 1.0e-30] {
        let mean = [0.11f32, 0.22, 0.33];
        let std = [0.5f32, 0.25, 0.125];
        let norm = Normalization { mean, std, rescale };
        for channel in 0..3 {
            for value in [0.0f32, 1.0, 64.0, 128.0, 255.0] {
                let expected = f64::from(value)
                    .mul_add(f64::from(rescale), -f64::from(mean[channel]))
                    / f64::from(std[channel]);
                let actual = f64::from(norm.apply(value, channel));
                assert!(actual.is_finite(), "rescale {rescale} value {value}");
                assert!(
                    (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                    "rescale {rescale} channel {channel} value {value}: {actual} vs {expected}"
                );
            }
        }
    }
}
