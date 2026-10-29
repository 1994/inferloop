use super::*;
use std::env::var;
use std::fs::read;
use std::path::Path;
use std::result::Result as StdResult;

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
fn rgb8_resize_matches_torchvision_antialiased_bicubic() {
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
    let unchanged = ImageProcessor::resize(&image, image.height, image.width);
    assert_eq!(
        unchanged,
        image
            .pixels
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>()
    );
    for (height, width, expected) in [(6, 8, up.as_slice()), (2, 3, down.as_slice())] {
        let actual = ImageProcessor::resize(&image, height, width);
        let expected: Vec<f32> = expected
            .iter()
            .flat_map(|v| [f32::from(*v); RGB_CHANNELS])
            .collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn resized_patches_preserve_scalar_bits_and_zero_sign() -> Result<()> {
    let mut saw_negative_zero = false;
    for mean in [[0.5; RGB_CHANNELS], [0.0; RGB_CHANNELS]] {
        let processor = ImageProcessor {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            min_pixels: 1024,
            max_pixels: 16_777_216,
            mean,
            std: [0.5; RGB_CHANNELS],
            rescale: RESCALE,
        };
        for (height, width) in [(3, 5), (33, 65), (68, 57), (129, 191), (257, 257)] {
            let mut state = 0x7bee_cafeu32;
            let pixels = (0..height * width * RGB_CHANNELS)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state.to_be_bytes()[0]
                })
                .collect();
            let image = RgbImage {
                height,
                width,
                pixels,
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
            assert_eq!(actual.grid, (1, target_height / 16, target_width / 16));
            assert_eq!(actual.pixels.len(), expected.len());
            for (actual, expected) in actual.pixels.iter().zip(&expected) {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{height}x{width}");
                saw_negative_zero |= expected.to_bits() == (-0.0f32).to_bits();
            }
        }
    }
    assert!(
        saw_negative_zero,
        "zero-mean fixture must exercise signed zero"
    );
    Ok(())
}

#[test]
fn fast_resize_matches_independent_torchvision_pixels_after_normalization() -> Result<()> {
    let source = [
        0, 30, 240, 60, 255, 90, 180, 10, 220, 40, 255, 75, 160, 0, 120,
    ];
    let pixels: Vec<u8> = source.into_iter().flat_map(|v| [v; RGB_CHANNELS]).collect();
    let norm = Normalization {
        mean: [0.5; RGB_CHANNELS],
        std: [0.5; RGB_CHANNELS],
        rescale: RESCALE,
    };
    let up: [u8; 48] = [
        0, 0, 27, 211, 212, 48, 173, 255, 12, 25, 67, 166, 172, 96, 161, 210, 54, 101, 153, 71, 88,
        197, 136, 70, 128, 153, 163, 55, 64, 180, 106, 34, 224, 174, 96, 121, 105, 47, 75, 105,
        255, 184, 64, 152, 124, 0, 60, 138,
    ];
    let down: [u8; 6] = [50, 134, 156, 168, 97, 82];
    for (height, width, expected) in [(6, 8, up.as_slice()), (2, 3, down.as_slice())] {
        let actual = image_resize::resize(&pixels, 3, 5, height, width)?;
        assert_eq!(actual.len(), expected.len() * RGB_CHANNELS);
        for (index, (value, expected)) in actual
            .iter()
            .zip(expected.iter().flat_map(|v| [v; 3]))
            .enumerate()
        {
            assert_eq!(
                norm.apply(f32::from(*value), index % RGB_CHANNELS)
                    .to_bits(),
                norm.apply(f32::from(*expected), index % RGB_CHANNELS)
                    .to_bits()
            );
        }
    }
    Ok(())
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

#[expect(
    clippy::suboptimal_flops,
    reason = "The scalar reference preserves the processor's specified rounding order"
)]
fn scalar_normalization(processor: &ImageProcessor, value: f32, channel: usize) -> f32 {
    if processor.rescale == RESCALE {
        (value - processor.mean[channel] * 255.0) / (processor.std[channel] * 255.0)
    } else {
        value.mul_add(processor.rescale, -processor.mean[channel]) / processor.std[channel]
    }
}

#[test]
fn unresized_rgb8_patches_match_scalar_layout_and_normalization() -> Result<()> {
    use infer_ir::ImageResizePolicy;
    for (height, width, patch, temporal, merge) in [
        (8, 12, 2, 2, 2),
        (16, 16, 16, 2, 1),
        (32, 32, 16, 1, 2),
        (32, 32, 16, 2, 2),
        (64, 64, 16, 2, 2),
        (48, 48, 24, 2, 2),
        (96, 96, 24, 2, 2),
        (64, 64, 32, 2, 2),
    ] {
        let pixels: Vec<u8> = (0..height * width * RGB_CHANNELS)
            .map(|index| u8::try_from((index * 37 + 11) % 256))
            .collect::<StdResult<_, _>>()
            .map_err(|_| Error::invalid("fixture byte"))?;
        for rescale in [RESCALE, 0.3, 0.0, 1.0e-30] {
            let processor = ImageProcessor {
                patch_size: patch,
                temporal_patch_size: temporal,
                merge_size: merge,
                min_pixels: 1,
                max_pixels: height * width,
                mean: [0.11, 0.22, 0.33],
                std: [0.5, 0.25, 0.125],
                rescale,
            };
            let disabled = processor.preprocess_rgb_with_resize(
                height,
                width,
                &pixels,
                ImageResizePolicy::Disabled,
            )?;
            let smart = processor.preprocess_rgb_with_resize(
                height,
                width,
                &pixels,
                ImageResizePolicy::Smart,
            )?;
            assert_eq!(disabled.grid, (1, height / patch, width / patch));
            assert_eq!(disabled.grid, smart.grid);
            assert_eq!(disabled.pixels.len(), smart.pixels.len());
            assert_scalar_patch_values(&processor, &pixels, width, &disabled);
            for (actual, expected) in smart.pixels.iter().zip(&disabled.pixels) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
    Ok(())
}

fn assert_scalar_patch_values(
    processor: &ImageProcessor,
    pixels: &[u8],
    width: usize,
    prepared: &PreprocessedImage,
) {
    let patch_area = processor.patch_size * processor.patch_size;
    let channel_width = processor.temporal_patch_size * patch_area;
    let patch_width = RGB_CHANNELS * channel_width;
    let merge_area = processor.merge_size * processor.merge_size;
    let blocks_per_row = prepared.grid.2 / processor.merge_size;
    assert_eq!(
        prepared.pixels.len(),
        prepared.grid.1 * prepared.grid.2 * patch_width
    );
    for (index, &actual) in prepared.pixels.iter().enumerate() {
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
        let value = f32::from(pixels[(row * width + col) * RGB_CHANNELS + channel]);
        let expected = scalar_normalization(processor, value, channel);
        assert_eq!(actual.to_bits(), expected.to_bits(), "patch value {index}");
    }
}

#[test]
fn disabled_resize_preserves_dimensions_and_rejects_unaligned_images() -> Result<()> {
    use infer_ir::ImageResizePolicy;
    let processor = ImageProcessor {
        patch_size: 2,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 256,
        max_pixels: 1024,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: RESCALE,
    };
    let pixels = vec![255; 4 * 8 * RGB_CHANNELS];
    let processed =
        processor.preprocess_rgb_with_resize(4, 8, &pixels, ImageResizePolicy::Disabled)?;
    assert_eq!(processed.grid, (1, 2, 4));
    assert!(
        processed
            .pixels
            .iter()
            .all(|value| value.to_bits() == 1.0f32.to_bits())
    );
    assert_ne!(
        processor.target_dimensions(4, 8, ImageResizePolicy::Smart)?,
        (4, 8)
    );
    assert!(
        processor
            .target_dimensions(3, 8, ImageResizePolicy::Disabled)
            .is_err()
    );
    assert!(
        processor
            .target_dimensions(0, 8, ImageResizePolicy::Disabled)
            .is_err()
    );
    Ok(())
}

#[test]
fn disabled_resize_matches_embedded_transformers_reference() -> Result<()> {
    use infer_ir::ImageResizePolicy;
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
    // The fixture is the first 2x2 patch block produced by Transformers 5.17.0's
    // Qwen2VLImageProcessor with do_resize=false, on a 128x256 RGB8 image whose
    // row-major byte at index i is i % 251. Extract the same 32x32 input block.
    let mut pixels = Vec::with_capacity(32 * 32 * RGB_CHANNELS);
    for y in 0..32 {
        for x in 0..32 {
            for channel in 0..RGB_CHANNELS {
                let value = ((y * 256 + x) * RGB_CHANNELS + channel) % 251;
                pixels.push(u8::try_from(value).map_err(|_| Error::invalid("fixture byte"))?);
            }
        }
    }
    let result =
        processor.preprocess_rgb_with_resize(32, 32, &pixels, ImageResizePolicy::Disabled)?;
    let expected = include_bytes!("../fixtures/qwen3vl-no-resize-first-block.f32");
    assert_eq!(result.grid, (1, 2, 2));
    assert_eq!(expected.len(), result.pixels.len() * size_of::<f32>());
    for (actual, bytes) in result
        .pixels
        .iter()
        .zip(expected.as_chunks::<{ size_of::<f32>() }>().0)
    {
        let reference = f32::from_le_bytes(*bytes);
        assert!((actual - reference).abs() <= 1.0e-6);
    }
    Ok(())
}

#[test]
fn unresized_full_image_matches_embedded_transformers_patch_block() -> Result<()> {
    use infer_ir::ImageResizePolicy;
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
    let (height, width) = (128, 256);
    let pixels: Vec<u8> = (0..height * width * RGB_CHANNELS)
        .map(|index| u8::try_from(index % 251))
        .collect::<StdResult<_, _>>()
        .map_err(|_| Error::invalid("fixture byte"))?;
    let result = processor.preprocess_rgb_with_resize(
        height,
        width,
        &pixels,
        ImageResizePolicy::Disabled,
    )?;
    assert_eq!(result.grid, (1, 8, 16));
    assert_eq!(
        result.pixels.len(),
        result.grid.1 * result.grid.2 * processor.patch_width()
    );
    let expected = include_bytes!("../fixtures/qwen3vl-no-resize-first-block.f32");
    assert!(expected.len().is_multiple_of(size_of::<f32>()));
    let reference = expected.as_chunks::<{ size_of::<f32>() }>().0;
    assert_eq!(
        reference.len(),
        processor.merge_size * processor.merge_size * processor.patch_width()
    );
    assert!(result.pixels.len() >= reference.len());
    for (actual, bytes) in result.pixels.iter().zip(reference) {
        assert!((actual - f32::from_le_bytes(*bytes)).abs() <= 1.0e-6);
    }
    Ok(())
}

/// Optional independent Transformers fixture generated with `do_resize=false`.
#[test]
fn disabled_resize_matches_transformers_reference() -> Result<()> {
    use infer_ir::ImageResizePolicy;
    let Ok(directory) = var("INFER_NO_RESIZE_GOLDEN_DIR") else {
        return Ok(());
    };
    let processor = package_processor()?;
    let path = Path::new(&directory);
    let pixels = read(path.join("rgb.u8")).map_err(|error| Error::invalid(error.to_string()))?;
    let expected =
        read(path.join("patches.f32")).map_err(|error| Error::invalid(error.to_string()))?;
    let result =
        processor.preprocess_rgb_with_resize(128, 256, &pixels, ImageResizePolicy::Disabled)?;
    assert_eq!(result.grid, (1, 8, 16));
    assert_eq!(expected.len(), result.pixels.len() * size_of::<f32>());
    let mut max_error = 0.0_f32;
    for (actual, bytes) in result
        .pixels
        .iter()
        .zip(expected.as_chunks::<{ size_of::<f32>() }>().0)
    {
        let reference = f32::from_le_bytes(*bytes);
        max_error = max_error.max((actual - reference).abs());
    }
    assert!(max_error <= 1.0e-6, "reference difference {max_error}");
    Ok(())
}
