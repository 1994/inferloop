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
