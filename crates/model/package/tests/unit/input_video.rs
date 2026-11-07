use super::*;
use std::env::var;
use std::fs::read;
use std::path::Path;

fn frame(height: usize, width: usize) -> RgbImage {
    RgbImage {
        height,
        width,
        pixels: vec![0; height * width * RGB_CHANNELS],
    }
}

fn clip(frames: usize, fps: f64) -> RawVideo {
    RawVideo {
        fps,
        frames: (0..frames).map(|_| frame(1, 1)).collect(),
    }
}

fn processor() -> VideoProcessor {
    VideoProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 4_096,
        max_pixels: 25_165_824,
        mean: [0.5; RGB_CHANNELS],
        std: [0.5; RGB_CHANNELS],
        rescale: 1.0 / 255.0,
        fps: 2.0,
        min_frames: 4,
        max_frames: 768,
    }
}

#[test]
fn sampling_matches_the_reference_sampler() {
    let processor = processor();
    let cases = [
        (30, 30.0, vec![0, 10, 19, 29]),
        (60, 30.0, vec![0, 20, 39, 59]),
        (10, 5.0, vec![0, 3, 6, 9]),
        (3, 3.0, vec![0, 1, 2]),
    ];
    for (frames, fps, expected) in cases {
        assert_eq!(
            processor.sample_frames(&clip(frames, fps)).unwrap(),
            expected
        );
    }
    // A long clip is capped at `max_frames`, a short one at its decoded length.
    let capped = processor.sample_frames(&clip(2_000, 30.0)).unwrap();
    assert_eq!(capped.len(), 133);
    assert_eq!((capped[0], capped[132]), (0, 1_999));
    let mut short = processor.clone();
    short.max_frames = 10;
    assert_eq!(short.sample_frames(&clip(2_000, 30.0)).unwrap().len(), 10);
    assert!(processor.sample_frames(&clip(0, 30.0)).is_err());
    assert!(processor.sample_frames(&clip(10, 0.0)).is_err());
}

#[test]
fn smart_resize_matches_the_reference_grid() -> Result<()> {
    let processor = processor();
    // The package's video budget is 4096..=25165824 pixels over the whole clip.
    assert_eq!(processor.smart_resize(4, 60, 100)?, (64, 96));
    assert_eq!(processor.smart_resize(4, 480, 640)?, (480, 640));
    // 32 frames at 720p exceed the budget and shrink to the next 32-multiple grid.
    assert_eq!(processor.smart_resize(32, 720, 1280)?, (640, 1152));
    // An odd sampled count rounds its temporal budget to an even group.
    assert_eq!(processor.smart_resize(5, 60, 100)?, (64, 96));
    // Frames below one patch side scale up before the pixel budget applies.
    assert_eq!(processor.smart_resize(2, 10, 10)?, (64, 64));
    assert!(processor.smart_resize(1, 60, 100).is_err());
    assert!(processor.smart_resize(4, 0, 100).is_err());
    Ok(())
}

#[test]
fn temporal_taps_keep_adjacent_frames_distinct() -> Result<()> {
    let processor = processor();
    let mut dark = frame(32, 32);
    let mut bright = frame(32, 32);
    bright.pixels.fill(255);
    dark.pixels.fill(0);
    let processed = processor.preprocess(&RawVideo {
        fps: 30.0,
        frames: vec![dark, bright],
    })?;
    // Two frames fill one temporal group; 64x64 resizes into a 4x4 patch grid.
    assert_eq!(processed.image.grid, (1, 4, 4));
    assert_eq!(processed.image.pixels.len(), 16 * processor.patch_width());
    let taps = processor.patch_width() / processor.temporal_patch_size;
    // Mean 0.5 / std 0.5 maps the two frames to -1 and +1 in the temporal taps.
    assert!((processed.image.pixels[0] + 1.0).abs() < 1e-6);
    assert!((processed.image.pixels[taps] - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn frame_resize_matches_serial_pixels_and_order_across_worker_chunks() -> Result<()> {
    for count in [
        0,
        1,
        PARALLEL_FRAMES - 1,
        PARALLEL_FRAMES,
        PARALLEL_FRAMES + 3,
    ] {
        let frames: Vec<_> = (0..count)
            .map(|index| {
                let mut image = frame(2, 3);
                image.pixels.fill(u8::try_from(index * 17).unwrap());
                image
            })
            .collect();
        let borrowed: Vec<_> = frames.iter().collect();
        let expected: Vec<_> = frames
            .iter()
            .map(|frame| ImageProcessor::resize(frame, 4, 5))
            .collect();
        assert_eq!(
            resize_frames(&borrowed, 4, 5)?,
            expected,
            "frame count {count}"
        );
    }
    Ok(())
}

#[test]
fn odd_clips_pad_their_last_frame_into_a_whole_temporal_patch() -> Result<()> {
    let mut processor = processor();
    processor.fps = 10.0;
    processor.min_frames = 1;
    let frames: Vec<RgbImage> = (0..5)
        .map(|index| {
            let mut image = frame(32, 32);
            image.pixels.fill(u8::try_from(index * 40).unwrap_or(0));
            image
        })
        .collect();
    let processed = processor.preprocess(&RawVideo { fps: 10.0, frames })?;
    // Five sampled frames pad to six, so three temporal groups reach the tower.
    assert_eq!(processed.image.grid.0, 3);
    assert_eq!(processed.timestamps.len(), 3);
    Ok(())
}

#[test]
fn incremental_groups_match_complete_clip_pixels_and_timestamps() -> Result<()> {
    let mut processor = processor();
    processor.fps = 5.0;
    processor.min_frames = 1;
    for (count, fps, height, width) in [(3, 5.0, 32, 32), (13, 7.0, 34, 62), (30, 10.0, 64, 65)] {
        let frames: Vec<_> = (0..count)
            .map(|index| {
                let mut image = frame(height, width);
                for (offset, pixel) in image.pixels.iter_mut().enumerate() {
                    *pixel = u8::try_from((index * 37 + offset * 13) % 256).unwrap();
                }
                image
            })
            .collect();
        let clip = RawVideo { fps, frames };
        let reference = processor.preprocess(&clip)?;
        let plan = processor.plan(count, fps, height, width)?;
        assert_eq!(plan.indices(), processor.sample_frames(&clip)?);
        let mut pixels = Vec::new();
        let mut timestamps = Vec::new();
        for (group, indices) in plan
            .indices()
            .chunks(processor.temporal_patch_size)
            .enumerate()
        {
            let frames: Vec<_> = indices
                .iter()
                .map(|index| clip.frames[*index].clone())
                .collect();
            let processed = plan.preprocess_group(group, &frames)?;
            assert_eq!(
                processed.image.grid,
                (1, reference.image.grid.1, reference.image.grid.2)
            );
            pixels.extend(processed.image.pixels);
            timestamps.extend(processed.timestamps);
        }
        assert_eq!(plan.groups(), reference.image.grid.0);
        assert_eq!(pixels, reference.image.pixels);
        assert_eq!(timestamps, reference.timestamps);
        assert!(plan.preprocess_group(plan.groups(), &[]).is_err());
        assert!(plan.preprocess_group(0, &[]).is_err());
    }
    Ok(())
}

#[test]
fn configuration_comes_from_the_package() {
    let Ok(package) = var("INFER_VISION_PACKAGE") else {
        return;
    };
    let Ok(config) = read(Path::new(&package).join("video_preprocessor_config.json")) else {
        return;
    };
    let processor = VideoProcessor::from_json(&config).unwrap();
    assert_eq!(processor.patch_size, 16);
    assert_eq!(processor.temporal_patch_size, 2);
    assert_eq!(processor.merge_size, 2);
    assert_eq!(processor.min_pixels, 4_096);
    assert_eq!(processor.max_pixels, 25_165_824);
    assert_eq!(processor.fps, 2.0);
    assert_eq!(processor.min_frames, 4);
    assert_eq!(processor.max_frames, 768);
    assert_eq!(processor.patch_width(), 1_536);
}

/// The pipeline must reproduce the reference processor's pixels and grid.
///
/// Skipped unless both the package and the video golden are provided.
#[test]
fn preprocessing_matches_the_reference_processor() -> Result<()> {
    const TOLERANCE: f32 = 1e-6;
    let (Ok(package), Ok(golden)) = (
        var("INFER_VISION_PACKAGE"),
        var("INFER_VISION_VIDEO_GOLDEN"),
    ) else {
        return Ok(());
    };
    let config = read(Path::new(&package).join("video_preprocessor_config.json"))
        .map_err(|error| Error::invalid(error.to_string()))?;
    let processor = VideoProcessor::from_json(&config)?;
    let mut file = crate::SafetensorsFile::open(&golden)?;
    let expected = file.read_f32("pixel_values", 1 << 30)?;
    let grid = file.read_bytes("grid_thw", 1 << 20)?;
    let shape = file.read_bytes("shape", 1 << 20)?;
    let fps = file.read_f32("fps", 1 << 20)?;
    let pixels = file.read_bytes("frames", 1 << 30)?;
    let values = |bytes: &[u8]| -> Vec<i64> {
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| i64::from_le_bytes(*chunk))
            .collect()
    };
    let shape = values(&shape);
    let grid = values(&grid);
    let (frames, height, width) = (
        usize::try_from(shape[0]).map_err(|_| Error::invalid("golden frame count"))?,
        usize::try_from(shape[1]).map_err(|_| Error::invalid("golden height"))?,
        usize::try_from(shape[2]).map_err(|_| Error::invalid("golden width"))?,
    );
    let stride = height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(RGB_CHANNELS))
        .ok_or_else(|| Error::invalid("golden frame shape"))?;
    let frames = (0..frames)
        .map(|index| RgbImage {
            height,
            width,
            pixels: pixels[index * stride..(index + 1) * stride].to_vec(),
        })
        .collect();
    let video = RawVideo {
        fps: f64::from(fps.data[0]),
        frames,
    };
    let processed = processor.preprocess(&video)?;
    assert_eq!(
        processed.image.grid,
        (
            usize::try_from(grid[0]).map_err(|_| Error::invalid("golden grid"))?,
            usize::try_from(grid[1]).map_err(|_| Error::invalid("golden grid"))?,
            usize::try_from(grid[2]).map_err(|_| Error::invalid("golden grid"))?,
        )
    );
    assert_eq!(processed.image.pixels.len(), expected.data.len());
    let mut max_abs = 0f32;
    for (value, gold) in processed.image.pixels.iter().zip(expected.data.iter()) {
        max_abs = max_abs.max((value - gold).abs());
    }
    assert!(
        max_abs <= TOLERANCE,
        "video preprocessing deviates by {max_abs}"
    );
    Ok(())
}
