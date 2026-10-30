use super::{
    MediaRequest, MediaService, SourcePolicy, TempPath, check_video_frame_limit, image_digest,
    load_image, load_image_with, load_video_with, parse_rate, read_url, run_bounded, source_digest,
};
use crate::cpu::{CpuConfig, CpuPool, cancelled_context};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use infer_ir::Modality;
use infer_models::{ImageProcessor, MediaLimits, MultimodalAssets, TextAssets};
use std::env::{temp_dir, var};
use std::fs::{remove_file, write};
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::process::id;
use std::time::{SystemTime, UNIX_EPOCH};

const ONE_PIXEL_PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

fn noisy_image(width: u32, height: u32) -> image::RgbImage {
    let mut image = image::RgbImage::new(width, height);
    let mut state = 0x1234_5678_u32;
    for pixel in image.pixels_mut() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *pixel = image::Rgb([
            u8::try_from(state & 0xff).expect("byte"),
            u8::try_from((state >> 8) & 0xff).expect("byte"),
            u8::try_from((state >> 16) & 0xff).expect("byte"),
        ]);
    }
    image
}

fn encoded(image: image::RgbImage, format: image::ImageFormat) -> Vec<u8> {
    let mut output = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut output, format)
        .expect("encode fixture");
    output.into_inner()
}

fn identity_processor() -> ImageProcessor {
    ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 256,
        max_pixels: 65_536,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    }
}

#[test]
fn image_identity_tracks_source_and_every_processor_parameter() {
    let source = source_digest(b"abc", None).expect("source identity");
    assert_eq!(
        source.0,
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ]
    );
    let processor = identity_processor();
    let identity = image_digest(source, &processor).expect("image identity");
    assert_eq!(identity, image_digest(source, &processor).unwrap());
    let other = source_digest(b"abd", None).unwrap();
    assert_ne!(identity, image_digest(other, &processor).unwrap());
    let mut variants = Vec::new();
    for axis in 0..5 {
        let mut changed = processor.clone();
        match axis {
            0 => changed.patch_size += 1,
            1 => changed.temporal_patch_size += 1,
            2 => changed.merge_size += 1,
            3 => changed.min_pixels += 1,
            _ => changed.max_pixels += 1,
        }
        variants.push(changed);
    }
    for channel in 0..3 {
        let mut mean = processor.clone();
        mean.mean[channel] += 0.25;
        variants.push(mean);
        let mut std = processor.clone();
        std.std[channel] += 0.25;
        variants.push(std);
    }
    let mut scale = processor;
    scale.rescale *= 2.0;
    variants.push(scale);
    for changed in variants {
        assert_ne!(identity, image_digest(source, &changed).unwrap());
    }
    assert!(source_digest(b"", Some(&cancelled_context())).is_err());
}

#[test]
fn image_source_identity_follows_content_not_the_url() {
    let bytes = STANDARD.decode(ONE_PIXEL_PNG).expect("fixture");
    let uri = format!("data:image/png;base64,{}", STANDARD.encode(&bytes));
    let path = temp_dir().join(format!(
        "inferloop-image-identity-{}-{}.png",
        id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (temp, mut file) = TempPath::create_at(path).expect("fixture file");
    file.write_all(&bytes).unwrap();
    drop(file);
    let (inline, inline_source) = load_image_with(&uri, SourcePolicy::Public, None).unwrap();
    let (file, file_source) = load_image_with(
        temp.path.to_str().unwrap(),
        SourcePolicy::TrustedLocal,
        None,
    )
    .unwrap();
    assert_eq!(inline, file);
    assert_eq!(inline_source.0, file_source.0);
    let mut trailing = bytes;
    trailing.push(0);
    let alternative = format!("data:image/png;base64,{}", STANDARD.encode(trailing));
    let (same_pixels, other_source) =
        load_image_with(&alternative, SourcePolicy::Public, None).unwrap();
    assert_eq!(inline, same_pixels);
    assert_ne!(inline_source.0, other_source.0);
}

#[tokio::test]
#[ignore = "requires a real Qwen3-VL package in INFER_VISION_PACKAGE"]
async fn prepared_images_preserve_order_and_source_identity() {
    let package = PathBuf::from(var("INFER_VISION_PACKAGE").unwrap());
    let assets = TextAssets::open(&package, 4096).unwrap();
    let multimodal = MultimodalAssets::open(&package).unwrap().unwrap();
    let processor = multimodal.images.clone();
    let service = MediaService::new(multimodal, MediaLimits::default());
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 2,
        ..CpuConfig::default()
    })
    .unwrap();
    let image = STANDARD.decode(ONE_PIXEL_PNG).unwrap();
    let mut second = image.clone();
    second.push(0);
    let requests = [image, second]
        .into_iter()
        .map(|bytes| MediaRequest {
            modality: Modality::Image,
            url: format!("data:image/png;base64,{}", STANDARD.encode(bytes)),
        })
        .collect::<Vec<_>>();
    let expected = requests
        .iter()
        .map(|request| {
            let bytes = STANDARD
                .decode(request.url.split_once(',').unwrap().1)
                .unwrap();
            image_digest(source_digest(&bytes, None).unwrap(), &processor).unwrap()
        })
        .collect::<Vec<_>>();
    let (prepared, media) = pool
        .run(0, move |context| {
            service.prepare(
                &assets,
                "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>describe",
                &requests,
                context,
            )
        })
        .await
        .unwrap();
    assert_eq!(media.len(), 2);
    assert_eq!(media[0].content_digest, expected[0]);
    assert_eq!(media[1].content_digest, expected[1]);
    assert_ne!(media[0].content_digest, media[1].content_digest);
    assert_eq!(prepared.images[0].pixels, prepared.images[1].pixels);
    assert_eq!(
        media[0].item.as_ref().unwrap().pixels,
        media[1].item.as_ref().unwrap().pixels
    );
}

#[test]
fn data_uri_png_is_decoded_into_rgb8() {
    let image = load_image(&format!("data:image/png;base64,{ONE_PIXEL_PNG}")).expect("tiny PNG");
    assert_eq!((image.width, image.height), (1, 1));
    assert_eq!(image.pixels.len(), 3);
}

#[test]
fn native_png_conversion_preserves_rgb_values_for_color_and_grayscale_inputs() {
    use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};

    let cases = [
        (
            DynamicImage::ImageRgb8(RgbImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap()),
            vec![1, 2, 3, 4, 5, 6],
        ),
        (
            DynamicImage::ImageRgba8(
                RgbaImage::from_raw(2, 1, vec![7, 8, 9, 0, 10, 11, 12, 255]).unwrap(),
            ),
            vec![7, 8, 9, 10, 11, 12],
        ),
        (
            DynamicImage::ImageLuma8(GrayImage::from_raw(2, 1, vec![13, 14]).unwrap()),
            vec![13, 13, 13, 14, 14, 14],
        ),
    ];
    for (image, expected) in cases {
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let uri = format!(
            "data:image/png;base64,{}",
            STANDARD.encode(encoded.into_inner())
        );
        let decoded = load_image(&uri).expect("native PNG");
        assert_eq!((decoded.width, decoded.height), (2, 1));
        assert_eq!(decoded.pixels, expected);
    }
}

#[test]
fn valid_inline_image_may_exceed_the_short_url_limit() {
    let png = encoded(noisy_image(128, 128), image::ImageFormat::Png);
    assert!(png.len() > 8192, "fixture must exceed the URL limit");
    let uri = format!("data:image/png;base64,{}", STANDARD.encode(&png));
    assert!(uri.len() > 8192);
    let image = load_image(&uri).expect("large inline PNG");
    assert_eq!((image.width, image.height), (128, 128));
}

#[test]
fn inline_image_can_exceed_the_former_encoded_byte_limit() {
    let mut bytes = STANDARD.decode(ONE_PIXEL_PNG).expect("fixture");
    bytes.resize(24 * 1024 * 1024 + 1, 0);
    let uri = format!("data:image/png;base64,{}", STANDARD.encode(bytes));
    let image = load_image(&uri).expect("valid image over former byte cap");
    assert_eq!((image.width, image.height), (1, 1));
}

#[test]
fn decoded_image_can_exceed_the_former_dimensions_and_allocation_limits() {
    let (width, height) = (8193, 2732);
    let bytes = encoded(image::RgbImage::new(width, height), image::ImageFormat::Png);
    let uri = format!("data:image/png;base64,{}", STANDARD.encode(bytes));
    let image = load_image(&uri).expect("valid large image");
    assert_eq!(
        (image.width, image.height),
        (width as usize, height as usize)
    );
    assert_eq!(image.pixels.len(), width as usize * height as usize * 3);
    assert_eq!(&image.pixels[..3], &[0, 0, 0]);
    assert_eq!(&image.pixels[image.pixels.len() - 3..], &[0, 0, 0]);
}

#[test]
fn common_native_image_codecs_decode_without_external_tools() {
    for format in [image::ImageFormat::Jpeg, image::ImageFormat::WebP] {
        let bytes = encoded(noisy_image(64, 48), format);
        let mime = match format {
            image::ImageFormat::Jpeg => "jpeg",
            image::ImageFormat::WebP => "webp",
            _ => unreachable!(),
        };
        let uri = format!("data:image/{mime};base64,{}", STANDARD.encode(bytes));
        let image = load_image(&uri).expect("native codec");
        assert_eq!((image.width, image.height), (64, 48));
    }
}

#[test]
fn malformed_image_is_rejected() {
    let mut png = STANDARD.decode(ONE_PIXEL_PNG).expect("fixture");
    png[16..20].copy_from_slice(&100_u32.to_be_bytes());
    png[20..24].copy_from_slice(&100_u32.to_be_bytes());
    let crc = crc32fast::hash(&png[12..29]);
    png[29..33].copy_from_slice(&crc.to_be_bytes());
    let uri = format!("data:image/png;base64,{}", STANDARD.encode(png));
    assert!(load_image(&uri).is_err());
}

#[test]
fn malformed_data_uris_and_bounded_video_inputs_are_rejected() {
    assert!(load_image("data:image/png;base64,!!!").is_err());
    let encoded_over_limit = format!("data:image/png;base64,{}", "A".repeat(5000));
    assert!(read_url(&encoded_over_limit, Some(1), SourcePolicy::Public, None).is_err());
    assert!(load_image("ftp://example.invalid/a.png").is_err());
}

#[test]
fn public_source_policy_rejects_server_local_paths() {
    for url in ["file:///etc/passwd", "/etc/passwd"] {
        let error = read_url(url, Some(1024), SourcePolicy::Public, None).expect_err("local path");
        assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
    }
    assert!(
        read_url(
            "data:image/png;base64,AAAA",
            Some(1024),
            SourcePolicy::Public,
            None
        )
        .is_ok()
    );
}

#[test]
fn source_fps_is_preserved_not_clamped() {
    assert!((parse_rate("1/2").expect("half fps") - 0.5).abs() < f64::EPSILON);
    assert!((parse_rate("30000/1001").expect("ntsc") - 29.970_029_970_03).abs() < 1e-9);
    assert!(parse_rate("0/1").is_err());
    assert!(parse_rate("1/0").is_err());
}

#[test]
fn limits_are_per_modality_and_all_zero_is_text_only() {
    let limits = MediaLimits::default();
    assert!(!limits.text_only());
    assert!(limits.check(&Modality::Image, 1).is_ok());
    assert!(limits.check(&Modality::Image, 5).is_err());
    assert!(limits.check(&Modality::Video, 2).is_err());
    let text_only = MediaLimits {
        images: 0,
        videos: 0,
    };
    assert!(text_only.text_only());
    assert!(text_only.check(&Modality::Image, 1).is_err());
}

#[cfg(unix)]
#[test]
fn stuck_subprocess_is_killed_reaped_and_bounded() {
    use std::process::Command;
    use std::time::{Duration, Instant};
    let mut command = Command::new("/bin/sh");
    command.args(["-c", "sleep 30"]);
    let started = Instant::now();
    let error = run_bounded(&mut command, 1024, 1024, Duration::from_millis(100), None)
        .expect_err("stuck subprocess");
    assert_eq!(error.code, infer_core::ErrorCode::Backend);
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[cfg(unix)]
#[test]
fn failing_subprocess_drains_stderr_and_reaps() {
    use std::process::Command;
    use std::time::Duration;
    let mut command = Command::new("/bin/sh");
    command.args(["-c", "echo boom >&2; exit 7"]);
    let output = run_bounded(&mut command, 1024, 1024, Duration::from_secs(2), None)
        .expect("failing subprocess ran");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("boom"));
}

#[cfg(unix)]
#[test]
fn subprocess_stdout_cap_is_enforced() {
    use std::process::Command;
    use std::time::Duration;
    let mut command = Command::new("/bin/sh");
    command.args(["-c", "yes x | head -c 4096"]);
    let error = run_bounded(&mut command, 16, 1024, Duration::from_secs(2), None)
        .expect_err("capped output");
    assert_eq!(error.code, infer_core::ErrorCode::Capacity);
}

#[test]
fn temp_path_is_exclusive_and_removed_on_drop() {
    let path = temp_dir().join(format!(
        "inferloop-test-{}-{}.tmp",
        id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    write(&path, b"occupied").expect("seed");
    assert!(TempPath::create_at(path.clone()).is_err());
    remove_file(&path).expect("remove seed");
    let (temp, file) = TempPath::create_at(path.clone()).expect("exclusive");
    drop(file);
    assert!(path.exists());
    drop(temp);
    assert!(!path.exists());
}

#[test]
fn video_frame_limit_is_explicit_and_not_silently_truncated() {
    assert!(check_video_frame_limit(Some(768)).is_ok());
    let error = check_video_frame_limit(Some(769)).expect_err("over-limit clip");
    assert_eq!(error.code, infer_core::ErrorCode::Capacity);
    assert!(check_video_frame_limit(None).is_ok());
}

#[cfg(unix)]
#[test]
fn lossless_video_roundtrip_preserves_frames_and_fps() {
    use std::io::Write;
    use std::process::Command;
    use std::time::Duration;

    let width = 100u32;
    let height = 60u32;
    let frames: Vec<Vec<u8>> = (0..4u8)
        .map(|frame| {
            (0..height as usize)
                .flat_map(|y| {
                    (0..width as usize).flat_map(move |x| {
                        [
                            u8::try_from(y).unwrap_or(0),
                            u8::try_from(x + usize::from(frame) * 10).unwrap_or(0),
                            frame,
                        ]
                    })
                })
                .collect()
        })
        .collect();
    let raw: Vec<u8> = frames.concat();
    let (raw_path, mut raw_file) = TempPath::create("raw-video", "rgb").expect("raw temp");
    raw_file.write_all(&raw).expect("raw write");
    drop(raw_file);
    let (clip, clip_file) = TempPath::create("clip", "mkv").expect("clip temp");
    drop(clip_file);
    let mut encode = Command::new("ffmpeg");
    encode.args([
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "100x60",
        "-r",
        "30",
        "-i",
        &raw_path.path.to_string_lossy(),
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "gbrp",
        "-an",
        &clip.path.to_string_lossy(),
    ]);
    let encoded = run_bounded(&mut encode, 1 << 20, 1 << 20, Duration::from_secs(10), None)
        .expect("ffmpeg encode");
    assert!(encoded.status.success(), "{:?}", encoded.stderr);
    let video = load_video_with(
        &clip.path.to_string_lossy(),
        SourcePolicy::TrustedLocal,
        None,
    )
    .expect("lossless video");
    assert!((video.fps - 30.0).abs() < f64::EPSILON);
    assert_eq!(video.frames.len(), 4);
    for (loaded, expected) in video.frames.iter().zip(&frames) {
        assert_eq!(loaded.width, width as usize);
        assert_eq!(loaded.height, height as usize);
        assert_eq!(&loaded.pixels, expected);
    }
}

#[cfg(unix)]
#[test]
fn cancellation_guard_kills_and_reaps_a_subprocess() {
    use std::process::Command;
    use std::time::{Duration, Instant};
    let context = cancelled_context();
    let mut command = Command::new("/bin/sh");
    command.args(["-c", "sleep 30"]);
    let started = Instant::now();
    let error = run_bounded(
        &mut command,
        1024,
        1024,
        Duration::from_secs(5),
        Some(&context),
    )
    .expect_err("cancelled subprocess");
    assert_eq!(error.code, infer_core::ErrorCode::Capacity);
    assert!(started.elapsed() < Duration::from_secs(2));
}
