use super::options::{ImageProcessingOptions, check_working_set};
use super::{
    MEDIA_STAGING_BYTES, SourcePolicy, TempPath, check_video_frame_limit, load_image,
    load_video_with, parse_rate, read_url, run_bounded,
};
use crate::cpu::{CpuPool, cancelled_context};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use infer_models::MediaLimits;
use infer_models::{ImageProcessor, RawImage, TextAssets};
use std::env::temp_dir;
use std::fs::{remove_file, write};
use std::io::Cursor;
use std::process::id;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

const ONE_PIXEL_PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

#[test]
fn scoped_image_options_replace_the_request_scope() {
    let defaults = ImageProcessingOptions::parse(
        r#"{"images_kwargs":{"min_pixels":4096,"max_pixels":262144}}"#,
    )
    .unwrap();
    let request =
        ImageProcessingOptions::parse(r#"{"images_kwargs":{"max_pixels":65536}}"#).unwrap();
    let processor = ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 1024,
        max_pixels: 1_048_576,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    };
    let effective = defaults
        .merge(Some(&request))
        .unwrap()
        .processor(&processor)
        .unwrap();
    assert_eq!((effective.min_pixels, effective.max_pixels), (1024, 65536));
    let scoped = ImageProcessingOptions::parse(
        r#"{"images_kwargs":{"size":{"shortest_edge":1024,"longest_edge":65536}}}"#,
    )
    .unwrap()
    .processor(&processor)
    .unwrap();
    assert_eq!(scoped.smart_resize(512, 512).unwrap(), (256, 256));
    let flat_default = ImageProcessingOptions::parse(r#"{"min_pixels":4096}"#).unwrap();
    let effective = flat_default
        .merge(Some(&request))
        .unwrap()
        .processor(&processor)
        .unwrap();
    assert_eq!((effective.min_pixels, effective.max_pixels), (4096, 65536));
    let duplicate_default = ImageProcessingOptions::parse(r#"{"max_pixels":262144}"#).unwrap();
    assert!(duplicate_default.merge(Some(&request)).is_err());
    assert!(
        ImageProcessingOptions::parse(
            r#"{"max_pixels":65536,"images_kwargs":{"max_pixels":131072}}"#
        )
        .is_err()
    );
}

#[test]
fn image_resize_flag_respects_request_scope_and_rejects_conflicts() {
    use infer_ir::ImageResizePolicy;

    let defaults = ImageProcessingOptions::parse(r#"{"do_resize":false}"#).unwrap();
    assert_eq!(defaults.resize_policy(), ImageResizePolicy::Disabled);
    let request = ImageProcessingOptions::parse(r#"{"do_resize":true}"#).unwrap();
    assert_eq!(
        defaults.merge(Some(&request)).unwrap().resize_policy(),
        ImageResizePolicy::Smart
    );
    let scoped = ImageProcessingOptions::parse(r#"{"images_kwargs":{"do_resize":false}}"#).unwrap();
    assert_eq!(scoped.resize_policy(), ImageResizePolicy::Disabled);
    assert!(
        ImageProcessingOptions::parse(r#"{"do_resize":true,"images_kwargs":{"do_resize":false}}"#)
            .is_err()
    );
}

#[test]
fn image_resize_options_merge_and_preserve_shared_defaults() {
    let defaults =
        ImageProcessingOptions::parse(r#"{"min_pixels":65536,"max_pixels":1048576}"#).unwrap();
    let request = ImageProcessingOptions::parse(r#"{"max_pixels":262144}"#).unwrap();
    let processor = ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 1024,
        max_pixels: 16_777_216,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    };
    let effective = defaults
        .merge(Some(&request))
        .unwrap()
        .processor(&processor)
        .unwrap();
    assert_eq!(
        (effective.min_pixels, effective.max_pixels),
        (65536, 262_144)
    );
    assert_eq!(effective.smart_resize(1024, 1024).unwrap(), (512, 512));
    assert_eq!(
        (processor.min_pixels, processor.max_pixels),
        (1024, 16_777_216)
    );
    assert_eq!(
        defaults.processor(&processor).unwrap().max_pixels,
        1_048_576
    );
    let mut zero_minimum = processor.clone();
    zero_minimum.min_pixels = 0;
    assert_eq!(
        ImageProcessingOptions::default()
            .processor(&zero_minimum)
            .unwrap()
            .min_pixels,
        0
    );
    let size =
        ImageProcessingOptions::parse(r#"{"size":{"shortest_edge":65536,"longest_edge":262144}}"#)
            .unwrap();
    assert_eq!(
        size.processor(&processor)
            .unwrap()
            .smart_resize(1024, 1024)
            .unwrap(),
        (512, 512)
    );
    let image = RawImage {
        height: 1024,
        width: 1024,
        pixels: vec![0; 1024 * 1024 * 3],
    };
    assert!(check_working_set(&[image], &effective).is_ok());
    // At equal dimensions, the patch buffer and its handoff copy fit the per-item budget.
    let image = RawImage {
        height: 704,
        width: 704,
        pixels: vec![0; 704 * 704 * 3],
    };
    assert!(check_working_set(&[image], &defaults.processor(&processor).unwrap()).is_ok());
}

#[test]
fn image_resize_options_reject_ambiguous_unsupported_and_unbounded_inputs() {
    for value in [
        r#"{"size":{"shortest_edge":1}}"#,
        r#"{"size":{"shortest_edge":1,"longest_edge":2},"max_pixels":2}"#,
        r#"{"max_pixels":0}"#,
        r#"{"min_pixels":100,"max_pixels":50}"#,
        r#"{"do_resize":"false"}"#,
        r#"{"images_kwargs":null}"#,
        r#"{"images_kwargs":{"size":{"shortest_edge":null,"longest_edge":2}}}"#,
        r#"{"images_kwargs":{"unsupported":1}}"#,
        r#"{"max_pixels":16777217}"#,
    ] {
        assert!(
            ImageProcessingOptions::parse(value).is_err(),
            "accepted {value}"
        );
    }
    let processor = ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 16_777_216,
        max_pixels: 16_777_216,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    };
    let image = RawImage {
        height: 32,
        width: 32,
        pixels: vec![0; 32 * 32 * 3],
    };
    assert!(check_working_set(&[image], &processor).is_err());
}

/// The real package fixture is required to exercise tokenizer, grid and media handoff together.
#[tokio::test]
async fn image_request_overrides_are_isolated_and_align_media() {
    use crate::cpu::{CpuConfig, CpuPool};
    use std::env::var;
    use std::path::Path;
    use std::sync::Arc;

    let Ok(package) = var("INFER_VISION_PACKAGE") else {
        return;
    };
    let root = Path::new(&package);
    let multimodal = infer_models::MultimodalAssets::open(root)
        .expect("package metadata")
        .expect("vision model");
    let assets = Arc::new(TextAssets::open(root, 4096).expect("tokenizer"));
    let defaults = ImageProcessingOptions::parse(r#"{"max_pixels":262144}"#).unwrap();
    let service = Arc::new(
        super::MediaService::new(multimodal, MediaLimits::default()).with_image_options(defaults),
    );
    let uri = format!(
        "data:image/png;base64,{}",
        STANDARD.encode(encoded(
            image::RgbImage::new(1024, 1024),
            image::ImageFormat::Png
        ))
    );
    let pool = CpuPool::new(CpuConfig::default()).unwrap();
    let run = |request_options: Option<ImageProcessingOptions>| {
        let assets = assets.clone();
        let service = service.clone();
        let uri = uri.clone();
        pool.run(MEDIA_STAGING_BYTES, move |context| {
            let requests = [super::MediaRequest {
                modality: infer_ir::Modality::Image,
                url: uri,
            }];
            service.prepare_with_options(
                &assets,
                "<|vision_start|><|image_pad|><|vision_end|>describe",
                &requests,
                context,
                request_options.as_ref(),
            )
        })
    };
    let requested = ImageProcessingOptions::parse(r#"{"max_pixels":65536}"#).unwrap();
    let (normal, overridden) = tokio::join!(run(None), run(Some(requested)));
    let (normal, normal_media) = normal.expect("service defaults");
    let (overridden, overridden_media) = overridden.expect("request override");
    assert_eq!(normal.images[0].grid, (1, 32, 32));
    assert_eq!(overridden.images[0].grid, (1, 16, 16));
    for (prepared, media) in [(normal, normal_media), (overridden, overridden_media)] {
        assert_eq!(
            prepared.images[0].grid,
            media[0].item.as_ref().unwrap().grid
        );
        assert_eq!(prepared.spans[0], media[0].item.as_ref().unwrap().span);
        assert_eq!(
            prepared.spans[0].len(),
            prepared.images[0].grid.1 * prepared.images[0].grid.2 / 4
        );
    }

    assert_scoped_image_override(&pool, &service, &assets, &uri).await;
    assert_unresized_image(&pool, &service, &assets).await;

    let run_video = |request_options: Option<ImageProcessingOptions>| {
        let assets = assets.clone();
        let service = service.clone();
        pool.run(MEDIA_STAGING_BYTES, move |context| {
            let requests = [super::MediaRequest {
                modality: infer_ir::Modality::Video,
                url: "invalid-scheme:video".to_owned(),
            }];
            service.prepare_with_options(
                &assets,
                "<|vision_start|><|video_pad|><|vision_end|>describe",
                &requests,
                context,
                request_options.as_ref(),
            )
        })
    };
    let error = run_video(None).await.expect_err("invalid URL");
    assert!(error.message.starts_with("media URL must be"));
    let options = ImageProcessingOptions::parse(r#"{"max_pixels":65536}"#).unwrap();
    let error = run_video(Some(options))
        .await
        .expect_err("video image options");
    assert_eq!(
        error.message,
        "image processor options do not apply to video"
    );
}

async fn assert_unresized_image(
    pool: &CpuPool,
    service: &Arc<super::MediaService>,
    assets: &Arc<TextAssets>,
) {
    let image = image::RgbImage::from_pixel(256, 128, image::Rgb([32, 64, 128]));
    let uri = format!(
        "data:image/png;base64,{}",
        STANDARD.encode(encoded(image, image::ImageFormat::Png))
    );
    let options = ImageProcessingOptions::parse(r#"{"do_resize":false}"#).unwrap();
    let service = service.clone();
    let assets = assets.clone();
    let (prepared, media) = pool
        .run(MEDIA_STAGING_BYTES, move |context| {
            let requests = [super::MediaRequest {
                modality: infer_ir::Modality::Image,
                url: uri,
            }];
            service.prepare_with_options(
                &assets,
                "<|vision_start|><|image_pad|><|vision_end|>describe",
                &requests,
                context,
                Some(&options),
            )
        })
        .await
        .expect("unresized image");
    assert_eq!(prepared.images[0].grid, (1, 8, 16));
    assert_eq!(
        prepared.images[0].grid,
        media[0].item.as_ref().unwrap().grid
    );
    assert_eq!(prepared.spans[0], media[0].item.as_ref().unwrap().span);
    assert_eq!(prepared.spans[0].len(), 32);
}

async fn assert_scoped_image_override(
    pool: &CpuPool,
    service: &super::MediaService,
    assets: &Arc<TextAssets>,
    uri: &str,
) {
    let defaults =
        ImageProcessingOptions::parse(r#"{"images_kwargs":{"max_pixels":262144}}"#).unwrap();
    let service = Arc::new(
        super::MediaService::new(service.assets.clone(), MediaLimits::default())
            .with_image_options(defaults),
    );
    let options =
        ImageProcessingOptions::parse(r#"{"images_kwargs":{"max_pixels":65536}}"#).unwrap();
    let assets = assets.clone();
    let uri = uri.to_owned();
    let (prepared, media) = pool
        .run(MEDIA_STAGING_BYTES, move |context| {
            let requests = [super::MediaRequest {
                modality: infer_ir::Modality::Image,
                url: uri,
            }];
            service.prepare_with_options(
                &assets,
                "<|vision_start|><|image_pad|><|vision_end|>describe",
                &requests,
                context,
                Some(&options),
            )
        })
        .await
        .expect("scoped override");
    assert_eq!(prepared.images[0].grid, (1, 16, 16));
    assert_eq!(
        prepared.images[0].grid,
        media[0].item.as_ref().unwrap().grid
    );
    assert_eq!(prepared.spans[0], media[0].item.as_ref().unwrap().span);
}

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

#[test]
fn data_uri_png_is_decoded_into_rgb8() {
    let image = load_image(&format!("data:image/png;base64,{ONE_PIXEL_PNG}")).expect("tiny PNG");
    assert_eq!((image.width, image.height), (1, 1));
    assert_eq!(image.pixels.len(), 3);
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
fn high_dimension_image_header_is_rejected_before_full_decode() {
    let mut png = STANDARD.decode(ONE_PIXEL_PNG).expect("fixture");
    png[16..20].copy_from_slice(&100_000_u32.to_be_bytes());
    png[20..24].copy_from_slice(&100_000_u32.to_be_bytes());
    let crc = crc32fast::hash(&png[12..29]);
    png[29..33].copy_from_slice(&crc.to_be_bytes());
    let uri = format!("data:image/png;base64,{}", STANDARD.encode(png));
    let error = load_image(&uri).expect_err("huge dimensions");
    assert_eq!(error.code, infer_core::ErrorCode::Capacity);
}

#[test]
fn malformed_and_oversized_data_uris_are_rejected_before_decode() {
    assert!(load_image("data:image/png;base64,!!!").is_err());
    let oversized = "A".repeat(MEDIA_STAGING_BYTES + 1);
    assert!(load_image(&format!("data:image/png;base64,{oversized}")).is_err());
    let encoded_over_limit = format!("data:image/png;base64,{}", "A".repeat(5000));
    assert!(read_url(&encoded_over_limit, 1, SourcePolicy::Public, None).is_err());
    assert!(load_image("ftp://example.invalid/a.png").is_err());
}

#[test]
fn public_source_policy_rejects_server_local_paths() {
    for url in ["file:///etc/passwd", "/etc/passwd"] {
        let error = read_url(url, 1024, SourcePolicy::Public, None).expect_err("local path");
        assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
    }
    assert!(
        read_url(
            "data:image/png;base64,AAAA",
            1024,
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
    assert!(limits.check(&infer_ir::Modality::Image, 1).is_ok());
    assert!(limits.check(&infer_ir::Modality::Image, 5).is_err());
    assert!(limits.check(&infer_ir::Modality::Video, 2).is_err());
    let text_only = MediaLimits {
        images: 0,
        videos: 0,
    };
    assert!(text_only.text_only());
    assert!(text_only.check(&infer_ir::Modality::Image, 1).is_err());
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

#[tokio::test]
async fn request_media_handoff_preserves_bits_and_releases_temporary_pixels() {
    use crate::cpu::CpuConfig;
    use infer_ir::Modality;
    use infer_models::{PreparedPrompt, PromptImage};

    let prepared = PreparedPrompt {
        tokens: vec![1, 2, 3, 4, 5, 6],
        images: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .enumerate()
            .map(|(index, value)| PromptImage {
                pixels: vec![value, -0.0, 1.25],
                grid: (1, 1, index + 1),
            })
            .collect(),
        spans: (0..6).map(|index| index..index + 1).collect(),
    };
    let pool = CpuPool::new(CpuConfig::default()).expect("CPU pool");
    pool.run(MEDIA_STAGING_BYTES, move |context| {
        for modality in [Modality::Image, Modality::Video] {
            let mut retained = prepared.clone();
            let mut released = prepared.clone();
            let expected = super::media_inputs(
                &mut retained,
                &modality,
                context,
                super::PreparedPixels::Retain,
            )?;
            let actual = super::media_inputs(
                &mut released,
                &modality,
                context,
                super::PreparedPixels::Release,
            )?;
            assert_eq!(retained, prepared);
            assert_eq!(released.tokens, prepared.tokens);
            assert_eq!(released.spans, prepared.spans);
            assert!(
                released
                    .images
                    .iter()
                    .all(|image| image.pixels.capacity() == 0)
            );
            assert_eq!(actual, expected);
            for (input, image) in actual.iter().zip(&prepared.images) {
                let item = input.item.as_ref().expect("prepared media");
                assert_eq!(
                    item.pixels
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    image
                        .pixels
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
        Ok(())
    })
    .await
    .expect("media handoff");
}

#[tokio::test]
async fn default_image_preparation_respects_the_cpu_byte_budget() {
    use crate::cpu::CpuConfig;
    use infer_core::ErrorCode;
    use infer_ir::Modality;
    use std::env::var;
    use std::path::Path;

    let Ok(package) = var("INFER_VISION_PACKAGE") else {
        return;
    };
    let root = Path::new(&package);
    let multimodal = infer_models::MultimodalAssets::open(root)
        .expect("package metadata")
        .expect("vision model");
    let assets = Arc::new(TextAssets::open(root, 4096).expect("tokenizer"));
    let service = Arc::new(super::MediaService::new(multimodal, MediaLimits::default()));
    let pixels = image::RgbImage::from_pixel(1024, 1024, image::Rgb([24, 64, 128]));
    let request = super::MediaRequest {
        modality: Modality::Image,
        url: format!(
            "data:image/png;base64,{}",
            STANDARD.encode(encoded(pixels, image::ImageFormat::Png))
        ),
    };
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: MEDIA_STAGING_BYTES,
        ..CpuConfig::default()
    })
    .expect("CPU pool");
    let limited_assets = assets.clone();
    let limited_service = service.clone();
    let limited_request = request.clone();
    let result = pool
        .run(MEDIA_STAGING_BYTES, move |context| {
            limited_service.prepare_request(
                &limited_assets,
                "<|vision_start|><|image_pad|><|vision_end|>",
                &[limited_request],
                context,
                None,
            )
        })
        .await;
    assert!(matches!(result, Err(error) if error.code == ErrorCode::Capacity));
    assert_eq!(pool.inspect().retained_bytes, 0);
    let funded = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: 64 << 20,
        ..CpuConfig::default()
    })
    .expect("funded CPU pool");
    let default_assets = assets.clone();
    let default_service = service.clone();
    let default_request = request.clone();
    let (default_tokens, media) = funded
        .run(MEDIA_STAGING_BYTES, move |context| {
            default_service.prepare_request(
                &default_assets,
                "<|vision_start|><|image_pad|><|vision_end|>",
                &[default_request],
                context,
                None,
            )
        })
        .await
        .expect("funded image preparation");
    assert_eq!(media.len(), 1);
    let default_digest = media[0].content_digest.clone();
    drop(media);
    let options = ImageProcessingOptions::parse(r#"{"do_resize":false}"#).expect("disabled resize");
    let (unresized_tokens, media) = funded
        .run(MEDIA_STAGING_BYTES, move |context| {
            service.prepare_request(
                &assets,
                "<|vision_start|><|image_pad|><|vision_end|>",
                &[request],
                context,
                Some(&options),
            )
        })
        .await
        .expect("funded image without resize");
    assert_eq!(unresized_tokens, default_tokens);
    assert_eq!(media.len(), 1);
    assert_eq!(media[0].content_digest, default_digest);
    assert_eq!(funded.inspect().retained_bytes, 0);
}

#[tokio::test]
async fn multi_image_digest_matches_each_prepared_payload() {
    use crate::cpu::CpuConfig;
    use infer_ir::Modality;
    use std::env::var;
    use std::path::Path;

    const IMAGE_COUNT: usize = 6;
    let Ok(package) = var("INFER_VISION_PACKAGE") else {
        return;
    };
    let root = Path::new(&package);
    let multimodal = infer_models::MultimodalAssets::open(root)
        .expect("package metadata")
        .expect("vision model");
    let assets = Arc::new(TextAssets::open(root, 4096).expect("tokenizer"));
    let service = Arc::new(super::MediaService::new(
        multimodal,
        MediaLimits {
            images: IMAGE_COUNT,
            videos: 1,
        },
    ));
    let requests: Vec<_> = [24_u8, 48, 72, 96, 144, 192]
        .into_iter()
        .map(|value| {
            let pixels = image::RgbImage::from_pixel(256, 256, image::Rgb([value, 64, 128]));
            super::MediaRequest {
                modality: Modality::Image,
                url: format!(
                    "data:image/png;base64,{}",
                    STANDARD.encode(encoded(pixels, image::ImageFormat::Png))
                ),
            }
        })
        .collect();
    let config = CpuConfig {
        max_bytes: 2 * IMAGE_COUNT * MEDIA_STAGING_BYTES,
        ..CpuConfig::default()
    };
    let pool = CpuPool::new(config).expect("CPU pool");
    let (prepared, media, request_tokens, request_media) = pool
        .run(2 * IMAGE_COUNT * MEDIA_STAGING_BYTES, move |context| {
            let prompt = "<|vision_start|><|image_pad|><|vision_end|>".repeat(IMAGE_COUNT);
            let (prepared, media) = service.prepare(&assets, &prompt, &requests, context)?;
            let (request_tokens, request_media) =
                service.prepare_request(&assets, &prompt, &requests, context, None)?;
            Ok((prepared, media, request_tokens, request_media))
        })
        .await
        .expect("multi-image preparation");
    assert_eq!(request_tokens, prepared.tokens);
    assert_eq!(request_media, media);
    assert_eq!(media.len(), IMAGE_COUNT);
    for ((image, span), media) in prepared.images.iter().zip(&prepared.spans).zip(media) {
        let item = media.item.expect("prepared item");
        assert_eq!(item.grid, image.grid);
        assert_eq!(item.span, *span);
        assert_eq!(item.pixels.as_ref(), image.pixels.as_slice());
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for value in &image.pixels {
            for byte in value.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        for axis in <[usize; 3]>::from(image.grid) {
            for byte in u64::try_from(axis).unwrap().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(media.content_digest, format!("fnv1a64:{hash:016x}"));
    }
}
