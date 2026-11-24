use super::{ImageProcessingOptions, ResizeMode};
use infer_core::Result;
use infer_models::ImageProcessor;

fn processor() -> ImageProcessor {
    ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 32 * 32,
        max_pixels: 1024 * 1024,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    }
}

#[test]
fn request_options_override_deployment_and_scope() -> Result<()> {
    let global = ImageProcessingOptions::parse(
        r#"{"size":{"shortest_edge":4096,"longest_edge":262144},"do_resize":false}"#,
    )?;
    let request = ImageProcessingOptions::parse(
        r#"{"images_kwargs":{"max_pixels":65536,"do_resize":true}}"#,
    )?;
    let effective = global.resolve(Some(&request), &processor())?;
    assert_eq!(effective.processor.min_pixels, 4096);
    assert_eq!(effective.processor.max_pixels, 65536);
    assert!(effective.resize == ResizeMode::Smart);
    assert!(effective.transform(512, 512)?.target()[0] < 512);
    Ok(())
}

#[test]
fn disabled_resize_requires_a_valid_patch_grid() -> Result<()> {
    let options = ImageProcessingOptions::parse(r#"{"do_resize":false}"#)?;
    let effective = options.resolve(None, &processor())?;
    assert_eq!(effective.transform(64, 96)?.target(), [64, 96]);
    assert!(effective.transform(65, 96).is_err());
    Ok(())
}

#[test]
fn malformed_or_invalid_resize_options_are_rejected() {
    for input in [
        r#"{"min_pixels":0}"#,
        r#"{"min_pixels":100,"max_pixels":10}"#,
        r#"{"size":{"shortest_edge":10,"longest_edge":0}}"#,
        r#"{"unknown":true}"#,
        "[]",
    ] {
        assert!(ImageProcessingOptions::parse(input).is_err(), "{input}");
    }
}

#[test]
fn huge_resize_options_fail_checked_geometry_without_a_fixed_pixel_cap() -> Result<()> {
    let json = format!(
        r#"{{"min_pixels":{},"max_pixels":{}}}"#,
        usize::MAX,
        usize::MAX
    );
    let options = ImageProcessingOptions::parse(&json)?;
    let effective = options.resolve(None, &processor())?;
    assert!(effective.transform(32, 32).is_err());
    Ok(())
}
