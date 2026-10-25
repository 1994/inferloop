use super::{MEDIA_STAGING_BYTES, load_image};
use infer_models::MediaLimits;

const ONE_PIXEL_PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

#[test]
fn data_uri_png_is_decoded_into_rgb8() {
    let image = load_image(&format!("data:image/png;base64,{ONE_PIXEL_PNG}")).expect("tiny PNG");
    assert_eq!((image.width, image.height), (1, 1));
    assert_eq!(image.pixels.len(), 3);
}

#[test]
fn malformed_and_oversized_data_uris_are_rejected() {
    assert!(load_image("data:image/png;base64,!!!").is_err());
    let oversized = "A".repeat(MEDIA_STAGING_BYTES + 1);
    assert!(load_image(&format!("data:image/png;base64,{oversized}")).is_err());
    assert!(load_image("ftp://example.invalid/a.png").is_err());
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
