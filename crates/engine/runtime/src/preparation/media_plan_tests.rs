use infer_core::MediaId;
use infer_ir::{
    ImageInput, ImageNormalization, ImagePatchLayout, ImageTransform, MediaInput, MediaItem,
    MediaPixels, MediaPlan, Modality,
};
use std::ops::Range;
use std::sync::Arc;

fn item(span: Range<usize>) -> MediaInput {
    MediaInput {
        id: MediaId::ONE,
        modality: Modality::Image,
        content_digest: "test".into(),
        item: Some(MediaItem {
            modality: Modality::Image,
            pixels: Arc::<[f32]>::from([0.0f32, 1.0, 2.0, 3.0]).into(),
            grid: (1, 1, 1),
            span,
        }),
    }
}

#[test]
fn media_plan_requires_ordered_disjoint_payload_spans() {
    let prompt = vec![1, 2, 3, 4].into();
    let plan = MediaPlan::from_sequence(&prompt, &[item(0..2), item(2..4)])
        .expect("valid media plan")
        .expect("media plan exists");
    assert_eq!(plan.media_tokens(), 4);
    assert!(MediaPlan::from_sequence(&prompt, &[item(0..2), item(1..3)]).is_err());
    assert!(MediaPlan::from_sequence(&prompt, &[item(0..5)]).is_err());
    let missing = MediaInput {
        item: None,
        ..item(0..1)
    };
    assert!(MediaPlan::from_sequence(&prompt, &[missing]).is_err());
}

#[test]
fn media_plan_rejects_mixed_modalities() {
    let prompt = vec![1, 2].into();
    let mut video = item(1..2);
    video.modality = Modality::Video;
    if let Some(item) = video.item.as_mut() {
        item.modality = Modality::Video;
    }
    assert!(MediaPlan::from_sequence(&prompt, &[item(0..1), video]).is_err());
}

#[test]
fn media_plan_retains_raw_rgb_and_checks_its_grid_and_modality() {
    let layout = ImagePatchLayout::new(16, 2, 2).unwrap();
    let norm = ImageNormalization::new([0.5; 3], [0.5; 3], 1.0 / 255.0).unwrap();
    let transform = ImageTransform::new([32, 32], [32, 32], layout, norm).unwrap();
    let mut input = item(0..1);
    let image = ImageInput::new(transform, vec![128; 32 * 32 * 3]).unwrap();
    let retained = image.retained_bytes();
    let raw = input.item.as_mut().unwrap();
    raw.pixels = MediaPixels::Image(image);
    raw.grid = transform.grid();
    let prompt = vec![1].into();
    let plan = MediaPlan::from_sequence(&prompt, &[input.clone()])
        .unwrap()
        .unwrap();
    assert!(plan.retained_bytes() >= retained);
    input.item.as_mut().unwrap().grid = (1, 4, 4);
    assert!(MediaPlan::from_sequence(&prompt, &[input.clone()]).is_err());
    input.modality = Modality::Video;
    let raw = input.item.as_mut().unwrap();
    raw.modality = Modality::Video;
    raw.grid = transform.grid();
    assert!(MediaPlan::from_sequence(&prompt, &[input]).is_err());
}
