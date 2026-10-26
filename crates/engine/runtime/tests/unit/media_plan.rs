use infer_core::MediaId;
use infer_ir::{MediaInput, MediaItem, MediaPlan, Modality};
use std::ops::Range;
use std::sync::Arc;

fn item(span: Range<usize>) -> MediaInput {
    MediaInput {
        id: MediaId::ONE,
        modality: Modality::Image,
        content_digest: "test".into(),
        item: Some(MediaItem {
            modality: Modality::Image,
            pixels: Arc::from([0.0f32, 1.0, 2.0, 3.0]),
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
