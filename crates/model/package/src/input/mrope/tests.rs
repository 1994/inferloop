use super::*;
use crate::PromptImage;
use std::iter::once;

#[test]
fn frequency_axes_match_interleaved_and_contiguous_sections() {
    let mut spec = PositionSpec {
        rope_theta: 5_000_000.0,
        rotary_fraction: 1.0,
        multimodal_sections: vec![24, 20, 20],
        interleaved: true,
    };
    let actual = axes(&spec, 64).unwrap();
    for (index, axis) in actual.iter().enumerate() {
        assert_eq!(*axis, if index < 60 { [0, 1, 2][index % 3] } else { 0 });
    }
    spec.interleaved = false;
    assert_eq!(
        axes(&spec, 64).unwrap(),
        [vec![0; 24], vec![1; 20], vec![2; 20]].concat()
    );
    assert!(axes(&spec, 32).is_err());
    spec.multimodal_sections.clear();
    assert_eq!(axes(&spec, 32).unwrap(), vec![0; 32]);
}

#[test]
fn image_axes_and_decode_position_are_independent_of_kv_length() {
    let prompt = PreparedPrompt {
        tokens: vec![0; 95],
        spans: once(4..81).collect(),
        images: vec![PromptImage {
            pixels: vec![],
            grid: (1, 14, 22),
        }],
    };
    let expected = image_positions(&prompt, 2).unwrap();
    assert_eq!(expected[3], [3, 3, 3]);
    assert_eq!(expected[4], [4, 4, 4]);
    assert_eq!(expected[80], [4, 10, 14]);
    assert_eq!(expected[81], [15, 15, 15]);
    assert_eq!(expected[94], [28, 28, 28]);
    let media: Vec<_> = (4..81)
        .map(|position| VisualPlacement {
            position,
            embedding: vec![],
            rotary_position: Some(expected[position]),
        })
        .collect();
    let (actual, next) = placed_positions(95, &media).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(next, 29);
    assert_eq!(placed_positions(95, &[]).unwrap().1, 95);
}

#[test]
fn media_geometry_is_validated_before_positions_are_emitted() {
    let mut prompt = PreparedPrompt {
        tokens: vec![0; 10],
        spans: vec![2..6, 7..9],
        images: vec![
            PromptImage {
                pixels: vec![],
                grid: (1, 4, 4),
            },
            PromptImage {
                pixels: vec![],
                grid: (1, 2, 4),
            },
        ],
    };
    let positions = image_positions(&prompt, 2).unwrap();
    assert_eq!(positions[6], [4, 4, 4]);
    assert_eq!(positions[7], [5, 5, 5]);
    assert_eq!(positions[9], [7, 7, 7]);
    prompt.spans[1] = 4..6;
    assert!(image_positions(&prompt, 2).is_err());
    assert!(image_positions(&prompt, 0).is_err());
}
