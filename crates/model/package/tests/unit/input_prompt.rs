use super::*;
use crate::default_registry;
use crate::image::ImageProcessor;
use crate::image::RgbImage;
use crate::input::text::TextAssets;
use infer_core::ModelId;
use infer_ir::MediaItem;
use std::env::var;
use std::fs::read;
use std::iter::once;
use std::path::Path;
use std::slice::from_ref;
use std::sync::Arc;

const IMAGE: u32 = 248_056;
const VISION_START: u32 = 248_053;
const VISION_END: u32 = 248_054;

fn shipped() -> Result<ImportedModel> {
    let root = var("INFER_VISION_PACKAGE").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../../examples/qwen3.8-27b").into()
    });
    let config = read(Path::new(&root).join("config.json"))
        .map_err(|error| Error::invalid(error.to_string()))?;
    let provider = default_registry().resolve(&config)?;
    provider.import(ModelId::ONE, &config)
}

fn encoder(imported: &ImportedModel) -> &ModalityEncoder {
    require_encoder(imported, &Modality::Image).expect("image encoder")
}

#[test]
fn placeholder_runs_cover_every_merged_patch() -> Result<()> {
    let imported = shipped()?;
    let encoder = encoder(&imported);
    let images = [
        PromptImage {
            pixels: Vec::new(),
            grid: (1, 4, 4),
        },
        PromptImage {
            pixels: Vec::new(),
            grid: (1, 4, 6),
        },
    ];
    let counts = image_token_counts(&imported, &images)?;
    // 4x4 patches merge into 2x2 groups, 4x6 into 2x3.
    assert_eq!(counts, vec![4, 6]);
    let tokens = [
        VISION_START,
        IMAGE,
        VISION_END,
        7,
        VISION_START,
        IMAGE,
        VISION_END,
    ];
    let expanded = expand_placeholders(&tokens, &counts, IMAGE)?;
    assert_eq!(expanded.tokens.len(), tokens.len() + 8);
    assert_eq!(expanded.spans, vec![1..5, 8..14]);
    for span in &expanded.spans {
        assert!(expanded.tokens[span.clone()].iter().all(|id| *id == IMAGE));
    }
    assert_eq!(&expanded.tokens[1..2], &[IMAGE]);
    assert_eq!(encoder.out_hidden_size, 5120);
    Ok(())
}

/// The end-to-end prompt path: render, tokenize, preprocess and expand.
///
/// Skipped unless the real package (with its tokenizer) is configured.
#[test]
fn prepared_prompt_lines_up_tokens_and_media() -> Result<()> {
    let Ok(package) = var("INFER_VISION_PACKAGE") else {
        return Ok(());
    };
    let imported = shipped()?;
    let assets = TextAssets::open(&package, 4096)?;
    let processor = ImageProcessor::from_json(
        &read(Path::new(&package).join("preprocessor_config.json"))
            .map_err(|error| Error::invalid(error.to_string()))?,
    )?;
    let text = "<|vision_start|><|image_pad|><|vision_end|>describe";
    let image = RawImage {
        height: 4,
        width: 4,
        pixels: vec![128; 4 * 4 * 3],
    };
    let prepared = prepare(
        &assets,
        &processor,
        &imported,
        text,
        from_ref(&image),
        &Modality::Image,
    )?;
    let shared = prepare_shared(
        &assets,
        &processor,
        &imported,
        text,
        from_ref(&image),
        &Modality::Image,
    )?;
    assert_eq!(shared.tokens, prepared.tokens);
    assert_eq!(shared.spans, prepared.spans);
    assert_eq!(shared.images[0].grid, prepared.images[0].grid);
    assert_eq!(&*shared.images[0].pixels, prepared.images[0].pixels);
    let image_token = *placeholders(&imported, &Modality::Image)?
        .first()
        .ok_or_else(|| Error::invalid("image placeholder is missing"))?;
    assert_eq!(prepared.images.len(), 1);
    assert_eq!(prepared.spans.len(), 1);
    let span = prepared.spans[0].clone();
    let (start, end) = (span.start, span.end);
    assert!(
        prepared.tokens[start..end]
            .iter()
            .all(|id| *id == image_token)
    );
    // The placeholder itself is gone; only the expanded run remains.
    assert_eq!(
        prepared
            .tokens
            .iter()
            .filter(|id| **id == image_token)
            .count(),
        span.len()
    );
    let text_only = prepare(
        &assets,
        &processor,
        &imported,
        "describe",
        &[],
        &Modality::Image,
    )?;
    assert!(text_only.spans.is_empty() && text_only.images.is_empty());
    // A prompt whose text carries no placeholder must not silently drop the image.
    assert!(
        prepare(
            &assets,
            &processor,
            &imported,
            "describe",
            &[image],
            &Modality::Image
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn placements_follow_the_expanded_spans() -> Result<()> {
    let imported = shipped()?;
    let counts = [4usize, 6];
    let tokens = [
        VISION_START,
        IMAGE,
        VISION_END,
        7,
        VISION_START,
        IMAGE,
        VISION_END,
    ];
    let expanded = expand_placeholders(&tokens, &counts, IMAGE)?;
    let prepared = PreparedPrompt {
        tokens: expanded.tokens,
        images: vec![
            PromptImage {
                pixels: Vec::new(),
                grid: (1, 4, 4),
            },
            PromptImage {
                pixels: Vec::new(),
                grid: (1, 4, 6),
            },
        ],
        spans: expanded.spans,
    };
    let hidden = 3;
    let encodings = vec![vec![1.0f32; 4 * hidden], vec![2.0f32; 6 * hidden]];
    let placements =
        place_visual_embeddings(&prepared, &imported, &Modality::Image, &encodings, hidden)?;
    assert_eq!(placements.len(), 10);
    assert_eq!(placements[0].position, 1);
    assert_eq!(placements[0].embedding, vec![1.0; hidden]);
    assert_eq!(placements[4].position, 8);
    assert_eq!(placements[4].embedding, vec![2.0; hidden]);
    // A short encoding, a wrong width or a missing span must all fail loudly.
    assert!(
        place_visual_embeddings(
            &prepared,
            &imported,
            &Modality::Image,
            &[vec![1.0f32; 4 * hidden]],
            hidden
        )
        .is_err()
    );
    assert!(
        place_visual_embeddings(
            &prepared,
            &imported,
            &Modality::Image,
            &[vec![1.0f32; 4 * hidden], vec![2.0f32; 6 * hidden + 1]],
            hidden
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn placeholder_mismatch_is_an_explicit_error() {
    let counts = [4usize];
    let none = [VISION_START, VISION_END];
    assert!(expand_placeholders(&none, &counts, IMAGE).is_err());
    let two = [IMAGE, IMAGE];
    assert!(expand_placeholders(&two, &counts, IMAGE).is_err());
    assert!(expand_placeholders(&[IMAGE], &[], IMAGE).is_err());
    assert!(expand_placeholders(&[IMAGE], &[0], IMAGE).is_err());
}

#[test]
fn missing_encoder_is_unsupported_not_silent() -> Result<()> {
    let mut imported = shipped()?;
    imported.modalities.clear();
    let Err(error) = require_encoder(&imported, &Modality::Image) else {
        panic!("a model without an image encoder must fail");
    };
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
    assert!(error.message.contains("Image"));
    let images = [PromptImage {
        pixels: Vec::new(),
        grid: (1, 4, 4),
    }];
    assert!(image_token_counts(&imported, &images).is_err());
    assert!(placeholders(&imported, &Modality::Image).is_err());
    assert!(placeholders(&imported, &Modality::Audio).is_err());
    Ok(())
}

#[test]
fn square_grid_matches_the_visual_token_count() -> Result<()> {
    let imported = shipped()?;
    let encoder = encoder(&imported);
    assert_eq!(visual_tokens((2, 4, 4), encoder)?, 8);
    assert_eq!(visual_tokens((1, 4, 6), encoder)?, 6);
    // Nine patches cannot split into merge units of four.
    assert!(visual_tokens((1, 3, 3), encoder).is_err());
    Ok(())
}

#[test]
fn media_layout_exposes_every_span_and_the_next_rotary_coordinate() -> Result<()> {
    let imported = shipped()?;
    let counts = [4usize, 6];
    let tokens = [
        VISION_START,
        IMAGE,
        VISION_END,
        7,
        VISION_START,
        IMAGE,
        VISION_END,
    ];
    let expanded = expand_placeholders(&tokens, &counts, IMAGE)?;
    let prepared = PreparedPrompt {
        tokens: expanded.tokens,
        images: vec![
            PromptImage {
                pixels: Vec::new(),
                grid: (1, 4, 4),
            },
            PromptImage {
                pixels: Vec::new(),
                grid: (1, 4, 6),
            },
        ],
        spans: expanded.spans,
    };
    let layout = media_layout(&prepared, &imported, &Modality::Image)?;
    assert_eq!(layout.spans.len(), 2);
    assert_eq!(layout.spans[0].range, 1..5);
    assert_eq!(layout.spans[1].range, 8..14);
    assert_eq!(layout.spans[1].rotary.as_ref().map(Vec::len), Some(6));
    assert_eq!(layout.next_rotary, 10);
    let positions = layout
        .positions
        .as_ref()
        .ok_or_else(|| Error::invalid("expected three-axis positions"))?;
    assert_eq!(positions.len(), prepared.tokens.len());
    assert_eq!(positions[1], [1, 1, 1]);
    assert_eq!(positions[2], [1, 1, 2]);
    assert_eq!(positions[4], [1, 2, 2]);
    assert_eq!(positions[8], [6, 6, 6]);
    assert_eq!(positions[13], [6, 7, 8]);
    // A span whose tokens are not the modality placeholder must fail.
    let mut tampered = prepared;
    tampered.tokens[2] = VISION_END;
    assert!(media_layout(&tampered, &imported, &Modality::Image).is_err());
    Ok(())
}

#[test]
fn media_plan_layout_matches_prepared_layout_and_rejects_invalid_geometry() -> Result<()> {
    let mut imported = shipped()?;
    let expanded = expand_placeholders(
        &[
            VISION_START,
            IMAGE,
            VISION_END,
            7,
            VISION_START,
            IMAGE,
            VISION_END,
        ],
        &[4, 6],
        IMAGE,
    )?;
    let prepared = PreparedPrompt {
        tokens: expanded.tokens,
        images: vec![
            PromptImage {
                pixels: vec![1.0],
                grid: (1, 4, 4),
            },
            PromptImage {
                pixels: vec![2.0],
                grid: (1, 4, 6),
            },
        ],
        spans: expanded.spans,
    };
    let mut plan = MediaPlan {
        modality: Modality::Image,
        prompt: prepared.tokens.clone().into(),
        items: prepared
            .images
            .iter()
            .zip(&prepared.spans)
            .map(|(image, span)| MediaItem {
                modality: Modality::Image,
                pixels: Arc::from(image.pixels.as_slice()),
                grid: image.grid,
                span: span.clone(),
            })
            .collect(),
    };
    assert_eq!(
        media_plan_layout(&plan, &imported)?,
        media_layout(&prepared, &imported, &Modality::Image)?
    );
    imported.model.position.multimodal_sections.clear();
    assert_eq!(
        media_plan_layout(&plan, &imported)?,
        media_layout(&prepared, &imported, &Modality::Image)?
    );
    plan.items[1].span = 2..8;
    assert!(media_plan_layout(&plan, &imported).is_err());
    plan.items[1].span = prepared.spans[1].clone();
    plan.items[1].grid = (1, 4, 4);
    assert!(media_plan_layout(&plan, &imported).is_err());
    plan.items[1].grid = (1, 4, 6);
    plan.prompt = vec![VISION_START; prepared.tokens.len()].into();
    assert!(media_plan_layout(&plan, &imported).is_err());
    plan.items.clear();
    assert_eq!(
        media_plan_layout(&plan, &imported)?.next_rotary,
        prepared.tokens.len()
    );
    plan.prompt = vec![VISION_START, IMAGE, VISION_END].into();
    assert!(media_plan_layout(&plan, &imported).is_err());
    Ok(())
}

/// Video prompts go through the same alignment contract as images.
///
/// Skipped unless the real package (with its tokenizer and video processor) is configured.
#[test]
fn prepared_video_prompt_expands_temporal_patches() -> Result<()> {
    let Ok(package) = var("INFER_VISION_PACKAGE") else {
        return Ok(());
    };
    let imported = shipped()?;
    let assets = TextAssets::open(&package, 4096)?;
    let processor = VideoProcessor::from_json(
        &read(Path::new(&package).join("video_preprocessor_config.json"))
            .map_err(|error| Error::invalid(error.to_string()))?,
    )?;
    // Four 32x32 frames fill two temporal groups and a 2x2 spatial patch grid.
    let video = RawVideo {
        fps: 30.0,
        frames: (0..4)
            .map(|index| RgbImage {
                height: 32,
                width: 32,
                pixels: vec![u8::try_from(index * 60).unwrap_or(0); 32 * 32 * 3],
            })
            .collect(),
    };
    let text = "<|vision_start|><|video_pad|><|vision_end|>describe";
    let prepared = prepare_video(
        &assets,
        &processor,
        &imported,
        text,
        from_ref(&video),
        &Modality::Video,
    )?;
    let placeholder = placeholders(&imported, &Modality::Video)?[0];
    // Each temporal group is its own timestamped, marker-delimited media item.
    assert_eq!(prepared.images.len(), 2);
    assert_eq!(prepared.images[0].grid, (1, 2, 2));
    assert_eq!(prepared.spans.len(), 2);
    let vision_start = assets.encode("<|vision_start|>", false)?[0];
    let vision_end = assets.encode("<|vision_end|>", false)?[0];
    for span in &prepared.spans {
        assert_eq!(span.len(), 1);
        assert!(
            prepared.tokens[span.clone()]
                .iter()
                .all(|token| *token == placeholder)
        );
        assert_eq!(prepared.tokens[span.start - 1], vision_start);
        assert_eq!(prepared.tokens[span.end], vision_end);
    }
    let empty = prepare_video(
        &assets,
        &processor,
        &imported,
        "describe",
        &[],
        &Modality::Video,
    )?;
    assert!(empty.spans.is_empty() && empty.images.is_empty());
    // Text without a video placeholder must not silently drop the clip.
    assert!(
        prepare_video(
            &assets,
            &processor,
            &imported,
            "describe",
            from_ref(&video),
            &Modality::Video
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn foreign_modality_placeholders_are_rejected() -> Result<()> {
    let imported = shipped()?;
    let video = placeholders(&imported, &Modality::Video)?[0];
    assert!(
        reject_foreign_placeholders(
            &imported,
            &[VISION_START, video, VISION_END],
            &Modality::Image
        )
        .is_err()
    );
    assert!(
        reject_foreign_placeholders(
            &imported,
            &[VISION_START, IMAGE, VISION_END],
            &Modality::Video
        )
        .is_err()
    );
    reject_foreign_placeholders(
        &imported,
        &[VISION_START, IMAGE, VISION_END],
        &Modality::Image,
    )?;
    Ok(())
}

#[test]
fn media_layout_rejects_uncovered_or_misaligned_spans() -> Result<()> {
    let imported = shipped()?;
    // A placeholder without media must not reach the model as ordinary text.
    let stray = PreparedPrompt {
        tokens: vec![VISION_START, IMAGE, VISION_END],
        images: Vec::new(),
        spans: Vec::new(),
    };
    assert!(media_layout(&stray, &imported, &Modality::Image).is_err());
    // Two four-token spans overlap by two tokens.
    let overlap = PreparedPrompt {
        tokens: vec![IMAGE; 6],
        images: vec![
            PromptImage {
                pixels: Vec::new(),
                grid: (1, 4, 4),
            },
            PromptImage {
                pixels: Vec::new(),
                grid: (1, 4, 4),
            },
        ],
        spans: vec![0..4, 2..6],
    };
    assert!(media_layout(&overlap, &imported, &Modality::Image).is_err());
    // A 1x4x6 grid merges into six tokens, not four.
    let short = PreparedPrompt {
        tokens: vec![IMAGE; 4],
        images: vec![PromptImage {
            pixels: Vec::new(),
            grid: (1, 4, 6),
        }],
        spans: once(0..4).collect(),
    };
    assert!(media_layout(&short, &imported, &Modality::Image).is_err());
    Ok(())
}

/// Multiple clips keep their placeholder order and each contributes one span per group.
///
/// Skipped unless the real package (with its tokenizer and video processor) is configured.
#[test]
fn prepared_video_prompt_orders_multiple_clips() -> Result<()> {
    let Ok(package) = var("INFER_VISION_PACKAGE") else {
        return Ok(());
    };
    let imported = shipped()?;
    let assets = TextAssets::open(&package, 4096)?;
    let processor = VideoProcessor::from_json(
        &read(Path::new(&package).join("video_preprocessor_config.json"))
            .map_err(|error| Error::invalid(error.to_string()))?,
    )?;
    let clip = |level: u8| RawVideo {
        fps: 30.0,
        frames: (0..4)
            .map(|_| RgbImage {
                height: 32,
                width: 32,
                pixels: vec![level; 32 * 32 * 3],
            })
            .collect(),
    };
    let text = "<|vision_start|><|video_pad|><|vision_end|> and \
                <|vision_start|><|video_pad|><|vision_end|>";
    let prepared = prepare_video(
        &assets,
        &processor,
        &imported,
        text,
        &[clip(10), clip(200)],
        &Modality::Video,
    )?;
    assert_eq!(prepared.images.len(), 4);
    assert_eq!(prepared.spans.len(), 4);
    assert!(
        prepared
            .spans
            .windows(2)
            .all(|pair| pair[0].end <= pair[1].start)
    );
    // The first clip's first group must precede the second clip's first group.
    assert_ne!(prepared.images[0].pixels, prepared.images[2].pixels);
    Ok(())
}
