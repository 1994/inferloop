//! Multimodal prompt assembly: media placeholders, expansion and encoder gating.
//!
//! The chat template emits one placeholder token per image. The processor then expands each
//! placeholder into one token per *merged* visual patch, so the token stream and the vision
//! embeddings line up one-to-one. Every mismatch is an error: a prompt that cannot be aligned
//! with its media must never fall back to the text-only path.
use crate::image::ImageProcessor;
use crate::image::PreprocessedImage;
use crate::input::text::TextAssets;
use crate::mrope::ROTARY_AXES;
use crate::mrope::image_positions;
use crate::video::{PreprocessedVideo, RawVideo, VideoProcessor};
use crate::vision::merge_unit;
use infer_core::{Error, Result};
use infer_ir::Modality;
use infer_spi::{ImportedModel, ModalityEncoder};
use std::iter::repeat_n;
use std::mem::take;
use std::ops::Range;

/// One preprocessed image attached to a prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptImage {
    /// Patch vectors from [`crate::ImageProcessor::preprocess`].
    pub pixels: Vec<f32>,
    /// `(temporal, height, width)` patch grid.
    pub grid: (usize, usize, usize),
}

impl PromptImage {
    /// Wraps one preprocessed image.
    #[must_use]
    pub fn new(processed: PreprocessedImage) -> Self {
        Self {
            pixels: processed.pixels,
            grid: processed.grid,
        }
    }
}

/// Number of merged visual tokens an image contributes.
/// # Errors
/// Rejects grids that do not divide by the merge unit or overflow.
pub fn visual_tokens(grid: (usize, usize, usize), encoder: &ModalityEncoder) -> Result<usize> {
    let unit = merge_unit(encoder)?;
    let patches = grid
        .0
        .checked_mul(grid.1)
        .and_then(|value| value.checked_mul(grid.2))
        .ok_or_else(|| Error::invalid("patch grid overflow"))?;
    if !patches.is_multiple_of(unit) {
        return Err(Error::invalid(
            "patch grid does not divide by the merge unit",
        ));
    }
    Ok(patches / unit)
}

/// Encoder of one modality, or an explicit failure when the model has none.
///
/// # Errors
/// Returns an unsupported error naming the modality when the family declares no encoder.
pub fn require_encoder<'a>(
    imported: &'a ImportedModel,
    modality: &Modality,
) -> Result<&'a ModalityEncoder> {
    imported
        .modalities
        .iter()
        .find(|plan| &plan.modality == modality)
        .and_then(|plan| plan.encoder.as_ref())
        .ok_or_else(|| {
            Error::unsupported(format!(
                "model declares no {modality:?} encoder, yet the request carries {modality:?} media"
            ))
        })
}

/// Placeholder token ids of one modality.
/// # Errors
/// Returns an unsupported error naming the modality when the family declares no placeholders.
pub fn placeholders<'a>(imported: &'a ImportedModel, modality: &Modality) -> Result<&'a [u32]> {
    imported
        .modalities
        .iter()
        .find(|plan| &plan.modality == modality)
        .map(|plan| plan.placeholder_tokens.as_slice())
        .filter(|tokens| !tokens.is_empty())
        .ok_or_else(|| {
            Error::unsupported(format!("model declares no {modality:?} placeholder tokens"))
        })
}

/// Reject placeholders of a modality the caller is not preparing.
///
/// A prompt may carry one modality per preparation call. Without this check a video token in an
/// image prompt would survive expansion and reach the model as ordinary text.
///
/// # Errors
/// Returns an unsupported error naming both modalities when a foreign placeholder is present.
fn reject_foreign_placeholders(
    imported: &ImportedModel,
    tokens: &[u32],
    modality: &Modality,
) -> Result<()> {
    for plan in &imported.modalities {
        if &plan.modality != modality
            && plan
                .placeholder_tokens
                .iter()
                .any(|token| tokens.contains(token))
        {
            return Err(Error::unsupported(format!(
                "prompt carries {:?} media while preparing {modality:?}",
                plan.modality
            )));
        }
    }
    Ok(())
}

/// Result of expanding media placeholders in a token stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedPrompt {
    /// Token stream with every placeholder replaced by its media's token run.
    pub tokens: Vec<u32>,
    /// Token range each image's embeddings occupy, in media order.
    pub spans: Vec<Range<usize>>,
}

/// Expand every media placeholder into one token per merged patch.
///
/// `counts[i]` is the token count of the `i`-th media item. The stream must hold exactly
/// `counts.len()` placeholders: a different number means the prompt and its media disagree.
///
/// # Errors
/// Rejects a placeholder count that disagrees with the media count, an empty media list or
/// overflowing token counts.
pub fn expand_placeholders(
    tokens: &[u32],
    counts: &[usize],
    placeholder: u32,
) -> Result<ExpandedPrompt> {
    if counts.is_empty() {
        return Err(Error::invalid("no media to expand"));
    }
    let found = tokens.iter().filter(|token| **token == placeholder).count();
    if found != counts.len() {
        return Err(Error::invalid(format!(
            "prompt holds {found} media placeholders for {} media items",
            counts.len()
        )));
    }
    let capacity = tokens.len()
        + counts
            .iter()
            .try_fold(0usize, |sum, count| sum.checked_add(*count))
            .ok_or_else(|| Error::invalid("expanded prompt overflow"))?;
    let mut expanded = Vec::with_capacity(capacity);
    let mut spans = Vec::with_capacity(counts.len());
    let mut media = counts.iter();
    for token in tokens {
        if *token != placeholder {
            expanded.push(*token);
            continue;
        }
        let count = *media
            .next()
            .ok_or_else(|| Error::invariant("media placeholders were counted inconsistently"))?;
        if count == 0 {
            return Err(Error::invalid("media contributes no visual tokens"));
        }
        let start = expanded.len();
        expanded.extend(repeat_n(placeholder, count));
        spans.push(start..expanded.len());
    }
    Ok(ExpandedPrompt {
        tokens: expanded,
        spans,
    })
}

/// One media item aligned with the prompt span it occupies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaSpan {
    /// Token range the item's embeddings replace.
    pub range: Range<usize>,
    /// Absolute three-axis rotary coordinate of every token in the span; absent for 1-D `RoPE`.
    pub rotary: Option<Vec<[usize; ROTARY_AXES]>>,
}

/// Media alignment of a prepared prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaLayout {
    /// Media spans, in placeholder order.
    pub spans: Vec<MediaSpan>,
    /// Next rotary coordinate after the prompt: where the first decoded token starts.
    pub next_rotary: usize,
    /// Three-axis coordinate of every prompt token; absent for 1-D `RoPE`.
    pub positions: Option<Vec<[usize; ROTARY_AXES]>>,
}

/// Align every media span with the placeholder tokens and rotary geometry of the model.
///
/// The embeddings themselves are validated and placed by [`place_visual_embeddings`]; callers
/// that already hold device tensors use this narrower contract to avoid copying them to the host.
///
/// # Errors
/// Rejects inconsistent span/media counts, a span that does not hold the modality placeholder, a
/// model without declared placeholders or a geometry that disagrees with the encoder.
pub fn media_layout(
    prepared: &PreparedPrompt,
    imported: &ImportedModel,
    modality: &Modality,
) -> Result<MediaLayout> {
    if prepared.spans.len() != prepared.images.len() {
        return Err(Error::invalid(
            "prepared prompt holds inconsistent media spans",
        ));
    }
    if prepared.spans.is_empty() {
        // A prompt that carries placeholders but no media would run them as ordinary text.
        if let Ok(tokens) = placeholders(imported, modality)
            && let Some(placeholder) = tokens.first()
            && prepared.tokens.contains(placeholder)
        {
            return Err(Error::invalid(
                "prompt holds media placeholders without media",
            ));
        }
        return Ok(MediaLayout {
            spans: Vec::new(),
            next_rotary: prepared.tokens.len(),
            positions: None,
        });
    }
    let encoder = require_encoder(imported, modality)?;
    let merge = encoder.spatial_merge_size;
    if merge == 0 {
        return Err(Error::invalid("media encoder merge size must be nonzero"));
    }
    let placeholder = *placeholders(imported, modality)?
        .first()
        .ok_or_else(|| Error::invalid("media modality declares no placeholder token"))?;
    let (mut spans, expanded) = validate_spans(prepared, placeholder, merge)?;
    let declared = prepared
        .tokens
        .iter()
        .filter(|token| **token == placeholder)
        .count();
    if declared != expanded {
        return Err(Error::invalid(format!(
            "prompt holds {declared} media placeholders for {expanded} aligned tokens"
        )));
    }
    let positions = if imported.model.position.multimodal_sections.is_empty() {
        None
    } else {
        Some(image_positions(prepared, merge)?)
    };
    if let Some(positions) = &positions {
        for span in &mut spans {
            span.rotary = Some(positions[span.range.clone()].to_vec());
        }
    }
    let next_rotary = positions
        .as_ref()
        .map_or(prepared.tokens.len(), |positions| {
            positions
                .iter()
                .filter_map(|coordinates| coordinates.iter().max().copied())
                .max()
                .unwrap_or(0)
                .saturating_add(1)
        });
    Ok(MediaLayout {
        spans,
        next_rotary,
        positions,
    })
}

/// Validate media spans against their grids and placeholder tokens.
fn validate_spans(
    prepared: &PreparedPrompt,
    placeholder: u32,
    merge: usize,
) -> Result<(Vec<MediaSpan>, usize)> {
    let mut spans = Vec::with_capacity(prepared.spans.len());
    let mut expanded = 0usize;
    let mut previous_end = 0usize;
    for (span, image) in prepared.spans.iter().zip(&prepared.images) {
        if span.start < previous_end || span.start >= span.end || span.end > prepared.tokens.len() {
            return Err(Error::invalid(
                "media spans must be ordered, disjoint and inside the prompt",
            ));
        }
        if prepared.tokens[span.clone()]
            .iter()
            .any(|token| *token != placeholder)
        {
            return Err(Error::invalid(
                "media span does not hold the modality placeholder token",
            ));
        }
        let (temporal, height, width) = image.grid;
        if temporal == 0
            || height == 0
            || width == 0
            || !height.is_multiple_of(merge)
            || !width.is_multiple_of(merge)
        {
            return Err(Error::invalid("media grid disagrees with the encoder"));
        }
        let expected = temporal
            .checked_mul(height / merge)
            .and_then(|value| value.checked_mul(width / merge))
            .ok_or_else(|| Error::invalid("media token count overflow"))?;
        if expected != span.len() {
            return Err(Error::invalid(format!(
                "media span holds {} tokens for a grid of {expected}",
                span.len()
            )));
        }
        expanded = expanded
            .checked_add(span.len())
            .ok_or_else(|| Error::invalid("media token count overflow"))?;
        previous_end = span.end;
        spans.push(MediaSpan {
            range: span.clone(),
            rotary: None,
        });
    }
    Ok((spans, expanded))
}

/// Visual token counts of every image, validated against the declared encoder.
/// # Errors
/// Rejects a model without an image encoder or a grid that disagrees with the merge unit.
pub fn image_token_counts(imported: &ImportedModel, images: &[PromptImage]) -> Result<Vec<usize>> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    let encoder = require_encoder(imported, &Modality::Image)?;
    images
        .iter()
        .map(|image| visual_tokens(image.grid, encoder))
        .collect()
}

/// One decoded image attached to a prompt.
///
/// Decoding compressed formats is deliberately out of scope: the caller hands over RGB8 bytes and
/// their dimensions, so this crate stays free of image codecs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawImage {
    pub height: usize,
    pub width: usize,
    /// `height × width × 3` bytes, red first.
    pub pixels: Vec<u8>,
}

/// A prompt with its media preprocessed and its placeholders expanded.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedPrompt {
    /// Token stream the model consumes.
    pub tokens: Vec<u32>,
    /// Preprocessed visual media (images or video clips), in placeholder order.
    pub images: Vec<PromptImage>,
    /// Token range each image's embeddings replace, in the same order.
    pub spans: Vec<Range<usize>>,
}

/// Tokenize `text`, preprocess `images` and expand every placeholder.
///
/// Text-only prompts take the ordinary path. A prompt carrying media fails loudly when the model
/// declares no encoder, no placeholder, or a placeholder count that disagrees with the media —
/// there is no silent fall back to interpreting the placeholders as text.
///
/// # Errors
/// Rejects a missing encoder or placeholder, a placeholder/media mismatch, malformed image buffers
/// or a preprocessing failure.
pub fn prepare(
    assets: &TextAssets,
    processor: &ImageProcessor,
    imported: &ImportedModel,
    text: &str,
    images: &[RawImage],
    modality: &Modality,
) -> Result<PreparedPrompt> {
    let tokens = assets.encode(text, false)?;
    reject_foreign_placeholders(imported, &tokens, modality)?;
    if images.is_empty() {
        return Ok(PreparedPrompt {
            tokens,
            images: Vec::new(),
            spans: Vec::new(),
        });
    }
    let encoder = require_encoder(imported, modality)?;
    let placeholder = *placeholders(imported, modality)?
        .first()
        .ok_or_else(|| Error::invalid("media modality declares no placeholder token"))?;
    let mut prepared = Vec::with_capacity(images.len());
    let mut counts = Vec::with_capacity(images.len());
    for image in images {
        let processed = processor.preprocess_rgb(image.height, image.width, &image.pixels)?;
        counts.push(visual_tokens(processed.grid, encoder)?);
        prepared.push(PromptImage::new(processed));
    }
    let expanded = expand_placeholders(&tokens, &counts, placeholder)?;
    Ok(PreparedPrompt {
        tokens: expanded.tokens,
        images: prepared,
        spans: expanded.spans,
    })
}

/// Tokenize a video prompt, preprocess the clips and expand every placeholder.
///
/// The marker and timestamp layout is the Qwen3-VL processor contract
/// (`<t seconds><|vision_start|>pads<|vision_end|>` per temporal group). A video family with a
/// different prompt format needs its own assembler rather than a silent reinterpretation.
/// Clips go through the same alignment contract as images: a missing encoder, a missing
/// placeholder or a placeholder count that disagrees with the clips is an error, never a silent
/// fall back to text.
///
/// # Errors
/// Rejects a missing encoder or placeholder, a placeholder/media mismatch, malformed frames or a
/// preprocessing failure.
pub fn prepare_video(
    assets: &TextAssets,
    processor: &VideoProcessor,
    imported: &ImportedModel,
    text: &str,
    videos: &[RawVideo],
    modality: &Modality,
) -> Result<PreparedPrompt> {
    prepare_video_parts(
        assets,
        processor,
        imported,
        text,
        videos
            .iter()
            .map(|video| processor.preprocess(video).map(|part| vec![part])),
        modality,
    )
}

/// Assemble videos whose temporal groups have already been processed independently.
///
/// The outer vector is in prompt order and each inner vector is in clip timestamp order. A part
/// may contain one or more complete temporal groups and is consumed without copying single-group
/// patch buffers.
///
/// # Errors
/// Rejects a missing encoder or placeholder, mismatched media counts or malformed group geometry.
pub fn prepare_preprocessed_video(
    assets: &TextAssets,
    processor: &VideoProcessor,
    imported: &ImportedModel,
    text: &str,
    videos: Vec<Vec<PreprocessedVideo>>,
    modality: &Modality,
) -> Result<PreparedPrompt> {
    prepare_video_parts(
        assets,
        processor,
        imported,
        text,
        videos.into_iter().map(Ok),
        modality,
    )
}

fn prepare_video_parts(
    assets: &TextAssets,
    processor: &VideoProcessor,
    imported: &ImportedModel,
    text: &str,
    videos: impl ExactSizeIterator<Item = Result<Vec<PreprocessedVideo>>>,
    modality: &Modality,
) -> Result<PreparedPrompt> {
    let tokens = assets.encode(text, false)?;
    reject_foreign_placeholders(imported, &tokens, modality)?;
    if videos.len() == 0 {
        return Ok(PreparedPrompt {
            tokens,
            images: Vec::new(),
            spans: Vec::new(),
        });
    }
    let encoder = require_encoder(imported, modality)?;
    let merge = encoder.spatial_merge_size;
    if processor.patch_size != encoder.patch_size
        || processor.temporal_patch_size != encoder.temporal_patch_size
        || processor.merge_size != merge
    {
        return Err(Error::invalid(
            "video processor geometry disagrees with the declared encoder",
        ));
    }
    let placeholder = *placeholders(imported, modality)?
        .first()
        .ok_or_else(|| Error::invalid("media modality declares no placeholder token"))?;
    let vision_start = single_token(assets, "<|vision_start|>")?;
    let vision_end = single_token(assets, "<|vision_end|>")?;
    let mut prepared = Vec::new();
    let mut segments = Vec::with_capacity(videos.len());
    for video in videos {
        let parts = video?;
        if parts.is_empty() {
            return Err(Error::invalid("video has no processed temporal groups"));
        }
        let mut group_tokens = Vec::new();
        let mut geometry = None;
        let mut last_time = None;
        for processed in parts {
            let (_, height, width) = processed.image.grid;
            if geometry.is_some_and(|value| value != (height, width))
                || processed
                    .timestamps
                    .windows(2)
                    .any(|times| times[0] > times[1])
                || last_time
                    .zip(processed.timestamps.first())
                    .is_some_and(|(last, first)| last > *first)
            {
                return Err(Error::invalid(
                    "video groups are inconsistent or out of order",
                ));
            }
            geometry = Some((height, width));
            last_time = processed.timestamps.last().copied();
            append_video_part(
                assets,
                processor,
                merge,
                processed,
                &mut prepared,
                &mut group_tokens,
            )?;
        }
        segments.push(group_tokens);
    }
    let (tokens, spans) =
        expand_video_tokens(&tokens, placeholder, vision_start, vision_end, &segments)?;
    Ok(PreparedPrompt {
        tokens,
        images: prepared,
        spans,
    })
}

fn append_video_part(
    assets: &TextAssets,
    processor: &VideoProcessor,
    merge: usize,
    mut processed: PreprocessedVideo,
    prepared: &mut Vec<PromptImage>,
    group_tokens: &mut Vec<VideoSegment>,
) -> Result<()> {
    let (groups, height, width) = processed.image.grid;
    if merge == 0
        || height == 0
        || width == 0
        || !height.is_multiple_of(merge)
        || !width.is_multiple_of(merge)
    {
        return Err(Error::invalid("video patch grid"));
    }
    let frame_seqlen = (height / merge)
        .checked_mul(width / merge)
        .ok_or_else(|| Error::invalid("video frame sequence overflow"))?;
    let chunk = height
        .checked_mul(width)
        .and_then(|patches| patches.checked_mul(processor.patch_width()))
        .ok_or_else(|| Error::invalid("video group width overflow"))?;
    let expected_pixels = groups
        .checked_mul(chunk)
        .ok_or_else(|| Error::invalid("video group split overflow"))?;
    if groups == 0
        || processed.image.pixels.len() != expected_pixels
        || processed.timestamps.len() != groups
        || processed.timestamps.iter().any(|t| !t.is_finite())
    {
        return Err(Error::invalid("video preprocessing split"));
    }
    for (group, seconds) in processed.timestamps.into_iter().enumerate() {
        let start = group * chunk;
        let pixels = if groups == 1 {
            take(&mut processed.image.pixels)
        } else {
            processed.image.pixels[start..start + chunk].to_vec()
        };
        prepared.push(PromptImage {
            pixels,
            grid: (1, height, width),
        });
        let timestamp = assets.encode(&format!("<{seconds:.1} seconds>"), false)?;
        if timestamp.is_empty() {
            return Err(Error::invalid("video timestamp tokenization"));
        }
        group_tokens.push(VideoSegment {
            timestamp,
            count: frame_seqlen,
        });
    }
    Ok(())
}

/// One timestamped segment of a video, in prompt order.
struct VideoSegment {
    /// `<t seconds>` tokenized with the package tokenizer.
    timestamp: Vec<u32>,
    /// Video placeholder tokens this segment covers.
    count: usize,
}

/// Splice each video placeholder into timestamped, marker-delimited segments.
///
/// Qwen3-VL's processor replaces one `<|video_pad|>` with one
/// `<t seconds><|vision_start|>pads<|vision_end|>` run per temporal group. The outer
/// `vision_start`/`vision_end` the caller wrote stay in place, exactly like the reference
/// processor, and the prompt's token stream therefore matches the checkpoint's training format.
fn expand_video_tokens(
    tokens: &[u32],
    placeholder: u32,
    vision_start: u32,
    vision_end: u32,
    videos: &[Vec<VideoSegment>],
) -> Result<(Vec<u32>, Vec<Range<usize>>)> {
    let found = tokens.iter().filter(|token| **token == placeholder).count();
    if found != videos.len() {
        return Err(Error::invalid(format!(
            "prompt holds {found} video placeholders for {} clips",
            videos.len()
        )));
    }
    let mut expanded = Vec::with_capacity(tokens.len());
    let mut spans = Vec::new();
    let mut videos = videos.iter();
    for token in tokens {
        if *token != placeholder {
            expanded.push(*token);
            continue;
        }
        let segments = videos
            .next()
            .ok_or_else(|| Error::invariant("video placeholders were counted inconsistently"))?;
        for segment in segments {
            if segment.count == 0 {
                return Err(Error::invalid("video segment contributes no tokens"));
            }
            expanded.extend_from_slice(&segment.timestamp);
            expanded.push(vision_start);
            let start = expanded.len();
            expanded.extend(repeat_n(placeholder, segment.count));
            spans.push(start..expanded.len());
            expanded.push(vision_end);
        }
    }
    Ok((expanded, spans))
}

/// Tokenize a single literal that must map to exactly one token.
fn single_token(assets: &TextAssets, literal: &str) -> Result<u32> {
    let ids = assets.encode(literal, false)?;
    match ids.as_slice() {
        [token] => Ok(*token),
        _ => Err(Error::invalid(format!(
            "{literal} does not tokenize to one token"
        ))),
    }
}

/// One visual embedding destined for a prompt position.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualPlacement {
    /// Token position the embedding replaces.
    pub position: usize,
    /// Hidden state of that visual token.
    pub embedding: Vec<f32>,
    /// Absolute three-axis rotary coordinate; absent for ordinary one-dimensional `RoPE`.
    pub rotary_position: Option<[usize; ROTARY_AXES]>,
}

/// Align encoded media with the expanded prompt.
///
/// `embeddings[i]` is the flattened `tokens × hidden` output of the `i`-th media item's encoder.
/// Every span must hold exactly the placeholder tokens of that item and every embedding must match
/// its span length: a mismatch would silently shift the media relative to the text.
///
/// # Errors
/// Rejects a media/embedding count mismatch, a wrong embedding width, a span that does not hold the
/// placeholder token, or a model without declared placeholders.
pub fn place_visual_embeddings(
    prepared: &PreparedPrompt,
    imported: &ImportedModel,
    modality: &Modality,
    embeddings: &[Vec<f32>],
    hidden: usize,
) -> Result<Vec<VisualPlacement>> {
    let layout = media_layout(prepared, imported, modality)?;
    if layout.spans.len() != embeddings.len() {
        return Err(Error::invalid(format!(
            "prompt holds {} media spans for {} encodings",
            layout.spans.len(),
            embeddings.len()
        )));
    }
    let mut placements = Vec::new();
    for (span, encoded) in layout.spans.iter().zip(embeddings) {
        let tokens = span.range.len();
        let expected = tokens
            .checked_mul(hidden)
            .ok_or_else(|| Error::invalid("media encoding width overflow"))?;
        if encoded.len() != expected {
            return Err(Error::invalid(format!(
                "encoding holds {} values for {tokens} tokens of width {hidden}",
                encoded.len()
            )));
        }
        for offset in 0..tokens {
            placements.push(VisualPlacement {
                position: span.range.start + offset,
                rotary_position: span.rotary.as_ref().map(|rotary| rotary[offset]),
                embedding: encoded[offset * hidden..(offset + 1) * hidden].to_vec(),
            });
        }
    }
    Ok(placements)
}

#[cfg(test)]
#[path = "../../tests/unit/input_prompt.rs"]
mod tests;
