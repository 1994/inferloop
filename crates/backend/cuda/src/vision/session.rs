//! Multimodal prompt execution on the resident path.
//!
//! Media tokens carry externally computed embeddings instead of a vocabulary lookup. The prompt
//! runs as fused chunks: each media row is copied into the lane's external slot on the device
//! stream, the embedding kernel selects that slot through its per-token override flag, and one
//! replay ingests the whole chunk. A prompt with a thousand visual tokens therefore costs
//! `ceil(tokens / prefill_width)` replays, not one decode step per token.
//!
//! The alignment between media placeholders and encodings is checked before any device work: a
//! prompt whose media is not covered would otherwise execute as ordinary text and silently drop
//! the image or clip.
use crate::constants::PREFILL_LANES;
use crate::resident::DeviceProgram;
use infer_core::{Error, Result};
use infer_ir::Modality;
use infer_models::{MediaLayout, PreparedPrompt, VisualPlacement, media_layout, mrope};
use infer_spi::ImportedModel;
use std::collections::{BTreeMap, BTreeSet};

/// Media placeholder tokens the model declares, across every modality.
#[must_use]
pub fn placeholder_tokens(imported: &ImportedModel) -> BTreeSet<u32> {
    imported
        .modalities
        .iter()
        .flat_map(|plan| plan.placeholder_tokens.iter().copied())
        .collect()
}

/// Positions of the media placeholders in `tokens`.
#[must_use]
pub fn media_positions(imported: &ImportedModel, tokens: &[u32]) -> Vec<usize> {
    let placeholders = placeholder_tokens(imported);
    tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| placeholders.contains(token))
        .map(|(position, _)| position)
        .collect()
}

/// Require that the encodings cover exactly the prompt's media placeholders.
///
/// # Errors
/// Rejects a prompt with media placeholders and no encodings, encodings for a prompt without
/// placeholders, duplicate positions, positions outside the prompt or zero-width encodings.
pub fn require_placements(
    imported: &ImportedModel,
    tokens: &[u32],
    placements: &[VisualPlacement],
) -> Result<()> {
    let declared = media_positions(imported, tokens);
    let mut given: Vec<usize> = placements.iter().map(|placed| placed.position).collect();
    given.sort_unstable();
    let unique = {
        let mut deduped = given.clone();
        deduped.dedup();
        deduped
    };
    if unique.len() != given.len() {
        return Err(Error::invalid("duplicate media placements"));
    }
    if declared.is_empty() && given.is_empty() {
        return Ok(());
    }
    if declared.is_empty() {
        return Err(Error::unsupported(
            "media encodings were supplied for a prompt without media placeholders",
        ));
    }
    if given.is_empty() {
        return Err(Error::unsupported(format!(
            "prompt holds {} media placeholder tokens without encodings",
            declared.len()
        )));
    }
    if declared != given {
        return Err(Error::invalid(format!(
            "prompt holds {} media placeholder tokens at different positions than the {} encodings",
            declared.len(),
            given.len()
        )));
    }
    if placements
        .iter()
        .any(|placed| placed.embedding.is_empty() || placed.position >= tokens.len())
    {
        return Err(Error::invalid("media placement is empty or out of range"));
    }
    Ok(())
}

/// Logits and the next rotary coordinate of a media prefill.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaPrefill {
    /// Logits of the last prompt token.
    pub logits: Vec<f32>,
    /// Rotary coordinate of the first decoded token.
    pub next_rotary: usize,
}

/// Prefill a prompt, overriding media positions with their encoder outputs.
///
/// `encodings[i]` is the flattened `tokens × hidden` output of the `i`-th media item, in
/// placeholder order. Every encoding is uploaded once; rows are then copied into lane slots on the
/// device stream inside each chunk.
///
/// # Errors
/// Rejects encodings that disagree with the prompt's media spans, an empty prompt, an invalid base
/// position and CUDA failures.
pub fn prefill(
    program: &mut DeviceProgram,
    imported: &ImportedModel,
    prepared: &PreparedPrompt,
    modality: &Modality,
    encodings: &[Vec<f32>],
    hidden: usize,
    position: usize,
) -> Result<MediaPrefill> {
    let layout = media_layout(prepared, imported, modality)?;
    if layout.spans.len() != encodings.len() {
        return Err(Error::invalid(format!(
            "prompt holds {} media spans for {} encodings",
            layout.spans.len(),
            encodings.len()
        )));
    }
    if prepared.tokens.is_empty() {
        return Err(Error::invalid("empty multimodal prompt"));
    }
    if position != 0 {
        return Err(Error::unsupported(
            "media prefill starts at a zero base position",
        ));
    }
    if hidden == 0 {
        return Err(Error::invalid("media hidden size must be nonzero"));
    }
    let mut data = Vec::new();
    let mut bases = Vec::with_capacity(encodings.len());
    for (span, encoded) in layout.spans.iter().zip(encodings) {
        let tokens = span.range.len();
        let expected = tokens
            .checked_mul(hidden)
            .ok_or_else(|| Error::invalid("media encoding width overflow"))?;
        if encoded.len() != expected {
            return Err(Error::invalid(format!(
                "media encoding holds {} values for {tokens} tokens of width {hidden}",
                encoded.len()
            )));
        }
        bases.push(data.len() / hidden);
        data.extend_from_slice(encoded);
    }
    let logits = if program.prefill_width() < PREFILL_LANES {
        // A program loaded without a prompt graph keeps the per-token path; the media still runs
        // through the embedding override, just without chunk fusion.
        prefill_per_token(program, &prepared.tokens, &layout, &data, hidden, position)?
    } else {
        prefill_chunked(program, prepared, &layout, &bases, data, hidden, position)?
    };
    Ok(MediaPrefill {
        logits,
        next_rotary: layout.next_rotary,
    })
}

/// Fused prompt chunks: one replay per `prefill_width` tokens.
fn prefill_chunked(
    program: &mut DeviceProgram,
    prepared: &PreparedPrompt,
    layout: &MediaLayout,
    bases: &[usize],
    data: Vec<f32>,
    hidden: usize,
    position: usize,
) -> Result<Vec<f32>> {
    let width = program.prefill_width();
    if width == 0 {
        return Err(Error::invalid("media prefill width"));
    }
    let media = if data.is_empty() {
        None
    } else {
        Some(program.device().runtime().upload(data)?)
    };
    let mut logits = Vec::new();
    let mut offset = 0;
    while offset < prepared.tokens.len() {
        let count = width.min(prepared.tokens.len() - offset);
        let end = offset + count;
        let last = end == prepared.tokens.len();
        let chunk = &prepared.tokens[offset..end];
        let output = if let Some(media) = &media {
            let mut rows = Vec::new();
            for (index, span) in layout.spans.iter().enumerate() {
                let start = span.range.start.max(offset);
                let stop = span.range.end.min(end);
                for token in start..stop {
                    rows.push((token - offset, bases[index] + (token - span.range.start)));
                }
            }
            let rotary = layout
                .positions
                .as_ref()
                .map(|positions| &positions[offset..end]);
            program.prefill_media_chunk(
                chunk,
                position + offset,
                media,
                &rows,
                hidden,
                rotary,
                last,
            )?
        } else {
            program.prefill_batch(chunk, position + offset, last)?
        };
        if last {
            logits = output
                .into_iter()
                .next_back()
                .map(|(_, row)| row)
                .ok_or_else(|| Error::invariant("media prefill logits"))?;
        }
        offset = end;
    }
    Ok(logits)
}

/// Unfused fallback for programs loaded without a prompt graph.
fn prefill_per_token(
    program: &mut DeviceProgram,
    tokens: &[u32],
    layout: &MediaLayout,
    data: &[f32],
    hidden: usize,
    position: usize,
) -> Result<Vec<f32>> {
    let mut placements = Vec::new();
    let mut embeddings = BTreeMap::new();
    let mut row = 0;
    for span in &layout.spans {
        for (offset, token) in (span.range.start..span.range.end).enumerate() {
            let rotary = span.rotary.as_ref().map(|rotary| rotary[offset]);
            placements.push(VisualPlacement {
                position: token,
                embedding: Vec::new(),
                rotary_position: rotary,
            });
            let start = row * hidden;
            embeddings.insert(
                token,
                data.get(start..start + hidden)
                    .ok_or_else(|| Error::invalid("media encoding row"))?,
            );
            row += 1;
        }
    }
    let (rotary, _) = mrope::placed_positions(tokens.len(), &placements)?;
    program.reset()?;
    let mut logits = Vec::new();
    for (offset, token) in tokens.iter().enumerate() {
        let last = offset + 1 == tokens.len();
        let (_, row) = program.step_positioned(
            token.to_owned(),
            rotary[offset].map(|axis| position + axis),
            position + offset,
            embeddings.get(&offset).copied(),
            last,
        )?;
        if last {
            logits = row;
        }
    }
    Ok(logits)
}

/// Greedy decode from a completed media prefill.
///
/// Keeping the prefill and the decode apart lets a caller inspect the prompt logits without
/// paying for a second prefill.
///
/// # Errors
/// Rejects an invalid budget or CUDA failures.
pub fn decode(
    program: &mut DeviceProgram,
    prompt_tokens: usize,
    prefill: &MediaPrefill,
    budget: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    if budget == 0 {
        return Err(Error::invalid("generation budget must be nonzero"));
    }
    let mut logits = prefill.logits.clone();
    let mut generated = Vec::with_capacity(budget);
    for _ in 0..budget {
        let token = argmax(&logits)?;
        if stop.contains(&token) {
            break;
        }
        generated.push(token);
        let position = prompt_tokens + generated.len() - 1;
        let (_, row) = program.step(
            token,
            prefill.next_rotary + generated.len() - 1,
            position,
            None,
            true,
        )?;
        logits = row;
    }
    Ok(generated)
}

/// Prefill and greedily decode a multimodal prompt.
///
/// # Errors
/// Rejects an invalid budget, a failed prefill or CUDA failures.
#[expect(
    clippy::too_many_arguments,
    reason = "The media contract names encodings and their width explicitly; a builder would obscure it"
)]
pub fn generate(
    program: &mut DeviceProgram,
    imported: &ImportedModel,
    prepared: &PreparedPrompt,
    modality: &Modality,
    encodings: &[Vec<f32>],
    hidden: usize,
    budget: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    if budget == 0 {
        return Err(Error::invalid("generation budget must be nonzero"));
    }
    let prefill = prefill(program, imported, prepared, modality, encodings, hidden, 0)?;
    decode(program, prepared.tokens.len(), &prefill, budget, stop)
}

/// Index of the largest logit, the greedy next token.
/// # Errors
/// Rejects an empty or non-finite logit row.
pub fn argmax(logits: &[f32]) -> Result<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (index, value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::invalid("logits must be finite"));
        }
        if best.is_none_or(|(_, current)| *value > current) {
            best = Some((index, *value));
        }
    }
    best.map(|(index, _)| u32::try_from(index).unwrap_or(u32::MAX))
        .ok_or_else(|| Error::invalid("logits must not be empty"))
}

#[cfg(test)]
#[path = "../../tests/unit/vision_session.rs"]
mod tests;
