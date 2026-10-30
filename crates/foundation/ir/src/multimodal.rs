use crate::TokenBuffer;
use infer_core::{Error, MediaId, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
    Extension(String),
}
/// Prepared pixels for one media item, owned by the request until its device work completes.
///
/// Preprocessing happens in the input layer before admission. The payload is never serialized
/// into a checkpoint: a media request cannot be checkpointed while its pixels are in flight.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaItem {
    pub modality: Modality,
    pub pixels: Arc<[f32]>,
    /// `(temporal, height, width)` patch grid.
    pub grid: (usize, usize, usize),
    /// Prompt token positions replaced by this item's embeddings.
    pub span: Range<usize>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaInput {
    pub id: MediaId,
    pub modality: Modality,
    /// Opaque, versioned content identity; consumers must not assume a specific hash algorithm.
    /// Image identities include the source and preprocessing configuration, and are not a
    /// replacement for model, backend, revision, or precision compatibility checks.
    pub content_digest: String,
    /// Bounded prepared payload; retained until the request's device frame completes.
    #[serde(skip)]
    pub item: Option<MediaItem>,
}
impl MediaInput {
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            .saturating_add(self.content_digest.capacity())
            .saturating_add(self.item.as_ref().map_or(0, |item| {
                size_of::<MediaItem>()
                    .saturating_add(item.pixels.len().saturating_mul(size_of::<f32>()))
            }))
    }
}
/// Fully prepared media attached to one prompt.
///
/// `prompt` is the expanded token stream the backend executes; the items carry the preprocessed
/// patch payload, not model topology. Keeping the whole prompt lets a chunked scheduler replay
/// media chunks without re-tokenizing or re-preprocessing.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaPlan {
    pub modality: Modality,
    pub prompt: TokenBuffer,
    pub items: Vec<MediaItem>,
}
impl MediaPlan {
    /// Validate the media descriptors and retain their prepared payloads.
    /// # Errors
    /// Rejects missing payloads, mixed modalities, empty/inverted/overlapping spans, out-of-range
    /// spans, empty grids and empty pixel buffers.
    pub fn from_sequence(prompt: &TokenBuffer, media: &[MediaInput]) -> Result<Option<Self>> {
        if media.is_empty() {
            return Ok(None);
        }
        if prompt.is_empty() {
            return Err(Error::invalid("media prompt has no tokens"));
        }
        let first = media
            .first()
            .ok_or_else(|| Error::invalid("media prompt is empty"))?
            .modality
            .clone();
        if first == Modality::Text {
            return Err(Error::invalid("media input cannot use the text modality"));
        }
        let mut items = Vec::with_capacity(media.len());
        let mut previous_end = 0usize;
        for input in media {
            if input.modality != first {
                return Err(Error::unsupported(
                    "mixed image and video in one prompt is not supported",
                ));
            }
            let item = input
                .item
                .clone()
                .ok_or_else(|| Error::invalid("media input has no prepared payload"))?;
            if item.modality != first {
                return Err(Error::invalid(
                    "media payload modality differs from its input",
                ));
            }
            if item.span.is_empty()
                || item.span.start < previous_end
                || item.span.end > prompt.len()
            {
                return Err(Error::invalid(
                    "media spans must be ordered, disjoint and inside the prompt",
                ));
            }
            let (temporal, height, width) = item.grid;
            if temporal == 0
                || height == 0
                || width == 0
                || temporal
                    .checked_mul(height)
                    .and_then(|n| n.checked_mul(width))
                    .is_none_or(|patches| patches == 0)
            {
                return Err(Error::invalid("media patch grid is empty or overflows"));
            }
            if item.pixels.is_empty() {
                return Err(Error::invalid("media pixel payload is empty"));
            }
            previous_end = item.span.end;
            items.push(item);
        }
        Ok(Some(Self {
            modality: first,
            prompt: prompt.clone(),
            items,
        }))
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    #[must_use]
    pub fn media_tokens(&self) -> usize {
        self.items.iter().map(|item| item.span.len()).sum()
    }
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            .saturating_add(self.items.capacity().saturating_mul(size_of::<MediaItem>()))
            .saturating_add(
                self.items
                    .iter()
                    .map(|item| item.pixels.len().saturating_mul(size_of::<f32>()))
                    .sum::<usize>(),
            )
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaStage {
    Fetch,
    Decode,
    Preprocess,
    Encode,
    Fusion,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaNode {
    pub id: u32,
    pub stage: MediaStage,
    pub inputs: Vec<u32>,
    pub modality: Modality,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalGraph {
    pub nodes: Vec<MediaNode>,
}
impl MultimodalGraph {
    /// Serialized order is topological; future or duplicate dependencies are invalid.
    ///
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for node in &self.nodes {
            if node.inputs.iter().any(|input| !seen.contains(input)) || !seen.insert(node.id) {
                return Err(Error::invalid(
                    "multimodal graph is not a unique topological ordering",
                ));
            }
        }
        Ok(())
    }
}
