//! Bounded multimodal input assets: provider import, processors and per-prompt limits.
//!
//! URL/network/decode work belongs to the service input layer; this module owns only the
//! provider-declared model geometry and the deterministic preprocessors that turn RGB8 media
//! into prompt-aligned patches.
use crate::image::ImageProcessor;
use crate::input::prompt::{prepare, prepare_video};
use crate::input::text::TextAssets;
use crate::input::video::VideoProcessor;
use crate::package_path;
use infer_core::{Error, ModelId, Result};
use infer_ir::Modality;
use infer_spi::ImportedModel;
use std::fs::{metadata, read};
use std::path::Path;
use std::sync::Arc;

/// Largest accepted package metadata file.
const METADATA_MAX_BYTES: u64 = 16 * 1024 * 1024;
/// Default per-prompt image limit.
const DEFAULT_IMAGE_LIMIT: usize = 2;
/// Default per-prompt video limit.
const DEFAULT_VIDEO_LIMIT: usize = 1;

/// Per-prompt media item limits, matching the `--limit-mm-per-prompt` contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaLimits {
    pub images: usize,
    pub videos: usize,
}
impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            images: DEFAULT_IMAGE_LIMIT,
            videos: DEFAULT_VIDEO_LIMIT,
        }
    }
}
impl MediaLimits {
    #[must_use]
    pub const fn text_only(self) -> bool {
        self.images == 0 && self.videos == 0
    }
    /// # Errors
    /// Rejects a zero limit for one modality while the other remains enabled.
    pub fn validate(self) -> Result<()> {
        if self.images == usize::MAX || self.videos == usize::MAX {
            return Err(Error::invalid("media limit is unbounded"));
        }
        Ok(())
    }
    /// # Errors
    /// Rejects a request above the configured image or video limit.
    pub fn check(self, modality: &Modality, count: usize) -> Result<()> {
        let limit = match modality {
            Modality::Image => self.images,
            Modality::Video => self.videos,
            Modality::Text | Modality::Audio | Modality::Extension(_) => {
                return Err(Error::unsupported("unsupported media modality"));
            }
        };
        if count > limit {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                format!("request carries {count} {modality:?} items, limit is {limit}"),
            ));
        }
        Ok(())
    }
}

/// Provider import and the two deterministic processors declared by a multimodal package.
pub struct MultimodalAssets {
    pub imported: ImportedModel,
    pub images: ImageProcessor,
    pub video: Option<VideoProcessor>,
}
impl MultimodalAssets {
    /// Open the multimodal metadata for `root`, or `None` when the package declares no media.
    ///
    /// # Errors
    /// Rejects malformed/oversized config or preprocessor files and unsupported modalities.
    pub fn open(root: &Path) -> Result<Option<Arc<Self>>> {
        let config = read_bounded(&package_path(root, "config.json")?)?;
        let registry = crate::default_registry();
        let provider = registry.resolve(&config)?;
        let imported = provider.import(ModelId::ONE, &config)?;
        if imported.modalities.is_empty() {
            return Ok(None);
        }
        let images = ImageProcessor::from_json(&read_bounded(&package_path(
            root,
            "preprocessor_config.json",
        )?)?)?;
        let video = if imported
            .modalities
            .iter()
            .any(|plan| plan.modality == Modality::Video)
        {
            Some(VideoProcessor::from_json(&read_bounded(&package_path(
                root,
                "video_preprocessor_config.json",
            )?)?)?)
        } else {
            None
        };
        Ok(Some(Arc::new(Self {
            imported,
            images,
            video,
        })))
    }

    /// Preprocess images and expand their placeholders in `prompt`.
    /// # Errors
    /// Rejects mismatched placeholder counts, geometry or unsupported modality declarations.
    pub fn prepare_images(
        &self,
        assets: &TextAssets,
        prompt: &str,
        images: &[crate::RawImage],
    ) -> Result<crate::PreparedPrompt> {
        prepare(
            assets,
            &self.images,
            &self.imported,
            prompt,
            images,
            &Modality::Image,
        )
    }

    /// Preprocess one clip and expand its timestamped placeholder groups.
    /// # Errors
    /// Rejects a package without a video processor, mismatched placeholder counts or geometry.
    pub fn prepare_video(
        &self,
        assets: &TextAssets,
        prompt: &str,
        videos: &[crate::RawVideo],
    ) -> Result<crate::PreparedPrompt> {
        let processor = self
            .video
            .as_ref()
            .ok_or_else(|| Error::unsupported("package declares no video processor"))?;
        prepare_video(
            assets,
            processor,
            &self.imported,
            prompt,
            videos,
            &Modality::Video,
        )
    }

    /// The processor's preprocessed image shape, exposed for bounded decoding tests.
    #[must_use]
    pub const fn image_processor(&self) -> &ImageProcessor {
        &self.images
    }
}

fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let metadata = metadata(path).map_err(|e| Error::invalid(e.to_string()))?;
    if metadata.len() > METADATA_MAX_BYTES {
        return Err(Error::new(
            infer_core::ErrorCode::Capacity,
            "multimodal package metadata exceeds budget",
        ));
    }
    read(path).map_err(|e| Error::invalid(e.to_string()))
}
