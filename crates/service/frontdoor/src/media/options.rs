//! Validated vLLM-style image resize options.
use infer_core::{Error, Result};
use infer_ir::{ImageNormalization, ImagePatchLayout, ImageTransform, Modality};
use infer_models::{
    ExpandedPrompt, ImageProcessor, ImportedModel, PreparedPrompt, PromptImage, RawImage, RgbImage,
    SharedPreparedPrompt, TextAssets, prepare, prepare_image_geometry, prepare_shared,
};
use serde::Deserialize;

/// Image resize overrides accepted globally or in `mm_processor_kwargs`.
///
/// Per-request values override deployment values. A `size` supplies pixel counts in
/// `shortest_edge` and `longest_edge`; explicit `min_pixels` and `max_pixels` override them.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProcessingOptions {
    do_resize: Option<bool>,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    size: Option<ImageSize>,
    images_kwargs: Option<ScopedOptions>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScopedOptions {
    do_resize: Option<bool>,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    size: Option<ImageSize>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageSize {
    shortest_edge: usize,
    longest_edge: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ResizeMode {
    Smart,
    Disabled,
}

pub(super) struct ResolvedOptions {
    pub processor: ImageProcessor,
    pub resize: ResizeMode,
    validate_target: bool,
}

impl ImageProcessingOptions {
    pub(super) const fn empty() -> Self {
        Self {
            do_resize: None,
            min_pixels: None,
            max_pixels: None,
            size: None,
            images_kwargs: None,
        }
    }

    /// Parse JSON resize options for a deployment.
    /// # Errors
    /// Rejects malformed JSON, unknown options and zero or inverted pixel ranges.
    pub fn parse(value: &str) -> Result<Self> {
        let options: Self = serde_json::from_str(value)
            .map_err(|error| Error::invalid(format!("image processor options: {error}")))?;
        options.validate()?;
        Ok(options)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_fields(self.min_pixels, self.max_pixels, self.size)?;
        if let Some(scoped) = &self.images_kwargs {
            validate_fields(scoped.min_pixels, scoped.max_pixels, scoped.size)?;
        }
        Ok(())
    }

    pub(super) fn resolve(
        &self,
        request: Option<&Self>,
        base: &ImageProcessor,
    ) -> Result<ResolvedOptions> {
        let mut resolved = ResolvedOptions {
            processor: base.clone(),
            resize: ResizeMode::Smart,
            validate_target: !self.is_empty() || request.is_some(),
        };
        self.apply(&mut resolved)?;
        if let Some(request) = request {
            request.apply(&mut resolved)?;
        }
        if resolved.processor.min_pixels > resolved.processor.max_pixels {
            return Err(Error::invalid("invalid effective image pixel range"));
        }
        Ok(resolved)
    }

    fn apply(&self, resolved: &mut ResolvedOptions) -> Result<()> {
        self.validate()?;
        apply_fields(
            resolved,
            self.do_resize,
            self.min_pixels,
            self.max_pixels,
            self.size,
        );
        if let Some(scoped) = &self.images_kwargs {
            apply_fields(
                resolved,
                scoped.do_resize,
                scoped.min_pixels,
                scoped.max_pixels,
                scoped.size,
            );
        }
        Ok(())
    }

    const fn is_empty(&self) -> bool {
        self.do_resize.is_none()
            && self.min_pixels.is_none()
            && self.max_pixels.is_none()
            && self.size.is_none()
            && self.images_kwargs.is_none()
    }
}

impl ResolvedOptions {
    pub fn transform(&self, height: usize, width: usize) -> Result<ImageTransform> {
        if self.resize == ResizeMode::Smart {
            return self.processor.transform(height, width);
        }
        ImageTransform::new(
            [height, width],
            [height, width],
            ImagePatchLayout::new(
                self.processor.patch_size,
                self.processor.temporal_patch_size,
                self.processor.merge_size,
            )?,
            ImageNormalization::new(
                self.processor.mean,
                self.processor.std,
                self.processor.rescale,
            )?,
        )
    }

    pub fn prepare_shared(
        &self,
        assets: &TextAssets,
        imported: &ImportedModel,
        prompt: &str,
        images: Vec<RawImage>,
    ) -> Result<SharedPreparedPrompt> {
        if self.resize == ResizeMode::Smart {
            self.validate_targets(&images)?;
            return prepare_shared(
                assets,
                &self.processor,
                imported,
                prompt,
                &images,
                &Modality::Image,
            );
        }
        let (expanded, images) =
            self.prepare_original(assets, imported, prompt, images, |processor, image| {
                processor.preprocess_unresized_shared(image)
            })?;
        Ok(SharedPreparedPrompt {
            tokens: expanded.tokens,
            spans: expanded.spans,
            images,
        })
    }

    pub fn prepare(
        &self,
        assets: &TextAssets,
        imported: &ImportedModel,
        prompt: &str,
        images: Vec<RawImage>,
    ) -> Result<PreparedPrompt> {
        if self.resize == ResizeMode::Smart {
            self.validate_targets(&images)?;
            return prepare(
                assets,
                &self.processor,
                imported,
                prompt,
                &images,
                &Modality::Image,
            );
        }
        let (expanded, images) =
            self.prepare_original(assets, imported, prompt, images, |processor, image| {
                processor.preprocess_unresized(image).map(PromptImage::new)
            })?;
        Ok(PreparedPrompt {
            tokens: expanded.tokens,
            spans: expanded.spans,
            images,
        })
    }

    fn prepare_original<T>(
        &self,
        assets: &TextAssets,
        imported: &ImportedModel,
        prompt: &str,
        images: Vec<RawImage>,
        preprocess: impl Fn(&ImageProcessor, &RgbImage) -> Result<T>,
    ) -> Result<(ExpandedPrompt, Vec<T>)> {
        let transforms = images
            .iter()
            .map(|image| self.transform(image.height, image.width))
            .collect::<Result<Vec<_>>>()?;
        let expanded = prepare_image_geometry(assets, imported, prompt, &transforms)?;
        let mut prepared = Vec::with_capacity(images.len());
        for image in images {
            let image = RgbImage {
                height: image.height,
                width: image.width,
                pixels: image.pixels,
            };
            prepared.push(preprocess(&self.processor, &image)?);
        }
        Ok((expanded, prepared))
    }

    fn validate_targets(&self, images: &[RawImage]) -> Result<()> {
        if self.validate_target {
            for image in images {
                self.transform(image.height, image.width)?;
            }
        }
        Ok(())
    }
}

fn validate_fields(min: Option<usize>, max: Option<usize>, size: Option<ImageSize>) -> Result<()> {
    if min == Some(0)
        || max == Some(0)
        || min.zip(max).is_some_and(|(min, max)| min > max)
        || size
            .is_some_and(|size| size.shortest_edge == 0 || size.shortest_edge > size.longest_edge)
    {
        return Err(Error::invalid("invalid image pixel range"));
    }
    Ok(())
}

const fn apply_fields(
    resolved: &mut ResolvedOptions,
    do_resize: Option<bool>,
    min: Option<usize>,
    max: Option<usize>,
    size: Option<ImageSize>,
) {
    if let Some(size) = size {
        resolved.processor.min_pixels = size.shortest_edge;
        resolved.processor.max_pixels = size.longest_edge;
    }
    if let Some(min) = min {
        resolved.processor.min_pixels = min;
    }
    if let Some(max) = max {
        resolved.processor.max_pixels = max;
    }
    if let Some(do_resize) = do_resize {
        resolved.resize = if do_resize {
            ResizeMode::Smart
        } else {
            ResizeMode::Disabled
        };
    }
}

#[cfg(test)]
#[path = "../../tests/unit/media_options.rs"]
mod tests;
