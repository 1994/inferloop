//! Validated image resize overrides and their host working-set bound.
use super::{MAX_IMAGE_PIXELS, MEDIA_STAGING_BYTES, RGB_CHANNELS};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::ImageResizePolicy;
use infer_models::{ImageProcessor, RawImage};
use serde::Deserialize;
use serde_json::Value;

/// Bicubic support spans four source-pixel units at unit scale.
const RESAMPLE_SUPPORT: usize = 4;
/// Account for endpoint rounding when bounding each row's tap count.
const RESAMPLE_ROUNDING: usize = 2;

/// Supported subset of vLLM's image processor keyword arguments.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProcessingOptions {
    do_resize: Option<bool>,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    size: Option<ImageSize>,
    images_kwargs: Option<ScopedImageOptions>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScopedImageOptions {
    do_resize: Option<bool>,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    size: Option<ImageSize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageSize {
    shortest_edge: usize,
    longest_edge: usize,
}

impl ImageProcessingOptions {
    pub(super) const fn is_empty(&self) -> bool {
        self.do_resize.is_none()
            && self.min_pixels.is_none()
            && self.max_pixels.is_none()
            && self.size.is_none()
            && match &self.images_kwargs {
                Some(scoped) => {
                    scoped.do_resize.is_none()
                        && scoped.min_pixels.is_none()
                        && scoped.max_pixels.is_none()
                        && scoped.size.is_none()
                }
                None => true,
            }
    }
    /// Parse the supported image resize options.
    /// # Errors
    /// Rejects unknown options, malformed JSON and invalid pixel ranges.
    pub fn parse(value: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(value).map_err(|error| {
            Error::unsupported(format!(
                "unsupported or invalid image processor options: {error}"
            ))
        })?;
        let object = value
            .as_object()
            .ok_or_else(|| Error::invalid("image processor options must be an object"))?;
        let scoped_null = object
            .get("images_kwargs")
            .and_then(Value::as_object)
            .is_some_and(|scope| scope.values().any(Value::is_null));
        if object.values().any(Value::is_null) || scoped_null {
            return Err(Error::unsupported(
                "null image processor options are not supported",
            ));
        }
        let parsed: Self = serde_json::from_value(value).map_err(|error| {
            Error::unsupported(format!(
                "unsupported or invalid image processor options: {error}"
            ))
        })?;
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<()> {
        let scoped = self.images_kwargs.as_ref();
        if scoped.is_some_and(|scope| {
            (self.do_resize.is_some() && scope.do_resize.is_some())
                || (self.min_pixels.is_some() && scope.min_pixels.is_some())
                || (self.max_pixels.is_some() && scope.max_pixels.is_some())
                || (self.size.is_some() && scope.size.is_some())
        }) {
            return Err(Error::unsupported(
                "image processor option appears in both flat and scoped kwargs",
            ));
        }
        let min = scoped
            .and_then(|scope| scope.min_pixels)
            .or(self.min_pixels);
        let max = scoped
            .and_then(|scope| scope.max_pixels)
            .or(self.max_pixels);
        let size = scoped
            .and_then(|scope| scope.size.as_ref())
            .or(self.size.as_ref());
        if size.is_some() && (min.is_some() || max.is_some()) {
            return Err(Error::unsupported(
                "size cannot be combined with min_pixels or max_pixels",
            ));
        }
        let (min, max) = size.map_or((min, max), |size| {
            (Some(size.shortest_edge), Some(size.longest_edge))
        });
        if min == Some(0)
            || max == Some(0)
            || min.is_some_and(|v| v > MAX_IMAGE_PIXELS)
            || max.is_some_and(|v| v > MAX_IMAGE_PIXELS)
            || min.zip(max).is_some_and(|(min, max)| min > max)
        {
            return Err(Error::invalid("invalid image processor pixel range"));
        }
        Ok(())
    }

    pub(super) fn merge(&self, request: Option<&Self>) -> Result<Self> {
        let mut merged = self.clone();
        if let Some(request) = request {
            merged.do_resize = request.do_resize.or(merged.do_resize);
            merged.min_pixels = request.min_pixels.or(merged.min_pixels);
            merged.max_pixels = request.max_pixels.or(merged.max_pixels);
            if request.size.is_some() {
                merged.size.clone_from(&request.size);
            }
            if request.images_kwargs.is_some() {
                merged.images_kwargs.clone_from(&request.images_kwargs);
            }
        }
        merged.validate()?;
        Ok(merged)
    }

    pub(super) fn resize_policy(&self) -> ImageResizePolicy {
        match self
            .images_kwargs
            .as_ref()
            .and_then(|scope| scope.do_resize)
            .or(self.do_resize)
        {
            Some(false) => ImageResizePolicy::Disabled,
            Some(true) | None => ImageResizePolicy::Smart,
        }
    }

    pub(super) fn processor(&self, original: &ImageProcessor) -> Result<ImageProcessor> {
        let mut result = original.clone();
        let scoped = self.images_kwargs.as_ref();
        if let Some(size) = scoped
            .and_then(|scope| scope.size.as_ref())
            .or(self.size.as_ref())
        {
            result.min_pixels = size.shortest_edge;
            result.max_pixels = size.longest_edge;
        }
        if let Some(min) = scoped
            .and_then(|scope| scope.min_pixels)
            .or(self.min_pixels)
        {
            result.min_pixels = min;
        }
        if let Some(max) = scoped
            .and_then(|scope| scope.max_pixels)
            .or(self.max_pixels)
        {
            result.max_pixels = max;
        }
        if result.min_pixels > result.max_pixels {
            return Err(Error::invalid(
                "invalid effective image processor pixel range",
            ));
        }
        Ok(result)
    }
}

/// Bound the currently selected CPU processor's intermediate and retained patch allocations.
pub(super) fn check_working_set(images: &[RawImage], processor: &ImageProcessor) -> Result<()> {
    check_working_set_with_resize(images, processor, ImageResizePolicy::Smart)
}

pub(super) fn check_working_set_with_resize(
    images: &[RawImage],
    processor: &ImageProcessor,
    resize: ImageResizePolicy,
) -> Result<()> {
    let budget = images
        .len()
        .checked_mul(MEDIA_STAGING_BYTES)
        .ok_or_else(capacity)?;
    let mut raw = 0usize;
    for image in images {
        raw = raw
            .checked_add(image.pixels.capacity())
            .ok_or_else(capacity)?;
    }
    let mut retained = 0usize;
    let mut copied = 0usize;
    for image in images {
        let (height, width) = processor.target_dimensions(image.height, image.width, resize)?;
        let pixels = height
            .checked_mul(width)
            .filter(|n| *n <= MAX_IMAGE_PIXELS)
            .ok_or_else(capacity)?;
        let output = pixels
            .checked_mul(RGB_CHANNELS * size_of::<f32>())
            .ok_or_else(capacity)?;
        let intermediate = image
            .height
            .checked_mul(width)
            .and_then(|n| n.checked_mul(RGB_CHANNELS * size_of::<f32>()))
            .ok_or_else(capacity)?;
        let weights = axis_bytes(image.width, width)?
            .checked_add(axis_bytes(image.height, height)?)
            .ok_or_else(capacity)?;
        let actual_patch = output
            .checked_mul(processor.temporal_patch_size)
            .ok_or_else(capacity)?;
        // patchify reserves exactly one flattened vector per spatial patch.
        let patch_capacity = actual_patch;
        let resize_peak = raw
            .checked_add(retained)
            .and_then(|n| n.checked_add(intermediate))
            .and_then(|n| n.checked_add(output))
            .and_then(|n| n.checked_add(weights))
            .ok_or_else(capacity)?;
        let patch_peak = raw
            .checked_add(retained)
            .and_then(|n| n.checked_add(output))
            .and_then(|n| n.checked_add(patch_capacity))
            .ok_or_else(capacity)?;
        if resize_peak.max(patch_peak) > budget {
            return Err(capacity());
        }
        retained = retained.checked_add(patch_capacity).ok_or_else(capacity)?;
        copied = copied.checked_add(actual_patch).ok_or_else(capacity)?;
    }
    if retained.checked_add(copied).ok_or_else(capacity)? > budget {
        return Err(capacity());
    }
    Ok(())
}

fn axis_bytes(source: usize, target: usize) -> Result<usize> {
    let taps = source
        .div_ceil(target)
        .max(1)
        .checked_mul(RESAMPLE_SUPPORT)
        .and_then(|n| n.checked_add(RESAMPLE_ROUNDING))
        .ok_or_else(capacity)?
        .min(source);
    let row = taps
        .checked_mul(size_of::<(usize, f64)>())
        .and_then(|n| n.checked_add(size_of::<Vec<(usize, f64)>>()))
        .ok_or_else(capacity)?;
    target.checked_mul(row).ok_or_else(capacity)
}

fn capacity() -> Error {
    Error::new(
        ErrorCode::Capacity,
        "image resize exceeds the media preparation budget",
    )
}
