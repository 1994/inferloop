//! Image preprocessing for the vision tower, following the Qwen2-VL family processor.
//!
//! The processor resizes with a bicubic kernel so both axes become multiples of
//! `patch_size × merge_size` and the pixel count lands in `[min_pixels, max_pixels]`, then
//! normalizes and lays the patches out in spatial-merge-block order with the flattened
//! `[channel, temporal, patch, patch]` vector the patch-embed projection consumes.
use infer_core::{Error, Result};

/// Channels of the RGB images the processor accepts.
pub(crate) const RGB_CHANNELS: usize = 3;
/// Full scale of one 8-bit channel.
const BYTE_SCALE: f64 = 255.0;
/// Reciprocal of [`BYTE_SCALE`], the processor's rescale factor.
const RESCALE: f32 = 1.0 / 255.0;
/// Largest aspect ratio the reference processor accepts.
const MAX_ASPECT_RATIO: usize = 200;
/// Output pixel centres sit half a pixel past the integer grid.
const HALF_PIXEL: f64 = 0.5;
/// Keys bicubic kernel with `a = -0.5`, in the reference's polynomial form.
const CUBIC_A: f64 = -0.5;
const CUBIC_QUADRATIC: f64 = 2.0;
const CUBIC_LINEAR: f64 = 3.0;
const CUBIC_PIVOT: f64 = 5.0;
const CUBIC_OFFSET: f64 = 8.0;
const CUBIC_TAIL: f64 = 4.0;
const CUBIC_NEAR_SPAN: f64 = 1.0;
const CUBIC_FAR_SPAN: f64 = 2.0;

/// Decoded source image in row-major RGB8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub height: usize,
    pub width: usize,
    /// `height × width × 3` bytes, red first.
    pub pixels: Vec<u8>,
}

/// Bicubic weight of `x` for the `a = -0.5` kernel used by the reference processor.
#[expect(
    clippy::suboptimal_flops,
    reason = "The kernel is written as the reference defines it; fused vs separate rounding is immaterial"
)]
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x <= CUBIC_NEAR_SPAN {
        ((CUBIC_A + CUBIC_QUADRATIC) * x - (CUBIC_A + CUBIC_LINEAR)).mul_add(x * x, 1.0)
    } else if x < CUBIC_FAR_SPAN {
        (((x - CUBIC_PIVOT) * x + CUBIC_OFFSET) * x - CUBIC_TAIL) * CUBIC_A
    } else {
        0.0
    }
}

/// Processor geometry and normalization, read from a package preprocessor config.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    /// Smallest accepted pixel count (`size.shortest_edge`).
    pub min_pixels: usize,
    /// Largest accepted pixel count (`size.longest_edge`).
    pub max_pixels: usize,
    pub mean: [f32; RGB_CHANNELS],
    pub std: [f32; RGB_CHANNELS],
    pub rescale: f32,
}

/// Patches of one image, already flattened for the patch-embed projection.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedImage {
    /// `patches × channels × temporal × patch × patch`, in spatial-merge-block order.
    pub pixels: Vec<f32>,
    /// `(temporal, height, width)` patch grid.
    pub grid: (usize, usize, usize),
}

/// Per-channel normalization applied to a rescaled RGB value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Normalization {
    pub mean: [f32; RGB_CHANNELS],
    pub std: [f32; RGB_CHANNELS],
    pub rescale: f32,
}

impl Normalization {
    /// Canonical RGB8 uses the fast processor's fused 255 normalization. Every other configured
    /// factor keeps the original multiply-add/divide path, including zero and tiny positive scales.
    #[expect(
        clippy::suboptimal_flops,
        reason = "Sequential operations are required to match the processor's fused-bit result"
    )]
    fn apply(self, value: f32, channel: usize) -> f32 {
        const CANONICAL_RESCALE: f32 = 1.0 / 255.0;
        const CANONICAL_INVERSE: f32 = 255.0;
        if self.rescale == CANONICAL_RESCALE {
            (value - self.mean[channel] * CANONICAL_INVERSE)
                / (self.std[channel] * CANONICAL_INVERSE)
        } else {
            value.mul_add(self.rescale, -self.mean[channel]) / self.std[channel]
        }
    }
}

/// Patch layout of one visual sample: `groups` temporal groups of `temporal` frames each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PatchGrid {
    /// Frames merged into one token along the temporal axis.
    pub temporal: usize,
    /// Patch rows after resizing.
    pub height: usize,
    /// Patch columns after resizing.
    pub width: usize,
    /// Side of one square patch in pixels.
    pub patch: usize,
    /// Patches merged into one encoder output along each spatial axis.
    pub merge: usize,
    /// Temporal groups: one for an image, `frames / temporal` for a video.
    pub groups: usize,
}

/// Lay out resized frames into merged patch vectors.
///
/// `frames` holds `groups × temporal` RGB frames, each `height × patch` rows by
/// `width × patch` columns, in presentation order. A still image passes the same frame for every
/// temporal tap, so images and videos share one layout.
///
/// # Errors
/// Rejects missing frames, frames that disagree with the grid or a grid that does not divide by
/// the merge size.
pub(crate) fn patchify(
    frames: &[&[f32]],
    grid: PatchGrid,
    norm: &Normalization,
) -> Result<Vec<f32>> {
    let resized_height = grid
        .height
        .checked_mul(grid.patch)
        .ok_or_else(|| Error::invalid("patch grid height overflow"))?;
    let resized_width = grid
        .width
        .checked_mul(grid.patch)
        .ok_or_else(|| Error::invalid("patch grid width overflow"))?;
    let expected_frames = grid
        .groups
        .checked_mul(grid.temporal)
        .ok_or_else(|| Error::invalid("patch frame count overflow"))?;
    if grid.temporal == 0
        || grid.merge == 0
        || grid.patch == 0
        || grid.height == 0
        || grid.width == 0
        || !grid.height.is_multiple_of(grid.merge)
        || !grid.width.is_multiple_of(grid.merge)
        || frames.len() != expected_frames
    {
        return Err(Error::invalid("patch layout"));
    }
    let frame_values = resized_height
        .checked_mul(resized_width)
        .and_then(|value| value.checked_mul(RGB_CHANNELS))
        .ok_or_else(|| Error::invalid("patch frame overflow"))?;
    if frames.iter().any(|frame| frame.len() != frame_values) {
        return Err(Error::invalid(
            "patch frame does not match the resized grid",
        ));
    }
    let patch_area = grid
        .patch
        .checked_mul(grid.patch)
        .ok_or_else(|| Error::invalid("patch area overflow"))?;
    let patch_width = RGB_CHANNELS
        .checked_mul(grid.temporal)
        .and_then(|value| value.checked_mul(patch_area))
        .ok_or_else(|| Error::invalid("patch width overflow"))?;
    let patches = grid
        .groups
        .checked_mul(grid.height)
        .and_then(|value| value.checked_mul(grid.width))
        .ok_or_else(|| Error::invalid("patch count overflow"))?;
    let mut pixels = Vec::with_capacity(
        patches
            .checked_mul(patch_width)
            .ok_or_else(|| Error::invalid("patch vector overflow"))?,
    );
    for group in 0..grid.groups {
        for block_row in 0..grid.height / grid.merge {
            for block_col in 0..grid.width / grid.merge {
                for in_row in 0..grid.merge {
                    for in_col in 0..grid.merge {
                        let row = block_row * grid.merge + in_row;
                        let col = block_col * grid.merge + in_col;
                        for channel in 0..RGB_CHANNELS {
                            for tap in 0..grid.temporal {
                                let frame = frames[group * grid.temporal + tap];
                                for patch_row in 0..grid.patch {
                                    for patch_col in 0..grid.patch {
                                        let y = row * grid.patch + patch_row;
                                        let x = col * grid.patch + patch_col;
                                        let value =
                                            frame[(y * resized_width + x) * RGB_CHANNELS + channel];
                                        pixels.push(norm.apply(value, channel));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(pixels)
}

impl ImageProcessor {
    /// Parse `preprocessor_config.json`.
    /// # Errors
    /// Rejects missing or inconsistent geometry and normalization fields.
    pub fn from_json(config: &[u8]) -> Result<Self> {
        let parsed: serde_json::Value =
            serde_json::from_slice(config).map_err(|error| Error::invalid(error.to_string()))?;
        let number = |value: &serde_json::Value, key: &str| -> Result<usize> {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| Error::invalid(format!("preprocessor config lacks {key}")))
        };
        let size = parsed
            .get("size")
            .ok_or_else(|| Error::invalid("preprocessor config lacks size"))?;
        let triplet = |key: &str| -> Result<[f32; RGB_CHANNELS]> {
            let values = parsed
                .get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| Error::invalid(format!("preprocessor config lacks {key}")))?;
            let mut out = [0f32; RGB_CHANNELS];
            for (slot, value) in out.iter_mut().zip(values) {
                *slot = value
                    .as_f64()
                    .map(|value| {
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "Normalization constants are small exact values"
                        )]
                        let scaled = value as f32;
                        scaled
                    })
                    .ok_or_else(|| Error::invalid(format!("preprocessor {key} entry")))?;
            }
            Ok(out)
        };
        let processor = Self {
            patch_size: number(&parsed, "patch_size")?,
            temporal_patch_size: number(&parsed, "temporal_patch_size")?,
            merge_size: number(&parsed, "merge_size")?,
            min_pixels: number(size, "shortest_edge")?,
            max_pixels: number(size, "longest_edge")?,
            mean: triplet("image_mean")?,
            std: triplet("image_std")?,
            rescale: RESCALE,
        };
        if processor.patch_size == 0
            || processor.temporal_patch_size == 0
            || processor.merge_size == 0
            || processor.min_pixels > processor.max_pixels
            || processor.std.contains(&0.0)
        {
            return Err(Error::invalid("invalid preprocessor configuration"));
        }
        Ok(processor)
    }

    /// Flattened width of one patch.
    #[must_use]
    pub const fn patch_width(&self) -> usize {
        RGB_CHANNELS * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Resized dimensions of one image: both axes are `patch × merge` multiples and the pixel
    /// count stays within the configured bounds, preserving the aspect ratio as closely as the
    /// grid allows.
    /// # Errors
    /// Rejects degenerate dimensions or an extreme aspect ratio.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Image pixel counts are far below the exact integer range of f64"
    )]
    pub fn smart_resize(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        let factor = self
            .patch_size
            .checked_mul(self.merge_size)
            .ok_or_else(|| Error::invalid("resize factor overflow"))?;
        if height == 0 || width == 0 {
            return Err(Error::invalid("image dimensions must be nonzero"));
        }
        if height.max(width) > MAX_ASPECT_RATIO * height.min(width) {
            return Err(Error::invalid("image aspect ratio exceeds 200"));
        }
        let (height, width, factor) = (height as f64, width as f64, factor as f64);
        let mut resized_height = (height / factor).round_ties_even() * factor;
        let mut resized_width = (width / factor).round_ties_even() * factor;
        let snap = |height: f64, width: f64| (height as usize, width as usize);
        if resized_height * resized_width > self.max_pixels as f64 {
            let beta = ((height * width) / self.max_pixels as f64).sqrt();
            resized_height = factor.max((height / beta / factor).floor() * factor);
            resized_width = factor.max((width / beta / factor).floor() * factor);
        } else if resized_height * resized_width < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (height * width)).sqrt();
            resized_height = (height * beta / factor).ceil() * factor;
            resized_width = (width * beta / factor).ceil() * factor;
        }
        let (resized_height, resized_width) = snap(resized_height, resized_width);
        if resized_height == 0 || resized_width == 0 {
            return Err(Error::invalid("resized dimensions must be nonzero"));
        }
        Ok((resized_height, resized_width))
    }

    /// Preprocess one image into patch vectors.
    /// # Errors
    /// Rejects malformed image buffers or geometry that does not tile by the patch grid.
    pub fn preprocess(&self, image: &RgbImage) -> Result<PreprocessedImage> {
        let expected = image
            .height
            .checked_mul(image.width)
            .and_then(|value| value.checked_mul(RGB_CHANNELS))
            .ok_or_else(|| Error::invalid("image dimensions overflow"))?;
        if image.pixels.len() != expected {
            return Err(Error::invalid("image buffer does not match its dimensions"));
        }
        let (resized_height, resized_width) = self.smart_resize(image.height, image.width)?;
        let resized = Self::resize(image, resized_height, resized_width);
        let grid_h = resized_height / self.patch_size;
        let grid_w = resized_width / self.patch_size;
        let merge = self.merge_size;
        if !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
            return Err(Error::invalid(
                "patch grid does not divide by the merge size",
            ));
        }
        // A still image repeats the same frame across the temporal taps.
        let frames = vec![resized.as_slice(); self.temporal_patch_size];
        let pixels = patchify(
            &frames,
            PatchGrid {
                temporal: self.temporal_patch_size,
                height: grid_h,
                width: grid_w,
                patch: self.patch_size,
                merge: self.merge_size,
                groups: 1,
            },
            &Normalization {
                mean: self.mean,
                std: self.std,
                rescale: self.rescale,
            },
        )?;
        Ok(PreprocessedImage {
            pixels,
            grid: (1, grid_h, grid_w),
        })
    }

    /// Separable bicubic resize of an RGB8 image into `height × width`, in `0..=255`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Resampled values are rounded into the 8-bit range before the narrowing cast"
    )]
    pub(crate) fn resize(image: &RgbImage, height: usize, width: usize) -> Vec<f32> {
        if image.height == height && image.width == width {
            // Unit-scale bicubic sampling reproduces each RGB8 value exactly.
            return image.pixels.iter().copied().map(f32::from).collect();
        }
        let intermediate = if image.width == width {
            // Unit-scale horizontal filtering preserves each RGB8 value exactly.
            image.pixels.iter().copied().map(f32::from).collect()
        } else {
            let horizontal = resample_axis(image.width, width);
            let mut intermediate = vec![0f32; image.height * width * RGB_CHANNELS];
            for y in 0..image.height {
                for x in 0..width {
                    let weights = &horizontal[x];
                    for channel in 0..RGB_CHANNELS {
                        let mut value = 0f64;
                        for &(tap, weight) in weights {
                            let index = (y * image.width + tap) * RGB_CHANNELS + channel;
                            value = weight.mul_add(f64::from(image.pixels[index]), value);
                        }
                        let rounded = value.round().clamp(0.0, BYTE_SCALE) as f32;
                        intermediate[(y * width + x) * RGB_CHANNELS + channel] = rounded;
                    }
                }
            }
            intermediate
        };
        if image.height == height {
            // The horizontal pass already rounded the result to RGB8 values.
            return intermediate;
        }
        let vertical = resample_axis(image.height, height);
        let mut output = vec![0f32; height * width * RGB_CHANNELS];
        for y in 0..height {
            let weights = &vertical[y];
            for x in 0..width {
                for channel in 0..RGB_CHANNELS {
                    let mut value = 0f64;
                    for &(tap, weight) in weights {
                        value = weight.mul_add(
                            f64::from(intermediate[(tap * width + x) * RGB_CHANNELS + channel]),
                            value,
                        );
                    }
                    let rounded = value.round().clamp(0.0, BYTE_SCALE) as f32;
                    output[(y * width + x) * RGB_CHANNELS + channel] = rounded;
                }
            }
        }
        output
    }
}

/// Normalized bicubic taps of every output position along one axis.
///
/// The downsampling support widens with the scale; border taps are truncated and renormalized.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Image axis lengths are far below the exact integer range of f64"
)]
fn resample_axis(source: usize, target: usize) -> Vec<Vec<(usize, f64)>> {
    const COEFFICIENT_LIMIT: f64 = 32768.0;
    const MAX_PRECISION: usize = 22;
    let scale = source as f64 / target as f64;
    let filter_scale = scale.max(1.0);
    let support = CUBIC_FAR_SPAN * filter_scale;
    let mut rows = Vec::with_capacity(target);
    for position in 0..target {
        let center = (position as f64 + HALF_PIXEL) * scale;
        let start = (center - support + HALF_PIXEL).trunc().max(0.0);
        let first = start as usize;
        let last = (center + support + HALF_PIXEL).trunc().min(source as f64) as usize;
        let mut taps = Vec::with_capacity(last - first);
        let mut total = 0f64;
        for tap in first..last {
            let weight = cubic((tap as f64 + HALF_PIXEL - center) / filter_scale);
            total += weight;
            taps.push((tap, weight));
        }
        if total != 0.0 {
            for (_, weight) in &mut taps {
                *weight /= total;
            }
        }
        rows.push(taps);
    }
    // Torchvision's CPU RGB8 path quantizes normalized taps to signed 16-bit
    // coefficients, then rounds/clamps after each separable pass.
    let maximum = rows.iter().flatten().map(|(_, w)| *w).fold(0.0, f64::max);
    let mut denominator = 1.0;
    for _ in 0..MAX_PRECISION {
        if (maximum * denominator).mul_add(2.0, HALF_PIXEL).trunc() >= COEFFICIENT_LIMIT {
            break;
        }
        denominator *= 2.0;
    }
    for (_, weight) in rows.iter_mut().flatten() {
        *weight = (*weight * denominator).round() / denominator;
    }
    rows
}

#[cfg(test)]
#[path = "../../tests/unit/input_image.rs"]
mod tests;
