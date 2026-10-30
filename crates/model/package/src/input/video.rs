//! Video preprocessing for the vision tower, following the Qwen3-VL video processor.
//!
//! Frames arrive already decoded: codecs stay out of this crate, exactly as they do for images.
//! The processor samples the source clip at a fixed rate, resizes every frame to one shared grid,
//! pads the clip to whole temporal patches and lays the frames out in the same
//! `[channel, temporal, patch, patch]` vectors the image path uses. The vision tower and prompt
//! assembly therefore see one media shape, whether the sample is a still or a clip.
use crate::image::{
    ImageProcessor, Normalization, PatchGrid, PreprocessedImage, RGB_CHANNELS, RgbImage, patchify,
};
use infer_core::{Error, Result};
use std::iter::repeat_n;
use std::num::NonZeroUsize;
use std::thread::available_parallelism;
use std::thread::scope as thread_scope;

/// Frames sampled per second of source video when the configuration omits a rate.
const DEFAULT_FPS: f64 = 2.0;
/// Fewest frames a clip is sampled to, matching the reference video processor.
const DEFAULT_MIN_FRAMES: usize = 4;
/// Most frames a clip is sampled to, matching the reference video processor.
const DEFAULT_MAX_FRAMES: usize = 768;
/// Smallest clip whose frame resize is split across threads.
const PARALLEL_FRAMES: usize = 8;
/// Largest aspect ratio the reference processor accepts.
const MAX_ASPECT_RATIO: usize = 200;

/// One preprocessed clip: patch vectors plus the timestamp of every temporal group.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedVideo {
    /// Patch vectors and `(temporal, height, width)` grid for the whole clip.
    pub image: PreprocessedImage,
    /// Timestamp of each temporal group, in prompt order.
    ///
    /// The Qwen3-VL processor prefixes every group with `<t seconds>` so the model can reason
    /// about time; the prompt assembler uses these values.
    pub timestamps: Vec<f64>,
}

/// Decoded video: frames in presentation order plus the source frame rate.
#[derive(Debug, Clone, PartialEq)]
pub struct RawVideo {
    /// Source frames per second; sampling derives the clip length from it.
    pub fps: f64,
    /// Decoded frames in presentation order.
    pub frames: Vec<RgbImage>,
}

/// Video processor geometry and sampling policy, read from a package video preprocessor config.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    /// Smallest accepted pixel count over the whole clip (`size.shortest_edge`).
    pub min_pixels: usize,
    /// Largest accepted pixel count over the whole clip (`size.longest_edge`).
    pub max_pixels: usize,
    pub mean: [f32; RGB_CHANNELS],
    pub std: [f32; RGB_CHANNELS],
    pub rescale: f32,
    /// Frames sampled per second of source video.
    pub fps: f64,
    /// Fewest frames sampled from a clip.
    pub min_frames: usize,
    /// Most frames sampled from a clip.
    pub max_frames: usize,
}

/// A clip's global sampling and resize decisions, reusable across independent temporal groups.
///
/// The plan keeps the original clip's timing and geometry when its frames are decoded in parts.
#[derive(Debug, Clone)]
pub struct VideoPlan {
    processor: VideoProcessor,
    indices: Vec<usize>,
    source_fps: f64,
    height: usize,
    width: usize,
    resized_height: usize,
    resized_width: usize,
}

impl VideoPlan {
    /// Source-frame indices in presentation order, to be supplied to the decoder.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Number of temporal groups in the processed clip.
    #[must_use]
    pub const fn groups(&self) -> usize {
        self.indices
            .len()
            .div_ceil(self.processor.temporal_patch_size)
    }

    /// Process one independently decoded temporal group while preserving the clip's global plan.
    ///
    /// `frames` must contain exactly the source frames selected for `group`, in index order.
    /// The last group may contain fewer than a full temporal patch; its last frame is repeated.
    /// Groups can be processed concurrently and assembled in increasing group order.
    ///
    /// # Errors
    /// Rejects an out-of-range group, wrong frame count, geometry or buffer length.
    pub fn preprocess_group(&self, group: usize, frames: &[RgbImage]) -> Result<PreprocessedVideo> {
        let temporal = self.processor.temporal_patch_size;
        let start = group
            .checked_mul(temporal)
            .filter(|start| *start < self.indices.len())
            .ok_or_else(|| Error::invalid("video temporal group is outside the plan"))?;
        let end = start.saturating_add(temporal).min(self.indices.len());
        let indices = &self.indices[start..end];
        if frames.len() != indices.len() {
            return Err(Error::invalid("video temporal group frame count"));
        }
        let expected = self
            .height
            .checked_mul(self.width)
            .and_then(|value| value.checked_mul(RGB_CHANNELS))
            .ok_or_else(|| Error::invalid("video frame dimensions overflow"))?;
        if frames.iter().any(|frame| {
            frame.height != self.height
                || frame.width != self.width
                || frame.pixels.len() != expected
        }) {
            return Err(Error::invalid("video temporal group frame geometry"));
        }
        let resized: Vec<_> = frames
            .iter()
            .map(|frame| ImageProcessor::resize(frame, self.resized_height, self.resized_width))
            .collect();
        let last = resized
            .last()
            .ok_or_else(|| Error::invalid("video temporal group is empty"))?;
        let mut views: Vec<_> = resized.iter().map(Vec::as_slice).collect();
        views.extend(repeat_n(last.as_slice(), temporal - views.len()));
        let grid_h = self.resized_height / self.processor.patch_size;
        let grid_w = self.resized_width / self.processor.patch_size;
        let pixels = patchify(
            &views,
            PatchGrid {
                temporal,
                height: grid_h,
                width: grid_w,
                patch: self.processor.patch_size,
                merge: self.processor.merge_size,
                groups: 1,
            },
            &Normalization {
                mean: self.processor.mean,
                std: self.processor.std,
                rescale: self.processor.rescale,
            },
        )?;
        #[expect(
            clippy::cast_precision_loss,
            reason = "Frame indices follow the same bounded f64 sampler as whole-clip preprocessing"
        )]
        let timestamp =
            (indices[0] as f64 + indices[indices.len() - 1] as f64) / (2.0 * self.source_fps);
        Ok(PreprocessedVideo {
            image: PreprocessedImage {
                pixels,
                grid: (1, grid_h, grid_w),
            },
            timestamps: vec![timestamp],
        })
    }
}

impl VideoProcessor {
    /// Parse `video_preprocessor_config.json`.
    ///
    /// Geometry and normalization are shared with the image processor; sampling policy comes from
    /// the optional `fps`, `min_frames` and `max_frames` keys.
    ///
    /// # Errors
    /// Rejects missing or inconsistent geometry, a non-positive sampling rate or an empty frame
    /// range.
    pub fn from_json(config: &[u8]) -> Result<Self> {
        let parsed: serde_json::Value =
            serde_json::from_slice(config).map_err(|error| Error::invalid(error.to_string()))?;
        // Geometry, normalization and their invariants are exactly the image processor's.
        let image = ImageProcessor::from_json(config)?;
        let number = |key: &str| -> Result<Option<f64>> {
            match parsed.get(key) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(value) => value.as_f64().map(Some).ok_or_else(|| {
                    Error::invalid(format!("video preprocessor {key} must be a number"))
                }),
            }
        };
        let frame_count = |key: &str, default: usize| -> Result<usize> {
            let Some(value) = number(key)? else {
                return Ok(default);
            };
            if !value.is_finite() || value < 1.0 || value > f64::from(u32::MAX) {
                return Err(Error::invalid(format!(
                    "video preprocessor {key} must be a positive frame count"
                )));
            }
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "The value is range-checked against the u32 frame ceiling above"
            )]
            let count = value as u32;
            usize::try_from(count)
                .map_err(|_| Error::invalid(format!("video preprocessor {key} does not fit usize")))
        };
        let fps = number("fps")?.unwrap_or(DEFAULT_FPS);
        let min_frames = frame_count("min_frames", DEFAULT_MIN_FRAMES)?;
        let max_frames = frame_count("max_frames", DEFAULT_MAX_FRAMES)?;
        if parsed
            .get("cap_pixels_per_frame")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            return Err(Error::unsupported(
                "video preprocessor cap_pixels_per_frame is not implemented",
            ));
        }
        if !fps.is_finite() || fps <= 0.0 {
            return Err(Error::invalid("video preprocessor fps must be positive"));
        }
        if min_frames == 0 || min_frames > max_frames {
            return Err(Error::invalid("video preprocessor frame range"));
        }
        Ok(Self {
            patch_size: image.patch_size,
            temporal_patch_size: image.temporal_patch_size,
            merge_size: image.merge_size,
            min_pixels: image.min_pixels,
            max_pixels: image.max_pixels,
            mean: image.mean,
            std: image.std,
            rescale: image.rescale,
            fps,
            min_frames,
            max_frames,
        })
    }

    /// Flattened width of one patch.
    #[must_use]
    pub const fn patch_width(&self) -> usize {
        RGB_CHANNELS * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Frame indices sampled from a clip, uniformly spaced like the reference `linspace` sampler.
    ///
    /// The clip length comes from the source rate and [`Self::fps`], clamped to
    /// [`Self::min_frames`], [`Self::max_frames`] and the decoded frame count.
    ///
    /// # Errors
    /// Rejects an empty clip, a non-positive source rate or an unrepresentable frame count.
    pub fn sample_frames(&self, video: &RawVideo) -> Result<Vec<usize>> {
        self.sample_frame_indices(video.frames.len(), video.fps)
    }

    /// Select source-frame indices from clip metadata, without materializing decoded frames.
    ///
    /// # Errors
    /// Rejects an empty clip or a non-positive source rate.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Frame counts follow the existing reference sampler's f64 contract"
    )]
    pub fn sample_frame_indices(&self, total: usize, source_fps: f64) -> Result<Vec<usize>> {
        if total == 0 {
            return Err(Error::invalid("video has no frames"));
        }
        if !source_fps.is_finite() || source_fps <= 0.0 {
            return Err(Error::invalid("video source fps must be positive"));
        }
        let seconds = total as f64 / source_fps;
        let wanted = (seconds * self.fps) as usize;
        let count = wanted.max(self.min_frames).min(self.max_frames).min(total);
        if count == 0 {
            return Err(Error::invalid("video samples no frames"));
        }
        if count == 1 {
            return Ok(vec![0]);
        }
        let last = (total - 1) as f64;
        let step = last / (count - 1) as f64;
        Ok((0..count)
            .map(|index| (index as f64 * step).round_ties_even() as usize)
            .collect())
    }

    /// Plan sampling and resizing once for incremental video processing.
    ///
    /// The count must be the actual number of source frames, not a duration-based estimate.
    /// The returned groups retain the same sample positions and timestamps as [`Self::preprocess`].
    ///
    /// # Errors
    /// Rejects invalid processor geometry, clip metadata or resize geometry.
    pub fn plan(
        &self,
        total: usize,
        source_fps: f64,
        height: usize,
        width: usize,
    ) -> Result<VideoPlan> {
        if self.patch_size == 0 || self.temporal_patch_size == 0 || self.merge_size == 0 {
            return Err(Error::invalid("video processor geometry must be nonzero"));
        }
        let indices = self.sample_frame_indices(total, source_fps)?;
        let (resized_height, resized_width) = self.smart_resize(indices.len(), height, width)?;
        Ok(VideoPlan {
            processor: self.clone(),
            indices,
            source_fps,
            height,
            width,
            resized_height,
            resized_width,
        })
    }

    /// Resized frame dimensions: both axes are `patch × merge` multiples and the whole clip stays
    /// inside the configured pixel budget.
    ///
    /// # Errors
    /// Rejects a clip shorter than one temporal patch, degenerate frame dimensions or an extreme
    /// aspect ratio.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Video pixel counts are far below the exact integer range of f64"
    )]
    pub fn smart_resize(
        &self,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize)> {
        let factor = self
            .patch_size
            .checked_mul(self.merge_size)
            .ok_or_else(|| Error::invalid("video resize factor overflow"))?;
        if frames < self.temporal_patch_size {
            return Err(Error::invalid(
                "video clip is shorter than one temporal patch",
            ));
        }
        if height == 0 || width == 0 {
            return Err(Error::invalid("video frame dimensions must be nonzero"));
        }
        let (mut height, mut width) = (height as f64, width as f64);
        if height < factor as f64 || width < factor as f64 {
            let scale = (factor as f64 / height).max(factor as f64 / width);
            height = (height * scale).trunc();
            width = (width * scale).trunc();
        }
        if height.max(width) > MAX_ASPECT_RATIO as f64 * height.min(width) {
            return Err(Error::invalid("video aspect ratio exceeds 200"));
        }
        let factor_f = factor as f64;
        let temporal = self.temporal_patch_size as f64;
        let mut resized_height = (height / factor_f).round_ties_even() * factor_f;
        let mut resized_width = (width / factor_f).round_ties_even() * factor_f;
        let temporal_tokens = (frames as f64 / temporal).round_ties_even() * temporal;
        let clip = temporal_tokens * resized_height * resized_width;
        if clip > self.max_pixels as f64 {
            let beta = ((frames as f64 * height * width) / self.max_pixels as f64).sqrt();
            resized_height = factor_f.max((height / beta / factor_f).floor() * factor_f);
            resized_width = factor_f.max((width / beta / factor_f).floor() * factor_f);
        } else if clip < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (frames as f64 * height * width)).sqrt();
            resized_height = (height * beta / factor_f).ceil() * factor_f;
            resized_width = (width * beta / factor_f).ceil() * factor_f;
        }
        let (resized_height, resized_width) = (resized_height as usize, resized_width as usize);
        if resized_height == 0 || resized_width == 0 {
            return Err(Error::invalid("resized video dimensions must be nonzero"));
        }
        Ok((resized_height, resized_width))
    }

    /// Preprocess one clip into patch vectors.
    ///
    /// Frames are resized concurrently: each frame is independent, so the CPU work scales with
    /// available cores while the patch layout stays deterministic.
    ///
    /// # Errors
    /// Rejects empty clips, frames that disagree with each other, malformed buffers or a grid that
    /// does not tile by the patch and merge sizes.
    #[expect(
        clippy::cast_precision_loss,
        reason = "Frame indices are far below the exact integer range of f64"
    )]
    pub fn preprocess(&self, video: &RawVideo) -> Result<PreprocessedVideo> {
        let indices = self.sample_frames(video)?;
        let first = video
            .frames
            .first()
            .ok_or_else(|| Error::invalid("video has no frames"))?;
        let (height, width) = (first.height, first.width);
        for frame in &video.frames {
            if frame.height != height || frame.width != width {
                return Err(Error::invalid("video frames disagree in size"));
            }
            let expected = frame
                .height
                .checked_mul(frame.width)
                .and_then(|value| value.checked_mul(RGB_CHANNELS))
                .ok_or_else(|| Error::invalid("video frame dimensions overflow"))?;
            if frame.pixels.len() != expected {
                return Err(Error::invalid(
                    "video frame buffer does not match its dimensions",
                ));
            }
        }
        let (resized_height, resized_width) = self.smart_resize(indices.len(), height, width)?;
        let grid_h = resized_height / self.patch_size;
        let grid_w = resized_width / self.patch_size;
        if !grid_h.is_multiple_of(self.merge_size) || !grid_w.is_multiple_of(self.merge_size) {
            return Err(Error::invalid(
                "video patch grid does not divide by the merge size",
            ));
        }
        let mut sampled: Vec<&RgbImage> =
            indices.iter().map(|index| &video.frames[*index]).collect();
        // The tower consumes whole temporal patches; a trailing partial patch repeats the last
        // frame, exactly as the reference processor does.
        let pad = (self.temporal_patch_size - sampled.len() % self.temporal_patch_size)
            % self.temporal_patch_size;
        if pad > 0 {
            let last = *sampled
                .last()
                .ok_or_else(|| Error::invalid("video has no sampled frames"))?;
            sampled.extend(repeat_n(last, pad));
        }
        let groups = sampled.len() / self.temporal_patch_size;
        // The reference timestamps average the first and last frame of each temporal group.
        let timestamps = indices
            .chunks(self.temporal_patch_size)
            .map(|group| {
                let last = group.last().copied().unwrap_or(0);
                let first = group.first().copied().unwrap_or(0);
                (first as f64 + last as f64) / (2.0 * video.fps)
            })
            .collect();
        let resized = resize_frames(&sampled, resized_height, resized_width)?;
        let frames: Vec<&[f32]> = resized.iter().map(Vec::as_slice).collect();
        let pixels = patchify(
            &frames,
            PatchGrid {
                temporal: self.temporal_patch_size,
                height: grid_h,
                width: grid_w,
                patch: self.patch_size,
                merge: self.merge_size,
                groups,
            },
            &Normalization {
                mean: self.mean,
                std: self.std,
                rescale: self.rescale,
            },
        )?;
        Ok(PreprocessedVideo {
            image: PreprocessedImage {
                pixels,
                grid: (groups, grid_h, grid_w),
            },
            timestamps,
        })
    }
}

/// Resize every sampled frame, splitting the work across cores for longer clips.
fn resize_frames(frames: &[&RgbImage], height: usize, width: usize) -> Result<Vec<Vec<f32>>> {
    let workers = available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(frames.len());
    if workers <= 1 || frames.len() < PARALLEL_FRAMES {
        return Ok(frames
            .iter()
            .map(|frame| ImageProcessor::resize(frame, height, width))
            .collect());
    }
    let chunk = frames.len().div_ceil(workers);
    thread_scope(|scope| {
        let handles: Vec<_> = frames
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|frame| ImageProcessor::resize(frame, height, width))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut resized = Vec::with_capacity(frames.len());
        for handle in handles {
            let mut part = handle
                .join()
                .map_err(|_| Error::invariant("frame resize worker panicked"))?;
            resized.append(&mut part);
        }
        Ok(resized)
    })
}

#[cfg(test)]
#[path = "../../tests/unit/input_video.rs"]
mod tests;
