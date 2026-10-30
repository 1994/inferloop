//! URL/data/media preparation for the `OpenAI` adapter.
//!
//! Decoding and network access stay in this input layer; the engine receives prepared patches or
//! RGB8 payloads accepted by its image processor, plus prompt spans. Video reads and commands retain their
//! time/frame bounds. Malformed inputs are explicit errors, never text-only fallbacks.
mod images;
mod options;
mod source;

use self::images::{Downloads, LoadedImage};
use self::options::{ResizeMode, ResolvedOptions};

use crate::cpu::{CpuContext, CpuPool};
use infer_core::{Error, ErrorCode, MediaId, Result};
use infer_ir::{MediaInput, MediaItem, MediaPixels, Modality};
use infer_media_input::{decode_data_uri, decode_rgb8};
use infer_models::{
    ImageProcessor, MediaLimits, MultimodalAssets, PreparedPrompt, RawImage, RawVideo, RgbImage,
    SharedPreparedPrompt, TextAssets, prepare_image_geometry,
};
use infer_spi::ImagePreprocessing;
use sha2::{Digest, Sha256};
use std::env::temp_dir;
use std::fs::{File, OpenOptions, metadata, read, remove_file};
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio, id};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::sleep as async_sleep;

pub use options::ImageProcessingOptions;

/// Largest accepted raw video-frame pixel count.
const MAX_VIDEO_PIXELS: usize = 16 * 1024 * 1024;
/// Largest accepted video download/decoded raw output.
const MAX_VIDEO_BYTES: usize = 24 * 1024 * 1024;
/// Largest accepted frame count decoded from one clip.
const MAX_VIDEO_FRAMES: usize = 768;
/// Chunk size for bounded decoder output draining.
const READ_CHUNK_BYTES: usize = 64 * 1024;
/// Check cancellation between source-hash chunks, including for large input files.
const HASH_CHUNK_BYTES: usize = 64 * 1024;
/// SHA-256 produces 256-bit source identities.
const SOURCE_DIGEST_BYTES: usize = 32;
/// Identity namespace for the CPU image decoder and preprocessing contract.
const IMAGE_IDENTITY_DOMAIN: &[u8] = b"inferloop-image-source-cpu-v1\0";
/// RGB channels in one decoded frame.
const RGB_CHANNELS: usize = 3;
/// Axes in one media patch grid.
const GRID_AXES: usize = 3;
/// FNV-1a 64-bit offset basis for the content digest.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime for the content digest.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Highest accepted source frame rate; lower rates such as 0.5 are preserved.
const MAX_SOURCE_FPS: f64 = 1000.0;
/// Bound for one subprocess stream that is not already media-sized.
const SUBPROCESS_STREAM_BYTES: usize = 1 << 20;
/// Wall-clock bound for one ffprobe invocation.
const PROBE_TIMEOUT_SECONDS: u64 = 5;
/// Wall-clock bound for one ffmpeg decode invocation.
const DECODE_TIMEOUT_SECONDS: u64 = 15;
/// Poll interval while waiting for a bounded subprocess.
const PROCESS_POLL_MILLIS: u64 = 10;
/// Owner-only mode for temporary media files.
#[cfg(unix)]
const TEMP_FILE_MODE: u32 = 0o600;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Where a media URL may be read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourcePolicy {
    /// Public HTTP input: data and remote URLs only.
    Public,
    /// Trusted local fixture/debug input: also accepts file:// and absolute paths.
    TrustedLocal,
}

/// Image decode selection; device capability and image format still determine availability.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ImageDecoder {
    /// Prefer a supported device decoder.
    #[default]
    Auto,
    /// Decode on the CPU before optional device image preprocessing.
    Cpu,
}

#[derive(Clone, Copy)]
struct SourceDigest([u8; SOURCE_DIGEST_BYTES]);

#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameKind {
    Image,
    Video,
}

/// One media reference parsed from an `OpenAI` content part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaRequest {
    pub modality: Modality,
    pub url: String,
}

/// Decoded-media service shared by the HTTP handlers.
pub struct MediaService {
    assets: Arc<MultimodalAssets>,
    limits: MediaLimits,
    source: SourcePolicy,
    image_preprocessing: Option<Arc<dyn ImagePreprocessing>>,
    image_options: ImageProcessingOptions,
    image_decoder: ImageDecoder,
}
impl MediaService {
    #[must_use]
    pub const fn new(assets: Arc<MultimodalAssets>, limits: MediaLimits) -> Self {
        Self {
            assets,
            limits,
            source: SourcePolicy::Public,
            image_preprocessing: None,
            image_options: ImageProcessingOptions::empty(),
            image_decoder: ImageDecoder::Auto,
        }
    }

    /// Select backend preprocessing only for transforms that the backend explicitly supports.
    #[must_use]
    pub fn with_image_preprocessing(mut self, capability: Arc<dyn ImagePreprocessing>) -> Self {
        self.image_preprocessing = Some(capability);
        self
    }

    /// Select whether eligible compressed images may be decoded by the backend.
    #[must_use]
    pub const fn with_image_decoder(mut self, decoder: ImageDecoder) -> Self {
        self.image_decoder = decoder;
        self
    }

    /// Apply validated deployment image options, overridden by any per-request values.
    #[must_use]
    pub const fn with_image_options(mut self, options: ImageProcessingOptions) -> Self {
        self.image_options = options;
        self
    }
    #[must_use]
    pub const fn limits(&self) -> MediaLimits {
        self.limits
    }

    /// Preprocess all media and expand their placeholders in `prompt`.
    /// # Errors
    /// Rejects mixed modalities, over-limit item counts, malformed URLs/data, decode failures and
    /// placeholder/patch mismatches.
    pub fn prepare(
        &self,
        assets: &TextAssets,
        prompt: &str,
        requests: &[MediaRequest],
        context: &CpuContext,
    ) -> Result<(PreparedPrompt, Vec<MediaInput>)> {
        self.prepare_inner(assets, prompt, requests, None, context)
    }

    /// Resolve remote media concurrently, preserving one slot per request.
    ///
    /// Inline sources leave an empty slot and are decoded later on a CPU worker.
    /// # Errors
    /// Returns a transport, cancellation or source-policy error.
    pub async fn resolve(
        &self,
        requests: &[MediaRequest],
        pool: &CpuPool,
    ) -> Result<Vec<Option<Vec<u8>>>> {
        source::resolve(requests, pool).await
    }

    /// Prepare resolved media for the selected backend and expand prompt placeholders.
    ///
    /// `downloads` must have one slot per request and contain bytes for every HTTP URL.
    /// Call on a CPU worker with its cancellation context; the returned inputs retain their data.
    /// # Errors
    /// Returns errors for unmatched sources, invalid options, media or prompt geometry.
    pub fn prepare_resolved(
        &self,
        assets: &TextAssets,
        prompt: &str,
        requests: &[MediaRequest],
        downloads: &[Option<Vec<u8>>],
        options: Option<&ImageProcessingOptions>,
        context: &CpuContext,
    ) -> Result<(Vec<u32>, Vec<MediaInput>)> {
        self.prepare_resolved_from(
            assets,
            prompt,
            requests,
            Downloads::Borrowed(downloads),
            options,
            context,
        )
    }

    /// Consume resolved HTTP bytes so compressed data can move directly into a media request.
    pub(crate) fn prepare_resolved_owned(
        &self,
        assets: &TextAssets,
        prompt: &str,
        requests: &[MediaRequest],
        downloads: Vec<Option<Vec<u8>>>,
        options: Option<&ImageProcessingOptions>,
        context: &CpuContext,
    ) -> Result<(Vec<u32>, Vec<MediaInput>)> {
        self.prepare_resolved_from(
            assets,
            prompt,
            requests,
            Downloads::Owned(downloads),
            options,
            context,
        )
    }

    fn prepare_resolved_from(
        &self,
        assets: &TextAssets,
        prompt: &str,
        requests: &[MediaRequest],
        mut downloads: Downloads<'_>,
        options: Option<&ImageProcessingOptions>,
        context: &CpuContext,
    ) -> Result<(Vec<u32>, Vec<MediaInput>)> {
        if downloads.as_slice().len() != requests.len()
            || requests
                .iter()
                .zip(downloads.as_slice())
                .any(|(request, bytes)| {
                    (request.url.starts_with("http://") || request.url.starts_with("https://"))
                        && bytes.is_none()
                })
        {
            return Err(Error::invariant("remote media sources were not resolved"));
        }
        if self.validate_requests(requests)? == Modality::Image {
            let (images, sources) =
                self.load_images(requests, &mut downloads, context, self.image_decoder)?;
            let selected = self
                .image_options
                .resolve(options, self.assets.image_processor())?;
            if let Some(capability) = &self.image_preprocessing {
                let transforms = images
                    .iter()
                    .map(|image| {
                        let [height, width] = image.source();
                        selected.transform(height, width)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if transforms
                    .iter()
                    .all(|transform| capability.supports(transform))
                {
                    return self.deferred_image_inputs(
                        assets, prompt, images, transforms, sources, &selected,
                    );
                }
            }
            let images = images
                .into_iter()
                .map(|image| image.into_raw(context))
                .collect::<Result<Vec<_>>>()?;
            let prepared =
                selected.prepare_shared(assets, &self.assets.imported, prompt, images)?;
            Self::shared_image_inputs(prepared, sources, &selected)
        } else {
            if options.is_some() {
                return Err(Error::unsupported(
                    "image processor options require image inputs",
                ));
            }
            let (prepared, media) = self.prepare_inner(
                assets,
                prompt,
                requests,
                Some(downloads.as_slice()),
                context,
            )?;
            Ok((prepared.tokens, media))
        }
    }

    fn deferred_image_inputs(
        &self,
        assets: &TextAssets,
        prompt: &str,
        images: Vec<LoadedImage>,
        transforms: Vec<infer_ir::ImageTransform>,
        sources: Vec<SourceDigest>,
        options: &ResolvedOptions,
    ) -> Result<(Vec<u32>, Vec<MediaInput>)> {
        if images.len() != transforms.len() || images.len() != sources.len() || images.is_empty() {
            return Err(Error::invariant(
                "deferred image metadata does not match inputs",
            ));
        }
        let prepared = prepare_image_geometry(assets, &self.assets.imported, prompt, &transforms)?;
        let mut media = Vec::with_capacity(images.len());
        for (index, (((image, transform), source), span)) in images
            .into_iter()
            .zip(transforms)
            .zip(sources)
            .zip(prepared.spans)
            .enumerate()
        {
            let id = u64::try_from(index)
                .ok()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| Error::invalid("media identity overflow"))?;
            let grid = transform.grid();
            let pixels = image.into_pixels(transform)?;
            let decoder_identity = match &pixels {
                MediaPixels::EncodedImage(image) => Some(image.info().decoder_identity()),
                MediaPixels::Image(_) | MediaPixels::Patches(_) => None,
            };
            media.push(MediaInput {
                id: MediaId::new(id)?,
                modality: Modality::Image,
                content_digest: image_digest_with_decoder(
                    source,
                    &options.processor,
                    options.resize,
                    decoder_identity,
                )?,
                item: Some(MediaItem {
                    modality: Modality::Image,
                    pixels,
                    grid,
                    span,
                }),
            });
        }
        Ok((prepared.tokens, media))
    }

    fn validate_requests(&self, requests: &[MediaRequest]) -> Result<Modality> {
        if requests.is_empty() {
            return Err(Error::invalid("media preparation needs at least one item"));
        }
        let modality = requests
            .first()
            .ok_or_else(|| Error::invalid("media request is empty"))?
            .modality
            .clone();
        if requests.iter().any(|request| request.modality != modality) {
            return Err(Error::unsupported(
                "mixed image and video in one prompt is not supported",
            ));
        }
        self.limits.check(&modality, requests.len())?;
        Ok(modality)
    }

    fn load_images(
        &self,
        requests: &[MediaRequest],
        downloads: &mut Downloads<'_>,
        context: &CpuContext,
        decoder: ImageDecoder,
    ) -> Result<(Vec<LoadedImage>, Vec<SourceDigest>)> {
        let capability = if decoder == ImageDecoder::Auto {
            self.image_preprocessing.as_deref()
        } else {
            None
        };
        let loaded = requests
            .iter()
            .enumerate()
            .map(|(index, request)| {
                let bytes = downloads.take(index);
                images::load(&request.url, bytes, self.source, context, capability)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(loaded.into_iter().unzip())
    }

    fn shared_image_inputs(
        prepared: SharedPreparedPrompt,
        sources: Vec<SourceDigest>,
        options: &ResolvedOptions,
    ) -> Result<(Vec<u32>, Vec<MediaInput>)> {
        if prepared.images.is_empty() || prepared.images.len() != prepared.spans.len() {
            return Err(Error::invalid(
                "prepared media is empty or has inconsistent spans",
            ));
        }
        if sources.len() != prepared.images.len() {
            return Err(Error::invariant(
                "image source identities do not match prepared images",
            ));
        }
        tracing::debug!(target: "infer::media", prompt_tokens = prepared.tokens.len(), tokens = ?prepared.tokens,
            spans = ?prepared.spans, grids = ?prepared.images.iter().map(|image| image.grid).collect::<Vec<_>>(),
            "prepared HTTP media");
        let mut media = Vec::with_capacity(prepared.images.len());
        for (index, ((image, span), source)) in prepared
            .images
            .into_iter()
            .zip(prepared.spans)
            .zip(sources)
            .enumerate()
        {
            let id = u64::try_from(index)
                .ok()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| Error::invalid("media identity overflow"))?;
            let item = MediaItem {
                modality: Modality::Image,
                pixels: image.pixels.into(),
                grid: image.grid,
                span,
            };
            media.push(MediaInput {
                id: MediaId::new(id)?,
                modality: Modality::Image,
                content_digest: image_digest_with_mode(source, &options.processor, options.resize)?,
                item: Some(item),
            });
        }
        Ok((prepared.tokens, media))
    }

    fn prepare_inner(
        &self,
        assets: &TextAssets,
        prompt: &str,
        requests: &[MediaRequest],
        downloads: Option<&[Option<Vec<u8>>]>,
        context: &CpuContext,
    ) -> Result<(PreparedPrompt, Vec<MediaInput>)> {
        let modality = self.validate_requests(requests)?;
        let image_options = if modality == Modality::Image {
            Some(
                self.image_options
                    .resolve(None, self.assets.image_processor())?,
            )
        } else {
            None
        };
        let (prepared, image_sources) = match modality {
            Modality::Image => {
                let mut sources = Downloads::Borrowed(downloads.unwrap_or(&[]));
                let (images, sources) =
                    self.load_images(requests, &mut sources, context, ImageDecoder::Cpu)?;
                let images = images
                    .into_iter()
                    .map(|image| image.into_raw(context))
                    .collect::<Result<Vec<_>>>()?;
                let options = image_options
                    .as_ref()
                    .ok_or_else(|| Error::invariant("image processor options"))?;
                (
                    options.prepare(assets, &self.assets.imported, prompt, images)?,
                    sources,
                )
            }
            Modality::Video => {
                let videos = requests
                    .iter()
                    .enumerate()
                    .map(|(index, request)| {
                        let bytes = downloads.and_then(|values| values[index].as_deref());
                        bytes.map_or_else(
                            || load_video_with(&request.url, self.source, Some(context)),
                            |bytes| load_video_bytes(bytes, Some(context)),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                (
                    self.assets.prepare_video(assets, prompt, &videos)?,
                    Vec::new(),
                )
            }
            Modality::Text | Modality::Audio | Modality::Extension(_) => {
                return Err(Error::unsupported("unsupported media modality"));
            }
        };
        tracing::debug!(target: "infer::media", prompt_tokens = prepared.tokens.len(), tokens = ?prepared.tokens,
            spans = ?prepared.spans, grids = ?prepared.images.iter().map(|image| image.grid).collect::<Vec<_>>(),
            "prepared HTTP media");
        if prepared.images.is_empty() || prepared.images.len() != prepared.spans.len() {
            return Err(Error::invalid(
                "prepared media is empty or has inconsistent spans",
            ));
        }
        if modality == Modality::Image && image_sources.len() != prepared.images.len() {
            return Err(Error::invariant(
                "image source identities do not match prepared images",
            ));
        }
        let mut sources = image_sources.into_iter();
        let mut media = Vec::with_capacity(prepared.images.len());
        for (index, (image, span)) in prepared.images.iter().zip(&prepared.spans).enumerate() {
            let id = u64::try_from(index)
                .ok()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| Error::invalid("media identity overflow"))?;
            let item = MediaItem {
                modality: modality.clone(),
                pixels: Arc::<[f32]>::from(image.pixels.as_slice()).into(),
                grid: image.grid,
                span: span.clone(),
            };
            media.push(MediaInput {
                id: MediaId::new(id)?,
                modality: modality.clone(),
                content_digest: if modality == Modality::Image {
                    let source = sources
                        .next()
                        .ok_or_else(|| Error::invariant("missing image source identity"))?;
                    let options = image_options
                        .as_ref()
                        .ok_or_else(|| Error::invariant("image processor options"))?;
                    image_digest_with_mode(source, &options.processor, options.resize)?
                } else {
                    digest(&image.pixels, image.grid)
                },
                item: Some(item),
            });
        }
        Ok((prepared, media))
    }
}

fn source_digest(bytes: &[u8], context: Option<&CpuContext>) -> Result<SourceDigest> {
    let mut digest = Sha256::new();
    if let Some(context) = context {
        context.check()?;
    }
    for chunk in bytes.chunks(HASH_CHUNK_BYTES) {
        if let Some(context) = context {
            context.check()?;
        }
        digest.update(chunk);
    }
    Ok(SourceDigest(digest.finalize().into()))
}

#[cfg(test)]
fn image_digest(source: SourceDigest, processor: &ImageProcessor) -> Result<String> {
    image_digest_with_mode(source, processor, ResizeMode::Smart)
}

fn image_digest_with_mode(
    source: SourceDigest,
    processor: &ImageProcessor,
    resize: ResizeMode,
) -> Result<String> {
    image_digest_with_decoder(source, processor, resize, None)
}

fn image_digest_with_decoder(
    source: SourceDigest,
    processor: &ImageProcessor,
    resize: ResizeMode,
    decoder_identity: Option<&str>,
) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(IMAGE_IDENTITY_DOMAIN);
    digest.update(source.0);
    if let Some(identity) = decoder_identity {
        digest.update(b"device-decoder\0");
        digest.update(identity.as_bytes());
    }
    if resize == ResizeMode::Disabled {
        digest.update(b"resize-disabled");
    }
    for value in [
        processor.patch_size,
        processor.temporal_patch_size,
        processor.merge_size,
        processor.min_pixels,
        processor.max_pixels,
    ] {
        digest.update(
            u64::try_from(value)
                .map_err(|_| Error::invalid("image identity geometry overflow"))?
                .to_le_bytes(),
        );
    }
    for value in processor
        .mean
        .into_iter()
        .chain(processor.std)
        .chain([processor.rescale])
    {
        digest.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("sha256-source-v1:{:x}", digest.finalize()))
}

fn digest(pixels: &[f32], grid: (usize, usize, usize)) -> String {
    let mut hash = FNV_OFFSET_BASIS;
    for value in pixels {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    for value in <[usize; GRID_AXES]>::from(grid) {
        for byte in u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

/// Decode one image URL/data URI into RGB8.
/// # Errors
/// Rejects unsupported schemes, malformed bytes, overflowing dimensions and decode failures.
#[cfg(test)]
pub fn load_image(url: &str) -> Result<RawImage> {
    load_image_with(url, SourcePolicy::Public, None).map(|(image, _)| image)
}

#[cfg(test)]
fn load_image_with(
    url: &str,
    source: SourcePolicy,
    context: Option<&CpuContext>,
) -> Result<(RawImage, SourceDigest)> {
    let bytes = read_url(url, None, source, context)?;
    load_image_bytes(&bytes, context)
}

fn load_image_bytes(
    bytes: &[u8],
    context: Option<&CpuContext>,
) -> Result<(RawImage, SourceDigest)> {
    let identity = source_digest(bytes, context)?;
    match decode_rgb8(bytes) {
        Ok(rgb) => {
            let (width, height, pixels) = rgb.into_parts();
            return Ok((rgb_image(width, height, pixels)?, identity));
        }
        Err(error) if error.code == ErrorCode::Capacity => return Err(error),
        Err(_) => {}
    }
    // Try the existing external decoder for formats unavailable to the in-process decoder.
    let temp = materialize(bytes, "image")?;
    let probe = probe(&temp.path, context)?;
    let raw = decode_frames(
        &temp.path,
        probe.width,
        probe.height,
        1,
        context,
        FrameKind::Image,
    )?;
    let frame = raw
        .into_iter()
        .next()
        .ok_or_else(|| Error::invalid("image decoder produced no frame"))?;
    Ok((rgb_image(probe.width, probe.height, frame)?, identity))
}

/// Decode one video URL/data URI into source-rate RGB frames.
/// # Errors
/// Rejects unsupported schemes, over-limit payloads/geometry, missing ffmpeg and decode failures.
fn load_video_with(
    url: &str,
    source: SourcePolicy,
    context: Option<&CpuContext>,
) -> Result<RawVideo> {
    let bytes = read_url(url, Some(MAX_VIDEO_BYTES), source, context)?;
    load_video_bytes(&bytes, context)
}

fn load_video_bytes(bytes: &[u8], context: Option<&CpuContext>) -> Result<RawVideo> {
    let temp = materialize(bytes, "video")?;
    let probe = probe(&temp.path, context)?;
    if u64::from(probe.width) * u64::from(probe.height) > MAX_VIDEO_PIXELS as u64 {
        return Err(Error::new(
            ErrorCode::Capacity,
            "video frame dimensions exceed the bounded limit",
        ));
    }
    check_video_frame_limit(probe.total_frames)?;
    let frames = decode_frames(
        &temp.path,
        probe.width,
        probe.height,
        MAX_VIDEO_FRAMES,
        context,
        FrameKind::Video,
    )?;
    if frames.is_empty() {
        return Err(Error::invalid("video decoder produced no frames"));
    }
    let frames = frames
        .into_iter()
        .map(|pixels| rgb_frame(probe.width, probe.height, pixels))
        .collect::<Result<Vec<_>>>()?;
    Ok(RawVideo {
        fps: probe.fps,
        frames,
    })
}

fn check_image_dimensions(width: u32, height: u32) -> Result<()> {
    let bytes = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|pixels| pixels.checked_mul(RGB_CHANNELS));
    if width == 0 || height == 0 || bytes.is_none_or(|bytes| bytes > isize::MAX as usize) {
        return Err(Error::new(
            ErrorCode::Capacity,
            "image dimensions cannot be represented",
        ));
    }
    Ok(())
}

fn rgb_image(width: u32, height: u32, pixels: Vec<u8>) -> Result<RawImage> {
    let width = usize::try_from(width).map_err(|_| Error::invalid("image width overflow"))?;
    let height = usize::try_from(height).map_err(|_| Error::invalid("image height overflow"))?;
    let expected = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(RGB_CHANNELS))
        .ok_or_else(|| Error::invalid("image dimensions overflow"))?;
    check_image_dimensions(
        u32::try_from(width).unwrap_or(u32::MAX),
        u32::try_from(height).unwrap_or(u32::MAX),
    )?;
    if pixels.len() != expected {
        return Err(Error::invalid("decoded image buffer has the wrong size"));
    }
    Ok(RawImage {
        height,
        width,
        pixels,
    })
}

fn rgb_frame(width: u32, height: u32, pixels: Vec<u8>) -> Result<RgbImage> {
    let raw = rgb_image(width, height, pixels)?;
    Ok(RgbImage {
        height: raw.height,
        width: raw.width,
        pixels: raw.pixels,
    })
}

fn read_url(
    url: &str,
    limit: Option<usize>,
    source: SourcePolicy,
    context: Option<&CpuContext>,
) -> Result<Vec<u8>> {
    if url.is_empty() {
        return Err(Error::invalid("media URL is empty"));
    }
    if url.starts_with("data:") {
        return decode_data_uri(url, limit);
    }
    if let Some(path) = url.strip_prefix("file://") {
        require_local(source, url)?;
        return read_file(Path::new(path), limit);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return fetch_http(url, limit, context);
    }
    if Path::new(url).is_absolute() {
        require_local(source, url)?;
        return read_file(Path::new(url), limit);
    }
    Err(Error::unsupported(
        "media URL must be data:, file://, absolute path, http:// or https://",
    ))
}

fn require_local(source: SourcePolicy, url: &str) -> Result<()> {
    if source == SourcePolicy::TrustedLocal {
        Ok(())
    } else {
        Err(Error::unsupported(format!(
            "public media input cannot reference server-local path {url:?}"
        )))
    }
}

fn read_file(path: &Path, limit: Option<usize>) -> Result<Vec<u8>> {
    if let Some(limit) = limit {
        let metadata = metadata(path).map_err(|e| Error::invalid(e.to_string()))?;
        if metadata.len() > limit as u64 {
            return Err(Error::new(
                ErrorCode::Capacity,
                "media file exceeds the bounded limit",
            ));
        }
    }
    read(path).map_err(|e| Error::invalid(e.to_string()))
}

/// A bounded subprocess result with both streams drained.
#[derive(Debug)]
struct BoundedOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Run one external decoder with a wall-clock deadline and bounded both-stream draining.
///
/// The command owns its process group so a timeout can terminate any children it started. The
/// two reader threads are bounded by the caller's item budget and are joined before returning;
/// no child or reader survives an error path.
fn run_bounded(
    command: &mut Command,
    stdout_limit: usize,
    stderr_limit: usize,
    timeout: Duration,
    context: Option<&CpuContext>,
) -> Result<BoundedOutput> {
    if let Some(context) = context {
        context.check()?;
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .spawn()
        .map_err(|error| Error::unsupported(format!("media subprocess unavailable: {error}")))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::invariant("media subprocess stdout missing"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| Error::invariant("media subprocess stderr missing"))?;
    let out_reader = read_capped(stdout, stdout_limit);
    let err_reader = read_capped(stderr, stderr_limit);
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(context) = context
            && let Err(error) = context.check()
        {
            kill_group(child.id());
            let _ = child.kill();
            let _ = child.wait();
            let _ = out_reader.join();
            let _ = err_reader.join();
            return Err(error);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                kill_group(child.id());
                let _ = child.kill();
                let _ = child.wait();
                let _ = out_reader.join();
                let _ = err_reader.join();
                return Err(Error::new(ErrorCode::Backend, "media subprocess timed out"));
            }
            Ok(None) => thread::sleep(Duration::from_millis(PROCESS_POLL_MILLIS)),
            Err(error) => {
                kill_group(child.id());
                let _ = child.kill();
                let _ = child.wait();
                let _ = out_reader.join();
                let _ = err_reader.join();
                return Err(Error::invalid(error.to_string()));
            }
        }
    };
    let (stdout, stdout_overflow) = join_reader(out_reader)?;
    let (stderr, stderr_overflow) = join_reader(err_reader)?;
    if stdout_overflow || stderr_overflow {
        return Err(Error::new(
            ErrorCode::Capacity,
            "media subprocess output exceeds the bounded limit",
        ));
    }
    Ok(BoundedOutput {
        status,
        stdout,
        stderr,
    })
}

fn join_reader(handle: thread::JoinHandle<io::Result<(Vec<u8>, bool)>>) -> Result<(Vec<u8>, bool)> {
    handle
        .join()
        .map_err(|_| Error::invariant("media reader thread panicked"))?
        .map_err(|error| Error::invalid(error.to_string()))
}

fn read_capped(
    mut pipe: impl Read + Send + 'static,
    limit: usize,
) -> thread::JoinHandle<io::Result<(Vec<u8>, bool)>> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut overflow = false;
        let mut chunk = vec![0u8; READ_CHUNK_BYTES];
        loop {
            let read = pipe.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            if bytes.len().saturating_add(read) > limit {
                overflow = true;
            } else {
                bytes.extend_from_slice(&chunk[..read]);
            }
        }
        Ok((bytes, overflow))
    })
}

fn kill_group(process_group: u32) {
    #[cfg(unix)]
    {
        let _ = Command::new("/bin/kill")
            .args(["-KILL", "--", &format!("-{process_group}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(unix))]
    {
        let _ = process_group;
    }
}

fn fetch_http(url: &str, limit: Option<usize>, context: Option<&CpuContext>) -> Result<Vec<u8>> {
    let context = context.ok_or_else(|| Error::invariant("network media needs a CPU context"))?;
    context.check()?;
    context.runtime()?.block_on(async {
        let request = source::fetch(url, limit);
        tokio::pin!(request);
        loop {
            context.check()?;
            tokio::select! {
                result = &mut request => return result,
                () = async_sleep(Duration::from_millis(PROCESS_POLL_MILLIS)) => {}
            }
        }
    })
}

fn materialize(bytes: &[u8], kind: &str) -> Result<TempPath> {
    let (temp, mut file) = TempPath::create(kind, "media")?;
    file.write_all(bytes)
        .map_err(|e| Error::invalid(e.to_string()))?;
    Ok(temp)
}

/// Probe stream geometry and source frame metadata with ffprobe.
struct Probe {
    width: u32,
    height: u32,
    fps: f64,
    total_frames: Option<usize>,
}

fn probe(path: &Path, context: Option<&CpuContext>) -> Result<Probe> {
    let mut command = Command::new("ffprobe");
    command.args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        &path.to_string_lossy(),
    ]);
    let output = run_bounded(
        &mut command,
        SUBPROCESS_STREAM_BYTES,
        SUBPROCESS_STREAM_BYTES,
        Duration::from_secs(PROBE_TIMEOUT_SECONDS),
        context,
    )?;
    if !output.status.success() {
        return Err(Error::invalid(format!(
            "media probe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| Error::invalid(format!("media probe JSON: {error}")))?;
    let stream = value
        .get("streams")
        .and_then(serde_json::Value::as_array)
        .and_then(|streams| streams.first())
        .ok_or_else(|| Error::invalid("media has no video stream"))?;
    let number = |key: &str| -> Result<u64> {
        stream
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::invalid(format!("media {key} is missing")))
    };
    let width = u32::try_from(number("width")?).map_err(|_| Error::invalid("media width"))?;
    let height = u32::try_from(number("height")?).map_err(|_| Error::invalid("media height"))?;
    check_image_dimensions(width, height)?;
    let rate = stream
        .get("avg_frame_rate")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("30/1");
    let fps = parse_rate(rate)?;
    let total_frames = stream
        .get("nb_frames")
        .and_then(serde_json::Value::as_str)
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| {
            let duration = stream
                .get("duration")
                .and_then(serde_json::Value::as_str)
                .and_then(|value| value.parse::<f64>().ok())?;
            let estimate = (duration * fps).ceil();
            (estimate.is_finite() && estimate >= 0.0).then(|| usize_from_frames(estimate))
        });
    Ok(Probe {
        width,
        height,
        fps,
        total_frames,
    })
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "The estimate is finite, nonnegative and bounded by the frame limit check"
)]
fn usize_from_frames(value: f64) -> usize {
    if value > usize::MAX as f64 {
        usize::MAX
    } else {
        value as usize
    }
}

fn check_video_frame_limit(total_frames: Option<usize>) -> Result<()> {
    if total_frames.is_some_and(|frames| frames > MAX_VIDEO_FRAMES) {
        return Err(Error::new(
            ErrorCode::Capacity,
            "video frame count exceeds the bounded decoder limit",
        ));
    }
    Ok(())
}

fn parse_rate(rate: &str) -> Result<f64> {
    let (num, den) = rate.split_once('/').unwrap_or((rate, "1"));
    let num: f64 = num
        .parse()
        .map_err(|_| Error::invalid("media frame rate"))?;
    let den: f64 = den
        .parse()
        .map_err(|_| Error::invalid("media frame rate"))?;
    if !num.is_finite() || !den.is_finite() || num <= 0.0 || den <= 0.0 {
        return Err(Error::invalid("media frame rate must be positive"));
    }
    let fps = num / den;
    if !fps.is_finite() || fps <= 0.0 || fps > MAX_SOURCE_FPS {
        return Err(Error::invalid(
            "media frame rate is outside the supported range",
        ));
    }
    Ok(fps)
}

/// Decode bounded RGB24 frames through ffmpeg.
fn decode_frames(
    path: &Path,
    width: u32,
    height: u32,
    frames: usize,
    context: Option<&CpuContext>,
    kind: FrameKind,
) -> Result<Vec<Vec<u8>>> {
    let frame_bytes = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|n| n.checked_mul(RGB_CHANNELS))
        .ok_or_else(|| Error::invalid("media frame size overflow"))?;
    let total_bytes = frame_bytes
        .checked_mul(frames)
        .ok_or_else(|| Error::invalid("decoded media size overflow"))?;
    if frame_bytes == 0 || (kind == FrameKind::Video && total_bytes > MAX_VIDEO_BYTES) {
        return Err(Error::new(
            ErrorCode::Capacity,
            "decoded media exceeds the bounded limit",
        ));
    }
    let requested = frames.saturating_add(1);
    let mut command = Command::new("ffmpeg");
    command.args([
        "-v",
        "error",
        "-nostdin",
        "-i",
        &path.to_string_lossy(),
        "-map",
        "0:v:0",
        "-an",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        &requested.to_string(),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]);
    let output = run_bounded(
        &mut command,
        if kind == FrameKind::Image {
            total_bytes
        } else {
            MAX_VIDEO_BYTES
        },
        SUBPROCESS_STREAM_BYTES,
        Duration::from_secs(DECODE_TIMEOUT_SECONDS),
        context,
    )?;
    if !output.status.success() {
        return Err(Error::invalid(format!(
            "media decode failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    if !output.stdout.len().is_multiple_of(frame_bytes) {
        return Err(Error::invalid("decoded media byte count is inconsistent"));
    }
    let count = output.stdout.len() / frame_bytes;
    if count > frames {
        return Err(Error::new(
            ErrorCode::Capacity,
            "video exceeds the bounded decoder frame limit",
        ));
    }
    Ok(output
        .stdout
        .chunks_exact(frame_bytes)
        .map(<[u8]>::to_vec)
        .collect())
}

fn create_exclusive(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(TEMP_FILE_MODE);
    options
        .open(path)
        .map_err(|e| Error::invalid(e.to_string()))
}

/// Temporary path removed on drop.
struct TempPath {
    path: PathBuf,
}
impl TempPath {
    #[cfg(test)]
    fn create_at(path: PathBuf) -> Result<(Self, File)> {
        let file = create_exclusive(&path)?;
        Ok((Self { path }, file))
    }

    fn create(kind: &str, extension: &str) -> Result<(Self, File)> {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| Error::invalid(e.to_string()))?
            .as_nanos();
        let path = temp_dir().join(format!(
            "inferloop-{kind}-{}-{nanos}-{sequence}.{extension}",
            id()
        ));
        let file = create_exclusive(&path)?;
        Ok((Self { path }, file))
    }
}
impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = remove_file(&self.path);
    }
}

#[cfg(test)]
#[path = "../tests/unit/media.rs"]
mod tests;
