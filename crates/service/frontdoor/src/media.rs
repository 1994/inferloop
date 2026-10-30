//! Bounded URL/data/media preparation for the `OpenAI` adapter.
//!
//! Decoding and network access stay in this input layer; the engine receives only preprocessed
//! patch payloads plus their prompt spans. Every read is size-bounded and every external command
//! gets a hard time/frame bound. A malformed or over-limit input is an explicit error, never a
//! text-only fallback.
use crate::cpu::CpuContext;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use infer_core::{Error, ErrorCode, MediaId, Result};
use infer_ir::{MediaInput, MediaItem, Modality};
use infer_models::{
    ImageProcessor, MediaLimits, MultimodalAssets, PreparedPrompt, RawImage, RawVideo, RgbImage,
    TextAssets,
};
use sha2::{Digest, Sha256};
use std::env::temp_dir;
use std::fs::{File, OpenOptions, metadata, read, remove_file};
use std::io::{self, Cursor, Read, Write};
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

/// Largest accepted image payload after decoding or download.
const MAX_IMAGE_BYTES: usize = 24 * 1024 * 1024;
/// Largest accepted raw RGB image pixel count.
const MAX_IMAGE_PIXELS: usize = 16 * 1024 * 1024;
/// Largest accepted image edge during in-process decode.
const MAX_IMAGE_EDGE: u32 = 8192;
/// Largest accepted decoder allocation during in-process decode.
const MAX_IMAGE_ALLOC_BYTES: u64 = 64 << 20;
/// Largest accepted video download/decoded raw output.
const MAX_VIDEO_BYTES: usize = 24 * 1024 * 1024;
/// Largest accepted frame count decoded from one clip.
const MAX_VIDEO_FRAMES: usize = 768;
/// Chunk size for bounded decoder output draining.
const READ_CHUNK_BYTES: usize = 64 * 1024;
/// Check cancellation between source-hash chunks, including for large input files.
const HASH_CHUNK_BYTES: usize = 64 * 1024;
/// Identity namespace for the CPU image decoder and preprocessing contract.
const IMAGE_IDENTITY_DOMAIN: &[u8] = b"inferloop-image-source-cpu-v1\0";
/// Largest accepted URL or data-URI text.
const MAX_URL_BYTES: usize = 8192;
/// Base64 expansion numerator for a bounded data-URI text length.
const BASE64_EXPANSION_NUMERATOR: usize = 4;
/// Base64 expansion denominator for a bounded data-URI text length.
const BASE64_EXPANSION_DENOMINATOR: usize = 3;
/// Header and MIME headroom allowed around a base64 data-URI payload.
const DATA_URI_HEADER_BYTES: usize = 4096;
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
/// HTTP fetch timeout.
const HTTP_TIMEOUT_SECONDS: u32 = 20;
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

#[derive(Clone, Copy)]
struct SourceDigest([u8; 32]);

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
}
impl MediaService {
    #[must_use]
    pub const fn new(assets: Arc<MultimodalAssets>, limits: MediaLimits) -> Self {
        Self {
            assets,
            limits,
            source: SourcePolicy::Public,
        }
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
        let (prepared, image_sources) = match modality {
            Modality::Image => {
                let decoded = requests
                    .iter()
                    .map(|request| load_image_with(&request.url, self.source, Some(context)))
                    .collect::<Result<Vec<_>>>()?;
                let (images, sources): (Vec<_>, Vec<_>) = decoded.into_iter().unzip();
                (
                    self.assets.prepare_images(assets, prompt, &images)?,
                    sources,
                )
            }
            Modality::Video => {
                let videos = requests
                    .iter()
                    .map(|request| load_video_with(&request.url, self.source, Some(context)))
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
                pixels: Arc::from(image.pixels.as_slice()),
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
                    image_digest(source, self.assets.image_processor())?
                } else {
                    digest(&item)
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

fn image_digest(source: SourceDigest, processor: &ImageProcessor) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(IMAGE_IDENTITY_DOMAIN);
    digest.update(source.0);
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

fn digest(item: &MediaItem) -> String {
    let mut hash = FNV_OFFSET_BASIS;
    for value in item.pixels.iter() {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    for value in <[usize; GRID_AXES]>::from(item.grid) {
        for byte in u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

/// Decode one image URL/data URI into bounded RGB8.
/// # Errors
/// Rejects unsupported schemes, over-limit payloads, malformed bytes and decode failures.
#[cfg(test)]
pub fn load_image(url: &str) -> Result<RawImage> {
    load_image_with(url, SourcePolicy::Public, None).map(|(image, _)| image)
}

fn load_image_with(
    url: &str,
    source: SourcePolicy,
    context: Option<&CpuContext>,
) -> Result<(RawImage, SourceDigest)> {
    let bytes = read_url(url, MAX_IMAGE_BYTES, source, context)?;
    let identity = source_digest(&bytes, context)?;
    match decode_png(&bytes) {
        Ok(rgb) => {
            return Ok((
                rgb_image(rgb.width(), rgb.height(), rgb.into_raw())?,
                identity,
            ));
        }
        Err(error) if error.code == ErrorCode::Capacity => return Err(error),
        Err(_) => {}
    }
    // A format the in-process decoder cannot identify (or a corrupted stream) is an explicit
    // decode failure; ffmpeg is retained only for the video container path.
    let temp = materialize(&bytes, "image")?;
    let probe = probe(&temp.path, context)?;
    let raw = decode_frames(&temp.path, probe.width, probe.height, 1, context)?;
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
    let bytes = read_url(url, MAX_VIDEO_BYTES, source, context)?;
    let temp = materialize(&bytes, "video")?;
    let probe = probe(&temp.path, context)?;
    check_video_frame_limit(probe.total_frames)?;
    let frames = decode_frames(
        &temp.path,
        probe.width,
        probe.height,
        MAX_VIDEO_FRAMES,
        context,
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
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    if width == 0 || height == 0 || pixels > MAX_IMAGE_PIXELS as u64 {
        return Err(Error::new(
            ErrorCode::Capacity,
            "image dimensions exceed the bounded media limit",
        ));
    }
    Ok(())
}

fn decode_png(bytes: &[u8]) -> Result<image::RgbImage> {
    // Read the header before materializing the full image. A compressed high-dimension file
    // then fails here rather than allocating the claimed dimensions.
    let probe = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| Error::invalid(format!("image format: {error}")))?;
    let (width, height) = probe
        .into_dimensions()
        .map_err(|error| Error::invalid(format!("image dimensions: {error}")))?;
    check_image_dimensions(width, height)?;
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| Error::invalid(format!("image format: {error}")))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_EDGE);
    limits.max_image_height = Some(MAX_IMAGE_EDGE);
    limits.max_alloc = Some(MAX_IMAGE_ALLOC_BYTES);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|error| Error::invalid(format!("image decode: {error}")))?;
    Ok(image.into_rgb8())
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
    limit: usize,
    source: SourcePolicy,
    context: Option<&CpuContext>,
) -> Result<Vec<u8>> {
    if url.is_empty() {
        return Err(Error::invalid("media URL is empty"));
    }
    if url.starts_with("data:") {
        let encoded_limit = limit.saturating_mul(BASE64_EXPANSION_NUMERATOR)
            / BASE64_EXPANSION_DENOMINATOR
            + DATA_URI_HEADER_BYTES;
        if url.len() > encoded_limit {
            return Err(Error::new(
                ErrorCode::Capacity,
                "media data URI text exceeds the bounded limit",
            ));
        }
        return read_data_uri(url, limit);
    }
    if url.len() > MAX_URL_BYTES {
        return Err(Error::invalid("media URL is too long"));
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

fn read_data_uri(url: &str, limit: usize) -> Result<Vec<u8>> {
    let (header, payload) = url
        .split_once(',')
        .ok_or_else(|| Error::invalid("malformed data URI"))?;
    if !header.ends_with(";base64") {
        return Err(Error::unsupported("only base64 data URIs are supported"));
    }
    let bytes = BASE64
        .decode(payload)
        .map_err(|error| Error::invalid(format!("data URI base64: {error}")))?;
    if bytes.len() > limit {
        return Err(Error::new(
            ErrorCode::Capacity,
            "media data URI exceeds the bounded limit",
        ));
    }
    Ok(bytes)
}

fn read_file(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let metadata = metadata(path).map_err(|e| Error::invalid(e.to_string()))?;
    if metadata.len() > limit as u64 {
        return Err(Error::new(
            ErrorCode::Capacity,
            "media file exceeds the bounded limit",
        ));
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

fn fetch_http(url: &str, limit: usize, context: Option<&CpuContext>) -> Result<Vec<u8>> {
    let (temp, file) = TempPath::create("download", "media")?;
    drop(file);
    let mut command = Command::new("curl");
    command.args([
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--max-time",
        &HTTP_TIMEOUT_SECONDS.to_string(),
        "--max-filesize",
        &limit.to_string(),
        "--output",
        &temp.path.to_string_lossy(),
        url,
    ]);
    let output = run_bounded(
        &mut command,
        SUBPROCESS_STREAM_BYTES,
        SUBPROCESS_STREAM_BYTES,
        Duration::from_secs(u64::from(HTTP_TIMEOUT_SECONDS) + PROBE_TIMEOUT_SECONDS),
        context,
    )?;
    if !output.status.success() {
        return Err(Error::invalid(format!(
            "media download failed; {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    read_file(&temp.path, limit)
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
    if u64::from(width).saturating_mul(u64::from(height)) > MAX_IMAGE_PIXELS as u64 {
        return Err(Error::new(
            ErrorCode::Capacity,
            "media frame dimensions exceed the bounded limit",
        ));
    }
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
) -> Result<Vec<Vec<u8>>> {
    let frame_bytes = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|n| n.checked_mul(RGB_CHANNELS))
        .ok_or_else(|| Error::invalid("media frame size overflow"))?;
    if frame_bytes == 0 || frame_bytes.saturating_mul(frames) > MAX_VIDEO_BYTES {
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
        MAX_VIDEO_BYTES,
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
