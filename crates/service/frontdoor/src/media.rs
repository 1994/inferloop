//! Bounded URL/data/media preparation for the `OpenAI` adapter.
//!
//! Decoding and network access stay in this input layer; the engine receives only preprocessed
//! patch payloads plus their prompt spans. Every read is size-bounded and every external command
//! gets a hard time/frame bound. A malformed or over-limit input is an explicit error, never a
//! text-only fallback.
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use infer_core::{Error, ErrorCode, MediaId, Result};
use infer_ir::{MediaInput, MediaItem, Modality};
use infer_models::{
    MediaLimits, MultimodalAssets, PreparedPrompt, RawImage, RawVideo, RgbImage, TextAssets,
};
use std::env::temp_dir;
use std::fs::{metadata, read, remove_file, write};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, id};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Largest accepted image payload after decoding or download.
const MAX_IMAGE_BYTES: usize = 24 * 1024 * 1024;
/// Largest accepted raw RGB image pixel count.
const MAX_IMAGE_PIXELS: usize = 16 * 1024 * 1024;
/// Largest accepted video download/decoded raw output.
const MAX_VIDEO_BYTES: usize = 24 * 1024 * 1024;
/// Host staging credit reserved for one media item before decode.
pub const MEDIA_STAGING_BYTES: usize = MAX_VIDEO_BYTES + 1024 * 1024;
/// Largest accepted frame count decoded from one clip.
const MAX_VIDEO_FRAMES: usize = 64;
/// Longest accepted clip duration handed to ffmpeg.
const MAX_VIDEO_SECONDS: u32 = 10;
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
/// Lowest accepted source frame rate.
const MIN_FPS: f64 = 1.0;
/// Highest accepted source frame rate.
const MAX_FPS: f64 = 240.0;
/// HTTP fetch timeout.
const HTTP_TIMEOUT_SECONDS: u32 = 20;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

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
}
impl MediaService {
    #[must_use]
    pub const fn new(assets: Arc<MultimodalAssets>, limits: MediaLimits) -> Self {
        Self { assets, limits }
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
        let prepared = match modality {
            Modality::Image => {
                let images = requests
                    .iter()
                    .map(|request| load_image(&request.url))
                    .collect::<Result<Vec<_>>>()?;
                self.assets.prepare_images(assets, prompt, &images)?
            }
            Modality::Video => {
                let videos = requests
                    .iter()
                    .map(|request| load_video(&request.url))
                    .collect::<Result<Vec<_>>>()?;
                self.assets.prepare_video(assets, prompt, &videos)?
            }
            Modality::Text | Modality::Audio | Modality::Extension(_) => {
                return Err(Error::unsupported("unsupported media modality"));
            }
        };
        if prepared.images.len() != prepared.spans.len() {
            return Err(Error::invalid("prepared media spans are inconsistent"));
        }
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
                content_digest: digest(&item),
                item: Some(item),
            });
        }
        Ok((prepared, media))
    }
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
pub fn load_image(url: &str) -> Result<RawImage> {
    let bytes = read_url(url, MAX_IMAGE_BYTES)?;
    if let Ok(image) = image::load_from_memory(&bytes) {
        let rgb = image.to_rgb8();
        return rgb_image(rgb.width(), rgb.height(), rgb.into_raw());
    }
    // JPEG/WebP may not be enabled in the host image crate; ffmpeg covers those formats.
    let temp = materialize(&bytes, "image")?;
    let (width, height, _) = probe(&temp.path)?;
    let raw = decode_frames(&temp.path, width, height, 1)?;
    let frame = raw
        .into_iter()
        .next()
        .ok_or_else(|| Error::invalid("image decoder produced no frame"))?;
    rgb_image(width, height, frame)
}

/// Decode one video URL/data URI into source-rate RGB frames.
/// # Errors
/// Rejects unsupported schemes, over-limit payloads/geometry, missing ffmpeg and decode failures.
pub fn load_video(url: &str) -> Result<RawVideo> {
    let bytes = read_url(url, MAX_VIDEO_BYTES)?;
    let temp = materialize(&bytes, "video")?;
    let (width, height, fps) = probe(&temp.path)?;
    let frames = decode_frames(&temp.path, width, height, MAX_VIDEO_FRAMES)?;
    if frames.is_empty() {
        return Err(Error::invalid("video decoder produced no frames"));
    }
    let frames = frames
        .into_iter()
        .map(|pixels| rgb_frame(width, height, pixels))
        .collect::<Result<Vec<_>>>()?;
    Ok(RawVideo { fps, frames })
}

fn rgb_image(width: u32, height: u32, pixels: Vec<u8>) -> Result<RawImage> {
    let width = usize::try_from(width).map_err(|_| Error::invalid("image width overflow"))?;
    let height = usize::try_from(height).map_err(|_| Error::invalid("image height overflow"))?;
    let expected = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(RGB_CHANNELS))
        .ok_or_else(|| Error::invalid("image dimensions overflow"))?;
    if width == 0 || height == 0 || width.saturating_mul(height) > MAX_IMAGE_PIXELS {
        return Err(Error::new(
            ErrorCode::Capacity,
            "image dimensions exceed the bounded media limit",
        ));
    }
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

fn read_url(url: &str, limit: usize) -> Result<Vec<u8>> {
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
        return read_file(Path::new(path), limit);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return fetch_http(url, limit);
    }
    if Path::new(url).is_absolute() {
        return read_file(Path::new(url), limit);
    }
    Err(Error::unsupported(
        "media URL must be data:, file://, absolute path, http:// or https://",
    ))
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

fn fetch_http(url: &str, limit: usize) -> Result<Vec<u8>> {
    let temp = TempPath::new("download", "media")?;
    let status = Command::new("curl")
        .args([
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
        ])
        .status()
        .map_err(|error| Error::unsupported(format!("curl is required for media URLs: {error}")))?;
    if !status.success() {
        return Err(Error::invalid(format!(
            "media download failed with {status}"
        )));
    }
    read_file(&temp.path, limit)
}

fn materialize(bytes: &[u8], kind: &str) -> Result<TempPath> {
    let temp = TempPath::new(kind, "media")?;
    write(&temp.path, bytes).map_err(|e| Error::invalid(e.to_string()))?;
    Ok(temp)
}

/// Probe stream geometry with ffprobe.
fn probe(path: &Path) -> Result<(u32, u32, f64)> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            &path.to_string_lossy(),
        ])
        .output()
        .map_err(|error| Error::unsupported(format!("ffprobe is required for media: {error}")))?;
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
    Ok((width, height, fps))
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
    Ok((num / den).clamp(MIN_FPS, MAX_FPS))
}

/// Decode bounded RGB24 frames through ffmpeg.
fn decode_frames(path: &Path, width: u32, height: u32, frames: usize) -> Result<Vec<Vec<u8>>> {
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
    let mut child = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostdin",
            "-i",
            &path.to_string_lossy(),
            "-map",
            "0:v:0",
            "-an",
            "-vsync",
            "0",
            "-frames:v",
            &frames.to_string(),
            "-t",
            &MAX_VIDEO_SECONDS.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| Error::unsupported(format!("ffmpeg is required for media: {error}")))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::invariant("ffmpeg stdout missing"))?;
    let mut bytes = Vec::new();
    stdout
        .by_ref()
        .take((MAX_VIDEO_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::invalid(e.to_string()))?;
    let output = child
        .wait_with_output()
        .map_err(|e| Error::invalid(e.to_string()))?;
    if !output.status.success() {
        return Err(Error::invalid(format!(
            "media decode failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    if bytes.len() > MAX_VIDEO_BYTES || !bytes.len().is_multiple_of(frame_bytes) {
        return Err(Error::invalid("decoded media byte count is inconsistent"));
    }
    Ok(bytes
        .chunks_exact(frame_bytes)
        .map(<[u8]>::to_vec)
        .collect())
}

/// Temporary path removed on drop.
struct TempPath {
    path: PathBuf,
}
impl TempPath {
    fn new(kind: &str, extension: &str) -> Result<Self> {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| Error::invalid(e.to_string()))?
            .as_nanos();
        let path = temp_dir().join(format!(
            "inferloop-{kind}-{}-{nanos}-{sequence}.{extension}",
            id()
        ));
        Ok(Self { path })
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
