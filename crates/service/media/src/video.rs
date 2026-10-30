//! Asynchronous probing and sampled frame decoding for seekable video sources.
use crate::DecodedImage;
use infer_core::{Error, ErrorCode, Result};
use serde_json::Value;
use std::mem::take;
use std::path::Path;
use std::process::Stdio;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, ChildStdout, Command};

/// A frame index above this value is not exact in ffmpeg's floating-point filter expressions.
const MAX_EXACT_FILTER_INDEX: u64 = 1 << 53;
/// Channels in a decoded RGB frame.
const RGB_CHANNELS: usize = 3;

/// Video stream properties and its exact decoded frame count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoMetadata {
    width: u32,
    height: u32,
    fps: f64,
    frames: usize,
}

impl VideoMetadata {
    /// Source frame width in pixels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }
    /// Source frame height in pixels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
    /// Average source frame rate, used by the model's sampling contract.
    #[must_use]
    pub const fn fps(self) -> f64 {
        self.fps
    }
    /// Actual source frame count, not an estimate from duration.
    #[must_use]
    pub const fn frames(self) -> usize {
        self.frames
    }
}

/// Probe a seekable video and count decoded frames to preserve the model's sampling contract.
///
/// No runtime worker is held while the external probe runs. Dropping the future kills its child.
/// The caller must keep the input file available for the probe and subsequent decoding.
///
/// # Errors
/// Rejects missing tools, malformed streams, unsupported geometry or a failed frame count.
pub async fn probe_video(path: &Path) -> Result<VideoMetadata> {
    let stream = probe(path).await?;
    metadata(&stream, "nb_read_frames")?
        .ok_or_else(|| Error::invalid("video decoder did not report a frame count"))
}

async fn probe(path: &Path) -> Result<Value> {
    let mut command = Command::new("ffprobe");
    command.kill_on_drop(true).args([
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        "-count_frames",
    ]);
    let output = command.arg(path).output().await.map_err(|error| {
        Error::new(
            ErrorCode::Backend,
            format!("could not start media probe: {error}"),
        )
    })?;
    if !output.status.success() {
        return Err(Error::invalid(format!(
            "media probe failed: {}",
            output.status
        )));
    }
    let value: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| Error::invalid(format!("media probe JSON: {error}")))?;
    value
        .get("streams")
        .and_then(Value::as_array)
        .and_then(|streams| streams.first())
        .cloned()
        .ok_or_else(|| Error::invalid("media has no video stream"))
}

fn metadata(stream: &Value, count_field: &str) -> Result<Option<VideoMetadata>> {
    let dimension = |name: &str| -> Result<u32> {
        let value = stream
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value != 0)
            .ok_or_else(|| Error::invalid(format!("video {name} is missing or invalid")))?;
        Ok(value)
    };
    let rate = stream
        .get("avg_frame_rate")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::invalid("video frame rate is missing"))?;
    let (num, den) = rate.split_once('/').unwrap_or((rate, "1"));
    let num: f64 = num
        .parse()
        .map_err(|_| Error::invalid("video frame rate"))?;
    let den: f64 = den
        .parse()
        .map_err(|_| Error::invalid("video frame rate"))?;
    let fps = num / den;
    if !num.is_finite() || !den.is_finite() || den <= 0.0 || !fps.is_finite() || fps <= 0.0 {
        return Err(Error::invalid("video frame rate must be positive"));
    }
    let frames = stream
        .get(count_field)
        .and_then(Value::as_str)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0);
    let width = dimension("width")?;
    let height = dimension("height")?;
    Ok(frames.map(|frames| VideoMetadata {
        width,
        height,
        fps,
        frames,
    }))
}

/// A stream of selected RGB frames backed by one asynchronous decoder process.
///
/// The input file must remain available until the stream is finished or dropped. Dropping this
/// value terminates the child. Partial reads are retained if a `next_frame` future is cancelled.
pub struct SampledVideo {
    child: Child,
    stdout: ChildStdout,
    metadata: VideoMetadata,
    frame_bytes: usize,
    buffer: Vec<u8>,
    filled: usize,
    remaining: usize,
    complete: bool,
}

impl SampledVideo {
    /// Start decoding the strictly increasing source frame indices in presentation order.
    ///
    /// Only selected RGB frames cross the process boundary; unselected decoded frames are dropped
    /// by ffmpeg. At most one unfinished frame is retained by this reader.
    ///
    /// # Errors
    /// Rejects invalid indices, unrepresentable frames or failure to start the decoder.
    pub fn start(path: &Path, metadata: VideoMetadata, indices: &[usize]) -> Result<Self> {
        if indices.is_empty()
            || indices.windows(2).any(|pair| pair[0] >= pair[1])
            || indices.last().is_none_or(|index| *index >= metadata.frames)
            || indices.last().is_some_and(|index| {
                u64::try_from(*index).map_or(true, |index| index > MAX_EXACT_FILTER_INDEX)
            })
        {
            return Err(Error::invalid(
                "video frame indices must be ordered and in range",
            ));
        }
        let frame_bytes = usize::try_from(metadata.width)
            .ok()
            .and_then(|width| {
                usize::try_from(metadata.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(RGB_CHANNELS))
            .filter(|bytes| *bytes > 0 && isize::try_from(*bytes).is_ok())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::Capacity,
                    "video frame size cannot be represented",
                )
            })?;
        let filter = format!("select={}", selection(indices));
        let mut command = Command::new("ffmpeg");
        command
            .kill_on_drop(true)
            .args(["-v", "quiet", "-nostdin", "-i"])
            .arg(path)
            .args(["-map", "0:v:0", "-an", "-sn", "-vf"])
            .arg(filter)
            .args(["-fps_mode", "passthrough", "-frames:v"])
            .arg(indices.len().to_string())
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = command.spawn().map_err(|error| {
            Error::new(
                ErrorCode::Backend,
                format!("could not start video decoder: {error}"),
            )
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::invariant("video decoder has no stdout"))?;
        Ok(Self {
            child,
            stdout,
            metadata,
            frame_bytes,
            buffer: Vec::new(),
            filled: 0,
            remaining: indices.len(),
            complete: false,
        })
    }

    /// Read the next selected frame, or `None` after the decoder successfully exits.
    ///
    /// # Errors
    /// Reports truncated output, allocation failures, or a failed decoder exit.
    pub async fn next_frame(&mut self) -> Result<Option<DecodedImage>> {
        if self.complete {
            return Ok(None);
        }
        if self.remaining == 0 {
            self.finish().await?;
            return Ok(None);
        }
        if self.buffer.is_empty() {
            self.buffer
                .try_reserve_exact(self.frame_bytes)
                .map_err(|error| {
                    Error::new(
                        ErrorCode::Capacity,
                        format!("video frame allocation: {error}"),
                    )
                })?;
            self.buffer.resize(self.frame_bytes, 0);
        }
        while self.filled < self.frame_bytes {
            let read = self
                .stdout
                .read(&mut self.buffer[self.filled..])
                .await
                .map_err(|error| Error::invalid(format!("video frame read: {error}")))?;
            if read == 0 {
                self.finish().await?;
                return Err(Error::invalid(
                    "video decoder ended before all selected frames",
                ));
            }
            self.filled += read;
        }
        self.filled = 0;
        self.remaining -= 1;
        Ok(Some(DecodedImage::from_rgb8(
            self.metadata.width,
            self.metadata.height,
            take(&mut self.buffer),
        )?))
    }

    async fn finish(&mut self) -> Result<()> {
        let status = self.child.wait().await.map_err(|error| {
            Error::new(ErrorCode::Backend, format!("video decoder wait: {error}"))
        })?;
        self.complete = true;
        if !status.success() {
            return Err(Error::invalid(format!("video decoder failed: {status}")));
        }
        Ok(())
    }
}

/// Build a balanced filter tree so selection costs logarithmic comparisons per decoded frame.
fn selection(indices: &[usize]) -> String {
    if indices.len() == 1 {
        return format!("eq(n\\,{})", indices[0]);
    }
    let middle = indices.len() / 2;
    let left = selection(&indices[..middle]);
    let right = selection(&indices[middle..]);
    format!("if(lt(n\\,{})\\,{left}\\,{right})", indices[middle])
}

#[cfg(test)]
#[path = "../tests/unit/video.rs"]
mod tests;
