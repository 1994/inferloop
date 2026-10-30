//! In-process decoding into checked RGB8 pixels.
use infer_core::{Error, ErrorCode, Result};
use std::io::Cursor;

/// Channels in a decoded RGB image.
const CHANNELS: usize = 3;

/// An owned, row-major RGB8 image whose dimensions match its buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl DecodedImage {
    /// Consume the image and return its width, height and RGB8 bytes.
    #[must_use]
    pub fn into_parts(self) -> (u32, u32, Vec<u8>) {
        (self.width, self.height, self.pixels)
    }
}

/// Decode an image supported by the configured in-process codecs into RGB8.
///
/// # Errors
/// Returns an error for unknown or malformed formats, unrepresentable dimensions or decode errors.
pub fn decode_rgb8(bytes: &[u8]) -> Result<DecodedImage> {
    let mut probe = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| Error::invalid(format!("image format: {error}")))?;
    probe.limits(image::Limits::no_limits());
    let (width, height) = probe
        .into_dimensions()
        .map_err(|error| Error::invalid(format!("image dimensions: {error}")))?;
    let size = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|pixels| pixels.checked_mul(CHANNELS));
    if width == 0 || height == 0 || size.is_none_or(|size| size > isize::MAX as usize) {
        return Err(Error::new(
            ErrorCode::Capacity,
            "image dimensions cannot be represented",
        ));
    }
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| Error::invalid(format!("image format: {error}")))?;
    reader.limits(image::Limits::no_limits());
    let image = reader
        .decode()
        .map_err(|error| Error::invalid(format!("image decode: {error}")))?
        .into_rgb8();
    Ok(DecodedImage {
        width: image.width(),
        height: image.height(),
        pixels: image.into_raw(),
    })
}
