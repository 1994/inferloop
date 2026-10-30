//! Backend-independent contracts for deferred RGB8 image preparation.
use infer_core::{Error, Result};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

/// Channels in one interleaved RGB pixel.
const RGB_CHANNELS: usize = 3;

/// Image patch geometry selected by the model's processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImagePatchLayout {
    patch: usize,
    temporal: usize,
    merge: usize,
}

impl ImagePatchLayout {
    /// # Errors
    /// Rejects zero dimensions and overflowing output geometry.
    pub fn new(patch: usize, temporal: usize, merge: usize) -> Result<Self> {
        if patch == 0 || temporal == 0 || merge == 0 {
            return Err(Error::invalid("image patch dimensions must be nonzero"));
        }
        patch
            .checked_mul(patch)
            .and_then(|value| value.checked_mul(temporal))
            .and_then(|value| value.checked_mul(RGB_CHANNELS))
            .and_then(|_| patch.checked_mul(merge))
            .ok_or_else(|| Error::invalid("image patch dimensions overflow"))?;
        Ok(Self {
            patch,
            temporal,
            merge,
        })
    }

    #[must_use]
    pub const fn patch(self) -> usize {
        self.patch
    }
    #[must_use]
    pub const fn temporal(self) -> usize {
        self.temporal
    }
    #[must_use]
    pub const fn merge(self) -> usize {
        self.merge
    }
}

/// Channel-wise rescaling and normalization for image pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageNormalization {
    mean: [f32; RGB_CHANNELS],
    std: [f32; RGB_CHANNELS],
    rescale: f32,
}

impl ImageNormalization {
    /// # Errors
    /// Rejects nonfinite values and zero standard deviations.
    pub fn new(mean: [f32; RGB_CHANNELS], std: [f32; RGB_CHANNELS], rescale: f32) -> Result<Self> {
        if !rescale.is_finite()
            || mean.iter().any(|value| !value.is_finite())
            || std.iter().any(|value| !value.is_finite() || *value == 0.0)
        {
            return Err(Error::invalid("invalid image normalization"));
        }
        Ok(Self { mean, std, rescale })
    }

    #[must_use]
    pub const fn mean(self) -> [f32; RGB_CHANNELS] {
        self.mean
    }
    #[must_use]
    pub const fn std(self) -> [f32; RGB_CHANNELS] {
        self.std
    }
    #[must_use]
    pub const fn rescale(self) -> f32 {
        self.rescale
    }
}

/// Validated RGB8 source dimensions and the requested processor output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageTransform {
    source: [usize; 2],
    target: [usize; 2],
    layout: ImagePatchLayout,
    normalization: ImageNormalization,
}

impl ImageTransform {
    /// # Errors
    /// Rejects empty, overflowing or incompatible source and target dimensions.
    pub fn new(
        source: [usize; 2],
        target: [usize; 2],
        layout: ImagePatchLayout,
        normalization: ImageNormalization,
    ) -> Result<Self> {
        let factor = layout.patch * layout.merge;
        if source.contains(&0)
            || target.contains(&0)
            || !target.iter().all(|axis| axis.is_multiple_of(factor))
        {
            return Err(Error::invalid("image source or target dimensions"));
        }
        rgb_bytes(source)?;
        rgb_bytes(target)?
            .checked_mul(layout.temporal)
            .and_then(|value| value.checked_mul(size_of::<f32>()))
            .ok_or_else(|| Error::invalid("image target dimensions overflow"))?;
        Ok(Self {
            source,
            target,
            layout,
            normalization,
        })
    }

    #[must_use]
    pub const fn source(self) -> [usize; 2] {
        self.source
    }
    #[must_use]
    pub const fn target(self) -> [usize; 2] {
        self.target
    }
    #[must_use]
    pub const fn layout(self) -> ImagePatchLayout {
        self.layout
    }
    #[must_use]
    pub const fn normalization(self) -> ImageNormalization {
        self.normalization
    }
    #[must_use]
    pub const fn grid(self) -> (usize, usize, usize) {
        (
            1,
            self.target[0] / self.layout.patch,
            self.target[1] / self.layout.patch,
        )
    }
}

/// Decoded image retained by the request while its backend processing is in flight.
#[derive(Clone, PartialEq)]
pub struct ImageInput {
    transform: ImageTransform,
    pixels: Arc<Vec<u8>>,
}

/// Compressed image format understood by a backend decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageEncoding {
    /// A JPEG bitstream.
    Jpeg,
}

/// Metadata checked by a backend before accepting a compressed image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedImageInfo {
    source: [usize; 2],
    encoding: ImageEncoding,
    decoder_identity: String,
}

impl EncodedImageInfo {
    /// Construct metadata for a nonempty source and a stable decoder identity.
    /// # Errors
    /// Rejects empty or overflowing dimensions and an empty decoder identity.
    pub fn new(
        source: [usize; 2],
        encoding: ImageEncoding,
        decoder_identity: String,
    ) -> Result<Self> {
        if source.contains(&0) || decoder_identity.is_empty() {
            return Err(Error::invalid("invalid compressed image metadata"));
        }
        rgb_bytes(source)?;
        Ok(Self {
            source,
            encoding,
            decoder_identity,
        })
    }

    #[must_use]
    pub const fn source(&self) -> [usize; 2] {
        self.source
    }
    #[must_use]
    pub const fn encoding(&self) -> ImageEncoding {
        self.encoding
    }
    #[must_use]
    pub fn decoder_identity(&self) -> &str {
        &self.decoder_identity
    }
}

/// Compressed source and transform retained until the backend finishes decoding it.
#[derive(Clone, PartialEq)]
pub struct EncodedImageInput {
    transform: ImageTransform,
    info: EncodedImageInfo,
    bytes: Arc<Vec<u8>>,
}

impl Debug for EncodedImageInput {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("EncodedImageInput")
            .field("transform", &self.transform)
            .field("info", &self.info)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl EncodedImageInput {
    /// Retain compressed bytes and their backend-probed geometry.
    ///
    /// A backend must still validate that the stream matches this metadata before device writes.
    /// # Errors
    /// Rejects empty bytes or a transform that does not match the source dimensions.
    pub fn new(transform: ImageTransform, info: EncodedImageInfo, bytes: Vec<u8>) -> Result<Self> {
        if bytes.is_empty() || transform.source() != info.source() {
            return Err(Error::invalid(
                "compressed image buffer or geometry mismatch",
            ));
        }
        Ok(Self {
            transform,
            info,
            bytes: Arc::new(bytes),
        })
    }

    #[must_use]
    pub const fn transform(&self) -> ImageTransform {
        self.transform
    }
    #[must_use]
    pub const fn info(&self) -> &EncodedImageInfo {
        &self.info
    }
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.bytes.capacity()
    }
    /// Clone the owner without copying the compressed bytes.
    #[must_use]
    pub fn shared_bytes(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.bytes)
    }
}

impl Debug for ImageInput {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("ImageInput")
            .field("transform", &self.transform)
            .field("bytes", &self.pixels.len())
            .finish()
    }
}

impl ImageInput {
    /// Move RGB8 bytes into a shared owner after checking the transform's source dimensions.
    /// # Errors
    /// Rejects a byte buffer that does not cover the declared RGB8 source.
    pub fn new(transform: ImageTransform, pixels: Vec<u8>) -> Result<Self> {
        if pixels.len() != rgb_bytes(transform.source)? {
            return Err(Error::invalid(
                "RGB image buffer does not match its dimensions",
            ));
        }
        Ok(Self {
            transform,
            pixels: Arc::new(pixels),
        })
    }

    #[must_use]
    pub const fn transform(&self) -> ImageTransform {
        self.transform
    }
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
    /// Heap bytes retained by the shared RGB8 allocation.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.pixels.capacity()
    }
    /// Clone only the shared owner, not its RGB8 bytes.
    #[must_use]
    pub fn shared_pixels(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.pixels)
    }
}

fn rgb_bytes(dimensions: [usize; 2]) -> Result<usize> {
    dimensions[0]
        .checked_mul(dimensions[1])
        .and_then(|value| value.checked_mul(RGB_CHANNELS))
        .ok_or_else(|| Error::invalid("image dimensions overflow"))
}

#[cfg(test)]
#[path = "../tests/unit/image.rs"]
mod tests;
