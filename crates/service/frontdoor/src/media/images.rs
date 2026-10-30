//! Load compressed media for a capable backend or decode it on a CPU worker.
use super::{SourceDigest, SourcePolicy, load_image_bytes, read_url, source_digest};
use crate::cpu::CpuContext;
use infer_core::Result;
use infer_ir::{EncodedImageInfo, EncodedImageInput, ImageInput, ImageTransform, MediaPixels};
use infer_models::RawImage;
use infer_spi::ImagePreprocessing;
use std::borrow::Cow;

/// Borrow public-call inputs, or take ownership of bytes resolved by the HTTP ingress.
pub(super) enum Downloads<'a> {
    Borrowed(&'a [Option<Vec<u8>>]),
    Owned(Vec<Option<Vec<u8>>>),
}

impl<'a> Downloads<'a> {
    pub fn as_slice(&self) -> &[Option<Vec<u8>>] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes,
        }
    }

    pub fn take(&mut self, index: usize) -> Option<Cow<'a, [u8]>> {
        match self {
            Self::Borrowed(bytes) => bytes.get(index)?.as_deref().map(Cow::Borrowed),
            Self::Owned(bytes) => bytes.get_mut(index)?.take().map(Cow::Owned),
        }
    }
}

pub(super) enum LoadedImage {
    Rgb(RawImage),
    Encoded {
        info: EncodedImageInfo,
        bytes: Vec<u8>,
    },
}

impl LoadedImage {
    pub fn source(&self) -> [usize; 2] {
        match self {
            Self::Rgb(image) => [image.height, image.width],
            Self::Encoded { info, .. } => info.source(),
        }
    }

    pub fn into_pixels(self, transform: ImageTransform) -> Result<MediaPixels> {
        match self {
            Self::Rgb(image) => Ok(MediaPixels::Image(ImageInput::new(
                transform,
                image.pixels,
            )?)),
            Self::Encoded { info, bytes } => Ok(MediaPixels::EncodedImage(EncodedImageInput::new(
                transform, info, bytes,
            )?)),
        }
    }

    pub fn into_raw(self, context: &CpuContext) -> Result<RawImage> {
        match self {
            Self::Rgb(image) => Ok(image),
            Self::Encoded { bytes, .. } => {
                load_image_bytes(&bytes, Some(context)).map(|pair| pair.0)
            }
        }
    }
}

pub(super) fn load(
    url: &str,
    download: Option<Cow<'_, [u8]>>,
    policy: SourcePolicy,
    context: &CpuContext,
    capability: Option<&dyn ImagePreprocessing>,
) -> Result<(LoadedImage, SourceDigest)> {
    let bytes = if let Some(bytes) = download {
        bytes
    } else {
        Cow::Owned(read_url(url, None, policy, Some(context))?)
    };
    if let Some(capability) = capability {
        if let Some(info) = capability.inspect_encoded(&bytes)? {
            let source = source_digest(&bytes, Some(context))?;
            return Ok((
                LoadedImage::Encoded {
                    info,
                    bytes: bytes.into_owned(),
                },
                source,
            ));
        }
    }
    let (image, source) = load_image_bytes(&bytes, Some(context))?;
    Ok((LoadedImage::Rgb(image), source))
}
