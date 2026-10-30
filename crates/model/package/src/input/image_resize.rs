//! SIMD RGB8 resizing with the reference processor's pass order.
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use infer_core::{Error, Result};

use super::image::RGB_CHANNELS;

pub(super) fn resize(
    pixels: &[u8],
    source_height: usize,
    source_width: usize,
    height: usize,
    width: usize,
) -> Result<Vec<u8>> {
    let source_width_u32 = dimension(source_width)?;
    let source_height_u32 = dimension(source_height)?;
    let width_u32 = dimension(width)?;
    let height_u32 = dimension(height)?;
    if pixels.len() != rgb_bytes(source_height, source_width)? {
        return Err(Error::invalid("image buffer does not match its dimensions"));
    }
    rgb_bytes(source_height, width)?;
    rgb_bytes(height, width)?;
    let source = ImageRef::new(source_width_u32, source_height_u32, pixels, PixelType::U8x3)
        .map_err(|error| Error::invalid(format!("image resize source: {error}")))?;
    let mut horizontal = Image::new(width_u32, source_height_u32, PixelType::U8x3);
    let mut output = Image::new(width_u32, height_u32, PixelType::U8x3);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    let mut resizer = Resizer::new();
    // The library otherwise chooses vertical-first, changing intermediate rounding.
    resizer
        .resize(&source, &mut horizontal, &options)
        .map_err(|error| Error::invalid(format!("horizontal image resize: {error}")))?;
    resizer
        .resize(&horizontal, &mut output, &options)
        .map_err(|error| Error::invalid(format!("vertical image resize: {error}")))?;
    Ok(output.into_vec())
}

fn dimension(value: usize) -> Result<u32> {
    u32::try_from(value)
        .ok()
        .filter(|value| *value != 0)
        .ok_or_else(|| Error::invalid("image resize dimension is out of range"))
}

fn rgb_bytes(height: usize, width: usize) -> Result<usize> {
    height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(RGB_CHANNELS))
        .ok_or_else(|| Error::invalid("image resize size overflow"))
}
