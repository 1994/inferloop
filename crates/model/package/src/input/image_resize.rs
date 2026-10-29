//! SIMD RGB8 resampling for normalization that does not retain the sign of zero.
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use infer_core::{Error, Result};
use std::mem::size_of;

use super::image::RGB_CHANNELS;

/// Upper bound on the number of input pixels spanned by the bicubic filter at unit scale.
const FILTER_SPAN: usize = 4;
/// Extra coefficient slot allocated by `fast_image_resize` for each output coordinate.
const WINDOW_PADDING: usize = 1;

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
    // The library normally applies the vertical pass first for RGB8. Separate calls preserve
    // the existing horizontal-then-vertical rounding and clipping contract.
    resizer
        .resize(&source, &mut horizontal, &options)
        .map_err(|error| Error::invalid(format!("horizontal image resize: {error}")))?;
    resizer
        .resize(&horizontal, &mut output, &options)
        .map_err(|error| Error::invalid(format!("vertical image resize: {error}")))?;
    drop(horizontal);
    drop(resizer);
    Ok(output.into_vec())
}

/// Extra scratch beyond the original scalar quote: both axes' maximum coefficient storage.
/// The original F32 image allocations already cover the smaller RGB8 images. Coefficients are
/// bounded against the pinned library's allocation layout.
pub(super) fn extra_scratch(
    source_height: usize,
    source_width: usize,
    height: usize,
    width: usize,
) -> Result<usize> {
    let horizontal = axis_scratch(source_width, width)?;
    let vertical = axis_scratch(source_height, height)?;
    horizontal
        .checked_add(vertical)
        .ok_or_else(|| Error::invalid("image resize scratch overflow"))
}

fn axis_scratch(source: usize, target: usize) -> Result<usize> {
    if source == 0 || target == 0 {
        return Err(Error::invalid("image resize dimensions must be nonzero"));
    }
    if source == target {
        return Ok(0);
    }
    let window = source
        .div_ceil(target)
        .max(1)
        .checked_mul(FILTER_SPAN)
        .and_then(|value| value.checked_add(WINDOW_PADDING))
        .ok_or_else(|| Error::invalid("image resize filter overflow"))?;
    let coefficients = window
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| {
            window
                .min(source)
                .checked_mul(size_of::<i16>())
                .and_then(|taps| bytes.checked_add(taps))
        })
        .and_then(|bytes| bytes.checked_add(size_of::<(u32, u32)>()))
        .and_then(|bytes| bytes.checked_add(size_of::<Vec<i16>>() + 2 * size_of::<u64>()))
        .ok_or_else(|| Error::invalid("image resize filter overflow"))?;
    target
        .checked_mul(coefficients)
        .ok_or_else(|| Error::invalid("image resize scratch overflow"))
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
