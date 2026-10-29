//! Experimental RGB8 preprocessing on CUDA; not connected to serving.
//!
//! The implementation uses safe, scalar cuTile loads. It records two kernel nodes and submits
//! one graph per image; its block-per-element design is for correctness qualification, not a
//! performance claim. The caller must retain the returned tensor until device consumers finish.
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::ImageResizePolicy;
use infer_models::ImageProcessor;
use std::sync::Arc;

#[cfg(test)]
#[path = "../../tests/unit/vision_image_preprocess.rs"]
mod tests;

/// Channels in a decoded RGB image.
const CHANNELS: usize = 3;
/// Input byte full scale.
const BYTE_SCALE: f64 = 255.0;
/// Half a source pixel, used for bicubic center alignment.
const HALF_PIXEL: f64 = 0.5;
/// Cubic filter radius, before widening for antialiasing.
const FILTER_RADIUS: f64 = 2.0;
/// Cubic interpolation parameter used by the reference RGB8 processor.
const CUBIC_A: f64 = -0.5;
/// Cubic interpolation linear coefficient.
const CUBIC_LINEAR: f64 = 3.0;
/// Cubic interpolation far-span pivot.
const CUBIC_PIVOT: f64 = 5.0;
/// Cubic interpolation far-span offset.
const CUBIC_OFFSET: f64 = 8.0;
/// Cubic interpolation far-span tail coefficient.
const CUBIC_TAIL: f64 = 4.0;
/// Limit on the signed coefficient magnitude used by the reference CPU processor.
const COEFFICIENT_LIMIT: f64 = 32768.0;
/// Upper limit on the fixed-point filter precision.
const MAX_PRECISION: usize = 22;
/// Rows in a vision projection tile.
const ROW_TILE: usize = 32;
/// Host and device copies of the two coefficient tables or normalizer arrays.
const INPUT_COPIES: usize = 4;
/// Normalization scale used by the qualified RGB8 reference path.
const RESCALE: f32 = 1.0 / 255.0;

/// A device-resident flattened patch tensor and its original, unpadded grid.
pub struct DeviceImagePatches {
    /// Patch rows are padded to the vision projection tile; padded values are zero.
    pub pixels: Arc<Tensor<f32>>,
    /// Temporal, height and width patch grid.
    pub grid: (usize, usize, usize),
}

/// Preprocess a decoded RGB8 image into device-resident patches.
///
/// `budget_bytes` bounds the input, temporary image, output, coefficient buffers and temporary
/// host coefficient storage. This synchronous experimental entry point preserves the resources
/// until both kernel nodes complete. The resize policy follows the model processor's geometry;
/// disabled resize requires the input dimensions to tile by the patch and merge sizes. It supports
/// a canonical `1/255` rescale and positive finite normalization values; unsupported settings
/// fail explicitly.
///
/// # Errors
/// Rejects unsupported settings, invalid RGB data, overflowing geometry, a budget too small for
/// the operation, unsafe coefficient ranges, and CUDA allocation, compilation or execution errors.
pub fn preprocess_rgb8(
    device: &CudaDevice,
    processor: &ImageProcessor,
    height: usize,
    width: usize,
    pixels: &[u8],
    resize: ImageResizePolicy,
    budget_bytes: usize,
) -> Result<DeviceImagePatches> {
    let plan = Plan::new(processor, height, width, pixels.len(), resize, budget_bytes)?;
    let horizontal = Axis::new(width, plan.target_width, plan.horizontal_taps)?;
    let vertical = Axis::new(height, plan.target_height, plan.vertical_taps)?;
    let input = device.upload(pixels.to_vec(), &[pixels.len()])?;
    let hx = device.upload(horizontal.indices, &[plan.horizontal_entries])?;
    let hw = device.upload(horizontal.weights, &[plan.horizontal_entries])?;
    let vy = device.upload(vertical.indices, &[plan.vertical_entries])?;
    let vw = device.upload(vertical.weights, &[plan.vertical_entries])?;
    let means = device.upload(processor.mean.to_vec(), &[CHANNELS])?;
    let deviations = device.upload(processor.std.to_vec(), &[CHANNELS])?;
    let mut intermediate = api::zeros::<u8>(&[plan.intermediate_elements])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut output = api::zeros::<f32>(&[plan.output_elements])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(kernels::horizontal(
            (&mut intermediate).partition([1]),
            &input,
            &hx,
            &hw,
            plan.width_i32,
            plan.target_width_i32,
            plan.horizontal_taps_i32,
            horizontal.denominator,
        ))?;
        scope.record(kernels::vertical_patch(
            (&mut output).partition([1]),
            &intermediate,
            &vy,
            &vw,
            &means,
            &deviations,
            plan.target_width_i32,
            plan.vertical_taps_i32,
            vertical.denominator,
            plan.patch_i32,
            plan.merge_i32,
            plan.temporal_i32,
            plan.real_patches_i32,
        ))?;
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    drop(graph);
    let output = output
        .reshape(&[plan.padded_patches, plan.patch_width])
        .map_err(device_error)?;
    Ok(DeviceImagePatches {
        pixels: Arc::new(output),
        grid: (
            1,
            plan.target_height / processor.patch_size,
            plan.target_width / processor.patch_size,
        ),
    })
}

struct Plan {
    target_height: usize,
    target_width: usize,
    horizontal_taps: usize,
    vertical_taps: usize,
    horizontal_entries: usize,
    vertical_entries: usize,
    intermediate_elements: usize,
    output_elements: usize,
    padded_patches: usize,
    patch_width: usize,
    width_i32: i32,
    target_width_i32: i32,
    horizontal_taps_i32: i32,
    vertical_taps_i32: i32,
    patch_i32: i32,
    merge_i32: i32,
    temporal_i32: i32,
    real_patches_i32: i32,
}

fn validate_processor(processor: &ImageProcessor) -> Result<()> {
    if processor.rescale != RESCALE
        || processor.patch_size == 0
        || processor.merge_size == 0
        || processor.temporal_patch_size == 0
        || processor.min_pixels == 0
        || processor.max_pixels < processor.min_pixels
        || processor.mean.iter().any(|x| !x.is_finite())
        || processor.std.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(Error::unsupported(
            "unsupported image preprocessing configuration",
        ));
    }
    for (&mean, &std) in processor.mean.iter().zip(&processor.std) {
        let scaled_mean = f64::from(mean) * BYTE_SCALE;
        let scaled_std = f64::from(std) * BYTE_SCALE;
        let finite_limit = f64::from(f32::MAX);
        if scaled_mean.abs() > finite_limit
            || scaled_std > finite_limit
            || scaled_mean.abs() / scaled_std > finite_limit
            || (BYTE_SCALE - scaled_mean).abs() / scaled_std > finite_limit
        {
            return Err(Error::unsupported(
                "image normalization exceeds finite f32 range",
            ));
        }
    }
    Ok(())
}

fn check_budget(
    input: usize,
    intermediate_elements: usize,
    output_elements: usize,
    filter_entries: usize,
    budget: usize,
) -> Result<()> {
    // The host temporarily holds F64 coefficients, I32 weights and indices; the device owns
    // both I32 arrays. Include both the caller's original and the H2D staging RGB payload.
    let bytes = input
        .checked_mul(2)
        .and_then(|x| x.checked_add(intermediate_elements))
        .and_then(|x| x.checked_add(output_elements.checked_mul(size_of::<f32>())?))
        .and_then(|x| x.checked_add(CHANNELS * INPUT_COPIES * size_of::<f32>()))
        .and_then(|x| {
            x.checked_add(
                filter_entries.checked_mul(size_of::<f64>() + INPUT_COPIES * size_of::<i32>())?,
            )
        })
        .ok_or_else(|| Error::invalid("image working set overflow"))?;
    if bytes > budget {
        return Err(Error::invalid(
            "image preprocessing exceeds its byte budget",
        ));
    }
    Ok(())
}

fn validate_input(
    processor: &ImageProcessor,
    height: usize,
    width: usize,
    input_len: usize,
) -> Result<usize> {
    let input = height
        .checked_mul(width)
        .and_then(|x| x.checked_mul(CHANNELS))
        .ok_or_else(|| Error::invalid("RGB image dimensions overflow"))?;
    if input == 0 || input != input_len {
        return Err(Error::invalid("RGB image payload shape"));
    }
    for value in [
        height,
        width,
        processor.patch_size,
        processor.merge_size,
        processor.temporal_patch_size,
    ] {
        i32::try_from(value)
            .map_err(|_| Error::invalid("image geometry exceeds CUDA index range"))?;
    }
    Ok(input)
}

impl Plan {
    fn new(
        processor: &ImageProcessor,
        height: usize,
        width: usize,
        input_len: usize,
        resize: ImageResizePolicy,
        budget: usize,
    ) -> Result<Self> {
        validate_processor(processor)?;
        let input = validate_input(processor, height, width, input_len)?;
        let (target_height, target_width) = processor.target_dimensions(height, width, resize)?;
        for value in [target_height, target_width] {
            i32::try_from(value)
                .map_err(|_| Error::invalid("image geometry exceeds CUDA index range"))?;
        }
        let patch = processor.patch_size;
        let merge = processor.merge_size;
        let factor = patch
            .checked_mul(merge)
            .ok_or_else(|| Error::invalid("patch merge factor overflow"))?;
        if !target_height.is_multiple_of(factor) || !target_width.is_multiple_of(factor) {
            return Err(Error::invalid(
                "image grid does not divide by patch and merge",
            ));
        }
        let patches = (target_height / patch)
            .checked_mul(target_width / patch)
            .ok_or_else(|| Error::invalid("image patch count overflow"))?;
        let padded = patches
            .checked_next_multiple_of(ROW_TILE)
            .ok_or_else(|| Error::invalid("padded patch count overflow"))?;
        let patch_width = CHANNELS
            .checked_mul(processor.temporal_patch_size)
            .and_then(|x| x.checked_mul(patch))
            .and_then(|x| x.checked_mul(patch))
            .ok_or_else(|| Error::invalid("patch width overflow"))?;
        let output_elements = padded
            .checked_mul(patch_width)
            .ok_or_else(|| Error::invalid("image output overflow"))?;
        let intermediate_elements = height
            .checked_mul(target_width)
            .and_then(|x| x.checked_mul(CHANNELS))
            .ok_or_else(|| Error::invalid("image intermediate overflow"))?;
        let horizontal_taps = tap_bound(width, target_width)?;
        let vertical_taps = tap_bound(height, target_height)?;
        let horizontal_entries = target_width
            .checked_mul(horizontal_taps)
            .ok_or_else(|| Error::invalid("image filter overflow"))?;
        let vertical_entries = target_height
            .checked_mul(vertical_taps)
            .ok_or_else(|| Error::invalid("image filter overflow"))?;
        let filter_entries = horizontal_entries
            .checked_add(vertical_entries)
            .ok_or_else(|| Error::invalid("image filter overflow"))?;
        check_budget(
            input,
            intermediate_elements,
            output_elements,
            filter_entries,
            budget,
        )?;
        for value in [
            input,
            target_height,
            target_width,
            height,
            width,
            intermediate_elements,
            output_elements,
            padded,
            horizontal_entries,
            vertical_entries,
            patches,
            patch_width,
        ] {
            i32::try_from(value)
                .map_err(|_| Error::invalid("image geometry exceeds CUDA index range"))?;
        }
        let to_i32 = |value| {
            i32::try_from(value)
                .map_err(|_| Error::invalid("image geometry exceeds CUDA index range"))
        };
        Ok(Self {
            target_height,
            target_width,
            horizontal_taps,
            vertical_taps,
            horizontal_entries,
            vertical_entries,
            intermediate_elements,
            output_elements,
            padded_patches: padded,
            patch_width,
            width_i32: to_i32(width)?,
            target_width_i32: to_i32(target_width)?,
            horizontal_taps_i32: to_i32(horizontal_taps)?,
            vertical_taps_i32: to_i32(vertical_taps)?,
            patch_i32: to_i32(patch)?,
            merge_i32: to_i32(merge)?,
            temporal_i32: to_i32(processor.temporal_patch_size)?,
            real_patches_i32: to_i32(patches)?,
        })
    }
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Image dimensions are bounded before device execution"
)]
fn tap_bound(source: usize, target: usize) -> Result<usize> {
    let scale = (source as f64 / target as f64).max(1.0);
    let taps = (FILTER_RADIUS * scale * 2.0).ceil() as usize + 2;
    if taps == 0 {
        return Err(Error::invalid("invalid image filter width"));
    }
    Ok(taps)
}

struct Axis {
    indices: Vec<i32>,
    weights: Vec<i32>,
    denominator: i32,
}
impl Axis {
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Image axes and coefficient tables were bounded by the checked plan"
    )]
    fn new(source: usize, target: usize, taps: usize) -> Result<Self> {
        let scale = source as f64 / target as f64;
        let filter_scale = scale.max(1.0);
        let support = FILTER_RADIUS * filter_scale;
        let mut indices = vec![0i32; target * taps];
        let mut coefficients = vec![0.0f64; target * taps];
        let mut maximum = 0.0f64;
        for position in 0..target {
            let center = (position as f64 + HALF_PIXEL) * scale;
            let first = (center - support + HALF_PIXEL).trunc().max(0.0) as usize;
            let last = ((center + support + HALF_PIXEL).trunc() as usize).min(source);
            if last < first || last - first > taps {
                return Err(Error::invariant("image filter bounds"));
            }
            let mut total = 0.0;
            for (slot, tap) in (first..last).enumerate() {
                let weight = cubic((tap as f64 + HALF_PIXEL - center) / filter_scale);
                indices[position * taps + slot] =
                    i32::try_from(tap).map_err(|_| Error::invalid("image filter index"))?;
                coefficients[position * taps + slot] = weight;
                total += weight;
            }
            if total != 0.0 {
                for weight in &mut coefficients[position * taps..position * taps + last - first] {
                    *weight /= total;
                    maximum = maximum.max(*weight);
                }
            }
        }
        let mut denominator = 1.0;
        for _ in 0..MAX_PRECISION {
            if (maximum * denominator).mul_add(2.0, HALF_PIXEL).trunc() >= COEFFICIENT_LIMIT {
                break;
            }
            denominator *= 2.0;
        }
        let weights: Vec<i32> = coefficients
            .into_iter()
            .map(|weight| (weight * denominator).round() as i32)
            .collect();
        let denominator = denominator as i32;
        for row in weights.chunks_exact(taps) {
            let bound = row
                .iter()
                .try_fold(0i64, |sum, weight| {
                    sum.checked_add(i64::from(*weight).abs())
                })
                .and_then(|sum| sum.checked_mul(BYTE_SCALE as i64))
                .and_then(|sum| sum.checked_add(i64::from(denominator / 2)))
                .ok_or_else(|| Error::invalid("image filter accumulator overflow"))?;
            if bound > i64::from(i32::MAX) {
                return Err(Error::unsupported("image filter accumulator exceeds i32"));
            }
        }
        Ok(Self {
            indices,
            weights,
            denominator,
        })
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "The reference uses this evaluation order before coefficient quantization"
)]
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x <= 1.0 {
        ((CUBIC_A + 2.0) * x - (CUBIC_A + CUBIC_LINEAR)).mul_add(x * x, 1.0)
    } else if x < 2.0 {
        (((x - CUBIC_PIVOT) * x + CUBIC_OFFSET) * x - CUBIC_TAIL) * CUBIC_A
    } else {
        0.0
    }
}

#[cutile::module]
mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_0, Shape_1, StoreTileAtCurrentBlock, Tensor_1,
        convert_scalar, get_tile_block_id, tile_to_scalar,
    };

    #[cutile::entry()]
    fn horizontal(
        out: &mut Tensor<u8, { [1] }>,
        input: &Tensor<u8, { [-1] }>,
        indices: &Tensor<i32, { [-1] }>,
        weights: &Tensor<i32, { [-1] }>,
        width: i32,
        target_width: i32,
        taps: i32,
        denominator: i32,
    ) {
        let i = get_tile_block_id().0;
        let channel = i % 3i32;
        let x = (i / 3i32) % target_width;
        let y = i / (3i32 * target_width);
        let mut total = 0i32;
        let mut j = 0i32;
        while j < taps {
            let index: i32 = tile_to_scalar(
                indices
                    .partition(shape![1])
                    .load([x * taps + j])
                    .reshape(shape![]),
            );
            let weight: i32 = tile_to_scalar(
                weights
                    .partition(shape![1])
                    .load([x * taps + j])
                    .reshape(shape![]),
            );
            let value: u8 = tile_to_scalar(
                input
                    .partition(shape![1])
                    .load([(y * width + index) * 3i32 + channel])
                    .reshape(shape![]),
            );
            let float_value: f32 = convert_scalar(value);
            let value: i32 = convert_scalar(float_value);
            total = total + value * weight;
            j = j + 1i32;
        }
        let mut value = (total + denominator / 2i32) / denominator;
        if value < 0i32 {
            value = 0i32;
        }
        if value > 255i32 {
            value = 255i32;
        }
        let float_value: f32 = convert_scalar(value);
        let byte: u8 = convert_scalar(float_value);
        out.store(byte.broadcast(shape![1]));
    }

    #[expect(
        clippy::useless_let_if_seq,
        clippy::suboptimal_flops,
        reason = "cuTile control flow is statement-based and normalization follows the reference rounding"
    )]
    #[cutile::entry()]
    fn vertical_patch(
        out: &mut Tensor<f32, { [1] }>,
        input: &Tensor<u8, { [-1] }>,
        indices: &Tensor<i32, { [-1] }>,
        weights: &Tensor<i32, { [-1] }>,
        means: &Tensor<f32, { [-1] }>,
        stds: &Tensor<f32, { [-1] }>,
        target_width: i32,
        taps: i32,
        denominator: i32,
        patch: i32,
        merge: i32,
        temporal: i32,
        real_patches: i32,
    ) {
        let position = get_tile_block_id().0;
        let patch_width = 3i32 * temporal * patch * patch;
        let row = position / patch_width;
        let mut output = 0.0f32;
        if row < real_patches {
            let patch_offset = position % patch_width;
            let channel = patch_offset / (temporal * patch * patch);
            let dy = (patch_offset / patch) % patch;
            let dx = patch_offset % patch;
            let grid_width = target_width / patch;
            let mx = row % merge;
            let my = (row / merge) % merge;
            let gx = (row / (merge * merge)) % (grid_width / merge);
            let gy = row / (merge * grid_width);
            let pixel_x = (gx * merge + mx) * patch + dx;
            let pixel_y = (gy * merge + my) * patch + dy;
            let mut total = 0i32;
            let mut tap = 0i32;
            while tap < taps {
                let index: i32 = tile_to_scalar(
                    indices
                        .partition(shape![1])
                        .load([pixel_y * taps + tap])
                        .reshape(shape![]),
                );
                let weight: i32 = tile_to_scalar(
                    weights
                        .partition(shape![1])
                        .load([pixel_y * taps + tap])
                        .reshape(shape![]),
                );
                let value: u8 = tile_to_scalar(
                    input
                        .partition(shape![1])
                        .load([(index * target_width + pixel_x) * 3i32 + channel])
                        .reshape(shape![]),
                );
                let float_value: f32 = convert_scalar(value);
                let value: i32 = convert_scalar(float_value);
                total = total + value * weight;
                tap = tap + 1i32;
            }
            let mut value = (total + denominator / 2i32) / denominator;
            if value < 0i32 {
                value = 0i32;
            }
            if value > 255i32 {
                value = 255i32;
            }
            let pixel: f32 = convert_scalar(value);
            let mean: f32 =
                tile_to_scalar(means.partition(shape![1]).load([channel]).reshape(shape![]));
            let std: f32 =
                tile_to_scalar(stds.partition(shape![1]).load([channel]).reshape(shape![]));
            output = (pixel - mean * 255.0f32) / (std * 255.0f32);
        }
        out.store(output.broadcast(shape![1]));
    }
}
