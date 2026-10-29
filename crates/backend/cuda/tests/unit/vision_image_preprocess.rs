//! CPU shape/limit checks and explicitly selected CUDA numeric qualification.
use super::{Axis, Plan, kernels, preprocess_rgb8, tap_bound};
use crate::device::CudaDevice;
use cutile::cutile_compiler::compile_api::KernelCompiler;
use infer_core::{Error, Result};
use infer_ir::ImageResizePolicy;
use infer_models::{ImageProcessor, RgbImage};
use std::error::Error as StdError;
use std::result::Result as StdResult;

const BUDGET: usize = 64 * 1024 * 1024;
const TOLERANCE: f32 = 1e-6;

fn processor(patch: usize, temporal: usize, merge: usize) -> ImageProcessor {
    ImageProcessor {
        patch_size: patch,
        temporal_patch_size: temporal,
        merge_size: merge,
        min_pixels: 1024,
        max_pixels: 8192,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    }
}

#[test]
fn rejects_malformed_shape_and_budget_before_allocating() -> Result<()> {
    let p = processor(16, 2, 2);
    assert!(Plan::new(&p, 8, 8, 8 * 8 * 3 - 1, ImageResizePolicy::Smart, BUDGET).is_err());
    assert!(Plan::new(&p, 8, 8, 8 * 8 * 3, ImageResizePolicy::Smart, 1).is_err());
    let plan = Plan::new(&p, 8, 8, 8 * 8 * 3, ImageResizePolicy::Smart, BUDGET)?;
    assert!(plan.output_elements >= plan.intermediate_elements);
    Ok(())
}

#[test]
fn disabled_resize_preserves_source_geometry_and_requires_aligned_edges() -> Result<()> {
    let p = processor(16, 2, 2);
    let disabled = Plan::new(
        &p,
        128,
        128,
        128 * 128 * 3,
        ImageResizePolicy::Disabled,
        BUDGET,
    )?;
    let smart = Plan::new(
        &p,
        128,
        128,
        128 * 128 * 3,
        ImageResizePolicy::Smart,
        BUDGET,
    )?;
    assert_eq!((disabled.target_height, disabled.target_width), (128, 128));
    assert_ne!((smart.target_height, smart.target_width), (128, 128));
    assert!(
        Plan::new(
            &p,
            127,
            128,
            127 * 128 * 3,
            ImageResizePolicy::Disabled,
            BUDGET
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn filter_rejects_unrepresentable_or_insufficient_taps() {
    assert!(Axis::new(10, 20, 1).is_err());
}

#[test]
fn unit_scale_filter_preserves_each_source_pixel_at_edges_and_interior() -> Result<()> {
    for size in [1, 2, 31, 128] {
        let taps = tap_bound(size, size)?;
        let axis = Axis::new(size, size, taps)?;
        for (position, (indices, weights)) in axis
            .indices
            .chunks_exact(taps)
            .zip(axis.weights.chunks_exact(taps))
            .enumerate()
        {
            assert_eq!(weights.iter().sum::<i32>(), axis.denominator);
            for (&index, &weight) in indices.iter().zip(weights) {
                if weight != 0 {
                    assert_eq!(usize::try_from(index).ok(), Some(position));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn image_kernels_compile_without_a_device() -> StdResult<(), Box<dyn StdError>> {
    KernelCompiler::new(kernels::__module_ast_self, "kernels", "horizontal")
        .target("sm_120")
        .strides(&[
            ("out", &[1]),
            ("input", &[1]),
            ("indices", &[1]),
            ("weights", &[1]),
        ])
        .compile()?;
    KernelCompiler::new(kernels::__module_ast_self, "kernels", "vertical_patch")
        .target("sm_120")
        .strides(&[
            ("out", &[1]),
            ("input", &[1]),
            ("indices", &[1]),
            ("weights", &[1]),
            ("means", &[1]),
            ("stds", &[1]),
        ])
        .compile()?;
    Ok(())
}

#[test]
#[ignore = "requires a reserved CUDA device; compared against the CPU processor"]
fn gpu_matches_cpu_for_resize_and_patch_layout() -> Result<()> {
    let device = CudaDevice::new(0)?;
    for (height, width, patch, temporal, merge, resize) in [
        (31, 45, 16, 2, 2, ImageResizePolicy::Smart),
        (128, 128, 16, 2, 2, ImageResizePolicy::Smart),
        (41, 73, 8, 1, 2, ImageResizePolicy::Smart),
        (128, 128, 16, 2, 2, ImageResizePolicy::Disabled),
    ] {
        let processor = processor(patch, temporal, merge);
        let pixels: Vec<u8> = (0..height * width * 3)
            .map(|i| u8::try_from(i % 251))
            .collect::<StdResult<_, _>>()
            .map_err(|_| Error::invariant("image fixture value exceeds u8"))?;
        let input = RgbImage {
            height,
            width,
            pixels,
        };
        let mut reference_processor = processor.clone();
        if resize == ImageResizePolicy::Disabled {
            // Smart resize with these bounds leaves the aligned source unchanged, giving an
            // independent CPU reference for disabled resize above the original pixel maximum.
            reference_processor.max_pixels = height * width;
        }
        let reference = reference_processor.preprocess(&input)?;
        let device_output = preprocess_rgb8(
            &device,
            &processor,
            height,
            width,
            &input.pixels,
            resize,
            BUDGET,
        )?;
        if device_output.grid != reference.grid {
            return Err(Error::invariant("image grid mismatch"));
        }
        let values = device.read(&device_output.pixels)?;
        if values.len() < reference.pixels.len() {
            return Err(Error::invariant("image output is short"));
        }
        for (actual, expected) in values.iter().zip(&reference.pixels) {
            if !actual.is_finite() || (actual - expected).abs() > TOLERANCE {
                return Err(Error::invariant(
                    "GPU image pixels disagree with the CPU reference",
                ));
            }
        }
        if values[reference.pixels.len()..]
            .iter()
            .any(|value| *value != 0.0)
        {
            return Err(Error::invariant("image output padding is nonzero"));
        }
    }
    Ok(())
}
