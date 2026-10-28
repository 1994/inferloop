//! CPU shape/limit checks and explicitly selected CUDA numeric qualification.
use super::{Axis, Plan, kernels, preprocess_rgb8};
use crate::device::CudaDevice;
use cutile::cutile_compiler::compile_api::KernelCompiler;
use infer_core::{Error, Result};
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
    assert!(Plan::new(&p, 8, 8, 8 * 8 * 3 - 1, BUDGET).is_err());
    assert!(Plan::new(&p, 8, 8, 8 * 8 * 3, 1).is_err());
    let plan = Plan::new(&p, 8, 8, 8 * 8 * 3, BUDGET)?;
    assert!(plan.output_elements >= plan.intermediate_elements);
    Ok(())
}

#[test]
fn filter_rejects_unrepresentable_or_insufficient_taps() {
    assert!(Axis::new(10, 20, 1).is_err());
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
    for (height, width, patch, temporal, merge) in
        [(31, 45, 16, 2, 2), (128, 128, 16, 2, 2), (41, 73, 8, 1, 2)]
    {
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
        let reference = processor.preprocess(&input)?;
        let device_output =
            preprocess_rgb8(&device, &processor, height, width, &input.pixels, BUDGET)?;
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
