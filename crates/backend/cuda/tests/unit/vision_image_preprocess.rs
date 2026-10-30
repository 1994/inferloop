use super::{ImageBatch, Processor};
use crate::device::CudaDevice;
use crate::device::asynchronous::block_on;
use infer_core::{Error, Result};
use infer_models::{ImageProcessor, RgbImage};
use std::sync::Arc;

#[test]
#[ignore = "requires a qualified CUDA device"]
fn device_image_batches_match_cpu_patches_and_zero_padding() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let processor = Processor::load(&device)?
        .ok_or_else(|| Error::unsupported("CUDA image kernel is not qualified on this device"))?;
    for (source, batch, maximum) in [
        ([256, 256], 16, 1024 * 1024),
        ([333, 571], 16, 1024 * 1024),
        ([225, 353], 16, 1024 * 1024),
        ([333, 571], 16, 32 * 1024),
        ([1024, 1024], 1, 1024 * 1024),
    ] {
        check_batch(&device, &processor, source, batch, maximum)?;
    }
    Ok(())
}

fn check_batch(
    device: &CudaDevice,
    processor: &Processor,
    source: [usize; 2],
    batch: usize,
    maximum: usize,
) -> Result<()> {
    let reference = ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 32 * 32,
        max_pixels: maximum,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    };
    let target = reference.smart_resize(source[0], source[1])?;
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for sample in 0..batch {
        let image = RgbImage {
            height: source[0],
            width: source[1],
            pixels: (0..source[0] * source[1] * 3)
                .map(|index| (index.wrapping_mul(37).wrapping_add(sample * 53)).to_le_bytes()[0])
                .collect(),
        };
        expected.push(reference.preprocess(&image)?.pixels);
        inputs.extend_from_slice(&image.pixels);
    }
    let transform = reference.transform(source[0], source[1])?;
    assert_eq!(transform.target(), <[usize; 2]>::from(target));
    let images = ImageBatch::new(Arc::new(inputs), batch, transform)?;
    let result = block_on(processor.process(device, &images))?;
    let values = device.read(&result)?;
    let stride = values.len() / batch;
    for (index, reference) in expected.iter().enumerate() {
        let actual = &values[index * stride..(index + 1) * stride];
        assert_eq!(
            actual[..reference.len()]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            reference
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(
            actual[reference.len()..]
                .iter()
                .all(|value| value.to_bits() == 0)
        );
    }
    Ok(())
}
