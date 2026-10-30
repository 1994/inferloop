use super::{ImageBatch, Processor};
use crate::device::CudaDevice;
use crate::device::asynchronous::block_on;
use crate::device::queue_test_gate;
use infer_core::{Error, Result};
use infer_models::{ImageProcessor, RgbImage};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

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

#[test]
#[ignore = "requires a qualified CUDA device"]
fn image_preprocessing_poll_does_not_wait_for_busy_stream() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let processor = Processor::load(&device)?
        .ok_or_else(|| Error::unsupported("CUDA image kernel is not qualified on this device"))?;
    for side in [32, 1024] {
        check_busy_stream(&device, &processor, side)?;
    }
    Ok(())
}

fn check_busy_stream(device: &CudaDevice, processor: &Processor, side: usize) -> Result<()> {
    use std::future::{Future, poll_fn};
    use std::pin::pin;
    use std::task::Poll;
    use std::thread::yield_now;

    let reference = ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: side * side,
        max_pixels: side * side,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    };
    let transform = reference.transform(side, side)?;
    let images = ImageBatch::new(Arc::new(vec![128; side * side * 3]), 1, transform)?;
    let released = Arc::new(AtomicBool::new(false));
    let expired = Arc::new(AtomicBool::new(false));
    let (gate, timeout) = (Arc::clone(&released), Arc::clone(&expired));
    queue_test_gate(device, move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !gate.load(Ordering::Acquire) {
            if Instant::now() >= deadline {
                timeout.store(true, Ordering::Release);
                break;
            }
            yield_now();
        }
    })?;
    let mut future = pin!(processor.process(device, &images));
    let first = block_on(poll_fn(|context| {
        Poll::Ready(future.as_mut().poll(context))
    }));
    released.store(true, Ordering::Release);
    assert!(
        !expired.load(Ordering::Acquire),
        "image preprocessing blocked on the GPU stream"
    );
    assert!(first.is_pending());
    let output = block_on(future)?;
    assert_eq!(device.read(&output)?.len(), output.size());
    Ok(())
}

#[test]
#[ignore = "requires a qualified CUDA device"]
fn simultaneous_image_uploads_keep_distinct_staging_until_completion() -> Result<()> {
    use std::future::{Future, poll_fn};
    use std::pin::pin;
    use std::task::Poll;
    use std::thread::yield_now;

    let device = CudaDevice::new(0)?;
    let processor = Processor::load(&device)?
        .ok_or_else(|| Error::unsupported("CUDA image kernel is not qualified on this device"))?;
    let reference = ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 32 * 32,
        max_pixels: 32 * 32,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    };
    let transform = reference.transform(32, 32)?;
    let images = [11, 229].map(|color| RgbImage {
        height: 32,
        width: 32,
        pixels: vec![color; 32 * 32 * 3],
    });
    let first = ImageBatch::new(Arc::new(images[0].pixels.clone()), 1, transform)?;
    let second = ImageBatch::new(Arc::new(images[1].pixels.clone()), 1, transform)?;
    let released = Arc::new(AtomicBool::new(false));
    let expired = Arc::new(AtomicBool::new(false));
    let (gate, timeout) = (Arc::clone(&released), Arc::clone(&expired));
    queue_test_gate(&device, move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !gate.load(Ordering::Acquire) {
            if Instant::now() >= deadline {
                timeout.store(true, Ordering::Release);
                break;
            }
            yield_now();
        }
    })?;
    let mut first = pin!(processor.process(&device, &first));
    let mut second = pin!(processor.process(&device, &second));
    let states = block_on(poll_fn(|context| {
        Poll::Ready((first.as_mut().poll(context), second.as_mut().poll(context)))
    }));
    released.store(true, Ordering::Release);
    assert!(!expired.load(Ordering::Acquire));
    assert!(states.0.is_pending() && states.1.is_pending());
    for (output, image) in [
        (block_on(first)?, &images[0]),
        (block_on(second)?, &images[1]),
    ] {
        let actual = device.read(&output)?;
        let expected = reference.preprocess(image)?.pixels;
        assert_eq!(&actual[..expected.len()], expected.as_slice());
        assert!(actual[expected.len()..].iter().all(|value| *value == 0.0));
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
