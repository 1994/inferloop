use super::{ImageBatch, PORTABLE_PTX, Processor};
use crate::device::asynchronous::block_on;
use crate::device::queue_test_gate;
use crate::device::{CudaDevice, device_error};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::{EncodedImageInfo, EncodedImageInput, ImageEncoding};
use infer_models::{ImageProcessor, RgbImage};
use infer_spi::DeviceHostExecutor;
use std::env::var;
use std::fs::read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

type HostJob = Box<dyn FnOnce() + Send>;

struct TestHostExecutor {
    sender: Option<mpsc::SyncSender<HostJob>>,
    worker: Option<JoinHandle<()>>,
}

impl TestHostExecutor {
    fn new() -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<HostJob>(2);
        let worker = thread::Builder::new()
            .name("nvjpeg-test-worker".into())
            .spawn(move || {
                while let Ok(work) = receiver.recv() {
                    work();
                }
            })
            .map_err(|error| Error::new(ErrorCode::Backend, error.to_string()))?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
        })
    }
}

impl DeviceHostExecutor for TestHostExecutor {
    fn submit(&self, work: HostJob) -> Result<()> {
        self.sender
            .as_ref()
            .ok_or_else(|| Error::invariant("test executor stopped"))?
            .try_send(work)
            .map_err(|error| Error::new(ErrorCode::Capacity, error.to_string()))
    }
}

impl Drop for TestHostExecutor {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            eprintln!("nvJPEG test worker panicked");
        }
    }
}

fn install_host_executor(processor: &Processor) -> Result<()> {
    processor
        .capability()
        .set_host_executor(Arc::new(TestHostExecutor::new()?))
}

#[test]
#[ignore = "requires nvJPEG hardware and JPEG/RGB fixtures in INFER_NVJPEG_JPEG_FIXTURE and INFER_NVJPEG_RGB_FIXTURE"]
fn hardware_jpeg_preprocessing_matches_independent_rgb_reference() -> Result<()> {
    let path = var("INFER_NVJPEG_JPEG_FIXTURE")
        .map_err(|_| Error::invalid("INFER_NVJPEG_JPEG_FIXTURE is required"))?;
    let rgb_path = var("INFER_NVJPEG_RGB_FIXTURE")
        .map_err(|_| Error::invalid("INFER_NVJPEG_RGB_FIXTURE is required"))?;
    let bytes = read(path).map_err(|error| Error::invalid(error.to_string()))?;
    let pixels = read(rgb_path).map_err(|error| Error::invalid(error.to_string()))?;
    let device = CudaDevice::new(0)?;
    let processor = Processor::load(&device)?
        .ok_or_else(|| Error::unsupported("CUDA image kernel is not qualified on this device"))?;
    let portable = Processor::load_kernel(&device, PORTABLE_PTX)?;
    for processor in [&processor, &portable] {
        install_host_executor(processor)?;
        let capability = processor.capability();
        let info = capability
            .inspect_encoded(&bytes)?
            .ok_or_else(|| Error::unsupported("fixture requires nvJPEG hardware decode"))?;
        assert_eq!(info.source(), [1024, 1024]);
        assert!(capability.inspect_encoded(b"not a JPEG")?.is_none());
        let progressive = include_bytes!("../fixtures/progressive-16x16.jpg");
        assert!(capability.inspect_encoded(progressive)?.is_none());
        let image = RgbImage {
            height: 1024,
            width: 1024,
            pixels: pixels.clone(),
        };
        for maximum in [1024 * 1024, 256 * 256] {
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
            let transform = reference.transform(image.height, image.width)?;
            let expected = reference.preprocess(&image)?.pixels;
            let encoded = EncodedImageInput::new(transform, info.clone(), bytes.clone())?;
            let output = block_on(processor.process_encoded(&device, &encoded))?;
            let actual = device.read(&output)?;
            assert_eq!(
                actual[..expected.len()]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert!(
                actual[expected.len()..]
                    .iter()
                    .all(|value| value.to_bits() == 0)
            );
        }
        let wrong = EncodedImageInfo::new(
            [1024, 1056],
            ImageEncoding::Jpeg,
            info.decoder_identity().into(),
        )?;
        let reference = ImageProcessor {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            min_pixels: 32 * 32,
            max_pixels: 1024 * 1024,
            mean: [0.5; 3],
            std: [0.5; 3],
            rescale: 1.0 / 255.0,
        };
        let encoded =
            EncodedImageInput::new(reference.transform(1024, 1056)?, wrong, bytes.clone())?;
        assert!(block_on(processor.process_encoded(&device, &encoded)).is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requires nvJPEG hardware and JPEG fixture in INFER_NVJPEG_JPEG_FIXTURE"]
fn jpeg_preprocessing_poll_does_not_wait_for_busy_stream() -> Result<()> {
    use std::future::{Future, poll_fn};
    use std::pin::pin;
    use std::task::Poll;
    use std::thread::yield_now;

    let path = var("INFER_NVJPEG_JPEG_FIXTURE")
        .map_err(|_| Error::invalid("INFER_NVJPEG_JPEG_FIXTURE is required"))?;
    let bytes = read(path).map_err(|error| Error::invalid(error.to_string()))?;
    let device = CudaDevice::new(0)?;
    let processor = Processor::load(&device)?
        .ok_or_else(|| Error::unsupported("CUDA image kernel is not qualified on this device"))?;
    install_host_executor(&processor)?;
    let info = processor
        .capability()
        .inspect_encoded(&bytes)?
        .ok_or_else(|| Error::unsupported("fixture requires nvJPEG hardware decode"))?;
    let reference = ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 32 * 32,
        max_pixels: 1024 * 1024,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    };
    let [height, width] = info.source();
    let image = EncodedImageInput::new(reference.transform(height, width)?, info, bytes)?;
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
    let mut future = pin!(processor.process_encoded(&device, &image));
    let first = block_on(poll_fn(|context| {
        Poll::Ready(future.as_mut().poll(context))
    }));
    released.store(true, Ordering::Release);
    assert!(
        !expired.load(Ordering::Acquire),
        "JPEG preprocessing blocked on the GPU stream"
    );
    assert!(first.is_pending());
    let output = block_on(future)?;
    assert_eq!(device.read(&output)?.len(), output.size());
    Ok(())
}

#[test]
#[ignore = "requires nvJPEG hardware and two JPEG/RGB fixtures in INFER_NVJPEG_*_FIXTURE"]
fn simultaneous_jpeg_submissions_keep_distinct_decoder_state_until_completion() -> Result<()> {
    use std::future::{Future, poll_fn};
    use std::pin::pin;
    use std::task::Poll;
    use std::thread::yield_now;

    fn fixture(key: &str) -> Result<Vec<u8>> {
        let path = var(key).map_err(|_| Error::invalid(format!("{key} is required")))?;
        read(path).map_err(|error| Error::invalid(error.to_string()))
    }
    let encoded = [
        fixture("INFER_NVJPEG_JPEG_FIXTURE")?,
        fixture("INFER_NVJPEG_SECOND_JPEG_FIXTURE")?,
    ];
    let pixels = [
        fixture("INFER_NVJPEG_RGB_FIXTURE")?,
        fixture("INFER_NVJPEG_SECOND_RGB_FIXTURE")?,
    ];
    let device = CudaDevice::new(0)?;
    let processor = Processor::load(&device)?
        .ok_or_else(|| Error::unsupported("CUDA image kernel is not qualified on this device"))?;
    install_host_executor(&processor)?;
    let reference = ImageProcessor {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 32 * 32,
        max_pixels: 256 * 256,
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale: 1.0 / 255.0,
    };
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for (bytes, pixels) in encoded.into_iter().zip(pixels) {
        let info = processor
            .capability()
            .inspect_encoded(&bytes)?
            .ok_or_else(|| Error::unsupported("fixture requires nvJPEG hardware decode"))?;
        let [height, width] = info.source();
        let transform = reference.transform(height, width)?;
        expected.push(
            reference
                .preprocess(&RgbImage {
                    height,
                    width,
                    pixels,
                })?
                .pixels,
        );
        inputs.push(EncodedImageInput::new(transform, info, bytes)?);
    }
    assert_ne!(expected[0], expected[1]);
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
    let mut first = pin!(processor.process_encoded(&device, &inputs[0]));
    let mut second = pin!(processor.process_encoded(&device, &inputs[1]));
    let states = block_on(poll_fn(|context| {
        Poll::Ready((first.as_mut().poll(context), second.as_mut().poll(context)))
    }));
    released.store(true, Ordering::Release);
    assert!(
        !expired.load(Ordering::Acquire),
        "JPEG submissions blocked on the GPU stream"
    );
    assert!(states.0.is_pending() && states.1.is_pending());
    for (result, expected) in [block_on(first)?, block_on(second)?]
        .into_iter()
        .zip(expected)
    {
        let actual = device.read(&result)?;
        assert_eq!(&actual[..expected.len()], expected.as_slice());
        assert!(actual[expected.len()..].iter().all(|value| *value == 0.0));
    }
    Ok(())
}

#[test]
#[ignore = "requires a qualified CUDA device"]
fn device_image_batches_match_cpu_patches_and_zero_padding() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let processor = Processor::load(&device)?
        .ok_or_else(|| Error::unsupported("CUDA image kernel is not qualified on this device"))?;
    let portable = Processor::load_kernel(&device, PORTABLE_PTX)?;
    for processor in [&processor, &portable] {
        for (source, batch, maximum) in [
            ([32, 32], 4, 1024 * 1024),
            ([32, 64], 3, 1024 * 1024),
            ([96, 160], 3, 1024 * 1024),
            ([256, 256], 16, 1024 * 1024),
            ([333, 571], 16, 1024 * 1024),
            ([225, 353], 16, 1024 * 1024),
            ([333, 571], 16, 32 * 1024),
            ([1024, 1024], 1, 1024 * 1024),
        ] {
            check_batch(&device, processor, source, batch, maximum)?;
        }
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

#[test]
#[ignore = "requires a qualified CUDA device"]
fn image_upload_staging_reuses_capacity_without_reusing_inflight_data() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let processor = Processor::load(&device)?
        .ok_or_else(|| Error::unsupported("CUDA image kernel is not qualified on this device"))?;
    let first = processor.pixels.stage(&[7; 33])?;
    let first_pointer = first.buffer().map_err(device_error)?.as_ptr();
    let concurrent = processor.pixels.stage(&[11; 33])?;
    assert_ne!(
        first_pointer,
        concurrent.buffer().map_err(device_error)?.as_ptr()
    );
    drop(first);
    let reused = processor.pixels.stage(&[19; 50])?;
    let buffer = reused.buffer().map_err(device_error)?;
    assert_eq!(first_pointer, buffer.as_ptr());
    assert_eq!(&buffer.as_slice()[..50], &[19; 50]);
    assert_eq!(
        &concurrent.buffer().map_err(device_error)?.as_slice()[..33],
        &[11; 33]
    );
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
