use super::*;
use crate::device::asynchronous::block_on;

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn fp8_prefix_copy_preserves_head_strides_and_untouched_tails() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let (heads, source_capacity, target_capacity, dim, covered) = (3, 64, 128, 32, 17);
    let data: Vec<_> = (0..heads * source_capacity * dim)
        .map(|i| f8e4m3fn(u8::try_from(16 + i % 64).unwrap()))
        .collect();
    let source = api::copy_host_vec_to_device(&Arc::new(data.clone()))
        .sync_on(&device.stream)
        .map_err(device_error)?
        .reshape(&[heads, source_capacity, dim])
        .map_err(device_error)?;
    let mut cached = api::zeros::<f8e4m3fn>(&[heads, covered, dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    block_on(copy_state(&device, &mut cached, &source, true, covered))?;
    let initial = vec![f8e4m3fn(3); heads * target_capacity * dim];
    let mut restored = api::copy_host_vec_to_device(&Arc::new(initial))
        .sync_on(&device.stream)
        .map_err(device_error)?
        .reshape(&[heads, target_capacity, dim])
        .map_err(device_error)?;
    block_on(copy_state(&device, &mut restored, &cached, true, covered))?;
    let restored = restored
        .to_host_vec()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    for head in 0..heads {
        for row in 0..target_capacity {
            for column in 0..dim {
                let expected = if row < covered {
                    data[(head * source_capacity + row) * dim + column].0
                } else {
                    3
                };
                assert_eq!(
                    restored[(head * target_capacity + row) * dim + column].0,
                    expected
                );
            }
        }
    }
    assert!(block_on(copy_state(&device, &mut cached, &source, true, covered + 1)).is_err());
    Ok(())
}
