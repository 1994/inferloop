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
    let runtime = device.runtime();
    let layout = KvLayout {
        heads,
        rows: source_capacity,
        dim,
    };
    let source = KvState {
        keys: runtime.upload(data.clone())?,
        values: runtime.upload(data.clone())?,
        layout,
    };
    let cached = KvState::zeros(
        &device,
        KvLayout {
            rows: covered,
            ..layout
        },
    )?;
    block_on(copy_kv(&device, &cached, &source, covered))?;
    let initial = vec![f8e4m3fn(3); heads * target_capacity * dim];
    let restored = KvState {
        keys: runtime.upload(initial.clone())?,
        values: runtime.upload(initial)?,
        layout: KvLayout {
            rows: target_capacity,
            ..layout
        },
    };
    block_on(copy_kv(&device, &restored, &cached, covered))?;
    let restored = runtime.read(&restored.keys)?;
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
    assert!(block_on(copy_kv(&device, &cached, &source, covered + 1)).is_err());
    Ok(())
}
