use super::*;
use crate::device::asynchronous::block_on;
use crate::resident::narrow_kv::NarrowElement;
use infer_cuda_kernels::runtime::{f8e4m3fn, f16};

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
    let source = NarrowState::Fp8(KvState {
        keys: runtime.upload(data.clone())?,
        values: runtime.upload(data.clone())?,
        layout,
    });
    let cached = NarrowState::zeros(
        &device,
        KvLayout {
            rows: covered,
            ..layout
        },
        NarrowElement::Fp8,
    )?;
    block_on(copy_kv(&device, &cached, &source, covered))?;
    let initial = vec![f8e4m3fn(3); heads * target_capacity * dim];
    let restored = NarrowState::Fp8(KvState {
        keys: runtime.upload(initial.clone())?,
        values: runtime.upload(initial)?,
        layout: KvLayout {
            rows: target_capacity,
            ..layout
        },
    });
    block_on(copy_kv(&device, &restored, &cached, covered))?;
    let restored = runtime.read(&restored.as_fp8().expect("an FP8 cache").keys)?;
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

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn half_precision_prefix_copy_keeps_values_and_rejects_other_elements() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let (heads, capacity, dim, covered) = (2, 32, 16, 9);
    let data: Vec<_> = (0..heads * capacity * dim)
        .map(|i| f16::from_f32(f32::from(u16::try_from(i % 97).unwrap()) / 8.0))
        .collect();
    let layout = KvLayout {
        heads,
        rows: capacity,
        dim,
    };
    let runtime = device.runtime();
    let source = NarrowState::F16(KvState {
        keys: runtime.upload(data.clone())?,
        values: runtime.upload(data.clone())?,
        layout,
    });
    let cached = NarrowState::zeros(
        &device,
        KvLayout {
            rows: covered,
            ..layout
        },
        NarrowElement::F16,
    )?;
    assert_eq!(cached.bytes(), (2 * heads * covered * dim * 2) as u64);
    block_on(copy_kv(&device, &cached, &source, covered))?;
    let NarrowState::F16(state) = &cached else {
        panic!("a half-precision cache");
    };
    let values = runtime.read(&state.values)?;
    for head in 0..heads {
        for row in 0..covered {
            for column in 0..dim {
                assert_eq!(
                    values[(head * covered + row) * dim + column],
                    data[(head * capacity + row) * dim + column]
                );
            }
        }
    }
    let fp8 = NarrowState::zeros(&device, layout, NarrowElement::Fp8)?;
    assert!(block_on(copy_kv(&device, &fp8, &source, covered)).is_err());
    Ok(())
}
