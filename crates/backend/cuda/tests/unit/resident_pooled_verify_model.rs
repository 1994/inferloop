use crate::tuning::pool_dense;
use crate::{
    device::CudaDevice,
    loading::{LoadOptions, LoadedModel},
    resident::{DeviceProgram, attention_decode::pooled, slot_batch::SlotPool, slot_convolution},
};
use infer_core::{Error, ModelId, Result};
use std::env::var;

#[derive(Clone, Copy)]
enum Operation {
    Attention,
    Convolution,
    Dense,
}

impl Operation {
    fn builds(self) -> usize {
        match self {
            Self::Attention => pooled::builds(),
            Self::Convolution => slot_convolution::builds(),
            Self::Dense => super::pool_gemv::captures(),
        }
    }

    fn reference(self, loaded: &LoadedModel, width: usize) -> Result<SlotPool> {
        let build = || loaded.slot_pool(width, 64, Some(width * 64));
        match self {
            Self::Attention => pooled::with_row_reference(build),
            Self::Convolution => slot_convolution::with_row_reference(build),
            Self::Dense => pool_dense::with_private_reference(build),
        }
    }
}

#[test]
#[ignore = "requires CUDA and INFER_TEST_MODEL; run release inside safe-run"]
fn pooled_attention_preserves_checkpoint_model_bits() -> Result<()> {
    compare_model(Operation::Attention)
}

#[test]
#[ignore = "requires CUDA and INFER_TEST_MODEL; run release inside safe-run"]
fn pooled_convolution_preserves_checkpoint_model_bits() -> Result<()> {
    compare_model(Operation::Convolution)
}

#[test]
#[ignore = "requires CUDA/checkpoint/private cache; full small-pool model/commit/rebind states"]
fn measured_pool_dense_preserves_private_checkpoint_bits() -> Result<()> {
    compare_model(Operation::Dense)
}

fn compare_model(operation: Operation) -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|error| Error::invalid(error.to_string()))?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            mtp_depth: 2,
            autotune: matches!(operation, Operation::Dense),
            chunked_recurrent: false,
            ..LoadOptions::default()
        },
    )?;
    let widths = match operation {
        Operation::Dense => [6, 8],
        Operation::Attention | Operation::Convolution => [2, 5],
    };
    for width in widths {
        compare_pool(&loaded, width, operation)?;
    }
    Ok(())
}

fn compare_pool(loaded: &LoadedModel, width: usize, operation: Operation) -> Result<()> {
    let before = operation.builds();
    let mut candidate = loaded.slot_pool(width, 64, Some(width * 64))?;
    assert!(
        operation.builds() > before,
        "actual target must capture optimized verification"
    );
    let before = operation.builds();
    let mut reference = operation.reference(loaded, width)?;
    match operation {
        Operation::Dense => assert!(
            operation.builds() > before,
            "private reference actually captured"
        ),
        Operation::Attention | Operation::Convolution => assert_eq!(
            operation.builds(),
            before,
            "reference must retain the original operation"
        ),
    }
    compare_states(loaded.device(), &candidate, &reference, width)?;
    let mut worker = loaded.sequence_pooled(64)?;
    drop(worker.prefill_batch(&[101, 102, 103], 0, false)?);
    bind_worker(loaded, &mut candidate, &worker, width)?;
    bind_worker(loaded, &mut reference, &worker, width)?;
    compare_states(loaded.device(), &candidate, &reference, width)?;
    for round in 0..4 {
        if round == 3 {
            for slot in 0..width {
                candidate.release(slot);
                reference.release(slot);
            }
            worker.reset()?;
            drop(worker.prefill_batch(&[104, 105, 106], 0, false)?);
            bind_worker(loaded, &mut candidate, &worker, width)?;
            bind_worker(loaded, &mut reference, &worker, width)?;
        }
        let mut active: Vec<_> = (0..width)
            .filter(|slot| round != 1 || slot % 2 == 0)
            .collect();
        if round % 2 == 1 {
            active.reverse();
        }
        let mut lanes = Vec::new();
        let mut commits = Vec::new();
        for &slot in &active {
            let count = 1 + (slot + round) % 3;
            let accepted = (slot + round) % count;
            let position = candidate.cursors()[slot];
            assert_eq!(position, reference.cursors()[slot]);
            for offset in 0..count {
                lanes.push((
                    slot,
                    offset,
                    u32::try_from(100 + slot * 3 + round + offset).unwrap(),
                    position + offset,
                    position + offset,
                ));
            }
            commits.push((slot, accepted));
        }
        let actual = candidate.run_verify(loaded.device(), &lanes)?;
        let expected = reference.run_verify(loaded.device(), &lanes)?;
        for &(slot, offset, _, _, _) in &lanes {
            let index = slot * 3 + offset;
            bits(&actual[index].0, &expected[index].0);
            bits(&actual[index].1, &expected[index].1);
        }
        compare_states(loaded.device(), &candidate, &reference, width)?;
        candidate.commit_verify_many(loaded.device(), &commits)?;
        reference.commit_verify_many(loaded.device(), &commits)?;
        assert_eq!(candidate.cursors(), reference.cursors());
        compare_states(loaded.device(), &candidate, &reference, width)?;
    }
    Ok(())
}

fn compare_states(device: &CudaDevice, a: &SlotPool, b: &SlotPool, width: usize) -> Result<()> {
    for slot in 0..width {
        let (a, a_packed) = a.test_states(slot);
        let (b, b_packed) = b.test_states(slot);
        assert_eq!(
            a_packed.keys().collect::<Vec<_>>(),
            b_packed.keys().collect::<Vec<_>>()
        );
        for (id, (keys, values)) in a_packed {
            let (expected_keys, expected_values) = &b_packed[id];
            for (actual, expected) in [(keys, expected_keys), (values, expected_values)] {
                assert_eq!(
                    actual.shape(),
                    expected.shape(),
                    "slot={slot} state={id:?} cache shape"
                );
                let actual = device.read_borrowed(actual)?;
                let expected = device.read_borrowed(expected)?;
                assert_eq!(actual.len(), expected.len());
                for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(a.0, b.0, "slot={slot} state={id:?} byte={index}");
                }
            }
        }
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
        for (id, actual) in a {
            let expected = &b[id];
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert_eq!(a.shape(), b.shape(), "slot={slot} state={id:?} state shape");
                bits(&device.read_borrowed(a)?, &device.read_borrowed(b)?);
            }
        }
    }
    Ok(())
}

fn bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element={index}: {a} != {b}");
    }
}

/// Exercise nonzero prompt-worker transfer, plus lease/page reassignment before slot reuse.
fn bind_worker(
    loaded: &LoadedModel,
    pool: &mut SlotPool,
    worker: &DeviceProgram,
    width: usize,
) -> Result<()> {
    for slot in 0..width {
        assert!(pool.take_slot(slot));
        assert!(pool.assign(slot, 64, loaded.device())?);
        pool.bind(
            slot,
            worker.states(),
            worker.fp8_states(),
            worker.position(),
            loaded.device(),
        )?;
    }
    Ok(())
}
