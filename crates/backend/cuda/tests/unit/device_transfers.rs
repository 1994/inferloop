//! Eager transfers keep their owners alive until the stream passes them, and copy exactly the
//! ranges they name.
use super::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::Result;
use infer_cuda_kernels::runtime::{Allocation, HeadMajor};
use std::sync::Arc;

/// Large enough that the copy is still running when the host drops its source.
const LARGE: usize = 1 << 26;

/// A distinct finite value per index, exact for every index a test uses.
fn value(index: usize) -> f32 {
    f32::from_bits(u32::try_from(index).unwrap_or_default())
}

fn ramp(len: usize) -> Vec<f32> {
    (0..len).map(value).collect()
}

#[test]
#[ignore = "requires CUDA; run release serially inside safe-run"]
fn enqueued_copy_retains_a_dropped_source_until_reaped() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let values = ramp(LARGE);
    let tensor = device.upload(values.clone(), &[LARGE])?;
    let memory = Arc::downgrade(&tensor);
    let source = Allocation::adopt(tensor)?;
    let target = runtime.zeros::<f32>(LARGE)?;
    runtime.enqueue(|transfers| transfers.copy(target.buffer(), source.buffer()))?;
    drop(source);
    assert!(
        memory.upgrade().is_some(),
        "the batch holds the source while the copy may still run"
    );
    device.synchronize()?;
    assert!(
        memory.upgrade().is_none(),
        "a synchronized device releases what finished transfers held"
    );
    assert_eq!(runtime.read(&target)?, values);
    Ok(())
}

#[test]
#[ignore = "requires CUDA; run release serially inside safe-run"]
fn copy_moves_disjoint_ranges_and_rejects_partial_overlap() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let memory = runtime.upload(ramp(8))?;
    let whole = memory.buffer();
    runtime.enqueue(|transfers| transfers.copy(whole.slice(0..4)?, whole.slice(4..8)?))?;
    runtime.enqueue(|transfers| transfers.copy(whole.slice(2..6)?, whole.slice(2..6)?))?;
    assert!(
        runtime
            .enqueue(|transfers| transfers.copy(whole.slice(0..4)?, whole.slice(2..6)?))
            .is_err()
    );
    assert!(
        runtime
            .enqueue(|transfers| transfers.copy(whole.slice(0..4)?, whole.slice(0..3)?))
            .is_err()
    );
    let upper = ramp(8).split_off(4);
    assert_eq!(runtime.read(&memory)?, [upper.clone(), upper].concat());
    Ok(())
}

#[test]
#[ignore = "requires CUDA; run release serially inside safe-run"]
fn head_copy_moves_the_prefix_of_every_head_and_keeps_the_tails() -> Result<()> {
    const HEADS: usize = 2;
    const DIM: usize = 3;
    const SOURCE: usize = 4;
    const TARGET: usize = 5;
    const ROWS: usize = 2;
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let source = runtime.upload(ramp(HEADS * SOURCE * DIM))?;
    let target = runtime.upload(vec![-1.0; HEADS * TARGET * DIM])?;
    runtime.enqueue(|transfers| {
        transfers.copy_heads(
            HeadMajor::new(target.buffer(), [HEADS, TARGET, DIM])?,
            HeadMajor::new(source.buffer(), [HEADS, SOURCE, DIM])?,
            ROWS,
        )
    })?;
    let mut expected = vec![-1.0; HEADS * TARGET * DIM];
    for head in 0..HEADS {
        for element in 0..ROWS * DIM {
            expected[head * TARGET * DIM + element] = value(head * SOURCE * DIM + element);
        }
    }
    assert_eq!(runtime.read(&target)?, expected);
    assert!(HeadMajor::new(source.buffer(), [HEADS, TARGET, DIM]).is_err());
    assert!(
        runtime
            .enqueue(|transfers| {
                transfers.copy_heads(
                    HeadMajor::new(target.buffer(), [HEADS, TARGET, DIM])?,
                    HeadMajor::new(source.buffer(), [HEADS, SOURCE, DIM])?,
                    SOURCE + 1,
                )
            })
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA; run release serially inside safe-run"]
fn staging_refills_a_slot_only_after_the_device_read_it() -> Result<()> {
    const UPLOADS: usize = 4;
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let mut staging = runtime.staging::<f32>(LARGE, 1)?;
    let rows = (0..UPLOADS)
        .map(|upload| {
            (0..LARGE)
                .map(|index| value(index ^ upload))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let targets = (0..UPLOADS)
        .map(|_| runtime.zeros::<f32>(LARGE))
        .collect::<Result<Vec<_>>>()?;
    for (row, target) in rows.iter().zip(&targets) {
        runtime.enqueue(|transfers| transfers.upload(&mut staging, target.buffer(), row))?;
    }
    for (row, target) in rows.iter().zip(&targets) {
        assert!(
            runtime.read(target)? == *row,
            "an upload kept what it staged"
        );
    }
    assert!(
        runtime
            .enqueue(|transfers| {
                transfers.upload(&mut staging, targets[0].buffer(), &rows[0][..1])
            })
            .is_err(),
        "a destination of another length is rejected"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA; run release serially inside safe-run"]
fn recorded_copy_replays_from_a_source_only_the_graph_holds() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let values = ramp(LARGE);
    let source = runtime.upload(values.clone())?;
    let target = runtime.zeros::<f32>(LARGE)?;
    let graph = runtime.capture(|recording| recording.copy(target.buffer(), source.buffer()))?;
    drop(source);
    graph
        .launch()
        .sync_on(device.stream())
        .map_err(device_error)?;
    assert_eq!(runtime.read(&target)?, values);
    Ok(())
}
