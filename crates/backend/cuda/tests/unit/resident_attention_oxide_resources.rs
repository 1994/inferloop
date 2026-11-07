//! The SIMT graph must keep its owned operands alive after the caller drops them.
use super::*;
use cuda_core::f8e4m3fn;
use cutile::prelude::DeviceOp;
use infer_cuda_kernels::contract::{KvBuffers, KvCache};

/// Positions of the fixture's cache.
const CAPACITY: usize = 128;
/// Graph replays after every operand was dropped.
const REPLAYS: usize = 3;

fn resource_contract(device: &CudaDevice, support: &Support, lanes: usize) -> Result<()> {
    let runtime = device.runtime();
    let query = runtime.upload(vec![0.0f32; lanes * HEADS * DIM])?;
    let keys = runtime.upload(vec![f8e4m3fn(0); KV_HEADS * CAPACITY * DIM])?;
    let values = runtime.upload(vec![f8e4m3fn(0); KV_HEADS * CAPACITY * DIM])?;
    let metadata = runtime.upload(vec![0i32, 1, 0, 0])?;
    let table = runtime.upload(vec![0i32, 1, 2, 3])?;
    let output = runtime.upload(vec![0.0f32; lanes * HEADS * DIM])?;
    let graph = runtime.capture(|recording| {
        support
            .record(
                recording,
                DecodeBuffers {
                    output: output.buffer(),
                    query: query.buffer(),
                    cache: KvCache {
                        buffers: KvBuffers::Fp8 {
                            keys: keys.operand(),
                            values: values.operand(),
                        },
                        groups: KV_HEADS,
                        dim: DIM,
                        scales: [0.5, 2.0],
                    },
                    metadata: metadata.buffer(),
                    table: table.buffer(),
                },
                (HEADS, KV_HEADS, DIM, lanes, CAPACITY),
                0,
            )
            .map_err(device_error)
    })?;
    drop((query, keys, values, metadata, table));
    // These allocations may be reclaimed only after the graph releases its resources.
    for _ in 0..REPLAYS {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let result = runtime.read(&output)?;
        assert!(
            result.iter().all(|value| value.to_bits() == 0),
            "retained zero fixture changed"
        );
    }
    println!(
        "{}",
        serde_json::json!({"oxide_graph_resources":true,
        "operands_dropped_before_replay":true,"replays":REPLAYS})
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA/safe-run; graph operand lifetime"]
fn rust_attention_graph_retains_operands() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let support =
        Support::load(&device)?.ok_or_else(|| infer_core::Error::invalid("SM120 required"))?;
    resource_contract(&device, &support, 64)
}
#[test]
#[ignore = "requires CUDA/safe-run; shared KV graph operand lifetime"]
fn shared_attention_graph_retains_operands() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let support = Support::load_shared(&device)?
        .ok_or_else(|| infer_core::Error::invalid("SM120 required"))?;
    assert_eq!(support.rows_per_cta, 12);
    resource_contract(&device, &support, 256)?;
    resource_contract(&device, &support, 2048)
}

#[test]
#[ignore = "requires CUDA/safe-run; vector KV graph operand lifetime"]
fn vector_attention_graph_retains_operands() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let support = Support::load_vector(&device)?
        .ok_or_else(|| infer_core::Error::invalid("SM120 required"))?;
    assert_eq!(support.rows_per_cta, 12);
    resource_contract(&device, &support, 256)
}

#[test]
#[ignore = "requires CUDA/safe-run; combined KV graph operand lifetime"]
fn combined_attention_graph_retains_operands() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let support = Support::load_combined(&device)?
        .ok_or_else(|| infer_core::Error::invalid("SM120 required"))?;
    assert_eq!(support.rows_per_cta, 12);
    resource_contract(&device, &support, 256)
}
