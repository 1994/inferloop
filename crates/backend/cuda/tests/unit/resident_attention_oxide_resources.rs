//! The SIMT graph must retain inputs and honor cross-stream output exclusivity.
use super::*;
use crate::device::device_error;
use cuda_core::f8e4m3fn;
use infer_cuda_kernels::tile::runtime::Graph;

fn upload<T: DType>(device: &CudaDevice, data: Vec<T>, shape: &[usize]) -> Result<Tensor<T>> {
    api::copy_host_vec_to_device(&Arc::new(data))
        .sync_on(&device.stream)
        .map_err(device_error)?
        .reshape(shape)
        .map_err(device_error)
}
fn resource_contract(device: &CudaDevice, support: &Support, lanes: usize) -> Result<()> {
    let query = upload(
        device,
        vec![0.0; lanes * HEADS * DIM],
        &[lanes, HEADS * DIM],
    )?;
    let mut keys = upload(
        device,
        vec![f8e4m3fn(0); KV_HEADS * 128 * DIM],
        &[KV_HEADS, 128, DIM],
    )?;
    let values = upload(
        device,
        vec![f8e4m3fn(0); KV_HEADS * 128 * DIM],
        &[KV_HEADS, 128, DIM],
    )?;
    let metadata = upload(device, vec![0, 1, 0, 0], &[4])?;
    let table = upload(device, vec![0, 1, 2, 3], &[4])?;
    let mut output = upload(
        device,
        vec![0.0; lanes * HEADS * DIM],
        &[lanes * HEADS, DIM],
    )?;
    let graph = Graph::scope(&device.stream, |scope| {
        support.record(
            scope,
            Bindings {
                output: &mut output,
                query: &query,
                keys: &keys,
                values: &values,
                metadata: &metadata,
                table: &table,
            },
            (HEADS, KV_HEADS, DIM, lanes, 128),
            0,
            [0.5, 2.0],
        )?;
        Ok(())
    })
    .map_err(device_error)?;
    let other = ExecutionContext::new(device.stream.device().new_stream().map_err(device_error)?);
    KernelOutputStored::retain(&(&mut keys).partition([1, 128, DIM]), &other)
        .map_err(device_error)?;
    assert!(
        graph.launch().sync_on(&device.stream).is_err(),
        "input write on another stream must block graph reads"
    );
    drop(other);
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let other = ExecutionContext::new(device.stream.device().new_stream().map_err(device_error)?);
    KernelInputStored::retain(&&output, &other).map_err(device_error)?;
    assert!(
        graph.launch().sync_on(&device.stream).is_err(),
        "output read on another stream must block graph writes"
    );
    drop(other);
    drop((query, keys, values, metadata, table));
    // These allocations may be reclaimed only after the graph releases its resources.
    for _ in 0..3 {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let result = device.read_borrowed(&output)?;
        assert!(
            result.iter().all(|value| value.to_bits() == 0),
            "retained zero fixture changed"
        );
    }
    println!(
        "{}",
        serde_json::json!({"oxide_graph_resources":true,"input_exclusive_access":true,
        "output_exclusive_access":true,"operands_dropped_before_replay":true,"replays":3})
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA/safe-run; graph operand lifetime and cross-stream access contract"]
fn rust_attention_graph_retains_operands_and_write_lease() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let support =
        Support::load(&device)?.ok_or_else(|| infer_core::Error::invalid("SM120 required"))?;
    resource_contract(&device, &support, 64)
}
#[test]
#[ignore = "requires CUDA/safe-run; shared KV graph operand lifetime and exclusive access"]
fn shared_attention_graph_retains_operands_and_write_lease() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let support = Support::load_shared(&device)?
        .ok_or_else(|| infer_core::Error::invalid("SM120 required"))?;
    assert_eq!(support.rows_per_cta, 12);
    resource_contract(&device, &support, 256)?;
    resource_contract(&device, &support, 2048)
}

#[test]
#[ignore = "requires CUDA/safe-run; vector KV graph ownership and exclusive access"]
fn vector_attention_graph_retains_operands_and_write_lease() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let support = Support::load_vector(&device)?
        .ok_or_else(|| infer_core::Error::invalid("SM120 required"))?;
    assert_eq!(support.rows_per_cta, 12);
    resource_contract(&device, &support, 256)
}

#[test]
#[ignore = "requires CUDA/safe-run; combined KV graph ownership and exclusive access"]
fn combined_attention_graph_retains_operands_and_write_lease() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let support = Support::load_combined(&device)?
        .ok_or_else(|| infer_core::Error::invalid("SM120 required"))?;
    assert_eq!(support.rows_per_cta, 12);
    resource_contract(&device, &support, 256)
}
