//! The cuTile source of the `pool_gemv_bf16` cell, through the contract, against the shared cases.
#[path = "../../../cuda-kernels/acceptance/pool_gemv.rs"]
pub mod acceptance;

use super::*;
use crate::device::{CudaDevice, device_error};
use acceptance::{Case, OUTPUTS, ROWS};
use cutile::half::bf16;
use infer_core::{Error, Result};
use std::sync::Arc;

/// Reduction lengths the cuTile kernel tiles evenly.
const COLUMNS: [usize; 3] = [64, 128, 256];
/// Output tile and reduction tile the cases run with.
const TILE: (usize, usize) = (16, 64);

fn upload<T: DType>(device: &CudaDevice, data: Vec<T>, shape: &[usize]) -> Result<Tensor<T>> {
    api::copy_host_vec_to_device(&Arc::new(data))
        .sync_on(&device.stream)
        .map_err(device_error)?
        .reshape(shape)
        .map_err(device_error)
}

fn check(device: &CudaDevice, case: Case) -> Result<()> {
    let columns = case.columns;
    let input = upload(device, case.input(), &[ROWS, columns])?;
    let weight = upload(
        device,
        case.weight_bits()
            .into_iter()
            .map(bf16::from_bits)
            .collect(),
        &[OUTPUTS, columns],
    )?;
    let output = upload(device, vec![f32::NAN; ROWS * OUTPUTS], &[ROWS, OUTPUTS])?;
    let tile = LinearTiling::new(TILE.0, TILE.1)?;
    let pointer = input.device_pointer().cu_deviceptr();
    let view = input.view(&[ROWS, columns]).map_err(device_error)?;
    let kernels = device.kernels();
    let weight = Arc::new(weight);
    let graph = CudaGraph::scope(&device.stream, |scope| {
        record_gemv(
            &kernels,
            scope,
            buffer(weight.as_ref())?,
            &view,
            pointer,
            buffer(&output)?,
            columns,
            tile,
        )
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    case.check(&device.read_borrowed(&output)?)
        .map_err(Error::invariant)
}

#[test]
#[ignore = "requires a CUDA device"]
fn tile_source_passes_the_shared_cases() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    COLUMNS
        .into_iter()
        .try_for_each(|columns| check(&device, Case { columns }))
}
