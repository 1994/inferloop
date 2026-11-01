//! Reset must clear every byte of rank-1 and recurrent state, including a partial tile.
use super::Clear;
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error;

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn captured_reset_clears_every_state_shape() -> Result<(), Box<dyn Error>> {
    let device = CudaDevice::new(0)?;
    for shape in [vec![257], vec![2, 128, 128], vec![1, 64, 128]] {
        let elements = shape.iter().product();
        let mut tensor = api::zeros::<f32>(&shape).sync_on(&device.stream)?;
        let source = device.upload(vec![7.25f32; elements], &shape)?;
        let reset = Graph::scope(&device.stream, |scope| scope.record(Clear(&mut tensor)))?;
        for _ in 0..2 {
            reset.update(api::memcpy(&mut tensor, &source))?;
            reset.launch().sync_on(&device.stream)?;
            let values = device.read_borrowed(&tensor).map_err(device_error)?;
            assert_eq!(values.len(), elements);
            assert!(values.iter().all(|value| value.to_bits() == 0));
        }
    }
    Ok(())
}
