use super::update;
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::Result;
use infer_cuda_kernels::tile::runtime::Graph;
use std::time::Instant;

#[test]
#[ignore = "requires an NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn metadata_ordering_and_overhead() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let mut input = api::zeros::<i32>(&[4])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut output = api::zeros::<i32>(&[4])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = Graph::scope(&device.stream, |scope| {
        scope.record(api::memcpy(&mut output, &input))?;
        Ok(())
    })
    .map_err(device_error)?;
    let output = Arc::new(output);
    // Alternating fields check stale updates, negative inactive lanes and MTP flags.
    for values in [
        [0, 248_319, -1, 0],
        [32_767, 0, 3, 1],
        [61, 18, 61, 0],
        [i32::MIN, i32::MAX, 0x7fa0_0001, -1],
    ] {
        update(&graph, &mut input, values)?;
        let actual = graph
            .launch()
            .then(|()| (&output).to_host_vec())
            .sync_on(&device.stream)
            .map_err(device_error)?;
        assert_eq!(actual, values);
    }
    let mut legacy = Vec::new();
    let mut asynchronous = Vec::new();
    for round in 0..6 {
        // Alternate order to limit clock/thermal ordering bias.
        for offset in 0..2 {
            let old = (round + offset) % 2 == 0;
            let start = Instant::now();
            for token in 0..200 {
                let values = [token, token + 1, token, token % 2];
                if old {
                    let uploaded = device.upload(values.to_vec(), &[4])?;
                    graph
                        .update(api::memcpy(&mut input, &uploaded))
                        .map_err(device_error)?;
                } else {
                    update(&graph, &mut input, values)?;
                }
                graph
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(device_error)?;
            }
            let elapsed = start.elapsed().as_secs_f64() * 1e6 / 200.0;
            if old {
                legacy.push(elapsed);
            } else {
                asynchronous.push(elapsed);
            }
            let actual = (&output)
                .to_host_vec()
                .sync_on(&device.stream)
                .map_err(device_error)?;
            assert_eq!(actual, [199, 200, 199, 1]);
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "device": device.name()?, "iterations_per_round": 200,
            "legacy_wall_us": legacy, "async_scalar_wall_us": asynchronous,
            "scope": "metadata update plus empty copy graph and final sync; not model throughput"
        })
    );
    Ok(())
}

#[test]
#[ignore = "requires an NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn rotary_metadata_preserves_independent_axes() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let mut input = api::zeros::<i32>(&[8])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut output = api::zeros::<i32>(&[8])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = Graph::scope(&device.stream, |scope| {
        scope.record(api::memcpy(&mut output, &input))?;
        Ok(())
    })
    .map_err(device_error)?;
    let output = Arc::new(output);
    for (control, coordinates) in [
        ([4, 17, 81, 1], [4, 9, 14]),
        ([29, 151_645, 95, 0], [29, 29, 29]),
    ] {
        super::update_rotary(&graph, &mut input, control, coordinates)?;
        let actual = graph
            .launch()
            .then(|()| (&output).to_host_vec())
            .sync_on(&device.stream)
            .map_err(device_error)?;
        assert_eq!(&actual[..4], &control);
        assert_eq!(actual[4], control[0]);
        assert_eq!(actual[5], i32::try_from(coordinates[1]).unwrap());
        assert_eq!(actual[6], i32::try_from(coordinates[2]).unwrap());
    }
    Ok(())
}
