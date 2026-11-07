//! The AOT runtime on a real device: memory, copies, queries and graph replay.
use super::*;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

/// Ready futures resolve on their first poll; anything else is a runtime bug here.
fn now<T>(future: impl Future<Output = T>) -> T {
    match pin!(future).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("AOT runtime futures complete before they are returned"),
    }
}

fn ramp(len: u16) -> Vec<f32> {
    (0..len).map(f32::from).collect()
}

#[test]
#[ignore = "requires a CUDA device"]
fn memory_round_trips_through_slices_rows_and_reinterpretation() -> Result<()> {
    let (device, _) = Device::open(0)?;
    let values = device.upload(ramp(64))?;
    assert_eq!(device.read(&values)?, ramp(64));
    let rows = values.rows(4)?;
    assert_eq!(device.read(&rows[2])?, ramp(64)[32..48]);
    assert_eq!(device.read(&values.slice(5..9)?)?, ramp(64)[5..9]);
    assert!(values.slice(60..70).is_err());
    let bits = values.reinterpret::<u32>()?;
    assert_eq!(device.read(&bits)?[3], 3.0_f32.to_bits());
    let zeros = now(device.zeroed::<f32>(16))?;
    assert_eq!(device.read(&zeros)?, vec![0.0; 16]);
    assert!(values.watch().watches(&rows[1]));
    assert_eq!(values.identity(), rows[3].identity());
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn transfers_copy_buffers_and_head_prefixes() -> Result<()> {
    let (device, _) = Device::open(0)?;
    let source = device.upload(ramp(64))?;
    let target = device.zeros::<f32>(16)?;
    let rows = source.rows(4)?;
    device.enqueue(|transfers| transfers.copy(target.operand(), rows[1].operand()))?;
    assert_eq!(device.read(&target)?, ramp(64)[16..32]);
    // Two heads of eight positions of width four, copied as three positions into capacity four.
    let heads = device.zeros::<f32>(2 * 4 * 4)?;
    device.enqueue(|transfers| {
        transfers.copy_heads(
            HeadMajor::new(heads.operand(), [2, 4, 4])?,
            HeadMajor::new(source.operand(), [2, 8, 4])?,
            3,
        )
    })?;
    let copied = device.read(&heads)?;
    assert_eq!(copied[..12], ramp(64)[..12]);
    assert_eq!(copied[12..16], [0.0; 4]);
    assert_eq!(copied[16..28], ramp(64)[32..44]);
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn queries_describe_the_device_and_driver() -> Result<()> {
    let (device, arch) = Device::open(0)?;
    assert!(device.attribute(Attribute::ComputeMajor)? >= 7);
    assert_eq!(device.attribute(Attribute::WarpSize)?, 32);
    assert!(device.attribute(Attribute::Multiprocessors)? > 0);
    assert_ne!(device.name()?, "");
    let (free, total) = device.memory_info()?;
    assert!(free <= total && total > 0);
    assert!(driver_version()? >= 12_000);
    assert_eq!(
        arch,
        Arch::from_sm(
            device.attribute(Attribute::ComputeMajor)?,
            device.attribute(Attribute::ComputeMinor)?
        )
    );
    assert!(device.shares_stream(&device.clone()));
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_captured_graph_replays_on_its_own_stream() -> Result<()> {
    let (device, _) = Device::open(0)?;
    let (other, _) = Device::open(0)?;
    let graph = device.capture(|_| Ok(()))?;
    assert!(graph.runs_on(&device) && !graph.runs_on(&other));
    now(device.run(graph.launch().followed_by(|| graph.launch())))?;
    device.submit(&graph)?;
    assert!(other.submit(&graph).is_err());
    graph.drain()?;
    now(device.fence())
}

/// The attribute numbers this runtime spells out match the driver header's.
#[cfg(all(feature = "tile", target_os = "linux"))]
#[test]
fn attribute_numbers_match_the_driver_header() {
    use cuda_core::sys::{
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE as CLOCK,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as MAJOR,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as MINOR,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH as BUS,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE as L2,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE as MEMORY_CLOCK,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as SMS,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE as WARP,
    };
    let header = |value: u32| i32::try_from(value).ok();
    for (attribute, value) in [
        (Attribute::ComputeMajor, MAJOR),
        (Attribute::ComputeMinor, MINOR),
        (Attribute::Multiprocessors, SMS),
        (Attribute::WarpSize, WARP),
        (Attribute::ClockKhz, CLOCK),
        (Attribute::MemoryClockKhz, MEMORY_CLOCK),
        (Attribute::MemoryBusWidthBits, BUS),
        (Attribute::L2CacheBytes, L2),
    ] {
        assert_eq!(Some(attribute.driver()), header(value), "{attribute:?}");
    }
}
