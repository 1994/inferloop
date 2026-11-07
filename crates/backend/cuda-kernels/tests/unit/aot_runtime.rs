//! The AOT runtime on a real device: memory, copies, queries, graph replay, PTX launches, pinned
//! memory and timing.
use super::*;
use std::pin::pin;
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

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

/// Stores `value` at `out[threadIdx.x]`; the driver compiles it for whatever device runs it.
const FILL_PTX: &str = r"
.version 7.0
.target sm_52
.address_size 64
.visible .entry fill(.param .u64 out, .param .u32 value)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [value];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r2, %tid.x;
    mul.wide.u32 %rd3, %r2, 4;
    add.s64 %rd2, %rd2, %rd3;
    st.global.u32 [%rd2], %r1;
    ret;
}
";

/// Threads of one fill launch, and the elements it writes.
const FILL_LEN: u32 = 32;

fn fill(
    device: &Device,
    function: &Arc<Function>,
    out: &Allocation<u32>,
    value: u32,
) -> KernelLaunch {
    // SAFETY: `fill` takes an address and a u32, and writes `FILL_LEN` u32s at the address, which
    // `out` holds; every caller keeps `out` alive until the launch or its graph has run.
    unsafe {
        KernelLaunch::new(
            Arc::clone(function),
            device.stream_id(),
            1,
            FILL_LEN,
            vec![Argument::buffer(out.buffer()), Argument::U32(value)],
        )
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn ptx_launches_run_in_operations_and_captures() -> Result<()> {
    let (device, _) = Device::open(0)?;
    let (other, _) = Device::open(0)?;
    let function = Arc::new(device.load_ptx(FILL_PTX)?.function("fill")?);
    assert!(device.load_ptx(FILL_PTX)?.function("missing").is_err());
    let out = device.zeros::<u32>(FILL_LEN as usize)?;
    let launch = fill(&device, &function, &out, 7);
    now(device.operation(move |operation| operation.launch(launch)))?;
    assert_eq!(device.read(&out)?, vec![7; FILL_LEN as usize]);
    let made = now(device.operation(|operation| {
        // SAFETY: the launch below writes every element before the host reads it.
        let made = unsafe { operation.overwritten::<u32>(FILL_LEN as usize) }?;
        operation.launch(fill(&device, &function, &made, 8))?;
        Ok(made)
    }))?;
    assert_eq!(device.read(&made)?, vec![8; FILL_LEN as usize]);
    let launch = fill(&device, &function, &out, 9);
    let owner = launch.owner();
    let graph = device.capture(|recording| recording.foreign(launch, vec![owner]))?;
    device.sync(graph.launch())?;
    assert_eq!(device.read(&out)?, vec![9; FILL_LEN as usize]);
    let stray = fill(&device, &function, &out, 1);
    assert!(now(other.operation(move |operation| operation.launch(stray))).is_err());
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn readbacks_and_uploads_move_through_pinned_memory() -> Result<()> {
    let (device, _) = Device::open(0)?;
    let source = device.upload((0..16).collect::<Vec<u32>>())?;
    let pinned = Arc::new(Mutex::new(device.pinned::<u32>(8)?));
    assert!(Readback::new(source.clone(), 12..20, Arc::clone(&pinned), 0).is_err());
    let mut copies = Copies::with_capacity(2);
    copies.push(Readback::new(source.clone(), 4..8, Arc::clone(&pinned), 2)?);
    copies.push(Readback::new(
        source.clone(),
        15..16,
        Arc::clone(&pinned),
        7,
    )?);
    now(device.read_back(copies))?;
    let expected = [0, 0, 4, 5, 6, 7, 0, 15];
    assert_eq!(
        pinned
            .lock()
            .map_err(|_| Error::invariant("poisoned"))?
            .as_slice(),
        expected
    );
    let mut overflowing = Copies::with_capacity(1);
    overflowing.push(Readback::new(source, 0..4, Arc::clone(&pinned), 6)?);
    assert!(now(device.read_back(overflowing)).is_err());

    let target = device.zeros::<u32>(4)?;
    let mut staging = device.staging::<u32>(4, 2)?;
    device.enqueue(|transfers| transfers.upload(&mut staging, target.operand(), &[1, 2, 3, 4]))?;
    assert_eq!(device.read(&target)?, [1, 2, 3, 4]);
    assert!(
        device
            .enqueue(|transfers| transfers.upload(&mut staging, target.operand(), &[1, 2]))
            .is_err()
    );
    assert!(device.staging::<u32>(4, 0).is_err());

    let mut host = device.pinned::<u32>(4)?;
    host.as_mut_slice().copy_from_slice(&[5, 6, 7, 8]);
    now(device.operation(|operation| {
        // SAFETY: `host` is borrowed unchanged until the operation has completed.
        unsafe { operation.upload(&target, &host, 3) }
    }))?;
    assert_eq!(device.read(&target)?, [5, 6, 7, 4]);
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn events_time_replays_and_captured_records() -> Result<()> {
    let (device, _) = Device::open(0)?;
    let function = Arc::new(device.load_ptx(FILL_PTX)?.function("fill")?);
    let out = device.zeros::<u32>(FILL_LEN as usize)?;
    let (start, end) = (timing_event(&device)?, timing_event(&device)?);
    let launch = fill(&device, &function, &out, 3);
    let owner = launch.owner();
    let graph = device.capture(|recording| {
        record_event(recording, &start)?;
        recording.foreign(launch, vec![owner])?;
        record_event(recording, &end)
    })?;
    device.sync(graph.launch())?;
    assert!(start.elapsed_ms(&end)? >= 0.0);

    let timing = Timing {
        warmup: Duration::from_millis(1),
        rep: Duration::from_millis(5),
        min_reps: 3,
        max_reps: 50,
        clear_l2: true,
    };
    let (a, b) = device.time_paired(
        &timing,
        || device.sync(graph.launch()),
        || device.sync(graph.launch().followed_by(|| graph.launch())),
    )?;
    assert_eq!(a.times_ms().len(), b.times_ms().len());
    assert!((3..=50).contains(&a.times_ms().len()));
    let mut sorted = a.times_ms().to_vec();
    sorted.sort_by(f32::total_cmp);
    let within = |low: Option<&f32>, high: Option<&f32>| {
        low.zip(high)
            .is_some_and(|(low, high)| (*low..=*high).contains(&a.median_ms()))
    };
    assert!(within(sorted.first(), sorted.last()));
    assert!(a.median_ms().is_finite() && b.median_ms() >= 0.0);
    let bounds = Timing {
        min_reps: 0,
        ..timing
    };
    assert!(device.time_paired(&bounds, || Ok(()), || Ok(())).is_err());
    Ok(())
}
