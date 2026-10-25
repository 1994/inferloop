use super::*;
use std::hint::black_box;
use std::thread::yield_now;
use std::time::{Duration, Instant};

type Tensors = Vec<Tensor<f32>>;

fn fixtures(device: &CudaDevice) -> Result<(Tensors, Tensors)> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for lane in 0..33_u16 {
        let size = if lane == 32 { 248_320 } else { 5120 };
        let input = device.upload(vec![f32::from(lane); size], &[size])?;
        inputs.push(Arc::try_unwrap(input).map_err(|_| Error::invariant("fixture ownership"))?);
        outputs.push(
            api::zeros::<f32>(&[size])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        );
    }
    Ok((inputs, outputs))
}

#[test]
#[ignore = "requires NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn pinned_readbacks_ordering_and_overhead() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let (mut inputs, mut outputs) = fixtures(&device)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        for (output, input) in outputs.iter_mut().zip(&inputs) {
            scope.record(api::memcpy(output, input))?;
        }
        Ok(())
    })
    .map_err(device_error)?;
    let sources: Vec<_> = outputs
        .into_iter()
        .map(|output| Some(ReadbackSource::whole(Arc::new(output))))
        .collect();
    let mut readbacks = Readbacks::default();
    for token in 0..8_u16 {
        let data = device.upload(vec![f32::from(token); 5120], &[5120])?;
        graph
            .update(api::memcpy(&mut inputs[0], &data))
            .map_err(device_error)?;
        let mut selected = sources.clone();
        if token % 2 == 0 {
            selected[1] = None;
        }
        let results = readbacks.run(&device, &graph, &selected)?;
        for (lane, row) in results.iter().enumerate() {
            if selected[lane].is_none() {
                assert_eq!(row.as_slice(), &[0.0_f32; 0]);
                continue;
            }
            let expected = if lane == 0 {
                token
            } else {
                u16::try_from(lane).map_err(device_error)?
            };
            assert!(
                row.iter()
                    .all(|value| value.to_bits() == f32::from(expected).to_bits())
            );
            assert_eq!(row.len(), if lane == 32 { 248_320 } else { 5120 });
        }
    }
    let retained = Arc::clone(
        readbacks.buffers[0]
            .as_ref()
            .ok_or_else(|| Error::invariant("pinned reuse"))?,
    );
    let _ = readbacks.run(&device, &graph, &sources)?;
    assert!(Arc::ptr_eq(
        &retained,
        readbacks.buffers[0]
            .as_ref()
            .ok_or_else(|| Error::invariant("pinned reuse"))?
    ));
    let over_budget = vec![sources[32].clone(); 68];
    assert_eq!(
        readbacks
            .run(&device, &graph, &over_budget)
            .err()
            .map(|e| e.code),
        Some(ErrorCode::Capacity)
    );
    let other = CudaDevice::new(0)?;
    assert!(readbacks.run(&other, &graph, &sources).is_err());
    measure(&device, &graph, &sources, &mut readbacks)?;
    let mut single = Readbacks::default();
    let _ = single.run(&device, &graph, &sources[32..])?;
    measure(&device, &graph, &sources[32..], &mut single)?;
    let empty = readbacks.run(&device, &graph, &[None, None])?;
    assert_eq!(empty, vec![Vec::<f32>::new(), Vec::new()]);
    partial_readback(&device, &graph, &mut readbacks)?;
    readbacks.poisoned = true;
    assert!(readbacks.run(&device, &graph, &sources).is_err());
    Ok(())
}

/// A mid-tensor range copies exactly those elements; out-of-bounds is rejected.
fn partial_readback(
    device: &CudaDevice,
    graph: &CudaGraph<()>,
    readbacks: &mut Readbacks,
) -> Result<()> {
    let ramp = device.upload((0..8192_u16).map(f32::from).collect::<Vec<f32>>(), &[8192])?;
    let partial = [Some(ReadbackSource {
        tensor: Arc::clone(&ramp),
        skip: 999,
        len: 2345,
    })];
    let rows = readbacks.run(device, graph, &partial)?;
    let expected: Vec<f32> = (999..3344_u16).map(f32::from).collect();
    assert_eq!(rows.len(), 1);
    assert!(
        rows[0]
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    let overflow = [Some(ReadbackSource {
        tensor: ramp,
        skip: 8000,
        len: 193,
    })];
    assert!(readbacks.run(device, graph, &overflow).is_err());
    Ok(())
}

fn measure(
    device: &CudaDevice,
    graph: &CudaGraph<()>,
    sources: &[Option<ReadbackSource>],
    readbacks: &mut Readbacks,
) -> Result<()> {
    let mut legacy = Vec::new();
    let mut pinned = Vec::new();
    for round in 0..6 {
        for offset in 0..2 {
            let old = (round + offset) % 2 == 0;
            let start = Instant::now();
            for _ in 0..100 {
                let rows = if old {
                    let mut reads = DeviceOpVec::with_capacity(sources.len());
                    for source in sources {
                        let source = source
                            .as_ref()
                            .ok_or_else(|| Error::invariant("benchmark source"))?;
                        reads.push(Arc::clone(&source.tensor).to_host_vec());
                    }
                    graph
                        .launch()
                        .then(move |()| reads)
                        .sync_on(&device.stream)
                        .map_err(device_error)?
                } else {
                    readbacks.run(device, graph, sources)?
                };
                black_box(rows);
            }
            let us = start.elapsed().as_secs_f64() * 1e6 / 100.0;
            if old {
                legacy.push(us);
            } else {
                pinned.push(us);
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "device": device.name()?, "rows": sources.len(), "iterations_per_round": 100,
            "legacy_wall_us": legacy, "pinned_wall_us": pinned,
            "scope": "33 graph D2D copies plus selected output D2H, including host Vec materialization; rows=33 is prefill Full, rows=1 is logits only; not model throughput"
        })
    );
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn pinned_float_and_integer_graphs_complete_in_order_and_reuse_buffers() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let float_input = device.upload(vec![3.5_f32; 5], &[5])?;
    let mut integer_input = Arc::try_unwrap(device.upload(vec![0_i32; 6], &[6])?)
        .map_err(|_| Error::invariant("integer fixture ownership"))?;
    let mut middle = api::zeros::<i32>(&[6])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut integer_output = api::zeros::<i32>(&[6])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut float_output = api::zeros::<f32>(&[5])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(api::memcpy(&mut float_output, &float_input))?;
        scope.record(api::memcpy(&mut middle, &integer_input))?;
        Ok(())
    })
    .map_err(device_error)?;
    let tail = CudaGraph::scope(&device.stream, |scope| {
        scope.record(api::memcpy(&mut integer_output, &middle))?;
        Ok(())
    })
    .map_err(device_error)?;
    let floats = [Some(ReadbackSource::whole(Arc::new(float_output))), None];
    let integer_output = Arc::new(integer_output);
    let mut float_reads = Readbacks::default();
    let mut token_reads = Readbacks::<i32>::default();
    for round in 0..8 {
        let expected = vec![round, 0, i32::MAX - round, 1, -round, 0];
        let input = device.upload(expected.clone(), &[6])?;
        graph
            .update(api::memcpy(&mut integer_input, &input))
            .map_err(device_error)?;
        let (rows, tokens) = float_reads.run_with_tokens(
            &device,
            &graph,
            &floats,
            (&tail, &mut token_reads, Arc::clone(&integer_output)),
        )?;
        assert_eq!(rows, vec![vec![3.5; 5], Vec::new()]);
        assert_eq!(tokens, vec![expected.clone()]);
        float_reads.submit_with_tokens(
            &device,
            &graph,
            &floats,
            (&tail, &mut token_reads, Arc::clone(&integer_output)),
        )?;
        let (rows, tokens) = poll_pair(&mut float_reads, &mut token_reads)?;
        assert_eq!(rows, vec![vec![3.5; 5], Vec::new()]);
        assert_eq!(tokens, vec![expected]);
    }
    let retained = Arc::clone(
        token_reads.buffers[0]
            .as_ref()
            .ok_or_else(|| Error::invariant("integer pinned reuse"))?,
    );
    let _ = float_reads.run_with_tokens(
        &device,
        &graph,
        &floats,
        (&tail, &mut token_reads, Arc::clone(&integer_output)),
    )?;
    assert!(Arc::ptr_eq(
        &retained,
        token_reads.buffers[0]
            .as_ref()
            .ok_or_else(|| Error::invariant("integer pinned reuse"))?
    ));
    let other = CudaDevice::new(0)?;
    assert!(
        float_reads
            .run_with_tokens(
                &other,
                &graph,
                &floats,
                (&tail, &mut token_reads, Arc::clone(&integer_output)),
            )
            .is_err()
    );
    token_reads.poisoned = true;
    assert!(
        float_reads
            .run_with_tokens(
                &device,
                &graph,
                &floats,
                (&tail, &mut token_reads, integer_output),
            )
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn split_readback_is_pending_until_stream_completion_and_rejects_reuse() -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread::yield_now;
    use std::time::Duration;

    let device = CudaDevice::new(0)?;
    let input = device.upload(vec![1.0_f32, 2.0, 3.0, 4.0], &[4])?;
    let mut output = api::zeros::<f32>(&[4])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(api::memcpy(&mut output, &input))?;
        Ok(())
    })
    .map_err(device_error)?;
    let sources = [Some(ReadbackSource::whole(Arc::new(output))), None];
    let mut reads = Readbacks::default();
    // Allocate before gating: pinned allocation is allowed to synchronize at startup.
    reads.prepare(&device, &sources)?;
    let released = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    let gate = Arc::clone(&released);
    let expired = Arc::clone(&timed_out);
    queue_test_gate(&device, move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !gate.load(Ordering::Acquire) {
            if Instant::now() >= deadline {
                expired.store(true, Ordering::Release);
                break;
            }
            yield_now();
        }
    })?;
    reads.submit(&device, &graph, &sources)?;
    assert!(
        !timed_out.load(Ordering::Acquire),
        "submit waited for GPU completion"
    );
    assert!(reads.poll()?.is_none());
    assert!(reads.submit(&device, &graph, &sources).is_err());
    assert!(reads.run(&device, &graph, &sources).is_err());
    released.store(true, Ordering::Release);
    // The submission must retain captured resources after the public graph is dropped.
    drop(graph);
    let deadline = Instant::now() + Duration::from_secs(5);
    let rows = loop {
        if let Some(rows) = reads.poll()? {
            break rows;
        }
        assert!(Instant::now() < deadline, "readback failed to complete");
        yield_now();
    };
    assert_eq!(rows, vec![vec![1.0, 2.0, 3.0, 4.0], Vec::new()]);
    assert!(reads.poll().is_err());
    assert!(!reads.poisoned);
    Ok(())
}

fn poll_pair(
    float_reads: &mut Readbacks,
    token_reads: &mut Readbacks<i32>,
) -> Result<(Rows<f32>, Rows<i32>)> {
    let mut foreign = Readbacks::<i32>::default();
    assert!(float_reads.poll_with_tokens(&mut foreign).is_err());
    assert!(float_reads.poll().is_err());
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(output) = float_reads.poll_with_tokens(token_reads)? {
            return Ok(output);
        }
        assert!(Instant::now() < deadline);
        yield_now();
    }
}
