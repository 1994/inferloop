//! Fixed task groups isolate completion polling from scheduler/admission variability.
use super::*;
use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::thread::{current, park_timeout};

fn digest(outputs: &[TaskOutput]) -> u64 {
    let mut hash = DefaultHasher::new();
    for row in outputs {
        row.request.hash(&mut hash);
        row.output.next_token.hash(&mut hash);
        row.output.tokens.hash(&mut hash);
        for value in row
            .output
            .logits
            .iter()
            .chain(row.output.hidden.iter().flatten())
        {
            value.to_bits().hash(&mut hash);
        }
    }
    hash.finish()
}

fn measured_step(
    backend: &mut CudaBackend,
    program: &ExecutionProgram,
    tasks: &[ExecutionTask],
) -> TestResult<serde_json::Value> {
    let plan = plan(backend, program, tasks)?;
    let started = Instant::now();
    let mut ticket = backend.submit_borrowed(program, &plan, tasks)?;
    let submitted = started.elapsed();
    let mut polls = 0;
    let mut poll_wall = Duration::ZERO;
    let mut interval = 25_u64;
    let deadline = started + Duration::from_secs(30);
    let outputs = loop {
        let polling = Instant::now();
        let output = backend.poll(&mut ticket)?;
        poll_wall += polling.elapsed();
        polls += 1;
        if let Some(outputs) = output {
            break outputs;
        }
        assert!(
            Instant::now() < deadline,
            "fixed batch completion timed out"
        );
        interval = (interval * 2).min(1000);
        park_timeout(Duration::from_micros(interval));
    };
    let elapsed = started.elapsed();
    Ok(serde_json::json!({
        "tasks": tasks.len(),
        "prefills": tasks.iter().filter(|t| !matches!(t.tokens, ExecutionInput::Decode { .. })).count(),
        "wall_us": elapsed.as_micros(), "submit_wall_us": submitted.as_micros(),
        "poll_calls": polls, "poll_wall_us": poll_wall.as_micros(),
        "output_digest": digest(&outputs),
        "accepted_tokens": outputs.iter().map(|row| row.output.tokens.len()).collect::<Vec<_>>(),
    }))
}

#[test]
#[ignore = "real MTP2 checkpoint/CUDA; fixed-batch diagnostic under safe-run with profile paths"]
fn fixed_groups_measure_host_polling_and_device_graphs() -> TestResult {
    let output = var("INFER_TEST_PROFILE_OUT")?;
    let mut report = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    let (mut backend, program, states) = fixture_with_width(2, &[OutputReadout::Logits; 4], 2048)?;
    let owner = current();
    backend.set_waker(Arc::new(move || owner.unpark()));
    let mut reference = BTreeMap::new();
    for mixed in [false, true] {
        for repeat in 0..6 {
            for &state in &states {
                backend.reset_state(state)?;
            }
            for round in 0..17 {
                let active = if mixed && round == 0 {
                    &states[..1]
                } else {
                    &states
                };
                let mut tasks = tasks(&backend, active, round)?;
                for task in &mut tasks {
                    task.plain_greedy = true;
                    if let ExecutionInput::Prefill { readout, .. } = &mut task.tokens {
                        *readout = OutputReadout::Logits;
                    }
                }
                let mut record = measured_step(&mut backend, &program, &tasks)?;
                let key = (mixed, round);
                if repeat == 0 {
                    reference.insert(key, record["output_digest"].clone());
                } else {
                    assert_eq!(reference.get(&key), Some(&record["output_digest"]));
                }
                record["mixed"] = mixed.into();
                record["repeat"] = repeat.into();
                record["round"] = round.into();
                record["warmup"] = (repeat == 0).into();
                writeln!(report, "{record}")?;
            }
        }
    }
    for state in states {
        backend.release_state(state)?;
    }
    backend.validate_state_ownership(&[])?;
    Ok(())
}
