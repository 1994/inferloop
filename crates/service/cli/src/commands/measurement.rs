//! Measurement commands.
#[cfg(native_backend)]
use super::{
    BenchmarkOptions, ProfileOptions, backend, config, example, results, run_to_idle,
    selected_engine, write_json,
};
use super::{CompareOptions, print, read_json};
#[cfg(native_backend)]
use crate::constants::MAX_IDLE_TICKS;
use infer_core::{Error, Result};
#[cfg(native_backend)]
use infer_ir::{RequestInput, Workload};
#[cfg(native_backend)]
use infer_runtime::{ReplayAction, RuntimeConfig};
#[cfg(native_backend)]
use std::time::Instant;

/// Maximum request count one benchmark run admits.
#[cfg(native_backend)]
const MAX_BENCHMARK_REQUESTS: usize = 256;

#[cfg(native_backend)]
pub fn profile(options: ProfileOptions, backend_choice: backend::Selection) -> Result<()> {
    let ProfileOptions {
        journal,
        output,
        config: config_path,
        package,
        host_memory_mib,
    } = options;
    let actions: Vec<ReplayAction> = read_json(&journal)?;
    let mut engine = selected_engine(
        config(config_path.as_deref())?,
        None,
        package.as_deref(),
        host_memory_mib,
        backend_choice,
    )?;
    let mut events = Vec::new();
    for action in actions {
        engine.replay(&[action])?;
        events.extend(engine.drain_events());
    }
    if !engine.is_idle() {
        events.extend(run_to_idle(&mut engine)?);
    }
    let trace:Vec<_>=events.into_iter().map(|e|serde_json::json!({"name":format!("{:?}",e.kind),"cat":format!("{:?}",e.object_kind),"ph":"i","s":"t","ts":e.timestamp_us,"pid":1,"tid":e.object_kind as u16,"args":{"object_id":e.object_id,"correlation_id":e.correlation_id,"arg0":e.arg0,"arg1":e.arg1}})).collect();
    let artifact = if package.is_some() {
        engine.backend().profile()
    } else {
        serde_json::json!({"displayTimeUnit":"us","traceEvents":trace})
    };
    write_json(&output, &artifact)?;
    print(
        &serde_json::json!({"output":output,"scope":if package.is_some(){"selected backend execution profile"}else{"CPU/control semantic timeline"},"backend":engine.program().backend,"gpu_counters":false}),
    )?;

    Ok(())
}
#[cfg(native_backend)]
pub fn benchmark(options: BenchmarkOptions, backend_choice: backend::Selection) -> Result<()> {
    let BenchmarkOptions {
        requests,
        input_tokens,
        output_tokens,
        ttft_slo_us,
        tpot_slo_us,
        output,
        package,
        host_memory_mib,
    } = options;
    let mut engine = selected_engine(
        RuntimeConfig::default(),
        None,
        package.as_deref(),
        host_memory_mib,
        backend_choice,
    )?;
    if requests == 0
        || requests > MAX_BENCHMARK_REQUESTS
        || input_tokens == 0
        || input_tokens
            .checked_add(output_tokens)
            .is_none_or(|v| v > engine.model().max_sequence)
        || output_tokens == 0
    {
        return Err(Error::invalid(
            "benchmark limits: 1..256 requests, total context within model capacity",
        ));
    }
    let start = Instant::now();
    let mut ids = Vec::new();
    for id in 1..=requests {
        let mut request = example(
            id as u64,
            Workload::Generate {
                max_new_tokens: output_tokens,
            },
        )?;
        request.input = RequestInput::Sequence {
            tokens: (0..input_tokens)
                .map(|i| {
                    u32::try_from(i % engine.model().vocab_size)
                        .map_err(|_| Error::invalid("vocabulary exceeds token ABI"))
                })
                .collect::<Result<Vec<_>>>()?
                .into(),
            media: vec![],
        };
        ids.push(request.id);
        engine.submit(request)?;
    }
    for _ in 0..MAX_IDLE_TICKS {
        if engine.is_idle() {
            break;
        }
        engine.tick(u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX))?;
        engine.drain_events();
    }
    if !engine.is_idle() {
        return Err(Error::invariant("benchmark did not finish"));
    }
    let samples = results(&engine, &ids)?
        .into_iter()
        .map(|r| r.measurement)
        .collect::<Vec<_>>();
    let report = infer_quality::benchmark(
        engine.inspect().backend,
        format!(
            "closed-generate-v2:requests={requests}:input={input_tokens}:output={output_tokens}"
        ),
        &samples,
        (u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)).max(1),
        ttft_slo_us,
        tpot_slo_us,
    )?;
    if let Some(path) = output {
        write_json(&path, &report)?;
    }
    print(&report)?;

    Ok(())
}
pub fn compare(options: CompareOptions) -> Result<()> {
    let CompareOptions {
        baseline,
        candidate,
        correctness_passed,
        max_p99_regression,
    } = options;
    let verdict = infer_quality::experiment(
        &read_json(&baseline)?,
        &read_json(&candidate)?,
        correctness_passed,
        max_p99_regression,
    )?;
    print(&verdict)?;
    if !verdict.accepted {
        return Err(Error::invariant("experiment rejected"));
    }

    Ok(())
}
