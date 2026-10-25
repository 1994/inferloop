//! Execution CLI support.
use crate::backend::SelectedBackend;
use crate::constants::DEVICE_POLL_INTERVAL;
use crate::constants::MAX_IDLE_TICKS;
use infer_core::event::SemanticEvent;
use infer_core::{Error, RequestId, Result};
use infer_ir::CanonicalRequest;
use infer_runtime::Engine;
use std::thread::sleep;
use std::time::Instant;

pub fn run_to_idle(engine: &mut Engine<SelectedBackend>) -> Result<Vec<SemanticEvent>> {
    run_to_idle_started(engine, engine.now_us(), Instant::now())
}

fn run_to_idle_started(
    engine: &mut Engine<SelectedBackend>,
    clock: u64,
    started: Instant,
) -> Result<Vec<SemanticEvent>> {
    let mut events = engine.drain_events();
    let start = engine.inspect().global_progress_epoch;
    for tick in 0..MAX_IDLE_TICKS {
        if engine.is_idle() {
            return Ok(events);
        }
        let elapsed = if engine.program().backend.is_device() {
            u64::try_from(started.elapsed().as_micros())
                .map_err(|_| Error::invalid("CLI clock overflow"))?
        } else {
            tick
        };
        engine.tick(
            clock
                .checked_add(elapsed)
                .ok_or_else(|| Error::invalid("CLI clock overflow"))?,
        )?;
        events.extend(engine.drain_events());
        if engine.program().backend.is_device() && engine.inspect().inflight_step.is_some() {
            sleep(DEVICE_POLL_INTERVAL);
        }
    }
    Err(Error::invariant(format!(
        "runtime exceeded tick limit; initial progress epoch {start}"
    )))
}
pub fn results(
    engine: &Engine<SelectedBackend>,
    ids: &[RequestId],
) -> Result<Vec<infer_runtime::CompletedRequest>> {
    ids.iter()
        .map(|id| {
            engine
                .request(*id)?
                .completed
                .clone()
                .ok_or_else(|| Error::invariant("request has no completion"))
        })
        .collect()
}
pub fn run_requests(
    engine: &mut Engine<SelectedBackend>,
    requests: Vec<CanonicalRequest>,
) -> Result<Vec<SemanticEvent>> {
    let clock = engine.now_us();
    let started = Instant::now();
    for request in requests {
        engine.submit(request)?;
    }
    run_to_idle_started(engine, clock, started)
}
