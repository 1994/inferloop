//! A bounded worker queue that wakes the most recently active idle worker first.
use super::Job;
use infer_core::{Error, ErrorCode, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

#[derive(Default)]
struct Worker {
    idle: bool,
    notified: bool,
}

struct State {
    jobs: VecDeque<Job>,
    workers: Vec<Worker>,
    preferred: usize,
    closed: bool,
}

pub(super) struct WorkQueue {
    state: Mutex<State>,
    ready: Vec<Condvar>,
    capacity: usize,
}

impl WorkQueue {
    pub(super) fn new(workers: usize, capacity: usize) -> Self {
        Self {
            state: Mutex::new(State {
                jobs: VecDeque::new(),
                workers: (0..workers).map(|_| Worker::default()).collect(),
                preferred: 0,
                closed: false,
            }),
            ready: (0..workers).map(|_| Condvar::new()).collect(),
            capacity,
        }
    }

    fn submit(&self, job: Job) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::invariant("CPU work queue poisoned"))?;
        if state.closed || state.jobs.len() >= self.capacity {
            return Err(Error::new(ErrorCode::Capacity, "CPU queue unavailable"));
        }
        state.jobs.try_reserve(1).map_err(|error| {
            Error::new(
                ErrorCode::Capacity,
                format!("CPU work queue allocation: {error}"),
            )
        })?;
        state.jobs.push_back(job);
        let available = |worker: &Worker| worker.idle && !worker.notified;
        let preferred = state.preferred;
        let target = if available(&state.workers[preferred]) {
            Some(preferred)
        } else {
            state.workers.iter().position(available)
        };
        if let Some(index) = target {
            state.workers[index].notified = true;
        }
        drop(state);
        if let Some(index) = target {
            self.ready[index].notify_one();
        }
        Ok(())
    }

    /// Mark a completed worker available before publishing its result to the caller.
    pub(super) fn available(&self, index: usize) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::invariant("CPU work queue poisoned"))?;
        if !state.workers[index].idle {
            state.workers[index].idle = true;
            state.workers[index].notified = false;
        }
        state.preferred = index;
        drop(state);
        Ok(())
    }

    /// Wait for queued work, draining existing jobs after the last sender closes.
    pub(super) fn next(&self, index: usize) -> Result<Option<Job>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::invariant("CPU work queue poisoned"))?;
        loop {
            if let Some(job) = state.jobs.pop_front() {
                state.workers[index].idle = false;
                state.workers[index].notified = false;
                return Ok(Some(job));
            }
            if state.closed {
                return Ok(None);
            }
            state.workers[index].idle = true;
            state.workers[index].notified = false;
            state = self.ready[index]
                .wait(state)
                .map_err(|_| Error::invariant("CPU work queue poisoned"))?;
        }
    }

    fn close(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.closed = true;
        }
        for ready in &self.ready {
            ready.notify_all();
        }
    }
}

/// The last sender closes the queue and wakes every parked worker.
pub(super) struct WorkSender(pub(super) Arc<WorkQueue>);

impl WorkSender {
    pub(super) fn submit(&self, job: Job) -> Result<()> {
        self.0.submit(job)
    }
}

impl Drop for WorkSender {
    fn drop(&mut self) {
        self.0.close();
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cpu_queue.rs"]
mod tests;
