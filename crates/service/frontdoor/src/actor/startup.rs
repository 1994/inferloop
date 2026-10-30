//! Startup responsibilities.
use super::{RuntimeActor, RuntimeHandle, mpsc, observation_bytes};
use crate::cpu::CpuConfig;
use crate::cpu::CpuPool;
use crate::ingress::Registry;
use infer_core::Result;
use infer_runtime::Engine;
use infer_spi::BackendProvider;
use std::sync::atomic::AtomicU64;
use std::{sync::Arc, sync::atomic::AtomicBool, time::Duration, time::Instant};

/// Microseconds in one millisecond, for the device control timeout conversion.
const MICROS_PER_MILLI: u64 = 1000;
/// Depth of the device submission queue owned by the threaded engine backend.
const ENGINE_SUBMISSION_QUEUE: usize = 32;
/// Bound on the actor's bulk command channel.
const COMMAND_CHANNEL_CAPACITY: usize = 256;
/// Bound on the actor's control command channel.
const CONTROL_CHANNEL_CAPACITY: usize = 64;

impl RuntimeHandle {
    /// # Errors
    /// Returns a backend/state error if stale completions cannot be drained or an I/O error if the runtime thread cannot start.
    pub fn start<B: BackendProvider + Send + 'static>(engine: Engine<B>) -> Result<Self>
    where
        B::Ticket: Send,
    {
        let config = CpuConfig {
            device_control_timeout_ms: engine
                .config()
                .resource_timeout_us
                .div_ceil(MICROS_PER_MILLI),
            ..CpuConfig::default()
        };
        Self::start_with_config(engine, config)
    }
    /// # Errors
    /// Rejects invalid CPU limits, unforkable preparation providers or startup failures.
    pub fn start_with_config<B: BackendProvider + Send + 'static>(
        engine: Engine<B>,
        config: CpuConfig,
    ) -> Result<Self>
    where
        B::Ticket: Send,
    {
        config.validate()?;
        let request_ids = Arc::new(AtomicU64::new(engine.request_id_high_watermark()));
        let preparer = engine.request_preparer()?;
        let cpu = CpuPool::new(config.clone())?;
        let delivery = CpuPool::named(
            CpuConfig {
                workers: 1,
                max_jobs: config.max_jobs,
                ..config.clone()
            },
            "delivery",
        )?;
        let observation = CpuPool::named(
            CpuConfig {
                workers: 1,
                max_jobs: 2,
                ..config.clone()
            },
            "observe",
        )?;
        let observation_bytes = observation_bytes(engine.config())?;
        let engine = engine.into_threaded(
            ENGINE_SUBMISSION_QUEUE,
            Duration::from_millis(config.device_control_timeout_ms),
        )?;
        let (sender, receiver) = mpsc::sync_channel(COMMAND_CHANNEL_CAPACITY);
        let (control, control_rx) = mpsc::sync_channel(CONTROL_CHANNEL_CAPACITY);
        let stop = Arc::new(AtomicBool::new(false));
        let actor_stop = stop.clone();
        let clock_started = Instant::now();
        let clock_origin_us = engine.now_us();
        let placement = engine.config().cpu.placement.scheduler.clone();
        let thread = placement.spawn("infer-scheduler".into(), move |_| {
            RuntimeActor::new(engine, receiver, control_rx, actor_stop, clock_started).run();
        })?;
        Ok(Self {
            commands: Arc::new(sender),
            control: Arc::new(control),
            wake: thread.thread().clone(),
            stop,
            clock_started,
            clock_origin_us,
            request_ids,
            preparer,
            cpu,
            delivery,
            observation,
            observation_bytes,
            ingress: Registry::new(config.max_jobs, config.max_bytes),
            config,
        })
    }
}
