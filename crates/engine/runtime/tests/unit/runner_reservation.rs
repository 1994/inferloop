use super::{deadline, request, support};
use infer_core::{Error, ErrorCode, ModelId, Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, OutputReadout, PrecisionPlan,
    StepPlan, TaskOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::{BackendProvider, ResourceCommand, ResourceReply};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, ThreadId},
    time::{Duration, Instant},
};

struct QuoteBackend {
    inner: support::ProtocolBackend,
    prepared_on: Arc<Mutex<Option<ThreadId>>>,
    quotes: Arc<AtomicUsize>,
    fail: bool,
    geometry: Arc<Mutex<Option<infer_ir::PrefillGeometry>>>,
    profile: Arc<Mutex<Option<infer_ir::ExecutionProfileInspection>>>,
}
impl BackendProvider for QuoteBackend {
    type Ticket = <support::ProtocolBackend as BackendProvider>::Ticket;
    fn execution_profile(&self) -> Option<infer_ir::ExecutionProfileInspection> {
        self.profile.lock().ok().and_then(|profile| profile.clone())
    }
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn prepare_state_reservation(&mut self, _: usize, _: OutputReadout) -> Result<()> {
        if self.fail {
            return Err(Error::new(ErrorCode::Backend, "quote preparation failed"));
        }
        *self
            .prepared_on
            .lock()
            .map_err(|e| Error::invariant(e.to_string()))? = Some(thread::current().id());
        Ok(())
    }
    fn state_reservation_bytes_for(&self, _: usize, _: OutputReadout) -> Result<Option<u64>> {
        if self
            .prepared_on
            .lock()
            .map_err(|e| Error::invariant(e.to_string()))?
            .is_none()
        {
            return Err(Error::invariant("quote preceded preparation"));
        }
        self.quotes.fetch_add(1, Ordering::AcqRel);
        Ok(Some(1024))
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        let prepared = self
            .prepared_on
            .lock()
            .map_err(|e| Error::invariant(e.to_string()))?;
        Ok(Some(if prepared.is_some() { 2048 } else { 0 }))
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.inner.reserve_state_for(state, capacity, readout)
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.inner.reset_state(state)
    }
    fn state_prefill_geometry(&self, _: StateId) -> Result<Option<infer_ir::PrefillGeometry>> {
        self.geometry
            .lock()
            .map(|geometry| *geometry)
            .map_err(|e| Error::invariant(e.to_string()))
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)
    }
    fn submit(
        &mut self,
        p: &ExecutionProgram,
        s: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.inner.submit(p, s, tasks)
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        self.inner.poll(ticket)
    }
}
fn engine(fail: bool) -> Result<Engine<QuoteBackend>> {
    let model = support::model(ModelId::ONE);
    let backend = QuoteBackend {
        inner: support::ProtocolBackend::new(16, 8, &model)?,
        prepared_on: Arc::default(),
        quotes: Arc::default(),
        fail,
        geometry: Arc::default(),
        profile: Arc::default(),
    };
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )
}

#[test]
fn direct_admission_prepares_before_sizing_and_capacity_check() -> Result<()> {
    let mut engine = engine(false)?;
    assert_eq!(engine.backend().free_state_bytes()?, Some(0));
    engine.submit(request(1)?)?;
    assert_eq!(engine.backend().quotes.load(Ordering::Acquire), 1);
    assert_eq!(
        *engine
            .backend()
            .prepared_on
            .lock()
            .map_err(|e| Error::invariant(e.to_string()))?,
        Some(thread::current().id())
    );
    Ok(())
}

#[test]
fn threaded_quote_publishes_reclaimed_capacity_before_acknowledgement() -> Result<()> {
    let direct = engine(false)?;
    let prepared_on = direct.backend().prepared_on.clone();
    let quotes = direct.backend().quotes.clone();
    let mut engine = direct.into_threaded(4, Duration::from_millis(20))?;
    assert_eq!(engine.backend().free_state_bytes()?, Some(0));
    let prepared = engine.request_preparer()?.prepare(request(1)?)?;
    let mut quote = engine.admission_quote(&prepared)?;
    let limit = Instant::now() + Duration::from_secs(3);
    let bytes = loop {
        if let Some(ResourceReply::ReservationBytes(bytes)) = quote.poll()? {
            break bytes;
        }
        deadline(limit)?;
    };
    // No maintenance/tick between acknowledgement and the snapshot read.
    assert_eq!(bytes, Some(1024));
    assert_eq!(engine.backend().free_state_bytes()?, Some(2048));
    assert_eq!(quotes.load(Ordering::Acquire), 1);
    assert_ne!(
        *prepared_on
            .lock()
            .map_err(|e| Error::invariant(e.to_string()))?,
        Some(thread::current().id())
    );
    engine.submit_quoted_with_trace(prepared, bytes, None)?;
    Ok(())
}

#[test]
fn failed_preparation_does_not_publish_a_reservation_price() -> Result<()> {
    let mut engine = engine(true)?;
    let mut quote = engine
        .backend_mut()
        .begin_resource(ResourceCommand::ReservationBytes {
            capacity: 16,
            readout: OutputReadout::Logits,
        })?;
    assert_eq!(quote.poll().err().map(|e| e.code), Some(ErrorCode::Backend));
    assert_eq!(engine.backend().quotes.load(Ordering::Acquire), 0);
    assert_eq!(engine.backend().free_state_bytes()?, Some(0));
    assert_eq!(
        engine.submit(request(1)?).err().map(|e| e.code),
        Some(ErrorCode::Backend)
    );
    assert_eq!(engine.backend().quotes.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn threaded_reservation_publishes_and_retires_actual_prompt_geometry() -> Result<()> {
    let direct = engine(false)?;
    let geometry = infer_ir::PrefillGeometry {
        width: 1024,
        narrow_width: 64,
    };
    let shared_geometry = direct.backend().geometry.clone();
    *shared_geometry
        .lock()
        .map_err(|e| Error::invariant(e.to_string()))? = Some(geometry);
    let mut engine = direct.into_threaded(4, Duration::from_millis(20))?;
    assert_eq!(engine.backend().state_prefill_geometry(StateId::ONE)?, None);
    let before = engine.backend().resource_epoch();
    let mut reserve = engine
        .backend_mut()
        .begin_resource(ResourceCommand::Reserve {
            state: StateId::ONE,
            capacity: 16,
            readout: OutputReadout::Logits,
        })?;
    let limit = Instant::now() + Duration::from_secs(3);
    while reserve.poll()?.is_none() {
        deadline(limit)?;
    }
    // An acknowledged reservation must carry its shapes before the first planning round.
    assert_eq!(
        engine.backend().state_prefill_geometry(StateId::ONE)?,
        Some(geometry)
    );
    assert!(engine.backend().resource_epoch() > before);
    let epoch = engine.backend().resource_epoch();
    let changed = infer_ir::PrefillGeometry {
        width: 512,
        narrow_width: 32,
    };
    *shared_geometry
        .lock()
        .map_err(|e| Error::invariant(e.to_string()))? = Some(changed);
    let mut reset = engine
        .backend_mut()
        .begin_resource(ResourceCommand::Reset {
            state: StateId::ONE,
        })?;
    while reset.poll()?.is_none() {
        deadline(limit)?;
    }
    assert_eq!(
        engine.backend().state_prefill_geometry(StateId::ONE)?,
        Some(changed)
    );
    assert!(engine.backend().resource_epoch() > epoch);
    engine.backend_mut().release_state(StateId::ONE)?;
    let mut barrier = engine
        .backend_mut()
        .begin_resource(ResourceCommand::ReservationBytes {
            capacity: 16,
            readout: OutputReadout::Logits,
        })?;
    while barrier.poll()?.is_none() {
        deadline(limit)?;
    }
    assert_eq!(engine.backend().state_prefill_geometry(StateId::ONE)?, None);
    Ok(())
}

#[test]
fn threaded_resource_ack_publishes_the_current_execution_profile() -> Result<()> {
    use infer_ir::{ConcurrencyInspection, ExecutionProfileInspection};
    let direct = engine(false)?;
    let shared_profile = direct.backend().profile.clone();
    let cold = ExecutionProfileInspection {
        concurrency: Some(ConcurrencyInspection {
            memory_bound: 24,
            roofline_bound: Some(52),
            selected: 24,
        }),
        prefill_width: 1024,
        pooled_prefill_width: None,
        batch_width: 3,
        draft_depth: 2,
        automatic_prefill: true,
        arena_budget_bytes: 1024,
        attention_profile: "bf16".into(),
        precise_prefix_positions: Some(0),
    };
    *shared_profile
        .lock()
        .map_err(|error| Error::invariant(error.to_string()))? = Some(cold.clone());
    let mut threaded = direct.into_threaded(4, Duration::from_millis(20))?;
    assert_eq!(threaded.backend().execution_profile(), Some(cold.clone()));
    let mut current = cold;
    current.pooled_prefill_width = Some(1024);
    current.concurrency.as_mut().unwrap().selected = 15;
    *shared_profile
        .lock()
        .map_err(|error| Error::invariant(error.to_string()))? = Some(current.clone());
    let prepared = threaded.request_preparer()?.prepare(request(1)?)?;
    let mut quote = threaded.admission_quote(&prepared)?;
    let limit = Instant::now() + Duration::from_secs(3);
    while quote.poll()?.is_none() {
        deadline(limit)?;
    }
    assert_eq!(threaded.backend().execution_profile(), Some(current));
    *shared_profile
        .lock()
        .map_err(|error| Error::invariant(error.to_string()))? = None;
    let mut quote = threaded.admission_quote(&prepared)?;
    let limit = Instant::now() + Duration::from_secs(3);
    while quote.poll()?.is_none() {
        deadline(limit)?;
    }
    assert_eq!(threaded.backend().execution_profile(), None);
    Ok(())
}
