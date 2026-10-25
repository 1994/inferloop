use super::support::{self, ProtocolBackend};
use infer_core::{Error, ErrorCode, ModelId, Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, OutputReadout, PrecisionPlan,
    StepPlan, TaskOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig, runner::ThreadedBackend};
use infer_spi::{BackendProvider, ResourceCommand};
use std::{
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, ThreadId},
    time::{Duration, Instant},
};

#[derive(Default)]
struct Observed {
    advance: AtomicBool,
    failure: AtomicBool,
    maintenance_calls: AtomicUsize,
    polls: AtomicUsize,
    submitted: AtomicUsize,
    owned: AtomicUsize,
    ticket_drops: AtomicUsize,
    premature_drops: AtomicUsize,
    backend_drops: AtomicUsize,
    wrong_thread: AtomicUsize,
    owner: Mutex<Option<ThreadId>>,
    cleanup_ready: Mutex<Option<Weak<AtomicBool>>>,
    wake: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}
impl Observed {
    fn record_owner(&self, caller: ThreadId) -> Result<()> {
        let current = thread::current().id();
        let mut owner = self
            .owner
            .lock()
            .map_err(|error| Error::invariant(error.to_string()))?;
        if current == caller || owner.is_some_and(|expected| expected != current) {
            self.wrong_thread.fetch_add(1, Ordering::AcqRel);
        }
        owner.get_or_insert(current);
        drop(owner);
        Ok(())
    }
}

struct MaintainedTicket {
    inner: <ProtocolBackend as BackendProvider>::Ticket,
    ready: Arc<AtomicBool>,
    done: bool,
    observed: Arc<Observed>,
}
impl Drop for MaintainedTicket {
    fn drop(&mut self) {
        self.observed.ticket_drops.fetch_add(1, Ordering::AcqRel);
        if !self.done {
            self.observed.premature_drops.fetch_add(1, Ordering::AcqRel);
        }
    }
}

struct MaintainedBackend {
    inner: ProtocolBackend,
    observed: Arc<Observed>,
    caller: ThreadId,
    pending: Option<Arc<AtomicBool>>,
}
impl Drop for MaintainedBackend {
    fn drop(&mut self) {
        self.observed.backend_drops.fetch_add(1, Ordering::AcqRel);
    }
}
impl BackendProvider for MaintainedBackend {
    type Ticket = MaintainedTicket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn set_waker(&mut self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.observed.record_owner(self.caller).unwrap();
        *self.observed.wake.lock().unwrap() = Some(wake);
    }
    fn maintenance(&mut self) -> Result<()> {
        self.observed.record_owner(self.caller)?;
        self.observed
            .maintenance_calls
            .fetch_add(1, Ordering::AcqRel);
        // Only this owner hook can advance the simulated host command/fence.
        if self.observed.advance.load(Ordering::Acquire)
            && let Some(ready) = &self.pending
        {
            ready.store(true, Ordering::Release);
        }
        if self.observed.failure.load(Ordering::Acquire) {
            return Err(Error::new(ErrorCode::Backend, "maintenance failure"));
        }
        Ok(())
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.observed.record_owner(self.caller)?;
        self.inner.reserve_state_for(state, capacity, readout)?;
        self.observed.owned.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.observed.record_owner(self.caller)?;
        self.inner.release_state(state)?;
        self.observed.owned.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.observed.record_owner(self.caller)?;
        let ready = Arc::new(AtomicBool::new(false));
        *self
            .observed
            .cleanup_ready
            .lock()
            .map_err(|error| Error::invariant(error.to_string()))? = Some(Arc::downgrade(&ready));
        let inner = self.inner.submit(program, step, tasks)?;
        self.pending = Some(ready.clone());
        self.observed.submitted.fetch_add(1, Ordering::AcqRel);
        Ok(MaintainedTicket {
            inner,
            ready,
            done: false,
            observed: self.observed.clone(),
        })
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        self.observed.record_owner(self.caller)?;
        self.observed.polls.fetch_add(1, Ordering::AcqRel);
        if !ticket.ready.load(Ordering::Acquire) {
            return Ok(None);
        }
        let result = self.inner.poll(&mut ticket.inner)?;
        if result.is_some() {
            ticket.done = true;
            self.pending = None;
        }
        Ok(result)
    }
}

struct Fixture {
    engine: Engine<ThreadedBackend<MaintainedBackend>>,
    observed: Arc<Observed>,
}
impl Fixture {
    fn new(advance: bool) -> Result<Self> {
        let model = support::model(ModelId::ONE);
        let observed = Arc::new(Observed::default());
        observed.advance.store(advance, Ordering::Release);
        let backend = MaintainedBackend {
            inner: ProtocolBackend::new(16, 8, &model)?,
            observed: observed.clone(),
            caller: thread::current().id(),
            pending: None,
        };
        let mut registry = KernelRegistry::default();
        registry.register(&support::DeclaredKernels)?;
        let engine = Engine::new(
            backend,
            model,
            PrecisionPlan::f32(),
            &registry,
            RuntimeConfig {
                max_num_seqs: 1,
                ..RuntimeConfig::default()
            },
        )?
        .into_threaded(4, Duration::from_millis(20))?;
        Ok(Self { engine, observed })
    }
    fn reserve(&mut self, deadline: Instant) -> Result<()> {
        let mut reserve = self
            .engine
            .backend_mut()
            .begin_resource(ResourceCommand::Reserve {
                state: StateId::ONE,
                capacity: 16,
                readout: OutputReadout::Full,
            })?;
        while reserve.poll()?.is_none() {
            wait(deadline)?;
        }
        Ok(())
    }
    fn step(&self) -> Arc<StepPlan> {
        Arc::new(StepPlan {
            id: infer_core::StepId::ONE,
            decision: infer_core::DecisionId::ONE,
            program: self.engine.program().id,
            role: infer_ir::ExecutionRole::Prefill,
            work: vec![infer_ir::PlannedWork {
                request: infer_core::RequestId::ONE,
                state: StateId::ONE,
                token_count: 3,
                role: infer_ir::ExecutionRole::Prefill,
            }],
            cost: infer_ir::CostEstimate::default(),
            graph: None,
            quantum_overrun: false,
        })
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // Cleanup can force the fence even when the old worker never calls maintenance.
        self.observed.advance.store(true, Ordering::Release);
        if let Ok(pending) = self.observed.cleanup_ready.lock()
            && let Some(ready) = pending.as_ref().and_then(Weak::upgrade)
        {
            ready.store(true, Ordering::Release);
        }
    }
}
fn tasks() -> Vec<ExecutionTask> {
    vec![ExecutionTask {
        plain_greedy: false,
        request: infer_core::RequestId::ONE,
        state: StateId::ONE,
        tokens: vec![1, 2, 3].into(),
        sampling: None,
    }]
}
fn wait(deadline: Instant) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(Error::invariant("maintenance owner did not drain"));
    }
    thread::sleep(Duration::from_micros(100));
    Ok(())
}

#[test]
fn backend_notification_wakes_the_parked_device_owner() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    fixture.reserve(Instant::now() + Duration::from_secs(3))?;
    // Resource completion lets the idle owner reach its one-second park. The device
    // notification must resume maintenance without another command or compute batch.
    thread::sleep(Duration::from_millis(20));
    let before = fixture.observed.maintenance_calls.load(Ordering::Acquire);
    let wake = fixture.observed.wake.lock().unwrap().clone().unwrap();
    wake();
    let deadline = Instant::now() + Duration::from_millis(200);
    while fixture.observed.maintenance_calls.load(Ordering::Acquire) == before {
        wait(deadline)?;
    }
    assert_eq!(fixture.observed.wrong_thread.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn owner_maintenance_advances_an_outstanding_ticket() -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut fixture = Fixture::new(true)?;
    fixture.reserve(deadline)?;
    let program = fixture.engine.program().clone();
    let step = fixture.step();
    let mut ticket = fixture
        .engine
        .backend_mut()
        .submit_shared(&program, step, tasks())?;
    let outputs = loop {
        if let Some(outputs) = fixture.engine.backend_mut().poll(&mut ticket)? {
            break outputs;
        }
        wait(deadline)?;
    };
    assert_eq!(outputs.len(), 1);
    while fixture.observed.ticket_drops.load(Ordering::Acquire) == 0 {
        wait(deadline)?;
    }
    assert_eq!(fixture.observed.submitted.load(Ordering::Acquire), 1);
    assert!(fixture.observed.maintenance_calls.load(Ordering::Acquire) > 0);
    assert_eq!(fixture.observed.wrong_thread.load(Ordering::Acquire), 0);
    assert_eq!(fixture.observed.ticket_drops.load(Ordering::Acquire), 1);
    assert_eq!(fixture.observed.premature_drops.load(Ordering::Acquire), 0);
    fixture.engine.backend_mut().release_state(StateId::ONE)?;
    while fixture.engine.backend().pending_resource_releases() {
        fixture.engine.backend_mut().maintenance()?;
        wait(deadline)?;
    }
    assert_eq!(fixture.observed.owned.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn maintenance_failure_preserves_flight_and_release_until_the_fence() -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut fixture = Fixture::new(false)?;
    fixture.reserve(deadline)?;
    let program = fixture.engine.program().clone();
    let step = fixture.step();
    let mut ticket = fixture
        .engine
        .backend_mut()
        .submit_shared(&program, step, tasks())?;
    while fixture.observed.polls.load(Ordering::Acquire) == 0 {
        wait(deadline)?;
    }
    fixture.observed.failure.store(true, Ordering::Release);
    loop {
        if let Err(error) = fixture.engine.backend().free_state_bytes() {
            assert_eq!(error.code, ErrorCode::Backend);
            assert_eq!(error.message, "maintenance failure");
            break;
        }
        wait(deadline)?;
    }
    assert!(fixture.engine.backend_mut().poll(&mut ticket)?.is_none());
    assert_eq!(fixture.observed.owned.load(Ordering::Acquire), 1);
    assert_eq!(fixture.observed.ticket_drops.load(Ordering::Acquire), 0);
    assert_eq!(fixture.observed.backend_drops.load(Ordering::Acquire), 0);
    drop(ticket);
    fixture.engine.backend_mut().release_state(StateId::ONE)?;
    let polls = fixture.observed.polls.load(Ordering::Acquire);
    while fixture.observed.polls.load(Ordering::Acquire) < polls + 2 {
        wait(deadline)?;
    }
    assert_eq!(fixture.observed.owned.load(Ordering::Acquire), 1);
    assert_eq!(fixture.observed.ticket_drops.load(Ordering::Acquire), 0);
    fixture.observed.advance.store(true, Ordering::Release);
    while fixture.engine.backend().pending_resource_releases()
        || fixture.observed.ticket_drops.load(Ordering::Acquire) == 0
    {
        fixture.engine.backend_mut().maintenance()?;
        wait(deadline)?;
    }
    assert_eq!(fixture.observed.owned.load(Ordering::Acquire), 0);
    assert_eq!(fixture.observed.ticket_drops.load(Ordering::Acquire), 1);
    assert_eq!(fixture.observed.premature_drops.load(Ordering::Acquire), 0);
    assert_eq!(fixture.observed.wrong_thread.load(Ordering::Acquire), 0);
    let observed = fixture.observed.clone();
    drop(fixture);
    while observed.backend_drops.load(Ordering::Acquire) == 0 {
        wait(deadline)?;
    }
    assert_eq!(observed.backend_drops.load(Ordering::Acquire), 1);
    Ok(())
}
