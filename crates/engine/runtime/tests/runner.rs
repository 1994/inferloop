#[path = "unit/runner_bytes_capacity.rs"]
mod bytes_capacity;
#[path = "unit/runner_capacity_progress.rs"]
mod capacity_progress;
#[path = "unit/runner_maintenance.rs"]
mod maintenance;
#[path = "unit/runner_queue_deadline.rs"]
mod queue_deadline;
#[path = "unit/runner_reservation.rs"]
mod reservation;
mod support;
use infer_core::{Error, FinishReason, ModelId, Result, StateId};
use infer_ir::{
    CanonicalRequest, DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, ModelOutput,
    OutputReadout, PrecisionPlan, StepPlan, TaskOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig, runner::ThreadedBackend};
use std::collections::{BTreeMap, BTreeSet};
use std::thread::sleep;

use infer_spi::{BackendProvider, ResourceCommand, ResourceReply, ResourceTicket};
use std::{
    sync::Arc, sync::atomic::AtomicBool, sync::atomic::AtomicUsize, sync::atomic::Ordering,
    sync::mpsc, time::Duration, time::Instant,
};
use support::ProtocolBackend;

struct ReserveGate {
    inner: ProtocolBackend,
    draft_depth: usize,
    gate: Option<mpsc::Receiver<()>>,
    entered: mpsc::Sender<()>,
    owned: Arc<AtomicUsize>,
    reservation_limit: Arc<AtomicUsize>,
    byte_capacity: Arc<AtomicUsize>,
    reservation_attempts: Arc<AtomicUsize>,
    observed_work: Arc<AtomicUsize>,
    observed_sampling: Arc<AtomicUsize>,
    poll_gate: Arc<AtomicBool>,
    poll_permits: Arc<AtomicUsize>,
}
impl BackendProvider for ReserveGate {
    type Ticket = <ProtocolBackend as BackendProvider>::Ticket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn speculation_capability(&self) -> infer_ir::SpeculationCapability {
        infer_ir::SpeculationCapability {
            draft_depth: self.draft_depth,
            greedy_only: true,
        }
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
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        // The threaded runner reserves through a resource command, which reaches this entry point;
        // keeping the gate on `reserve_state` alone would bypass it.
        self.reservation_attempts.fetch_add(1, Ordering::AcqRel);
        if self.owned.load(Ordering::Acquire) >= self.reservation_limit.load(Ordering::Acquire) {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "resident budget busy",
            ));
        }
        self.inner.reserve_state_for(state, capacity, readout)?;
        self.owned.fetch_add(1, Ordering::AcqRel);
        self.entered
            .send(())
            .map_err(|e| Error::invariant(e.to_string()))?;
        if let Some(gate) = self.gate.take() {
            gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
        }
        Ok(())
    }
    fn state_capacity_bytes(&self) -> Result<Option<u64>> {
        match self.byte_capacity.load(Ordering::Acquire) {
            usize::MAX => Ok(None),
            bytes => u64::try_from(bytes)
                .map(Some)
                .map_err(|e| Error::invalid(e.to_string())),
        }
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        Ok(self.state_capacity_bytes()?.map(|bytes| {
            bytes.saturating_sub(
                u64::try_from(self.owned.load(Ordering::Acquire))
                    .unwrap_or(u64::MAX)
                    .saturating_mul(1024),
            )
        }))
    }
    fn state_reservation_bytes_for(&self, _: usize, _: OutputReadout) -> Result<Option<u64>> {
        Ok(self.state_capacity_bytes()?.map(|_| 1024))
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.inner.reset_state(state)
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)?;
        self.owned.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }
    fn recycle_output(&mut self, state: StateId, output: ModelOutput) -> Result<()> {
        self.inner.recycle_output(state, output)
    }
    fn recycle_batch(&mut self, outputs: Vec<TaskOutput>) -> Result<()> {
        self.inner.recycle_batch(outputs)
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.observed_work
            .store(step.work.as_ptr() as usize, Ordering::Release);
        self.observed_sampling.fetch_add(
            tasks.iter().filter(|task| task.sampling.is_some()).count(),
            Ordering::Release,
        );
        self.inner.submit(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        if !self.poll_gate.load(Ordering::Acquire) || self.poll_permits.load(Ordering::Acquire) == 0
        {
            return Ok(None);
        }
        let result = self.inner.poll(ticket)?;
        if result.is_some() {
            self.poll_permits.fetch_sub(1, Ordering::AcqRel);
        }
        Ok(result)
    }
}
type TestEngine = Engine<ThreadedBackend<ReserveGate>>;
struct Fixture {
    engine: TestEngine,
    release: mpsc::Sender<()>,
    entered: mpsc::Receiver<()>,
    owned: Arc<AtomicUsize>,
    reservation_limit: Arc<AtomicUsize>,
    byte_capacity: Arc<AtomicUsize>,
    reservation_attempts: Arc<AtomicUsize>,
    observed_work: Arc<AtomicUsize>,
    observed_sampling: Arc<AtomicUsize>,
    poll_gate: Arc<AtomicBool>,
    poll_permits: Arc<AtomicUsize>,
}
impl Fixture {
    fn new(gated: bool) -> Result<Self> {
        Self::with_draft(gated, 0)
    }
    fn with_draft(gated: bool, draft_depth: usize) -> Result<Self> {
        Self::with_timeouts(gated, draft_depth, None, Duration::from_millis(20))
    }
    fn with_timeouts(
        gated: bool,
        draft_depth: usize,
        queue_wait_us: Option<u64>,
        timeout: Duration,
    ) -> Result<Self> {
        let ir = support::model(ModelId::ONE);
        let (release, gate) = mpsc::channel();
        let (entered, start) = mpsc::channel();
        let owned = Arc::new(AtomicUsize::new(0));
        let reservation_limit = Arc::new(AtomicUsize::new(usize::MAX));
        let byte_capacity = Arc::new(AtomicUsize::new(usize::MAX));
        let reservation_attempts = Arc::new(AtomicUsize::new(0));
        let observed_work = Arc::new(AtomicUsize::new(0));
        let observed_sampling = Arc::new(AtomicUsize::new(0));
        let poll_gate = Arc::new(AtomicBool::new(true));
        let poll_permits = Arc::new(AtomicUsize::new(usize::MAX));
        let backend = ReserveGate {
            inner: ProtocolBackend::new(16, 8, &ir)?,
            draft_depth,
            gate: gated.then_some(gate),
            entered,
            owned: owned.clone(),
            reservation_limit: reservation_limit.clone(),
            byte_capacity: byte_capacity.clone(),
            reservation_attempts: reservation_attempts.clone(),
            observed_work: observed_work.clone(),
            observed_sampling: observed_sampling.clone(),
            poll_gate: poll_gate.clone(),
            poll_permits: poll_permits.clone(),
        };
        let mut registry = KernelRegistry::default();
        registry.register(&support::DeclaredKernels)?;
        let engine = Engine::new(
            backend,
            ir,
            PrecisionPlan::f32(),
            &registry,
            RuntimeConfig {
                max_num_seqs: 1,
                // Queue-deadline scenes can wait beyond the default device timeout.
                submission_timeout_us: u64::try_from(timeout.as_micros())
                    .unwrap_or(u64::MAX)
                    .max(RuntimeConfig::default().submission_timeout_us),
                max_queue_wait_us: queue_wait_us
                    .unwrap_or_else(|| RuntimeConfig::default().max_queue_wait_us),
                ..RuntimeConfig::default()
            },
        )?
        .into_threaded(4, timeout)?;
        Ok(Self {
            engine,
            release,
            entered: start,
            owned,
            reservation_limit,
            byte_capacity,
            reservation_attempts,
            observed_work,
            observed_sampling,
            poll_gate,
            poll_permits,
        })
    }
    fn admit(&mut self, id: u64) -> Result<()> {
        self.admit_request(request(id)?)
    }
    fn admit_request(&mut self, request: CanonicalRequest) -> Result<()> {
        let prepared = self.engine.request_preparer()?.prepare(request)?;
        let mut quote = self.engine.admission_quote(&prepared)?;
        let limit = Instant::now() + Duration::from_secs(3);
        let bytes = loop {
            if let Some(ResourceReply::ReservationBytes(bytes)) = quote.poll()? {
                break bytes;
            }
            deadline(limit)?;
        };
        self.engine.submit_quoted_with_trace(prepared, bytes, None)
    }
    fn drain(&mut self) -> Result<()> {
        let limit = Instant::now() + Duration::from_secs(3);
        while !self.engine.is_idle() {
            self.engine.tick(self.engine.now_us() + 1)?;
            deadline(limit)?;
        }
        self.engine.check_invariants()
    }
}
fn poll_quote(ticket: &mut ResourceTicket) -> Result<Option<u64>> {
    let limit = Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(ResourceReply::ReservationBytes(bytes)) = ticket.poll()? {
            return Ok(bytes);
        }
        if Instant::now() >= limit {
            return Err(Error::invariant("threaded quote did not acknowledge"));
        }
        sleep(Duration::from_millis(1));
    }
}
fn deadline(limit: Instant) -> Result<()> {
    if Instant::now() >= limit {
        return Err(Error::invariant("resource owner did not drain"));
    }
    sleep(Duration::from_micros(100));
    Ok(())
}
fn request(id: u64) -> Result<CanonicalRequest> {
    serde_json::from_value(serde_json::json!({"id":id,"model":1,"input":{"Sequence":{"tokens":[1,2,3]}},"workload":{"Generate":{"max_new_tokens":2}}})).map_err(|e| Error::invalid(e.to_string()))
}
#[test]
fn resource_timeout_retains_ownership_until_ack_and_later_requests_progress() -> Result<()> {
    let mut fixture = Fixture::new(true)?;
    fixture.admit(1)?;
    fixture
        .entered
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    fixture.engine.tick(20_001)?;
    assert!(matches!(
        fixture
            .engine
            .pending_terminal(infer_core::RequestId::ONE)
            .map(|done| done.reason),
        Some(FinishReason::Failed(_))
    ));
    assert_eq!(fixture.owned.load(Ordering::Acquire), 1);
    assert_eq!(fixture.engine.inspect().state.sequence_count, 1);
    fixture.engine.check_invariants()?;
    fixture
        .release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    fixture.drain()?;
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    fixture.admit(2)?;
    fixture.drain()?;
    assert_eq!(
        fixture
            .engine
            .request(infer_core::RequestId::new(2)?)?
            .completed
            .as_ref()
            .map(|done| &done.reason),
        Some(&FinishReason::Length)
    );
    Ok(())
}
#[test]
fn reservation_cancel_is_idempotent_and_waits_for_real_ack() -> Result<()> {
    let mut fixture = Fixture::new(true)?;
    fixture.admit(1)?;
    fixture
        .entered
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    assert_eq!(fixture.engine.cancel(infer_core::RequestId::ONE)?, []);
    assert_eq!(fixture.engine.cancel(infer_core::RequestId::ONE)?, []);
    assert_eq!(fixture.owned.load(Ordering::Acquire), 1);
    fixture.engine.check_invariants()?;
    fixture
        .release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    fixture.drain()?;
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    Ok(())
}
#[test]
fn published_but_unconsumed_reservation_is_compensated_on_ticket_drop() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    let ticket = fixture
        .engine
        .backend_mut()
        .begin_resource(ResourceCommand::Reserve {
            state: StateId::ONE,
            capacity: 16,
            readout: OutputReadout::Full,
        })?;
    fixture
        .entered
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    // Sizing is ordered behind reservation publication; observing its reply makes the race deterministic.
    let mut barrier =
        fixture
            .engine
            .backend_mut()
            .begin_resource(ResourceCommand::ReservationBytes {
                capacity: 16,
                readout: OutputReadout::Full,
            })?;
    let limit = Instant::now() + Duration::from_secs(3);
    while barrier.poll()?.is_none() {
        deadline(limit)?;
    }
    assert_eq!(fixture.owned.load(Ordering::Acquire), 1);
    drop(ticket);
    while fixture.owned.load(Ordering::Acquire) > 0
        || fixture.engine.backend().pending_resource_releases()
    {
        deadline(limit)?;
    }
    Ok(())
}

#[test]
fn batch_metadata_is_shared_across_the_device_owner_handoff() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    let mut reserve = fixture
        .engine
        .backend_mut()
        .begin_resource(ResourceCommand::Reserve {
            state: StateId::ONE,
            capacity: 16,
            readout: OutputReadout::Full,
        })?;
    let limit = Instant::now() + Duration::from_secs(3);
    while reserve.poll()?.is_none() {
        deadline(limit)?;
    }
    let program = fixture.engine.program().clone();
    let step = Arc::new(StepPlan {
        id: infer_core::StepId::ONE,
        decision: infer_core::DecisionId::ONE,
        program: program.id,
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
    });
    let mut ticket = fixture.engine.backend_mut().submit_shared(
        &program,
        step.clone(),
        vec![ExecutionTask {
            plain_greedy: false,
            request: infer_core::RequestId::ONE,
            state: StateId::ONE,
            tokens: vec![1, 2, 3].into(),
            sampling: None,
            media: None,
        }],
    )?;
    while fixture.engine.backend_mut().poll(&mut ticket)?.is_none() {
        deadline(limit)?;
    }
    assert_eq!(
        fixture.observed_work.load(Ordering::Acquire),
        step.work.as_ptr() as usize
    );
    fixture.engine.backend_mut().release_state(StateId::ONE)?;
    while fixture.engine.backend().pending_resource_releases() {
        deadline(limit)?;
    }
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn dropped_compute_ticket_waits_for_fence_then_allows_next_submission() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    let mut reserve = fixture
        .engine
        .backend_mut()
        .begin_resource(ResourceCommand::Reserve {
            state: StateId::ONE,
            capacity: 16,
            readout: OutputReadout::Full,
        })?;
    let limit = Instant::now() + Duration::from_secs(3);
    while reserve.poll()?.is_none() {
        deadline(limit)?;
    }
    fixture.poll_gate.store(false, Ordering::Release);
    let program = fixture.engine.program().clone();
    let step = Arc::new(StepPlan {
        id: infer_core::StepId::ONE,
        decision: infer_core::DecisionId::ONE,
        program: program.id,
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
    });
    let tasks = vec![ExecutionTask {
        plain_greedy: false,
        request: infer_core::RequestId::ONE,
        state: StateId::ONE,
        tokens: vec![1, 2, 3].into(),
        sampling: None,
        media: None,
    }];
    let ticket =
        fixture
            .engine
            .backend_mut()
            .submit_shared_borrowed(&program, step.clone(), &tasks)?;
    while fixture.observed_work.load(Ordering::Acquire) == 0 {
        deadline(limit)?;
    }
    drop(ticket);
    assert_eq!(
        fixture
            .engine
            .backend_mut()
            .submit_shared_borrowed(&program, step.clone(), &tasks)
            .err()
            .map(|e| e.code),
        Some(infer_core::ErrorCode::Conflict)
    );
    fixture.poll_gate.store(true, Ordering::Release);
    let mut next = loop {
        match fixture
            .engine
            .backend_mut()
            .submit_shared_borrowed(&program, step.clone(), &tasks)
        {
            Ok(ticket) => break ticket,
            Err(error)
                if matches!(
                    error.code,
                    infer_core::ErrorCode::Capacity | infer_core::ErrorCode::Conflict
                ) =>
            {
                deadline(limit)?;
            }
            Err(error) => return Err(error),
        }
    };
    while fixture.engine.backend_mut().poll(&mut next)?.is_none() {
        deadline(limit)?;
    }
    fixture.engine.backend_mut().release_state(StateId::ONE)?;
    while fixture.engine.backend().pending_resource_releases() {
        deadline(limit)?;
    }
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn independent_ready_work_is_preplanned_and_new_admission_invalidates_it() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    for id in 1..=3 {
        fixture.admit(id)?;
    }
    fixture.poll_gate.store(false, Ordering::Release);
    let limit = Instant::now() + Duration::from_secs(3);
    while !fixture.engine.inspect().cpu.prepared_next {
        fixture.engine.tick(fixture.engine.now_us() + 1)?;
        deadline(limit)?;
    }
    assert!(fixture.engine.inspect().inflight_step.is_some());
    let prepared = fixture.engine.request_preparer()?.prepare(request(4)?)?;
    fixture
        .engine
        .submit_quoted_with_trace(prepared, None, None)?;
    assert!(!fixture.engine.inspect().cpu.prepared_next);
    fixture.engine.cancel(infer_core::RequestId::new(2)?)?;
    fixture.poll_gate.store(true, Ordering::Release);
    fixture.drain()?;
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    assert_eq!(
        fixture
            .engine
            .request(infer_core::RequestId::new(2)?)?
            .completed
            .as_ref()
            .map(|done| &done.reason),
        Some(&FinishReason::Cancelled)
    );
    for id in [1, 3, 4] {
        assert_eq!(
            fixture
                .engine
                .request(infer_core::RequestId::new(id)?)?
                .completed
                .as_ref()
                .map(|done| &done.reason),
            Some(&FinishReason::Length)
        );
    }
    Ok(())
}

#[test]
fn threaded_dispatch_preserves_speculation_and_supplies_decode_sampling() -> Result<()> {
    for depth in [0, 2] {
        let mut fixture = Fixture::with_draft(false, depth)?;
        assert_eq!(
            fixture
                .engine
                .backend()
                .speculation_capability()
                .draft_depth,
            depth
        );
        fixture.admit(1)?;
        fixture.drain()?;
        assert_eq!(
            fixture.observed_sampling.load(Ordering::Acquire) > 0,
            depth > 0,
            "a loaded draft must receive sampling through the threaded service path"
        );
    }
    Ok(())
}

#[test]
fn reservation_capacity_waits_for_progress_and_retries_without_losing_tokens() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    fixture.reservation_limit.store(1, Ordering::Release);
    fixture.poll_gate.store(false, Ordering::Release);
    fixture.admit(1)?;
    fixture
        .entered
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    fixture.admit(2)?;
    let limit = Instant::now() + Duration::from_secs(3);
    while fixture.reservation_attempts.load(Ordering::Acquire) < 2 {
        deadline(limit)?;
    }
    // Consume the failed acknowledgement and stabilize the snapshot. The busy
    // backend must not receive a new reservation on every scheduler poll.
    for _ in 0..100 {
        fixture.engine.tick(fixture.engine.now_us() + 1)?;
        deadline(limit)?;
    }
    let attempts = fixture.reservation_attempts.load(Ordering::Acquire);
    for _ in 0..100 {
        fixture.engine.tick(fixture.engine.now_us() + 1)?;
        deadline(limit)?;
    }
    assert_eq!(
        fixture.reservation_attempts.load(Ordering::Acquire),
        attempts
    );
    assert!(
        fixture
            .engine
            .request(infer_core::RequestId::new(2)?)?
            .completed
            .is_none()
    );
    fixture.engine.check_invariants()?;
    fixture.poll_gate.store(true, Ordering::Release);
    fixture.drain()?;
    for id in [1, 2] {
        let record = fixture.engine.request(infer_core::RequestId::new(id)?)?;
        assert_eq!(
            record.completed.as_ref().map(|done| &done.reason),
            Some(&FinishReason::Length)
        );
        assert_eq!(record.generated.len(), 2);
    }
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn deferred_reservation_can_time_out_or_cancel_without_a_backend_state() -> Result<()> {
    for cancel in [false, true] {
        let mut fixture = Fixture::new(false)?;
        fixture.reservation_limit.store(0, Ordering::Release);
        fixture.admit(1)?;
        let limit = Instant::now() + Duration::from_secs(3);
        for _ in 0..100 {
            fixture.engine.tick(fixture.engine.now_us() + 1)?;
            deadline(limit)?;
        }
        assert_eq!(fixture.reservation_attempts.load(Ordering::Acquire), 1);
        fixture.engine.check_invariants()?;
        if cancel {
            fixture.engine.cancel(infer_core::RequestId::ONE)?;
        } else {
            fixture.engine.tick(20_001)?;
        }
        fixture.drain()?;
        let reason = &fixture
            .engine
            .request(infer_core::RequestId::ONE)?
            .completed
            .as_ref()
            .ok_or_else(|| Error::invariant("deferred reservation did not terminate"))?
            .reason;
        if cancel {
            assert_eq!(reason, &FinishReason::Cancelled);
        } else {
            assert!(matches!(reason, FinishReason::Failed(_)));
        }
        assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    }
    Ok(())
}

/// A direct backend that settles resource commands inline but owns reservation intent, so a
/// busy `Reserve` is always observed on the engine's very first poll instead of a later one.
struct InlineReservationIntent {
    inner: ProtocolBackend,
    attempts: Arc<AtomicUsize>,
}
impl BackendProvider for InlineReservationIntent {
    type Ticket = <ProtocolBackend as BackendProvider>::Ticket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn tracks_reservation_intent(&self) -> bool {
        true
    }
    fn state_reservation_bytes(&self, capacity: usize) -> Result<Option<u64>> {
        self.inner.state_reservation_bytes(capacity)
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn reserve_state_for(
        &mut self,
        _state: StateId,
        _capacity: usize,
        _readout: OutputReadout,
    ) -> Result<()> {
        self.attempts.fetch_add(1, Ordering::AcqRel);
        Err(Error::new(
            infer_core::ErrorCode::Capacity,
            "resident budget busy",
        ))
    }
    fn release_state(&mut self, _state: StateId) -> Result<()> {
        // The rejected reservation never created backend state.
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.inner.submit(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        self.inner.poll(ticket)
    }
}

#[test]
fn inline_busy_reservation_defers_instead_of_failing_admission() -> Result<()> {
    // A direct backend settles `begin_resource` inline, so the busy rejection is always
    // observed on the engine's first poll. When the backend owns reservation intent the
    // rejection must be deferred exactly like a later acknowledgement: keep the request
    // and its host state, and publish no retry until resource ownership progresses.
    let ir = support::model(ModelId::ONE);
    let attempts = Arc::new(AtomicUsize::new(0));
    let backend = InlineReservationIntent {
        inner: ProtocolBackend::new(16, 8, &ir)?,
        attempts: attempts.clone(),
    };
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    let mut engine = Engine::new(
        backend,
        ir,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig {
            max_num_seqs: 1,
            ..RuntimeConfig::default()
        },
    )?;
    let prepared = engine.request_preparer()?.prepare(request(1)?)?;
    let mut quote = engine.admission_quote(&prepared)?;
    let bytes = loop {
        if let Some(ResourceReply::ReservationBytes(bytes)) = quote.poll()? {
            break bytes;
        }
    };
    engine.submit_quoted_with_trace(prepared, bytes, None)?;
    assert_eq!(attempts.load(Ordering::Acquire), 1);
    engine.check_invariants()?;
    for _ in 0..100 {
        engine.tick(engine.now_us() + 1)?;
    }
    assert_eq!(attempts.load(Ordering::Acquire), 1);
    assert!(
        engine
            .request(infer_core::RequestId::ONE)?
            .completed
            .is_none()
    );
    engine.cancel(infer_core::RequestId::ONE)?;
    let limit = Instant::now() + Duration::from_secs(3);
    while !engine.is_idle() {
        engine.tick(engine.now_us() + 1)?;
        deadline(limit)?;
    }
    engine.check_invariants()?;
    Ok(())
}

/// A direct backend that withholds the busy rejection of the second reservation until the
/// scene releases it. This makes the ordering explicit: execution progress happens while the
/// rejection is still in flight, so the engine can observe the rejection only at the later epoch.
struct DelayedBusyReservation {
    inner: ProtocolBackend,
    attempts: Arc<AtomicUsize>,
    states: BTreeSet<StateId>,
    owned: usize,
    poll_gate: bool,
    held: Option<infer_spi::ResourceResponder>,
}
impl DelayedBusyReservation {
    fn new(ir: &ModelIr) -> Result<Self> {
        Ok(Self {
            inner: ProtocolBackend::new(16, 8, ir)?,
            attempts: Arc::new(AtomicUsize::new(0)),
            states: BTreeSet::new(),
            owned: 0,
            poll_gate: true,
            held: None,
        })
    }
    fn release_capacity(&mut self) -> Result<()> {
        self.held
            .take()
            .ok_or_else(|| Error::invariant("no held capacity rejection"))?
            .send(Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "resident budget busy",
            )))
            .map_err(|_| Error::invariant("capacity rejection receiver disappeared"))?;
        Ok(())
    }
}
impl BackendProvider for DelayedBusyReservation {
    type Ticket = <ProtocolBackend as BackendProvider>::Ticket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn tracks_reservation_intent(&self) -> bool {
        true
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn begin_resource(&mut self, command: ResourceCommand) -> Result<ResourceTicket> {
        if matches!(command, ResourceCommand::Reserve { .. }) {
            let attempt = self.attempts.fetch_add(1, Ordering::AcqRel) + 1;
            if attempt == 2 {
                let (ticket, reply) = ResourceTicket::channel();
                self.held = Some(reply);
                return Ok(ticket);
            }
        }
        let (ticket, reply) = ResourceTicket::channel();
        let result = infer_spi::execute_resource(self, command);
        let _ = reply.send(result);
        Ok(ticket)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        if self.owned > 0 {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "resident budget busy",
            ));
        }
        self.inner.reserve_state_for(state, capacity, readout)?;
        self.states.insert(state);
        self.owned += 1;
        Ok(())
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        // A rejected reservation never created backend state; cancellation still
        // releases the host state that was reserved for it.
        if self.states.remove(&state) {
            self.inner.release_state(state)?;
            self.owned = self.owned.saturating_sub(1);
        }
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.inner.submit(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        if self.poll_gate {
            self.inner.poll(ticket)
        } else {
            Ok(None)
        }
    }
}

#[test]
fn busy_rejection_observed_after_execution_progress_is_retried_once() -> Result<()> {
    let ir = support::model(ModelId::ONE);
    let backend = DelayedBusyReservation::new(&ir)?;
    let attempts = backend.attempts.clone();
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    let mut engine = Engine::new(
        backend,
        ir,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig {
            max_num_seqs: 1,
            ..RuntimeConfig::default()
        },
    )?;
    let mut first = request(1)?;
    first.workload = serde_json::from_value(serde_json::json!({
        "Generate": {"max_new_tokens":120}
    }))
    .map_err(|e| Error::invalid(e.to_string()))?;
    engine.submit(first)?;
    engine.submit(request(2)?)?;
    assert_eq!(attempts.load(Ordering::Acquire), 2);
    let limit = Instant::now() + Duration::from_secs(3);
    // Complete one execution step while request 2's busy rejection is held in flight.
    let generated = engine.request(infer_core::RequestId::ONE)?.generated.len();
    while engine.request(infer_core::RequestId::ONE)?.generated.len() == generated {
        engine.tick(engine.now_us() + 1)?;
        deadline(limit)?;
    }
    let before = attempts.load(Ordering::Acquire);
    // No further execution progress may arrive after this point.
    engine.backend_mut().poll_gate = false;
    engine.backend_mut().release_capacity()?;
    // The rejected attempt started before that progress, so it must probe once more.
    while attempts.load(Ordering::Acquire) == before {
        engine.tick(engine.now_us() + 1)?;
        deadline(limit)?;
    }
    assert!(
        engine
            .request(infer_core::RequestId::new(2)?)?
            .completed
            .is_none()
    );
    engine.backend_mut().poll_gate = true;
    engine.cancel(infer_core::RequestId::ONE)?;
    engine.cancel(infer_core::RequestId::new(2)?)?;
    while !engine.is_idle() {
        engine.tick(engine.now_us() + 1)?;
        deadline(limit)?;
    }
    engine.check_invariants()
}

#[test]
fn runtime_applies_backend_recommendation_without_cli() -> Result<()> {
    let ir = support::model(ModelId::ONE);
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels)?;
    let backend = ProtocolBackend::new(16, 8, &ir)?.with_recommendation(3);
    let engine = Engine::new(
        backend,
        ir,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )?;
    assert_eq!(engine.config().max_num_seqs, 3);
    Ok(())
}

#[test]
fn runtime_preserves_explicit_max_num_seqs() -> Result<()> {
    let ir = support::model(ModelId::ONE);
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels)?;
    let backend = ProtocolBackend::new(16, 8, &ir)?.with_recommendation(3);
    let config = RuntimeConfig {
        max_num_seqs: 7,
        max_num_seqs_explicit: true,
        ..RuntimeConfig::default()
    };
    let engine = Engine::new(backend, ir, PrecisionPlan::f32(), &kernels, config)?;
    assert_eq!(engine.config().max_num_seqs, 7);
    Ok(())
}

#[test]
fn media_request_requires_a_multimodal_backend() -> Result<()> {
    use infer_ir::{MediaInput, MediaItem, Modality};
    use std::sync::Arc;

    let ir = support::model(ModelId::ONE);
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels)?;
    let mut engine = Engine::new(
        ProtocolBackend::new(16, 8, &ir)?,
        ir,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )?;
    let media = MediaInput {
        id: infer_core::MediaId::ONE,
        modality: Modality::Image,
        content_digest: "fixture".into(),
        item: Some(MediaItem {
            modality: Modality::Image,
            pixels: Arc::from([0.0f32, 1.0, 2.0, 3.0]),
            grid: (1, 1, 1),
            span: 0..1,
        }),
    };
    let request = CanonicalRequest {
        id: infer_core::RequestId::new(7)?,
        model: ModelId::ONE,
        session: None,
        input: infer_ir::RequestInput::Sequence {
            tokens: vec![1, 2].into(),
            media: vec![media],
        },
        workload: infer_ir::Workload::Generate { max_new_tokens: 1 },
        qos: infer_ir::Qos::default(),
        sampling: infer_ir::Sampling::default(),
        extensions: BTreeMap::new(),
    };
    let error = engine.submit(request).unwrap_err();
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
    Ok(())
}

#[test]
fn prepared_media_reaches_the_scheduler_and_backend() -> Result<()> {
    use infer_ir::{MediaInput, MediaItem, Modality};
    use std::sync::Arc;

    let ir = support::model(ModelId::ONE);
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels)?;
    let backend = ProtocolBackend::new(16, 8, &ir)?.multimodal();
    let media_tasks = backend.media_tasks();
    let mut engine = Engine::new(
        backend,
        ir,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )?;
    let media = MediaInput {
        id: infer_core::MediaId::ONE,
        modality: Modality::Image,
        content_digest: "fixture".into(),
        item: Some(MediaItem {
            modality: Modality::Image,
            pixels: Arc::from([0.0f32, 1.0, 2.0, 3.0]),
            grid: (1, 1, 1),
            span: 0..1,
        }),
    };
    let request = CanonicalRequest {
        id: infer_core::RequestId::new(8)?,
        model: ModelId::ONE,
        session: None,
        input: infer_ir::RequestInput::Sequence {
            tokens: vec![1, 2].into(),
            media: vec![media],
        },
        workload: infer_ir::Workload::Generate { max_new_tokens: 1 },
        qos: infer_ir::Qos::default(),
        sampling: infer_ir::Sampling::default(),
        extensions: BTreeMap::new(),
    };
    engine.submit(request)?;
    for now in 0..64 {
        if engine.is_idle() {
            break;
        }
        engine.tick(now)?;
    }
    assert!(engine.is_idle());
    assert!(media_tasks.load(Ordering::Relaxed) > 0);
    Ok(())
}

#[test]
fn automatic_mtp_is_disabled_for_media_instead_of_rejecting_it() -> Result<()> {
    use infer_ir::{MediaInput, MediaItem, Modality};
    use std::sync::Arc;

    let ir = support::model(ModelId::ONE);
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels)?;
    let backend = ProtocolBackend::new(16, 8, &ir)?
        .multimodal()
        .requiring_engine_sampling(2);
    let media_reservations = backend.media_reservations();
    let media_quotes = backend.media_quotes();
    let media_tasks = backend.media_tasks();
    let mut engine = Engine::new(
        backend,
        ir,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )?;
    let media = MediaInput {
        id: infer_core::MediaId::ONE,
        modality: Modality::Image,
        content_digest: "fixture".into(),
        item: Some(MediaItem {
            modality: Modality::Image,
            pixels: Arc::from([0.0f32, 1.0, 2.0, 3.0]),
            grid: (1, 1, 1),
            span: 0..1,
        }),
    };
    let request = CanonicalRequest {
        id: infer_core::RequestId::new(9)?,
        model: ModelId::ONE,
        session: None,
        input: infer_ir::RequestInput::Sequence {
            tokens: vec![1, 2].into(),
            media: vec![media],
        },
        workload: infer_ir::Workload::Generate { max_new_tokens: 1 },
        qos: infer_ir::Qos::default(),
        sampling: infer_ir::Sampling::default(),
        extensions: BTreeMap::new(),
    };
    engine.submit(request)?;
    for now in 0..64 {
        if engine.is_idle() {
            break;
        }
        engine.tick(now)?;
    }
    assert!(engine.is_idle());
    assert_eq!(media_reservations.load(Ordering::Relaxed), 1);
    assert!(media_quotes.load(Ordering::Relaxed) > 0);
    assert!(media_tasks.load(Ordering::Relaxed) > 0);
    Ok(())
}

#[test]
fn threaded_media_request_uses_the_media_quote_command() -> Result<()> {
    use infer_ir::{MediaInput, MediaItem, Modality};
    use std::sync::Arc;
    use std::time::Duration;

    // Failure-first regression: before the typed media quote command, the threaded admission
    // path always sent ReservationBytes and priced draft state; this counter stayed zero.
    let ir = support::model(ModelId::ONE);
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels)?;
    let backend = ProtocolBackend::new(16, 8, &ir)?
        .multimodal()
        .requiring_engine_sampling(2)
        .priced_quotes(1024, 4096);
    let media_quotes = backend.media_quotes();
    let normal_quotes = backend.normal_quotes();
    let mut engine = Engine::new(
        backend,
        ir,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )?
    .into_threaded(4, Duration::from_secs(5))?;
    let media = MediaInput {
        id: infer_core::MediaId::ONE,
        modality: Modality::Image,
        content_digest: "fixture".into(),
        item: Some(MediaItem {
            modality: Modality::Image,
            pixels: Arc::from([0.0f32, 1.0, 2.0, 3.0]),
            grid: (1, 1, 1),
            span: 0..1,
        }),
    };
    let media_request = CanonicalRequest {
        id: infer_core::RequestId::new(10)?,
        model: ModelId::ONE,
        session: None,
        input: infer_ir::RequestInput::Sequence {
            tokens: vec![1, 2].into(),
            media: vec![media],
        },
        workload: infer_ir::Workload::Generate { max_new_tokens: 1 },
        qos: infer_ir::Qos::default(),
        sampling: infer_ir::Sampling::default(),
        extensions: BTreeMap::new(),
    };
    // The normal path must still quote its own price; only the media request uses 4096.
    let text = request(11)?;
    let prepared_text = engine.request_preparer()?.prepare(text)?;
    let mut text_quote = engine.admission_quote(&prepared_text)?;
    let text_bytes = poll_quote(&mut text_quote)?;
    assert_eq!(text_bytes, Some(1024));
    assert!(normal_quotes.load(Ordering::Acquire) > 0);
    assert_eq!(media_quotes.load(Ordering::Acquire), 0);
    drop(text_quote);

    let prepared = engine.request_preparer()?.prepare(media_request)?;
    let mut quote = engine.admission_quote(&prepared)?;
    let bytes = poll_quote(&mut quote)?;
    assert_eq!(bytes, Some(4096), "media quote must use the media price");
    assert!(media_quotes.load(Ordering::Acquire) > 0);
    engine.submit_quoted_with_trace(prepared, bytes, None)?;
    let limit = Instant::now() + Duration::from_secs(3);
    while !engine.is_idle() {
        engine.tick(engine.now_us() + 1)?;
        if Instant::now() >= limit {
            return Err(Error::invariant("threaded media request did not drain"));
        }
        sleep(Duration::from_millis(1));
    }
    assert!(
        media_quotes.load(Ordering::Acquire) > 0,
        "threaded media admission did not use the media quote command"
    );
    Ok(())
}
