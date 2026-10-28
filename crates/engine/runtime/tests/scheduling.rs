mod support;

use infer_kernel_api::KernelRegistry;
use infer_spi::{AdmissionPolicy, BackendProvider, SchedulingPolicy};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::slice::from_ref;
use support::ProtocolBackend;

use infer_core::*;
use infer_ir::*;
use infer_runtime::*;

#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn registry() -> KernelRegistry {
    let mut r = KernelRegistry::default();
    r.register(&support::DeclaredKernels).unwrap();
    r
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn backend() -> ProtocolBackend {
    ProtocolBackend::tagged("scheduling-weights", 16, 8, &support::model(ModelId::ONE)).unwrap()
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn engine(config: RuntimeConfig) -> Engine<ProtocolBackend> {
    let b = backend();
    let m = support::model(ModelId::ONE);
    Engine::new(b, m, PrecisionPlan::f32(), &registry(), config).unwrap()
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn request(id: u64, input: usize) -> CanonicalRequest {
    CanonicalRequest {
        id: RequestId::new(id).unwrap(),
        model: ModelId::ONE,
        session: None,
        input: RequestInput::Sequence {
            tokens: vec![3; input].into(),
            media: vec![],
        },
        workload: Workload::Generate { max_new_tokens: 8 },
        qos: Qos::default(),
        sampling: Sampling::default(),
        prefix_boundary: None,
        extensions: BTreeMap::new(),
    }
}
#[expect(
    clippy::panic,
    reason = "This test helper rejects unexpected output variants so an incorrect fixture cannot pass silently"
)]
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn finish<B: BackendProvider, P: SchedulingPolicy>(e: &mut Engine<B, P>, start: u64) {
    for tick in start..start + 1000 {
        e.tick(tick).unwrap();
        e.check_invariants().unwrap();
        if e.is_idle() {
            return;
        }
    }
    panic!("scheduler did not finish");
}

#[test]
fn admission_quota_is_transactional_and_running_cancel_holds_reservation() {
    let mut config = RuntimeConfig::default();
    config.admission.default_tenant.max_active_requests = 1;
    let mut e = engine(config);
    e.submit(request(1, 3)).unwrap();
    let pages = e.inspect().state.allocated_pages;
    assert!(
        e.submit(request(2, 3))
            .unwrap_err()
            .message
            .contains("TenantRequests")
    );
    assert!(e.request(RequestId::new(2).unwrap()).is_err());
    assert_eq!(e.inspect().state.allocated_pages, pages);
    e.cancel(RequestId::new(1).unwrap()).unwrap();
    assert_eq!(e.inspect().state.allocated_pages, 0);
    e.submit(request(2, 3)).unwrap();
    e.tick(0).unwrap();
    let running_pages = e.inspect().state.allocated_pages;
    assert!(running_pages > 0);
    e.cancel(RequestId::new(2).unwrap()).unwrap();
    assert!(e.submit(request(3, 3)).is_err());
    assert_eq!(e.inspect().state.allocated_pages, running_pages);
    e.tick(1).unwrap();
    e.submit(request(3, 3)).unwrap();
    finish(&mut e, 2);
    assert_eq!(e.inspect().state.allocated_pages, 0);
}

#[test]
fn infeasible_ttft_rejection_does_not_consume_id_or_state() {
    let mut config = RuntimeConfig::default();
    config.admission.reject_infeasible_slo = true;
    config.cost_per_token_us = 100;
    let mut e = engine(config);
    let mut r = request(1, 10);
    r.qos.ttft_slo_us = Some(50);
    assert!(
        e.submit(r.clone())
            .unwrap_err()
            .message
            .contains("SloInfeasible")
    );
    assert_eq!(e.inspect().active_requests, 0);
    assert_eq!(e.inspect().state.allocated_pages, 0);
    assert_eq!(e.journal().unwrap(), [] as [ReplayAction; 0]);
    r.qos.ttft_slo_us = Some(2000);
    e.submit(r).unwrap();
    assert!(
        !e.request(RequestId::new(1).unwrap())
            .unwrap()
            .admission
            .slo_at_risk
    );
    finish(&mut e, 0);
}
#[test]
fn strict_admission_also_checks_decode_tpot_feasibility() {
    let mut config = RuntimeConfig::default();
    config.admission.reject_infeasible_slo = true;
    config.cost_per_token_us = 100;
    let mut e = engine(config);
    let mut r = request(1, 3);
    r.qos.tpot_slo_us = Some(10);
    assert!(
        e.submit(r.clone())
            .unwrap_err()
            .message
            .contains("SloInfeasible")
    );
    assert_eq!(e.inspect().state.allocated_pages, 0);
    r.qos.tpot_slo_us = Some(200);
    e.submit(r).unwrap();
    finish(&mut e, 0);
}

struct CustomAdmission(infer_scheduler::ResourceAdmission);
impl AdmissionPolicy for CustomAdmission {
    fn identity(&self) -> &'static str {
        "test-custom-admission-v1"
    }
    fn check(&self, input: &AdmissionInput<'_>) -> Result<AdmissionDecision> {
        self.0.check(input)
    }
}
#[test]
fn admission_provider_is_installed_before_requests_and_bound_to_checkpoint() {
    let mut e = engine(RuntimeConfig::default())
        .with_admission_policy(CustomAdmission(
            infer_scheduler::ResourceAdmission::new(AdmissionConfig::default()).unwrap(),
        ))
        .unwrap();
    e.submit(request(1, 3)).unwrap();
    let snapshot = e.snapshot().unwrap();
    assert_eq!(
        snapshot.admission_provider_identity,
        "test-custom-admission-v1"
    );
    assert!(Engine::restore(backend(), &registry(), snapshot.clone()).is_err());
    let b = backend();
    let costs = infer_scheduler::CalibratedCosts::new(
        format!("{}:{}", b.identity(), snapshot.program.id),
        snapshot.config.cost_model.clone(),
    )
    .unwrap();
    let mut restored = Engine::restore_with_planning(
        b,
        &registry(),
        snapshot,
        infer_workloads::NativeWorkloads,
        costs,
        CustomAdmission(
            infer_scheduler::ResourceAdmission::new(AdmissionConfig::default()).unwrap(),
        ),
    )
    .unwrap();
    finish(&mut restored, 0);
    assert!(
        restored
            .with_admission_policy(CustomAdmission(
                infer_scheduler::ResourceAdmission::new(AdmissionConfig::default()).unwrap()
            ))
            .is_err()
    );
}

#[test]
fn mixed_decode_prefill_executes_without_changing_generated_tokens() {
    let mut config = RuntimeConfig {
        max_num_batched_tokens: 4,
        ..Default::default()
    };
    config.scheduler.prefill_chunk_tokens = 2;
    let mut e = engine(config);
    e.submit(request(1, 3)).unwrap();
    e.tick(0).unwrap();
    e.tick(1).unwrap();
    e.submit(request(2, 25)).unwrap();
    e.tick(2).unwrap();
    let step = e.decisions().back().unwrap().step.as_ref().unwrap();
    assert_eq!(step.role, ExecutionRole::Mixed);
    assert!(
        step.work
            .iter()
            .any(|w| w.request.get() == 1 && w.role == ExecutionRole::Decode)
    );
    assert!(
        step.work.iter().any(|w| w.request.get() == 2
            && w.role == ExecutionRole::Prefill
            && w.token_count == 2)
    );
    finish(&mut e, 3);
    let mut baseline = engine(RuntimeConfig {
        max_num_seqs: 1,
        ..Default::default()
    });
    baseline.submit(request(1, 3)).unwrap();
    baseline.submit(request(2, 25)).unwrap();
    finish(&mut baseline, 0);
    for id in [1, 2] {
        let id = RequestId::new(id).unwrap();
        assert_eq!(
            e.request(id).unwrap().completed.as_ref().unwrap().output,
            baseline
                .request(id)
                .unwrap()
                .completed
                .as_ref()
                .unwrap()
                .output
        );
    }
    assert_eq!(e.inspect().state.allocated_pages, 0);
}

#[test]
fn tpot_target_tracks_last_token_instead_of_acceptance_time() {
    let mut e = engine(RuntimeConfig::default());
    let mut r = request(1, 3);
    r.qos.ttft_slo_us = Some(100);
    r.qos.tpot_slo_us = Some(5);
    e.submit(r).unwrap();
    e.tick(0).unwrap();
    let (_, snapshot) = e.quiesce(10).unwrap();
    assert!(snapshot.is_some());
    e.submit(request(2, 30)).unwrap();
    e.tick(11).unwrap();
    let evidence = e
        .decisions()
        .back()
        .unwrap()
        .selected
        .iter()
        .find(|s| s.request.get() == 1)
        .unwrap();
    assert_eq!(evidence.role, ExecutionRole::Decode);
    assert_eq!(evidence.target_us, Some(15));
    assert_eq!(evidence.slack_us, Some(3));
    assert_eq!(evidence.reason, SelectionReason::UrgentSlo);
    finish(&mut e, 12);
}

#[test]
fn singleton_larger_than_quantum_makes_progress_with_explicit_overrun() {
    let mut e = engine(RuntimeConfig {
        gpu_budget_us: 1,
        cost_per_token_us: 100,
        ..Default::default()
    });
    e.submit(request(1, 3)).unwrap();
    finish(&mut e, 0);
    assert!(
        e.decisions()
            .iter()
            .filter_map(|d| d.step.as_ref())
            .all(|s| s.quantum_overrun && s.work.len() == 1 && s.work[0].token_count == 1)
    );
    assert!(
        e.request(RequestId::new(1).unwrap())
            .unwrap()
            .completed
            .as_ref()
            .unwrap()
            .measurement
            .successful
    );
}
#[test]
fn logical_page_pressure_recomputes_and_replays_without_losing_generated_tokens() {
    let config = RuntimeConfig {
        num_gpu_blocks: 3,
        block_size: 4,
        max_num_batched_tokens: 2,
        max_num_seqs: 2,
        ..Default::default()
    };
    let mut e = engine(config.clone());
    e.submit(request(1, 3)).unwrap();
    e.submit(request(2, 3)).unwrap();
    assert_eq!(e.inspect().state.allocated_pages, 0);
    finish(&mut e, 0);
    assert!(e.inspect().preemptions > 0);
    assert_eq!(e.inspect().state.allocated_pages, 0);
    let mut baseline = engine(RuntimeConfig::default());
    baseline.submit(request(1, 3)).unwrap();
    baseline.submit(request(2, 3)).unwrap();
    finish(&mut baseline, 0);
    for id in [1, 2] {
        let id = RequestId::new(id).unwrap();
        assert_eq!(
            e.request(id).unwrap().completed.as_ref().unwrap().output,
            baseline
                .request(id)
                .unwrap()
                .completed
                .as_ref()
                .unwrap()
                .output
        );
    }
    let journal = e.journal().unwrap();
    let mut replay = engine(config);
    replay.replay(&journal).unwrap();
    assert_eq!(e.decisions(), replay.decisions());
    assert_eq!(e.inspect().preemptions, replay.inspect().preemptions);
    assert!(e.snapshot().is_ok());
}

struct ForgedPolicy {
    reject: bool,
}
impl SchedulingPolicy for ForgedPolicy {
    type Workspace = ();
    #[expect(
        clippy::unwrap_used,
        reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
    )]
    fn plan(
        &self,
        ready: &[ReadyWork],
        resources: &ResourceSnapshot,
        now: u64,
        decision: DecisionId,
        step: StepId,
    ) -> Result<SchedulingDecision> {
        if self.reject {
            return Ok(SchedulingDecision {
                window: None,
                id: decision,
                step: None,
                deferred: vec![DeferredWork {
                    request: ready[0].request,
                    reason: DeferReason::AtomicCostLimit,
                    required: u64::MAX,
                    available: 1,
                }],
                selected: vec![],
            });
        }
        let mut d = infer_scheduler::CostAwarePolicy.plan(ready, resources, now, decision, step)?;
        d.step.as_mut().unwrap().cost.gpu_us = 0;
        Ok(d)
    }
}
#[test]
fn provider_cannot_underreport_cost_or_forge_permanent_rejection() {
    for reject in [false, true] {
        let b = backend();
        let m = support::model(ModelId::ONE);
        let mut e = Engine::with_policy(
            b,
            m,
            PrecisionPlan::f32(),
            &registry(),
            RuntimeConfig::default(),
            ForgedPolicy { reject },
        )
        .unwrap();
        e.submit(request(1, 3)).unwrap();
        let pages = e.inspect().state.allocated_pages;
        assert_eq!(e.tick(0).unwrap_err().code, ErrorCode::Invariant);
        assert_eq!(
            e.request(RequestId::new(1).unwrap()).unwrap().status,
            RequestStatus::Runnable
        );
        assert_eq!(e.inspect().state.allocated_pages, pages);
    }
}

// Controlled timing fixtures isolate replay from machine timing variability.
// Real Host and Metal measurements are covered by their backend integration tests.
struct TimedBackend {
    inner: ProtocolBackend,
    elapsed: u64,
}
struct TimedTicket {
    inner: <ProtocolBackend as BackendProvider>::Ticket,
    done: bool,
}
impl BackendProvider for TimedBackend {
    type Ticket = TimedTicket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn supports_control_checkpoint(&self) -> bool {
        true
    }
    fn reserve_state(&mut self, state: StateId, capacity: usize) -> Result<()> {
        self.inner.reserve_state(state, capacity)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.inner.reserve_state_for(state, capacity, readout)
    }
    fn recycle_output(&mut self, state: StateId, output: ModelOutput) -> Result<()> {
        self.inner.recycle_output(state, output)
    }
    fn recycle_batch(&mut self, outputs: Vec<TaskOutput>) -> Result<()> {
        self.inner.recycle_batch(outputs)
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.inner.reset_state(state)
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)
    }
    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        self.inner.capture_execution_state()
    }
    fn restore_execution_state(&mut self, state: Option<&[u8]>) -> Result<()> {
        self.inner.restore_execution_state(state)
    }
    fn validate_state_ownership(&self, states: &[(StateId, usize, usize)]) -> Result<()> {
        self.inner.validate_state_ownership(states)
    }
    fn validate_program(&self, m: &ModelIr, p: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(m, p)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn submit(
        &mut self,
        p: &ExecutionProgram,
        s: &StepPlan,
        t: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        Ok(TimedTicket {
            inner: self.inner.submit(p, s, t)?,
            done: false,
        })
    }
    fn poll(&mut self, t: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        let result = self.inner.poll(&mut t.inner)?;
        t.done = result.is_some();
        Ok(result)
    }
    fn completion_timing(&self, t: &Self::Ticket) -> Option<ExecutionTiming> {
        t.done.then_some(ExecutionTiming {
            elapsed_us: self.elapsed,
            // The source has to match the declared backend kind or the engine discards the sample.
            source: TimingSource::CudaGpu,
        })
    }
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn timed_engine(config: RuntimeConfig, elapsed: u64) -> Engine<TimedBackend> {
    let b = backend();
    let m = support::model(ModelId::ONE);
    Engine::new(
        TimedBackend { inner: b, elapsed },
        m,
        PrecisionPlan::f32(),
        &registry(),
        config,
    )
    .unwrap()
}
#[test]
fn measured_cost_feedback_is_replayed_and_checkpointed_deterministically() {
    let config = RuntimeConfig {
        max_num_batched_tokens: 4,
        gpu_budget_us: 200,
        ..Default::default()
    };
    let mut original = timed_engine(config.clone(), 800);
    original.submit(request(1, 3)).unwrap();
    original.submit(request(2, 25)).unwrap();
    original.tick(0).unwrap();
    let (_, checkpoint) = original.quiesce(1).unwrap();
    let checkpoint = checkpoint.unwrap();
    assert_eq!(checkpoint.pending_cost_observations.len(), 1);
    let mut restored = Engine::restore(
        TimedBackend {
            inner: backend(),
            elapsed: 800,
        },
        &registry(),
        checkpoint,
    )
    .unwrap();
    finish(&mut original, 2);
    finish(&mut restored, 2);
    assert_eq!(original.decisions(), restored.decisions());
    assert_eq!(original.inspect().cost_model, restored.inspect().cost_model);
    assert_eq!(
        original.inspect().cost_model.last_source,
        Some(TimingSource::CudaGpu)
    );
    assert!(original.inspect().cost_model.observations > 0);
    let journal = original.journal().unwrap();
    assert!(
        journal
            .iter()
            .any(|a| matches!(a, ReplayAction::CostFeedback(_)))
    );
    let mut replay = timed_engine(config, 8_000_000);
    replay.replay(&journal).unwrap();
    assert_eq!(original.decisions(), replay.decisions());
    assert_eq!(original.inspect().cost_model, replay.inspect().cost_model);
    for id in [1, 2] {
        let id = RequestId::new(id).unwrap();
        assert_eq!(
            original.request(id).unwrap().completed,
            replay.request(id).unwrap().completed
        );
        assert_eq!(
            original.request(id).unwrap().completed,
            restored.request(id).unwrap().completed
        );
    }
}

#[test]
fn memory_blocked_decode_preempts_even_when_another_request_is_feasible() {
    let mut config = RuntimeConfig {
        num_gpu_blocks: 3,
        block_size: 4,
        max_num_seqs: 1,
        max_num_batched_tokens: 6,
        ..Default::default()
    };
    config.scheduler.prefill_chunk_tokens = 6;
    config.scheduler.max_wait_us = 4;
    let mut e = engine(config);
    let mut first = request(1, 4);
    first.workload = Workload::Generate { max_new_tokens: 2 };
    first.qos.tenant = "decode".into();
    let mut second = request(2, 8);
    second.workload = Workload::Generate { max_new_tokens: 2 };
    second.qos.tenant = "prefill".into();
    second.qos.weight = 10;
    e.submit(first.clone()).unwrap();
    e.tick(0).unwrap();
    e.quiesce(1).unwrap();
    assert_eq!(e.request(first.id).unwrap().generated.len(), 1);
    e.submit(second.clone()).unwrap();
    // The higher-weight newcomer receives six prefill tokens and owns two pages.
    // Its final two tokens fit there, but the aged decode needs one more page.
    e.tick(2).unwrap();
    e.quiesce(3).unwrap();
    assert_eq!(e.inspect().state.free_pages, 0);
    assert_eq!(e.request(second.id).unwrap().prefill_done, 6);
    assert_eq!(e.inspect().preemptions, 0);
    e.tick(4).unwrap();
    assert_eq!(e.request(second.id).unwrap().preemptions, 1);
    assert!(
        e.decisions()
            .back()
            .unwrap()
            .deferred
            .iter()
            .any(|deferred| deferred.request == second.id
                && deferred.reason == DeferReason::PreemptionFocus { request: first.id })
    );
    assert!(matches!(
        e.request(first.id).unwrap().status,
        RequestStatus::Running { .. }
    ));
    finish(&mut e, 5);
    assert_eq!(e.inspect().state.allocated_pages, 0);
    let mut baseline = engine(RuntimeConfig::default());
    baseline.submit(first.clone()).unwrap();
    baseline.submit(second.clone()).unwrap();
    finish(&mut baseline, 0);
    for id in [first.id, second.id] {
        assert_eq!(
            e.request(id).unwrap().completed.as_ref().unwrap().output,
            baseline
                .request(id)
                .unwrap()
                .completed
                .as_ref()
                .unwrap()
                .output
        );
    }
}

#[test]
fn bounded_runtime_window_uses_remaining_ttft_service() -> Result<()> {
    let mut config = RuntimeConfig {
        candidate_limit: 1,
        max_num_batched_tokens: 1,
        cost_per_token_us: 250,
        ..RuntimeConfig::default()
    };
    config.scheduler.prefill_chunk_tokens = 1;
    let mut e = engine(config);
    let mut short = request(1, 2);
    short.qos.ttft_slo_us = Some(20_000);
    let mut long = request(2, 100);
    long.qos.ttft_slo_us = Some(30_000);
    e.submit(short)?;
    e.submit(long)?;
    e.tick(0)?;
    let selected = e
        .decisions()
        .back()
        .and_then(|decision| decision.selected.first())
        .ok_or_else(|| Error::invariant("missing first selection"))?;
    assert_eq!(selected.request, RequestId::new(2)?);
    e.check_invariants()
}

struct PressureBackend {
    inner: ProtocolBackend,
    blocked: BTreeSet<StateId>,
}
impl BackendProvider for PressureBackend {
    type Ticket = <ProtocolBackend as BackendProvider>::Ticket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        tokens: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.inner.reserve_state_for(state, tokens, readout)
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.inner.submit(program, step, tasks)
    }
    fn submit_borrowed(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<Self::Ticket> {
        self.inner.submit_borrowed(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        self.inner.poll(ticket)
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        Ok((!self.blocked.is_empty()).then_some(0))
    }
    fn state_page_growth(&self, state: StateId) -> Result<Option<PageGrowth>> {
        Ok(self.blocked.contains(&state).then_some(PageGrowth {
            block_size: 1,
            allocated_pages: usize::MAX,
            bytes_per_page: 1,
            cow_tail: true,
        }))
    }
}

#[test]
fn all_blocked_priority_candidates_do_not_hide_a_feasible_request() -> Result<()> {
    let b = PressureBackend {
        inner: backend(),
        blocked: BTreeSet::new(),
    };
    let mut e = Engine::new(
        b,
        support::model(ModelId::ONE),
        PrecisionPlan::f32(),
        &registry(),
        RuntimeConfig {
            candidate_limit: 2,
            ..RuntimeConfig::default()
        },
    )?;
    for id in 1..=3 {
        e.submit(request(id, 3))?;
    }
    for id in 1..=2 {
        let state = e
            .request(RequestId::new(id)?)?
            .state
            .ok_or_else(|| Error::invariant("missing request state"))?;
        e.backend_mut().blocked.insert(state);
    }
    e.tick(0)?;
    assert!(
        e.decisions()
            .back()
            .is_some_and(|decision| decision.step.is_none())
    );
    e.tick(1)?;
    assert!(e.decisions().back().is_some_and(|decision| {
        decision
            .selected
            .iter()
            .any(|selected| selected.request.get() == 3)
    }));
    assert!(e.inspect().inflight_step.is_some());
    e.check_invariants()
}

#[test]
fn continuous_calibration_covers_the_queue_and_resumes_its_checkpoint_cursor() -> Result<()> {
    let mut config = RuntimeConfig::default();
    config.cpu.maintenance_items = 1;
    let mut original = engine(config);
    for id in 1..=4 {
        let mut r = request(id, 16);
        r.qos.ttft_slo_us = Some(100_000);
        original.submit(r)?;
    }
    let snapshot = original.snapshot()?;
    let query = CostQuery::from_unit(
        snapshot.program.id,
        snapshot.program.backend,
        ExecutionRole::Prefill,
        1,
        1,
        CostEstimate {
            gpu_us: snapshot.config.cost_per_token_us,
            workspace_bytes: snapshot.program.workspace_bytes,
            ..Default::default()
        },
    );
    let feedback = ReplayAction::CostFeedback(CostObservation {
        step: StepId::ONE,
        work: vec![query].into(),
        timing: ExecutionTiming {
            elapsed_us: 100,
            source: TimingSource::CudaGpu,
        },
    });
    original.replay(from_ref(&feedback))?;
    original.poll_completed(1)?;
    let checkpoint = original.snapshot()?;
    assert_eq!(checkpoint.priority_cursor, Some(RequestId::ONE));
    let mut restored = Engine::restore(backend(), &registry(), checkpoint.clone())?;
    for tick in 2..=4 {
        for e in [&mut original, &mut restored] {
            e.replay(from_ref(&feedback))?;
            e.poll_completed(tick)?;
        }
        let a = original.snapshot()?;
        let b = restored.snapshot()?;
        assert_eq!(a.priority_cursor, Some(RequestId::new(tick)?));
        assert_eq!(a.priority_cursor, b.priority_cursor);
        assert_eq!(a.priority_sweep_epoch, 1);
        assert_eq!(a.queues, b.queues);
    }
    original.poll_completed(5)?;
    let finished = original.snapshot()?;
    assert_eq!(finished.priority_cursor, None);
    assert_eq!(finished.priority_epoch, 1);
    assert_eq!(finished.cost_epoch, 4);
    original.poll_completed(6)?;
    assert_eq!(original.snapshot()?.priority_cursor, Some(RequestId::ONE));
    let mut corrupted = checkpoint;
    corrupted.priority_sweep_epoch = corrupted.cost_epoch + 1;
    assert!(Engine::restore(backend(), &registry(), corrupted).is_err());
    Ok(())
}

#[test]
fn cold_backend_concurrency_reaches_engine_before_resources_are_sized() {
    let ir = support::model(ModelId::ONE);
    let profile: ExecutionProfileInspection = serde_json::from_value(serde_json::json!({
        "prefill_width":64,"batch_width":1,"draft_depth":0,"automatic_prefill":true,
        "arena_budget_bytes":1,"concurrency":{"memory_bound":48,"roofline_bound":32,"selected":32}
    }))
    .unwrap();
    let backend = ProtocolBackend::new(64, 32, &ir)
        .unwrap()
        .with_profile(profile);
    let mut e = Engine::new(
        backend,
        ir,
        PrecisionPlan::f32(),
        &registry(),
        RuntimeConfig::default(),
    )
    .unwrap();
    assert_eq!(e.config().max_num_seqs, 32);
    for id in 1..=32 {
        e.submit(request(id, 1)).unwrap();
    }
    finish(&mut e, 0);
    assert_eq!(e.inspect().active_requests, 0);
    e.check_invariants().unwrap();
}

#[test]
fn a_prefix_boundary_ends_one_prompt_chunk_on_a_page_and_asks_to_keep_it() {
    let backend = backend();
    let published = backend.published_prefixes();
    let model = support::model(ModelId::ONE);
    let mut e = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry(),
        RuntimeConfig::default(),
    )
    .unwrap();
    let page = e.config().block_size;
    for (boundary, valid) in [(0, false), (100, false), (2 * page + 3, true)] {
        let mut shared = request(1, 100);
        shared.prefix_boundary = Some(boundary);
        assert_eq!(e.submit(shared).is_ok(), valid, "boundary {boundary}");
    }
    finish(&mut e, 1);
    // The boundary rounds down to whole pages; the chunk ends there and nowhere else asks.
    assert_eq!(*published.lock().unwrap(), vec![2 * page]);
    e.submit(request(2, 100)).unwrap();
    finish(&mut e, 2000);
    assert_eq!(published.lock().unwrap().len(), 1);
    let output = |id| {
        e.request(RequestId::new(id).unwrap())
            .unwrap()
            .completed
            .as_ref()
            .and_then(|done| done.output.clone())
    };
    // Splitting the prompt at the boundary does not change what the request produces.
    assert!(output(1).is_some());
    assert_eq!(output(1), output(2));
}
