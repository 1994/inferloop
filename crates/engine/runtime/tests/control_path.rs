mod support;

use infer_kernel_api::KernelRegistry;
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::collections::BTreeMap;
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
    ProtocolBackend::tagged("control-path-weights", 16, 8, &support::model(ModelId::ONE)).unwrap()
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
fn request(id: u64, workload: Workload) -> CanonicalRequest {
    CanonicalRequest {
        id: RequestId::new(id).unwrap(),
        model: ModelId::ONE,
        session: None,
        input: RequestInput::Sequence {
            tokens: vec![1, 2, 3, 4, 5].into(),
            media: vec![],
        },
        workload,
        qos: Qos::default(),
        sampling: Sampling {
            temperature: 0.7,
            top_k: Some(8),
            seed: 13,
            ..Default::default()
        },
        prefix_boundary: None,
        extensions: BTreeMap::new(),
    }
}
fn workloads() -> Vec<CanonicalRequest> {
    let mut rerank = request(3, Workload::Rerank { top_k: 2 });
    rerank.input = RequestInput::Pairs {
        query: vec![1, 2].into(),
        documents: vec![vec![3, 4, 5, 6].into(), vec![8].into(), vec![9, 10].into()].into(),
    };
    vec![
        request(1, Workload::Generate { max_new_tokens: 6 }),
        request(
            2,
            Workload::Embed {
                pooling: Pooling::Mean,
                dimensions: Some(4),
                normalize: true,
            },
        ),
        rerank,
        request(
            4,
            Workload::Decision(DecisionSchema {
                questions: vec![
                    DecisionQuestion::Binary {
                        negative_token: 0,
                        positive_token: 1,
                    },
                    DecisionQuestion::Categorical {
                        options: vec![2, 3, 4],
                    },
                    DecisionQuestion::Ordinal {
                        options: vec![1, 2, 3],
                        values: vec![0.0, 1.0, 2.0],
                    },
                    DecisionQuestion::Continuous {
                        token: 1,
                        min: 0.0,
                        max: 10.0,
                    },
                ],
                calibration_temperature: 1.0,
                abstain_below: Some(0.9),
                readout: DecisionReadout::Projection,
            }),
        ),
    ]
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
        e.tick(tick * 100).unwrap();
        e.check_invariants().unwrap();
        if e.is_idle() {
            return;
        }
    }
    panic!("runtime did not finish");
}
#[test]
fn all_workloads_are_batch_and_chunk_invariant_and_release_state() {
    let mut baseline = engine(RuntimeConfig {
        max_num_seqs: 1,
        ..Default::default()
    });
    let mut candidate = engine(RuntimeConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 2,
        ..Default::default()
    });
    for r in workloads() {
        baseline.submit(r.clone()).unwrap();
        candidate.submit(r).unwrap();
    }
    finish(&mut baseline, 0);
    finish(&mut candidate, 0);
    for id in 1..=4 {
        let a = baseline
            .take_completed(RequestId::new(id).unwrap())
            .unwrap()
            .unwrap();
        let b = candidate
            .take_completed(RequestId::new(id).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(a.output, b.output);
        assert!(
            a.measurement.successful && b.measurement.successful,
            "{a:?} {b:?}"
        );
    }
    assert_eq!(baseline.inspect().state.allocated_pages, 0);
    assert_eq!(candidate.inspect().state.allocated_pages, 0);
    assert_eq!(baseline.inspect().active_requests, 0);
}
#[test]
fn journal_replays_scheduler_decisions_and_outputs_exactly() {
    let mut original = engine(RuntimeConfig {
        max_num_batched_tokens: 2,
        ..Default::default()
    });
    for r in workloads() {
        original.submit(r).unwrap();
    }
    original.tick(0).unwrap();
    original.cancel(RequestId::new(2).unwrap()).unwrap();
    finish(&mut original, 1);
    let journal = original.journal().unwrap();
    let mut replay = engine(original.config().clone());
    replay.replay(&journal).unwrap();
    assert_eq!(original.decisions(), replay.decisions());
    for id in 1..=4 {
        assert_eq!(
            original
                .request(RequestId::new(id).unwrap())
                .unwrap()
                .completed,
            replay
                .request(RequestId::new(id).unwrap())
                .unwrap()
                .completed
        );
    }
}
#[test]
fn checkpoint_roundtrip_and_weight_mismatch_rejection() {
    let mut original = engine(RuntimeConfig {
        max_num_batched_tokens: 2,
        ..Default::default()
    });
    original
        .submit(request(1, Workload::Generate { max_new_tokens: 6 }))
        .unwrap();
    let checkpoint = original.snapshot().unwrap();
    let bytes = serde_json::to_vec(&checkpoint).unwrap();
    let decoded = serde_json::from_slice(&bytes).unwrap();
    let mut restored = Engine::restore(backend(), &registry(), decoded).unwrap();
    finish(&mut original, 0);
    finish(&mut restored, 0);
    assert_eq!(
        original
            .request(RequestId::new(1).unwrap())
            .unwrap()
            .completed,
        restored
            .request(RequestId::new(1).unwrap())
            .unwrap()
            .completed
    );
    let other = ProtocolBackend::tagged(
        "control-path-other-weights",
        16,
        8,
        &support::model(ModelId::ONE),
    )
    .unwrap();
    assert!(Engine::restore(other, &registry(), checkpoint).is_err());
}
#[test]
fn checkpoint_after_partial_generation_resumes_the_same_trajectory() {
    let mut original = engine(RuntimeConfig::default());
    original
        .submit(request(1, Workload::Generate { max_new_tokens: 6 }))
        .unwrap();
    original.tick(0).unwrap();
    let (events, snapshot) = original.quiesce(1).unwrap();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EngineOutput::Token { .. }))
    );
    let snapshot = snapshot.unwrap();
    assert_eq!(
        snapshot.requests[&RequestId::new(1).unwrap()]
            .generated
            .len(),
        1
    );
    let snapshot = serde_json::from_slice(&serde_json::to_vec(&snapshot).unwrap()).unwrap();
    let mut restored = Engine::restore(backend(), &registry(), snapshot).unwrap();
    finish(&mut original, 2);
    finish(&mut restored, 2);
    assert_eq!(
        original
            .request(RequestId::new(1).unwrap())
            .unwrap()
            .completed,
        restored
            .request(RequestId::new(1).unwrap())
            .unwrap()
            .completed
    );
    let mut replay = engine(original.config().clone());
    replay.replay(&original.journal().unwrap()).unwrap();
    assert_eq!(original.decisions(), replay.decisions());
}
#[test]
fn consumed_request_id_cannot_alias_a_new_request() {
    let mut e = engine(RuntimeConfig {
        history_capacity: 2,
        ..Default::default()
    });
    assert_eq!(e.request_id_high_watermark(), 0);
    for id in 1..=3 {
        e.submit(request(id, Workload::Generate { max_new_tokens: 1 }))
            .unwrap();
        e.cancel(RequestId::new(id).unwrap()).unwrap();
        e.take_completed(RequestId::new(id).unwrap()).unwrap();
    }
    assert_eq!(e.request_id_high_watermark(), 3);
    let restored = Engine::restore(backend(), &registry(), e.snapshot().unwrap()).unwrap();
    assert_eq!(restored.request_id_high_watermark(), 3);
    for id in 1..=3 {
        assert_eq!(
            e.submit(request(id, Workload::Generate { max_new_tokens: 1 }))
                .unwrap_err()
                .code,
            ErrorCode::Conflict
        );
    }
    e.check_invariants().unwrap();
}
#[test]
fn weighted_service_advances_heavier_tenant_without_starvation() {
    let mut e = engine(RuntimeConfig {
        max_num_batched_tokens: 1,
        max_num_seqs: 1,
        ..Default::default()
    });
    let mut a = request(
        1,
        Workload::Generate {
            max_new_tokens: 100,
        },
    );
    let mut b = request(
        2,
        Workload::Generate {
            max_new_tokens: 100,
        },
    );
    a.qos.tenant = "a".into();
    b.qos.tenant = "b".into();
    b.qos.weight = 2;
    e.submit(a).unwrap();
    e.submit(b).unwrap();
    for tick in 0..100 {
        e.tick(tick).unwrap();
    }
    let a = e
        .request(RequestId::new(1).unwrap())
        .unwrap()
        .generated
        .len();
    let b = e
        .request(RequestId::new(2).unwrap())
        .unwrap()
        .generated
        .len();
    assert!(a > 0 && b > a && b <= a * 3 + 6, "a={a}, b={b}");
    e.cancel(RequestId::new(1).unwrap()).unwrap();
    e.cancel(RequestId::new(2).unwrap()).unwrap();
    finish(&mut e, 2);
    assert_eq!(e.inspect().state.allocated_pages, 0);
}
#[test]
fn corrupted_checkpoint_cursor_is_rejected() {
    let mut e = engine(RuntimeConfig::default());
    e.submit(request(1, Workload::Generate { max_new_tokens: 6 }))
        .unwrap();
    let mut json = serde_json::to_value(e.snapshot().unwrap()).unwrap();
    json["requests"]["1"]["prefill_done"] = serde_json::json!(usize::MAX);
    let invalid = serde_json::from_value(json).unwrap();
    assert!(Engine::restore(backend(), &registry(), invalid).is_err());
}
#[derive(Clone, Copy)]
enum Fault {
    Submit,
    Poll,
    Malformed,
    NeverComplete,
    Recover,
    InvalidTiming,
}
struct FaultyBackend {
    inner: ProtocolBackend,
    fault: Fault,
}
impl BackendProvider for FaultyBackend {
    type Ticket = <ProtocolBackend as BackendProvider>::Ticket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        // The engine creates sequences through this entry point.
        self.inner.reserve_state_for(state, capacity, readout)
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.inner.reset_state(state)
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)
    }
    fn recycle_output(&mut self, state: StateId, output: ModelOutput) -> Result<()> {
        self.inner.recycle_output(state, output)
    }
    fn recycle_batch(&mut self, outputs: Vec<TaskOutput>) -> Result<()> {
        self.inner.recycle_batch(outputs)
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
        if matches!(self.fault, Fault::Submit) {
            return Err(Error::new(ErrorCode::Backend, "injected submit failure"));
        }
        self.inner.submit(p, s, t)
    }
    fn poll(&mut self, t: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        match self.fault {
            Fault::Poll => Err(Error::new(ErrorCode::Backend, "injected poll failure")),
            Fault::NeverComplete => Ok(None),
            Fault::Recover | Fault::InvalidTiming => self.inner.poll(t),
            Fault::Malformed => {
                let mut outputs = self.inner.poll(t)?;
                if let Some(outputs) = &mut outputs {
                    outputs.clear();
                }
                Ok(outputs)
            }
            Fault::Submit => unreachable!(),
        }
    }
    fn completion_timing(&self, _: &Self::Ticket) -> Option<ExecutionTiming> {
        matches!(self.fault, Fault::InvalidTiming).then_some(ExecutionTiming {
            elapsed_us: 0,
            source: TimingSource::CpuWall,
        })
    }
}
#[test]
fn failure_after_device_completion_does_not_leave_requests_waiting_on_a_missing_ticket() {
    let backend = FaultyBackend {
        inner: backend(),
        fault: Fault::InvalidTiming,
    };
    let model = support::model(ModelId::ONE);
    let mut e = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry(),
        RuntimeConfig::default(),
    )
    .unwrap();
    e.submit(request(1, Workload::Generate { max_new_tokens: 2 }))
        .unwrap();
    e.tick(0).unwrap();
    assert_eq!(e.tick(1).unwrap_err().code, ErrorCode::InvalidInput);
    assert!(e.inspect().inflight_step.is_none());
    assert!(!e.inspect().ready && e.inspect().resource_release_pending);
    e.tick(2).unwrap();
    assert!(e.is_idle());
    assert_eq!(e.inspect().state.allocated_pages, 0);
    e.check_invariants().unwrap();
}
#[test]
fn unbounded_trace_configuration_is_rejected_before_ring_allocation() {
    for config in [
        RuntimeConfig {
            event_capacity: usize::MAX,
            ..Default::default()
        },
        RuntimeConfig {
            history_capacity: usize::MAX,
            ..Default::default()
        },
    ] {
        let backend = backend();
        let model = support::model(ModelId::ONE);
        let result = Engine::new(backend, model, PrecisionPlan::f32(), &registry(), config);
        assert_eq!(result.err().unwrap().code, ErrorCode::InvalidInput);
    }
}
#[test]
fn backend_failures_and_invalid_completions_release_all_owned_state() {
    for fault in [Fault::Submit, Fault::Poll, Fault::Malformed] {
        let b = FaultyBackend {
            inner: backend(),
            fault,
        };
        let model = support::model(ModelId::ONE);
        let mut e = Engine::new(
            b,
            model,
            PrecisionPlan::f32(),
            &registry(),
            RuntimeConfig::default(),
        )
        .unwrap();
        for r in workloads() {
            e.submit(r).unwrap();
        }
        finish(&mut e, 0);
        for id in 1..=4 {
            assert!(matches!(
                e.request(RequestId::new(id).unwrap()).unwrap().status,
                RequestStatus::Finished(FinishReason::Failed(_))
            ));
        }
        assert_eq!(e.inspect().state.allocated_pages, 0);
    }
}
#[test]
fn backend_timeout_is_diagnosed_without_recycling_inflight_state() {
    let b = FaultyBackend {
        inner: backend(),
        fault: Fault::NeverComplete,
    };
    let model = support::model(ModelId::ONE);
    let mut e = Engine::new(
        b,
        model,
        PrecisionPlan::f32(),
        &registry(),
        RuntimeConfig {
            submission_timeout_us: 10,
            ..Default::default()
        },
    )
    .unwrap();
    e.submit(request(1, Workload::Generate { max_new_tokens: 1 }))
        .unwrap();
    e.tick(0).unwrap();
    assert_eq!(e.tick(11).unwrap_err().code, ErrorCode::Invariant);
    assert_eq!(e.inspect().state.sequence_count, 1);
    assert!(e.last_diagnostic_snapshot().unwrap().inflight.is_some());
    e.check_invariants().unwrap();
    assert!(!e.inspect().ready);
    assert!(e.inspect().resource_release_pending);
    assert_eq!(
        e.submit(request(2, Workload::Generate { max_new_tokens: 1 }))
            .unwrap_err()
            .code,
        ErrorCode::Backend
    );
    let diagnostics = e.diagnostics();
    assert!(diagnostics.iter().any(
        |d| d.code == infer_observe::DiagnosticCode::SubmissionTimeout
            && d.resource_release_pending
    ));
    // Further polling is safe and does not repeatedly emit the timeout or free pages.
    e.tick(12).unwrap();
    assert_eq!(e.inspect().state.sequence_count, 1);
    e.cancel(RequestId::new(1).unwrap()).unwrap();
    e.backend_mut().fault = Fault::Recover;
    let outputs = e.tick(13).unwrap();
    assert!(outputs.iter().any(|output|matches!(output,EngineOutput::Finished(done) if matches!(done.reason,FinishReason::Failed(_)))));
    assert_eq!(e.inspect().state.allocated_pages, 0);
    assert!(!e.inspect().resource_release_pending);
    assert!(!e.inspect().ready);
    assert!(e.snapshot().is_err());
    assert!(
        Engine::restore(
            backend(),
            &registry(),
            e.last_diagnostic_snapshot().unwrap().clone()
        )
        .is_err()
    );
}
#[test]
fn admission_failure_and_queued_cancellation_leave_no_leaks() {
    let mut e = engine(RuntimeConfig {
        num_gpu_blocks: 1,
        block_size: 8,
        ..Default::default()
    });
    assert_eq!(
        e.submit(request(1, Workload::Generate { max_new_tokens: 10 }))
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
    assert_eq!(e.inspect().state.allocated_pages, 0);
    assert_eq!(e.inspect().active_requests, 0);
    e.submit(request(1, Workload::Generate { max_new_tokens: 1 }))
        .unwrap();
    e.cancel(RequestId::new(1).unwrap()).unwrap();
    e.check_invariants().unwrap();
    assert_eq!(e.inspect().state.allocated_pages, 0);
    assert_eq!(
        e.cancel(RequestId::new(1).unwrap()).unwrap(),
        [] as [EngineOutput; 0]
    );
}
struct DelayedBackend {
    inner: ProtocolBackend,
}
struct DelayedTicket {
    inner: <ProtocolBackend as BackendProvider>::Ticket,
    wait: bool,
}
impl BackendProvider for DelayedBackend {
    type Ticket = DelayedTicket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        // The engine creates sequences through this entry point.
        self.inner.reserve_state_for(state, capacity, readout)
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.inner.reset_state(state)
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)
    }
    fn recycle_output(&mut self, state: StateId, output: ModelOutput) -> Result<()> {
        self.inner.recycle_output(state, output)
    }
    fn recycle_batch(&mut self, outputs: Vec<TaskOutput>) -> Result<()> {
        self.inner.recycle_batch(outputs)
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
        Ok(DelayedTicket {
            inner: self.inner.submit(p, s, t)?,
            wait: true,
        })
    }
    fn poll(&mut self, t: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        if t.wait {
            t.wait = false;
            Ok(None)
        } else {
            self.inner.poll(&mut t.inner)
        }
    }
}
#[test]
fn in_flight_cancel_retains_state_until_completion() {
    let b = DelayedBackend { inner: backend() };
    let model = support::model(ModelId::ONE);
    let mut e = Engine::new(
        b,
        model,
        PrecisionPlan::f32(),
        &registry(),
        RuntimeConfig::default(),
    )
    .unwrap();
    e.submit(request(1, Workload::Generate { max_new_tokens: 6 }))
        .unwrap();
    e.tick(0).unwrap();
    assert!(e.snapshot().is_err());
    e.cancel(RequestId::new(1).unwrap()).unwrap();
    assert_eq!(e.inspect().state.sequence_count, 1);
    e.tick(1).unwrap();
    assert_eq!(e.inspect().state.sequence_count, 1);
    let events = e.tick(2).unwrap();
    assert_eq!(e.inspect().state.allocated_pages, 0);
    assert!(matches!(
        &events[0],
        EngineOutput::Finished(CompletedRequest {
            reason: FinishReason::Cancelled,
            ..
        })
    ));
    e.check_invariants().unwrap();
}
struct EmptyPolicy;
impl SchedulingPolicy for EmptyPolicy {
    type Workspace = ();
    fn plan(
        &self,
        ready: &[ReadyWork],
        _: &ResourceSnapshot,
        _: u64,
        id: DecisionId,
        _: StepId,
    ) -> Result<SchedulingDecision> {
        Ok(SchedulingDecision {
            window: None,
            id,
            step: None,
            deferred: ready
                .iter()
                .map(|work| DeferredWork {
                    request: work.request,
                    reason: DeferReason::StateCapacity,
                    required: 1,
                    available: 0,
                })
                .collect(),
            selected: vec![],
        })
    }
}
#[test]
fn scheduler_livelock_captures_diagnostic_snapshot() {
    let b = backend();
    let model = support::model(ModelId::ONE);
    let mut e = Engine::with_policy(
        b,
        model,
        PrecisionPlan::f32(),
        &registry(),
        RuntimeConfig::default(),
        EmptyPolicy,
    )
    .unwrap();
    e.submit(request(1, Workload::Generate { max_new_tokens: 1 }))
        .unwrap();
    e.tick(0).unwrap();
    e.tick(1).unwrap();
    assert_eq!(e.tick(2).unwrap_err().code, ErrorCode::Invariant);
    assert!(e.last_diagnostic_snapshot().is_some());
    assert_eq!(e.inspect().active_requests, 1);
    assert!(!e.inspect().ready);
    assert!(
        e.diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == infer_observe::DiagnosticCode::ProgressStall)
    );
    e.tick(3).unwrap();
    assert_eq!(e.inspect().active_requests, 0);
    assert_eq!(e.inspect().state.allocated_pages, 0);
}
#[test]
fn deadline_expiry_finishes_with_reason_and_no_state() {
    let mut e = engine(RuntimeConfig::default());
    let mut r = request(1, Workload::Generate { max_new_tokens: 1 });
    r.qos.deadline_us = Some(1);
    e.submit(r).unwrap();
    e.tick(2).unwrap();
    assert_eq!(
        e.request(RequestId::new(1).unwrap()).unwrap().status,
        RequestStatus::Finished(FinishReason::Deadline)
    );
    assert_eq!(e.inspect().state.allocated_pages, 0);
}

#[test]
fn replay_token_budget_evicts_payload_without_releasing_live_request_ownership() -> Result<()> {
    let mut engine = engine(RuntimeConfig {
        max_history_tokens: 5,
        ..Default::default()
    });
    engine.submit(request(1, Workload::Generate { max_new_tokens: 6 }))?;
    engine.submit(request(2, Workload::Generate { max_new_tokens: 6 }))?;
    assert!(engine.journal().is_err());
    assert_eq!(engine.inspect().active_requests, 2);
    engine.check_invariants()?;
    engine.cancel(RequestId::ONE)?;
    engine.cancel(RequestId::new(2)?)?;
    assert_eq!(engine.inspect().state.sequence_count, 0);
    Ok(())
}
