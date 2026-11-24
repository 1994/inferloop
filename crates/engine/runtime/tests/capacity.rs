use infer_test_support::fixtures as support;

use infer_core::{Error, ErrorCode, FinishReason, ModelId, RequestId, Result, StateId};
use infer_ir::{
    CanonicalRequest, DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, OutputReadout,
    PrecisionPlan, StepPlan, TaskOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::{BackendProvider, ResourceCommand, ResourceTicket, execute_resource};
use support::ProtocolBackend;

/// Injects the two resource-lane rejections the scene needs on top of the declared double.
struct Backpressure {
    inner: ProtocolBackend,
    submit_rejections: usize,
    reset_rejections: usize,
}
impl BackendProvider for Backpressure {
    type Ticket = <ProtocolBackend as BackendProvider>::Ticket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn formats(&self) -> infer_ir::BackendFormats {
        self.inner.formats()
    }
    fn validate_program(&self, m: &ModelIr, p: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(m, p)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        self.inner.execution_graph(model)
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
    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        self.inner.capture_execution_state()
    }
    fn restore_execution_state(&mut self, state: Option<&[u8]>) -> Result<()> {
        self.inner.restore_execution_state(state)
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.inner.reset_state(state)
    }
    fn begin_resource(&mut self, command: ResourceCommand) -> Result<ResourceTicket> {
        if matches!(command, ResourceCommand::Reset { .. }) && self.reset_rejections > 0 {
            self.reset_rejections -= 1;
            return Err(Error::new(
                ErrorCode::Capacity,
                "injected full resource lane",
            ));
        }
        let (ticket, responder) = ResourceTicket::channel();
        responder
            .send(execute_resource(self, command))
            .map_err(|_| Error::invariant("inline receiver disappeared"))?;
        Ok(ticket)
    }
    fn submit(
        &mut self,
        p: &ExecutionProgram,
        s: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        if self.submit_rejections > 0 {
            self.submit_rejections -= 1;
            return Err(Error::new(
                ErrorCode::Capacity,
                "injected full compute lane",
            ));
        }
        self.inner.submit(p, s, tasks)
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        self.inner.poll(ticket)
    }
    fn recycle_output(&mut self, state: StateId, output: infer_ir::ModelOutput) -> Result<()> {
        self.inner.recycle_output(state, output)
    }
    fn recycle_batch(&mut self, outputs: Vec<TaskOutput>) -> Result<()> {
        self.inner.recycle_batch(outputs)
    }
}
fn engine(submit_rejections: usize, reset_rejections: usize) -> Result<Engine<Backpressure>> {
    let ir = support::model(ModelId::ONE);
    let backend = Backpressure {
        inner: ProtocolBackend::new(4, 4, &ir)?,
        submit_rejections,
        reset_rejections,
    };
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels)?;
    Engine::new(
        backend,
        ir,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )
}
fn request(rerank: bool) -> Result<CanonicalRequest> {
    let input = if rerank {
        serde_json::json!({"Pairs":{"query":[1,2],"documents":[[3,4],[5,6]]}})
    } else {
        serde_json::json!({"Sequence":{"tokens":[1,2,3]}})
    };
    let workload = if rerank {
        serde_json::json!({"Rerank":{"top_k":2}})
    } else {
        serde_json::json!({"Generate":{"max_new_tokens":2}})
    };
    serde_json::from_value(serde_json::json!({"id":1,"model":1,"input":input,"workload":workload}))
        .map_err(|error| Error::invalid(error.to_string()))
}
fn drain(engine: &mut Engine<Backpressure>, start: u64) -> Result<()> {
    for now in start..start + 100 {
        engine.tick(now)?;
        engine.check_invariants()?;
        if engine.is_idle() {
            return Ok(());
        }
    }
    Err(Error::invariant("transient capacity did not recover"))
}
#[test]
fn compute_backpressure_retries_without_finishing_or_charging_request() -> Result<()> {
    let mut engine = engine(2, 0)?;
    engine.submit(request(false)?)?;
    for now in 0..2 {
        assert_eq!(engine.tick(now)?, [] as [infer_runtime::EngineOutput; 0]);
        assert!(engine.inspect().inflight_step.is_none());
        assert_eq!(engine.inspect().active_requests, 1);
        engine.check_invariants()?;
    }
    drain(&mut engine, 2)?;
    assert_eq!(
        engine
            .request(RequestId::ONE)?
            .completed
            .as_ref()
            .map(|done| &done.reason),
        Some(&FinishReason::Length)
    );
    assert_eq!(engine.inspect().state.allocated_pages, 0);
    Ok(())
}
#[test]
fn reset_backpressure_keeps_unit_transition_until_lane_recovers() -> Result<()> {
    let mut engine = engine(0, 2)?;
    engine.submit(request(true)?)?;
    let mut observed_waiter = false;
    for now in 0..100 {
        engine.tick(now)?;
        engine.check_invariants()?;
        observed_waiter |= engine.inspect().resource_waiters == 1;
        if engine.is_idle() {
            break;
        }
    }
    assert!(observed_waiter && engine.is_idle());
    eprintln!("DEBUG inspect={:?}", engine.inspect());
    assert!(
        engine
            .request(RequestId::ONE)?
            .completed
            .as_ref()
            .is_some_and(|done| done.output.is_some())
    );
    assert_eq!(engine.inspect().state.allocated_pages, 0);
    Ok(())
}
#[test]
fn cancelling_an_unpublished_reset_does_not_wait_for_an_ack_that_cannot_exist() -> Result<()> {
    let mut engine = engine(0, usize::MAX)?;
    engine.submit(request(true)?)?;
    for now in 0..100 {
        engine.tick(now)?;
        if engine.inspect().resource_waiters == 1 {
            break;
        }
    }
    assert_eq!(engine.inspect().resource_waiters, 1);
    engine.cancel(RequestId::ONE)?;
    engine.check_invariants()?;
    assert!(engine.is_idle());
    assert_eq!(engine.inspect().state.sequence_count, 0);
    assert_eq!(
        engine
            .request(RequestId::ONE)?
            .completed
            .as_ref()
            .map(|done| &done.reason),
        Some(&FinishReason::Cancelled)
    );
    Ok(())
}
