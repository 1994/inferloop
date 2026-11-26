//! What a finished request asks its backend to keep of its state.
use infer_test_support::fixtures as support;

use infer_core::{Error, ModelId, RequestId, Result, StateId};
use infer_ir::{
    CanonicalRequest, DeviceCapabilities, ExecutionProgram, ExecutionTask, FinishedState, ModelIr,
    OutputReadout, PrecisionPlan, StepPlan, TaskOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::{BackendProvider, ResourceCommand, ResourceTicket, execute_resource};
use support::ProtocolBackend;

/// The declared double, recording each finished state and what it was asked to keep.
struct Recording {
    inner: ProtocolBackend,
    finished: Vec<(StateId, FinishedState)>,
}
impl BackendProvider for Recording {
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
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)
    }
    fn finish_state(&mut self, state: StateId, kept: FinishedState) -> Result<()> {
        self.finished.push((state, kept));
        self.inner.release_state(state)
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.inner.reset_state(state)
    }
    fn begin_resource(&mut self, command: ResourceCommand) -> Result<ResourceTicket> {
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

fn engine() -> Result<Engine<Recording>> {
    let ir = support::model(ModelId::ONE);
    let backend = Recording {
        inner: ProtocolBackend::new(4, 4, &ir)?,
        finished: Vec::new(),
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

fn request(id: u64, on_finish: FinishedState) -> Result<CanonicalRequest> {
    let mut request: CanonicalRequest = serde_json::from_value(serde_json::json!({
        "id": id, "model": 1,
        "input": {"Sequence": {"tokens": [1, 2, 3]}},
        "workload": {"Generate": {"max_new_tokens": 2}}
    }))
    .map_err(|error| Error::invalid(error.to_string()))?;
    request.on_finish = on_finish;
    Ok(request)
}

fn drain(engine: &mut Engine<Recording>) -> Result<()> {
    for now in 0..100 {
        engine.tick(now)?;
        engine.check_invariants()?;
        if engine.is_idle() {
            return Ok(());
        }
    }
    Err(Error::invariant("requests did not finish"))
}

fn kept(engine: &Engine<Recording>) -> Vec<FinishedState> {
    engine
        .backend()
        .finished
        .iter()
        .map(|(_, kept)| *kept)
        .collect()
}

#[test]
fn a_request_that_finishes_on_its_own_passes_on_what_it_asked_to_keep() -> Result<()> {
    let mut engine = engine()?;
    engine.submit(request(1, FinishedState::PublishPrefix)?)?;
    engine.submit(request(2, FinishedState::Release)?)?;
    drain(&mut engine)?;
    let mut finished = kept(&engine);
    finished.sort_by_key(|kept| *kept == FinishedState::Release);
    assert_eq!(
        finished,
        [FinishedState::PublishPrefix, FinishedState::Release]
    );
    Ok(())
}

#[test]
fn a_cancelled_request_keeps_nothing() -> Result<()> {
    let mut engine = engine()?;
    engine.submit(request(1, FinishedState::PublishPrefix)?)?;
    engine.tick(0)?;
    engine.cancel(RequestId::ONE)?;
    drain(&mut engine)?;
    assert_eq!(kept(&engine), [FinishedState::Release]);
    Ok(())
}
