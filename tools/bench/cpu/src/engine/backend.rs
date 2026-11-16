//! A persistent device-contract double isolates Engine CPU allocations from native driver allocations.
use infer_core::{DeviceId, Error, Result, StateId, map::BoundedMap};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, ModelOutput, OutputReadout,
    StepPlan, TaskOutput,
};
use infer_model_recipes::decoder::lower;
use infer_spi::BackendProvider;
struct State {
    output: Option<ModelOutput>,
}
pub struct Backend {
    states: BoundedMap<StateId, State>,
    outputs: Vec<ModelOutput>,
    batches: Vec<Vec<TaskOutput>>,
    resources: infer_spi::ResourcePool,
    vocabulary: usize,
}
pub struct Ticket(Option<Vec<TaskOutput>>);
impl Backend {
    pub fn new(requests: usize, batch: usize, vocabulary: usize) -> Result<Self> {
        Ok(Self {
            states: BoundedMap::new(requests)?,
            outputs: (0..requests)
                .map(|_| ModelOutput {
                    next_token: None,
                    logits: vec![1.0; vocabulary],
                    hidden: Vec::new(),
                    // The output stage pushes decided tokens into this buffer, so it must
                    // arrive with capacity and never grow on a hot path.
                    tokens: Vec::with_capacity(8),
                })
                .collect(),
            batches: (0..4).map(|_| Vec::with_capacity(batch)).collect(),
            resources: infer_spi::ResourcePool::new(requests + 4)?,
            vocabulary,
        })
    }
}
impl BackendProvider for Backend {
    type Ticket = Ticket;
    fn identity(&self) -> &'static str {
        "persistent-device-contract-double"
    }
    fn formats(&self) -> infer_ir::BackendFormats {
        // A device-contract double that loads no checkpoint.
        infer_ir::BackendFormats::default()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        // A planning sample, not a claim that a device exists: this backend is a device-contract
        // double that runs no model, and the engine only reads the descriptor to plan.
        DeviceCapabilities {
            backend: infer_ir::DeviceBackend::Cuda(infer_ir::NvidiaCapabilities {
                architecture: infer_ir::NvidiaArchitecture {
                    compute_major: 12,
                    compute_minor: 0,
                    multiprocessors: 170,
                },
                tensor_core_generation: Some(5),
                warp_size: 32,
                graphs: true,
                tma: true,
                clusters: true,
                pinned_transfer: true,
                cuda_ipc: true,
                nvlink: false,
                gpu_direct: false,
            }),
            device: DeviceId::ONE,
            // These three match what this backend reported before, so the measured allocation
            // profile stays comparable.
            compute_dtypes: vec![infer_ir::DType::F32],
            memory_bytes: 0,
            unified_memory: true,
            profiling: false,
            speculation: infer_ir::SpeculationCapability::default(),
        }
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        lower(model)
    }
    fn validate_program(&self, _: &ModelIr, _: &ExecutionProgram) -> Result<()> {
        Ok(())
    }
    fn supports_recompute_preemption(&self) -> bool {
        true
    }
    fn begin_resource(
        &mut self,
        command: infer_spi::ResourceCommand,
    ) -> Result<infer_spi::ResourceTicket> {
        let (ticket, reply) = self.resources.channel()?;
        let _ = reply.send(infer_spi::execute_resource(self, command));
        Ok(ticket)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        _: usize,
        _readout: OutputReadout,
    ) -> Result<()> {
        let output = self
            .outputs
            .pop()
            .ok_or_else(|| Error::invariant("readback credit missing"))?;
        self.states.insert(
            state,
            State {
                output: Some(output),
            },
        )?;
        Ok(())
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        let state = self
            .states
            .remove(&state)
            .ok_or_else(|| Error::invariant("release state missing"))?;
        if let Some(output) = state.output {
            self.outputs.push(output);
        }
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Ticket> {
        self.submit_borrowed(program, step, &tasks)
    }
    fn submit_borrowed(
        &mut self,
        _: &ExecutionProgram,
        _step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<Ticket> {
        let mut batch = self
            .batches
            .pop()
            .ok_or_else(|| Error::invariant("batch credit missing"))?;
        for task in tasks {
            let state = self
                .states
                .get_mut(&task.state)
                .ok_or_else(|| Error::invariant("submit state missing"))?;
            let mut output = state
                .output
                .take()
                .ok_or_else(|| Error::invariant("readback still leased"))?;
            let readout = task.tokens.readout();
            output.logits.resize(
                if readout == OutputReadout::None {
                    0
                } else {
                    self.vocabulary
                },
                1.0,
            );
            batch.push(TaskOutput {
                request: task.request,
                output,
            });
        }
        Ok(Ticket(Some(batch)))
    }
    fn recycle_output(&mut self, state: StateId, output: ModelOutput) -> Result<()> {
        self.states
            .get_mut(&state)
            .ok_or_else(|| Error::invariant("recycle owner missing"))?
            .output = Some(output);
        Ok(())
    }
    fn recycle_batch(&mut self, mut outputs: Vec<TaskOutput>) -> Result<()> {
        outputs.clear();
        self.batches.push(outputs);
        Ok(())
    }
    fn poll(&mut self, ticket: &mut Ticket) -> Result<Option<Vec<TaskOutput>>> {
        Ok(ticket.0.take())
    }
}
