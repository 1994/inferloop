//! `BackendProvider` adaptation; lifecycle helpers live alongside their owning subsystem.
use super::{MetalBackend, MetalTicket};
use crate::constants::COMPLETION_POOL_SIZE;
use crate::constants::NANOS_PER_MICROSECOND;
use crate::device::command_timing;
use infer_core::{DeviceId, Error, ErrorCode, Result, StateId};
use infer_ir::{
    DType, DeviceBackend, DeviceCapabilities, ExecutionProgram, ExecutionTask, ExecutionTiming,
    KvCacheInspection, MetalCapabilities, ModelIr, PageGrowth, StepPlan, TaskOutput, TimingSource,
};
use infer_spi::BackendProvider;
use metal::MTLCommandBufferStatus;
use std::sync::Arc;

impl MetalBackend {
    /// Checkpoint formats and weight encodings this backend executes.
    pub const FORMATS: infer_ir::BackendFormats = infer_ir::BackendFormats {
        checkpoints: &[
            infer_ir::CheckpointFormat::Safetensors,
            infer_ir::CheckpointFormat::Gguf,
            infer_ir::CheckpointFormat::Mlx,
        ],
        encodings: &[
            infer_ir::ExecutableEncoding::Dense,
            infer_ir::ExecutableEncoding::MlxAffine,
        ],
    };
}

impl BackendProvider for MetalBackend {
    type Ticket = MetalTicket;
    fn begin_resource(
        &mut self,
        command: infer_spi::ResourceCommand,
    ) -> Result<infer_spi::ResourceTicket> {
        let (ticket, reply) = self.resources.channel()?;
        let _ = reply.send(infer_spi::execute_resource(self, command));
        Ok(ticket)
    }
    fn weight_backed_dataflow(&self) -> bool {
        true
    }
    fn identity(&self) -> &str {
        &self.identity
    }
    fn supports_control_checkpoint(&self) -> bool {
        true
    }
    fn formats(&self) -> infer_ir::BackendFormats {
        Self::FORMATS
    }
    fn device_identity(&self) -> Option<infer_ir::DeviceIdentity> {
        // One system device, addressed by no PCI slot; its family is the backend itself.
        Some(infer_ir::DeviceIdentity {
            ordinal: 0,
            name: self.device_name().to_owned(),
            architecture: "metal".into(),
            pci_address: None,
            memory_bytes: self.config.memory_bytes,
        })
    }
    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            backend: DeviceBackend::Metal(MetalCapabilities {
                simd_width: self.gpu.simd_width(),
            }),
            device: DeviceId::ONE,
            compute_dtypes: vec![DType::F32],
            memory_bytes: self.config.memory_bytes,
            unified_memory: true,
            profiling: true,
            speculation: infer_ir::SpeculationCapability::default(),
        }
    }
    fn state_recipe(&self) -> Option<&infer_ir::StateRecipe> {
        Some(&self.state_recipe)
    }
    fn state_reservation_bytes(&self, capacity: usize) -> Result<Option<u64>> {
        Ok(Some(
            self.state_bytes(capacity, infer_ir::OutputReadout::Full)?,
        ))
    }
    fn state_reservation_bytes_for(
        &self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<Option<u64>> {
        Ok(Some(self.state_bytes(capacity, readout)?))
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        let used = self
            .base_bytes()?
            .checked_add(self.config.probe_bytes)
            .and_then(|n| n.checked_add(self.inspect().reserved_bytes))
            .ok_or_else(|| Error::invalid("Metal state budget overflow"))?;
        Ok(Some(self.config.memory_bytes.saturating_sub(used)))
    }
    fn kv_cache(&self) -> Option<KvCacheInspection> {
        Some(self.kv_inspection())
    }
    fn state_page_growth(&self, state: StateId) -> Result<Option<PageGrowth>> {
        let s = self
            .sequences
            .get(&state)
            .ok_or_else(|| Error::invalid("unknown KV sequence"))?;
        Ok(Some(self.kv.page_growth(&s.blocks, s.tokens.len())?))
    }
    fn supports_recompute_preemption(&self) -> bool {
        true
    }
    fn reusable_prefix_for(&self, state: StateId, tokens: &[u32], maximum: usize) -> usize {
        let Some(sequence) = self.sequences.get(&state) else {
            return 0;
        };
        self.kv.matched_tokens_where(tokens, maximum, |cache| {
            sequence.readout != infer_ir::OutputReadout::Full
                || cache.readout == infer_ir::OutputReadout::Full
        })
    }
    fn reusable_prefix(&self, tokens: &[u32], maximum: usize) -> usize {
        self.kv.matched_tokens(tokens, maximum)
    }
    fn reuse_prefix(&mut self, id: StateId, tokens: &[u32], maximum: usize) -> Result<usize> {
        self.provider_reuse_prefix(id, tokens, maximum)
    }

    fn completion_timing(&self, ticket: &MetalTicket) -> Option<ExecutionTiming> {
        if !ticket.done
            || !Arc::ptr_eq(&self.owner, &ticket.owner)
            || ticket.command.status() != MTLCommandBufferStatus::Completed
        {
            return None;
        }
        let (_, ns) = command_timing(&ticket.command);
        (ns > 0).then_some(ExecutionTiming {
            elapsed_us: ns.div_ceil(NANOS_PER_MICROSECOND),
            source: TimingSource::MetalGpu,
        })
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        if model != &self.model {
            return Err(Error::invalid("execution graph model mismatch"));
        }
        Ok(self.graph.clone())
    }
    fn validate_program(&self, model: &ModelIr, p: &ExecutionProgram) -> Result<()> {
        self.provider_validate_program(model, p)
    }

    fn reserve_state(&mut self, id: StateId, capacity: usize) -> Result<()> {
        self.reserve_state_for(id, capacity, infer_ir::OutputReadout::Full)
    }
    fn reserve_state_for(
        &mut self,
        id: StateId,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<()> {
        self.provider_reserve_state_for(id, capacity, readout)
    }

    fn reset_state(&mut self, id: StateId) -> Result<()> {
        self.provider_reset_state(id)
    }

    fn release_state(&mut self, id: StateId) -> Result<()> {
        if self.inflight_states.contains(&id) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "Metal completion must precede release",
            ));
        }
        let s = self
            .sequences
            .remove(&id)
            .ok_or_else(|| Error::invalid("unknown Metal state"))?;
        self.kv.release(&s.blocks)?;
        Ok(())
    }

    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<MetalTicket> {
        self.submit_borrowed(program, step, &tasks)
    }
    fn submit_borrowed(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<MetalTicket> {
        self.provider_submit_borrowed(program, step, tasks)
    }

    fn recycle_output(&mut self, state: StateId, mut output: infer_ir::ModelOutput) -> Result<()> {
        if let Some(sequence) = self.sequences.get_mut(&state) {
            output.hidden.append(&mut sequence.hidden_spares);
            if sequence.readback.is_none() {
                sequence.readback = Some(output);
            }
        }
        Ok(())
    }
    fn recycle_batch(&mut self, mut outputs: Vec<TaskOutput>) -> Result<()> {
        outputs.clear();
        if self.completion_pool.len() < COMPLETION_POOL_SIZE {
            self.completion_pool.push(outputs);
        }
        Ok(())
    }
    fn poll(&mut self, t: &mut MetalTicket) -> Result<Option<Vec<TaskOutput>>> {
        self.provider_poll(t)
    }

    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        self.provider_capture_execution_state()
    }

    fn restore_execution_state(&mut self, data: Option<&[u8]>) -> Result<()> {
        self.provider_restore_execution_state(data)
    }

    fn validate_state_ownership(&self, states: &[(StateId, usize, usize)]) -> Result<()> {
        self.provider_validate_state_ownership(states)
    }
}
