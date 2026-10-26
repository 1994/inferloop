//! Backend extension contract.
use super::{ResourceCommand, ResourceTicket, execute_resource};
use infer_core::{Error, Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, ExecutionTiming, KvCacheInspection,
    ModelIr, PageGrowth, StepPlan, TaskOutput,
};
use std::sync::Arc;

/// Owned work crosses the submission boundary; a future CUDA implementation uses
/// persistent buffers and a bounded SPSC descriptor ring behind this contract.
pub trait BackendProvider {
    type Ticket;
    /// Begin a bounded resource transaction. Threaded owners enqueue; direct backends settle inline.
    /// # Errors
    /// Rejects unavailable command capacity or a stopped owner.
    fn begin_resource(&mut self, command: ResourceCommand) -> Result<ResourceTicket> {
        let (ticket, reply) = ResourceTicket::channel();
        let _ = reply.send(execute_resource(self, command));
        Ok(ticket)
    }
    /// True when reservation intent lives on an owner lane, so a `Reserve` the owner
    /// rejects for capacity can be absorbed as backpressure and its state released later.
    /// Direct backends settle inline and must reject admission instead of deferring.
    fn tracks_reservation_intent(&self) -> bool {
        false
    }
    fn identity(&self) -> &str;
    /// Reports loaded tensor dataflow execution without exposing concrete backend names.
    fn weight_backed_dataflow(&self) -> bool {
        false
    }
    /// Whether this backend has a bound multimodal encoder and can execute media prefill.
    fn supports_multimodal(&self) -> bool {
        false
    }
    /// True only if control checkpoints can reconstruct all execution state.
    /// A proxy captures execution state via its owner lane; direct drivers can capture synchronously.
    fn requires_async_checkpoint(&self) -> bool {
        false
    }
    fn supports_control_checkpoint(&self) -> bool {
        false
    }
    fn capabilities(&self) -> DeviceCapabilities;
    /// Speculative decode support, queried on hot paths where cloning capabilities would allocate.
    fn speculation_capability(&self) -> infer_ir::SpeculationCapability {
        infer_ir::SpeculationCapability::default()
    }
    /// Whether cold control/resource calls can be accepted without waiting behind CPU encoding.
    fn control_ready(&self) -> bool {
        true
    }
    /// Drive pending host commands without waiting for device execution.
    /// # Errors
    /// Returns a terminal owner/protocol error, retaining any in-flight resource ownership.
    fn maintenance(&mut self) -> Result<()> {
        Ok(())
    }
    fn pending_resource_releases(&self) -> bool {
        false
    }
    fn resource_epoch(&self) -> u64 {
        0
    }
    /// Install a persistent owner notification. Implementations must not mutate the engine in callbacks.
    fn set_waker(&mut self, _wake: Arc<dyn Fn() + Send + Sync>) {}
    /// Effective load-time graph geometry and speculation depth when the backend resolves
    /// them from device policy. Opaque compatibility drivers return None.
    fn execution_profile(&self) -> Option<infer_ir::ExecutionProfileInspection> {
        None
    }
    /// # Errors
    /// Returns an invalid-input or capacity error if the reservation size overflows or the state layout is unsupported.
    /// Cold compiled complete physical layout. Opaque compatibility drivers return None.
    fn state_recipe(&self) -> Option<&infer_ir::StateRecipe> {
        None
    }
    /// # Errors
    /// Rejects unsupported or overflowing state reservations.
    fn state_reservation_bytes(&self, _capacity: usize) -> Result<Option<u64>> {
        Ok(None)
    }
    /// Prepare an admission quote on the owning thread. Backends may reclaim idle,
    /// incompatible storage before reporting the reservation price and free capacity.
    /// Live state and its storage must remain unchanged.
    /// # Errors
    /// Returns allocator, ownership or device errors while preparing the quote.
    fn prepare_state_reservation(
        &mut self,
        _capacity: usize,
        _readout: infer_ir::OutputReadout,
    ) -> Result<()> {
        Ok(())
    }
    /// Size the retained state for the workload's output contract.
    /// # Errors
    /// Returns layout/capacity errors.
    fn state_reservation_bytes_for(
        &self,
        capacity: usize,
        _readout: infer_ir::OutputReadout,
    ) -> Result<Option<u64>> {
        self.state_reservation_bytes(capacity)
    }
    /// # Errors
    /// Returns a backend or invariant error if the physical allocator cannot report its available capacity.
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        Ok(None)
    }
    /// State-byte budget for a single queued reservation, independent of live leases.
    /// The runtime uses this only when the adapter owns deferred reservation intent.
    /// Returning None keeps admission bounded by current free bytes.
    /// Actual reservation must still check live physical capacity.
    /// # Errors
    /// Returns backend errors while reading the configured state budget.
    fn state_capacity_bytes(&self) -> Result<Option<u64>> {
        Ok(None)
    }
    fn kv_cache(&self) -> Option<KvCacheInspection> {
        None
    }
    /// # Errors
    /// Returns a not-found or invariant error for unknown state or an invalid physical page table.
    fn state_page_growth(&self, _state: StateId) -> Result<Option<PageGrowth>> {
        Ok(None)
    }
    /// Prompt replay geometry for this state's current resident program.
    /// Opaque or variable-row backends leave the scheduler's existing grant policy unchanged.
    /// # Errors
    /// Rejects unknown state ids or inconsistent resident geometry.
    fn state_prefill_geometry(&self, _state: StateId) -> Result<Option<infer_ir::PrefillGeometry>> {
        Ok(None)
    }
    fn reusable_prefix(&self, _tokens: &[u32], _maximum: usize) -> usize {
        0
    }
    fn reusable_prefix_for(&self, _state: StateId, tokens: &[u32], maximum: usize) -> usize {
        self.reusable_prefix(tokens, maximum)
    }
    fn reusable_prefix_shared(
        &self,
        state: StateId,
        tokens: infer_ir::TokenBuffer,
        maximum: usize,
    ) -> usize {
        self.reusable_prefix_for(state, &tokens, maximum)
    }
    /// Attach a prefix without requiring a copied descriptor payload.
    /// # Errors
    /// Returns prefix ownership/capacity errors.
    fn reuse_prefix_shared(
        &mut self,
        state: StateId,
        tokens: infer_ir::TokenBuffer,
        maximum: usize,
    ) -> Result<usize> {
        self.reuse_prefix(state, &tokens, maximum)
    }
    /// Attach complete immutable KV blocks and their matching recurrent state.
    /// Device copies may be deferred to the next submit; this call does not wait.
    ///
    /// # Errors
    /// Returns a not-found, capacity, or invariant error if cached leases cannot be attached to the sequence.
    fn reuse_prefix(&mut self, _state: StateId, _tokens: &[u32], _maximum: usize) -> Result<usize> {
        Ok(0)
    }
    fn supports_recompute_preemption(&self) -> bool {
        false
    }
    /// Only actual completed execution timing; caller logical clocks are not samples.
    fn completion_timing(&self, _ticket: &Self::Ticket) -> Option<ExecutionTiming> {
        None
    }
    /// # Errors
    /// Returns an unsupported or invalid-input error for incompatible backend, model, precision, kernels, or memory requirements.
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()>;
    /// Return the model-provider graph bound to this backend, without rebuilding its topology.
    /// # Errors
    /// Rejects a model different from the one loaded by this backend.
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph>;

    /// # Errors
    /// Returns an invalid-input, conflict, or capacity error for duplicate state or insufficient device storage.
    fn reserve_state(&mut self, _state: StateId, _capacity_tokens: usize) -> Result<()> {
        Ok(())
    }
    /// Prepare media-specific state pricing without speculative draft state.
    /// # Errors
    /// Returns invalid input or physical capacity errors.
    fn prepare_state_reservation_for_media(
        &mut self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<()> {
        self.prepare_state_reservation(capacity, readout)
    }
    /// Media-specific state reservation size without speculative draft state.
    /// # Errors
    /// Returns invalid input or physical capacity errors.
    fn state_reservation_bytes_for_media(
        &self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
        _media: &infer_ir::MediaPlan,
    ) -> Result<Option<u64>> {
        self.state_reservation_bytes_for(capacity, readout)
    }
    /// Reserve state for a media request without creating speculative draft state.
    /// # Errors
    /// Returns invalid input, ownership or physical capacity errors.
    fn reserve_state_for_media(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: infer_ir::OutputReadout,
        _media: &infer_ir::MediaPlan,
    ) -> Result<()> {
        self.reserve_state_for(state, capacity, readout)
    }

    /// Reserve persistent state matching the readout contract.
    /// # Errors
    /// Returns invalid input, ownership or physical capacity errors.
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        _readout: infer_ir::OutputReadout,
    ) -> Result<()> {
        self.reserve_state(state, capacity)
    }
    /// # Errors
    /// Returns a not-found or backend error if the sequence cannot be reset.
    fn reset_state(&mut self, _state: StateId) -> Result<()> {
        Ok(())
    }
    /// # Errors
    /// Returns a not-found, invariant, or backend error if owned state cannot be released.
    fn release_state(&mut self, _state: StateId) -> Result<()> {
        Ok(())
    }
    /// # Errors
    /// Returns a conflict error while execution is in flight or a backend/serialization error during capture.
    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }
    /// # Errors
    /// Returns an invariant error if sequence identities, cursors, or physical page leases disagree with runtime ownership.
    fn validate_state_ownership(&self, _states: &[(StateId, usize, usize)]) -> Result<()> {
        Ok(())
    }
    /// # Errors
    /// Returns a conflict, unsupported, invalid-input, or backend error if execution state cannot be validated and restored.
    fn restore_execution_state(&mut self, state: Option<&[u8]>) -> Result<()> {
        if state.is_some() {
            return Err(Error::unsupported("execution checkpoint not supported"));
        }
        Ok(())
    }
    /// # Errors
    /// Returns a validation, capacity, or backend error if tasks cannot be enqueued for the supplied program and step.
    /// An error must leave no device work in flight. Once device work is submitted,
    /// its ticket must be returned even if execution may later fail.
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket>;
    /// Transfer immutable batch metadata without copying its work arrays across owners.
    /// Direct providers may borrow it for synchronous encoding; asynchronous owners retain it.
    /// # Errors
    /// Follows the same no-unacknowledged-device-work contract as `submit`.
    fn submit_shared(
        &mut self,
        program: &ExecutionProgram,
        step: Arc<StepPlan>,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.submit(program, &step, tasks)
    }
    /// Borrow persistent task arrays during synchronous driver encoding.
    /// Compatibility providers copy metadata; native providers override this entry point.
    /// # Errors
    /// Follows `submit`: failure leaves no unacknowledged device readers.
    fn submit_borrowed(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<Self::Ticket> {
        self.submit(program, step, tasks.to_vec())
    }
    /// Publish shared work while retaining caller-owned task scratch.
    /// # Errors
    /// Follows the submission ownership contract.
    fn submit_shared_borrowed(
        &mut self,
        program: &ExecutionProgram,
        step: Arc<StepPlan>,
        tasks: &[ExecutionTask],
    ) -> Result<Self::Ticket> {
        self.submit_borrowed(program, &step, tasks)
    }
    /// Return consumed numerical storage to the device owner's persistent readback pool.
    /// # Errors
    /// Rejects a full recycler or invalid buffer ownership. Compatibility drivers simply release storage.
    fn recycle_output(&mut self, _state: StateId, _output: infer_ir::ModelOutput) -> Result<()> {
        Ok(())
    }
    /// Return the emptied completion descriptor array independently of numerical buffers.
    /// # Errors
    /// Rejects invalid recycler ownership/capacity.
    fn recycle_batch(&mut self, _outputs: Vec<TaskOutput>) -> Result<()> {
        Ok(())
    }
    /// Driver acceptance is distinct from publication. Pending acceptance still owns every lease.
    fn launch_accepted(&self, _ticket: &Self::Ticket) -> Option<bool> {
        Some(true)
    }
    /// None means an explicit in-flight wait. Failure is terminal for the ticket:
    /// neither success nor an error may be returned while device work still uses
    /// request-owned state. A slow command remains pending until completion.
    ///
    /// # Errors
    /// Returns a backend or invariant error for a failed command or a ticket that belongs to another backend.
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>>;
}
