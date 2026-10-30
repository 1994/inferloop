#[cfg(metal_backend)]
use super::metal;
use super::{BackendChoice, Selection, cuda, cuda_available, metal_available, resolve};
use crate::constants::MIB_U64;
#[cfg(cuda_backend)]
use infer_backend_cuda::{
    executor::{CudaBackend, CudaTicket},
    registry::CudaKernels,
};
#[cfg(metal_backend)]
use infer_backend_metal::{MetalBackend, MetalKernels, MetalTicket};
use infer_core::{Error, Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionStats, ExecutionTask, ExecutionTiming,
    KvCacheInspection, LayerProbe, ModelIr, OpTrace, PageGrowth, StepPlan, TaskOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_spi::BackendProvider;
use std::path::Path;
use std::sync::Arc;

pub enum SelectedBackend {
    #[cfg(cuda_backend)]
    Cuda(Box<CudaBackend>),
    #[cfg(metal_backend)]
    Metal(Box<MetalBackend>),
}
pub enum SelectedTicket {
    #[cfg(cuda_backend)]
    Cuda(CudaTicket),
    #[cfg(metal_backend)]
    Metal(MetalTicket),
}
macro_rules! forward {
    ($s:expr,$method:ident $(,$arg:expr)*) => {match $s {
        #[cfg(cuda_backend)]
        Self::Cuda(b)=>b.$method($($arg),*),
        #[cfg(metal_backend)]
        Self::Metal(b)=>b.$method($($arg),*),
    }};
}
impl SelectedBackend {
    pub fn model_ir(&self) -> &ModelIr {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => b.model(),
            #[cfg(cuda_backend)]
            Self::Cuda(b) => b.model(),
        }
    }
    pub fn registry(&self) -> Result<KernelRegistry> {
        let mut registry = KernelRegistry::default();
        match self {
            #[cfg(metal_backend)]
            Self::Metal(_) => registry.register(&MetalKernels)?,
            #[cfg(cuda_backend)]
            Self::Cuda(_) => registry.register(&CudaKernels)?,
        }
        Ok(registry)
    }
    pub fn fresh(&self) -> Result<Self> {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => Ok(Self::Metal(Box::new(b.fresh()?))),
            #[cfg(cuda_backend)]
            Self::Cuda(_) => Err(Error::unsupported(
                "CUDA diagnostic fork requires independent state admission; not yet installed",
            )),
        }
    }
    #[cfg_attr(
        metal_backend,
        expect(
            clippy::unnecessary_wraps,
            reason = "The shared interface supports the test reference executor, which has no per-operation execution statistics"
        )
    )]
    #[cfg_attr(
        cuda_backend,
        expect(
            clippy::missing_const_for_fn,
            reason = "The shared backend interface invokes non-const mutable Metal/host diagnostics in other builds"
        )
    )]
    pub fn execution_stats(&self) -> Option<ExecutionStats> {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => Some(b.inspect()),
            #[cfg(cuda_backend)]
            Self::Cuda(_) => None,
        }
    }
    pub fn inspection(&self) -> serde_json::Value {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => {
                serde_json::json!({"kind":"metal-dataflow","device":b.device_name(),"state":b.inspect(),"capabilities":b.capabilities(),"weight_load":b.load_plan()})
            }
            #[cfg(cuda_backend)]
            Self::Cuda(b) => {
                serde_json::json!({"kind":"cuda-resident", "execution":"synchronous", "capabilities":b.capabilities(), "state_pool":b.pool_inspection(), "profile":b.execution_profile()})
            }
        }
    }
    #[cfg_attr(
        cuda_backend,
        expect(
            clippy::missing_const_for_fn,
            reason = "The shared backend interface invokes non-const mutable Metal/host diagnostics in other builds"
        )
    )]
    pub fn trace(&self) -> serde_json::Value {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => serde_json::json!(b.traces()),
            #[cfg(cuda_backend)]
            Self::Cuda(_) => serde_json::json!([]),
        }
    }
    #[cfg_attr(
        cuda_backend,
        expect(
            clippy::missing_const_for_fn,
            clippy::needless_pass_by_ref_mut,
            reason = "The shared backend interface invokes non-const mutable Metal/host diagnostics in other builds"
        )
    )]
    pub fn drain_layer_probes(&mut self) -> Vec<LayerProbe> {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => b.drain_layer_probes(),
            #[cfg(cuda_backend)]
            Self::Cuda(_) => vec![],
        }
    }
    #[cfg_attr(
        cuda_backend,
        expect(
            clippy::needless_pass_by_ref_mut,
            reason = "The shared backend interface invokes non-const mutable Metal/host diagnostics in other builds"
        )
    )]
    pub fn enable_layer_probes(&mut self, bytes: u64) -> Result<()> {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => b.enable_layer_probes(bytes),
            #[cfg(cuda_backend)]
            Self::Cuda(_) => {
                if bytes == 0 {
                    Ok(())
                } else {
                    Err(Error::unsupported("CUDA layer probes are not installed"))
                }
            }
        }
    }
    pub fn profile(&self) -> serde_json::Value {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => b.profile(),
            #[cfg(cuda_backend)]
            Self::Cuda(_) => {
                serde_json::json!({"scope":"CUDA per-operation profiling unavailable", "traceEvents":[]})
            }
        }
    }
}
impl infer_agent::AgentBackend for SelectedBackend {
    fn fresh(&self) -> Result<Self> {
        Self::fresh(self)
    }
    fn registry(&self) -> Result<KernelRegistry> {
        Self::registry(self)
    }
    fn inspection(&self) -> serde_json::Value {
        Self::inspection(self)
    }
    fn traces(&self) -> Vec<OpTrace> {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => b.traces().iter().cloned().collect(),
            #[cfg(cuda_backend)]
            Self::Cuda(_) => vec![],
        }
    }
    fn probes(&self) -> Vec<LayerProbe> {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => b.layer_probes().iter().cloned().collect(),
            #[cfg(cuda_backend)]
            Self::Cuda(_) => vec![],
        }
    }
    fn trace_timing_scope(&self) -> &'static str {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(_) => "cpu_encoding",
            #[cfg(cuda_backend)]
            Self::Cuda(_) => "unavailable",
        }
    }
    fn profile(&self) -> serde_json::Value {
        Self::profile(self)
    }
    fn execution_stats(&self) -> Option<ExecutionStats> {
        Self::execution_stats(self)
    }
}
impl BackendProvider for SelectedBackend {
    type Ticket = SelectedTicket;
    fn control_ready(&self) -> bool {
        forward!(self, control_ready)
    }
    fn maintenance(&mut self) -> Result<()> {
        forward!(self, maintenance)
    }
    fn state_recipe(&self) -> Option<&infer_ir::StateRecipe> {
        forward!(self, state_recipe)
    }
    fn begin_resource(
        &mut self,
        command: infer_spi::ResourceCommand,
    ) -> Result<infer_spi::ResourceTicket> {
        forward!(self, begin_resource, command)
    }
    fn recycle_output(&mut self, state: StateId, output: infer_ir::ModelOutput) -> Result<()> {
        forward!(self, recycle_output, state, output)
    }
    fn recycle_batch(&mut self, outputs: Vec<TaskOutput>) -> Result<()> {
        forward!(self, recycle_batch, outputs)
    }

    fn identity(&self) -> &str {
        forward!(self, identity)
    }
    fn weight_backed_dataflow(&self) -> bool {
        forward!(self, weight_backed_dataflow)
    }
    fn supports_multimodal(&self) -> bool {
        forward!(self, supports_multimodal)
    }
    fn image_preprocessing(&self) -> Option<Arc<dyn infer_spi::ImagePreprocessing>> {
        forward!(self, image_preprocessing)
    }
    fn execution_profile(&self) -> Option<infer_ir::ExecutionProfileInspection> {
        match self {
            #[cfg(cuda_backend)]
            Self::Cuda(b) => Some(b.execution_profile()),
            #[cfg(metal_backend)]
            Self::Metal(_) => None,
        }
    }
    fn capabilities(&self) -> DeviceCapabilities {
        forward!(self, capabilities)
    }
    /// Wrappers must forward every defaulted trait method: the trait's default silently
    /// reports the conservative answer, which once disabled speculative decode unnoticed.
    ///
    /// `submit_shared`/`submit_shared_borrowed` are deliberately not implemented here: their
    /// defaults route through this wrapper's own `submit`, which already wraps the ticket.
    /// `launch_accepted` needs a per-variant ticket conversion, so it keeps the trait default.
    fn requires_async_checkpoint(&self) -> bool {
        forward!(self, requires_async_checkpoint)
    }
    fn pending_resource_releases(&self) -> bool {
        forward!(self, pending_resource_releases)
    }
    fn resource_epoch(&self) -> u64 {
        forward!(self, resource_epoch)
    }
    fn tracks_reservation_intent(&self) -> bool {
        forward!(self, tracks_reservation_intent)
    }
    fn set_waker(&mut self, wake: Arc<dyn Fn() + Send + Sync>) {
        forward!(self, set_waker, wake);
    }

    /// Delegated explicitly: the trait default reports no speculation, which would silently
    /// disable the engine's speculative path even when the wrapped backend supports it.
    fn speculation_capability(&self) -> infer_ir::SpeculationCapability {
        forward!(self, speculation_capability)
    }
    fn prepare_state_reservation(
        &mut self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<()> {
        forward!(self, prepare_state_reservation, capacity, readout)
    }
    fn prepare_state_reservation_for_media(
        &mut self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<()> {
        forward!(self, prepare_state_reservation_for_media, capacity, readout)
    }
    fn state_reservation_bytes(&self, capacity: usize) -> Result<Option<u64>> {
        forward!(self, state_reservation_bytes, capacity)
    }
    fn state_reservation_bytes_for(
        &self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<Option<u64>> {
        forward!(self, state_reservation_bytes_for, capacity, readout)
    }
    fn state_reservation_bytes_for_media(
        &self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
        media: &infer_ir::MediaPlan,
    ) -> Result<Option<u64>> {
        forward!(
            self,
            state_reservation_bytes_for_media,
            capacity,
            readout,
            media
        )
    }
    fn state_capacity_bytes(&self) -> Result<Option<u64>> {
        forward!(self, state_capacity_bytes)
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        forward!(self, free_state_bytes)
    }
    fn kv_cache(&self) -> Option<KvCacheInspection> {
        forward!(self, kv_cache)
    }
    fn state_prefill_geometry(&self, s: StateId) -> Result<Option<infer_ir::PrefillGeometry>> {
        forward!(self, state_prefill_geometry, s)
    }
    fn state_page_growth(&self, s: StateId) -> Result<Option<PageGrowth>> {
        forward!(self, state_page_growth, s)
    }
    fn reusable_prefix(&self, t: &[u32], m: usize) -> usize {
        forward!(self, reusable_prefix, t, m)
    }
    fn reusable_prefix_for(&self, state: StateId, tokens: &[u32], maximum: usize) -> usize {
        forward!(self, reusable_prefix_for, state, tokens, maximum)
    }
    fn reusable_prefix_shared(
        &self,
        state: StateId,
        tokens: infer_ir::TokenBuffer,
        maximum: usize,
    ) -> usize {
        forward!(self, reusable_prefix_shared, state, tokens, maximum)
    }
    fn reuse_prefix_shared(
        &mut self,
        state: StateId,
        tokens: infer_ir::TokenBuffer,
        maximum: usize,
    ) -> Result<usize> {
        forward!(self, reuse_prefix_shared, state, tokens, maximum)
    }
    fn reuse_prefix(&mut self, s: StateId, t: &[u32], m: usize) -> Result<usize> {
        forward!(self, reuse_prefix, s, t, m)
    }
    fn supports_recompute_preemption(&self) -> bool {
        forward!(self, supports_recompute_preemption)
    }
    fn completion_timing(&self, t: &SelectedTicket) -> Option<ExecutionTiming> {
        match (self, t) {
            #[cfg(metal_backend)]
            (Self::Metal(b), SelectedTicket::Metal(t)) => b.completion_timing(t),
            #[cfg(cuda_backend)]
            (Self::Cuda(b), SelectedTicket::Cuda(t)) => b.completion_timing(t),
        }
    }
    fn supports_control_checkpoint(&self) -> bool {
        forward!(self, supports_control_checkpoint)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        forward!(self, execution_graph, model)
    }
    fn validate_program(&self, m: &ModelIr, p: &ExecutionProgram) -> Result<()> {
        forward!(self, validate_program, m, p)
    }
    fn reserve_state(&mut self, id: StateId, n: usize) -> Result<()> {
        forward!(self, reserve_state, id, n)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<()> {
        forward!(self, reserve_state_for, state, capacity, readout)
    }
    fn reserve_state_for_media(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: infer_ir::OutputReadout,
        media: &infer_ir::MediaPlan,
    ) -> Result<()> {
        forward!(
            self,
            reserve_state_for_media,
            state,
            capacity,
            readout,
            media
        )
    }
    fn reset_state(&mut self, id: StateId) -> Result<()> {
        forward!(self, reset_state, id)
    }
    fn release_state(&mut self, id: StateId) -> Result<()> {
        forward!(self, release_state, id)
    }
    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        forward!(self, capture_execution_state)
    }
    fn validate_state_ownership(&self, s: &[(StateId, usize, usize)]) -> Result<()> {
        forward!(self, validate_state_ownership, s)
    }
    fn restore_execution_state(&mut self, s: Option<&[u8]>) -> Result<()> {
        forward!(self, restore_execution_state, s)
    }
    fn submit(
        &mut self,
        p: &ExecutionProgram,
        s: &StepPlan,
        t: Vec<ExecutionTask>,
    ) -> Result<SelectedTicket> {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => Ok(SelectedTicket::Metal(b.submit(p, s, t)?)),
            #[cfg(cuda_backend)]
            Self::Cuda(b) => Ok(SelectedTicket::Cuda(b.submit(p, s, t)?)),
        }
    }
    fn submit_borrowed(
        &mut self,
        p: &ExecutionProgram,
        s: &StepPlan,
        t: &[ExecutionTask],
    ) -> Result<SelectedTicket> {
        match self {
            #[cfg(metal_backend)]
            Self::Metal(b) => Ok(SelectedTicket::Metal(b.submit_borrowed(p, s, t)?)),
            #[cfg(cuda_backend)]
            Self::Cuda(b) => Ok(SelectedTicket::Cuda(b.submit_borrowed(p, s, t)?)),
        }
    }
    fn poll(&mut self, t: &mut SelectedTicket) -> Result<Option<Vec<TaskOutput>>> {
        match (self, t) {
            #[cfg(metal_backend)]
            (Self::Metal(b), SelectedTicket::Metal(t)) => b.poll(t),
            #[cfg(cuda_backend)]
            (Self::Cuda(b), SelectedTicket::Cuda(t)) => b.poll(t),
        }
    }
}
pub fn load(path: &Path, memory_mib: u64, selection: &Selection) -> Result<SelectedBackend> {
    let choice = resolve(selection.kind, cuda_available(), metal_available())?;
    // Zero means "derive the budget from the device"; every backend resolves it locally.
    let memory_bytes = if memory_mib == 0 {
        0
    } else {
        memory_mib
            .checked_mul(MIB_U64)
            .ok_or_else(|| Error::invalid("device budget overflow"))?
    };
    match choice {
        BackendChoice::Cuda => cuda::load(path, memory_bytes, selection),
        #[cfg(metal_backend)]
        BackendChoice::Metal => metal::load(path, memory_bytes, selection),
        _ => Err(Error::unsupported(
            "Metal backend requires macOS and a supported Metal device",
        )),
    }
}

#[cfg(all(test, metal_backend))]
#[path = "../../tests/unit/backend_selected.rs"]
mod readout_tests;
