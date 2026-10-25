// Each integration test compiles its own copy of this module, so whatever a given scene does not
// use is dead code in that target.
#![allow(
    dead_code,
    reason = "shared test source, included once per test target"
)]

//! The declared protocol double for runtime scenes.
//!
//! Scenes that exercise admission, completion identity, checkpointing or resource ownership need
//! an engine, not a transformer. This module supplies the three declarations such a scene needs -
//! a model descriptor, the operations it lowers to, and a backend that completes work with fixed
//! outputs - so the scene can stop depending on a full CPU inference executor.
//!
//! Nothing here executes a model. `ProtocolBackend` reports a planning sample rather than a
//! device, hands out fixed outputs, and reports no timing, because a backend that runs nothing
//! has no timing to report.
use infer_core::{DeviceId, Error, KernelId, ModelId, Result, StateId, map::BoundedMap};
use infer_ir::{
    BackboneKind, CapabilityRequirements, DType, DeviceCapabilities, ExecutionProgram,
    ExecutionTask, FeedForward, Head, Mixer, Modality, ModelIr, ModelOutput, Operation,
    OutputReadout, PositionSpec, PrecisionPlan, StateKind, StateRequirement, StepPlan, TaskOutput,
};
use infer_model_recipes::decoder::lower;
use infer_spi::{BackendProvider, KernelProvider, KernelRegistration, SourceLocation};
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A small decoder descriptor: two attention layers, hidden 8, intermediate 16, vocabulary 32.
#[must_use]
pub fn model(id: ModelId) -> ModelIr {
    const HIDDEN: usize = 8;
    const LAYERS: usize = 2;
    const KV_ELEMENTS_PER_HEAD: usize = 2;
    ModelIr {
        id,
        backbone: BackboneKind::Decoder,
        vocab_size: 32,
        hidden_size: HIDDEN,
        max_sequence: 128,
        mixers: vec![
            Mixer::Attention {
                query_heads: 1,
                kv_heads: 1,
                head_dim: HIDDEN,
                sliding_window: None,
                output_gate: false,
                qk_norm: false,
            };
            LAYERS
        ],
        feed_forward: FeedForward::Dense { intermediate: 16 },
        position: PositionSpec {
            rope_theta: 10_000.0,
            rotary_fraction: 1.0,
            multimodal_sections: vec![],
            interleaved: false,
        },
        norm_epsilon: 1e-5,
        norm_weight_offset: 0.0,
        heads: vec![
            Head::LanguageModel,
            Head::Embedding,
            Head::Rank,
            Head::Decision,
        ],
        modalities: vec![Modality::Text],
        state: (0..LAYERS)
            .map(|layer| StateRequirement {
                layer,
                kind: StateKind::AttentionKv,
                dtype: DType::F32,
                elements: HIDDEN * KV_ELEMENTS_PER_HEAD,
                per_token: true,
            })
            .collect(),
        tied_embeddings: false,
    }
}

/// The operations the descriptor above lowers to, declared so the engine can validate its plan.
pub struct DeclaredKernels;

const OPERATIONS: [Operation; 15] = [
    Operation::TokenEmbedding,
    Operation::RmsNorm,
    Operation::Attention,
    Operation::MatMul,
    Operation::Silu,
    Operation::Residual,
    Operation::LmHead,
    Operation::Pool,
    Operation::Rank,
    Operation::Decision,
    Operation::Split,
    Operation::Rope,
    Operation::Sigmoid,
    Operation::Multiply,
    Operation::GatedNorm,
];

impl KernelProvider for DeclaredKernels {
    fn kernels(&self) -> Vec<KernelRegistration> {
        OPERATIONS
            .into_iter()
            .enumerate()
            .map(|(index, operation)| KernelRegistration {
                backend: infer_ir::BackendKind::Cuda,
                id: KernelId::from_nonzero(
                    NonZeroU64::MIN.saturating_add(u64::try_from(index).unwrap_or(0)),
                ),
                operation,
                precision: PrecisionPlan::f32(),
                requirements: CapabilityRequirements {
                    compute_dtypes: vec![DType::F32],
                    ..CapabilityRequirements::default()
                },
                max_shape_elements: usize::MAX,
                workspace_bytes: 0,
                priority: 0,
                estimated_ns: 1,
                // Report the crate and file this module was compiled into, so a suite in another
                // crate that includes it does not claim to be the runtime's.
                source: SourceLocation {
                    crate_name: env!("CARGO_PKG_NAME").into(),
                    file: file!().into(),
                    function: "DeclaredKernels::kernels".into(),
                },
            })
            .collect()
    }
}

struct State {
    history: Vec<u32>,
    output_cursor: usize,
}

pub struct ProtocolBackend {
    host_sampling_draft_depth: usize,
    multimodal: bool,
    media_tasks: Arc<AtomicUsize>,
    media_reservations: Arc<AtomicUsize>,
    media_quotes: Arc<AtomicUsize>,
    normal_quotes: Arc<AtomicUsize>,
    quote_prices: Option<(u64, u64)>,
    identity: String,
    states: BoundedMap<StateId, State>,
    template: ModelOutput,
    resources: infer_spi::ResourcePool,
    hidden: usize,
    capacity: usize,
    script: Vec<usize>,
}

impl ProtocolBackend {
    /// # Errors
    /// Returns an invalid-input error when the configured request bound cannot be reserved.
    pub fn new(requests: usize, batch: usize, ir: &ModelIr) -> Result<Self> {
        Self::tagged("declared-protocol-double", requests, batch, ir)
    }

    /// The same double with a declared ramp that peaks at `token`.
    ///
    /// The default ramp peaks at the last vocabulary entry, which is the end-of-sequence token of
    /// some fixtures - a scene that counts generated tokens then stops after one. Declaring the peak
    /// states which token the sequence should continue with, so a token budget can be exercised
    /// without depending on a fixture's stop token.
    ///
    /// # Errors
    /// Returns an invalid-input error when the token is outside the vocabulary or the request bound
    /// cannot be reserved.
    pub fn peaking_at(token: usize, requests: usize, batch: usize, ir: &ModelIr) -> Result<Self> {
        if token >= ir.vocab_size {
            return Err(Error::invalid(format!(
                "declared peak token {token} is outside a vocabulary of {}",
                ir.vocab_size
            )));
        }
        let mut backend = Self::tagged("declared", requests, batch, ir)?;
        // A declared peak rather than a ramp: every other token is lower, so sampling's argmax is
        // the token the scene named no matter which index the sampler walks from.
        backend.template.logits = (0..ir.vocab_size)
            .map(|index| if index == token { 1.0 } else { 0.0 })
            .collect();
        Ok(backend)
    }

    /// A declared sequence of peaks, advanced once per logits readout for each request.
    pub fn scripted(tokens: Vec<usize>, ir: &ModelIr) -> Result<Self> {
        if tokens.iter().any(|&token| token >= ir.vocab_size) {
            return Err(Error::invalid("script token outside vocabulary"));
        }
        let mut backend = Self::new(16, 8, ir)?;
        backend.script = tokens;
        Ok(backend)
    }

    /// The same double with a declared weight tag.
    ///
    /// A snapshot records the backend's `identity` as its weights fingerprint, so a scene that
    /// checks a restore rejects different weights gives each side a different tag. The double
    /// declares that identity instead of hashing tensors it never loads.
    ///
    /// # Errors
    /// Returns an invalid-input error when the configured request bound cannot be reserved.
    pub fn tagged(tag: &str, requests: usize, batch: usize, ir: &ModelIr) -> Result<Self> {
        let _ = batch;
        // A declared ramp rather than a constant: sampling has to pick a token that is not the stop
        // token for scenes that count generated tokens, and the ramp makes that deterministic
        // instead of depending on which index sampling happens to choose.
        let mut logits = Vec::with_capacity(ir.vocab_size);
        let mut value = 0.0_f32;
        for _ in 0..ir.vocab_size {
            logits.push(value);
            value += 1.0;
        }
        Ok(Self {
            host_sampling_draft_depth: 0,
            multimodal: false,
            media_tasks: Arc::new(AtomicUsize::new(0)),
            media_reservations: Arc::new(AtomicUsize::new(0)),
            media_quotes: Arc::new(AtomicUsize::new(0)),
            normal_quotes: Arc::new(AtomicUsize::new(0)),
            quote_prices: None,
            identity: tag.to_owned(),
            states: BoundedMap::new(requests)?,
            template: ModelOutput {
                next_token: None,
                logits,
                hidden: vec![vec![0.5; ir.hidden_size]],
                tokens: Vec::with_capacity(8),
            },
            resources: infer_spi::ResourcePool::new(requests + 4)?,
            hidden: ir.hidden_size,
            capacity: requests,
            script: Vec::new(),
        })
    }
}

impl ProtocolBackend {
    /// Advertise a bound multimodal encoder to admission and scheduler scenes.
    #[must_use]
    pub const fn multimodal(mut self) -> Self {
        self.multimodal = true;
        self
    }
    /// Number of submitted tasks that carried a prepared media plan.
    #[must_use]
    pub fn media_tasks(&self) -> Arc<AtomicUsize> {
        self.media_tasks.clone()
    }
    /// Number of media-specific reservations that suppressed speculative state.
    #[must_use]
    pub fn media_reservations(&self) -> Arc<AtomicUsize> {
        self.media_reservations.clone()
    }
    /// Number of media-specific byte quotes requested by admission.
    #[must_use]
    pub fn media_quotes(&self) -> Arc<AtomicUsize> {
        self.media_quotes.clone()
    }
    /// Number of normal reservation-byte quotes requested by admission.
    #[must_use]
    pub fn normal_quotes(&self) -> Arc<AtomicUsize> {
        self.normal_quotes.clone()
    }
    /// Return distinct normal and media byte prices for quote-path scenes.
    #[must_use]
    pub const fn priced_quotes(mut self, normal: u64, media: u64) -> Self {
        self.quote_prices = Some((normal, media));
        self
    }
    /// Advertise a loaded draft while rejecting any task that permits backend sampling.
    pub const fn requiring_engine_sampling(mut self, draft_depth: usize) -> Self {
        self.host_sampling_draft_depth = draft_depth;
        self
    }
}

impl BackendProvider for ProtocolBackend {
    type Ticket = Option<Vec<TaskOutput>>;
    fn identity(&self) -> &str {
        &self.identity
    }
    fn supports_multimodal(&self) -> bool {
        self.multimodal
    }
    fn reserve_state_for_media(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.media_reservations.fetch_add(1, Ordering::Relaxed);
        self.reserve_state_for(state, capacity, readout)
    }
    fn state_reservation_bytes_for(
        &self,
        capacity: usize,
        _readout: OutputReadout,
    ) -> Result<Option<u64>> {
        if let Some((normal, _)) = self.quote_prices {
            self.normal_quotes.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(normal));
        }
        self.state_reservation_bytes(capacity)
    }
    fn state_reservation_bytes_for_media(
        &self,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<Option<u64>> {
        if let Some((_, media)) = self.quote_prices {
            self.media_quotes.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(media));
        }
        self.state_reservation_bytes_for(capacity, readout)
    }
    fn prepare_state_reservation_for_media(
        &mut self,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.media_quotes.fetch_add(1, Ordering::Relaxed);
        self.prepare_state_reservation(capacity, readout)
    }
    fn capabilities(&self) -> DeviceCapabilities {
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
            compute_dtypes: vec![DType::F32],
            memory_bytes: 0,
            unified_memory: true,
            profiling: false,
            speculation: infer_ir::SpeculationCapability {
                draft_depth: self.host_sampling_draft_depth,
                greedy_only: self.host_sampling_draft_depth > 0,
            },
        }
    }
    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        // The double's whole device state is the token history it has been given per state. That
        // history has to survive a checkpoint: the engine resumes from the cursor it committed, and
        // a restored double with an empty history rejects the next decode as a stale cursor.
        let history: Vec<(&StateId, &Vec<u32>)> = self
            .states
            .iter()
            .map(|(id, state)| (id, &state.history))
            .collect();
        serde_json::to_vec(&history)
            .map(Some)
            .map_err(|error| Error::invalid(error.to_string()))
    }
    fn restore_execution_state(&mut self, state: Option<&[u8]>) -> Result<()> {
        let history: Vec<(StateId, Vec<u32>)> = state.map_or_else(
            || Ok(Vec::new()),
            |bytes| {
                serde_json::from_slice(bytes).map_err(|error| Error::invalid(error.to_string()))
            },
        )?;
        if history.len() > self.capacity {
            return Err(Error::invalid(
                "checkpoint holds more states than the double granted",
            ));
        }
        self.states.clear();
        for (id, tokens) in history {
            self.states.insert(
                id,
                State {
                    history: tokens,
                    output_cursor: 0,
                },
            )?;
        }
        Ok(())
    }
    fn supports_recompute_preemption(&self) -> bool {
        // A scene that reclaims pages by recomputing needs the backend to allow it; the double has
        // no device state to lose, so it always can.
        true
    }
    fn supports_control_checkpoint(&self) -> bool {
        // The double holds fixed outputs and no device state, so a control checkpoint reconstructs
        // it exactly; scenes that checkpoint need this to be true.
        true
    }
    fn execution_graph(&self, ir: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        lower(ir)
    }
    fn validate_program(&self, _: &ModelIr, _: &ExecutionProgram) -> Result<()> {
        Ok(())
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
        _tokens: usize,
        _readout: OutputReadout,
    ) -> Result<()> {
        self.states.insert(
            state,
            State {
                history: Vec::new(),
                output_cursor: 0,
            },
        )?;
        Ok(())
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        // A reset starts the sequence over, so the token cursor the engine sends next must start
        // from zero again; keeping the old history rejects the next prefill as a stale cursor.
        self.states
            .get_mut(&state)
            .ok_or_else(|| Error::invalid("unknown state"))?
            .history
            .clear();
        Ok(())
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.states
            .remove(&state)
            .ok_or_else(|| Error::invariant("release state missing"))?;
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.submit_borrowed(program, step, &tasks)
    }
    fn submit_borrowed(
        &mut self,
        _: &ExecutionProgram,
        _: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<Self::Ticket> {
        // A declared output per task. The double does not lease buffers: the engine may submit the
        // next unit before it returns the previous output, which a buffer pool would have to
        // tolerate anyway.
        let mut batch = Vec::with_capacity(tasks.len());
        for task in tasks {
            if task.media.is_some() {
                self.media_tasks.fetch_add(1, Ordering::Relaxed);
            }
            if task.media.is_none()
                && self.host_sampling_draft_depth > 0
                && (task.plain_greedy || task.sampling.is_some())
            {
                return Err(Error::invariant(
                    "constrained request allowed backend token selection",
                ));
            }
            let state = self
                .states
                .get_mut(&task.state)
                .ok_or_else(|| Error::invariant("submit state missing"))?;
            // The output must carry one hidden row per token in the sequence, because the engine
            // projects a hidden output only when the rows match the context it planned for.
            task.tokens.commit(&mut state.history)?;
            let rows = state.history.len();
            let readout = task.tokens.readout();
            let mut output = self.template.clone();
            if readout == OutputReadout::Logits && !self.script.is_empty() {
                output.logits.fill(0.0);
                let token = self.script[state.output_cursor.min(self.script.len() - 1)];
                output.logits[token] = 10.0;
                state.output_cursor += 1;
            }
            if readout == OutputReadout::None {
                output.logits.clear();
            }
            output.hidden.resize(
                if readout == OutputReadout::Full {
                    rows
                } else {
                    0
                },
                vec![0.5; self.hidden],
            );
            batch.push(TaskOutput {
                request: task.request,
                output,
            });
        }
        Ok(Some(batch))
    }
    fn recycle_output(&mut self, _state: StateId, _output: ModelOutput) -> Result<()> {
        // The double allocates its declared outputs, so a returned buffer needs no bookkeeping.
        Ok(())
    }
    fn recycle_batch(&mut self, _outputs: Vec<TaskOutput>) -> Result<()> {
        Ok(())
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        Ok(ticket.take())
    }
}
