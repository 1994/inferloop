//! Node-major verification: consecutive projections reuse weights in cache.
//! Recurrent checkpoints stay on the device; KV rollback is a logical prefix.
//! Prefill width 32 keeps one batched arena: auxiliaries record once per node.
use super::{
    ActivationArena, ProgramWeights,
    capture::{Capture, CaptureMode, error},
    fp8_cache::Fp8Caches,
    profile,
    prompt_scratch::PromptScratch,
};
use crate::attention_profile::precise_prefix_positions;
use crate::constants::ATTENTION_SPLIT_THRESHOLD;
use crate::constants::AUX_KERNEL_TILE;
use crate::constants::CB_DECODE_SLOTS;
use crate::constants::EXTRA_WIDE_PREFILL_LANES;
use crate::constants::F32_BYTES;
use crate::constants::INACTIVE_LANE_STATE_POSITION;
use crate::constants::MAX_PREFILL_LANES;
use crate::constants::MAX_VERIFICATION_WIDTH;
use crate::constants::METADATA_FIELDS;
use crate::constants::MID_PREFILL_LANES;
use crate::constants::MTP_KV_OFFSET;
use crate::constants::PREFILL_LANES;
use crate::constants::PREFILL_SPLIT_K;
use crate::constants::WIDE_PREFILL_LANES;
use crate::constants::WIDER_PREFILL_LANES;
use crate::constants::WIDEST_PREFILL_LANES;
use crate::device::cublaslt::Support;
use crate::device::{CudaDevice, ReadbackSource, device_error};
use crate::device::{PinnedUpload, Readbacks};
use crate::mlp::ProjectionWeight;
use cutile::prelude::*;
use infer_core::{Error, ErrorCode, Result, TensorId};
use infer_ir::{DataflowGraph, StateKind, TensorNode, TensorOp, TensorStorage};
use infer_models::mrope::ROTARY_AXES;
use std::collections::BTreeMap;
use std::future::Future;
use std::mem::take;
use std::result::Result as StdResult;
use std::slice::from_ref;
use std::sync::Mutex;

pub type States = BTreeMap<TensorId, Vec<Tensor<f32>>>;
pub type BatchOutput = Vec<(Vec<f32>, Vec<f32>)>;

pub struct BatchReadout {
    pub rows: BatchOutput,
    pub tokens: Option<Vec<Option<u32>>>,
}
impl BatchReadout {
    pub(super) const fn host(rows: BatchOutput) -> Self {
        Self { rows, tokens: None }
    }
}

/// Per-lane embedding overrides and the rotary coordinates they run at.
#[derive(Clone, Copy)]
pub(super) struct PromptOverrides<'a> {
    /// `lanes[lane]` selects that lane's external embedding slot.
    pub lanes: &'a [bool],
    /// Per-lane `(t, h, w)`; absent means `[position + lane; 3]`.
    pub rotary: Option<&'a [[usize; ROTARY_AXES]]>,
}

/// Per-lane metadata plus the batched owner, the shared prompt sidecar and rotary table.
type PromptMetadata<'a> = (
    &'a [Tensor<i32>],
    &'a Tensor<i32>,
    &'a Tensor<i32>,
    Option<&'a Tensor<i32>>,
);

pub(super) struct Lane {
    pub arena: ActivationArena,
    metadata: Tensor<i32>,
    checkpoints: States,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PromptReadout {
    Hidden,
    Logits,
    #[cfg(test)]
    StateOnly,
}

/// Prompt readout variants captured against the same shared scratch.
struct PromptGraphs {
    prefill: CudaGraph<()>,
    prefill_last: CudaGraph<()>,
    #[cfg(test)]
    state_prefill: Option<CudaGraph<()>>,
}

struct PromptCaptures {
    regular: PromptGraphs,
    precise: Option<PromptGraphs>,
}

pub(super) struct ProgramCaptures {
    pub batch: Option<BatchGraph>,
    pub prompt_batch: Option<BatchGraph>,
    pub prompt_narrow: Option<BatchGraph>,
    pub forward: CudaGraph<()>,
    pub prefill: CudaGraph<()>,
}

pub(super) struct BatchGraph {
    graph: Option<CudaGraph<()>>,
    readbacks: Readbacks,
    token_readout: Option<super::token_readout::TokenReadout>,
    prefill: CudaGraph<()>,
    prefill_last: CudaGraph<()>,
    precise: Option<PromptGraphs>,
    #[cfg(test)]
    state_prefill: Option<CudaGraph<()>>,
    restore: Vec<CudaGraph<()>>,
    /// Split-K partial windows the captured fp4 prompt GEMMs accumulate into;
    /// declared after the graphs so their executables drop first.
    _partials: BTreeMap<usize, Tensor<f32>>,
    /// Final prompt row the one-row logits projection reads; captured by the prompt graphs.
    prompt_row: Option<Tensor<f32>>,
    _recurrent: super::recurrent_prefill::Workspace,
    _fusion: Option<super::fusion_batch::Workspace>,
    /// Shared prompt scratch the prompt graphs captured against.
    _prompt_scratch: Option<Arc<Mutex<PromptScratch>>>,
    metadata: Vec<Tensor<i32>>,
    /// Owner of the batched metadata rows when the prompt graph uses one allocation.
    metadata_owner: Option<Tensor<i32>>,
    /// Prompt replay is awaited before either pinned staging owner is reused.
    metadata_upload: Vec<i32>,
    metadata_staging: PinnedUpload<i32>,
    rotary_staging: PinnedUpload<i32>,
    prefill_info: Option<Tensor<i32>>,
    /// Per-lane `(t, h, w)` rotary coordinates for prompt chunks; absent below prefill width 32.
    prefill_rotary: Option<Tensor<i32>>,
    hidden: Vec<Arc<Tensor<f32>>>,
    logits: Vec<Arc<Tensor<f32>>>,
    pub width: usize,
    /// Token capacity of the KV arena this graph's kernels address. Positions are rows in it,
    /// so a chunk that would write past the end is an internal inconsistency, not a request
    /// the engine may reject later.
    pub capacity: usize,
    /// Positions the state writes lag the `RoPE` position by, matching the MTP KV offset.
    state_offset: i32,
    /// Optional in-graph boundary events (`INFER_CUDA_PREFILL_PROFILE`); declared after
    /// the graphs so their executables are destroyed before the events they record.
    profile: Option<profile::PrefillProfile>,
    precise_profile: Option<profile::PrefillProfile>,
    // Batched width-32 arena anchors every slot allocation past graph replays;
    // declared last so it drops after the graphs that reference its rows.
    _arena: Option<ActivationArena>,
}

pub(super) struct BatchBuilder<'a> {
    pub device: &'a CudaDevice,
    pub graph: &'a DataflowGraph,
    pub weights: &'a ProgramWeights,
    pub nvfp4: &'a mut super::nvfp4_gemm::Workspace,
    pub delegated: Option<&'a mut Support>,
    pub attention: &'a mut super::attention_decode::Workspace,
    pub states: &'a mut States,
    pub fp8_states: &'a mut Fp8Caches,
    pub external: &'a Tensor<f32>,
    pub lane_external: &'a mut Vec<Tensor<f32>>,
    /// Contiguous storage behind `lane_external`, one external hidden row per prompt lane.
    pub lane_external_owner: Option<&'a Tensor<f32>>,
    pub lane_fusion: &'a mut Vec<Option<super::program::FusionWorkspace>>,
    /// One block table per lane, indexed exactly like `lane_external`; batched and prefill
    /// captures use entry 0 because their operators do not resolve KV positions.
    pub tables: &'a [Tensor<i32>],
    pub capacity: usize,
    pub width: usize,
}

/// Decode-only slot graph: lane `i` reads and writes the slot pool's `i`-th state set,
/// so one replay advances every leased sequence through the same weight pass.
///
/// Inactive lanes are masked with a `-1` state position (the state kernels gate writes on
/// it); slot tenants never rewind, so no rollback checkpoints are captured.
pub(super) struct SlotDecodeGraph {
    graph: CudaGraph<()>,
    state_only: Option<CudaGraph<()>>,
    readbacks: Readbacks,
    token_readout: Option<super::token_readout::TokenReadout>,
    metadata_owner: Tensor<i32>,
    metadata_upload: Vec<i32>,
    metadata_staging: PinnedUpload<i32>,
    // Batched hidden rows stay allocated for the graph's writes but are never read back.
    hidden: Arc<Tensor<f32>>,
    pub(super) external_hidden: bool,
    logits: Arc<Tensor<f32>>,
    pub width: usize,
    /// Host-side state position each slot's next replay must continue at.
    cursors: Vec<usize>,
    /// Optional in-graph boundary events (`INFER_CUDA_PREFILL_PROFILE`); declared after
    /// the graph so its executable is destroyed before the events it records.
    profile: Option<profile::SlotProfile>,
    // The shared batched arena anchors the graph's activations; declared last so the
    // graph drops first.
    _fusion: Option<super::fusion_batch::Workspace>,
    _arena: ActivationArena,
}

impl BatchBuilder<'_> {
    /// Batch graph plus every prompt graph the program captures, widest last.
    ///
    /// A prompt chunk costs one fixed-width replay whatever its token count, so a program may
    /// capture a second, narrower prompt graph and route short chunks to it.
    /// Capture the verification graph (unless the sequence will speculate through the
    /// shared slot pool) plus whichever prompt graphs the configured widths select.
    /// Pool-bound geometry omits the wide prompt arena when a narrow prompt graph is available.
    pub fn build_pair(
        &mut self,
        geometry: super::program::ProgramGeometry,
    ) -> Result<(Option<BatchGraph>, Option<BatchGraph>, Option<BatchGraph>)> {
        if self.weights.batch_width > MAX_VERIFICATION_WIDTH {
            return Err(Error::invalid("verification width exceeds 9"));
        }
        let batch = if geometry.verification > 1 {
            self.width = geometry.verification;
            Some(self.build()?)
        } else {
            None
        };
        let wide = self.weights.prefill_width;
        let narrow = geometry.prompts[1];
        if wide < PREFILL_LANES && wide > 1 && wide != self.weights.batch_width {
            return Err(Error::invalid(
                "prefill width must match verification width or be 32",
            ));
        }
        let prompt_narrow = if geometry.prompts[1] > 0 {
            self.width = narrow;
            Some(self.build()?)
        } else {
            None
        };
        let prompt = if geometry.prompts[0] > 0 {
            self.width = geometry.prompts[0];
            Some(self.build()?)
        } else {
            None
        };
        Ok((batch, prompt, prompt_narrow))
    }

    /// State positions a fused program writes, relative to its `RoPE` positions.
    fn state_offset(weights: &ProgramWeights) -> i32 {
        if weights.fusion.is_some() {
            -i32::try_from(MTP_KV_OFFSET).unwrap_or(0)
        } else {
            0
        }
    }

    /// Decode-only graph over per-slot state sets on one shared batched arena: Linear
    /// nodes read and write their `[width, size]` slots in place (no pack/unpack), pure
    /// activation nodes record once over every row, and state or metadata kernels record
    /// per lane against `lane_states[lane]`/`lane_fp8[lane]`. The builder's own
    /// `states`/`fp8_states` are placeholders
    /// and stay untouched.
    /// # Errors
    /// Rejects mismatched lane slices, a width outside `2..=CB_DECODE_SLOTS` or CUDA
    /// capture failures.
    pub fn build_slots(
        &mut self,
        lane_states: &mut [States],
        lane_fp8: &mut [Fp8Caches],
    ) -> Result<SlotDecodeGraph> {
        let width = lane_states.len();
        if !(2..=CB_DECODE_SLOTS).contains(&width)
            || lane_fp8.len() != width
            || self.lane_external.len() < width
            || self.lane_fusion.len() < width
        {
            return Err(Error::invalid("slot decode lanes"));
        }
        let mut arena = slot_arena(self.device, self.graph, width)?;
        let (metadata_owner, metadata) = batched_metadata(self.device, width)?;
        let mut profile = profile::SlotProfile::from_env(self.device);
        let mut fusion =
            super::fusion_batch::Workspace::allocate(self.device, self.weights, width)?;
        let graph = self.capture_slots(
            &mut arena,
            (&metadata, &metadata_owner, &mut fusion),
            (lane_states, lane_fp8),
            &mut profile,
            true,
        )?;
        let state_only = if self.weights.fusion.is_some() {
            Some(self.capture_slots(
                &mut arena,
                (&metadata, &metadata_owner, &mut fusion),
                (lane_states, lane_fp8),
                &mut profile,
                false,
            )?)
        } else {
            None
        };
        let hidden = take_result(&mut arena, self.graph.hidden)?;
        let logits = take_result(&mut arena, self.graph.logits)?;
        let token_readout = super::token_readout::TokenReadout::contiguous(
            self.device,
            &logits,
            width,
            self.weights,
        )?;
        Ok(SlotDecodeGraph {
            token_readout,
            graph,
            state_only,
            readbacks: Readbacks::contiguous(self.device, &hidden, &logits, width)?,
            metadata_owner,
            metadata_upload: Vec::with_capacity(width * METADATA_FIELDS),
            metadata_staging: PinnedUpload::chain(self.device, width * METADATA_FIELDS)?,
            hidden,
            external_hidden: self.weights.fusion.is_some(),
            logits,
            width,
            cursors: vec![0; width],
            profile,
            _fusion: fusion,
            _arena: arena,
        })
    }

    fn capture_slots(
        &mut self,
        arena: &mut ActivationArena,
        metadata: (
            &[Tensor<i32>],
            &Tensor<i32>,
            &mut Option<super::fusion_batch::Workspace>,
        ),
        states: (&mut [States], &mut [Fp8Caches]),
        profile: &mut Option<profile::SlotProfile>,
        read_logits: bool,
    ) -> Result<CudaGraph<()>> {
        let (metadata, metadata_owner, fusion) = metadata;
        let (lane_states, lane_fp8) = states;
        let width = lane_states.len();
        let _pool =
            super::batch_projection::allocate(self.device, self.graph, width, self.weights)?;
        let quant_plan = super::nvfp4_gemm::QuantPairPlan::new(self.graph, self.weights);
        self.nvfp4.begin_quant_capture();
        let result = CudaGraph::scope(&self.device.stream, |scope| {
            let mut no_fusion = None;
            for (index, node) in self.graph.nodes.iter().enumerate() {
                self.nvfp4.enter_quant_node(index, quant_plan.role(index));
                if !read_logits && node.outputs.iter().any(|id| Some(*id) == self.graph.logits) {
                    self.nvfp4.clear_quant_reuse();
                    continue;
                }
                if read_logits && let Some(profile) = profile.as_mut() {
                    profile.graph().boundary(scope, index, node)?;
                }
                if let Some(workspace) = fusion.as_mut()
                    && super::fusion_batch::record(
                        self,
                        scope,
                        node,
                        arena,
                        metadata,
                        metadata_owner,
                        workspace,
                    )?
                {
                    continue;
                }
                if node.op == TensorOp::Linear {
                    super::batch_projection::record_slots(
                        scope,
                        node,
                        arena,
                        self.weights,
                        self.nvfp4,
                    )?;
                    continue;
                }
                match dispatch32(node, self.weights) {
                    Dispatch32::Linear => return Err(error("slot decode linear dispatch")),
                    Dispatch32::Batched => Capture {
                        scope,
                        arena,
                        weights: self.weights,
                        nvfp4: self.nvfp4,
                        attention: self.attention,
                        states: self.states,
                        fp8_states: self.fp8_states,
                        metadata: &metadata[0],
                        table: &self.tables[0],
                        external: self.external,
                        capacity: self.capacity,
                        fusion: &mut no_fusion,
                        mode: CaptureMode::Pooled,
                    }
                    .record(node)?,
                    Dispatch32::Row => {
                        for lane in 0..width {
                            Capture {
                                scope,
                                arena,
                                weights: self.weights,
                                nvfp4: self.nvfp4,
                                attention: self.attention,
                                states: &mut lane_states[lane],
                                fp8_states: &mut lane_fp8[lane],
                                metadata: &metadata[lane],
                                table: &self.tables[lane],
                                external: &self.lane_external[lane],
                                capacity: self.capacity,
                                fusion: &mut self.lane_fusion[lane],
                                mode: CaptureMode::Row(lane),
                            }
                            .record_row(node, lane)?;
                        }
                    }
                }
            }
            if read_logits && let Some(profile) = profile.as_mut() {
                profile.graph().finish(scope)?;
            }
            Ok(())
        });
        self.nvfp4.end_quant_capture();
        result.map_err(device_error)
    }

    pub(super) fn build_program(
        &mut self,
        geometry: super::program::ProgramGeometry,
        arena: &mut ActivationArena,
        metadata: &Tensor<i32>,
        fusion: &mut Option<super::program::FusionWorkspace>,
        forward_captured: bool,
    ) -> Result<ProgramCaptures> {
        let (batch, prompt_batch, prompt_narrow) = self.build_pair(geometry)?;
        let forward = if forward_captured {
            self.capture_single(arena, metadata, fusion, false)?
        } else {
            CudaGraph::scope(&self.device.stream, |_| Ok(())).map_err(device_error)?
        };
        let prefill = if forward_captured {
            self.capture_single(arena, metadata, fusion, true)?
        } else {
            CudaGraph::scope(&self.device.stream, |_| Ok(())).map_err(device_error)?
        };
        Ok(ProgramCaptures {
            batch,
            prompt_batch,
            prompt_narrow,
            forward,
            prefill,
        })
    }

    pub(super) fn capture_single(
        &mut self,
        arena: &mut ActivationArena,
        metadata: &Tensor<i32>,
        fusion: &mut Option<super::program::FusionWorkspace>,
        skip_logits: bool,
    ) -> Result<CudaGraph<()>> {
        CudaGraph::scope(&self.device.stream, |scope| {
            Capture {
                scope,
                arena,
                weights: self.weights,
                nvfp4: self.nvfp4,
                attention: self.attention,
                states: self.states,
                fp8_states: self.fp8_states,
                metadata,
                table: &self.tables[0],
                external: self.external,
                capacity: self.capacity,
                fusion,
                mode: CaptureMode::Flat,
            }
            .record_program(self.graph, skip_logits)
        })
        .map_err(device_error)
    }

    pub fn build(&mut self) -> Result<BatchGraph> {
        let width = self.width;
        if !(2..=MAX_VERIFICATION_WIDTH).contains(&width)
            && ![
                PREFILL_LANES,
                MID_PREFILL_LANES,
                MAX_PREFILL_LANES,
                WIDE_PREFILL_LANES,
                EXTRA_WIDE_PREFILL_LANES,
                WIDER_PREFILL_LANES,
                WIDEST_PREFILL_LANES,
            ]
            .contains(&width)
        {
            return Err(Error::invalid("verification width"));
        }
        if width >= PREFILL_LANES {
            return self.build32();
        }
        // Verification records stateless nodes once over a batched arena whose rows the lanes
        // alias; per-lane stateful nodes and the prompt captures keep their lane views.
        let mut shared = Some(ActivationArena::new_batched(
            self.device,
            self.graph,
            width,
            usize::try_from(self.device.profile()?.arena_budget_bytes()).map_err(device_error)?,
        )?);
        let mut lanes = self.allocate_lanes(width, shared.as_ref())?;
        let mut projections =
            super::batch_projection::allocate(self.device, self.graph, width, self.weights)?;
        let graph = Some(self.capture(&mut lanes, &mut projections, None, shared.as_mut())?);
        let prefill = self.capture(&mut lanes, &mut projections, Some(false), None)?;
        let prefill_last = self.capture(&mut lanes, &mut projections, Some(true), None)?;
        let mut restore = Vec::new();
        let mut metadata = Vec::new();
        let mut hidden = Vec::new();
        let mut logits = Vec::new();
        for mut lane in lanes {
            restore.push(
                CudaGraph::scope(&self.device.stream, |scope| {
                    for (id, tensors) in &lane.checkpoints {
                        for (output, input) in self
                            .states
                            .get_mut(id)
                            .ok_or_else(|| error("checkpoint state"))?
                            .iter_mut()
                            .zip(tensors)
                        {
                            scope.record(api::memcpy(output, input))?;
                        }
                    }
                    Ok(())
                })
                .map_err(device_error)?,
            );
            hidden.push(take_result(&mut lane.arena, self.graph.hidden)?);
            logits.push(take_result(&mut lane.arena, self.graph.logits)?);
            metadata.push(lane.metadata);
        }
        let token_readout = super::token_readout::TokenReadout::new(
            self.device,
            &logits,
            width,
            logits[0].size(),
            self.weights,
        )?;
        Ok(BatchGraph {
            token_readout,
            graph,
            state_offset: Self::state_offset(self.weights),
            readbacks: Readbacks::pairs(self.device, &hidden, &logits)?,
            prefill,
            prefill_last,
            precise: None,
            #[cfg(test)]
            state_prefill: None,
            restore,
            _partials: BTreeMap::new(),
            prompt_row: None,
            _recurrent: super::recurrent_prefill::Workspace::default(),
            _fusion: None,
            _prompt_scratch: None,
            prefill_info: None,
            prefill_rotary: None,
            metadata,
            metadata_owner: None,
            metadata_upload: Vec::with_capacity(self.width * METADATA_FIELDS),
            metadata_staging: PinnedUpload::single(self.device, self.width * METADATA_FIELDS)?,
            rotary_staging: PinnedUpload::single(self.device, self.width * ROTARY_AXES)?,
            hidden,
            logits,
            width,
            capacity: self.capacity,
            profile: None,
            precise_profile: None,
            _arena: shared,
        })
    }

    /// Prefill width 32: one batched arena, one metadata tensor per lane.
    /// The prompt arena, with the one-row logits buffer its final chunk projects when the
    /// vocabulary projection supports it.
    fn prompt_arena(&self) -> Result<(ActivationArena, Option<Tensor<f32>>)> {
        let budget =
            usize::try_from(self.device.profile()?.arena_budget_bytes()).map_err(device_error)?;
        if !super::prompt_logits::supported(self.graph, self.weights) {
            let arena = ActivationArena::new_batched(self.device, self.graph, self.width, budget)?;
            return Ok((arena, None));
        }
        let arena = ActivationArena::new_prompt(self.device, self.graph, self.width, budget)?;
        let row = api::zeros::<f32>(&[1, self.external.size()])
            .sync_on(&self.device.stream)
            .map_err(device_error)?;
        Ok((arena, Some(row)))
    }

    fn build32(&mut self) -> Result<BatchGraph> {
        #[cfg(test)]
        self.attention.taps.prepare(
            self.device,
            self.graph,
            self.weights,
            self.width,
            self.capacity,
        )?;
        self.attention
            .prepare_prefill(self.device, self.graph, self.weights, self.width);
        let scratch = self
            .weights
            .prompt_scratch
            .get_or_allocate(self.width, || self.allocate_prompt_scratch())?;
        let mut shared = scratch
            .lock()
            .map_err(|_| Error::invariant("prompt scratch poisoned"))?;
        let (mut arena, mut prompt_row) = shared_prompt_views(&shared)?;
        let (metadata_owner, metadata) = batched_metadata(self.device, self.width)?;
        let prefill_info = api::zeros::<i32>(&[METADATA_FIELDS])
            .sync_on(&self.device.stream)
            .map_err(device_error)?;
        let prefill_rotary = self.prompt_rotary()?;
        let mut profile = profile::PrefillProfile::from_env(self.device);
        let mut precise_profile =
            if self.weights.fusion.is_none() && self.weights.attention_profile.is_bf16() {
                profile::PrefillProfile::from_env(self.device)
            } else {
                None
            };
        let captures = self.capture_prompt_readouts(
            &mut arena,
            (
                metadata.as_slice(),
                &metadata_owner,
                &prefill_info,
                prefill_rotary.as_ref(),
            ),
            (&mut profile, &mut precise_profile),
            &mut shared,
            prompt_row.as_mut(),
        )?;
        drop(shared);
        let hidden = vec![take_result(&mut arena, self.graph.hidden)?];
        let logits = vec![take_result(&mut arena, self.graph.logits)?];
        Ok(BatchGraph {
            token_readout: None,
            graph: None,
            state_offset: Self::state_offset(self.weights),
            readbacks: Readbacks::pairs(self.device, &hidden, &logits)?,
            prefill: captures.regular.prefill,
            prefill_last: captures.regular.prefill_last,
            precise: captures.precise,
            #[cfg(test)]
            state_prefill: captures.regular.state_prefill,
            restore: Vec::new(),
            _partials: BTreeMap::new(),
            prompt_row,
            _recurrent: super::recurrent_prefill::Workspace::default(),
            _fusion: None,
            _prompt_scratch: Some(Arc::clone(&scratch)),
            prefill_info: Some(prefill_info),
            prefill_rotary,
            metadata,
            metadata_owner: Some(metadata_owner),
            metadata_upload: Vec::with_capacity(self.width * METADATA_FIELDS),
            metadata_staging: PinnedUpload::single(self.device, self.width * METADATA_FIELDS)?,
            rotary_staging: PinnedUpload::single(self.device, self.width * ROTARY_AXES)?,
            hidden,
            logits,
            width: self.width,
            capacity: self.capacity,
            profile,
            precise_profile,
            _arena: Some(arena),
        })
    }

    fn capture_prompt_readouts(
        &mut self,
        arena: &mut ActivationArena,
        metadata: PromptMetadata<'_>,
        profiles: (
            &mut Option<profile::PrefillProfile>,
            &mut Option<profile::PrefillProfile>,
        ),
        scratch: &mut PromptScratch,
        mut prompt_row: Option<&mut Tensor<f32>>,
    ) -> Result<PromptCaptures> {
        let (profile, precise_profile) = profiles;
        let regular = self.capture_prompt_graphs(
            arena,
            metadata,
            profile,
            scratch,
            prompt_row.as_deref_mut(),
        )?;
        // Both variants retain the same arenas, state and metadata. A configured canonical
        // prefix needs its own graphs; without one the variant is never replayed.
        let precise = if self.weights.fusion.is_none()
            && self.weights.attention_profile.is_bf16()
            && precise_prefix_positions() > 0
        {
            self.attention.precise_prefill(true);
            let result =
                self.capture_prompt_graphs(arena, metadata, precise_profile, scratch, prompt_row);
            self.attention.precise_prefill(false);
            Some(result?)
        } else {
            None
        };
        Ok(PromptCaptures { regular, precise })
    }

    fn capture_prompt_graphs(
        &mut self,
        arena: &mut ActivationArena,
        metadata: PromptMetadata<'_>,
        profile: &mut Option<profile::PrefillProfile>,
        scratch: &mut PromptScratch,
        mut prompt_row: Option<&mut Tensor<f32>>,
    ) -> Result<PromptGraphs> {
        let PromptScratch {
            partials,
            recurrent,
            fusion,
            ..
        } = scratch;
        let prefill = self.capture32(
            arena,
            metadata,
            PromptReadout::Hidden,
            profile.as_mut(),
            (partials, recurrent, fusion),
            prompt_row.as_deref_mut(),
        )?;
        #[cfg(test)]
        let state_prefill = if super::prefill_demand::enabled() {
            Some(self.capture32(
                arena,
                metadata,
                PromptReadout::StateOnly,
                None,
                (partials, recurrent, fusion),
                prompt_row.as_deref_mut(),
            )?)
        } else {
            None
        };
        let prefill_last = self.capture32(
            arena,
            metadata,
            PromptReadout::Logits,
            profile.as_mut(),
            (partials, recurrent, fusion),
            prompt_row,
        )?;
        Ok(PromptGraphs {
            prefill,
            prefill_last,
            #[cfg(test)]
            state_prefill,
        })
    }

    /// Only three-axis models need the per-lane table; text-only prompts keep the cheaper
    /// `prefill_info` path and avoid an upload per chunk.
    fn prompt_rotary(&self) -> Result<Option<Tensor<i32>>> {
        if !self.weights.mrope {
            return Ok(None);
        }
        Ok(Some(
            api::zeros::<i32>(&[self.width * ROTARY_AXES])
                .sync_on(&self.device.stream)
                .map_err(device_error)?,
        ))
    }

    /// Prompt scratch for this width, allocated once per program and shared by its sequences.
    fn allocate_prompt_scratch(&self) -> Result<PromptScratch> {
        let (arena, row) = self.prompt_arena()?;
        Ok(PromptScratch {
            arena,
            row,
            partials: self.allocate_prompt_partials()?,
            recurrent: super::recurrent_prefill::Workspace::new(
                self.device,
                self.graph,
                self.weights,
                self.width,
            )?,
            fusion: super::fusion_batch::Workspace::allocate(
                self.device,
                self.weights,
                self.width,
            )?,
        })
    }

    fn allocate_prompt_partials(&self) -> Result<BTreeMap<usize, Tensor<f32>>> {
        let mut partials = BTreeMap::new();
        for node in &self.graph.nodes {
            if node.op != TensorOp::Linear
                || self.weights.input_scales.contains_key(&node.inputs[1])
                || !matches!(
                    self.weights.projections.get(&node.inputs[1]),
                    Some(ProjectionWeight::Fp4(..))
                )
            {
                continue;
            }
            let rows = self
                .graph
                .tensors
                .iter()
                .find(|tensor| tensor.id == node.outputs[0])
                .ok_or_else(|| Error::invalid("prefill linear output"))?
                .elements()?;
            if partials.contains_key(&rows) {
                continue;
            }
            partials.insert(
                rows,
                api::zeros::<f32>(&[PREFILL_SPLIT_K * self.width, rows])
                    .sync_on(&self.device.stream)
                    .map_err(device_error)?,
            );
        }
        Ok(partials)
    }

    /// Record a verification node once over the batched arena the lanes alias: projections read
    /// and write every lane's row in place, and stateless nodes cover all rows in one launch.
    /// False leaves the node to the per-lane path.
    fn record_shared(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        metadata: &Tensor<i32>,
    ) -> StdResult<bool, DeviceError> {
        match dispatch32(node, self.weights) {
            Dispatch32::Linear if !self.weights.constants.contains_key(&node.inputs[0]) => {
                super::batch_projection::record_slots(
                    scope,
                    node,
                    arena,
                    self.weights,
                    self.nvfp4,
                )?;
            }
            Dispatch32::Batched => {
                let mut no_fusion = None;
                Capture {
                    scope,
                    arena,
                    weights: self.weights,
                    nvfp4: self.nvfp4,
                    attention: self.attention,
                    states: self.states,
                    fp8_states: self.fp8_states,
                    metadata,
                    table: &self.tables[0],
                    external: self.external,
                    capacity: self.capacity,
                    fusion: &mut no_fusion,
                    mode: CaptureMode::Batched,
                }
                .record(node)?;
            }
            Dispatch32::Linear | Dispatch32::Row => return Ok(false),
        }
        Ok(true)
    }

    /// Verification lanes append their rows, then attend in one grouped launch that reads each
    /// KV block once for all three lanes. False when this node keeps the per-lane path.
    fn record_verify_attention(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        lanes: &mut [Lane],
        prefill: Option<bool>,
    ) -> StdResult<bool, DeviceError> {
        let width = lanes.len();
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return Ok(false);
        };
        let Some(&id) = node.states.first() else {
            return Ok(false);
        };
        let eligible = prefill.is_none()
            && width == super::attention_decode::VERIFY_LANES
            && window.is_none_or(|w| w >= ATTENTION_SPLIT_THRESHOLD)
            && self.attention.supports_verify(query_heads, head_dim)
            && !self.weights.constants.contains_key(&node.inputs[0])
            && self.fp8_states.contains_key(&id)
            && lanes.iter().all(|lane| !lane.checkpoints.contains_key(&id));
        if !eligible {
            return Ok(false);
        }
        for (index, lane) in lanes.iter_mut().enumerate() {
            Capture {
                scope,
                arena: &mut lane.arena,
                weights: self.weights,
                nvfp4: self.nvfp4,
                attention: self.attention,
                states: self.states,
                fp8_states: self.fp8_states,
                metadata: &lane.metadata,
                table: &self.tables[0],
                external: &self.lane_external[index],
                capacity: self.capacity,
                fusion: &mut self.lane_fusion[index],
                mode: CaptureMode::Flat,
            }
            .record_attention_append(node)?;
        }
        let (keys, values) = self
            .fp8_states
            .remove(&id)
            .ok_or_else(|| error("verification FP8 state"))?;
        let mut outputs = Vec::with_capacity(width);
        for lane in lanes.iter_mut() {
            let slot = lane.arena.slot(node.outputs[0]).map_err(error)?;
            let output = lane.arena.buffers[slot]
                .take()
                .ok_or_else(|| error("verification attention output"))?;
            let shape = output
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim).map_err(error))
                .collect::<StdResult<Vec<_>, _>>()?;
            outputs.push((slot, shape, output.reshape(&[query_heads, head_dim])?));
        }
        let recorded = {
            let queries = lanes
                .iter()
                .map(|lane| {
                    lane.arena
                        .get(node.inputs[0])
                        .map_err(error)?
                        .view(&[query_heads, head_dim])
                        .map_err(error)
                })
                .collect::<StdResult<Vec<_>, DeviceError>>()?;
            let [first, second, third] = outputs.as_mut_slice() else {
                return Err(error("verification lanes"));
            };
            self.attention.record_verify(
                scope,
                [&mut first.2, &mut second.2, &mut third.2],
                [&queries[0], &queries[1], &queries[2]],
                [&lanes[0].metadata, &lanes[1].metadata, &lanes[2].metadata],
                (&keys, &values, &self.tables[0]),
                (query_heads, kv_heads, head_dim),
                i32::try_from(window.unwrap_or(0)).map_err(error)?,
                self.weights.kv_scales[&id],
            )
        };
        self.fp8_states.insert(id, (keys, values));
        for ((slot, shape, output), lane) in outputs.into_iter().zip(lanes.iter_mut()) {
            lane.arena.buffers[slot] = Some(output.reshape(&shape)?);
        }
        recorded?;
        Ok(true)
    }

    fn capture(
        &mut self,
        lanes: &mut [Lane],
        projections: &mut super::batch_projection::Workspace,
        prefill: Option<bool>,
        mut shared: Option<&mut ActivationArena>,
    ) -> Result<CudaGraph<()>> {
        let width = lanes.len();
        let device = self.device;
        self.nvfp4.end_quant_capture();
        let result = CudaGraph::scope(&device.stream, |scope| {
            for node in &self.graph.nodes {
                let logits = node.outputs.iter().any(|id| Some(*id) == self.graph.logits);
                if logits && prefill == Some(false) {
                    continue;
                }
                if let Some(arena) = shared.as_deref_mut()
                    && self.record_shared(scope, node, arena, &lanes[0].metadata)?
                {
                    continue;
                }
                if width > 1 && node.op == TensorOp::Linear && !(logits && prefill.is_some()) {
                    super::batch_projection::record(
                        scope,
                        node,
                        lanes,
                        self.weights,
                        projections,
                        self.nvfp4,
                    )?;
                    continue;
                }
                if self.record_verify_attention(scope, node, lanes, prefill)? {
                    continue;
                }
                for (index, lane) in lanes.iter_mut().enumerate() {
                    if logits && prefill.is_some() && index + 1 != width {
                        continue;
                    }
                    let mut capture = Capture {
                        scope,
                        arena: &mut lane.arena,
                        weights: self.weights,
                        nvfp4: self.nvfp4,
                        attention: self.attention,
                        states: self.states,
                        fp8_states: self.fp8_states,
                        metadata: &lane.metadata,
                        // Flat lanes share the program's single cache, so they share its table.
                        table: &self.tables[0],
                        external: &self.lane_external[index],
                        capacity: self.capacity,
                        fusion: &mut self.lane_fusion[index],
                        mode: CaptureMode::Flat,
                    };
                    let snapshot = prefill
                        .is_none()
                        .then(|| node.states.first())
                        .flatten()
                        .and_then(|id| lane.checkpoints.get_mut(id));
                    if let Some(snapshot) = snapshot {
                        if super::capture_checkpoint::flat_checkpoint_enabled() {
                            capture.record_flat_checkpoint(node, snapshot)?;
                        } else {
                            capture.record(node)?;
                            for (output, input) in
                                snapshot.iter_mut().zip(&capture.states[&node.states[0]])
                            {
                                scope.record(api::memcpy(output, input))?;
                            }
                        }
                    } else {
                        capture.record(node)?;
                    }
                }
            }
            Ok(())
        });
        self.nvfp4.end_quant_capture();
        result.map_err(device_error)
    }

    /// One graph covering all 32 lanes: Linear reads batched slots directly,
    /// shape-agnostic auxiliaries record once, state/metadata kernels stay per-lane.
    fn record_prompt_pair(
        &self,
        scope: &Scope,
        first: &TensorNode,
        second: &TensorNode,
        arena: &mut ActivationArena,
    ) -> StdResult<bool, DeviceError> {
        let profile = self.device.profile().map_err(error)?;
        super::prefill_projection::record_pair(scope, first, second, arena, (self.weights, profile))
    }

    fn capture32(
        &mut self,
        arena: &mut ActivationArena,
        metadata: PromptMetadata<'_>,
        readout: PromptReadout,
        profile: Option<&mut profile::PrefillProfile>,
        workspaces: (
            &mut BTreeMap<usize, Tensor<f32>>,
            &mut super::recurrent_prefill::Workspace,
            &mut Option<super::fusion_batch::Workspace>,
        ),
        mut prompt_row: Option<&mut Tensor<f32>>,
    ) -> Result<CudaGraph<()>> {
        let (metadata, metadata_batch, prefill_info, prefill_rotary) = metadata;
        let (partials, recurrent, fusion) = workspaces;
        let last = readout == PromptReadout::Logits;
        #[cfg(test)]
        let demand = (readout == PromptReadout::StateOnly)
            .then(|| super::prefill_demand::StateDemand::new(self.graph));
        let quant_plan = super::nvfp4_gemm::QuantPairPlan::new(self.graph, self.weights);
        let gdn_roles = gdn_ba_roles(self.graph, self.weights);
        let pairs = gdn_ba_pairs(self.graph, self.weights);
        let mlp = super::prefill_projection::MlpPlan::new(
            self.graph,
            self.weights,
            (arena.lanes(), &*self.nvfp4),
        );
        self.nvfp4.begin_quant_capture();
        let result = CudaGraph::scope(&self.device.stream, |scope| {
            let mut boundaries = profile.map(|profile| profile.graph(last));
            let mut fused_second = None;
            for (index, node) in self.graph.nodes.iter().enumerate() {
                if fused_second == Some(index) || mlp.absorbs(index) {
                    continue;
                }
                let logits = node.outputs.iter().any(|id| Some(*id) == self.graph.logits);
                let skip = logits && !last;
                #[cfg(test)]
                let skip = skip || demand.as_ref().is_some_and(|plan| !plan.retains(index));
                if self.skip_quant(&quant_plan, index, skip) {
                    continue;
                }
                if let Some(boundaries) = boundaries.as_mut() {
                    boundaries.boundary(scope, index, node)?;
                }
                if super::norm_maximum::record(
                    scope,
                    index,
                    &self.graph.nodes,
                    arena,
                    self.weights,
                    self.nvfp4,
                )? {
                    continue;
                }
                if !skip && let Some(&second) = pairs.get(&index) {
                    let paired = &self.graph.nodes[second];
                    let paired_skip = paired
                        .outputs
                        .iter()
                        .any(|id| Some(*id) == self.graph.logits)
                        && !last;
                    #[cfg(test)]
                    let paired_skip =
                        paired_skip || demand.as_ref().is_some_and(|plan| !plan.retains(second));
                    let fused =
                        !paired_skip && self.record_prompt_pair(scope, node, paired, arena)?;
                    if fused {
                        if let Some(boundaries) = boundaries.as_mut() {
                            boundaries.boundary(scope, second, paired)?;
                        }
                        fused_second = Some(second);
                        continue;
                    }
                }
                if logits && let Some(row) = prompt_row.as_deref_mut() {
                    super::prompt_logits::record(
                        scope,
                        node,
                        arena,
                        self.weights,
                        self.nvfp4,
                        (row, prefill_info),
                    )?;
                    continue;
                }
                if mlp.record(
                    scope,
                    (index, &self.graph.nodes),
                    arena,
                    self.weights,
                    self.nvfp4,
                )? {
                    continue;
                }
                self.record_prompt_node(
                    scope,
                    node,
                    gdn_roles.get(&index).copied(),
                    arena,
                    (metadata, metadata_batch, prefill_info, prefill_rotary),
                    (partials, recurrent, fusion),
                )?;
            }
            if let Some(boundaries) = boundaries.as_mut() {
                boundaries.finish(scope)?;
            }
            Ok(())
        });
        self.nvfp4.end_quant_capture();
        result.map_err(device_error)
    }

    /// Gather every lane of a target prompt chunk in one launch instead of one per lane.
    /// False leaves draft programs and geometries without contiguous external rows to the
    /// per-lane kernel.
    fn record_prompt_embedding(
        &self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        metadata_batch: &Tensor<i32>,
    ) -> StdResult<bool, DeviceError> {
        let lanes = arena.lanes();
        let (TensorOp::Embedding, None, Some(external)) =
            (&node.op, &self.weights.fusion, self.lane_external_owner)
        else {
            return Ok(false);
        };
        let Some(weight) = self.weights.embeddings.get(&node.inputs[0]) else {
            return Ok(false);
        };
        let slot = arena.slot(node.outputs[0]).map_err(error)?;
        let size = arena.get(node.outputs[0]).map_err(error)?.size();
        let dimension = size / lanes;
        if size != lanes * dimension
            || !dimension.is_multiple_of(AUX_KERNEL_TILE)
            || external.size() < size
            || metadata_batch.size() < lanes * METADATA_FIELDS
        {
            return Ok(false);
        }
        let output = arena.buffers[slot]
            .take()
            .ok_or_else(|| error("prompt embedding output"))?;
        let mut rows = output.reshape(&[lanes, dimension])?;
        // The owner holds rows for the widest prompt graph; this chunk reads its first lanes.
        let range = 0..size;
        let external = external.slice(from_ref(&range)).map_err(error)?;
        let external = external.view(&[lanes, dimension]).map_err(error)?;
        let recorded = scope
            .record(
                super::attention::attention::embedding_rows(
                    (&mut rows).partition([1, AUX_KERNEL_TILE]),
                    weight,
                    &external,
                    metadata_batch,
                )
                .generics(vec![dimension.to_string(), METADATA_FIELDS.to_string()]),
            )
            .map(drop);
        arena.buffers[slot] = Some(rows.reshape(&[size])?);
        recorded.map(|()| true)
    }

    fn record_prompt_node(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        role: Option<super::prefill_projection::GdnRole>,
        arena: &mut ActivationArena,
        metadata: PromptMetadata<'_>,
        workspaces: (
            &mut BTreeMap<usize, Tensor<f32>>,
            &mut super::recurrent_prefill::Workspace,
            &mut Option<super::fusion_batch::Workspace>,
        ),
    ) -> StdResult<(), DeviceError> {
        let (metadata, metadata_batch, prefill_info, prefill_rotary) = metadata;
        let (partials, recurrent, fusion) = workspaces;
        if self.record_prompt_embedding(scope, node, arena, metadata_batch)? {
            return Ok(());
        }
        if let Some(workspace) = fusion.as_mut()
            && super::fusion_batch::record(
                self,
                scope,
                node,
                arena,
                metadata,
                metadata_batch,
                workspace,
            )?
        {
            return Ok(());
        }
        if recurrent.record(scope, node, arena, self.weights, self.states, prefill_info)? {
            return Ok(());
        }
        if self.record_prompt_dynamic(scope, node, arena, prefill_info, prefill_rotary)? {
            return Ok(());
        }
        self.record_prompt_dispatch(scope, node, role, arena, metadata, partials)
    }

    /// Record the width-32 dispatch for a node whose special prefill cases did not apply.
    fn record_prompt_dispatch(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        role: Option<super::prefill_projection::GdnRole>,
        arena: &mut ActivationArena,
        metadata: &[Tensor<i32>],
        partials: &mut BTreeMap<usize, Tensor<f32>>,
    ) -> StdResult<(), DeviceError> {
        let mut no_fusion = None;
        match dispatch32(node, self.weights) {
            Dispatch32::Linear => {
                super::prefill_projection::record(
                    scope,
                    (node, role),
                    arena,
                    (self.weights, self.device.profile().map_err(error)?),
                    partials,
                    self.nvfp4,
                    self.delegated.as_deref_mut(),
                )?;
            }
            Dispatch32::Batched => {
                Capture {
                    scope,
                    arena: &mut *arena,
                    weights: self.weights,
                    nvfp4: self.nvfp4,
                    attention: self.attention,
                    states: self.states,
                    fp8_states: self.fp8_states,
                    metadata: &metadata[0],
                    table: &self.tables[0],
                    external: self.external,
                    capacity: self.capacity,
                    fusion: &mut no_fusion,
                    mode: CaptureMode::Batched,
                }
                .record(node)?;
            }
            Dispatch32::Row => {
                // Row kernels take scalar per-lane inputs, so a fused program records
                // its embedding and normalization against that lane's own hidden.
                for (lane, meta) in metadata.iter().enumerate().take(self.width) {
                    Capture {
                        scope,
                        arena: &mut *arena,
                        weights: self.weights,
                        nvfp4: self.nvfp4,
                        attention: self.attention,
                        states: self.states,
                        fp8_states: self.fp8_states,
                        metadata: meta,
                        // Every lane of a prompt chunk appends into the one cache this
                        // program owns, so they all resolve through one table.
                        table: &self.tables[0],
                        external: &self.lane_external[lane],
                        capacity: self.capacity,
                        fusion: &mut self.lane_fusion[lane],
                        mode: CaptureMode::Row(lane),
                    }
                    .record_row(node, lane)?;
                }
            }
        }
        Ok(())
    }

    /// Record a prompt-wide sequence kernel for dynamic `RoPE` or attention nodes.
    fn record_prompt_dynamic(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        prefill_info: &Tensor<i32>,
        prefill_rotary: Option<&Tensor<i32>>,
    ) -> StdResult<bool, DeviceError> {
        let dynamic = !node
            .inputs
            .iter()
            .any(|id| self.weights.constants.contains_key(id));
        if !dynamic {
            return Ok(false);
        }
        match node.op {
            TensorOp::Rope { .. } => {
                let (metadata, mode) = prefill_rotary
                    .map_or((prefill_info, CaptureMode::Prefill), |rotary| {
                        (rotary, CaptureMode::PrefillRotary)
                    });
                self.record_prompt_sequence(scope, node, arena, metadata, mode)?;
            }
            TensorOp::Attention { .. } => {
                self.record_prompt_sequence(
                    scope,
                    node,
                    arena,
                    prefill_info,
                    CaptureMode::Prefill,
                )?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Record one prompt-wide sequence kernel against a chosen metadata binding.
    fn record_prompt_sequence(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        metadata: &Tensor<i32>,
        mode: CaptureMode,
    ) -> StdResult<(), DeviceError> {
        let mut no_fusion = None;
        Capture {
            scope,
            arena,
            weights: self.weights,
            nvfp4: self.nvfp4,
            attention: self.attention,
            states: self.states,
            fp8_states: self.fp8_states,
            metadata,
            table: &self.tables[0],
            external: self.external,
            capacity: self.capacity,
            fusion: &mut no_fusion,
            mode,
        }
        .record(node)
    }

    fn skip_quant(
        &mut self,
        plan: &super::nvfp4_gemm::QuantPairPlan,
        index: usize,
        skip: bool,
    ) -> bool {
        self.nvfp4
            .enter_quant_node(index, (!skip).then(|| plan.role(index)).flatten());
        skip
    }

    #[expect(
        unsafe_code,
        reason = "Audited lane aliasing: the batched arena is stored in the BatchGraph that owns \
                  every graph captured against its lane views"
    )]
    fn allocate_lanes(&self, width: usize, shared: Option<&ActivationArena>) -> Result<Vec<Lane>> {
        let mut lanes = Vec::new();
        let mut bytes = 0usize;
        for _ in 0..width {
            let mut checkpoints = BTreeMap::new();
            for spec in &self.graph.tensors {
                if !matches!(
                    spec.storage,
                    TensorStorage::State {
                        kind: StateKind::Conv | StateKind::LinearAttention,
                        ..
                    }
                ) {
                    continue;
                }
                let mut copies = Vec::new();
                for source in &self.states[&spec.id] {
                    bytes = source
                        .size()
                        .checked_mul(F32_BYTES)
                        .and_then(|n| bytes.checked_add(n))
                        .ok_or_else(|| Error::invalid("checkpoint overflow"))?;
                    if u64::try_from(bytes).map_err(device_error)?
                        > self.device.profile()?.checkpoint_budget_bytes()
                    {
                        return Err(Error::invalid("verification checkpoints exceed budget"));
                    }
                    let shape = source
                        .shape()
                        .iter()
                        .map(|v| usize::try_from(*v).map_err(device_error))
                        .collect::<Result<Vec<_>>>()?;
                    copies.push(
                        api::zeros::<f32>(&shape)
                            .sync_on(&self.device.stream)
                            .map_err(device_error)?,
                    );
                }
                checkpoints.insert(spec.id, copies);
            }
            let arena = if let Some(shared) = shared {
                // SAFETY: `build` keeps `shared` alive in the BatchGraph beside these lanes'
                // graphs; per-lane and whole-slot nodes are recorded in sequence on one stream.
                unsafe { shared.lane_view(lanes.len())? }
            } else {
                ActivationArena::new(
                    self.device,
                    self.graph,
                    usize::try_from(self.device.profile()?.arena_budget_bytes())
                        .map_err(device_error)?,
                )?
            };
            lanes.push(Lane {
                arena,
                metadata: api::zeros::<i32>(&[METADATA_FIELDS])
                    .sync_on(&self.device.stream)
                    .map_err(device_error)?,
                checkpoints,
            });
        }
        Ok(lanes)
    }
}

/// Node dispatch for prefill width 32 captures.
pub(super) enum Dispatch32 {
    Linear,
    Batched,
    Row,
}

/// Batched recording needs every data input as a `[32, size]` slot; a constant
/// data input is a single row and degrades the node to per-lane recording.
/// Norm weights stay shared constants and never degrade the node.
pub(super) fn dispatch32(node: &TensorNode, weights: &ProgramWeights) -> Dispatch32 {
    let constant = |index: usize| weights.constants.contains_key(&node.inputs[index]);
    let batched = match &node.op {
        TensorOp::Linear => return Dispatch32::Linear,
        TensorOp::Silu | TensorOp::Sigmoid | TensorOp::Add | TensorOp::Multiply => !node
            .inputs
            .iter()
            .any(|id| weights.constants.contains_key(id)),
        TensorOp::Norm { .. } | TensorOp::Split { .. } => !constant(0),
        TensorOp::GatedNorm { .. } => !constant(0) && !constant(1),
        _ => false,
    };
    if batched {
        Dispatch32::Batched
    } else {
        Dispatch32::Row
    }
}

/// Move one captured output out of the activation arena.
pub(super) fn take_result(
    arena: &mut ActivationArena,
    id: Option<TensorId>,
) -> Result<Arc<Tensor<f32>>> {
    let slot = arena.slot(id.ok_or_else(|| Error::invalid("batch result binding"))?)?;
    Ok(Arc::new(
        arena.buffers[slot]
            .take()
            .ok_or_else(|| Error::invariant("batch result buffer"))?,
    ))
}

impl BatchGraph {
    /// Device hidden rows of this graph's prompt lanes, valid after a replay.
    pub(crate) fn hidden_rows(&self) -> Arc<Tensor<f32>> {
        Arc::clone(&self.hidden[0])
    }

    pub(crate) async fn run_async(
        &mut self,
        device: &CudaDevice,
        tokens: &[u32],
        position: usize,
        prefill: Option<bool>,
        read_hidden: bool,
    ) -> Result<BatchOutput> {
        Ok(self
            .run_readout_async(device, tokens, position, prefill, read_hidden, false)
            .await?
            .rows)
    }

    pub(crate) async fn run_readout_async(
        &mut self,
        device: &CudaDevice,
        tokens: &[u32],
        position: usize,
        prefill: Option<bool>,
        read_hidden: bool,
        plain_greedy: bool,
    ) -> Result<BatchReadout> {
        if self.width >= PREFILL_LANES {
            return self
                .run32_async(
                    device,
                    tokens,
                    position,
                    prefill,
                    read_hidden,
                    PromptOverrides {
                        lanes: &[],
                        rotary: None,
                    },
                )
                .await
                .map(BatchReadout::host);
        }
        let device_tokens = plain_greedy && prefill.is_none() && self.token_readout.is_some();
        let graph = match prefill {
            None => self
                .graph
                .as_ref()
                .ok_or_else(|| Error::invalid("prompt-only batch cannot verify"))?,
            Some(false) => &self.prefill,
            Some(true) => &self.prefill_last,
        };
        Self::update_metadata(
            graph,
            &mut self.metadata,
            self.width,
            self.capacity,
            tokens,
            position,
            self.state_offset,
            PromptOverrides {
                lanes: &[],
                rotary: None,
            },
        )?;
        if prefill.is_some() {
            // A prompt graph's append and recurrent kernels resolve positions from this sidecar
            // rather than from the per-lane metadata. run32 writes it; this narrow path replayed
            // the same graphs without writing it, so they read the previous chunk's base - which
            // silently placed KV rows outside the chunk and, with a block table, out of the arena.
            Self::validate_prefill_state(tokens.len(), position, self.state_offset, self.capacity)?;
            super::metadata::update(
                graph,
                self.prefill_info
                    .as_mut()
                    .ok_or_else(|| Error::invariant("prefill metadata"))?,
                [
                    i32::try_from(position).map_err(device_error)?,
                    i32::try_from(tokens.len()).map_err(device_error)?,
                    self.state_offset,
                    0,
                ],
            )?;
        }
        let mut sources = Vec::with_capacity(2 * tokens.len());
        for lane in 0..tokens.len() {
            let logits = !device_tokens
                && (prefill.is_none() || prefill == Some(true) && lane + 1 == tokens.len());
            sources
                .push(read_hidden.then(|| ReadbackSource::whole(Arc::clone(&self.hidden[lane]))));
            sources.push(logits.then(|| ReadbackSource::whole(Arc::clone(&self.logits[lane]))));
        }
        if device_tokens {
            let active: Vec<_> = (0..tokens.len()).collect();
            let (rows, tokens) = self
                .token_readout
                .as_mut()
                .ok_or_else(|| Error::invariant("batch token readout"))?
                .run_async(device, graph, &mut self.readbacks, &sources, &active)
                .await?;
            return Ok(BatchReadout {
                rows,
                tokens: Some(tokens.into_iter().map(Some).collect()),
            });
        }
        let mut rows = self
            .readbacks
            .run_async(device, graph, &sources)
            .await?
            .into_iter();
        (0..tokens.len())
            .map(|_| {
                Ok((
                    rows.next()
                        .ok_or_else(|| Error::invariant("batch hidden readback"))?,
                    rows.next()
                        .ok_or_else(|| Error::invariant("batch logits readback"))?,
                ))
            })
            .collect::<Result<_>>()
            .map(BatchReadout::host)
    }

    /// State positions this graph lags its `RoPE` position by.
    fn validate_prefill_state(
        count: usize,
        position: usize,
        state_offset: i32,
        capacity: usize,
    ) -> Result<()> {
        let last_row = i64::try_from(position).map_err(device_error)?
            + i64::try_from(count).map_err(device_error)?
            - 1
            + i64::from(state_offset);
        if last_row >= i64::try_from(capacity).map_err(device_error)? {
            return Err(Error::new(
                ErrorCode::Capacity,
                format!(
                    "prompt state rows end at {last_row} (base {position}, {count} tokens, offset {state_offset}), past the {capacity}-token KV arena"
                ),
            ));
        }
        Ok(())
    }
    #[must_use]
    pub const fn state_offset(&self) -> i32 {
        self.state_offset
    }

    /// One lane's metadata tuple, with the same inactive-tail and capacity contract as the
    /// per-lane updater. Both the scalar and the batched staging paths consume this function.
    fn metadata_rows(
        width: usize,
        capacity: usize,
        tokens: &[u32],
        position: usize,
        state_offset: i32,
        overrides: PromptOverrides<'_>,
    ) -> Result<Vec<[i32; METADATA_FIELDS]>> {
        let mut rows = Vec::with_capacity(width);
        for lane in 0..width {
            let token = tokens.get(lane).copied().unwrap_or(0);
            let pos = i32::try_from(position + lane).map_err(device_error)?;
            let state_pos = if width >= PREFILL_LANES && lane >= tokens.len() {
                -1
            } else {
                pos.saturating_add(state_offset)
            };
            // A KV row past the arena would be written out of bounds by the state kernels, and
            // the earlier kernels only report that on a checked load. Refuse it here, where the
            // numbers are still host-side.
            // `-1` is the inactive sentinel the kernels skip; anything below it, or at or past
            // the arena depth, would address a row the state kernels do not check on store.
            if state_pos < -1 || state_pos >= i32::try_from(capacity).map_err(device_error)? {
                return Err(Error::new(
                    ErrorCode::Capacity,
                    format!(
                        "state position {state_pos} (base {position}, lane {lane}, offset {state_offset}) \
                         exceeds the {capacity}-token KV arena"
                    ),
                ));
            }
            rows.push([
                pos,
                i32::try_from(token).map_err(device_error)?,
                state_pos,
                i32::from(overrides.lanes.get(lane).copied().unwrap_or(false)),
            ]);
        }
        Ok(rows)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Launch fields are independent scalars; bundling them would hide the metadata contract"
    )]
    fn update_metadata(
        graph: &CudaGraph<()>,
        metadata: &mut [Tensor<i32>],
        width: usize,
        capacity: usize,
        tokens: &[u32],
        position: usize,
        state_offset: i32,
        overrides: PromptOverrides<'_>,
    ) -> Result<()> {
        let rows = Self::metadata_rows(width, capacity, tokens, position, state_offset, overrides)?;
        for (metadata, values) in metadata.iter_mut().take(width).zip(rows) {
            super::metadata::update(graph, metadata, values)?;
        }
        Ok(())
    }

    /// Stage all prompt lanes' metadata in one host-to-device copy instead of one tiny
    /// `graph.update` kernel per lane. The rows stay in the same owner the captured graph
    /// already points at, so only the transfer granularity changes.
    fn update_metadata_batch(
        &mut self,
        device: &CudaDevice,
        tokens: &[u32],
        position: usize,
        overrides: PromptOverrides<'_>,
    ) -> Result<()> {
        let owner = self
            .metadata_owner
            .as_mut()
            .ok_or_else(|| Error::invariant("batched metadata owner"))?;
        stage_metadata_batch(
            device,
            owner,
            (&mut self.metadata_upload, &mut self.metadata_staging),
            self.width,
            self.capacity,
            tokens,
            position,
            self.state_offset,
            overrides,
        )
    }

    fn validate_prompt_positions(&self, count: usize, position: usize) -> Result<()> {
        // The slot-decode path already refuses a state position past the arena
        // (`stage_lanes`); the prompt path has to as well, because the prefill append and
        // decode kernels address that many rows of KV without a checked load.
        let last_row = i64::try_from(position).map_err(device_error)?
            + i64::try_from(count).map_err(device_error)?
            - 1
            + i64::from(self.state_offset);
        if last_row >= i64::try_from(self.capacity).map_err(device_error)? {
            return Err(Error::new(
                ErrorCode::Capacity,
                format!(
                    "prompt state rows end at {last_row} (base {position}, {} tokens, offset {}), \
                     past the {}-token KV arena",
                    count, self.state_offset, self.capacity
                ),
            ));
        }
        Ok(())
    }

    fn prompt_readout_graph<'a>(
        regular: (&'a CudaGraph<()>, &'a CudaGraph<()>),
        selected: Option<&'a PromptGraphs>,
        state: Option<&'a CudaGraph<()>>,
        readout: Option<bool>,
        read_hidden: bool,
    ) -> Result<&'a CudaGraph<()>> {
        let prefill = selected.map_or(regular.0, |graphs| &graphs.prefill);
        let last = selected.map_or(regular.1, |graphs| &graphs.prefill_last);
        match readout {
            Some(false) => {
                #[cfg(test)]
                if !read_hidden && super::prefill_demand::enabled() {
                    let state = selected.map_or(state, |graphs| graphs.state_prefill.as_ref());
                    return Ok(state.unwrap_or(prefill));
                }
                #[cfg(not(test))]
                let _ = (read_hidden, state);
                Ok(prefill)
            }
            Some(true) => Ok(last),
            None => Err(Error::invalid("prompt-only batch cannot verify")),
        }
    }

    /// Batched slots read back in one copy each; per-lane rows are split on host.
    async fn run32_async(
        &mut self,
        device: &CudaDevice,
        tokens: &[u32],
        position: usize,
        prefill: Option<bool>,
        read_hidden: bool,
        overrides: PromptOverrides<'_>,
    ) -> Result<BatchOutput> {
        self.validate_prompt_positions(tokens.len(), position)?;
        self.update_prompt_rotary(device, tokens.len(), position, overrides.rotary)?;
        // Stage batched metadata before taking the graph borrow; the owner is a disjoint
        // field and the copy is ordered on the same stream as the later replay.
        let staged_metadata = self.metadata_owner.is_some();
        if staged_metadata {
            self.update_metadata_batch(device, tokens, position, overrides)?;
        }
        let precise = self.precise.is_some()
            && position
                .checked_add(tokens.len())
                .is_some_and(|end| end <= precise_prefix_positions());
        #[cfg(test)]
        let state = self.state_prefill.as_ref();
        #[cfg(not(test))]
        let state = None;
        let graph = Self::prompt_readout_graph(
            (&self.prefill, &self.prefill_last),
            self.precise.as_ref().filter(|_| precise),
            state,
            prefill,
            read_hidden,
        )?;
        if !staged_metadata {
            Self::update_metadata(
                graph,
                &mut self.metadata,
                self.width,
                self.capacity,
                tokens,
                position,
                self.state_offset,
                overrides,
            )?;
        }
        super::metadata::update(
            graph,
            self.prefill_info
                .as_mut()
                .ok_or_else(|| Error::invariant("prefill metadata"))?,
            [
                i32::try_from(position).map_err(device_error)?,
                i32::try_from(tokens.len()).map_err(device_error)?,
                self.state_offset,
                0,
            ],
        )?;
        let hidden_dim = self.hidden[0].size() / self.width;
        // A one-row logits slot already holds the final lane.
        let single_logits = self.prompt_row.is_some();
        let vocabulary = self.logits[0].size() / if single_logits { 1 } else { self.width };
        let sources = [
            read_hidden.then(|| ReadbackSource {
                tensor: Arc::clone(&self.hidden[0]),
                skip: 0,
                len: tokens.len() * hidden_dim,
            }),
            (prefill == Some(true)).then(|| ReadbackSource {
                tensor: Arc::clone(&self.logits[0]),
                skip: if single_logits {
                    0
                } else {
                    (tokens.len() - 1) * vocabulary
                },
                len: vocabulary,
            }),
        ];
        let profile = if precise {
            &self.precise_profile
        } else {
            &self.profile
        };
        if sources.iter().all(Option::is_none) && profile.is_none() {
            device.run_op(graph.launch()).await?;
            return Ok(vec![(Vec::new(), Vec::new()); tokens.len()]);
        }
        let rows = self.readbacks.run_async(device, graph, &sources).await;
        if rows.is_ok()
            && let Some(profile) = profile
        {
            profile.report(prefill == Some(true), tokens.len(), position);
        }
        Self::prompt_rows(rows?, (tokens.len(), hidden_dim), read_hidden)
    }

    fn prompt_rows(
        rows: Vec<Vec<f32>>,
        (lanes, hidden_dim): (usize, usize),
        read_hidden: bool,
    ) -> Result<BatchOutput> {
        let mut rows = rows.into_iter();
        let hidden_rows = rows
            .next()
            .ok_or_else(|| Error::invariant("batch hidden readback"))?;
        let mut logits = rows
            .next()
            .ok_or_else(|| Error::invariant("batch logits readback"))?;
        (0..lanes)
            .map(|lane| {
                let hidden = if read_hidden {
                    hidden_rows[lane * hidden_dim..(lane + 1) * hidden_dim].to_vec()
                } else {
                    Vec::new()
                };
                let logits = if lane + 1 == lanes {
                    take(&mut logits)
                } else {
                    Vec::new()
                };
                Ok((hidden, logits))
            })
            .collect()
    }

    /// Write the per-lane `(t, h, w)` table a prompt chunk's M-RoPE kernel reads.
    fn update_prompt_rotary(
        &mut self,
        device: &CudaDevice,
        lanes: usize,
        position: usize,
        rotary: Option<&[[usize; ROTARY_AXES]]>,
    ) -> Result<()> {
        let Some(table) = self.prefill_rotary.as_mut() else {
            return Ok(());
        };
        if rotary.is_some_and(|rotary| rotary.len() != lanes) {
            return Err(Error::invalid("prompt rotary table width"));
        }
        let mut values = vec![0i32; self.width * ROTARY_AXES];
        for lane in 0..lanes {
            // Prompt-local coordinates shift with the base, like the 1-D fallback does.
            let axes = rotary.map_or([position + lane; ROTARY_AXES], |rotary| {
                let axes = rotary[lane];
                [axes[0] + position, axes[1] + position, axes[2] + position]
            });
            let base = lane * ROTARY_AXES;
            values[base] = i32::try_from(axes[0]).map_err(device_error)?;
            values[base + 1] = i32::try_from(axes[1]).map_err(device_error)?;
            values[base + 2] = i32::try_from(axes[2]).map_err(device_error)?;
        }
        self.rotary_staging.copy(device, table, &values)
    }

    /// Replay one prompt chunk whose media lanes carry precomputed embeddings.
    ///
    /// The caller has already copied each media row into its lane's external slot on the device
    /// stream; `overrides[lane]` selects that slot in the embedding kernel.
    pub(crate) async fn run_media_async(
        &mut self,
        device: &CudaDevice,
        tokens: &[u32],
        position: usize,
        read_logits: bool,
        overrides: PromptOverrides<'_>,
    ) -> Result<BatchOutput> {
        if overrides.lanes.len() != tokens.len() {
            return Err(Error::invalid("media prefill overrides"));
        }
        self.run32_async(
            device,
            tokens,
            position,
            Some(read_logits),
            false,
            overrides,
        )
        .await
    }

    #[cfg(test)]
    #[expect(
        clippy::used_underscore_binding,
        reason = "Test observes the retained graph owner allocation"
    )]
    pub(super) fn test_fusion_scratch_bytes(&self) -> usize {
        self._fusion
            .as_ref()
            .map_or(0, super::fusion_batch::Workspace::bytes)
    }

    /// Passive readback of the most recent prompt replay; never launches a graph.
    #[cfg(test)]
    pub(super) fn test_retained_prompt_hidden(
        &self,
        device: &CudaDevice,
        position: usize,
        rows: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if rows == 0 || rows > self.width || self.hidden.len() != 1 {
            return Err(Error::invalid("test retained prompt rows"));
        }
        let info = device.read_borrowed(
            self.prefill_info
                .as_ref()
                .ok_or_else(|| Error::invariant("test retained prompt metadata"))?,
        )?;
        let expected = [
            i32::try_from(position).map_err(device_error)?,
            i32::try_from(rows).map_err(device_error)?,
        ];
        if info.get(..2) != Some(expected.as_slice()) {
            return Err(Error::invalid(
                "test retained prompt is not the requested replay",
            ));
        }
        let values = device.read_borrowed(&self.hidden[0])?;
        let columns = values.len() / self.width;
        if columns == 0 || values.len() % self.width != 0 {
            return Err(Error::invariant("test retained prompt hidden shape"));
        }
        Ok(values
            .chunks_exact(columns)
            .take(rows)
            .map(<[f32]>::to_vec)
            .collect())
    }

    pub(crate) fn restore_async<'a>(
        &self,
        device: &'a CudaDevice,
        count: usize,
    ) -> impl Future<Output = Result<()>> + Send + 'a {
        let launch = self
            .restore
            .get(count - 1)
            .map(CudaGraph::launch)
            .ok_or_else(|| Error::invalid("batch commit count"));
        async move { device.run_op(launch?).await }
    }
}

/// Stage one prompt chunk's batched metadata rows into their existing owner.
///
/// The pinned source remains owned by the graph. The caller awaits replay completion
/// before the next prompt chunk can overwrite this staging storage.
#[expect(
    clippy::too_many_arguments,
    reason = "The staging contract names each launch scalar explicitly instead of hiding them"
)]
fn stage_metadata_batch(
    device: &CudaDevice,
    owner: &mut Tensor<i32>,
    (upload, staging): (&mut Vec<i32>, &mut PinnedUpload<i32>),
    width: usize,
    capacity: usize,
    tokens: &[u32],
    position: usize,
    state_offset: i32,
    overrides: PromptOverrides<'_>,
) -> Result<()> {
    let rows =
        BatchGraph::metadata_rows(width, capacity, tokens, position, state_offset, overrides)?;
    upload.clear();
    for row in rows {
        upload.extend_from_slice(&row);
    }
    staging.copy(device, owner, upload)
}

/// One active lane of a slot decode replay: slot index, token, `RoPE` position, state position.
pub type SlotLane = (usize, u32, usize, usize);

impl SlotDecodeGraph {
    #[cfg(test)]
    pub(super) fn test_hidden(&self, device: &CudaDevice, slot: usize) -> Result<Vec<f32>> {
        if slot >= self.width {
            return Err(Error::invalid("test hidden slot"));
        }
        let values = device.read_borrowed(&self.hidden)?;
        let columns = values.len() / self.width;
        Ok(values[slot * columns..(slot + 1) * columns].to_vec())
    }

    /// Validate the lane set against the cursors, then stage every slot's metadata for
    /// the next replay. `graph` only retains the metadata writes until that replay.
    #[allow(
        clippy::too_many_arguments,
        reason = "disjoint field borrows of the graph"
    )]
    fn stage_lanes(
        metadata: &mut Vec<i32>,
        cursors: &[usize],
        width: usize,
        external_hidden: bool,
        lanes: &[SlotLane],
        capacity: usize,
        vocabulary: usize,
    ) -> Result<()> {
        if lanes.is_empty() || lanes.len() > width {
            return Err(Error::invalid("slot decode lane count"));
        }
        let mut per_slot = vec![None; width];
        for &(slot, token, rope_pos, state_pos) in lanes {
            if slot >= width
                || per_slot[slot].is_some()
                || usize::try_from(token).map_err(device_error)? >= vocabulary
                || state_pos >= capacity
            {
                return Err(Error::invalid("slot decode slot, token or position"));
            }
            if state_pos != cursors[slot] {
                return Err(Error::invalid("nonsequential slot state position"));
            }
            per_slot[slot] = Some((token, rope_pos, state_pos));
        }
        metadata.clear();
        for entry in per_slot.iter().take(width) {
            let values = match *entry {
                Some((token, rope_pos, state_pos)) => [
                    i32::try_from(rope_pos).map_err(device_error)?,
                    i32::try_from(token).map_err(device_error)?,
                    i32::try_from(state_pos).map_err(device_error)?,
                    i32::from(external_hidden),
                ],
                None => [0, 0, INACTIVE_LANE_STATE_POSITION, 0],
            };
            metadata.extend_from_slice(&values);
        }
        Ok(())
    }

    /// Replay one decode step for the active lanes in `lanes` (output order); every other
    /// lane is masked with a `-1` state position so its slot's state stays untouched.
    ///
    /// Each active lane's state position must equal the slot cursor (same nonsequential
    /// rejection as the single-step path) and advances it by one on success. Returns one
    /// `(hidden, logits)` pair per active lane; external-hidden draft graphs return
    /// hidden rows for their next step, while target decode uses logits-only readout.
    /// # Errors
    /// Rejects empty/oversized lane sets, duplicate or unknown slots, out-of-vocabulary
    /// tokens, out-of-capacity state positions, stale cursors and CUDA replay failures.
    pub(crate) async fn run_lanes_async(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        capacity: usize,
        vocabulary: usize,
        read_logits: bool,
    ) -> Result<BatchOutput> {
        Ok(self
            .run_lanes_readout_async(device, lanes, capacity, vocabulary, read_logits, false)
            .await?
            .rows)
    }

    pub(crate) async fn run_lanes_readout_async(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        capacity: usize,
        vocabulary: usize,
        read_logits: bool,
        plain_greedy: bool,
    ) -> Result<BatchReadout> {
        let device_tokens = read_logits && plain_greedy && self.token_readout.is_some();
        let graph = if read_logits {
            &self.graph
        } else {
            self.state_only
                .as_ref()
                .ok_or_else(|| Error::invalid("state-only draft graph"))?
        };
        Self::stage_lanes(
            &mut self.metadata_upload,
            &self.cursors,
            self.width,
            self.external_hidden,
            lanes,
            capacity,
            vocabulary,
        )?;
        self.metadata_staging
            .copy(device, &mut self.metadata_owner, &self.metadata_upload)?;
        let vocabulary = self.logits.size() / self.width;
        let mut sources = Vec::with_capacity(2 * lanes.len());
        for &(slot, ..) in lanes {
            sources.push(
                (read_logits && self.external_hidden).then(|| ReadbackSource {
                    tensor: Arc::clone(&self.hidden),
                    skip: slot * (self.hidden.size() / self.width),
                    len: self.hidden.size() / self.width,
                }),
            );
            sources.push((read_logits && !device_tokens).then(|| ReadbackSource {
                tensor: Arc::clone(&self.logits),
                skip: slot * vocabulary,
                len: vocabulary,
            }));
        }
        let (output, tokens) = if device_tokens {
            let active: Vec<_> = lanes.iter().map(|&(slot, ..)| slot).collect();
            let (rows, tokens) = self
                .token_readout
                .as_mut()
                .ok_or_else(|| Error::invariant("slot token readout"))?
                .run_async(device, graph, &mut self.readbacks, &sources, &active)
                .await?;
            (Some(rows), Some(tokens.into_iter().map(Some).collect()))
        } else {
            (None, None)
        };
        let rows = if output.is_none() {
            self.readbacks.run_async(device, graph, &sources).await?
        } else {
            Vec::new()
        };
        if read_logits
            && let Some(profile) = &self.profile
            && let Some(&(.., rope_pos, _)) = lanes.first()
        {
            profile.report("slot_decode", lanes.len(), rope_pos);
        }
        let mut rows = rows.into_iter();
        for &(slot, ..) in lanes {
            self.cursors[slot] += 1;
        }
        if let Some(rows) = output {
            return Ok(BatchReadout { rows, tokens });
        }
        lanes
            .iter()
            .map(|_| {
                Ok((
                    rows.next()
                        .ok_or_else(|| Error::invariant("slot hidden readback"))?,
                    rows.next()
                        .ok_or_else(|| Error::invariant("slot logits readback"))?,
                ))
            })
            .collect::<Result<_>>()
            .map(BatchReadout::host)
    }

    /// Replay the state-only graph for `lanes` without a readback or a stream sync: a
    /// draft catch-up needs the state writes, never the outputs. Completion is carried
    /// by the next readback on this stream or the pool's drop barrier.
    /// # Errors
    /// Rejects the same lane/position errors as [`Self::run_lanes`], a missing state-only
    /// graph, and CUDA launch failures detected at enqueue time.
    #[expect(
        unsafe_code,
        reason = "Audited detached replay: single-stream ordering after staged metadata, \
                  pool-owned buffers, drained by the owning pool's drop barrier"
    )]
    pub fn replay_detached(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        capacity: usize,
        vocabulary: usize,
    ) -> Result<()> {
        let graph = self
            .state_only
            .as_ref()
            .ok_or_else(|| Error::invalid("state-only draft graph"))?;
        Self::stage_lanes(
            &mut self.metadata_upload,
            &self.cursors,
            self.width,
            self.external_hidden,
            lanes,
            capacity,
            vocabulary,
        )?;
        self.metadata_staging
            .copy(device, &mut self.metadata_owner, &self.metadata_upload)?;
        // SAFETY: the engine is single-stream, so this replay is ordered after the staged
        // metadata and uploads and before any later replay; every buffer it touches is
        // pool-owned, and the pool's drop barrier drains the stream before anything frees.
        unsafe { graph.launch().async_on(&device.stream) }.map_err(device_error)?;
        for &(slot, ..) in lanes {
            self.cursors[slot] += 1;
        }
        Ok(())
    }

    /// Drain the capture stream: the pool's teardown barrier for detached replays.
    /// # Errors
    /// Returns the driver error when the stream cannot be synchronized.
    #[expect(
        unsafe_code,
        reason = "Audited stream drain at teardown: no work is enqueued after the owning \
                  pool starts dropping"
    )]
    pub(super) fn barrier(&self) -> Result<()> {
        // SAFETY: teardown only; no work is enqueued after the owning pool starts dropping.
        unsafe { self.graph.stream().synchronize() }.map_err(device_error)
    }

    /// Host-side state cursors of every slot.
    #[must_use]
    pub fn cursors(&self) -> &[usize] {
        &self.cursors
    }

    /// Seed a slot's cursor after binding a sequence's covered prefix into it.
    /// # Errors
    /// Rejects unknown slot indices.
    pub fn set_cursor(&mut self, slot: usize, position: usize) -> Result<()> {
        let cursor = self
            .cursors
            .get_mut(slot)
            .ok_or_else(|| Error::invalid("slot decode cursor"))?;
        *cursor = position;
        Ok(())
    }
}

/// Beta and alpha input positions of a GDN `Delta` node.
const GDN_BETA_INPUT: usize = 1;
const GDN_ALPHA_INPUT: usize = 2;
/// Pair output width must fit one prefill 64-column tile.
const GDN_BA_MAX_ROWS: usize = 64;

/// GDN beta/alpha producer roles from one `Delta` node's semantic inputs.
fn gdn_ba_roles(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
) -> BTreeMap<usize, super::prefill_projection::GdnRole> {
    gdn_ba_roles_by(graph, |id| {
        let Some(ProjectionWeight::Dense(weight)) = weights.projections.get(&id) else {
            return None;
        };
        Some((
            usize::try_from(weight.shape()[0]).ok()?,
            usize::try_from(weight.shape()[1]).ok()?,
        ))
    })
}

fn gdn_ba_roles_by(
    graph: &DataflowGraph,
    shape: impl Fn(TensorId) -> Option<(usize, usize)>,
) -> BTreeMap<usize, super::prefill_projection::GdnRole> {
    let mut producer = BTreeMap::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        for id in &node.outputs {
            producer.insert(*id, index);
        }
    }
    let mut roles = BTreeMap::new();
    for (delta, node) in graph.nodes.iter().enumerate() {
        if !matches!(&node.op, TensorOp::Delta { .. }) || node.inputs.len() <= GDN_ALPHA_INPUT {
            continue;
        }
        let (beta, alpha) = (node.inputs[GDN_BETA_INPUT], node.inputs[GDN_ALPHA_INPUT]);
        let (Some(&first), Some(&second)) = (producer.get(&beta), producer.get(&alpha)) else {
            continue;
        };
        if !gdn_ba_ordered(graph, first, second, delta) {
            continue;
        }
        let (Some(&weight_b), Some(&weight_a)) = (
            graph.nodes[first].inputs.get(1),
            graph.nodes[second].inputs.get(1),
        ) else {
            continue;
        };
        let (Some(beta_shape), Some(alpha_shape)) = (shape(weight_b), shape(weight_a)) else {
            continue;
        };
        if !gdn_ba_weight_compatible(beta_shape, alpha_shape) {
            continue;
        }
        roles.insert(first, super::prefill_projection::GdnRole::Beta);
        roles.insert(second, super::prefill_projection::GdnRole::Alpha);
    }
    roles
}

fn gdn_ba_pairs(graph: &DataflowGraph, weights: &ProgramWeights) -> BTreeMap<usize, usize> {
    let roles = gdn_ba_roles(graph, weights);
    let mut pairs = BTreeMap::new();
    let mut iter = roles.iter().peekable();
    while let Some((&first, &role)) = iter.next() {
        if role != super::prefill_projection::GdnRole::Beta {
            continue;
        }
        if let Some(&(&second, _)) = iter.peek()
            && roles.get(&second) == Some(&super::prefill_projection::GdnRole::Alpha)
        {
            pairs.insert(first, second);
        }
    }
    pairs
}

fn gdn_ba_ordered(graph: &DataflowGraph, first: usize, second: usize, delta: usize) -> bool {
    if second != first + 1 || second >= delta {
        return false;
    }
    let node = &graph.nodes[delta];
    if !matches!(&node.op, TensorOp::Delta { .. }) || node.inputs.len() <= GDN_ALPHA_INPUT {
        return false;
    }
    let (beta, alpha) = (node.inputs[GDN_BETA_INPUT], node.inputs[GDN_ALPHA_INPUT]);
    let (first_node, second_node) = (&graph.nodes[first], &graph.nodes[second]);
    first_node.outputs.as_slice() == [beta]
        && second_node.outputs.as_slice() == [alpha]
        && first_node.layer == node.layer
        && second_node.layer == node.layer
        && gdn_ba_nodes(first_node, second_node)
}

fn gdn_ba_nodes(first: &TensorNode, second: &TensorNode) -> bool {
    first.layer == second.layer
        && first.op == TensorOp::Linear
        && second.op == TensorOp::Linear
        && first.inputs.len() == 2
        && second.inputs.len() == 2
        && first.inputs[0] == second.inputs[0]
        && first.outputs.len() == 1
        && second.outputs.len() == 1
}

fn gdn_ba_weight_compatible(beta: (usize, usize), alpha: (usize, usize)) -> bool {
    beta == alpha && (1..=GDN_BA_MAX_ROWS).contains(&beta.0) && beta.1 > 0
}

/// One allocation for every prompt lane's metadata, with a non-owning row view per lane.
/// The batched embedding gather reads token ids from this contiguous owner.
#[expect(
    unsafe_code,
    reason = "Audited metadata lane views: the owner is returned and stored with the rows, \
              each view addresses a disjoint lane, and graphs retain the owner"
)]
pub(super) fn batched_metadata(
    device: &CudaDevice,
    width: usize,
) -> Result<(Tensor<i32>, Vec<Tensor<i32>>)> {
    let fields = METADATA_FIELDS;
    let elements = width
        .checked_mul(fields)
        .ok_or_else(|| Error::invalid("metadata size overflow"))?;
    let owner = api::zeros::<i32>(&[elements])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut rows = Vec::with_capacity(width);
    for lane in 0..width {
        // SAFETY: `owner` is returned to the caller and stored beside `rows` for the
        // whole graph lifetime; each view addresses one disjoint metadata row.
        rows.push(unsafe { super::arena::row_view(&owner, width, lane)? });
    }
    Ok((owner, rows))
}

/// Views of the shared prompt arena and logits row for one program's prompt graphs.
#[expect(
    unsafe_code,
    reason = "Audited prompt alias: graphs retain the shared scratch, and sequences replay prompt graphs one at a time on the device stream, reading their outputs before another starts"
)]
fn shared_prompt_views(shared: &PromptScratch) -> Result<(ActivationArena, Option<Tensor<f32>>)> {
    // SAFETY: see the expectation above.
    let arena = unsafe { shared.arena.shared_view()? };
    let row = match &shared.row {
        Some(row) => {
            // SAFETY: see the expectation above; one lane spans the whole row.
            let view = unsafe { super::arena::row_view(row, 1, 0)? };
            Some(view.reshape(&[1, row.size()]).map_err(device_error)?)
        }
        None => None,
    };
    Ok((arena, row))
}

fn slot_arena(device: &CudaDevice, graph: &DataflowGraph, width: usize) -> Result<ActivationArena> {
    ActivationArena::new_batched(
        device,
        graph,
        width,
        usize::try_from(device.profile()?.arena_budget_bytes()).map_err(device_error)?,
    )
}

#[cfg(test)]
#[path = "../../tests/unit/resident_batch_ba_pair.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_metadata_batch.rs"]
mod metadata_batch_tests;
