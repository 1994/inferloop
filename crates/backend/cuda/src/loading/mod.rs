//! Native model loading independent of diagnostic runners and CPU reference execution.
use crate::attention_profile::precise_prefix_positions;
use crate::constants::EXTRA_WIDE_PREFILL_LANES;
use crate::constants::WIDER_PREFILL_LANES;
use crate::constants::WIDEST_PREFILL_LANES;
use crate::resident::prompt_logits;
use std::path::PathBuf;
mod concurrency;
mod kv_scales;
mod weights;
use crate::constants::FUSED_VERIFY_LANES;
use crate::constants::MAX_MTP_DEPTH;
use crate::constants::MAX_PREFILL_LANES;
use crate::constants::MAX_VERIFICATION_WIDTH;
use crate::constants::MID_PREFILL_LANES;
use crate::constants::PREFILL_LANES;
use crate::constants::WIDE_PREFILL_LANES;
use crate::device::DeviceProfile;
use crate::mlp::ProjectionWeight;

use crate::resident::fp8_cache::Fp8Caches;
use crate::resident::slot_batch::SlotPool;
use crate::tuning::attention::mma::prepare;
use crate::vision::VisionWeights;
use crate::vision::footprint;
use crate::vision::merger;
use crate::vision::patch_embed;
use crate::vision::patch_width;
use crate::vision::tower;
use infer_models::vision::gather_positions;
use infer_models::vision::position_taps;
pub use kv_scales::load as load_kv_scales;
mod bindings;
mod budget;
pub(crate) use budget::PooledServingQuote;
mod experts;
mod phase;
use crate::resident::dflash2::DraftWeights;
use crate::{
    device::CudaDevice,
    resident::{DeviceProgram, ProgramWeights},
    tuning::TuningReport,
};
use infer_core::{Error, ModelId, Result};
use infer_ir::{DataflowGraph, ModelIr};
use infer_models::QuantizedPackage;
use std::path::Path;
use std::sync::Arc;
pub use weights::{Projection, floats};

/// Load-time execution policy; each request receives independent mutable device state.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub prefill_width: usize,
    pub verification_width: usize,
    /// None follows the checkpoint declaration; Some overrides KV storage.
    pub fp8_kv: Option<bool>,
    pub mtp_depth: usize,
    /// Measure GEMV tiles for geometries the local or shipped tables do not cover. On by default:
    /// tile size is a property of the machine, so the first load measures and caches its winners
    /// instead of asking the operator to choose them.
    pub autotune: bool,
    /// Capture the chunked gated-delta prompt kernel instead of the exact per-token kernel.
    /// The two lowerings differ in the last bits, so this is an explicit numerical choice.
    pub chunked_recurrent: bool,
    /// Target prefill attention numerical profile. `Canonical` preserves the existing exact
    /// path; `Bf16` is the experimental FA4 profile for target FP8-KV prefill only. The profile
    /// is resolved once here and never changes the draft, weights, KV storage or quality gate.
    pub attention_profile: crate::AttentionProfile,
    /// Bind the vision tower; false serves text only, as vLLM does with all-zero multimodal
    /// limits, and rejects image inputs.
    pub vision: bool,
    /// Separate `DFlash2` draft package to speculate with; excludes an MTP depth.
    pub draft_model: Option<PathBuf>,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            prefill_width: 1,
            verification_width: 0,
            fp8_kv: None,
            mtp_depth: 0,
            autotune: true,
            chunked_recurrent: false,
            attention_profile: crate::AttentionProfile::Canonical,
            vision: true,
            draft_model: None,
        }
    }
}
impl LoadOptions {
    fn resolve_verification(&mut self) -> Result<()> {
        self.validate()?;
        if self.mtp_depth > 0 {
            let width = self.mtp_depth + 1;
            if self.verification_width == 0 {
                self.verification_width = width;
            }
            if self.verification_width != width {
                return Err(Error::invalid(
                    "verification width must equal MTP depth plus one",
                ));
            }
        }
        self.validate()
    }
    fn validate(&self) -> Result<()> {
        if ![
            0,
            1,
            FUSED_VERIFY_LANES,
            PREFILL_LANES,
            MID_PREFILL_LANES,
            MAX_PREFILL_LANES,
            WIDE_PREFILL_LANES,
            EXTRA_WIDE_PREFILL_LANES,
            WIDER_PREFILL_LANES,
            WIDEST_PREFILL_LANES,
        ]
        .contains(&self.prefill_width)
            || self.verification_width > MAX_VERIFICATION_WIDTH
            || self.verification_width == 1
            || (self.prefill_width == FUSED_VERIFY_LANES
                && ![0, FUSED_VERIFY_LANES].contains(&self.verification_width))
        {
            return Err(Error::invalid("CUDA model prefill/verification widths"));
        }
        if self.mtp_depth > MAX_MTP_DEPTH {
            return Err(Error::invalid("CUDA MTP draft depth"));
        }
        if self.mtp_depth > 0 && self.draft_model.is_some() {
            return Err(Error::invalid(
                "an MTP depth and a separate draft model are exclusive",
            ));
        }
        Ok(())
    }
}

/// Immutable draft graph and weights for MTP speculation.
struct Draft {
    graph: DataflowGraph,
    weights: ProgramWeights,
}

/// Immutable shared model resources. CUDA graphs retain weights independently of this factory.
/// Mutable Conv/GDN/KV and activations are allocated separately for each request.
pub struct LoadedModel {
    device: CudaDevice,
    profile: DeviceProfile,
    model: ModelIr,
    graph: DataflowGraph,
    weights: ProgramWeights,
    draft: Option<Draft>,
    /// Attached `DFlash2` draft weights, shared by the executor's draft engine.
    dflash2: Option<Arc<DraftWeights>>,
    vision: Option<LoadedVision>,
    imported: infer_spi::ImportedModel,
    mtp_depth: usize,
    /// Whether the prompt graph width was chosen automatically at load time.
    automatic_prefill: bool,
    tuning: TuningReport,
    requirements: infer_ir::CapabilityRequirements,
    sequence_prices: budget::SequencePrices,
}

/// Vision tower bound alongside the text model.
///
/// Encoding runs the real kernels; a family without an image encoder simply has no `LoadedVision`,
/// so a caller holding one can never silently skip encoding.
pub struct LoadedVision {
    weights: VisionWeights,
    encoder: infer_spi::ModalityEncoder,
}

impl LoadedVision {
    /// Bind the declared encoder, or nothing when the family has no image modality.
    /// # Errors
    /// Rejects declared encoders whose weights or geometry are unusable.
    fn bind(device: &CudaDevice, package: &mut QuantizedPackage) -> Result<Option<Self>> {
        let declared = package
            .imported
            .modalities
            .iter()
            .find(|plan| plan.modality == infer_ir::Modality::Image)
            .and_then(|plan| plan.encoder.clone());
        let Some(encoder) = declared else {
            return Ok(None);
        };
        let weights = VisionWeights::load(device, package, &encoder)?;
        Ok(Some(Self { weights, encoder }))
    }

    /// Declared geometry of the bound tower.
    #[must_use]
    pub const fn encoder(&self) -> &infer_spi::ModalityEncoder {
        &self.encoder
    }

    /// Checked request-owned device bytes for a prepared media plan.
    /// # Errors
    /// Rejects empty/overflowing grids or encoder geometry mismatches.
    pub fn footprint(&self, media: &infer_ir::MediaPlan) -> Result<u64> {
        footprint(&self.encoder, media)
    }

    /// Exact bytes of every retained raw and prepared vision owner.
    #[must_use]
    pub fn inventory_bytes(&self) -> u64 {
        self.weights.inventory_bytes()
    }

    /// Encode one preprocessed image into `merged_tokens × text_hidden` embeddings.
    /// # Errors
    /// Rejects a patch grid that disagrees with the pixel buffer or failed CUDA execution.
    pub fn encode(
        &self,
        device: &CudaDevice,
        image: &infer_models::PromptImage,
    ) -> Result<Vec<f32>> {
        let (temporal, height, width) = image.grid;
        let patches = temporal
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| Error::invalid("image patch grid overflow"))?;
        if patches == 0 {
            return Err(Error::invalid("image patch grid is empty"));
        }
        let width = patch_width(&self.encoder)?;
        if image.pixels.len() != patches * width {
            return Err(Error::invalid("image pixels do not match the patch grid"));
        }
        // The position contribution is a host-side gather of the learned table, then the tower
        // consumes patch embeddings and the merger folds each merge group back into text width.
        let taps = position_taps(image.grid, &self.encoder)?;
        let positions = gather_positions(self.weights.position_table(), &self.encoder, &taps)?;
        let embedded = patch_embed(
            device,
            &self.encoder,
            &self.weights,
            &image.pixels,
            &positions,
            patches,
        )?;
        let hidden = tower(
            device,
            &self.weights,
            &self.encoder,
            &embedded,
            image.grid,
            patches,
        )?;
        merger(device, &self.weights, &self.encoder, &hidden, patches)
    }
}

impl LoadedModel {
    pub(crate) fn draft_graph(&self) -> Option<&DataflowGraph> {
        self.draft.as_ref().map(|draft| &draft.graph)
    }

    /// Attached `DFlash2` draft weights.
    pub(crate) const fn dflash2(&self) -> Option<&Arc<DraftWeights>> {
        self.dflash2.as_ref()
    }

    /// Tokens one speculative step may propose: the MTP depth or the `DFlash2` block's masks.
    #[must_use]
    pub fn draft_depth(&self) -> usize {
        self.dflash2
            .as_ref()
            .map_or(self.mtp_depth, |draft| draft.geometry.block - 1)
    }

    /// Target lanes one pooled verification replays per slot, zero without speculation.
    fn speculation_width(&self) -> usize {
        match self.draft_depth() {
            0 => 0,
            depth => depth + 1,
        }
    }

    /// Effective graph widths and speculation depth after automatic resolution.
    #[must_use]
    pub fn execution_profile(&self) -> infer_ir::ExecutionProfileInspection {
        infer_ir::ExecutionProfileInspection {
            concurrency: None,
            prefill_width: self.weights.prefill_width,
            pooled_prefill_width: None,
            batch_width: self.weights.batch_width,
            draft_depth: self.draft_depth(),
            automatic_prefill: self.automatic_prefill,
            arena_budget_bytes: self.profile.arena_budget_bytes(),
            attention_profile: self.weights.attention_profile.name().to_owned(),
            precise_prefix_positions: Some(precise_prefix_positions()),
        }
    }
    /// # Errors
    /// Rejects unsupported formats, invalid policies, budgets or CUDA loading errors.
    pub fn open(
        device: CudaDevice,
        root: impl AsRef<Path>,
        id: ModelId,
        mut options: LoadOptions,
    ) -> Result<Self> {
        options.resolve_verification()?;
        // Query hardware once: the profile keys machine-local tuning artifacts and derives the
        // device budget policy, so no per-model or per-board table needs maintaining here.
        let profile = device.profile()?.clone();
        let package_phase = phase::Phase::start(
            "package",
            format!(
                "reading checkpoint metadata from {}",
                root.as_ref().display()
            ),
        );
        let mut package = QuantizedPackage::open(root, id).map_err(|e| package_phase.fail(&e))?;
        attach_draft(&mut package, &mut options).map_err(|e| package_phase.fail(&e))?;
        package_phase.finish();
        options.fp8_kv =
            Some(options.fp8_kv.unwrap_or_else(|| {
                package.kv_cache_dtype == Some(infer_models::TensorDtype::F8E4m3)
            }));
        let automatic_prefill = options.prefill_width == 0;
        if automatic_prefill {
            options.prefill_width = PREFILL_LANES;
        }
        let mtp_depth = options.mtp_depth;
        let (mut weights, mut draft, tuning) =
            bindings::load(&device, &profile, &mut package, &options)?;
        let dflash2 = bind_dflash2(&device, &mut package, &weights)?;
        if automatic_prefill
            && !weights.projections.iter().any(|(id, weight)| {
                matches!(weight, ProjectionWeight::Fp4(..))
                    && !weights.input_scales.contains_key(id)
            })
        {
            let geometry_phase =
                phase::Phase::start("geometry", "resolving the prompt graph width");
            // One request owns one arena, so the width decision is bounded by the profiled
            // activation-arena budget, not by a fixed share of total memory.
            //
            // Both this budget and the recurrent cap below are load-bearing: letting live
            // free memory fund the 128-lane 27B prompt graph measured worse on three of four
            // cases (short TTFT 0.058 -> 0.084 s, batch4 0.247 -> 0.291 s, hot-prefix
            // 0.090 -> 0.115 s) while the long case barely moved (0.436 -> 0.424 s). Do not
            // widen a quantized recurrent prompt graph without an end-to-end measurement.
            // Quantized recurrent graphs keep the scalar Delta path for numerical
            // stability. Limit their prompt graph size while sharing projection loads.
            // The arena is the real bound; this is only a cap for the ladder below. Quantized
            // recurrent graphs stay at the established width, dense ones may take the wide rung
            // when the arena has room for it.
            let quantized_recurrent = package
                .graph
                .nodes
                .iter()
                .any(|node| matches!(node.op, infer_ir::TensorOp::Delta { .. }))
                && (!weights.input_scales.is_empty() || !weights.fp8_inputs.is_empty());
            // One request owns one arena, so the width decision is bounded by the profiled
            // activation-arena budget, not by a fixed share of total memory.
            select_prompt_width(
                &package.graph,
                quantized_recurrent,
                profile.arena_budget_bytes(),
                &mut weights,
            )?;
            geometry_phase.finish();
        }
        // Newly supported wide explicit buckets need the same tail and draft geometry as
        // automatic wide prompts. The token cap is still enforced by the scheduler.
        if automatic_prefill || options.prefill_width > WIDE_PREFILL_LANES {
            if !automatic_prefill {
                weights.narrow_prefill_width = MID_PREFILL_LANES;
            }
            // The draft primes once per prompt chunk; matching the target width turns a 2048-lane
            // target chunk into one draft replay instead of sixteen 128-lane ones, whose attention
            // grid fills a fraction of the device.
            if let Some(draft) = draft.as_mut()
                && weights.prefill_width > draft.weights.prefill_width
                && prompt_logits::required_bytes(
                    &draft.graph,
                    &draft.weights,
                    weights.prefill_width,
                )? as u64
                    <= profile.arena_budget_bytes()
            {
                draft.weights.prefill_width = weights.prefill_width;
            }
            tracing::info!(
                target: "infer::load",
                prefill_width = weights.prefill_width,
                narrow_prefill_width = weights.narrow_prefill_width,
                draft_prefill_width = draft.as_ref().map_or(0, |d| d.weights.prefill_width),
                "prompt graph width selected"
            );
        }
        let requirements = package.imported.requirements.clone();
        let vision = bind_vision_phase(&device, &mut package, options.vision)?;
        let imported = package.imported.clone();
        let loaded = Self {
            device,
            profile,
            model: package.imported.model,
            graph: package.graph,
            weights,
            draft,
            dflash2,
            vision,
            imported,
            mtp_depth,
            automatic_prefill,
            tuning,
            requirements,
            sequence_prices: budget::SequencePrices::default(),
        };
        loaded.prepare_attention()?;
        tracing::info!(
            target: "infer::load",
            attention_profile = loaded.weights.attention_profile.name(),
            "target attention numerical profile"
        );
        Ok(loaded)
    }
    fn prepare_attention(&self) -> Result<()> {
        let phase = phase::Phase::start("attention_tuning", "preparing hardware attention choices");
        prepare(&self.device, &self.graph, &self.weights).map_err(|e| phase.fail(&e))?;
        if let Some(draft) = &self.draft {
            prepare(&self.device, &draft.graph, &draft.weights).map_err(|e| phase.fail(&e))?;
        }
        phase.finish();
        Ok(())
    }
    /// Provider-imported description of this model, including its modalities.
    #[must_use]
    pub const fn imported(&self) -> &infer_spi::ImportedModel {
        &self.imported
    }

    /// Bound vision tower, absent for text-only families.
    #[must_use]
    pub const fn vision(&self) -> Option<&LoadedVision> {
        self.vision.as_ref()
    }
    /// Device capabilities this model's provider declared as required.
    #[must_use]
    pub const fn requirements(&self) -> &infer_ir::CapabilityRequirements {
        &self.requirements
    }
    /// Speculative draft depth configured at load time; zero disables speculation.
    #[must_use]
    pub const fn mtp_depth(&self) -> usize {
        self.mtp_depth
    }
    /// Automatic tiling decisions from this load: what was measured, and what fell back.
    #[must_use]
    pub const fn tuning(&self) -> &TuningReport {
        &self.tuning
    }
    #[must_use]
    pub const fn model(&self) -> &ModelIr {
        &self.model
    }
    #[must_use]
    pub const fn graph(&self) -> &DataflowGraph {
        &self.graph
    }
    #[must_use]
    pub const fn profile(&self) -> &DeviceProfile {
        &self.profile
    }
    #[must_use]
    pub const fn device(&self) -> &CudaDevice {
        &self.device
    }
    /// # Errors
    /// Rejects invalid capacity, exceeded state budgets or CUDA graph capture failures.
    pub fn sequence(&self, capacity: usize) -> Result<DeviceProgram> {
        self.validate_capacity(capacity)?;
        DeviceProgram::new(
            &self.device,
            &self.graph,
            &self.weights,
            capacity,
            self.model.hidden_size,
            self.model.vocab_size,
        )
    }
    /// A sequence that will speculate through the shared slot pool skips its private
    /// verification graph; the pool's captured graph replaces it.
    /// # Errors
    /// Rejects invalid capacity, exceeded state budgets or CUDA graph capture failures.
    pub(crate) fn sequence_pooled(&self, capacity: usize) -> Result<DeviceProgram> {
        self.validate_capacity(capacity)?;
        DeviceProgram::new_pooled(
            &self.device,
            &self.graph,
            &self.weights,
            capacity,
            self.model.hidden_size,
            self.model.vocab_size,
        )
    }
    pub(crate) fn ensure_verification(&self, program: &mut DeviceProgram) -> Result<()> {
        program.ensure_verification(&self.graph, &self.weights)
    }
    /// # Errors
    /// Rejects invalid capacity, missing draft configuration or capture failures.
    pub fn draft(&self, capacity: usize) -> Result<Option<DeviceProgram>> {
        self.validate_capacity(capacity)?;
        let Some(draft) = &self.draft else {
            return Ok(None);
        };
        DeviceProgram::new(
            &self.device,
            &draft.graph,
            &draft.weights,
            capacity,
            self.model.hidden_size,
            self.model.vocab_size,
        )
        .map(Some)
    }

    pub(crate) fn draft_slot_pool(
        &self,
        width: usize,
        capacity: usize,
    ) -> Result<Option<SlotPool>> {
        self.validate_capacity(capacity)?;
        let Some(draft) = &self.draft else {
            return Ok(None);
        };
        if draft.weights.fusion.is_none()
            || draft.graph.tensors.iter().any(|tensor| {
                matches!(
                    tensor.storage,
                    infer_ir::TensorStorage::State {
                        kind: infer_ir::StateKind::Conv | infer_ir::StateKind::LinearAttention,
                        ..
                    }
                )
            })
        {
            return Ok(None);
        }
        SlotPool::new(
            &self.device,
            &draft.graph,
            &draft.weights,
            width,
            capacity,
            self.model.hidden_size,
            (self.model.vocab_size, 0),
            None,
        )
        .map(Some)
    }

    /// Continuous-batching slot pool over the target weights: `width` slots of `capacity`
    /// tokens each, plus one shared graph captured against them. With a draft loaded the
    /// graph is the pooled speculation graph of `mtp_depth + 1` candidate lanes per slot.
    /// # Errors
    /// Rejects invalid geometry, exceeded device budgets or CUDA capture failures.
    /// Wide-prompt target and draft programs at `capacity` that slot-resident sequences prompt
    /// through.
    /// # Errors
    /// Rejects invalid capacity or CUDA graph capture failures.
    pub(crate) fn prompt_worker(
        &self,
        capacity: usize,
        kv: Option<Fp8Caches>,
        prompt_width: usize,
    ) -> Result<(DeviceProgram, Option<DeviceProgram>)> {
        self.validate_capacity(capacity)?;
        let geometry = (capacity, self.model.hidden_size, self.model.vocab_size);
        let program = DeviceProgram::new_worker(
            &self.device,
            &self.graph,
            &self.weights,
            geometry,
            kv,
            prompt_width,
        )?;
        let draft = self
            .draft
            .as_ref()
            .map(|draft| {
                DeviceProgram::new_worker(
                    &self.device,
                    &draft.graph,
                    &draft.weights,
                    geometry,
                    None,
                    prompt_width,
                )
            })
            .transpose()?;
        Ok((program, draft))
    }

    /// Target and draft programs that prefill straight into `slot` of the target and draft
    /// pools: their recurrent states and KV caches are the slots' storage.
    /// # Safety
    /// Both pools must outlive the programs, and a program may replay only while its sequence
    /// owns the slot.
    /// # Errors
    /// Rejects unknown slots or CUDA graph capture failures.
    #[expect(
        unsafe_code,
        reason = "Audited slot alias: forwarded from SlotPool::slot_views"
    )]
    pub(crate) unsafe fn slot_programs(
        &self,
        target: &SlotPool,
        draft: Option<&SlotPool>,
        slot: usize,
    ) -> Result<(DeviceProgram, Option<DeviceProgram>)> {
        let geometry = (
            target.capacity(),
            self.model.hidden_size,
            self.model.vocab_size,
        );
        // SAFETY: forwarded from this function's contract.
        let views = unsafe { target.slot_views(slot)? };
        let program =
            DeviceProgram::new_on_slot(&self.device, &self.graph, &self.weights, geometry, views)?;
        let draft = match (&self.draft, draft) {
            (Some(model), Some(pool)) => {
                // SAFETY: forwarded from this function's contract.
                let views = unsafe { pool.slot_views(slot)? };
                Some(DeviceProgram::new_on_slot(
                    &self.device,
                    &model.graph,
                    &model.weights,
                    geometry,
                    views,
                )?)
            }
            _ => None,
        };
        Ok((program, draft))
    }

    /// Drop unused prompt widths before a different serving geometry is allocated.
    pub(crate) fn reclaim_idle_prompt_scratch(&self) -> Result<usize> {
        let mut released = self.weights.prompt_scratch.reclaim_unused(&self.device)?;
        if let Some(draft) = &self.draft {
            released += draft.weights.prompt_scratch.reclaim_unused(&self.device)?;
        }
        Ok(released)
    }

    pub(crate) fn slot_pool(
        &self,
        width: usize,
        capacity: usize,
        kv_tokens: Option<usize>,
    ) -> Result<SlotPool> {
        self.validate_capacity(capacity)?;
        SlotPool::new(
            &self.device,
            &self.graph,
            &self.weights,
            width,
            capacity,
            self.model.hidden_size,
            (self.model.vocab_size, self.speculation_width()),
            kv_tokens,
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/loading_mod.rs"]
mod tests;

/// Attach `options.draft_model` to `package` and verify with its block width.
fn attach_draft(package: &mut QuantizedPackage, options: &mut LoadOptions) -> Result<()> {
    let Some(draft) = &options.draft_model else {
        return Ok(());
    };
    package.attach_dflash2(draft)?;
    let block = package
        .dflash2
        .as_ref()
        .map_or(0, |draft| draft.config.dflash_config.block_size);
    if !(2..=MAX_VERIFICATION_WIDTH).contains(&block) {
        return Err(Error::unsupported(format!(
            "DFlash2 block {block} exceeds the verification width limit"
        )));
    }
    options.verification_width = block;
    Ok(())
}

/// Upload the attached `DFlash2` draft, sharing the target's embedding and head.
fn bind_dflash2(
    device: &CudaDevice,
    package: &mut QuantizedPackage,
    weights: &ProgramWeights,
) -> Result<Option<Arc<DraftWeights>>> {
    if package.dflash2.is_none() {
        return Ok(None);
    }
    let phase = phase::Phase::start("dflash2", "binding the DFlash2 draft");
    let draft = DraftWeights::load(device, package, weights).map_err(|e| phase.fail(&e))?;
    phase.finish();
    tracing::info!(target: "infer::load", bytes = draft.bytes(), "DFlash2 draft bound");
    Ok(Some(Arc::new(draft)))
}

/// Pick the prompt graph width from the arena budget.
///
/// A wider prompt graph is arithmetically identical to several narrower chunks: keys and values
/// always pass through the cache, and the prompt GEMM accumulates per element over K. It is also
/// fewer replays, which is where the measured gains come from. The 256-lane rung is only offered
/// to quantized recurrent models, where it measured -4.4% long-prompt TTFT with identical tokens;
/// on the dense models it measured neutral on the long case and 1.3% worse on the short one.
fn select_prompt_width(
    graph: &DataflowGraph,
    quantized_recurrent: bool,
    arena_budget: u64,
    weights: &mut ProgramWeights,
) -> Result<()> {
    let recurrent_limit = if quantized_recurrent {
        MID_PREFILL_LANES
    } else {
        MAX_PREFILL_LANES
    };
    let mut selected = 0usize;
    for width in [MAX_PREFILL_LANES, MID_PREFILL_LANES]
        .into_iter()
        .filter(|&width| width <= recurrent_limit)
    {
        let needed = prompt_logits::required_bytes(graph, weights, width)?;
        if needed as u64 <= arena_budget {
            selected = width;
            break;
        }
    }
    if selected != 0 && selected < MAX_PREFILL_LANES {
        let rungs: &[usize] = if quantized_recurrent {
            &[
                WIDEST_PREFILL_LANES,
                WIDER_PREFILL_LANES,
                EXTRA_WIDE_PREFILL_LANES,
                WIDE_PREFILL_LANES,
                MAX_PREFILL_LANES,
            ]
        } else {
            &[MAX_PREFILL_LANES]
        };
        for &wide in rungs {
            // Target and draft retain separate prompt workspaces. Keep the automatic
            // fast pair on the joint 1024-row rung so long private states remain fundable.
            if weights.batch_width > 1
                && weights.attention_profile.is_bf16()
                && wide > WIDER_PREFILL_LANES
            {
                continue;
            }
            let needed = prompt_logits::required_bytes(graph, weights, wide)?;
            tracing::info!(
                target: "infer::load",
                width = wide,
                arena_bytes = needed,
                arena_budget,
                "prompt width rung"
            );
            if needed as u64 <= arena_budget {
                weights.narrow_prefill_width = selected;
                selected = wide;
                break;
            }
        }
    }
    if selected != 0 {
        weights.prefill_width = selected;
    }
    Ok(())
}

/// Bind the optional tower with a single phase even when text-only serving skips it.
fn bind_vision_phase(
    device: &CudaDevice,
    package: &mut QuantizedPackage,
    enabled: bool,
) -> Result<Option<LoadedVision>> {
    if !enabled {
        tracing::info!(target: "infer::load", "text-only serving: vision tower not loaded");
        return Ok(None);
    }
    let phase = phase::Phase::start("vision", "binding the vision tower");
    let vision = bind_vision(device, package, true, &phase)?;
    phase.finish();
    Ok(vision)
}

/// The vision tower, unless the service is text only.
fn bind_vision(
    device: &CudaDevice,
    package: &mut QuantizedPackage,
    enabled: bool,
    phase: &phase::Phase,
) -> Result<Option<LoadedVision>> {
    if !enabled {
        tracing::info!(target: "infer::load", "text-only serving: vision tower not loaded");
        return Ok(None);
    }
    LoadedVision::bind(device, package).map_err(|e| phase.fail(&e))
}

#[cfg(test)]
#[path = "../../tests/unit/loading_support.rs"]
mod support;

#[cfg(test)]
#[path = "../../tests/unit/loading_shape_ab.rs"]
mod shape_ab_tests;

#[cfg(test)]
#[path = "../../tests/unit/loading_prefill_width.rs"]
mod prefill_width_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_quant_reuse_model.rs"]
mod quant_reuse_model_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_attention_prefill_model.rs"]
mod attention_prefill_model_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_decode_path.rs"]
mod decode_path_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_prefill_demand_model.rs"]
mod prefill_demand_model_tests;

#[cfg(test)]
#[path = "../../tests/unit/loading_parallel_quant.rs"]
mod parallel_quant_tests;

#[cfg(test)]
#[path = "../../tests/unit/loading_parallel_quant_pool.rs"]
mod parallel_quant_pool_tests;

#[cfg(test)]
#[path = "../../tests/unit/loading_pool_gemv_cost.rs"]
mod pool_gemv_cost_tests;

#[cfg(test)]
#[path = "../../tests/unit/loading_pool_gemv_model.rs"]
mod pool_gemv_model_tests;

#[cfg(test)]
#[path = "../../tests/unit/loading_pool_greedy_lifecycle.rs"]
mod pool_greedy_lifecycle_tests;

#[cfg(test)]
#[path = "../../tests/unit/loading_oxide_attention.rs"]
mod oxide_attention_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_recurrent_oxide_model.rs"]
mod oxide_gdn_tests;

#[cfg(test)]
#[path = "../../tests/unit/loading_gated_norm_model.rs"]
mod gated_norm_model_tests;

#[cfg(test)]
#[path = "../../tests/unit/loading_prompt_width_buckets.rs"]
mod prompt_width_bucket_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_norm_maximum_model.rs"]
mod norm_maximum_model_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_norm_fp4_quant.rs"]
mod norm_fp4_quant_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_norm_fp4_model.rs"]
mod norm_fp4_model_tests;

#[cfg(test)]
#[path = "../../tests/unit/vendor_nvfp4_fixture.rs"]
pub(crate) mod nvfp4_fixture;

#[cfg(test)]
#[path = "../../tests/unit/resident_vendor_fp4_model.rs"]
mod vendor_fp4_model_tests;
