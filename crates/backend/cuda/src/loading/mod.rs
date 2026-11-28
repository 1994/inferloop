//! Native model loading independent of diagnostic runners and CPU reference execution.
use crate::attention_profile::precise_prefix_positions;
use crate::constants::EXTRA_WIDE_PREFILL_LANES;
use crate::constants::WIDER_PREFILL_LANES;
use crate::constants::WIDEST_PREFILL_LANES;
use crate::resident::prefill_burst::{self, PromptBurst};
use crate::resident::prompt_logits;
use crate::resident::require_kernel_geometry;
use head_padding::pad_package;
use std::path::PathBuf;
mod concurrency;
pub(crate) mod grouped_int_tiled;
pub use grouped_int_tiled::GroupedIntTiled;
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

use crate::device::image_preprocess::Capability as ImageCapability;
use crate::device::{CudaImageProcessor as ImageProcessor, ImageBatch};
use crate::resident::narrow_kv::NarrowCaches;
use crate::resident::slot_batch::SlotPool;
use crate::tuning::attention::mma::prepare;
use crate::vision::VisionWeights;
use crate::vision::encode_media;
use crate::vision::footprint;
use crate::vision::patch_width;
use crate::vision::program::patch_embed_device_pixels;
use crate::vision::program::patch_embed_device_positions;
use infer_models::vision::position_taps;
pub use kv_scales::load as load_kv_scales;
mod bindings;
mod budget;
pub(crate) use budget::PooledServingQuote;
pub(crate) use budget::kv_block_bytes;
mod experts;
mod head_padding;
mod phase;
use crate::resident::dflash2::DraftWeights;
use crate::{
    device::CudaDevice,
    resident::{DeviceProgram, ProgramWeights},
    tuning::TuningReport,
};
use infer_core::{Error, ModelId, Result};
use infer_cuda_kernels::runtime::Allocation;
use infer_ir::{DataflowGraph, EncodedImageInput, ImageInput, ImageTransform, ModelIr};
use infer_models::QuantizedPackage;
use infer_spi::ImagePreprocessing;
use std::path::Path;
use std::sync::Arc;
/// Interleaved channels in a decoded RGB image.
const IMAGE_CHANNELS: usize = 3;
pub use weights::floats;

/// The KV storage `package` declares: its checkpoint's KV scheme or its provider's default, else
/// F32.
/// # Errors
/// Rejects a declared dtype no KV cache stores.
fn declared_kv_cache(package: &QuantizedPackage) -> Result<infer_ir::KvCacheEncoding> {
    use infer_ir::KvCacheEncoding;
    use infer_models::TensorDtype;
    match package.kv_cache_dtype {
        None | Some(TensorDtype::F32) => Ok(KvCacheEncoding::F32),
        Some(TensorDtype::F16) => Ok(KvCacheEncoding::F16),
        Some(TensorDtype::F8E4m3) => Ok(KvCacheEncoding::Fp8E4m3),
        Some(other) => Err(Error::unsupported(format!(
            "declared KV cache dtype {other:?}"
        ))),
    }
}

/// Load-time execution policy; each request receives independent mutable device state.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub prefill_width: usize,
    pub verification_width: usize,
    /// KV storage; `None` follows the package's declaration, then F32.
    pub kv_cache: Option<infer_ir::KvCacheEncoding>,
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
    /// Storage of the target's gated-delta states; `F32` unless a numerical acceptance run selects
    /// BF16. Resolved once here like the attention profile.
    pub recurrent_profile: crate::RecurrentProfile,
    /// Bind the vision tower; false serves text only, as vLLM does with all-zero multimodal
    /// limits, and rejects image inputs.
    pub vision: bool,
    /// Separate `DFlash2` draft package to speculate with; excludes an MTP depth.
    pub draft_model: Option<PathBuf>,
    /// How much device time to measure; fixed here because measurement points are captured.
    pub device_timing: infer_ir::DeviceTimingConfig,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            prefill_width: 1,
            verification_width: 0,
            kv_cache: None,
            mtp_depth: 0,
            autotune: true,
            chunked_recurrent: false,
            attention_profile: crate::AttentionProfile::Canonical,
            recurrent_profile: crate::RecurrentProfile::Bf16,
            vision: true,
            draft_model: None,
            device_timing: infer_ir::DeviceTimingConfig::default(),
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
    device_timing: infer_ir::DeviceTimingConfig,
}

/// Vision tower bound alongside the text model.
///
/// Encoding runs the real kernels; a family without an image encoder simply has no `LoadedVision`,
/// so a caller holding one can never silently skip encoding.
pub struct LoadedVision {
    weights: VisionWeights,
    encoder: infer_spi::ModalityEncoder,
    image_preprocessor: Option<ImageProcessor>,
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
        let image_preprocessor = ImageProcessor::load(device)?;
        Ok(Some(Self {
            weights,
            encoder,
            image_preprocessor,
        }))
    }

    pub(crate) fn image_preprocessing(&self) -> Option<Arc<dyn ImagePreprocessing>> {
        self.image_preprocessor
            .as_ref()
            .map(ImageProcessor::capability)
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

    /// Encode one image and return the final embedding plus declared deepstack features.
    /// # Errors
    /// Rejects a patch grid that disagrees with the pixel buffer or failed CUDA execution.
    pub fn encode_with_deepstack(
        &self,
        device: &CudaDevice,
        image: &infer_models::PromptImage,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let encoded = self.encode_media(device, image)?;
        Ok((encoded.embeddings, encoded.features))
    }

    /// Encode one image into everything the text decoder reads for it.
    /// # Errors
    /// Rejects a patch grid that disagrees with the pixel buffer or failed CUDA execution.
    pub fn encode_media(
        &self,
        device: &CudaDevice,
        image: &infer_models::PromptImage,
    ) -> Result<MediaEncoding> {
        self.encode_pixels(device, image.grid, &image.pixels)
    }

    /// Encode one preprocessed image into `merged_tokens × text_hidden` embeddings.
    /// # Errors
    /// Rejects a patch grid that disagrees with the pixel buffer or failed CUDA execution.
    pub fn encode(
        &self,
        device: &CudaDevice,
        image: &infer_models::PromptImage,
    ) -> Result<Vec<f32>> {
        self.encode_pixels(device, image.grid, &image.pixels)
            .map(|encoded| encoded.embeddings)
    }

    pub(crate) fn encode_pixels(
        &self,
        device: &CudaDevice,
        grid: (usize, usize, usize),
        pixels: &[f32],
    ) -> Result<MediaEncoding> {
        let (temporal, height, width) = grid;
        let patches = temporal
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| Error::invalid("image patch grid overflow"))?;
        if patches == 0 {
            return Err(Error::invalid("image patch grid is empty"));
        }
        let width = patch_width(&self.encoder)?;
        if patches.checked_mul(width) != Some(pixels.len()) {
            return Err(Error::invalid("image pixels do not match the patch grid"));
        }
        // The tower consumes patch embeddings and the merger folds each group into text width.
        let taps = position_taps(grid, &self.encoder)?;
        let embedded = patch_embed_device_positions(
            device,
            &self.encoder,
            &self.weights,
            pixels,
            &taps,
            patches,
        )?;
        self.finish_encoding(device, grid, patches, &embedded)
    }

    pub(crate) async fn encode_image(
        &self,
        device: &CudaDevice,
        grid: (usize, usize, usize),
        image: &ImageInput,
    ) -> Result<MediaEncoding> {
        let transform = image.transform();
        let patches = self.validate_image_transform(grid, transform)?;
        let processor = self.image_preprocessor()?;
        let batch = ImageBatch::new(image.shared_pixels(), 1, transform)?;
        let input = processor.process(device, &batch).await?;
        self.encode_image_patches(device, grid, patches, &input)
    }

    pub(crate) async fn encode_encoded_image(
        &self,
        device: &CudaDevice,
        grid: (usize, usize, usize),
        image: &EncodedImageInput,
    ) -> Result<MediaEncoding> {
        let patches = self.validate_image_transform(grid, image.transform())?;
        let input = self
            .image_preprocessor()?
            .process_encoded(device, image)
            .await?;
        self.encode_image_patches(device, grid, patches, &input)
    }

    fn validate_image_transform(
        &self,
        grid: (usize, usize, usize),
        transform: ImageTransform,
    ) -> Result<usize> {
        let layout = transform.layout();
        if transform.grid() != grid
            || layout.patch() != self.encoder.patch_size
            || layout.temporal() != self.encoder.temporal_patch_size
            || layout.merge() != self.encoder.spatial_merge_size
            || self.encoder.in_channels != IMAGE_CHANNELS
            || !ImageCapability.supports(&transform)
        {
            return Err(Error::unsupported(
                "CUDA image processor cannot execute this transform",
            ));
        }
        grid.0
            .checked_mul(grid.1)
            .and_then(|value| value.checked_mul(grid.2))
            .ok_or_else(|| Error::invalid("image patch grid overflow"))
    }

    fn image_preprocessor(&self) -> Result<&ImageProcessor> {
        self.image_preprocessor
            .as_ref()
            .ok_or_else(|| Error::unsupported("CUDA image processor is unavailable"))
    }

    fn encode_image_patches(
        &self,
        device: &CudaDevice,
        grid: (usize, usize, usize),
        patches: usize,
        input: &Allocation<f32>,
    ) -> Result<MediaEncoding> {
        let taps = position_taps(grid, &self.encoder)?;
        let embedded =
            patch_embed_device_pixels(device, &self.encoder, &self.weights, input, &taps, patches)?;
        self.finish_encoding(device, grid, patches, &embedded)
    }

    fn finish_encoding(
        &self,
        device: &CudaDevice,
        grid: (usize, usize, usize),
        patches: usize,
        embedded: &[f32],
    ) -> Result<MediaEncoding> {
        let (embeddings, features) = encode_media(
            device,
            &self.weights,
            &self.encoder,
            embedded,
            grid,
            patches,
        )?;
        Ok(MediaEncoding {
            embeddings,
            features,
        })
    }
}

/// One image in text width: its merged embedding rows and, per declared deepstack level, the
/// feature rows the text decoder adds to the same tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaEncoding {
    pub embeddings: Vec<f32>,
    pub features: Vec<Vec<f32>>,
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
        let padding = pad_package(&mut package, &options).map_err(|e| package_phase.fail(&e))?;
        require_kernel_geometry(&package.graph).map_err(|e| package_phase.fail(&e))?;
        package_phase.finish();
        let kv_cache = match options.kv_cache {
            Some(explicit) => explicit,
            None => declared_kv_cache(&package)?,
        };
        options.kv_cache = Some(kv_cache);
        let automatic_prefill = options.prefill_width == 0;
        if automatic_prefill {
            options.prefill_width = PREFILL_LANES;
        }
        let mtp_depth = options.mtp_depth;
        let (mut weights, mut draft, tuning) =
            bindings::load(&device, &profile, &mut package, &options, &padding)?;
        let dflash2 = bind_dflash2(&device, &mut package, &weights)?;
        if automatic_prefill
            && !weights
                .projections
                .iter()
                .any(|(id, weight)| weight.is_nvfp4() && !weights.input_scales.contains_key(id))
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
            let quantized_recurrent = quantized_recurrent(&package.graph, &weights);
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
        if !automatic_prefill {
            require_prompt_width_fits(&package.graph, &weights, profile.arena_budget_bytes())?;
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
            device_timing: options.device_timing,
        };
        loaded.prepare_attention()?;
        tracing::info!(
            target: "infer::load",
            attention_profile = loaded.weights.attention_profile.name(),
            recurrent_profile = loaded.weights.recurrent_profile.name(),
            "target attention and recurrent-state numerical profiles"
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
    /// Device timing the model loaded with.
    #[must_use]
    pub const fn device_timing(&self) -> infer_ir::DeviceTimingConfig {
        self.device_timing
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

    /// The draft head's slot pool beside a target pool of `width` slots of `capacity` tokens.
    /// `kv_tokens` pages its KV like the target's, from a ledger of its own.
    /// # Errors
    /// Rejects invalid geometry, exceeded device budgets or CUDA capture failures.
    pub(crate) fn draft_slot_pool(
        &self,
        width: usize,
        capacity: usize,
        kv_tokens: Option<usize>,
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
            kv_tokens,
        )
        .map(Some)
    }

    /// Continuous-batching slot pool over the target weights: `width` slots of `capacity`
    /// tokens each, plus one shared graph captured against them. With a draft loaded the
    /// graph is the pooled speculation graph of `mtp_depth + 1` candidate lanes per slot.
    /// # Errors
    /// Rejects invalid geometry, exceeded device budgets or CUDA capture failures.
    /// Wide-prompt target and draft programs at `capacity` that slot-resident sequences prompt
    /// through, over the target and draft pools' shared paged caches when they page.
    /// # Errors
    /// Rejects invalid capacity or CUDA graph capture failures.
    pub(crate) fn prompt_worker(
        &self,
        capacity: usize,
        (kv, draft_kv): (Option<NarrowCaches>, Option<NarrowCaches>),
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
                    draft_kv,
                    prompt_width,
                )
            })
            .transpose()?;
        Ok((program, draft))
    }

    /// One prompt graph over every resident program.
    /// # Safety
    /// The slot pool behind `programs` must outlive the graph.
    /// # Errors
    /// Rejects geometries the graph does not cover or CUDA capture failures.
    #[expect(
        unsafe_code,
        reason = "Forwards the caller's slot-pool lifetime contract to the burst capture"
    )]
    pub(crate) unsafe fn prompt_burst(
        &self,
        capacity: usize,
        programs: &mut [&mut DeviceProgram],
        shared: prefill_burst::Shared<'_>,
    ) -> Result<PromptBurst> {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            PromptBurst::new(
                &self.device,
                &self.graph,
                &self.weights,
                (capacity, self.model.hidden_size, self.model.vocab_size),
                programs,
                shared,
            )
        }
    }

    #[cfg(test)]
    pub(crate) const fn test_weights(&self) -> &ProgramWeights {
        &self.weights
    }

    /// Device bytes multi-sequence prompt graphs add to a pool beside a worker of `worker_rows`.
    /// # Errors
    /// Rejects invalid graphs or overflowing sizes.
    pub(crate) fn prompt_burst_bytes(&self, width: usize, worker_rows: usize) -> Result<u64> {
        prefill_burst::required_bytes(
            &self.graph,
            &self.weights,
            (width, worker_rows),
            (self.model.hidden_size, self.model.vocab_size),
        )
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
/// Whether the graph has recurrent (Delta) layers fed by quantized projections.
fn quantized_recurrent(graph: &DataflowGraph, weights: &ProgramWeights) -> bool {
    graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, infer_ir::TensorOp::Delta { .. }))
        && (!weights.input_scales.is_empty() || !weights.fp8_inputs.is_empty())
}

/// An explicit prompt width must fit the activation arena a request is funded with; otherwise
/// every prompt would fail when its arena is planned, long after the server reported ready.
fn require_prompt_width_fits(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    arena_budget: u64,
) -> Result<()> {
    let width = weights.prefill_width;
    let needed = prompt_logits::required_bytes(graph, weights, width)? as u64;
    if needed <= arena_budget {
        return Ok(());
    }
    let mut fitting = None;
    for candidate in [
        WIDEST_PREFILL_LANES,
        WIDER_PREFILL_LANES,
        EXTRA_WIDE_PREFILL_LANES,
        WIDE_PREFILL_LANES,
        MAX_PREFILL_LANES,
        MID_PREFILL_LANES,
        PREFILL_LANES,
    ] {
        if candidate < width
            && prompt_logits::required_bytes(graph, weights, candidate)? as u64 <= arena_budget
        {
            fitting = Some(candidate);
            break;
        }
    }
    let advice = fitting.map_or_else(
        || "no supported prompt width fits".to_owned(),
        |fitting| format!("the widest that fits is {fitting}"),
    );
    Err(Error::invalid(format!(
        "a {width}-token prompt step needs {needed} bytes of activation arena but the device \
         funds {arena_budget}; {advice}"
    )))
}

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
    // The widest rung the arena funds becomes the wide prompt graph, and the width chosen
    // above stays as each slot's narrow graph. Dense graphs start from 128 lanes and quantized
    // recurrent ones from 64, so only rungs above the narrow width are wide.
    if selected != 0 {
        let rungs = [
            WIDEST_PREFILL_LANES,
            WIDER_PREFILL_LANES,
            EXTRA_WIDE_PREFILL_LANES,
            WIDE_PREFILL_LANES,
            MAX_PREFILL_LANES,
        ];
        for wide in rungs.into_iter().filter(|&wide| wide > selected) {
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
mod support_tests;

#[cfg(test)]
mod shape_ab_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod prefill_width_tests;

#[cfg(test)]
mod quant_reuse_model_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod attention_prefill_model_tests;

#[cfg(test)]
mod decode_path_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod prefill_demand_model_tests;

#[cfg(test)]
mod parallel_quant_tests;

#[cfg(test)]
mod parallel_quant_pool_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod pool_gemv_cost_tests;

#[cfg(test)]
mod pool_gemv_model_tests;

#[cfg(test)]
mod pool_greedy_lifecycle_tests;

#[cfg(test)]
mod oxide_attention_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod gated_norm_model_tests;

#[cfg(test)]
mod prompt_width_bucket_tests;

#[cfg(test)]
mod norm_maximum_model_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod norm_fp4_quant_tests;

#[cfg(test)]
mod norm_fp4_model_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
pub(crate) mod nvfp4_fixture_tests;

#[cfg(test)]
mod vendor_fp4_model_tests;
