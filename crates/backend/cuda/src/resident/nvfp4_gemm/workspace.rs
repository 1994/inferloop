//! Per-program activation quantization buffers, retained as long as captured graphs.
use super::{
    kernels,
    quant_reuse::{CaptureQuantReuse, QuantKey, QuantKind, QuantPairRole},
};
use crate::constants::FP8_BLOCK_COLUMNS;
use crate::constants::NVFP4_GROUP_SIZE;
use crate::constants::NVFP4_QUANT_CODES_TILE;
use crate::constants::NVFP4_QUANT_SCALES_TILE;
use crate::constants::PREFILL_LANES;
use crate::tuning::fp8_gemm::native_search_trials;
use crate::tuning::fp8_gemm::prefill_choice;

use super::{SplitCandidate, projection_split_policy};
use crate::constants::VERIFY_SPLIT_SCRATCH_BUDGET;
use crate::resident::prefill_projection::fused_projections;
use crate::resident::program::ProgramGeometry;
use crate::tuning::fp8_gemm::Choice;
use crate::tuning::fp8_gemm::WeightKey;
use crate::tuning::fp8_gemm::probe_inputs::Scope as ProbeInputsScope;
use crate::tuning::fp8_gemm::select;
#[cfg(test)]
use crate::tuning::resident_gemm::MainLoop;
use crate::tuning::resident_gemm::ResidentTuningSession;
use crate::tuning::resident_gemm::SearchFamily;
use crate::tuning::resident_gemm::native_tile;
use crate::tuning::resident_gemm::nvfp4_search_trials;
use crate::tuning::verify_split_qualified;

use crate::tuning::resident_gemm::GemmConfig;

use crate::tuning::resident_gemm::QUANTIZATION_ROWS;
use crate::tuning::resident_gemm::WeightKey as ResidentGemmWeightKey;
use crate::tuning::resident_gemm::select as resident_gemm_select;
use crate::tuning::swiglu::{Contract as SwigluContract, select as select_swiglu_tuned};

/// Two cache states times the baseline/candidate arms.
const SWIGLU_TRIAL_FACTOR: usize = 4;
use crate::{
    device::{CudaDevice, device_error, vendor_fp8},
    mlp::ProjectionWeight,
    resident::ProgramWeights,
};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, TensorOp};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::env::var;

#[cfg(test)]
use std::cell::Cell;
use std::result::Result as StdResult;

type Quantized = (Tensor<f4e2m1fnx2>, Tensor<f8e4m3fn>);
/// Fused gate/up tile. Two F32 accumulators plus the grouped quantization epilogue fit at 64 by
/// 64; a 128-row tile spills (2.5 ms instead of 0.40 ms at 1024 x 17408 x 5120 on RTX 5090,
/// against 0.45 ms for the separate projections and quantizer).
const SWIGLU_TILE_ROWS: usize = 64;
const SWIGLU_TILE_COLUMNS: usize = 64;
/// Candidate row tiling measured as one alternative to the conservative tile.
const SWIGLU_WIDE_ROWS: usize = 128;
/// Candidate column tiling retained only when its quantization layout is legal.
const SWIGLU_WIDE_COLUMNS: usize = 128;

/// One legal fused-SwiGLU loading-time tile candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwigluTile {
    pub rows: usize,
    pub columns: usize,
}

impl SwigluTile {
    /// Conservative 64x64 tile qualified by the existing measurements.
    pub const FALLBACK: Self = Self {
        rows: SWIGLU_TILE_ROWS,
        columns: SWIGLU_TILE_COLUMNS,
    };
}

/// Shape-specific candidates in conservative-first order.
///
/// The 128-row candidate is retained only when every row/column extent and the quantization
/// group layout are legal. It is a candidate, not a default: the 64x64 fallback stays qualified
/// until full-output and paired quantization-inclusive evidence selects otherwise.
pub fn swiglu_candidates(rows: usize, columns: usize, hidden: usize) -> Vec<SwigluTile> {
    let mut candidates = Vec::new();
    for tile in [
        SwigluTile::FALLBACK,
        SwigluTile {
            rows: SWIGLU_WIDE_ROWS,
            columns: SWIGLU_TILE_COLUMNS,
        },
        SwigluTile {
            rows: SWIGLU_WIDE_ROWS,
            columns: SWIGLU_WIDE_COLUMNS,
        },
    ] {
        if rows.is_multiple_of(tile.rows)
            && hidden.is_multiple_of(tile.columns)
            && columns.is_multiple_of(NVFP4_GROUP_SIZE)
        {
            candidates.push(tile);
        }
    }
    candidates
}

/// Select the qualified default; candidate search may choose another tile only after its own
/// exact-output and paired evidence exists.
pub fn select_swiglu_tile(rows: usize, columns: usize, hidden: usize) -> Option<SwigluTile> {
    swiglu_candidates(rows, columns, hidden)
        .contains(&SwigluTile::FALLBACK)
        .then_some(SwigluTile::FALLBACK)
}

/// Native W4A4 K step used by split windows.
const NVFP4_K_STEP: usize = 256;
const NVFP4_SPLIT_TILE: [usize; 2] = [64, 64];
/// FP8 native K step.
const FP8_K_STEP: usize = 128;
/// Keyed by `(rows, columns, activation-scale columns)`.
type Fp8Key = (usize, usize, usize);
type StrategyKey = (ResidentGemmWeightKey, usize, usize, usize);
/// Quantized activation and its per-token scales.
type Fp8Buffers = (Tensor<f8e4m3fn>, Tensor<f32>);
type Fp8SplitShape = (usize, usize, usize, usize, usize);
type Fp8StrategyKey = (WeightKey, Fp8SplitShape);

struct SplitBuffer {
    plan: SplitCandidate,
    partials: Tensor<f32>,
}

pub struct Workspace {
    quant_reuse: CaptureQuantReuse,
    norm_maximum: Option<super::super::norm_maximum::Pending>,
    strategies: BTreeMap<StrategyKey, GemmConfig>,
    buffers: BTreeMap<(usize, usize), Quantized>,
    fp8_strategies: BTreeMap<(WeightKey, usize, usize, usize), Choice>,
    /// Lt handle, descriptors and original weight allocations outlive every program capture.
    vendor_fp8: vendor_fp8::Support,
    /// Explicit diagnostic override; production automatic choices still require exact gates.
    vendor_diagnostic: bool,
    /// Keyed by `(rows, columns, scale_columns)`: per-channel and block-scaled FP8 share an
    /// operand geometry but need different activation-scale extents, so geometry alone would let
    /// whichever projection is visited first decide the layout for the other.
    fp8: BTreeMap<Fp8Key, Fp8Buffers>,
    /// Proven/selected fused `SwiGLU` tiles by `(rows, columns, hidden)`.
    swiglu_tiles: BTreeMap<
        (
            infer_core::TensorId,
            infer_core::TensorId,
            usize,
            usize,
            usize,
        ),
        SwigluTile,
    >,
    split_nvfp4: BTreeMap<(usize, usize, usize), SplitBuffer>,
    split_nvfp4_enabled: BTreeSet<StrategyKey>,
    split_fp8: BTreeMap<Fp8SplitShape, SplitBuffer>,
    split_fp8_enabled: BTreeSet<Fp8StrategyKey>,
    split_scratch_bytes: u64,
}

impl Workspace {
    /// Start a fresh graph-local permission epoch; no device memory is changed.
    pub(crate) const fn begin_quant_capture(&mut self) {
        self.norm_maximum = None;
        self.quant_reuse.begin();
    }
    pub(crate) fn enter_quant_node(&mut self, index: usize, role: Option<QuantPairRole>) {
        if self
            .norm_maximum
            .is_some_and(|pending| pending.next != index)
        {
            self.norm_maximum = None;
        }
        self.quant_reuse.enter(index, role);
    }
    pub(crate) const fn end_quant_capture(&mut self) {
        self.norm_maximum = None;
        self.quant_reuse.end();
    }
    pub(crate) const fn clear_quant_reuse(&mut self) {
        self.norm_maximum = None;
        self.quant_reuse.clear();
    }
    #[cfg(test)]
    pub(crate) const fn set_quant_reuse_enabled(&mut self, enabled: bool) {
        self.quant_reuse.set_enabled(enabled);
    }
    #[cfg(test)]
    pub(crate) const fn quant_reuse_stats(&self) -> (usize, usize) {
        let stats = self.quant_reuse.stats();
        (stats.recorded, stats.reused)
    }
    #[cfg(test)]
    pub(crate) const fn quant_reuse_totals(&self) -> (usize, usize) {
        let stats = self.quant_reuse.total();
        (stats.recorded, stats.reused)
    }
    #[cfg(test)]
    pub(crate) fn quantized_fp8(
        &self,
        rows: usize,
        columns: usize,
    ) -> Option<(&Tensor<f8e4m3fn>, &Tensor<f32>)> {
        self.fp8
            .get(&(rows, columns, 1))
            .map(|(codes, scales)| (codes, scales))
    }
    #[cfg(test)]
    pub(super) fn split_plan_count(&self) -> usize {
        self.split_nvfp4.len()
    }
    #[cfg(test)]
    pub(super) const fn split_scratch_bytes(&self) -> u64 {
        self.split_scratch_bytes
    }
    #[cfg(test)]
    pub(crate) fn quantized_fp4(
        &self,
        rows: usize,
        columns: usize,
    ) -> Option<(&Tensor<f4e2m1fnx2>, &Tensor<f8e4m3fn>)> {
        self.buffers
            .get(&(rows, columns))
            .map(|(codes, scales)| (codes, scales))
    }

    /// Test evidence for prepared strategies, grouped by shape/config; this is not a replay trace.
    #[cfg(test)]
    pub(crate) fn prepared_configs(&self) -> serde_json::Value {
        let mut groups = BTreeMap::<(&str, usize, usize, usize, String), usize>::new();
        for ((_, rows, outputs, columns), config) in &self.strategies {
            *groups
                .entry(("nvfp4", *rows, *outputs, *columns, format!("{config:?}")))
                .or_default() += 1;
        }
        for ((_, rows, outputs, columns), config) in &self.fp8_strategies {
            *groups
                .entry((
                    "fp8-channel",
                    *rows,
                    *outputs,
                    *columns,
                    format!("{config:?}"),
                ))
                .or_default() += 1;
        }
        serde_json::json!(
            groups
                .into_iter()
                .map(|((mode, rows, outputs, columns, config), count)| {
                    serde_json::json!({"mode":mode, "m":rows, "n":outputs, "k":columns,
                "prepared_config":config, "weight_count":count})
                })
                .collect::<Vec<_>>()
        )
    }

    pub(crate) fn program(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        geometry: ProgramGeometry,
    ) -> Result<Self> {
        let widths = geometry.widths();
        let skipped: BTreeSet<_> = geometry
            .prompts
            .into_iter()
            .filter(|&rows| rows > 1 && rows != geometry.verification)
            .flat_map(|rows| {
                fused_projections(graph, weights, rows)
                    .into_iter()
                    .map(move |index| (index, rows))
            })
            .collect();
        Self::new_planned(device, graph, weights, &widths, &skipped)
    }

    pub(crate) fn ensure_width(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        width: usize,
    ) -> Result<()> {
        self.extend(device, graph, weights, &[width], &BTreeSet::new())
    }

    pub(crate) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<Self> {
        Self::new_planned(device, graph, weights, widths, &BTreeSet::new())
    }

    fn new_planned(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
        skipped: &BTreeSet<(usize, usize)>,
    ) -> Result<Self> {
        let mut workspace = Self {
            quant_reuse: CaptureQuantReuse::default(),
            norm_maximum: None,
            strategies: BTreeMap::new(),
            buffers: BTreeMap::new(),
            fp8: BTreeMap::new(),
            fp8_strategies: BTreeMap::new(),
            split_nvfp4: BTreeMap::new(),
            split_nvfp4_enabled: BTreeSet::new(),
            split_fp8: BTreeMap::new(),
            split_fp8_enabled: BTreeSet::new(),
            split_scratch_bytes: 0,
            vendor_fp8: vendor_fp8::Support::new_with(device, true),
            vendor_diagnostic: vendor_diagnostic_enabled(),
            swiglu_tiles: BTreeMap::new(),
        };
        workspace.extend(device, graph, weights, widths, skipped)?;
        workspace.tune_swiglu(device, graph, weights, widths)?;
        Ok(workspace)
    }

    /// Register every geometry this workspace may search before the first measurement. This
    /// lets an exhausted shape report all still-unstarted shapes, not only the current one.
    fn plan_search_shapes(
        session: &ResidentTuningSession,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
        skipped: &BTreeSet<(usize, usize)>,
    ) -> Result<()> {
        let mut fp8 = BTreeSet::new();
        let mut nvfp4 = BTreeSet::new();
        for (index, node) in graph.nodes.iter().enumerate() {
            if node.op != TensorOp::Linear {
                continue;
            }
            let Some(input) = graph.tensors.iter().find(|t| t.id == node.inputs[0]) else {
                continue;
            };
            let Some(output) = graph.tensors.iter().find(|t| t.id == node.outputs[0]) else {
                continue;
            };
            let columns = input.elements()?;
            let outputs = output.elements()?;
            let weight_id = node.inputs[1];
            if weights.fp8_inputs.contains(&weight_id) {
                if matches!(
                    weights.projections.get(&weight_id),
                    Some(ProjectionWeight::Fp8(..))
                ) {
                    for &rows in widths {
                        if rows == 0
                            || (rows >= PREFILL_LANES && Some(node.outputs[0]) == graph.logits)
                        {
                            continue;
                        }
                        fp8.insert([rows, outputs, columns]);
                    }
                }
                continue;
            }
            if !weights.input_scales.contains_key(&weight_id)
                || !matches!(
                    weights.projections.get(&weight_id),
                    Some(ProjectionWeight::Fp4(..))
                )
            {
                continue;
            }
            for &rows in widths {
                if rows > 0 && !skipped.contains(&(index, rows)) {
                    nvfp4.insert([rows, outputs, columns]);
                }
            }
        }
        let fp8: Vec<_> = fp8.into_iter().collect();
        let nvfp4: Vec<_> = nvfp4.into_iter().collect();
        session.plan_gemm_search(SearchFamily::Fp8Channel, &fp8, native_search_trials())?;
        session.plan_gemm_search(SearchFamily::Nvfp4, &nvfp4, nvfp4_search_trials())?;
        Ok(())
    }

    fn split_allowed(
        device: &CudaDevice,
        key: (&str, usize, usize, usize, usize, u64),
    ) -> Result<bool> {
        #[cfg(test)]
        if DISABLE_SPLIT.with(Cell::get) {
            return Ok(false);
        }
        #[cfg(test)]
        if FORCE_SPLIT.with(Cell::get) {
            return Ok(true);
        }
        verify_split_qualified(device, key)
    }

    fn take_nvfp4_split(
        &mut self,
        strategy: StrategyKey,
        shape: (usize, usize, usize),
    ) -> Option<SplitBuffer> {
        #[cfg(test)]
        let enabled = FORCE_SPLIT.with(Cell::get) || self.split_nvfp4_enabled.contains(&strategy);
        #[cfg(not(test))]
        let enabled = self.split_nvfp4_enabled.contains(&strategy);
        enabled.then(|| self.split_nvfp4.remove(&shape)).flatten()
    }

    fn ensure_nvfp4_split(&mut self, device: &CudaDevice, key: StrategyKey) -> Result<()> {
        let (_, rows, outputs, columns) = key;
        let shape = (rows, outputs, columns);
        let sms = usize::try_from(device.profile()?.multiprocessors)
            .map_err(|_| Error::invariant("SM count"))?;
        #[cfg_attr(
            not(test),
            allow(unused_mut, reason = "mutation is test-only split override")
        )]
        let Some(mut plan) = projection_split_policy(
            rows,
            outputs,
            columns,
            sms,
            NVFP4_SPLIT_TILE,
            NVFP4_K_STEP,
            VERIFY_SPLIT_SCRATCH_BUDGET,
        )?
        else {
            return Ok(());
        };
        #[cfg(test)]
        let forced = FORCED_SPLITS.with(Cell::get);
        #[cfg(test)]
        if forced > 0 {
            if forced > columns.div_ceil(NVFP4_K_STEP) {
                return Ok(());
            }
            plan.splits = forced;
            plan.scratch_bytes = super::split_scratch_bytes(rows, outputs, forced)?;
        }
        if !Self::split_allowed(
            device,
            (
                "nvfp4",
                rows,
                outputs,
                columns,
                plan.splits,
                plan.scratch_bytes,
            ),
        )? {
            return Ok(());
        }
        if self.split_nvfp4.contains_key(&shape) {
            self.split_nvfp4_enabled.insert(key);
            return Ok(());
        }
        let total = self
            .split_scratch_bytes
            .checked_add(plan.scratch_bytes)
            .ok_or_else(|| Error::invalid("retained verify split scratch overflow"))?;
        if total > VERIFY_SPLIT_SCRATCH_BUDGET {
            return Ok(());
        }
        let partials = api::zeros::<f32>(&[plan.splits, rows, outputs])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        self.split_scratch_bytes = total;
        self.split_nvfp4
            .insert(shape, SplitBuffer { plan, partials });
        self.split_nvfp4_enabled.insert(key);
        Ok(())
    }

    fn ensure_fp8_split(
        &mut self,
        device: &CudaDevice,
        key: (WeightKey, usize, usize, usize),
        choice: Choice,
    ) -> Result<()> {
        let (weight_key, rows, outputs, columns) = key;
        let Some(tile) = choice.native_tile() else {
            return Ok(());
        };
        let shape = (rows, outputs, columns, tile[0], tile[1]);
        let strategy = (weight_key, shape);
        let sms = usize::try_from(device.profile()?.multiprocessors)
            .map_err(|_| Error::invariant("SM count"))?;
        let Some(plan) = projection_split_policy(
            rows,
            outputs,
            columns,
            sms,
            tile,
            FP8_K_STEP,
            VERIFY_SPLIT_SCRATCH_BUDGET,
        )?
        else {
            return Ok(());
        };
        if !Self::split_allowed(
            device,
            (
                "fp8-channel",
                rows,
                outputs,
                columns,
                plan.splits,
                plan.scratch_bytes,
            ),
        )? {
            return Ok(());
        }
        if self.split_fp8.contains_key(&shape) {
            self.split_fp8_enabled.insert(strategy);
            return Ok(());
        }
        let total = self
            .split_scratch_bytes
            .checked_add(plan.scratch_bytes)
            .ok_or_else(|| Error::invalid("retained verify split scratch overflow"))?;
        if total > VERIFY_SPLIT_SCRATCH_BUDGET {
            return Ok(());
        }
        let partials = api::zeros::<f32>(&[plan.splits, rows, outputs])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        self.split_scratch_bytes = total;
        self.split_fp8.insert(shape, SplitBuffer { plan, partials });
        self.split_fp8_enabled.insert(strategy);
        Ok(())
    }

    fn extend(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
        skipped: &BTreeSet<(usize, usize)>,
    ) -> Result<()> {
        Self::plan_search_shapes(&weights.resident_tuning, graph, weights, widths, skipped)?;
        let _probe_inputs = ProbeInputsScope::new(device)?;
        for (index, node) in graph.nodes.iter().enumerate() {
            if node.op != TensorOp::Linear
                || (!weights.input_scales.contains_key(&node.inputs[1])
                    && !weights.fp8_inputs.contains(&node.inputs[1]))
            {
                continue;
            }
            let columns = graph
                .tensors
                .iter()
                .find(|t| t.id == node.inputs[0])
                .ok_or_else(|| Error::invariant("FP4 input tensor"))?
                .elements()?;
            if weights.fp8_inputs.contains(&node.inputs[1]) {
                self.allocate_fp8(
                    device,
                    weights.projections.get(&node.inputs[1]),
                    columns,
                    widths,
                )?;
                self.prepare_channel(
                    device,
                    graph,
                    weights,
                    node.inputs[1],
                    node.outputs[0],
                    columns,
                    widths,
                )?;
                continue;
            }
            if let Some(weight @ ProjectionWeight::Fp4(..)) =
                weights.projections.get(&node.inputs[1])
                && let Some(&scale) = weights.input_scales.get(&node.inputs[1])
            {
                let outputs = graph
                    .tensors
                    .iter()
                    .find(|tensor| tensor.id == node.outputs[0])
                    .ok_or_else(|| Error::invariant("FP4 output tensor"))?
                    .elements()?;
                let weight_key = ResidentGemmWeightKey::new(weight, scale)?;
                for &rows in widths {
                    if rows > 0 && !skipped.contains(&(index, rows)) {
                        let key = (weight_key, rows, outputs, columns);
                        self.strategies.entry(key).or_insert_with(|| {
                            resident_gemm_select(
                                device,
                                weight,
                                scale,
                                [rows, outputs, columns],
                                weights.autotune,
                                &weights.resident_tuning,
                            )
                        });
                        #[cfg(test)]
                        if FORCED_LOOP.with(Cell::get) > 0
                            && let Some(config) = self.strategies.get_mut(&key)
                        {
                            config.main_loop = match FORCED_LOOP.with(Cell::get) {
                                1 => MainLoop::Plain,
                                _ => MainLoop::Prefetch8,
                            };
                        }
                        self.ensure_nvfp4_split(device, key)?;
                    }
                }
            }
            for &rows in widths {
                if rows == 0 || self.buffers.contains_key(&(rows, columns)) {
                    continue;
                }
                self.buffers.insert(
                    (rows, columns),
                    (
                        api::zeros::<f4e2m1fnx2>(&[rows, columns / 2])
                            .sync_on(&device.stream)
                            .map_err(device_error)?,
                        api::zeros::<f8e4m3fn>(&[rows, columns / NVFP4_GROUP_SIZE])
                            .sync_on(&device.stream)
                            .map_err(device_error)?,
                    ),
                );
            }
        }
        Ok(())
    }

    fn allocate_fp8(
        &mut self,
        device: &CudaDevice,
        weight: Option<&ProjectionWeight>,
        columns: usize,
        widths: &[usize],
    ) -> Result<()> {
        let block = matches!(weight, Some(ProjectionWeight::Fp8Block(..)));
        let scale_columns = if block {
            columns / FP8_BLOCK_COLUMNS
        } else {
            1
        };
        for &rows in widths {
            if rows > 0 && !self.fp8.contains_key(&(rows, columns, scale_columns)) {
                self.fp8.insert(
                    (rows, columns, scale_columns),
                    (
                        api::zeros::<f8e4m3fn>(&[rows, columns])
                            .sync_on(&device.stream)
                            .map_err(device_error)?,
                        api::zeros::<f32>(&[rows, scale_columns])
                            .sync_on(&device.stream)
                            .map_err(device_error)?,
                    ),
                );
            }
        }
        Ok(())
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Preparation carries graph weight/output identities and the captured widths before any CUDA scope starts"
    )]
    fn prepare_channel(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        weight_id: infer_core::TensorId,
        output_id: infer_core::TensorId,
        columns: usize,
        widths: &[usize],
    ) -> Result<()> {
        if let Some(weight @ ProjectionWeight::Fp8(codes, scales)) =
            weights.projections.get(&weight_id)
        {
            self.vendor_fp8.retain_channels(codes, scales);
            let outputs = graph
                .tensors
                .iter()
                .find(|tensor| tensor.id == output_id)
                .ok_or_else(|| Error::invariant("FP8 output tensor"))?
                .elements()?;
            let weight_key = WeightKey::new(weight)?;
            for &rows in widths {
                // Prompt captures omit the batched vocabulary head and use a single final lane.
                if rows == 0 || (rows >= PREFILL_LANES && Some(output_id) == graph.logits) {
                    continue;
                }
                let key = (weight_key, rows, outputs, columns);
                if self.fp8_strategies.contains_key(&key) {
                    continue;
                }
                let lt_prefill = self
                    .fp8
                    .get(&(rows, columns, 1))
                    .map(|(q, ..)| {
                        prefill_choice(
                            device,
                            &mut self.vendor_fp8,
                            q,
                            codes,
                            [rows, outputs, columns],
                        )
                    })
                    .transpose()?
                    .flatten();
                let choice = if let Some(choice) = lt_prefill {
                    choice
                } else if self.vendor_diagnostic {
                    let q = &self
                        .fp8
                        .get(&(rows, columns, 1))
                        .ok_or_else(|| Error::invariant("FP8 diagnostic activation"))?
                        .0;
                    let out = api::zeros::<f32>(&[rows, outputs])
                        .sync_on(&device.stream)
                        .map_err(device_error)?;
                    if let Err(error) = self.vendor_fp8.prepare(q, codes, &out) {
                        tracing::warn!(target: "infer::tuning", %error, "FP8 diagnostic preparation rejected: native fallback");
                    }
                    Choice::baseline(rows)
                } else {
                    select(
                        device,
                        &mut self.vendor_fp8,
                        weight,
                        [rows, outputs, columns],
                        weights.autotune,
                        &weights.resident_tuning,
                    )
                };
                self.fp8_strategies.insert(key, choice);
                self.ensure_fp8_split(device, key, choice)?;
            }
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Quantization records the exact view pointer alongside its original projection contract"
    )]
    pub(super) fn record_input(
        &mut self,
        scope: &Scope,
        weight: &ProjectionWeight,
        scale: f32,
        input: &TensorView<'_, f32>,
        output: &mut Tensor<f32>,
        columns: usize,
        input_pointer: Option<u64>,
    ) -> StdResult<(), DeviceError> {
        let ProjectionWeight::Fp4(..) = weight else {
            return Err(super::super::capture::error("FP4 workspace weight"));
        };
        let outputs = usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let rows = output.size() / outputs;
        let shape = (rows, outputs, columns);
        let strategy = (
            ResidentGemmWeightKey::new(weight, scale).map_err(super::super::capture::error)?,
            rows,
            outputs,
            columns,
        );
        let mut split = self.take_nvfp4_split(strategy, shape);
        let strategies = &self.strategies;
        let (q, qs) = self
            .buffers
            .get_mut(&(rows, columns))
            .ok_or_else(|| super::super::capture::error("FP4 quantization workspace missing"))?;
        let key = quant_key(
            input_pointer,
            input,
            output,
            rows,
            columns,
            QuantKind::Fp4(scale.to_bits()),
            (&*q, &*qs),
        );
        let (reuse, ticket) = self.quant_reuse.prepare(key);
        let mut record = || {
            if !reuse && rows.is_multiple_of(QUANTIZATION_ROWS) {
                scope.record(
                    kernels::quantize_rows(
                        (&mut *q).partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]]),
                        (&mut *qs).partition([QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]]),
                        input,
                        scale,
                    )
                    .generics(vec![columns.to_string()]),
                )?;
            } else if !reuse {
                scope.record(
                    kernels::quantize(
                        (&mut *q).partition(NVFP4_QUANT_CODES_TILE),
                        (&mut *qs).partition(NVFP4_QUANT_SCALES_TILE),
                        input,
                        scale,
                    )
                    .generics(vec![columns.to_string()]),
                )?;
            }
            Self::record_quantized_gemm(
                strategies,
                split.as_mut(),
                scope,
                weight,
                scale,
                (&*q, &*qs),
                output,
                (rows, columns),
            )
        };
        let result = record();
        if let Some(split) = split {
            self.split_nvfp4.insert(shape, split);
        }
        self.quant_reuse.finish(ticket, result.is_ok());
        result
    }

    /// Down projection of a gated MLP: quantize `silu(gate) * up` straight into the activation
    /// codes, then run the same row-tiled W4A4 GEMM as [`Self::record_input`].
    #[expect(
        clippy::too_many_arguments,
        reason = "The fused quantizer names both MLP halves next to the projection contract"
    )]
    pub(crate) fn record_swiglu_input(
        &mut self,
        scope: &Scope,
        weight: &ProjectionWeight,
        scale: f32,
        gate: &TensorView<'_, f32>,
        up: &TensorView<'_, f32>,
        output: &mut Tensor<f32>,
        columns: usize,
    ) -> StdResult<(), DeviceError> {
        let ProjectionWeight::Fp4(..) = weight else {
            return Err(super::super::capture::error("FP4 workspace weight"));
        };
        let outputs = usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let rows = output.size() / outputs;
        let shape = (rows, outputs, columns);
        let strategy = (
            ResidentGemmWeightKey::new(weight, scale).map_err(super::super::capture::error)?,
            rows,
            outputs,
            columns,
        );
        let mut split = self.take_nvfp4_split(strategy, shape);
        let strategies = &self.strategies;
        // The fused codes overwrite the shared quantization buffers; no reuse permit may span it.
        self.quant_reuse.clear();
        let (q, qs) = self
            .buffers
            .get_mut(&(rows, columns))
            .ok_or_else(|| super::super::capture::error("FP4 quantization workspace missing"))?;
        if !rows.is_multiple_of(QUANTIZATION_ROWS) {
            return Err(super::super::capture::error("fused SwiGLU row tile"));
        }
        scope.record(
            kernels::swiglu_quantize(
                (&mut *q).partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]]),
                (&mut *qs).partition([QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]]),
                gate,
                up,
                scale,
            )
            .generics(vec![columns.to_string()]),
        )?;
        let result = Self::record_quantized_gemm(
            strategies,
            split.as_mut(),
            scope,
            weight,
            scale,
            (&*q, &*qs),
            output,
            (rows, columns),
        );
        if let Some(split) = split {
            self.split_nvfp4.insert(shape, split);
        }
        result
    }

    fn tune_swiglu(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<()> {
        let profile = device.profile()?;
        for (gate, up, hidden, columns, input_scale, down_scale) in
            super::super::prefill_projection::swiglu_tuning_pairs(graph, weights)
        {
            let (Some(gate_weight), Some(up_weight)) =
                (weights.projections.get(&gate), weights.projections.get(&up))
            else {
                continue;
            };
            for &rows in widths {
                if rows < PREFILL_LANES {
                    continue;
                }
                let shape = [rows, columns, hidden];
                let trials =
                    swiglu_candidates(rows, columns, hidden).len().max(1) * SWIGLU_TRIAL_FACTOR;
                let search = weights.autotune
                    && weights.resident_tuning.permits_gemm_search(
                        SearchFamily::FusedSwiGLU,
                        shape,
                        trials,
                    )?;
                let tile = select_swiglu_tuned(
                    device,
                    profile,
                    gate_weight,
                    up_weight,
                    SwigluContract {
                        shape: (rows, columns, hidden),
                        scales: (input_scale, down_scale),
                    },
                    search,
                    &weights.resident_tuning,
                )?;
                self.swiglu_tiles
                    .insert((gate, up, rows, columns, hidden), tile);
            }
        }
        Ok(())
    }

    fn selected_swiglu(
        &self,
        gate: infer_core::TensorId,
        up: infer_core::TensorId,
        rows: usize,
        columns: usize,
        hidden: usize,
    ) -> Option<SwigluTile> {
        self.swiglu_tiles
            .get(&(gate, up, rows, columns, hidden))
            .copied()
            .or_else(|| select_swiglu_tile(rows, columns, hidden))
    }

    /// Geometry accepted by a legal fused prompt `SwiGLU` tile.
    pub(crate) const fn gate_up_swiglu_geometry(rows: usize, hidden: usize) -> bool {
        rows > 0
            && rows.is_multiple_of(SWIGLU_TILE_ROWS)
            && rows.is_multiple_of(QUANTIZATION_ROWS)
            && hidden.is_multiple_of(SWIGLU_TILE_COLUMNS)
    }

    /// Whether [`Self::record_gate_up_swiglu`] can run for this prompt width and MLP geometry.
    pub(crate) fn gate_up_swiglu_ready(
        &self,
        gate: infer_core::TensorId,
        up: infer_core::TensorId,
        rows: usize,
        columns: usize,
        hidden: usize,
    ) -> bool {
        self.selected_swiglu(gate, up, rows, columns, hidden)
            .is_some_and(|tile| {
                rows > 0
                    && rows.is_multiple_of(tile.rows)
                    && rows.is_multiple_of(QUANTIZATION_ROWS)
                    && hidden.is_multiple_of(tile.columns)
                    && self.buffers.contains_key(&(rows, columns))
                    && self.buffers.contains_key(&(rows, hidden))
            })
    }

    /// Gate and up projections of a gated MLP fused with the down projection's quantizer:
    /// quantize the shared input, then write the `silu(gate) * up` codes straight into the down
    /// projection's buffer for [`Self::record_prequantized`]. Neither F32 projection is written.
    pub(crate) fn record_gate_up_swiglu(
        &mut self,
        scope: &Scope,
        (gate_id, up_id): (infer_core::TensorId, infer_core::TensorId),
        (gate, up): (&ProjectionWeight, &ProjectionWeight),
        (input_scale, down_scale): (f32, f32),
        input: &TensorView<'_, f32>,
        (rows, columns, hidden): (usize, usize, usize),
    ) -> StdResult<(), DeviceError> {
        let (
            ProjectionWeight::Fp4(gate_codes, gate_scales, gate_global),
            ProjectionWeight::Fp4(up_codes, up_scales, up_global),
        ) = (gate, up)
        else {
            return Err(super::super::capture::error("fused SwiGLU weights"));
        };
        if !self.gate_up_swiglu_ready(gate_id, up_id, rows, columns, hidden) {
            return Err(super::super::capture::error("fused SwiGLU geometry"));
        }
        // Both writes replace shared quantization buffers; no reuse permit may span them.
        self.quant_reuse.clear();
        let tile = self
            .selected_swiglu(gate_id, up_id, rows, columns, hidden)
            .ok_or_else(|| super::super::capture::error("fused SwiGLU tile missing"))?;
        let (mut q, mut qs) = self
            .buffers
            .remove(&(rows, columns))
            .ok_or_else(|| super::super::capture::error("FP4 quantization workspace missing"))?;
        let quantized = scope
            .record(
                kernels::quantize_rows(
                    (&mut q).partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]]),
                    (&mut qs).partition([QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]]),
                    input,
                    input_scale,
                )
                .generics(vec![columns.to_string()]),
            )
            .map(drop);
        let recorded = quantized.and_then(|()| {
            let (codes, code_scales) = self.buffers.get_mut(&(rows, hidden)).ok_or_else(|| {
                super::super::capture::error("FP4 quantization workspace missing")
            })?;
            scope
                .record(
                    kernels::packed_swiglu(
                        codes.partition([tile.rows, tile.columns / 2]),
                        code_scales.partition([tile.rows, tile.columns / NVFP4_GROUP_SIZE]),
                        &q,
                        &qs,
                        gate_codes,
                        gate_scales,
                        up_codes,
                        up_scales,
                        1.0 / (gate_global * input_scale),
                        1.0 / (up_global * input_scale),
                        down_scale,
                    )
                    .generics(
                        [
                            columns,
                            tile.rows,
                            tile.columns,
                            tile.columns / 2,
                            tile.columns / NVFP4_GROUP_SIZE,
                        ]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                    ),
                )
                .map(drop)
        });
        self.buffers.insert((rows, columns), (q, qs));
        recorded
    }

    /// Down projection whose activation codes [`Self::record_gate_up_swiglu`] already wrote.
    pub(crate) fn record_prequantized(
        &mut self,
        scope: &Scope,
        weight: &ProjectionWeight,
        scale: f32,
        output: &mut Tensor<f32>,
        columns: usize,
    ) -> StdResult<(), DeviceError> {
        let outputs = usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let rows = output.size() / outputs;
        let shape = (rows, outputs, columns);
        let strategy = (
            ResidentGemmWeightKey::new(weight, scale).map_err(super::super::capture::error)?,
            rows,
            outputs,
            columns,
        );
        let mut split = self.take_nvfp4_split(strategy, shape);
        let strategies = &self.strategies;
        let (q, qs) = self
            .buffers
            .get(&(rows, columns))
            .ok_or_else(|| super::super::capture::error("FP4 quantization workspace missing"))?;
        let result = Self::record_quantized_gemm(
            strategies,
            split.as_mut(),
            scope,
            weight,
            scale,
            (q, qs),
            output,
            (rows, columns),
        );
        if let Some(split) = split {
            self.split_nvfp4.insert(shape, split);
        }
        result
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GEMM operands and split owner"
    )]
    fn record_quantized_gemm(
        strategies: &BTreeMap<StrategyKey, GemmConfig>,
        split: Option<&mut SplitBuffer>,
        scope: &Scope,
        weight: &ProjectionWeight,
        scale: f32,
        (q, qs): (&Tensor<f4e2m1fnx2>, &Tensor<f8e4m3fn>),
        output: &mut Tensor<f32>,
        (rows, columns): (usize, usize),
    ) -> StdResult<(), DeviceError> {
        {
            let outputs =
                usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
            let key = (
                ResidentGemmWeightKey::new(weight, scale).map_err(super::super::capture::error)?,
                rows,
                outputs,
                columns,
            );
            if let Some(split) = split {
                let ProjectionWeight::Fp4(codes, scales, global) = weight else {
                    return Err(super::super::capture::error("FP4 split weight"));
                };
                let tile = split.plan.tile;
                scope.record(
                    kernels::packed_split(
                        (&mut split.partials).partition([1, tile[0], tile[1]]),
                        q,
                        qs,
                        codes,
                        scales,
                    )
                    .generics(vec![
                        columns.to_string(),
                        tile[0].to_string(),
                        tile[1].to_string(),
                        split.plan.splits.to_string(),
                    ]),
                )?;
                scope.record(
                    kernels::sum_splits(
                        output.partition(tile),
                        &split.partials,
                        1.0 / (global * scale),
                    )
                    .generics(vec![
                        tile[0].to_string(),
                        tile[1].to_string(),
                        split.plan.splits.to_string(),
                    ]),
                )?;
                #[cfg(test)]
                SPLIT_RECORDS.set(SPLIT_RECORDS.get() + 1);
                return Ok(());
            }
            let config = strategies
                .get(&key)
                .copied()
                .unwrap_or_else(|| GemmConfig::baseline(rows));
            config.record(scope, weight, scale, q, qs, output, columns)?;
            Ok(())
        }
    }
    /// Keep the existing scale allocation; its permission lasts exactly one graph node.
    pub(crate) fn record_norm_maximum(
        &mut self,
        scope: &Scope,
        next: usize,
        input: &TensorView<'_, f32>,
        weight: &Tensor<f32>,
        output: &mut Tensor<f32>,
        (epsilon, offset): (f32, f32),
    ) -> StdResult<bool, DeviceError> {
        let rows = usize::try_from(output.shape()[0]).map_err(super::super::capture::error)?;
        let columns = usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let Some((_, scales)) = self.fp8.get_mut(&(rows, columns, 1)) else {
            return Ok(false);
        };
        let pointer = output.device_pointer().cu_deviceptr();
        scope.record(
            super::super::norm_maximum_kernel::kernels::norm_max(
                output.partition([1, columns.next_power_of_two()]),
                scales.partition([1, 1]),
                input,
                weight,
                epsilon,
                offset,
            )
            .generics(vec![
                columns.to_string(),
                columns.next_power_of_two().to_string(),
            ]),
        )?;
        self.norm_maximum = Some(super::super::norm_maximum::Pending {
            next,
            pointer,
            rows,
            columns,
        });
        Ok(true)
    }

    fn take_enabled_fp8(
        &mut self,
        weight_key: WeightKey,
        shape: Fp8SplitShape,
    ) -> Option<SplitBuffer> {
        #[cfg(test)]
        let enabled =
            FORCE_SPLIT.with(Cell::get) || self.split_fp8_enabled.contains(&(weight_key, shape));
        #[cfg(not(test))]
        let enabled = self.split_fp8_enabled.contains(&(weight_key, shape));
        enabled.then(|| self.split_fp8.remove(&shape)).flatten()
    }

    #[expect(clippy::too_many_arguments, reason = "explicit CUDA tensor bindings")]
    pub(crate) fn record_add_norm_maximum(
        &mut self,
        scope: &Scope,
        next: usize,
        input: &TensorView<'_, f32>,
        residual: &TensorView<'_, f32>,
        weight: &Tensor<f32>,
        sum_output: &mut Tensor<f32>,
        output: &mut Tensor<f32>,
        (epsilon, offset): (f32, f32),
    ) -> StdResult<bool, DeviceError> {
        let rows = usize::try_from(output.shape()[0]).map_err(super::super::capture::error)?;
        let columns = usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let Some((_, scales)) = self.fp8.get_mut(&(rows, columns, 1)) else {
            return Ok(false);
        };
        let pointer = output.device_pointer().cu_deviceptr();
        scope.record(
            super::super::norm_maximum_kernel::kernels::add_norm_max(
                sum_output.partition([1, columns.next_power_of_two()]),
                output.partition([1, columns.next_power_of_two()]),
                scales.partition([1, 1]),
                input,
                residual,
                weight,
                epsilon,
                offset,
            )
            .generics(vec![
                columns.to_string(),
                columns.next_power_of_two().to_string(),
            ]),
        )?;
        self.norm_maximum = Some(super::super::norm_maximum::Pending {
            next,
            pointer,
            rows,
            columns,
        });
        Ok(true)
    }

    pub(super) fn record_fp8_input(
        &mut self,
        scope: &Scope,
        weight: &ProjectionWeight,
        input: &TensorView<'_, f32>,
        output: &mut Tensor<f32>,
        columns: usize,
        input_pointer: Option<u64>,
    ) -> StdResult<(), DeviceError> {
        use super::super::capture::error;
        if !matches!(
            weight,
            ProjectionWeight::Fp8(..) | ProjectionWeight::Fp8Block(..)
        ) {
            return Err(error("FP8 workspace weight"));
        }
        let rows = usize::try_from(output.shape()[0]).map_err(error)?;
        // The scales layout follows the weight's quantization mode, and the key has to say which.
        let scale_columns = fp8_scale_columns(weight, columns);
        let maximum_ready = self.norm_maximum.take().is_some_and(|pending| {
            matches!(weight, ProjectionWeight::Fp8(..))
                && pending.matches(input_pointer, rows, columns)
        });
        let outputs = usize::try_from(output.shape()[1]).map_err(error)?;
        let choice = self
            .fp8_strategies
            .get(&(
                WeightKey::new(weight).map_err(error)?,
                rows,
                outputs,
                columns,
            ))
            .copied()
            .unwrap_or_else(|| Choice::baseline(rows));
        let weight_key = WeightKey::new(weight).map_err(error)?;
        let split_shape = choice
            .native_tile()
            .map(|tile| (rows, outputs, columns, tile[0], tile[1]));
        let mut split = split_shape.and_then(|shape| self.take_enabled_fp8(weight_key, shape));
        let (q, qs) = self
            .fp8
            .get_mut(&(rows, columns, scale_columns))
            .ok_or_else(|| error("FP8 quantization workspace missing"))?;
        let kind = matches!(weight, ProjectionWeight::Fp8(..)).then_some(QuantKind::Fp8Token);
        let key = kind.and_then(|kind| {
            quant_key(
                input_pointer,
                input,
                output,
                rows,
                columns,
                kind,
                (&*q, &*qs),
            )
        });
        let (reuse, ticket) = self.quant_reuse.prepare(key);
        let mut record = || {
            match weight {
                ProjectionWeight::Fp8Block(w, block_scale) => {
                    record_fp8_block(scope, q, qs, w, block_scale, input, output)?;
                }
                ProjectionWeight::Fp8(w, channel_scale) => {
                    if !reuse {
                        super::super::fp8_quant_reduction::record_with_maximum(
                            scope,
                            q,
                            qs,
                            input,
                            columns,
                            maximum_ready,
                        )?;
                        #[cfg(test)]
                        super::super::norm_maximum::record_consumption(maximum_ready);
                    }
                    if let Some(split) = split.as_mut() {
                        record_fp8_split(scope, split, &*q, w, output, &*qs, channel_scale)?;
                        return Ok(());
                    }
                    if self.vendor_diagnostic
                        && self
                            .vendor_fp8
                            .record(scope, &*q, w, output, &*qs, channel_scale)?
                    {
                        return Ok(());
                    }
                    choice.record(
                        scope,
                        &self.vendor_fp8,
                        &*q,
                        w,
                        output,
                        &*qs,
                        channel_scale,
                        columns,
                    )?;
                }
                _ => return Err(error("FP8 workspace weight")),
            }
            Ok(())
        };
        let result = record();
        if let (Some(split), Some(shape)) = (split, split_shape) {
            self.split_fp8.insert(shape, split);
        }
        self.quant_reuse.finish(ticket, result.is_ok());
        result
    }
}

fn record_fp8_block(
    scope: &Scope,
    q: &mut Tensor<f8e4m3fn>,
    qs: &mut Tensor<f32>,
    w: &Tensor<f8e4m3fn>,
    block_scale: &Tensor<f32>,
    input: &TensorView<'_, f32>,
    output: &mut Tensor<f32>,
) -> StdResult<(), DeviceError> {
    use super::super::fp8_gemm::kernels;
    let rows = usize::try_from(output.shape()[0]).map_err(super::super::capture::error)?;
    let columns = usize::try_from(input.shape()[1]).map_err(super::super::capture::error)?;
    let tile = native_tile(rows);
    let generics = [columns, tile[0], tile[1]]
        .map(|value| value.to_string())
        .to_vec();
    scope.record(
        kernels::quantize_block(
            (&mut *q).partition([1, FP8_BLOCK_COLUMNS]),
            (&mut *qs).partition([1, 1]),
            input,
        )
        .generics(vec![columns.to_string(), FP8_BLOCK_COLUMNS.to_string()]),
    )?;
    scope.record(
        kernels::matmul_block(output.partition(tile), &*q, w, &*qs, block_scale).generics(generics),
    )?;
    Ok(())
}

fn record_fp8_split(
    scope: &Scope,
    split: &mut SplitBuffer,
    q: &Tensor<f8e4m3fn>,
    w: &Tensor<f8e4m3fn>,
    output: &mut Tensor<f32>,
    qs: &Tensor<f32>,
    weight_scale: &Tensor<f32>,
) -> StdResult<(), DeviceError> {
    let columns = usize::try_from(q.shape()[1])
        .map_err(|_| super::super::capture::error("FP8 split columns"))?;
    let tile = split.plan.tile;
    let k_tiles = i32::try_from(columns.div_ceil(FP8_K_STEP))
        .map_err(|_| super::super::capture::error("FP8 split K tiles"))?;
    scope.record(
        super::super::fp8_gemm::kernels::matmul_split_tiled(
            (&mut split.partials).partition([1, tile[0], tile[1]]),
            q,
            w,
            k_tiles,
        )
        .generics(vec![
            columns.to_string(),
            tile[0].to_string(),
            tile[1].to_string(),
        ]),
    )?;
    scope.record(
        super::super::fp8_gemm::kernels::reduce_split_tiled(
            output.partition(tile),
            &split.partials,
            qs,
            weight_scale,
            i32::try_from(split.plan.splits)
                .map_err(|_| super::super::capture::error("FP8 split count"))?,
        )
        .generics(vec![tile[0].to_string(), tile[1].to_string()]),
    )?;
    #[cfg(test)]
    SPLIT_RECORDS.set(SPLIT_RECORDS.get() + 1);
    Ok(())
}

const fn fp8_scale_columns(weight: &ProjectionWeight, columns: usize) -> usize {
    match weight {
        ProjectionWeight::Fp8Block(..) => columns / FP8_BLOCK_COLUMNS,
        _ => 1,
    }
}

/// Only the certified whole-matrix paths supply a pointer; sliced/legacy views stay separate.
fn quant_key<Q: DType, S: DType>(
    input_pointer: Option<u64>,
    input: &TensorView<'_, f32>,
    output: &Tensor<f32>,
    rows: usize,
    columns: usize,
    kind: QuantKind,
    buffers: (&Tensor<Q>, &Tensor<S>),
) -> Option<QuantKey> {
    let input_pointer = input_pointer.filter(|pointer| *pointer != 0)?;
    let shape: [i32; 2] = input.shape().try_into().ok()?;
    let strides: [i32; 2] = input.strides().try_into().ok()?;
    if shape != [i32::try_from(rows).ok()?, i32::try_from(columns).ok()?]
        || strides != [i32::try_from(columns).ok()?, 1]
    {
        return None;
    }
    let input_end = input_pointer
        .checked_add(u64::try_from(input.size().checked_mul(size_of::<f32>())?).ok()?)?;
    let output_pointer = output.device_pointer().cu_deviceptr();
    let output_end = output_pointer
        .checked_add(u64::try_from(output.size().checked_mul(size_of::<f32>())?).ok()?)?;
    if input_pointer < output_end && output_pointer < input_end {
        return None;
    }
    Some(QuantKey {
        input_pointer,
        rows,
        columns,
        shape,
        strides,
        kind,
        codes_pointer: buffers.0.device_pointer().cu_deviceptr(),
        scales_pointer: buffers.1.device_pointer().cu_deviceptr(),
    })
}

/// Independent Lt diagnostics are confined to test executables, never production serving.
#[cfg(not(test))]
const fn vendor_diagnostic_enabled() -> bool {
    false
}
#[cfg(test)]
fn vendor_diagnostic_enabled() -> bool {
    var("INFER_CUDA_VENDOR_FP8").is_ok_and(|value| value == "1" || value == "true")
}

#[cfg(test)]
thread_local! {
    static FORCE_SPLIT: Cell<bool> = const { Cell::new(false) };
    static DISABLE_SPLIT: Cell<bool> = const { Cell::new(false) };
    static SPLIT_RECORDS: Cell<usize> = const { Cell::new(0) };
    static FORCED_SPLITS: Cell<usize> = const { Cell::new(0) };
    static FORCED_LOOP: Cell<u8> = const { Cell::new(0) };
}
#[cfg(test)]
pub(super) fn forced_split() -> bool {
    FORCE_SPLIT.with(Cell::get)
}
#[cfg(test)]
pub(super) fn with_forced_split<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_SPLIT.set(self.0);
        }
    }
    let _restore = Restore(FORCE_SPLIT.replace(true));
    run()
}

#[cfg(test)]
pub(super) fn with_disabled_split<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLE_SPLIT.set(self.0);
        }
    }
    let _restore = Restore(DISABLE_SPLIT.replace(true));
    run()
}
#[cfg(test)]
pub(super) fn reset_split_records() {
    SPLIT_RECORDS.set(0);
}
#[cfg(test)]
pub(super) fn split_records() -> usize {
    SPLIT_RECORDS.get()
}

#[cfg(test)]
pub(super) fn with_forced_splits<T>(splits: usize, run: impl FnOnce() -> T) -> T {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCED_SPLITS.set(self.0);
        }
    }
    let _restore = Restore(FORCED_SPLITS.replace(splits));
    run()
}

#[cfg(test)]
pub(super) fn with_forced_loop<T>(mode: u8, run: impl FnOnce() -> T) -> T {
    struct Restore(u8);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCED_LOOP.set(self.0);
        }
    }
    let _restore = Restore(FORCED_LOOP.replace(mode));
    run()
}
