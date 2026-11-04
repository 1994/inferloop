//! Per-program activation quantization buffers, retained as long as captured graphs.
mod preparation;
mod vendor_fp4;
use super::super::buffers::buffer;
use super::super::capture::error as capture_error;
use super::quant_reuse::{CaptureQuantReuse, QuantKey, QuantKind, QuantPairRole};
use crate::constants::FP8_BLOCK_COLUMNS;
use crate::constants::NVFP4_QUANT_CODES_TILE;
use crate::constants::NVFP4_QUANT_SCALES_TILE;
use crate::resident::norm_fp4;
use infer_cuda_kernels::contract::{
    Buffer, Fp4QuantKind, Fp4QuantPlan, Fp8, Fp8QuantKind, Fp8QuantPlan, GemmLoop, GemmPlan,
    Kernels, NormFp4Buffers, NormMaximumBuffers, QuantNormPlan, Recording, WeightRef,
};

use crate::device::vendor_fp8::Gemm;
use crate::resident::program::ProgramGeometry;
use crate::tuning::fp8_gemm::Choice;
use crate::tuning::fp8_gemm::WeightKey;
use crate::tuning::resident_gemm::native_tile;

use crate::constants::{DEFAULT_TILE_COLUMNS, DEFAULT_TILE_ROWS};
use crate::strategy::LinearTiling;
use crate::tuning::pool_dense::{self, Binding as PoolDenseBinding, Choice as PoolDenseChoice};
use crate::tuning::projection_budget::ProjectionShape;
use crate::tuning::resident_gemm::GemmConfig;
use crate::tuning::resident_gemm::swiglu::{self, Binding as SwigluBinding};
use cutile::half::bf16;

use crate::tuning::resident_gemm::QUANTIZATION_ROWS;
use crate::tuning::resident_gemm::WeightKey as ResidentGemmWeightKey;
use crate::{
    device::{CudaDevice, vendor_fp8},
    mlp::ProjectionWeight,
    resident::ProgramWeights,
};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use infer_core::Result;
use infer_ir::DataflowGraph;
use std::collections::BTreeMap;
#[cfg(test)]
use std::env::var;

use std::result::Result as StdResult;

type Quantized = (Tensor<f4e2m1fnx2>, Tensor<f8e4m3fn>);
/// Keyed by `(rows, columns, activation-scale columns)`.
type Fp8Key = (usize, usize, usize);
type StrategyKey = (ResidentGemmWeightKey, usize, usize, usize);
/// Quantized activation and its per-token scales.
type Fp8Buffers = (Tensor<f8e4m3fn>, Tensor<f32>);

pub struct Workspace {
    quant_reuse: CaptureQuantReuse,
    norm_maximum: Option<super::super::norm_maximum::Pending>,
    norm_fp4: Option<norm_fp4::Pending>,
    strategies: BTreeMap<StrategyKey, GemmConfig>,
    swiglu_strategies: BTreeMap<SwigluBinding, GemmConfig>,
    pool_dense_strategies: BTreeMap<PoolDenseBinding, PoolDenseChoice>,
    prompt_dense_strategies: BTreeMap<PoolDenseBinding, PoolDenseChoice>,
    buffers: BTreeMap<(usize, usize), Quantized>,
    fp8_strategies: BTreeMap<(WeightKey, usize, usize, usize), Choice>,
    /// Lt handle, descriptors and original weight allocations outlive every program capture.
    vendor_fp8: vendor_fp8::Support,
    vendor_fp4: vendor_fp8::nvfp4::Support,
    /// Explicit diagnostic override; production automatic choices still require exact gates.
    vendor_diagnostic: bool,
    /// Keyed by `(rows, columns, scale_columns)`: per-channel and block-scaled FP8 share an
    /// operand geometry but need different activation-scale extents, so geometry alone would let
    /// whichever projection is visited first decide the layout for the other.
    fp8: BTreeMap<Fp8Key, Fp8Buffers>,
}

impl Workspace {
    /// Start a fresh graph-local permission epoch; no device memory is changed.
    pub(crate) const fn begin_quant_capture(&mut self) {
        self.norm_maximum = None;
        self.norm_fp4 = None;
        self.quant_reuse.begin();
    }
    pub(crate) fn enter_quant_node(&mut self, index: usize, role: Option<QuantPairRole>) {
        if self
            .norm_maximum
            .is_some_and(|pending| pending.next != index)
        {
            self.norm_maximum = None;
        }
        if self.norm_fp4.is_some_and(|pending| pending.next != index) {
            self.norm_fp4 = None;
        }
        self.quant_reuse.enter(index, role);
    }
    pub(crate) const fn end_quant_capture(&mut self) {
        self.norm_maximum = None;
        self.norm_fp4 = None;
        self.quant_reuse.end();
    }
    pub(crate) const fn clear_quant_reuse(&mut self) {
        self.norm_maximum = None;
        self.norm_fp4 = None;
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
        for (binding, config) in &self.swiglu_strategies {
            let [rows, outputs, columns] = binding.shape();
            *groups
                .entry((
                    "nvfp4-swiglu",
                    rows,
                    outputs,
                    columns,
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
        let mut workspace = Self::new(device, graph, weights, &widths)?;
        workspace.prepare_pool_dense(device, graph, weights, geometry.verification)?;
        Ok(workspace)
    }

    pub(crate) fn pool_dense_choice(
        &self,
        weight: &ProjectionWeight,
        shape: ProjectionShape,
        tile: LinearTiling,
    ) -> PoolDenseChoice {
        let ProjectionWeight::Dense(weight) = weight else {
            return pool_dense::baseline(shape);
        };
        self.pool_dense_strategies
            .get(&PoolDenseBinding::new(weight.tensor(), shape, tile))
            .copied()
            .unwrap_or_else(|| pool_dense::baseline(shape))
    }

    pub(crate) fn ensure_width(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        width: usize,
    ) -> Result<()> {
        self.extend(device, graph, weights, &[width])
    }

    pub(crate) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<Self> {
        let vendor_fp8 = vendor_fp8::Support::new_with(device, true);
        let vendor_fp4 = vendor_fp8::nvfp4::Support::new(
            &vendor_fp8,
            &weights.resident_tuning,
            weights.autotune,
        );
        let mut workspace = Self {
            quant_reuse: CaptureQuantReuse::default(),
            norm_maximum: None,
            norm_fp4: None,
            strategies: BTreeMap::new(),
            swiglu_strategies: BTreeMap::new(),
            pool_dense_strategies: BTreeMap::new(),
            prompt_dense_strategies: BTreeMap::new(),
            buffers: BTreeMap::new(),
            fp8: BTreeMap::new(),
            fp8_strategies: BTreeMap::new(),
            vendor_fp8,
            vendor_fp4,
            vendor_diagnostic: vendor_diagnostic_enabled(),
        };
        workspace.extend(device, graph, weights, widths)?;
        for &rows in widths {
            workspace.prepare_prompt_rows(device, graph, weights, rows)?;
        }
        Ok(workspace)
    }

    /// Measured output tile of one prompt projection; the established tile when unmeasured.
    pub(crate) fn prompt_dense_choice(
        &self,
        weight: &Arc<Tensor<bf16>>,
        shape: ProjectionShape,
    ) -> PoolDenseChoice {
        LinearTiling::new(DEFAULT_TILE_ROWS, DEFAULT_TILE_COLUMNS)
            .ok()
            .and_then(|tile| {
                self.prompt_dense_strategies
                    .get(&PoolDenseBinding::new(weight, shape, tile))
                    .copied()
            })
            .unwrap_or(PoolDenseChoice::PromptBase)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Quantization records the exact view pointer alongside its original projection contract"
    )]
    pub(super) fn record_input(
        &mut self,
        (contract, scope): (&Kernels, &Scope),
        weight: &ProjectionWeight,
        scale: f32,
        input: Buffer<'_, f32>,
        output: &Tensor<f32>,
        columns: usize,
        certified: bool,
    ) -> StdResult<(), DeviceError> {
        let ProjectionWeight::Fp4(..) = weight else {
            return Err(super::super::capture::error("FP4 workspace weight"));
        };
        let rows = output.size()
            / usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let (q, qs) = self
            .buffers
            .get_mut(&(rows, columns))
            .ok_or_else(|| super::super::capture::error("FP4 quantization workspace missing"))?;
        let key = quant_key(
            certified,
            input,
            output,
            rows,
            columns,
            QuantKind::Fp4(scale.to_bits()),
            (&*q, &*qs),
        );
        let (reuse, ticket) = self.quant_reuse.prepare(key);
        let record = || {
            if !reuse {
                contract
                    .projection
                    .quantize_fp4(
                        &Recording::scope(scope),
                        &fp4_quant_plan(rows, columns),
                        buffer(&*q)?,
                        buffer(&*qs)?,
                        input,
                        scale,
                    )
                    .map_err(capture_error)?;
            }
            Self::record_quantized_gemm(
                &self.strategies,
                (contract, scope),
                weight,
                scale,
                (&*q, &*qs),
                output,
                (rows, columns),
            )
        };
        let result = record();
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
        (contract, scope): (&Kernels, &Scope),
        weight: &ProjectionWeight,
        scale: f32,
        gate: Buffer<'_, f32>,
        up: Buffer<'_, f32>,
        output: &Tensor<f32>,
        columns: usize,
    ) -> StdResult<(), DeviceError> {
        let ProjectionWeight::Fp4(..) = weight else {
            return Err(super::super::capture::error("FP4 workspace weight"));
        };
        let rows = output.size()
            / usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        // The fused codes overwrite the shared quantization buffers; no reuse permit may span it.
        self.quant_reuse.clear();
        let (q, qs) = self
            .buffers
            .get_mut(&(rows, columns))
            .ok_or_else(|| super::super::capture::error("FP4 quantization workspace missing"))?;
        if !rows.is_multiple_of(QUANTIZATION_ROWS) {
            return Err(super::super::capture::error("fused SwiGLU row tile"));
        }
        contract
            .projection
            .quantize_swiglu(
                &Recording::scope(scope),
                &fp4_quant_plan(rows, columns),
                buffer(&*q)?,
                buffer(&*qs)?,
                (gate, up),
                scale,
            )
            .map_err(capture_error)?;
        Self::record_quantized_gemm(
            &self.strategies,
            (contract, scope),
            weight,
            scale,
            (&*q, &*qs),
            output,
            (rows, columns),
        )
    }

    /// Whether [`Self::record_gate_up_swiglu`] can run for this prompt width and MLP geometry.
    pub(crate) fn gate_up_swiglu_ready(&self, rows: usize, columns: usize, hidden: usize) -> bool {
        swiglu::compatible(swiglu::baseline(), [rows, hidden, columns])
            && rows.is_multiple_of(QUANTIZATION_ROWS)
            && self.buffers.contains_key(&(rows, columns))
            && self.buffers.contains_key(&(rows, hidden))
    }

    /// Gate and up projections of a gated MLP fused with the down projection's quantizer:
    /// quantize the shared input, then write the `silu(gate) * up` codes straight into the down
    /// projection's buffer for [`Self::record_prequantized`]. Neither F32 projection is written.
    pub(crate) fn record_gate_up_swiglu(
        &mut self,
        (contract, scope): (&Kernels, &Scope),
        (gate, up): (&ProjectionWeight, &ProjectionWeight),
        (input_scale, down_scale): (f32, f32),
        input: Buffer<'_, f32>,
        (rows, columns, hidden): (usize, usize, usize),
    ) -> StdResult<(), DeviceError> {
        let pointer = input.address();
        let pending = self.norm_fp4.take();
        let (ProjectionWeight::Fp4(..), ProjectionWeight::Fp4(..)) = (gate, up) else {
            return Err(super::super::capture::error("fused SwiGLU weights"));
        };
        let binding = SwigluBinding::new(
            (gate, up),
            (input_scale, down_scale),
            [rows, hidden, columns],
        )
        .map_err(super::super::capture::error)?;
        let config = self
            .swiglu_strategies
            .get(&binding)
            .copied()
            .unwrap_or(swiglu::baseline());
        #[cfg(test)]
        let config = swiglu::recorded_config(config);
        if !self.gate_up_swiglu_ready(rows, columns, hidden) {
            return Err(super::super::capture::error("fused SwiGLU geometry"));
        }
        // Both writes replace shared quantization buffers; no reuse permit may span them.
        self.quant_reuse.clear();
        let (q, qs) = self
            .buffers
            .remove(&(rows, columns))
            .ok_or_else(|| super::super::capture::error("FP4 quantization workspace missing"))?;
        let ready = pending.is_some_and(|pending| {
            pending.binding
                == norm_fp4::Binding {
                    pointer,
                    rows,
                    columns,
                    scale_bits: input_scale.to_bits(),
                    codes: q.device_pointer().cu_deviceptr(),
                    scales: qs.device_pointer().cu_deviceptr(),
                }
        });
        #[cfg(test)]
        norm_fp4::record_consumption(ready);
        let quantized = if ready {
            Ok(())
        } else {
            buffer(&q).and_then(|codes| {
                contract
                    .projection
                    .quantize_fp4(
                        &Recording::scope(scope),
                        &fp4_quant_plan(rows, columns),
                        codes,
                        buffer(&qs)?,
                        input,
                        input_scale,
                    )
                    .map_err(capture_error)
            })
        };
        let recorded = quantized.and_then(|()| {
            let (codes, code_scales) = self.buffers.get_mut(&(rows, hidden)).ok_or_else(|| {
                super::super::capture::error("FP4 quantization workspace missing")
            })?;
            swiglu::record(
                (contract, scope),
                config,
                (gate, up),
                (input_scale, down_scale),
                (&q, &qs),
                (codes, code_scales),
                columns,
            )
        });
        self.buffers.insert((rows, columns), (q, qs));
        recorded
    }

    /// Down projection whose activation codes [`Self::record_gate_up_swiglu`] already wrote.
    pub(crate) fn record_prequantized(
        &self,
        (contract, scope): (&Kernels, &Scope),
        weight: &ProjectionWeight,
        scale: f32,
        output: &Tensor<f32>,
        columns: usize,
    ) -> StdResult<(), DeviceError> {
        let rows = output.size()
            / usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let (q, qs) = self
            .buffers
            .get(&(rows, columns))
            .ok_or_else(|| super::super::capture::error("FP4 quantization workspace missing"))?;
        Self::record_quantized_gemm(
            &self.strategies,
            (contract, scope),
            weight,
            scale,
            (q, qs),
            output,
            (rows, columns),
        )
    }

    fn record_quantized_gemm(
        strategies: &BTreeMap<StrategyKey, GemmConfig>,
        (contract, scope): (&Kernels, &Scope),
        weight: &ProjectionWeight,
        scale: f32,
        (q, qs): (&Tensor<f4e2m1fnx2>, &Tensor<f8e4m3fn>),
        output: &Tensor<f32>,
        (rows, columns): (usize, usize),
    ) -> StdResult<(), DeviceError> {
        {
            let outputs =
                usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
            let config = strategies
                .get(&(
                    ResidentGemmWeightKey::new(weight, scale)
                        .map_err(super::super::capture::error)?,
                    rows,
                    outputs,
                    columns,
                ))
                .copied()
                .unwrap_or_else(|| GemmConfig::baseline(rows));
            config.record((contract, scope), weight, scale, q, qs, output, columns)?;
            Ok(())
        }
    }
    /// Write existing dense and FP4 buffers in one RMS pass; permission is capture-local.
    pub(crate) fn record_norm_fp4(
        &mut self,
        (kernels, scope): (&Kernels, &Scope),
        next: usize,
        input: Buffer<'_, f32>,
        weight: &Tensor<f32>,
        output: &Tensor<f32>,
        (epsilon, offset, scale): (f32, f32, f32),
    ) -> StdResult<bool, DeviceError> {
        self.norm_fp4 = None;
        let rows = usize::try_from(output.shape()[0]).map_err(capture_error)?;
        let columns = usize::try_from(output.shape()[1]).map_err(capture_error)?;
        let Some((codes, scales)) = self.buffers.get(&(rows, columns)) else {
            return Ok(false);
        };
        let binding = norm_fp4::Binding {
            pointer: output.device_pointer().cu_deviceptr(),
            rows,
            columns,
            scale_bits: scale.to_bits(),
            codes: codes.device_pointer().cu_deviceptr(),
            scales: scales.device_pointer().cu_deviceptr(),
        };
        kernels
            .norm
            .fp4(
                &Recording::scope(scope),
                &QuantNormPlan {
                    rows,
                    dimension: columns,
                    epsilon,
                    offset,
                },
                scale,
                NormFp4Buffers {
                    out: buffer(output)?,
                    codes: buffer(codes)?,
                    scales: buffer(scales)?,
                    x: input,
                    weight: buffer(weight)?,
                },
            )
            .map_err(capture_error)?;
        self.norm_fp4 = Some(norm_fp4::Pending { next, binding });
        Ok(true)
    }

    /// Keep the existing scale allocation; its permission lasts exactly one graph node.
    pub(crate) fn record_norm_maximum(
        &mut self,
        (kernels, scope): (&Kernels, &Scope),
        next: usize,
        input: Buffer<'_, f32>,
        weight: &Tensor<f32>,
        output: &Tensor<f32>,
        (epsilon, offset): (f32, f32),
    ) -> StdResult<bool, DeviceError> {
        let rows = usize::try_from(output.shape()[0]).map_err(capture_error)?;
        let columns = usize::try_from(output.shape()[1]).map_err(capture_error)?;
        let Some((_, scales)) = self.fp8.get(&(rows, columns, 1)) else {
            return Ok(false);
        };
        let pointer = output.device_pointer().cu_deviceptr();
        kernels
            .norm
            .maximum(
                &Recording::scope(scope),
                &QuantNormPlan {
                    rows,
                    dimension: columns,
                    epsilon,
                    offset,
                },
                NormMaximumBuffers {
                    out: buffer(output)?,
                    maximum: buffer(scales)?,
                    x: input,
                    weight: buffer(weight)?,
                },
            )
            .map_err(capture_error)?;
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
        (contract, scope): (&Kernels, &Scope),
        weight: &ProjectionWeight,
        input: Buffer<'_, f32>,
        output: &Tensor<f32>,
        columns: usize,
        certified: bool,
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
                && pending.matches(certified.then_some(input.address()), rows, columns)
        });
        let (q, qs) = self
            .fp8
            .get_mut(&(rows, columns, scale_columns))
            .ok_or_else(|| error("FP8 quantization workspace missing"))?;
        let kind = matches!(weight, ProjectionWeight::Fp8(..)).then_some(QuantKind::Fp8Token);
        let key = kind
            .and_then(|kind| quant_key(certified, input, output, rows, columns, kind, (&*q, &*qs)));
        let (reuse, ticket) = self.quant_reuse.prepare(key);
        let record = || {
            let recording = Recording::scope(scope);
            match weight {
                ProjectionWeight::Fp8Block(w, block_scale) => {
                    record_fp8_block(
                        (contract, &recording),
                        WeightRef::Fp8 {
                            weight: buffer(w)?,
                            scales: buffer(block_scale)?,
                        },
                        input,
                        (buffer(&*q)?, buffer(&*qs)?),
                        (
                            buffer(&*output)?,
                            usize::try_from(output.shape()[1]).map_err(error)?,
                        ),
                        (rows, columns),
                    )?;
                }
                ProjectionWeight::Fp8(w, channel_scale) => {
                    if !reuse {
                        contract
                            .projection
                            .quantize_fp8(
                                &recording,
                                &Fp8QuantPlan {
                                    kind: super::super::fp8_quant_reduction::kind(
                                        columns,
                                        maximum_ready,
                                    ),
                                    rows,
                                    columns,
                                },
                                buffer(&*q)?,
                                buffer(&*qs)?,
                                input,
                            )
                            .map_err(error)?;
                        #[cfg(test)]
                        super::super::norm_maximum::record_consumption(maximum_ready);
                    }
                    let outputs = usize::try_from(output.shape()[1]).map_err(error)?;
                    let gemm = Gemm {
                        input: buffer(&*q)?,
                        weight: buffer(w)?,
                        output: buffer(&*output)?,
                        extents: [rows, outputs, columns],
                    };
                    let scales = [buffer(&*qs)?, buffer(channel_scale)?];
                    if self.vendor_diagnostic && self.vendor_fp8.record(&recording, gemm, scales)? {
                        return Ok(());
                    }
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
                    choice.record((contract, &recording), &self.vendor_fp8, gemm, scales)?;
                }
                _ => return Err(error("FP8 workspace weight")),
            }
            Ok(())
        };
        let result = record();
        self.quant_reuse.finish(ticket, result.is_ok());
        result
    }
}

/// Block-scaled FP8 quantizes every block on its own and takes no reuse or vendor path.
fn record_fp8_block(
    (contract, recording): (&Kernels, &Recording<'_>),
    weight: WeightRef<'_>,
    input: Buffer<'_, f32>,
    (q, qs): (Buffer<'_, Fp8>, Buffer<'_, f32>),
    (output, outputs): (Buffer<'_, f32>, usize),
    (rows, columns): (usize, usize),
) -> StdResult<(), DeviceError> {
    contract
        .projection
        .quantize_fp8(
            recording,
            &Fp8QuantPlan {
                kind: Fp8QuantKind::Block {
                    block: FP8_BLOCK_COLUMNS,
                },
                rows,
                columns,
            },
            q,
            qs,
            input,
        )
        .map_err(capture_error)?;
    contract
        .projection
        .gemm_fp8(
            recording,
            &GemmPlan {
                main_loop: GemmLoop::Plain,
                rows,
                outputs,
                columns,
                tile: native_tile(rows),
            },
            output,
            (q, qs),
            weight,
            Some(FP8_BLOCK_COLUMNS),
        )
        .map_err(capture_error)
}

/// Rows that fill whole quantization groups run them in one launch.
const fn fp4_quant_plan(rows: usize, columns: usize) -> Fp4QuantPlan {
    let (kind, codes_tile, scales_tile) = if rows.is_multiple_of(QUANTIZATION_ROWS) {
        (
            Fp4QuantKind::RowGroups {
                rows: QUANTIZATION_ROWS,
            },
            [QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]],
            [QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]],
        )
    } else {
        (
            Fp4QuantKind::Whole,
            NVFP4_QUANT_CODES_TILE,
            NVFP4_QUANT_SCALES_TILE,
        )
    };
    Fp4QuantPlan {
        kind,
        rows,
        columns,
        codes_tile,
        scales_tile,
    }
}

const fn fp8_scale_columns(weight: &ProjectionWeight, columns: usize) -> usize {
    match weight {
        ProjectionWeight::Fp8Block(..) => columns / FP8_BLOCK_COLUMNS,
        _ => 1,
    }
}

/// Only the certified whole-matrix paths may reuse quantized activations.
fn quant_key<Q: DType, S: DType>(
    certified: bool,
    input: Buffer<'_, f32>,
    output: &Tensor<f32>,
    rows: usize,
    columns: usize,
    kind: QuantKind,
    buffers: (&Tensor<Q>, &Tensor<S>),
) -> Option<QuantKey> {
    let input_pointer = certified.then_some(input.address()).filter(|p| *p != 0)?;
    if input.len() != rows.checked_mul(columns)? {
        return None;
    }
    let shape = [i32::try_from(rows).ok()?, i32::try_from(columns).ok()?];
    let strides = [shape[1], 1];
    let input_end = input_pointer
        .checked_add(u64::try_from(input.len().checked_mul(size_of::<f32>())?).ok()?)?;
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
#[path = "../../../tests/unit/resident_norm_fp4_permissions.rs"]
mod norm_fp4_permission_tests;
