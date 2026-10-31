//! Per-program activation quantization buffers, retained as long as captured graphs.
mod preparation;
mod vendor_fp4;
use super::{
    kernels,
    quant_reuse::{CaptureQuantReuse, QuantKey, QuantKind, QuantPairRole},
};
use crate::constants::FP8_BLOCK_COLUMNS;
use crate::constants::NVFP4_GROUP_SIZE;
use crate::constants::NVFP4_QUANT_CODES_TILE;
use crate::constants::NVFP4_QUANT_SCALES_TILE;
use crate::resident::{norm_fp4, norm_fp4_kernel};

use crate::resident::program::ProgramGeometry;
use crate::tuning::fp8_gemm::Choice;
use crate::tuning::fp8_gemm::WeightKey;
use crate::tuning::resident_gemm::native_tile;

use crate::strategy::LinearTiling;
use crate::tuning::pool_dense::{self, Binding as PoolDenseBinding, Choice as PoolDenseChoice};
use crate::tuning::projection_budget::ProjectionShape;
use crate::tuning::resident_gemm::GemmConfig;
use crate::tuning::resident_gemm::swiglu::{self, Binding as SwigluBinding};

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
            .get(&PoolDenseBinding::new(weight, shape, tile))
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
            buffers: BTreeMap::new(),
            fp8: BTreeMap::new(),
            fp8_strategies: BTreeMap::new(),
            vendor_fp8,
            vendor_fp4,
            vendor_diagnostic: vendor_diagnostic_enabled(),
        };
        workspace.extend(device, graph, weights, widths)?;
        Ok(workspace)
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
        let rows = output.size()
            / usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
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
                &self.strategies,
                scope,
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
        Self::record_quantized_gemm(
            &self.strategies,
            scope,
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
        scope: &Scope,
        (gate, up): (&ProjectionWeight, &ProjectionWeight),
        (input_scale, down_scale): (f32, f32),
        (input, pointer): (&TensorView<'_, f32>, u64),
        (rows, columns, hidden): (usize, usize, usize),
    ) -> StdResult<(), DeviceError> {
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
        let (mut q, mut qs) = self
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
            scope
                .record(
                    kernels::quantize_rows(
                        (&mut q).partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]]),
                        (&mut qs).partition([QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]]),
                        input,
                        input_scale,
                    )
                    .generics(vec![columns.to_string()]),
                )
                .map(drop)
        };
        let recorded = quantized.and_then(|()| {
            let (codes, code_scales) = self.buffers.get_mut(&(rows, hidden)).ok_or_else(|| {
                super::super::capture::error("FP4 quantization workspace missing")
            })?;
            swiglu::record(
                scope,
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
        scope: &Scope,
        weight: &ProjectionWeight,
        scale: f32,
        output: &mut Tensor<f32>,
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
            scope,
            weight,
            scale,
            (q, qs),
            output,
            (rows, columns),
        )
    }

    fn record_quantized_gemm(
        strategies: &BTreeMap<StrategyKey, GemmConfig>,
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
            config.record(scope, weight, scale, q, qs, output, columns)?;
            Ok(())
        }
    }
    /// Write existing dense and FP4 buffers in one RMS pass; permission is capture-local.
    pub(crate) fn record_norm_fp4(
        &mut self,
        scope: &Scope,
        next: usize,
        input: &TensorView<'_, f32>,
        weight: &Tensor<f32>,
        output: &mut Tensor<f32>,
        (epsilon, offset, scale): (f32, f32, f32),
    ) -> StdResult<bool, DeviceError> {
        self.norm_fp4 = None;
        let rows = usize::try_from(output.shape()[0]).map_err(super::super::capture::error)?;
        let columns = usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let Some((codes, scales)) = self.buffers.get_mut(&(rows, columns)) else {
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
        let block = columns.next_power_of_two();
        scope.record(
            norm_fp4_kernel::kernels::norm_fp4(
                output.partition([1, block]),
                codes.partition([1, block / 2]),
                scales.partition([1, block / NVFP4_GROUP_SIZE]),
                input,
                weight,
                epsilon,
                offset,
                scale,
            )
            .generics(
                [columns, block, block / NVFP4_GROUP_SIZE, block / 2]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            ),
        )?;
        self.norm_fp4 = Some(norm_fp4::Pending { next, binding });
        Ok(true)
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

    pub(super) fn record_fp8_input(
        &mut self,
        scope: &Scope,
        weight: &ProjectionWeight,
        input: &TensorView<'_, f32>,
        output: &mut Tensor<f32>,
        columns: usize,
        input_pointer: Option<u64>,
    ) -> StdResult<(), DeviceError> {
        use super::super::{capture::error, fp8_gemm::kernels};
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
            let tile = native_tile(rows);
            let generics = [columns, tile[0], tile[1]]
                .map(|value| value.to_string())
                .to_vec();
            match weight {
                ProjectionWeight::Fp8Block(w, block_scale) => {
                    scope.record(
                        kernels::quantize_block(
                            (&mut *q).partition([1, FP8_BLOCK_COLUMNS]),
                            (&mut *qs).partition([1, 1]),
                            input,
                        )
                        .generics(vec![columns.to_string(), FP8_BLOCK_COLUMNS.to_string()]),
                    )?;
                    scope.record(
                        kernels::matmul_block(output.partition(tile), &*q, w, &*qs, block_scale)
                            .generics(generics),
                    )?;
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
                    if self.vendor_diagnostic
                        && self
                            .vendor_fp8
                            .record(scope, &*q, w, output, &*qs, channel_scale)?
                    {
                        return Ok(());
                    }
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
        self.quant_reuse.finish(ticket, result.is_ok());
        result
    }
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
#[path = "../../../tests/unit/resident_norm_fp4_permissions.rs"]
mod norm_fp4_permission_tests;
