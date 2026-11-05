mod dense;
use super::buffers::{RowMajor, buffer, tensor_rows};
use super::prompt_scratch::Partials;
use super::{ProgramWeights, capture::error};
use crate::constants::PREFILL_GEMM_TILE_K;
use crate::constants::PREFILL_GEMM_TILE_N;
use crate::constants::PREFILL_LANES;
use crate::constants::PREFILL_NVFP4_TILE_N;
use crate::constants::PREFILL_SPLIT_K;
use crate::constants::{DEFAULT_TILE_COLUMNS, DEFAULT_TILE_ROWS};
use crate::device::CudaDevice;
use crate::device::cublaslt::Support;
use crate::mlp::Matrix;
use crate::strategy::LinearTiling;
use crate::tuning::pool_dense::Choice as PromptChoice;
use crate::{device::DeviceProfile, mlp::ProjectionWeight};
use cutile::half::bf16;
use cutile::prelude::*;
pub(super) use dense::DensePlan;
use infer_cuda_kernels::contract::{
    Bf16, Buffer, DenseKernel, DenseRowsPlan, Kernels, PromptPlan, Recording, SplitGemmPlan,
};
use infer_ir::TensorNode;
#[cfg(test)]
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

use crate::tuning::prefill::{delegate_dense, narrow_dense_layout};
const NARROW_DENSE_TILE: [usize; 2] = [PREFILL_LANES, 16];

/// Down projection absorbing `silu(gate) * up`: node indices of the `SiLU` and multiply it
/// replaces, then the gate and up tensors it reads.
type Swiglu = (usize, usize, infer_core::TensorId, infer_core::TensorId);

/// Target-graph NVFP4 down projections whose input is `silu(gate) * up` with no other reader of
/// the `SiLU` or product rows. Their prompt capture quantizes straight from gate and up.
fn swiglu_plan(
    graph: &infer_ir::DataflowGraph,
    weights: &ProgramWeights,
) -> BTreeMap<usize, Swiglu> {
    use infer_ir::TensorOp;
    let mut plan = BTreeMap::new();
    if weights.fusion.is_some() {
        return plan;
    }
    let (producer, readers) = dataflow(graph);
    let results = [graph.hidden, graph.logits];
    let sole = |id, reader: usize| {
        !results.contains(&Some(id)) && readers.get(&id).is_some_and(|r| r[..] == [reader])
    };
    for (index, node) in graph.nodes.iter().enumerate() {
        if node.op != TensorOp::Linear
            || !matches!(
                weights.projections.get(&node.inputs[1]),
                Some(ProjectionWeight::Fp4(..))
            )
            || !matches!(
                weights.activation_quantization(node.inputs[1]),
                Some(super::program::ActivationQuantization::Fp4(_))
            )
        {
            continue;
        }
        let product = node.inputs[0];
        let Some(&multiply) = producer.get(&product) else {
            continue;
        };
        let multiply_node = &graph.nodes[multiply];
        if multiply_node.op != TensorOp::Multiply || !sole(product, index) {
            continue;
        }
        let [left, right] = [multiply_node.inputs[0], multiply_node.inputs[1]];
        for (activated, up) in [(left, right), (right, left)] {
            if let Some(&silu) = producer.get(&activated)
                && graph.nodes[silu].op == TensorOp::Silu
                && sole(activated, multiply)
            {
                plan.insert(index, (silu, multiply, graph.nodes[silu].inputs[0], up));
                break;
            }
        }
    }
    plan
}

/// Gated-MLP capture plan of one prompt graph: the `SiLU` and multiply nodes absorbed into
/// their down projection, and the gate/up pairs that run as one fused pass at this width.
pub(super) struct MlpPlan {
    swiglu: BTreeMap<usize, Swiglu>,
    fused: BTreeMap<usize, FusedMlp>,
    fused_downs: BTreeSet<usize>,
    absorbed: BTreeSet<usize>,
}

impl MlpPlan {
    /// Warm and screen only down nodes whose F32 product this exact plan never materializes.
    pub(super) fn prepare_vendor(
        &self,
        device: &CudaDevice,
        graph: &infer_ir::DataflowGraph,
        weights: &ProgramWeights,
        arena: &super::ActivationArena,
        nvfp4: &mut super::nvfp4_gemm::Workspace,
    ) -> Result<(), DeviceError> {
        if weights.fusion.is_some() || !weights.attention_profile.is_bf16() {
            return Ok(());
        }
        let rows = arena.lanes();
        for &index in &self.fused_downs {
            let node = &graph.nodes[index];
            let Some(super::program::ActivationQuantization::Fp4(scale)) =
                weights.activation_quantization(node.inputs[1])
            else {
                continue;
            };
            let weight = weights
                .projections
                .get(&node.inputs[1])
                .ok_or_else(|| error("fused down vendor weight"))?;
            let columns = arena.get(node.inputs[0]).map_err(error)?.len() / rows;
            let Some((input, output)) = down_operands(arena, node)? else {
                continue;
            };
            nvfp4.prepare_vendor_down(
                device,
                weight,
                scale,
                (arena, input),
                arena.buffer(output).map_err(error)?,
                (rows, columns),
            )?;
        }
        Ok(())
    }

    /// Actual gate/up/down projection nodes retained by this prompt fusion plan.
    pub(super) fn fused_projections(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.fused.values().copied()
    }

    pub(super) fn new(
        graph: &infer_ir::DataflowGraph,
        weights: &ProgramWeights,
        geometry: (usize, &super::nvfp4_gemm::Workspace),
    ) -> Self {
        let swiglu = swiglu_plan(graph, weights);
        let fused = fused_mlp_plan(graph, weights, &swiglu, geometry);
        let fused_downs = fused.values().map(|&(.., down)| down).collect();
        let absorbed = swiglu
            .values()
            .flat_map(|&(silu, multiply, ..)| [silu, multiply])
            .chain(
                fused
                    .iter()
                    .map(|(&first, &(gate, up, _))| if first == gate { up } else { gate }),
            )
            .collect();
        Self {
            swiglu,
            fused,
            fused_downs,
            absorbed,
        }
    }

    /// These linear nodes are recorded by the fused gate/up kernel, never by standalone GEMM.
    pub(super) fn fuses_projection(&self, index: usize) -> bool {
        self.fused
            .values()
            .any(|&(gate, up, _)| index == gate || index == up)
    }

    /// Nodes another node of the plan records; the capture skips them.
    pub(super) fn absorbs(&self, index: usize) -> bool {
        self.absorbed.contains(&index)
    }

    /// Fuse normalization only when this plan actually records the following FP4 consumer.
    pub(super) fn record_norm(
        &self,
        scope: &Scope,
        (index, nodes): (usize, &[TensorNode]),
        arena: &super::ActivationArena,
        weights: &ProgramWeights,
        nvfp4: &mut super::nvfp4_gemm::Workspace,
    ) -> Result<bool, DeviceError> {
        if self.fused.contains_key(&(index + 1))
            && super::norm_fp4::record(scope, (index, nodes), arena, weights, nvfp4)?
        {
            return Ok(true);
        }
        super::norm_maximum::record(scope, index, nodes, arena, weights, nvfp4)
    }

    /// Record node `index` when the plan owns it. False leaves it to the ordinary dispatch.
    pub(super) fn record(
        &self,
        scope: &Scope,
        (index, nodes): (usize, &[TensorNode]),
        arena: &super::ActivationArena,
        weights: &ProgramWeights,
        nvfp4: &mut super::nvfp4_gemm::Workspace,
    ) -> Result<bool, DeviceError> {
        if let Some(&(gate, up, down)) = self.fused.get(&index) {
            let projections = (&nodes[gate], &nodes[up], &nodes[down]);
            record_fused_mlp(scope, projections, arena, weights, nvfp4)?;
            return Ok(true);
        }
        let node = &nodes[index];
        if self.fused_downs.contains(&index) {
            record_fused_down(scope, node, arena, weights, nvfp4)?;
            return Ok(true);
        }
        let Some(&(.., gate, up)) = self.swiglu.get(&index) else {
            return Ok(false);
        };
        if !record_swiglu(scope, node, (gate, up), arena, weights, nvfp4)? {
            return Err(error("fused SwiGLU projection"));
        }
        Ok(true)
    }
}

/// Producing node and reading nodes of every tensor.
type Dataflow = (
    BTreeMap<infer_core::TensorId, usize>,
    BTreeMap<infer_core::TensorId, Vec<usize>>,
);

fn dataflow(graph: &infer_ir::DataflowGraph) -> Dataflow {
    let mut producer = BTreeMap::new();
    let mut readers: BTreeMap<_, Vec<usize>> = BTreeMap::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        for id in &node.outputs {
            producer.insert(*id, index);
        }
        for id in &node.inputs {
            readers.entry(*id).or_default().push(index);
        }
    }
    (producer, readers)
}

/// Gate, up and down node indices of a gated MLP recorded as one fused gate/up pass.
type FusedMlp = (usize, usize, usize);

/// [`swiglu_plan`] entries whose gate and up projections can also run as one pass at `rows`
/// prompt lanes, keyed by whichever projection comes first. Both projections must read the same
/// input with the same FP4 activation scale and feed only the absorbed `SiLU` and multiply.
fn fused_mlp_plan(
    graph: &infer_ir::DataflowGraph,
    weights: &ProgramWeights,
    swiglu: &BTreeMap<usize, Swiglu>,
    (rows, nvfp4): (usize, &super::nvfp4_gemm::Workspace),
) -> BTreeMap<usize, FusedMlp> {
    use infer_ir::TensorOp;
    let (producer, readers) = dataflow(graph);
    let fp4_scale = |node: &TensorNode| match (
        weights.projections.get(&node.inputs[1]),
        weights.activation_quantization(node.inputs[1]),
    ) {
        (
            Some(ProjectionWeight::Fp4(..)),
            Some(super::program::ActivationQuantization::Fp4(scale)),
        ) => Some(scale.to_bits()),
        _ => None,
    };
    let sizes: BTreeMap<_, _> = graph
        .tensors
        .iter()
        .filter_map(|tensor| Some((tensor.id, tensor.elements().ok()?)))
        .collect();
    let mut plan = BTreeMap::new();
    for (&down, &(silu, multiply, gate_output, up_output)) in swiglu {
        let (Some(&gate), Some(&up)) = (producer.get(&gate_output), producer.get(&up_output))
        else {
            continue;
        };
        let [gate_node, up_node] = [&graph.nodes[gate], &graph.nodes[up]];
        let sole = |id, reader: usize| readers.get(&id).is_some_and(|r| r[..] == [reader]);
        let (Some(&columns), Some(&hidden)) =
            (sizes.get(&gate_node.inputs[0]), sizes.get(&gate_output))
        else {
            continue;
        };
        if gate_node.op != TensorOp::Linear
            || up_node.op != TensorOp::Linear
            || gate_node.inputs[0] != up_node.inputs[0]
            || fp4_scale(gate_node).is_none()
            || fp4_scale(gate_node) != fp4_scale(up_node)
            || !sole(gate_output, silu)
            || !sole(up_output, multiply)
            || sizes.get(&up_output) != Some(&hidden)
            || !nvfp4.gate_up_swiglu_ready(rows, columns, hidden)
        {
            continue;
        }
        plan.insert(gate.min(up), (gate, up, down));
    }
    plan
}

/// Record a [`fused_mlp_plan`] gate/up pass into the down projection's activation codes.
fn record_fused_mlp(
    scope: &Scope,
    (gate, up, down): (&TensorNode, &TensorNode, &TensorNode),
    arena: &super::ActivationArena,
    weights: &ProgramWeights,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
) -> Result<(), DeviceError> {
    let scale = |node: &TensorNode| match weights.activation_quantization(node.inputs[1]) {
        Some(super::program::ActivationQuantization::Fp4(scale)) => Ok(scale),
        _ => Err(error("fused SwiGLU activation scale")),
    };
    let weight = |node: &TensorNode| {
        weights
            .projections
            .get(&node.inputs[1])
            .ok_or_else(|| error("fused SwiGLU projection"))
    };
    let width = arena.lanes();
    let input = arena.view(gate.inputs[0]).map_err(error)?;
    let columns = input.size() / width;
    let input = buffer(input)?;
    let hidden = arena.get(gate.outputs[0]).map_err(error)?.len() / width;
    nvfp4.record_gate_up_swiglu(
        (&weights.kernels, scope),
        (weight(gate)?, weight(up)?),
        (scale(gate)?, scale(down)?),
        input,
        (width, columns, hidden),
    )
}

/// Record the down projection of a [`fused_mlp_plan`] entry from its prepared codes.
fn record_fused_down(
    scope: &Scope,
    node: &TensorNode,
    arena: &super::ActivationArena,
    weights: &ProgramWeights,
    nvfp4: &super::nvfp4_gemm::Workspace,
) -> Result<(), DeviceError> {
    let Some(super::program::ActivationQuantization::Fp4(scale)) =
        weights.activation_quantization(node.inputs[1])
    else {
        return Err(error("fused SwiGLU down scale"));
    };
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("prefill projection"))?;
    let width = arena.lanes();
    let columns = arena.get(node.inputs[0]).map_err(error)?.len() / width;
    if let Some((input, output)) = down_operands(arena, node)? {
        let output = arena.buffer(output).map_err(error)?;
        let output = RowMajor::new(output, width, output.len() / width)?;
        if nvfp4.record_vendor_down(scope, weight, scale, (arena, input), output, columns)? {
            return Ok(());
        }
        return nvfp4.record_prequantized(
            (&weights.kernels, scope),
            weight,
            scale,
            output,
            columns,
        );
    }
    let output = arena.activation(node.outputs[0]).map_err(error)?;
    nvfp4.record_prequantized(
        (&weights.kernels, scope),
        weight,
        scale,
        RowMajor::new(output, width, output.len() / width)?,
        columns,
    )?;
    Ok(())
}

/// Record a [`swiglu_plan`] down projection. False leaves nothing recorded.
fn record_swiglu(
    scope: &Scope,
    node: &TensorNode,
    (gate, up): (infer_core::TensorId, infer_core::TensorId),
    arena: &super::ActivationArena,
    weights: &ProgramWeights,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
) -> Result<bool, DeviceError> {
    let Some(super::program::ActivationQuantization::Fp4(scale)) =
        weights.activation_quantization(node.inputs[1])
    else {
        return Ok(false);
    };
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("prefill projection"))?;
    let width = arena.lanes();
    let output = arena.activation(node.outputs[0]).map_err(error)?;
    let output = RowMajor::new(output, width, output.len() / width)?;
    let gate = arena.view(gate).map_err(error)?;
    let columns = gate.size() / width;
    let gate = buffer(gate)?;
    let up = arena.activation(up).map_err(error)?;
    nvfp4.record_swiglu_input(
        (&weights.kernels, scope),
        weight,
        scale,
        gate,
        up,
        output,
        columns,
    )?;
    Ok(true)
}

/// Zero-copy prompt GEMM: the batched input slot holds all 32 rows and the
/// output slot is reshaped to `[32, rows]` in place; no packing kernels remain.
/// Fp4 weights split the K loop across `PREFILL_SPLIT_K` CTAs per output tile:
/// the prompt GEMM is latency-bound at one wave, so wider grids hide DRAM latency.
pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    arena: &super::ActivationArena,
    weights: &ProgramWeights,
    partials: &Partials,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
    delegated: Option<&mut Support>,
) -> Result<(), DeviceError> {
    if weights.constants.contains_key(&node.inputs[0]) {
        return Err(error("prefill32 constant linear input"));
    }
    let width = arena.lanes();
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.output(slot).map_err(error)?;
    let flat = output.size();
    let output = output.reshape(&[width, flat / width])?;
    let flat_input = arena.view(node.inputs[0]).map_err(error)?;
    let columns = flat_input.size() / width;
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("prefill projection"))?;
    if super::nvfp4_gemm::record_certified(
        (&weights.kernels, scope),
        nvfp4,
        weight,
        weights.activation_quantization(node.inputs[1]),
        buffer(flat_input)?,
        tensor_rows(&output)?,
        columns,
    )? {
        return Ok(());
    }
    match weight {
        ProjectionWeight::Fp8Block(..) => {
            return Err(error("block FP8 requires the resident FP8 GEMM"));
        }
        ProjectionWeight::Dense(w) => {
            record_prompt_dense(
                scope,
                weights,
                (flat_input, w),
                &output,
                (width, flat / width, columns),
                (nvfp4, delegated),
            )?;
        }
        ProjectionWeight::Fp8(..) => {
            weights
                .kernels
                .projection
                .prompt_fp8(
                    &Recording::scope(scope),
                    &PromptPlan {
                        rows: width,
                        outputs: flat / width,
                        columns,
                        tile: [PREFILL_LANES, PREFILL_GEMM_TILE_N],
                    },
                    buffer(&output)?,
                    buffer(flat_input)?,
                    weight.weight_ref()?,
                )
                .map_err(error)?;
        }
        ProjectionWeight::Fp4(..) => {
            record_legacy_fp4(
                (&weights.kernels, scope),
                weight,
                buffer(flat_input)?,
                &output,
                partials,
                (width, columns),
            )?;
        }
    }
    Ok(())
}

/// Two projections of one input, each with one output.
fn pairable(first: &TensorNode, second: &TensorNode) -> bool {
    first.inputs.len() == 2
        && second.inputs.len() == 2
        && first.inputs[0] == second.inputs[0]
        && first.outputs.len() == 1
        && second.outputs.len() == 1
}

/// The input and output slots of a pair whose extents match, all distinct.
fn pair_slots(
    first: &TensorNode,
    second: &TensorNode,
    arena: &super::ActivationArena,
    width: usize,
    (rows, columns): (usize, usize),
) -> Result<Option<(usize, usize, usize)>, DeviceError> {
    let expected = rows
        .checked_mul(width)
        .ok_or_else(|| error("paired projection extent overflow"))?;
    let input_slot = arena.slot(first.inputs[0]).map_err(error)?;
    let input_elements = arena.buffer(input_slot).map_err(error)?.len();
    if input_elements.checked_div(width) != Some(columns)
        || !input_elements.is_multiple_of(width)
        || arena.get(first.outputs[0]).map_err(error)?.len() != expected
        || arena.get(second.outputs[0]).map_err(error)?.len() != expected
    {
        return Ok(None);
    }
    let slot_b = arena.slot(first.outputs[0]).map_err(error)?;
    let slot_a = arena.slot(second.outputs[0]).map_err(error)?;
    if slot_b == slot_a || slot_b == input_slot || slot_a == input_slot {
        return Ok(None);
    }
    Ok(Some((input_slot, slot_b, slot_a)))
}

/// One dense prompt projection: the sparse router path, a vendor GEMM, or the measured tile.
fn record_prompt_dense(
    scope: &Scope,
    weights: &ProgramWeights,
    (input, w): (&Tensor<f32>, &Matrix<bf16>),
    output: &Tensor<f32>,
    (width, rows, columns): (usize, usize, usize),
    (nvfp4, delegated): (&super::nvfp4_gemm::Workspace, Option<&mut Support>),
) -> Result<(), DeviceError> {
    if records_precise_sparse(
        (&weights.kernels, scope),
        weights,
        (buffer(output)?, buffer(input)?, buffer(w)?),
        columns,
    )? {
        return Ok(());
    }
    // Wide projections are worth a vendor GEMM, square ones are not: `record_dense` decides from
    // the measured shape table.
    if records_vendor_gemm(
        (&weights.kernels, scope),
        delegated,
        input,
        w,
        output,
        (width, rows, columns),
    )? {
        return Ok(());
    }
    let choice = nvfp4.prompt_dense_choice(w, [width, rows, columns]);
    if choice != PromptChoice::PromptBase {
        return choice.record(
            (&weights.kernels, scope),
            w,
            buffer(input)?,
            tensor_rows(output)?,
            columns,
            LinearTiling::new(DEFAULT_TILE_ROWS, DEFAULT_TILE_COLUMNS).map_err(error)?,
        );
    }
    weights
        .kernels
        .projection
        .dense(
            &Recording::scope(scope),
            &DenseRowsPlan {
                kernel: DenseKernel::Prompt {
                    tile: [PREFILL_LANES, PREFILL_GEMM_TILE_N],
                },
                rows: width,
                outputs: rows,
                columns,
            },
            buffer(output)?,
            buffer(input)?,
            buffer(w)?,
        )
        .map_err(error)
}

/// Record two graph-identified sibling projections in one launch, retaining each K accumulation chain.
/// Dense weight/layout checks keep sparse, vendor and narrow-layout projections on their own paths.
pub(super) fn record_pair(
    scope: &Scope,
    first: &TensorNode,
    second: &TensorNode,
    arena: &super::ActivationArena,
    (weights, profile): (&ProgramWeights, &DeviceProfile),
) -> Result<bool, DeviceError> {
    if !pairable(first, second) || !weights.expert_banks.is_empty() {
        return Ok(false);
    }
    let (Some(ProjectionWeight::Dense(weight_b)), Some(ProjectionWeight::Dense(weight_a))) = (
        weights.projections.get(&first.inputs[1]),
        weights.projections.get(&second.inputs[1]),
    ) else {
        return Ok(false);
    };
    let [rows, columns] = weight_b.shape();
    if rows == 0 || columns == 0 || weight_a.shape() != weight_b.shape() {
        return Ok(false);
    }
    let width = arena.lanes();
    if width == 0
        || delegate_dense(width, rows, columns)
        || narrow_dense_layout(&profile.architecture, (width, rows, columns))
    {
        return Ok(false);
    }
    let Some((input_slot, slot_b, slot_a)) =
        pair_slots(first, second, arena, width, (rows, columns))?
    else {
        return Ok(false);
    };
    // `pair_slots` checked both outputs hold `width` rows of `rows` and the input `columns`.
    let output_b = arena.buffer(slot_b).map_err(error)?;
    let output_a = arena.buffer(slot_a).map_err(error)?;
    let flat_input = arena.buffer(input_slot).map_err(error)?;
    weights
        .kernels
        .projection
        .dense_pair(
            &Recording::scope(scope),
            &PromptPlan {
                rows: width,
                outputs: rows,
                columns,
                tile: [PREFILL_LANES, PREFILL_GEMM_TILE_N],
            },
            (output_b, output_a),
            flat_input,
            (buffer(weight_b)?, buffer(weight_a)?),
        )
        .map_err(error)?;
    Ok(true)
}

/// Record the established split-K projection with its existing operands and reduction.
fn record_legacy_fp4(
    (contract, scope): (&Kernels, &Scope),
    weight: &ProjectionWeight,
    input: Buffer<'_, f32>,
    output: &Tensor<f32>,
    partials: &Partials,
    (width, columns): (usize, usize),
) -> Result<(), DeviceError> {
    if width != PREFILL_LANES {
        return Err(error("legacy FP4 split prefill requires width 32"));
    }
    let rows = output.size() / width;
    let partial = partials.buffer(rows).map_err(error)?;
    let windows = columns.div_ceil(PREFILL_GEMM_TILE_K);
    let k_tiles = i32::try_from(windows.div_ceil(PREFILL_SPLIT_K))
        .map_err(|_| error("prefill split window"))?;
    contract
        .projection
        .prompt_fp4_split(
            &Recording::scope(scope),
            &SplitGemmPlan {
                prompt: PromptPlan {
                    rows: width,
                    outputs: rows,
                    columns,
                    tile: [width, PREFILL_NVFP4_TILE_N],
                },
                k_tiles,
                splits: i32::try_from(PREFILL_SPLIT_K).map_err(|_| error("prefill split count"))?,
            },
            buffer(output)?,
            partial,
            input,
            weight.weight_ref()?,
        )
        .map_err(error)
}

/// Whether the vendor GEMM recorded this projection instead of the tile kernel.
fn records_vendor_gemm(
    target: (&Kernels, &Scope),
    delegated: Option<&mut Support>,
    input: &Tensor<f32>,
    weight: &Matrix<bf16>,
    output: &Tensor<f32>,
    (width, rows, columns): (usize, usize, usize),
) -> Result<bool, DeviceError> {
    let Some(delegated) = delegated else {
        return Ok(false);
    };
    delegated.record_dense(
        target,
        buffer(input)?,
        weight,
        buffer(output)?,
        (width, rows, columns),
    )
}

// Preserve F32 router inputs through a high BF16 term and a low residual term.
fn records_precise_sparse(
    (contract, scope): (&Kernels, &Scope),
    weights: &ProgramWeights,
    (out, input, weight): (Buffer<'_, f32>, Buffer<'_, f32>, Buffer<'_, Bf16>),
    columns: usize,
) -> Result<bool, DeviceError> {
    if weights.expert_banks.is_empty() {
        return Ok(false);
    }
    let rows = input.len() / columns;
    contract
        .projection
        .dense(
            &Recording::scope(scope),
            &DenseRowsPlan {
                kernel: DenseKernel::Native,
                rows,
                outputs: out.len() / rows,
                columns,
            },
            out,
            input,
            weight,
        )
        .map_err(error)?;
    Ok(true)
}

/// Record a graph-identified small dense projection on the qualified narrow prompt rung.
pub(super) fn record_narrow(
    scope: &Scope,
    node: &TensorNode,
    arena: &super::ActivationArena,
    (weights, profile): (&ProgramWeights, &DeviceProfile),
) -> Result<bool, DeviceError> {
    let Some(ProjectionWeight::Dense(weight)) = weights.projections.get(&node.inputs[1]) else {
        return Ok(false);
    };
    let [rows, columns] = weight.shape();
    if rows == 0 || columns == 0 || !weights.expert_banks.is_empty() {
        return Ok(false);
    }
    let width = arena.lanes();
    if !narrow_dense_layout(&profile.architecture, (width, rows, columns))
        || delegate_dense(width, rows, columns)
    {
        return Ok(false);
    }
    let input_slot = arena.slot(node.inputs[0]).map_err(error)?;
    let output_slot = arena.slot(node.outputs[0]).map_err(error)?;
    if input_slot == output_slot {
        return Ok(false);
    }
    let Some(expected) = width.checked_mul(rows) else {
        return Err(error("control projection extent overflow"));
    };
    if arena.get(node.inputs[0]).map_err(error)?.len()
        != width
            .checked_mul(columns)
            .ok_or_else(|| error("control input extent overflow"))?
        || arena.get(node.outputs[0]).map_err(error)?.len() != expected
    {
        return Ok(false);
    }
    records_narrow_dense(
        (&weights.kernels, scope),
        profile,
        (
            arena.buffer(output_slot).map_err(error)?,
            arena.buffer(input_slot).map_err(error)?,
            buffer(weight)?,
        ),
        (width, rows, columns),
    )
}

fn records_narrow_dense(
    (contract, scope): (&Kernels, &Scope),
    profile: &DeviceProfile,
    (out, input, weight): (Buffer<'_, f32>, Buffer<'_, f32>, Buffer<'_, Bf16>),
    shape: (usize, usize, usize),
) -> Result<bool, DeviceError> {
    #[cfg(test)]
    if DENSE_REFERENCE.get() {
        return Ok(false);
    }
    if !narrow_dense_layout(&profile.architecture, shape) {
        return Ok(false);
    }
    contract
        .projection
        .dense(
            &Recording::scope(scope),
            &DenseRowsPlan {
                kernel: DenseKernel::PromptNarrow {
                    tile: NARROW_DENSE_TILE,
                },
                rows: shape.0,
                outputs: shape.1,
                columns: shape.2,
            },
            out,
            input,
            weight,
        )
        .map_err(error)?;
    #[cfg(test)]
    NARROW_DENSE_CAPTURES.set(NARROW_DENSE_CAPTURES.get() + 1);
    Ok(true)
}

#[cfg(test)]
thread_local! {
    static DENSE_REFERENCE: Cell<bool> = const { Cell::new(false) };
    static NARROW_DENSE_CAPTURES: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub fn narrow_dense_captures() -> usize {
    NARROW_DENSE_CAPTURES.get()
}

/// Captures the original dense kernel for a numerical comparison on this test thread.
#[cfg(test)]
pub fn with_dense_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_REFERENCE.set(self.0);
        }
    }
    let restore = Restore(DENSE_REFERENCE.replace(true));
    let value = run();
    drop(restore);
    value
}

#[cfg(test)]
#[path = "../../tests/unit/resident_prefill_projection.rs"]
mod tests;

/// The input and output slots of a fused down projection, or `None` when the
/// last-consumer arena gave them one slot: the vendor pack needs two disjoint slots.
fn down_operands(
    arena: &super::ActivationArena,
    node: &TensorNode,
) -> Result<Option<(usize, usize)>, DeviceError> {
    let input = arena.slot(node.inputs[0]).map_err(error)?;
    let output = arena.slot(node.outputs[0]).map_err(error)?;
    Ok((input != output).then_some((input, output)))
}
