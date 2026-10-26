use super::nvfp4_gemm::Workspace as Nvfp4Workspace;
use super::{ProgramWeights, capture::error, prefill_gemm::gemm};
use crate::constants::PREFILL_GEMM_TILE_K;
use crate::constants::PREFILL_GEMM_TILE_N;
use crate::constants::PREFILL_LANES;
use crate::constants::PREFILL_NVFP4_TILE_N;
use crate::constants::PREFILL_SPLIT_K;
use crate::constants::SMALL_GEMM_TILE;
use crate::device::cublaslt::Support;
use crate::{device::DeviceProfile, mlp::ProjectionWeight};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_ir::TensorNode;
#[cfg(test)]
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::thread::LocalKey;

use crate::tuning::prefill::narrow_dense_layout;
const NARROW_DENSE_TILE: [usize; 2] = [PREFILL_LANES, 16];

/// Semantic Delta input role, established once from graph dataflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GdnRole {
    Beta,
    Alpha,
}

/// Capture context for GDN beta/alpha pairing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnPairContext {
    /// Ordinary verification lanes: the existing Dense path keeps its two-component
    /// BF16 decomposition, so the single-component pair declines this context.
    Verify,
    /// Prompt chunk: the N16 prompt-only narrow layout must remain excluded so the
    /// role-selected N16 kernel, not the pair kernel, handles eligible beta/alpha rows.
    Prompt,
}

#[cfg(test)]
thread_local! {
    static PAIR_VERIFY: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    static PAIR_PROMPT: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

#[cfg(test)]
const fn pair_cell(context: GdnPairContext) -> &'static LocalKey<Cell<(usize, usize)>> {
    match context {
        GdnPairContext::Verify => &PAIR_VERIFY,
        GdnPairContext::Prompt => &PAIR_PROMPT,
    }
}

#[cfg(test)]
pub fn reset_pair_counts() {
    PAIR_VERIFY.with(|cell| cell.set((0, 0)));
    PAIR_PROMPT.with(|cell| cell.set((0, 0)));
}

#[cfg(test)]
pub fn pair_counts(context: GdnPairContext) -> (usize, usize) {
    pair_cell(context).with(Cell::get)
}

#[cfg(test)]
fn note_pair_attempt(context: GdnPairContext) {
    pair_cell(context).with(|cell| {
        let (attempts, records) = cell.get();
        cell.set((attempts + 1, records));
    });
}

#[cfg(test)]
fn note_pair_record(context: GdnPairContext) {
    pair_cell(context).with(|cell| {
        let (attempts, records) = cell.get();
        cell.set((attempts, records + 1));
    });
}

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

/// Gate/up projection pairs and their logical hidden/reduction widths for `SwiGLU` tuning.
pub fn swiglu_tuning_pairs(
    graph: &infer_ir::DataflowGraph,
    weights: &ProgramWeights,
) -> Vec<(
    infer_core::TensorId,
    infer_core::TensorId,
    usize,
    usize,
    f32,
    f32,
)> {
    let scale = |weight| match weights.activation_quantization(weight) {
        Some(super::program::ActivationQuantization::Fp4(scale)) => Some(scale),
        _ => None,
    };
    swiglu_plan(graph, weights)
        .into_iter()
        .filter_map(|(down, (_, _, gate, up))| {
            let (
                Some(ProjectionWeight::Fp4(gate_weight, ..)),
                Some(ProjectionWeight::Fp4(up_weight, ..)),
            ) = (weights.projections.get(&gate), weights.projections.get(&up))
            else {
                return None;
            };
            if gate_weight.shape() != up_weight.shape() {
                return None;
            }
            let gate_node = graph
                .nodes
                .iter()
                .find(|node| node.outputs.contains(&gate))?;
            let down_node = graph.nodes.get(down)?;
            let input_scale = scale(gate_node.inputs[1])?;
            let down_scale = scale(down_node.inputs[1])?;
            Some((
                gate,
                up,
                usize::try_from(gate_weight.shape()[0]).ok()?,
                usize::try_from(gate_weight.shape()[1]).ok()? * 2,
                input_scale,
                down_scale,
            ))
        })
        .collect()
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
    pub(super) fn new(
        graph: &infer_ir::DataflowGraph,
        weights: &ProgramWeights,
        geometry: (usize, &super::nvfp4_gemm::Workspace),
    ) -> Self {
        let swiglu = swiglu_plan(graph, weights);
        let (rows, workspace) = geometry;
        let fused = fused_mlp_plan(graph, weights, &swiglu, |gate, up, columns, hidden| {
            workspace.gate_up_swiglu_ready(gate, up, rows, columns, hidden)
        });
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

    /// Nodes another node of the plan records; the capture skips them.
    pub(super) fn absorbs(&self, index: usize) -> bool {
        self.absorbed.contains(&index)
    }

    /// Record node `index` when the plan owns it. False leaves it to the ordinary dispatch.
    pub(super) fn record(
        &self,
        scope: &Scope,
        (index, nodes): (usize, &[TensorNode]),
        arena: &mut super::ActivationArena,
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
    ready: impl Fn(infer_core::TensorId, infer_core::TensorId, usize, usize) -> bool,
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
            || !ready(gate_node.inputs[1], up_node.inputs[1], columns, hidden)
        {
            continue;
        }
        plan.insert(gate.min(up), (gate, up, down));
    }
    plan
}

/// Projection nodes replaced by the prompt fusion at this width. Workspace preparation uses
/// the same graph/geometry checks as capture, before the quantization buffers are allocated.
pub(super) fn fused_projections(
    graph: &infer_ir::DataflowGraph,
    weights: &ProgramWeights,
    rows: usize,
) -> BTreeSet<usize> {
    let swiglu = swiglu_plan(graph, weights);
    fused_mlp_plan(graph, weights, &swiglu, |_, _, _, hidden| {
        Nvfp4Workspace::gate_up_swiglu_geometry(rows, hidden)
    })
    .into_values()
    .flat_map(|(gate, up, _)| [gate, up])
    .collect()
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
    let input = arena.get(gate.inputs[0]).map_err(error)?;
    let columns = input.size() / width;
    let input = input.view(&[width, columns]).map_err(error)?;
    let hidden = arena.get(gate.outputs[0]).map_err(error)?.size() / width;
    nvfp4.record_gate_up_swiglu(
        scope,
        (gate.inputs[1], up.inputs[1]),
        (weight(gate)?, weight(up)?),
        (scale(gate)?, scale(down)?),
        &input,
        (width, columns, hidden),
    )
}

/// Record the down projection of a [`fused_mlp_plan`] entry from its prepared codes.
fn record_fused_down(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut super::ActivationArena,
    weights: &ProgramWeights,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
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
    let columns = arena.get(node.inputs[0]).map_err(error)?.size() / width;
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("prefill output"))?;
    let flat = output.size();
    let mut output = output.reshape(&[width, flat / width])?;
    nvfp4.record_prequantized(scope, weight, scale, &mut output, columns)?;
    arena.buffers[slot] = Some(output.reshape(&[flat])?);
    Ok(())
}

/// Record a [`swiglu_plan`] down projection. False leaves nothing recorded.
fn record_swiglu(
    scope: &Scope,
    node: &TensorNode,
    (gate, up): (infer_core::TensorId, infer_core::TensorId),
    arena: &mut super::ActivationArena,
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
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("prefill output"))?;
    let flat = output.size();
    let mut output = output.reshape(&[width, flat / width])?;
    let gate = arena.get(gate).map_err(error)?;
    let columns = gate.size() / width;
    let gate = gate.view(&[width, columns]).map_err(error)?;
    let up = arena
        .get(up)
        .map_err(error)?
        .view(&[width, columns])
        .map_err(error)?;
    nvfp4.record_swiglu_input(scope, weight, scale, &gate, &up, &mut output, columns)?;
    drop((gate, up));
    arena.buffers[slot] = Some(output.reshape(&[flat])?);
    Ok(true)
}

/// Zero-copy prompt GEMM:/// Zero-copy prompt GEMM: the batched input slot holds all 32 rows and the
/// output slot is reshaped to `[32, rows]` in place; no packing kernels remain.
/// Fp4 weights split the K loop across `PREFILL_SPLIT_K` CTAs per output tile:
/// the prompt GEMM is latency-bound at one wave, so wider grids hide DRAM latency.
pub(super) fn record(
    scope: &Scope,
    (node, role): (&TensorNode, Option<GdnRole>),
    arena: &mut super::ActivationArena,
    (weights, profile): (&ProgramWeights, &DeviceProfile),
    partials: &mut BTreeMap<usize, Tensor<f32>>,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
    delegated: Option<&mut Support>,
) -> Result<(), DeviceError> {
    if weights.constants.contains_key(&node.inputs[0]) {
        return Err(error("prefill32 constant linear input"));
    }
    let width = arena.lanes();
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("prefill output"))?;
    let flat = output.size();
    let mut output = output.reshape(&[width, flat / width])?;
    let flat_input = arena.get(node.inputs[0]).map_err(error)?;
    let input_pointer = flat_input.device_pointer().cu_deviceptr();
    let columns = flat_input.size() / width;
    let input = flat_input.view(&[width, columns]).map_err(error)?;
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("prefill projection"))?;
    if super::nvfp4_gemm::record_with_input_pointer(
        scope,
        nvfp4,
        weight,
        weights.activation_quantization(node.inputs[1]),
        &input,
        &mut output,
        columns,
        input_pointer,
    )? {
        arena.buffers[slot] = Some(output.reshape(&[flat])?);
        return Ok(());
    }
    let generics = vec![columns.to_string()];
    match weight {
        ProjectionWeight::Fp8Block(..) => {
            return Err(error("block FP8 requires the resident FP8 GEMM"));
        }
        ProjectionWeight::Dense(w) => {
            if records_precise_sparse(scope, weights, &input, w, &mut output, columns)? {
                arena.buffers[slot] = Some(output.reshape(&[flat])?);
                return Ok(());
            }
            // Wide projections are worth a vendor GEMM, square ones are not: `record_dense`
            // decides from the measured shape table.
            let rows = flat / width;
            let dims = (width, rows, columns);
            if records_vendor_gemm(scope, delegated, flat_input, w, &output, dims)? {
                arena.buffers[slot] = Some(output.reshape(&[flat])?);
                return Ok(());
            }
            if records_narrow_dense(scope, profile, role, &input, w, &mut output, dims)? {
                arena.buffers[slot] = Some(output.reshape(&[flat])?);
                return Ok(());
            }
            scope.record(
                gemm::dense(
                    (&mut output).partition([PREFILL_LANES, PREFILL_GEMM_TILE_N]),
                    &input,
                    w,
                )
                .generics(generics),
            )?;
        }
        ProjectionWeight::Fp8(w, scales) => {
            scope.record(
                gemm::fp8(
                    (&mut output).partition([PREFILL_LANES, PREFILL_GEMM_TILE_N]),
                    &input,
                    w,
                    scales,
                )
                .generics(generics),
            )?;
        }
        ProjectionWeight::Fp4(..) => {
            record_legacy_fp4(
                scope,
                weight,
                &input,
                &mut output,
                partials,
                (width, columns),
                generics,
            )?;
        }
    }
    arena.buffers[slot] = Some(output.reshape(&[flat])?);
    Ok(())
}

/// Record a GDN `in_proj_b` + `in_proj_a` pair as one launch: both are dense BF16
/// projections of the same input with equal output width. The caller establishes the
/// semantic pair from the graph (the beta and alpha inputs of one `Delta` node); this
/// function validates projection/layout compatibility and the capture context.
///
/// The pair kernel keeps each output's independent K accumulation order; only the
/// launch (and the shared `x` load) is fused. In prompt context the prompt-only N16
/// narrow layout stays excluded, so role-selected N16 wins; verification context is
/// not subject to that prompt-tile exclusion because its ordinary Dense path already
/// uses the same BF16 MMA contract.
///
/// Returns false when either node would take a different path (prompt N16 layout,
/// sparse experts, vendor GEMM, output width above one 64-column tile, or mismatched
/// input/weight geometry).
pub(super) fn record_pair(
    scope: &Scope,
    first: &TensorNode,
    second: &TensorNode,
    arena: &mut super::ActivationArena,
    (weights, profile): (&ProgramWeights, &DeviceProfile),
    context: GdnPairContext,
) -> Result<bool, DeviceError> {
    #[cfg(test)]
    note_pair_attempt(context);
    // Verify Dense uses the two-component BF16 decomposition in `small_gemm`
    // (high plus residual low), while `dense_pair` converts only the high component.
    // Keep verification on its existing F32 contract; prompt Dense already uses the
    // single-conversion kernel this pair matches.
    if context == GdnPairContext::Verify {
        return Ok(false);
    }
    if first.inputs.len() != 2
        || second.inputs.len() != 2
        || first.inputs[0] != second.inputs[0]
        || first.outputs.len() != 1
        || second.outputs.len() != 1
        || !weights.expert_banks.is_empty()
    {
        return Ok(false);
    }
    let (Some(ProjectionWeight::Dense(weight_b)), Some(ProjectionWeight::Dense(weight_a))) = (
        weights.projections.get(&first.inputs[1]),
        weights.projections.get(&second.inputs[1]),
    ) else {
        return Ok(false);
    };
    if weight_b.shape() != weight_a.shape() || weight_b.shape().len() != 2 {
        return Ok(false);
    }
    let rows = usize::try_from(weight_b.shape()[0]).map_err(error)?;
    let weight_columns = usize::try_from(weight_b.shape()[1]).map_err(error)?;
    if rows == 0
        || rows > PREFILL_GEMM_TILE_N
        || weight_columns == 0
        || weight_a.shape()[1] != weight_b.shape()[1]
    {
        return Ok(false);
    }
    let width = arena.lanes();
    let expected = rows * width;
    let input_slot = arena.slot(first.inputs[0]).map_err(error)?;
    let input_elements = arena.buffers[input_slot]
        .as_ref()
        .ok_or_else(|| error("paired projection input"))?
        .size();
    let columns = input_elements / width;
    if columns * width != input_elements
        || columns != weight_columns
        || arena.get(first.outputs[0]).map_err(error)?.size() != expected
        || arena.get(second.outputs[0]).map_err(error)?.size() != expected
        || (context == GdnPairContext::Prompt
            && narrow_dense_layout(&profile.architecture, rows, columns))
    {
        return Ok(false);
    }
    let slot_b = arena.slot(first.outputs[0]).map_err(error)?;
    let slot_a = arena.slot(second.outputs[0]).map_err(error)?;
    if slot_b == slot_a {
        return Ok(false);
    }
    let output_b = arena.buffers[slot_b]
        .take()
        .ok_or_else(|| error("paired projection b output"))?;
    let output_a = arena.buffers[slot_a]
        .take()
        .ok_or_else(|| error("paired projection a output"))?;
    let mut output_b = output_b.reshape(&[width, rows])?;
    let mut output_a = output_a.reshape(&[width, rows])?;
    let flat_input = arena.buffers[input_slot]
        .as_ref()
        .ok_or_else(|| error("paired projection input"))?;
    let input = flat_input.view(&[width, columns]).map_err(error)?;
    scope.record(
        gemm::dense_pair(
            (&mut output_b).partition([PREFILL_LANES, PREFILL_GEMM_TILE_N]),
            (&mut output_a).partition([PREFILL_LANES, PREFILL_GEMM_TILE_N]),
            &input,
            weight_b,
            weight_a,
        )
        .generics(vec![columns.to_string()]),
    )?;
    arena.buffers[slot_b] = Some(output_b.reshape(&[expected])?);
    arena.buffers[slot_a] = Some(output_a.reshape(&[expected])?);
    #[cfg(test)]
    note_pair_record(context);
    Ok(true)
}

/// Record the established split-K projection with its existing operands and reduction.
fn record_legacy_fp4(
    scope: &Scope,
    weight: &ProjectionWeight,
    input: &TensorView<'_, f32>,
    output: &mut Tensor<f32>,
    partials: &mut BTreeMap<usize, Tensor<f32>>,
    (width, columns): (usize, usize),
    generics: Vec<String>,
) -> Result<(), DeviceError> {
    let ProjectionWeight::Fp4(w, scales, global) = weight else {
        return Err(error("legacy FP4 prompt weight"));
    };
    if width != PREFILL_LANES {
        return Err(error("legacy FP4 split prefill requires width 32"));
    }
    let rows = output.size() / width;
    let partial = partials
        .get_mut(&rows)
        .ok_or_else(|| error("prefill split partials"))?;
    let windows = columns.div_ceil(PREFILL_GEMM_TILE_K);
    let k_tiles = i32::try_from(windows.div_ceil(PREFILL_SPLIT_K))
        .map_err(|_| error("prefill split window"))?;
    scope.record(
        gemm::nvfp4_split(
            partial.partition([width, PREFILL_NVFP4_TILE_N]),
            input,
            w,
            scales,
            k_tiles,
        )
        .generics(generics),
    )?;
    scope.record(gemm::reduce_split(
        output.partition([width, PREFILL_NVFP4_TILE_N]),
        &*partial,
        global.recip(),
        i32::try_from(PREFILL_SPLIT_K).map_err(|_| error("prefill split count"))?,
    ))?;
    Ok(())
}

/// Whether the vendor GEMM recorded this projection instead of the tile kernel.
fn records_vendor_gemm(
    scope: &Scope,
    delegated: Option<&mut Support>,
    input: &Tensor<f32>,
    weight: &Tensor<bf16>,
    output: &Tensor<f32>,
    (width, rows, columns): (usize, usize, usize),
) -> Result<bool, DeviceError> {
    let Some(delegated) = delegated else {
        return Ok(false);
    };
    delegated.record_dense(scope, input, weight, output, (width, rows, columns))
}

// Preserve F32 router inputs through a high BF16 term and a low residual term.
fn records_precise_sparse(
    scope: &Scope,
    weights: &ProgramWeights,
    input: &TensorView<'_, f32>,
    weight: &Tensor<bf16>,
    output: &mut Tensor<f32>,
    columns: usize,
) -> Result<bool, DeviceError> {
    if weights.expert_banks.is_empty() {
        return Ok(false);
    }
    scope.record(
        super::small_gemm::kernels::dense(output.partition(SMALL_GEMM_TILE), input, weight)
            .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
    )?;
    Ok(true)
}

fn records_narrow_dense(
    scope: &Scope,
    profile: &DeviceProfile,
    role: Option<GdnRole>,
    input: &TensorView<'_, f32>,
    weight: &Tensor<bf16>,
    output: &mut Tensor<f32>,
    shape: (usize, usize, usize),
) -> Result<bool, DeviceError> {
    #[cfg(test)]
    if DENSE_REFERENCE.get() {
        return Ok(false);
    }
    if role.is_none() || !narrow_dense_layout(&profile.architecture, shape.1, shape.2) {
        return Ok(false);
    }
    scope.record(
        gemm::dense_n16(output.partition(NARROW_DENSE_TILE), input, weight)
            .generics(vec![shape.2.to_string()]),
    )?;
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
