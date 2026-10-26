use super::nvfp4_gemm::GateUpMode;
use super::verify_projection_pair::Workspace as VerifyPairWorkspace;
use super::{ProgramWeights, capture::error, prefill_gemm::gemm};
use super::{pool_gemv, small_gemm::kernels as verify_gemm};
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

use crate::tuning::prefill::narrow_dense_layout;
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
    mode: GateUpMode,
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
        Self::build(graph, weights, geometry, GateUpMode::Prompt)
    }

    pub(super) fn for_verify(
        graph: &infer_ir::DataflowGraph,
        weights: &ProgramWeights,
        geometry: (usize, &super::nvfp4_gemm::Workspace),
    ) -> Self {
        Self::build(graph, weights, geometry, GateUpMode::Verify)
    }

    fn build(
        graph: &infer_ir::DataflowGraph,
        weights: &ProgramWeights,
        geometry: (usize, &super::nvfp4_gemm::Workspace),
        mode: GateUpMode,
    ) -> Self {
        let mut swiglu = swiglu_plan(graph, weights);
        let fused = fused_mlp_plan(graph, weights, &swiglu, geometry, mode);
        #[cfg(test)]
        let fused = if mode == GateUpMode::Verify && !VERIFY_MLP_ENABLED.with(Cell::get) {
            BTreeMap::new()
        } else {
            fused
        };
        let fused_downs: BTreeSet<_> = fused.values().map(|&(.., down)| down).collect();
        // Verify only absorbs complete packed passes; other row counts retain their original
        // SiLU/multiply/quantizer sequence, including the one-row quantizer for partial tiles.
        if mode == GateUpMode::Verify {
            swiglu.retain(|down, _| fused_downs.contains(down));
        }
        let absorbed = swiglu
            .values()
            .flat_map(|&(silu, multiply, ..)| [silu, multiply])
            .chain(fused.iter().map(
                |(&first, &(gate, up, _))| {
                    if first == gate { up } else { gate }
                },
            ))
            .collect();
        Self {
            mode,
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
            record_fused_mlp(scope, projections, arena, weights, nvfp4, self.mode)?;
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
    mode: GateUpMode,
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
        {
            continue;
        }
        let ready = match mode {
            GateUpMode::Prompt => nvfp4.gate_up_swiglu_ready(rows, columns, hidden),
            GateUpMode::Verify => {
                let (Some(gw), Some(uw), Some(scale)) = (
                    weights.projections.get(&gate_node.inputs[1]),
                    weights.projections.get(&up_node.inputs[1]),
                    fp4_scale(gate_node),
                ) else {
                    continue;
                };
                // The prepared down codes must not cross another operation that can overwrite
                // shared quantization storage. Prompt planning keeps its existing behavior.
                down > gate.max(up)
                    && (gate.min(up)..down).all(|i| [gate, up, silu, multiply].contains(&i))
                    && nvfp4.verify_gate_up_ready(
                        [rows, columns, hidden],
                        (gw, uw),
                        f32::from_bits(scale),
                    )
            }
        };
        if ready {
            plan.insert(gate.min(up), (gate, up, down));
        }
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
    mode: GateUpMode,
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
        (weight(gate)?, weight(up)?),
        (scale(gate)?, scale(down)?),
        &input,
        (mode, width, columns, hidden),
    )?;
    #[cfg(test)]
    if mode == GateUpMode::Verify {
        VERIFY_MLP_CAPTURES.with(|count| count.set(count.get().saturating_add(1)));
    }
    Ok(())
}

/// Record the down projection of a [`fused_mlp_plan`] entry from its prepared codes.
fn record_fused_down(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut super::ActivationArena,
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
    node: &TensorNode,
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
            if records_narrow_dense(scope, profile, &input, w, &mut output, dims)? {
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

/// Select the existing prompt arithmetic or two-term verification arithmetic.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PairContext {
    Prompt,
    Verify,
}

#[cfg(test)]
thread_local! {
    static VERIFY_PAIRS_ENABLED: Cell<bool> = const { Cell::new(true) };
    static VERIFY_PAIR_CAPTURES: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(super) fn verify_pair_captures() -> usize {
    VERIFY_PAIR_CAPTURES.with(Cell::get)
}

#[cfg(test)]
pub(super) fn with_verify_pairs<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            VERIFY_PAIRS_ENABLED.with(|value| value.set(self.0));
        }
    }
    let restore = Restore(VERIFY_PAIRS_ENABLED.with(|value| value.replace(enabled)));
    let result = run();
    drop(restore);
    result
}

/// Record the graph's beta/alpha pair from one shared input.
///
/// The unsplit path preserves each output's original K accumulation chain. A
/// supplied pooled verify workspace retains the two-term decomposition and F32
/// accumulation while changing the order of the partial reduction.
///
/// Returns false for narrow prompt layouts, sparse experts, small F32 verify
/// projections, incompatible geometry, or outputs wider than one tile.
///
/// # Errors
/// Propagates arena layout and CUDA capture errors.
pub(super) fn record_pair(
    scope: &Scope,
    first: &TensorNode,
    second: &TensorNode,
    arena: &mut super::ActivationArena,
    (weights, profile): (&ProgramWeights, &DeviceProfile),
    context: PairContext,
    scratch: Option<&mut VerifyPairWorkspace>,
) -> Result<bool, DeviceError> {
    #[cfg(test)]
    if context == PairContext::Verify && !VERIFY_PAIRS_ENABLED.with(Cell::get) {
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
    let rows_b = usize::try_from(weight_b.shape()[0]).map_err(error)?;
    let rows_a = usize::try_from(weight_a.shape()[0]).map_err(error)?;
    let weight_columns = usize::try_from(weight_b.shape()[1]).map_err(error)?;
    if rows_b == 0
        || rows_b != rows_a
        || rows_b > PREFILL_GEMM_TILE_N
        || weight_columns == 0
        || weight_a.shape()[1] != weight_b.shape()[1]
    {
        return Ok(false);
    }
    let width = arena.lanes();
    let expected = rows_b * width;
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
        || (context == PairContext::Prompt
            && narrow_dense_layout(&profile.architecture, (width, rows_b, columns)))
        // These rows use the existing F32 reduction, not the two-term MMA chain.
        || (context == PairContext::Verify
            && pool_gemv::supports(width, rows_b, columns))
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
    let mut output_b = output_b.reshape(&[width, rows_b])?;
    let mut output_a = output_a.reshape(&[width, rows_a])?;
    let flat_input = arena.buffers[input_slot]
        .as_ref()
        .ok_or_else(|| error("paired projection input"))?;
    let input = flat_input.view(&[width, columns]).map_err(error)?;
    match context {
        PairContext::Prompt => {
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
        }
        PairContext::Verify => {
            record_verify_pair(
                scope,
                &input,
                (weight_b, weight_a),
                (&mut output_b, &mut output_a),
                columns,
                scratch,
            )?;
            #[cfg(test)]
            VERIFY_PAIR_CAPTURES.with(|count| count.set(count.get().saturating_add(1)));
        }
    }
    arena.buffers[slot_b] = Some(output_b.reshape(&[expected])?);
    arena.buffers[slot_a] = Some(output_a.reshape(&[expected])?);
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
    input: &TensorView<'_, f32>,
    weight: &Tensor<bf16>,
    output: &mut Tensor<f32>,
    shape: (usize, usize, usize),
) -> Result<bool, DeviceError> {
    #[cfg(test)]
    if DENSE_REFERENCE.get() {
        return Ok(false);
    }
    if !narrow_dense_layout(&profile.architecture, shape) {
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

fn record_verify_pair(
    scope: &Scope,
    input: &TensorView<'_, f32>,
    weights: (&Tensor<bf16>, &Tensor<bf16>),
    (output_b, output_a): (&mut Tensor<f32>, &mut Tensor<f32>),
    columns: usize,
    scratch: Option<&mut VerifyPairWorkspace>,
) -> Result<(), DeviceError> {
    if let Some(scratch) = scratch
        && scratch.record(scope, input, weights, (&mut *output_b, &mut *output_a))?
    {
        return Ok(());
    }
    scope.record(
        verify_gemm::dense_pair(
            output_b.partition(SMALL_GEMM_TILE),
            output_a.partition(SMALL_GEMM_TILE),
            input,
            weights.0,
            weights.1,
        )
        .generics(vec![columns.to_string()]),
    )?;
    Ok(())
}

#[cfg(test)]
thread_local! {
    static VERIFY_MLP_ENABLED: Cell<bool> = const { Cell::new(true) };
    static VERIFY_MLP_CAPTURES: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(super) fn verify_mlp_captures() -> usize {
    VERIFY_MLP_CAPTURES.with(Cell::get)
}

#[cfg(test)]
pub(super) fn with_verify_mlp_disabled<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            VERIFY_MLP_ENABLED.with(|enabled| enabled.set(self.0));
        }
    }
    let _restore = Restore(VERIFY_MLP_ENABLED.with(|enabled| enabled.replace(false)));
    run()
}

#[cfg(test)]
#[path = "../../tests/unit/resident_verify_mlp_checkpoint.rs"]
mod verify_mlp_tests;
