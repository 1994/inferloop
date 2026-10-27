//! Allocate operands before planning; only captured standalone projections enter search.
use super::{Choice, Quantized, ResidentGemmWeightKey, WeightKey, Workspace};
use crate::constants::{FP8_BLOCK_COLUMNS, NVFP4_GROUP_SIZE, PREFILL_LANES};
use crate::device::{CudaDevice, device_error};
use crate::mlp::ProjectionWeight;
use crate::resident::{ProgramWeights, prefill_projection::MlpPlan};
use crate::tuning::{
    fp8_gemm::{self, prefill_choice, probe_inputs::Scope as ProbeInputsScope, select},
    projection_budget::{ProjectionKind, ProjectionShape},
    resident_gemm::{self, swiglu},
};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};

struct SwigluProjection {
    gate: TensorId,
    up: TensorId,
    down: TensorId,
    shape: ProjectionShape,
}
struct Projection {
    weight: TensorId,
    output: TensorId,
    shape: ProjectionShape,
    kind: ProjectionKind,
}
fn elements(graph: &DataflowGraph, id: TensorId) -> Result<usize> {
    graph
        .tensors
        .iter()
        .find(|t| t.id == id)
        .ok_or_else(|| Error::invariant("projection tensor"))?
        .elements()
}
fn quantized(node: &TensorNode, weights: &ProgramWeights) -> bool {
    node.op == TensorOp::Linear
        && (weights.input_scales.contains_key(&node.inputs[1])
            || weights.fp8_inputs.contains(&node.inputs[1]))
}
impl Workspace {
    pub(super) fn extend(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<()> {
        let _probe_inputs = ProbeInputsScope::new(device)?;
        self.allocate_inputs(device, graph, weights, widths)?;
        let projections = self.live_projections(graph, weights, widths)?;
        let fused = self.live_swiglu(graph, weights, widths)?;
        for projection in &projections {
            let trials = match projection.kind {
                ProjectionKind::Fp4 => resident_gemm::search_trials(),
                ProjectionKind::Fp8 => fp8_gemm::search_trials(),
                ProjectionKind::Fp4Lt | ProjectionKind::Swiglu => {
                    return Err(Error::invariant("standalone projection kind"));
                }
            };
            weights
                .resident_tuning
                .plan_projection(projection.kind, projection.shape, trials)?;
        }
        for projection in &fused {
            weights.resident_tuning.plan_projection(
                ProjectionKind::Swiglu,
                projection.shape,
                swiglu::search_trials(projection.shape),
            )?;
        }
        for projection in projections {
            let [rows, _, columns] = projection.shape;
            if projection.kind == ProjectionKind::Fp8 {
                self.prepare_channel(
                    device,
                    graph,
                    weights,
                    projection.weight,
                    projection.output,
                    columns,
                    &[rows],
                )?;
            } else {
                self.prepare_fp4(device, weights, &projection)?;
            }
        }
        for projection in &fused {
            self.prepare_swiglu(device, weights, projection)?;
        }
        weights.resident_tuning.report_projection_work()
    }

    fn live_swiglu(
        &self,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<Vec<SwigluProjection>> {
        let mut fused = Vec::new();
        for &rows in widths.iter().filter(|rows| **rows >= PREFILL_LANES) {
            let plan = MlpPlan::new(graph, weights, (rows, self));
            for (gate, up, down) in plan.fused_projections() {
                let gate = &graph.nodes[gate];
                fused.push(SwigluProjection {
                    gate: gate.inputs[1],
                    up: graph.nodes[up].inputs[1],
                    down: graph.nodes[down].inputs[1],
                    shape: [
                        rows,
                        elements(graph, gate.outputs[0])?,
                        elements(graph, gate.inputs[0])?,
                    ],
                });
            }
        }
        Ok(fused)
    }

    fn prepare_swiglu(
        &mut self,
        device: &CudaDevice,
        weights: &ProgramWeights,
        projection: &SwigluProjection,
    ) -> Result<()> {
        use crate::resident::program::ActivationQuantization;
        let weight = |id| {
            weights
                .projections
                .get(&id)
                .ok_or_else(|| Error::invariant("SwiGLU projection weight"))
        };
        let scale = |id| match weights.activation_quantization(id) {
            Some(ActivationQuantization::Fp4(scale)) => Ok(scale),
            _ => Err(Error::invariant("SwiGLU activation scale")),
        };
        let pair = (weight(projection.gate)?, weight(projection.up)?);
        let scales = (scale(projection.gate)?, scale(projection.down)?);
        if scales.0.to_bits() != scale(projection.up)?.to_bits() {
            return Err(Error::invariant("SwiGLU shared activation scale"));
        }
        let binding = swiglu::Binding::new(pair, scales, projection.shape)?;
        if self.swiglu_strategies.contains_key(&binding) {
            return Ok(());
        }
        let config = swiglu::select(
            device,
            pair,
            scales,
            projection.shape,
            weights.autotune,
            &weights.resident_tuning,
        );
        self.swiglu_strategies.insert(binding, config);
        Ok(())
    }

    fn live_projections(
        &self,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<Vec<Projection>> {
        let mut projections = Vec::new();
        for &rows in widths {
            if rows == 0 {
                continue;
            }
            let mlp = (rows >= PREFILL_LANES).then(|| MlpPlan::new(graph, weights, (rows, self)));
            for (index, node) in graph.nodes.iter().enumerate() {
                if !quantized(node, weights) {
                    continue;
                }
                let kind = match weights.projections.get(&node.inputs[1]) {
                    Some(ProjectionWeight::Fp4(..))
                        if !mlp.as_ref().is_some_and(|p| p.fuses_projection(index)) =>
                    {
                        ProjectionKind::Fp4
                    }
                    Some(ProjectionWeight::Fp8(..))
                        if !(rows >= PREFILL_LANES && Some(node.outputs[0]) == graph.logits) =>
                    {
                        ProjectionKind::Fp8
                    }
                    _ => continue,
                };
                projections.push(Projection {
                    weight: node.inputs[1],
                    output: node.outputs[0],
                    kind,
                    shape: [
                        rows,
                        elements(graph, node.outputs[0])?,
                        elements(graph, node.inputs[0])?,
                    ],
                });
            }
        }
        Ok(projections)
    }

    fn allocate_inputs(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<()> {
        for node in &graph.nodes {
            if !quantized(node, weights) {
                continue;
            }
            let columns = elements(graph, node.inputs[0])?;
            if weights.fp8_inputs.contains(&node.inputs[1]) {
                self.allocate_fp8(
                    device,
                    weights.projections.get(&node.inputs[1]),
                    columns,
                    widths,
                )?;
                continue;
            }
            for &rows in widths {
                if rows == 0 || self.buffers.contains_key(&(rows, columns)) {
                    continue;
                }
                let buffers: Quantized = (
                    api::zeros::<f4e2m1fnx2>(&[rows, columns / 2])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                    api::zeros::<f8e4m3fn>(&[rows, columns / NVFP4_GROUP_SIZE])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                );
                self.buffers.insert((rows, columns), buffers);
            }
        }
        Ok(())
    }

    fn prepare_fp4(
        &mut self,
        device: &CudaDevice,
        weights: &ProgramWeights,
        projection: &Projection,
    ) -> Result<()> {
        let [rows, outputs, columns] = projection.shape;
        let weight = weights
            .projections
            .get(&projection.weight)
            .ok_or_else(|| Error::invariant("FP4 preparation weight"))?;
        let scale = *weights
            .input_scales
            .get(&projection.weight)
            .ok_or_else(|| Error::invariant("FP4 preparation scale"))?;
        let key = (
            ResidentGemmWeightKey::new(weight, scale)?,
            rows,
            outputs,
            columns,
        );
        self.strategies.entry(key).or_insert_with(|| {
            resident_gemm::select(
                device,
                weight,
                scale,
                projection.shape,
                weights.autotune,
                &weights.resident_tuning,
            )
        });
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
        weight_id: TensorId,
        output_id: TensorId,
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
                if lt_prefill.is_some() {
                    weights
                        .resident_tuning
                        .complete_projection(ProjectionKind::Fp8, [rows, outputs, columns])?;
                }
            }
        }
        Ok(())
    }
}
