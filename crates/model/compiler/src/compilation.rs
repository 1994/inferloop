//! Compile provider-owned dataflow at load time; topology belongs to the model.
use infer_core::{Error, ProgramId, Result};
use infer_ir::{
    CapabilityRequirements, CompiledOp, DataflowGraph, DeviceCapabilities, ExecutionIr,
    ExecutionProgram, ModelIr, OperationIr, PrecisionPlan, TensorNode,
};
use infer_kernel_api::KernelRegistry;

/// Byte width of one scratch element; dataflow tensors are always `DType::F32`.
const F32_BYTES: u64 = 4;

/// # Errors
/// Returns an invalid-input or unsupported error for an invalid model graph or unsupported model operations.
pub fn lower(
    model: &ModelIr,
    graph: DataflowGraph,
    precision: PrecisionPlan,
) -> Result<ExecutionIr> {
    if precision != PrecisionPlan::f32() {
        return Err(Error::unsupported(
            "typed lowering for this precision provider is not installed",
        ));
    }
    model.validate()?;
    graph.validate()?;
    let operations = graph
        .nodes
        .iter()
        .map(|node| operation_ir(&graph, node, &precision))
        .collect::<Result<Vec<_>>>()?;
    Ok(ExecutionIr {
        model: model.id,
        precision,
        operations,
        dataflow: graph,
    })
}

/// The dataflow owns operation geometry and identity; a caller can only add capability limits.
fn operation_ir(
    graph: &DataflowGraph,
    node: &TensorNode,
    precision: &PrecisionPlan,
) -> Result<OperationIr> {
    let tensor = graph
        .tensors
        .iter()
        .find(|t| t.id == node.outputs[0])
        .ok_or_else(|| Error::invariant("lowered output"))?;
    Ok(OperationIr {
        id: node.id,
        operation: node
            .op
            .operation(graph.logits.is_some_and(|id| node.outputs.contains(&id))),
        layer: node.layer,
        shape: tensor.shape.clone(),
        requirements: CapabilityRequirements {
            compute_dtypes: vec![precision.compute],
            ..Default::default()
        },
    })
}

/// # Errors
/// Returns an invalid-input, unsupported, or capacity error if the graph cannot be lowered to compatible kernels within the workspace budget.
pub fn compile(
    id: ProgramId,
    ir: ExecutionIr,
    registry: &KernelRegistry,
    caps: &DeviceCapabilities,
    workspace_limit: u64,
) -> Result<ExecutionProgram> {
    ir.dataflow.validate()?;
    let mut planned = ir.dataflow.clone();
    planned.plan_lifetimes()?;
    if planned != ir.dataflow || ir.operations.len() != ir.dataflow.nodes.len() {
        return Err(Error::invalid(
            "execution IR dependency/lifetime/kernel mapping mismatch",
        ));
    }
    for (op, node) in ir.operations.iter().zip(&ir.dataflow.nodes) {
        let canonical = operation_ir(&ir.dataflow, node, &ir.precision)?;
        if op.id != canonical.id
            || op.operation != canonical.operation
            || op.shape != canonical.shape
            || op.layer != canonical.layer
            || !op
                .requirements
                .compute_dtypes
                .contains(&ir.precision.compute)
        {
            return Err(Error::invalid(
                "execution IR operation metadata/compute requirements mismatch",
            ));
        }
    }
    let mut operations = Vec::with_capacity(ir.operations.len());
    let mut workspace_bytes = 0;
    for op in ir.operations {
        let kernel = registry.select(&op, &ir.precision, caps, workspace_limit)?;
        workspace_bytes = workspace_bytes.max(kernel.workspace_bytes);
        operations.push(CompiledOp {
            op,
            kernel: kernel.id,
        });
    }
    let scratch = u64::try_from(ir.dataflow.scratch_elements)
        .ok()
        .and_then(|n| n.checked_mul(F32_BYTES))
        .ok_or_else(|| Error::invalid("dataflow workspace overflow"))?;
    workspace_bytes = workspace_bytes.max(scratch);
    if workspace_bytes > workspace_limit {
        return Err(Error::new(
            infer_core::ErrorCode::Capacity,
            "dataflow scratch exceeds program workspace budget",
        ));
    }
    Ok(ExecutionProgram {
        backend: caps.backend_kind(),
        id,
        model: ir.model,
        precision: ir.precision,
        operations,
        workspace_bytes,
        dataflow: ir.dataflow,
    })
}

#[cfg(test)]
mod tests;
