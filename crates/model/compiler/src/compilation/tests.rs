use super::*;
use infer_core::{DeviceId, ErrorCode, KernelId, ModelId, OpId, TensorId};
use infer_ir::{
    BackendKind, BackendRequirements, CudaRequirements, DType, DataflowGraph, DeviceBackend,
    NvidiaArchitecture, NvidiaCapabilities, Operation, SpeculationCapability, TensorNode, TensorOp,
    TensorSpec, TensorStorage,
};
use infer_spi::{KernelProvider, KernelRegistration, SourceLocation};

const OUTPUT_SHAPE: [usize; 2] = [2, 128];
const WORKSPACE_LIMIT: u64 = 4096;

fn ir() -> Result<ExecutionIr> {
    let input = TensorId::ONE;
    let output = TensorId::new(2)?;
    let mut dataflow = DataflowGraph {
        tensors: vec![
            TensorSpec {
                id: input,
                shape: OUTPUT_SHAPE.to_vec(),
                dtype: DType::F32,
                storage: TensorStorage::Weight {
                    slot: "input".into(),
                },
            },
            TensorSpec {
                id: output,
                shape: OUTPUT_SHAPE.to_vec(),
                dtype: DType::F32,
                storage: TensorStorage::Activation,
            },
        ],
        nodes: vec![TensorNode {
            id: OpId::ONE,
            layer: Some(0),
            op: TensorOp::Silu,
            inputs: vec![input],
            outputs: vec![output],
            states: vec![],
        }],
        hidden: Some(output),
        logits: Some(output),
        features: None,
        ..DataflowGraph::default()
    };
    dataflow.plan_lifetimes()?;
    Ok(ExecutionIr {
        model: ModelId::ONE,
        precision: PrecisionPlan::f32(),
        operations: vec![OperationIr {
            id: OpId::ONE,
            operation: Operation::Silu,
            shape: OUTPUT_SHAPE.to_vec(),
            layer: Some(0),
            requirements: CapabilityRequirements {
                compute_dtypes: vec![DType::F32],
                ..CapabilityRequirements::default()
            },
        }],
        dataflow,
    })
}

// A descriptor-only device sample; no backend or device execution enters these tests.
fn capabilities() -> DeviceCapabilities {
    DeviceCapabilities {
        backend: DeviceBackend::Cuda(NvidiaCapabilities {
            architecture: NvidiaArchitecture {
                compute_major: 9,
                compute_minor: 0,
                multiprocessors: 1,
            },
            tensor_core_generation: None,
            warp_size: 32,
            graphs: true,
            tma: false,
            clusters: false,
            pinned_transfer: false,
            cuda_ipc: false,
            nvlink: false,
            gpu_direct: false,
        }),
        device: DeviceId::ONE,
        compute_dtypes: vec![DType::F32, DType::Bf16],
        memory_bytes: WORKSPACE_LIMIT,
        unified_memory: false,
        profiling: false,
        speculation: SpeculationCapability::default(),
    }
}

struct Provider(usize);

impl KernelProvider for Provider {
    fn kernels(&self) -> Vec<KernelRegistration> {
        vec![KernelRegistration {
            backend: BackendKind::Cuda,
            id: KernelId::ONE,
            operation: Operation::Silu,
            precision: PrecisionPlan::f32(),
            requirements: CapabilityRequirements {
                compute_dtypes: vec![DType::F32],
                ..CapabilityRequirements::default()
            },
            max_shape_elements: self.0,
            workspace_bytes: 0,
            priority: 0,
            estimated_ns: 1,
            source: SourceLocation {
                crate_name: env!("CARGO_PKG_NAME").into(),
                file: file!().into(),
                function: "Provider::kernels".into(),
            },
        }]
    }
}

fn compile_with_limit(ir: ExecutionIr, elements: usize) -> Result<ExecutionProgram> {
    let mut registry = KernelRegistry::default();
    registry.register(&Provider(elements))?;
    compile(
        ProgramId::ONE,
        ir,
        &registry,
        &capabilities(),
        WORKSPACE_LIMIT,
    )
}

fn rejects(ir: ExecutionIr, elements: usize, code: ErrorCode) -> Result<()> {
    let error = compile_with_limit(ir, elements)
        .err()
        .ok_or_else(|| Error::invariant("compiler accepted an invalid execution IR"))?;
    assert_eq!(error.code, code);
    Ok(())
}

#[test]
fn canonical_metadata_compiles() -> Result<()> {
    let ir = ir()?;
    let expected = ir.operations[0].clone();
    let program = compile_with_limit(ir, 256)?;
    assert_eq!(program.operations[0].op, expected);
    assert_eq!(program.workspace_bytes, 1024);
    Ok(())
}

#[test]
fn forged_shape_cannot_bypass_kernel_limit() -> Result<()> {
    let mut forged = ir()?;
    // The real graph needs 256 elements; this kernel can only handle 128.
    rejects(forged.clone(), 128, ErrorCode::Unsupported)?;
    forged.operations[0].shape = vec![1];
    rejects(forged, 128, ErrorCode::InvalidInput)
}

#[test]
fn equal_element_count_does_not_allow_different_shape() -> Result<()> {
    let mut forged = ir()?;
    forged.operations[0].shape = vec![256];
    rejects(forged, 256, ErrorCode::InvalidInput)
}

#[test]
fn incorrect_layer_is_rejected() -> Result<()> {
    for layer in [None, Some(1)] {
        let mut forged = ir()?;
        forged.operations[0].layer = layer;
        rejects(forged, 256, ErrorCode::InvalidInput)?;
    }
    Ok(())
}

#[test]
fn compute_requirements_cannot_be_removed_or_replaced() -> Result<()> {
    for dtypes in [vec![], vec![DType::Bf16]] {
        let mut forged = ir()?;
        forged.operations[0].requirements.compute_dtypes = dtypes;
        rejects(forged, 256, ErrorCode::InvalidInput)?;
    }
    Ok(())
}

#[test]
fn additional_compute_and_backend_requirements_are_preserved() -> Result<()> {
    let mut constrained = ir()?;
    constrained.operations[0].requirements = CapabilityRequirements {
        // Order is immaterial; the caller may add supported compute requirements.
        compute_dtypes: vec![DType::Bf16, DType::F32],
        backend: Some(BackendRequirements::Cuda(CudaRequirements {
            graphs: true,
            ..CudaRequirements::default()
        })),
    };
    let expected = constrained.operations[0].requirements.clone();
    let program = compile_with_limit(constrained, 256)?;
    assert_eq!(program.operations[0].op.requirements, expected);
    Ok(())
}

#[test]
fn unsupported_additional_requirements_still_reject() -> Result<()> {
    let mut constrained = ir()?;
    constrained.operations[0]
        .requirements
        .compute_dtypes
        .push(DType::Fp8E4M3);
    rejects(constrained, 256, ErrorCode::Unsupported)?;
    let mut constrained = ir()?;
    constrained.operations[0].requirements.backend =
        Some(BackendRequirements::Cuda(CudaRequirements {
            tma: true,
            ..CudaRequirements::default()
        }));
    rejects(constrained, 256, ErrorCode::Unsupported)
}

#[test]
fn incorrect_operation_identity_and_kind_remain_rejected() -> Result<()> {
    let mut forged = ir()?;
    forged.operations[0].id = OpId::new(2)?;
    rejects(forged, 256, ErrorCode::InvalidInput)?;
    let mut forged = ir()?;
    forged.operations[0].operation = Operation::Sigmoid;
    rejects(forged, 256, ErrorCode::InvalidInput)
}
