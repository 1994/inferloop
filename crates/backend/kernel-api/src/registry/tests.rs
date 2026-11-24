use super::*;
use infer_core::OpId;
use infer_ir::{
    BackendKind, BackendRequirements, CapabilityRequirements, CudaRequirements, DeviceBackend,
    NvidiaArchitecture, NvidiaCapabilities, Operation, SpeculationCapability,
};
use infer_spi::SourceLocation;
/// A device-free capability sample: the tests check selection, not a device.
fn capabilities() -> DeviceCapabilities {
    DeviceCapabilities {
        backend: DeviceBackend::Cuda(NvidiaCapabilities {
            architecture: NvidiaArchitecture {
                compute_major: 12,
                compute_minor: 0,
                multiprocessors: 1,
            },
            tensor_core_generation: None,
            warp_size: 32,
            graphs: false,
            tma: false,
            clusters: false,
            pinned_transfer: false,
            cuda_ipc: false,
            nvlink: false,
            gpu_direct: false,
        }),
        device: infer_core::DeviceId::ONE,
        compute_dtypes: vec![infer_ir::DType::F32],
        memory_bytes: 0,
        unified_memory: true,
        profiling: false,
        speculation: SpeculationCapability::default(),
    }
}

struct Provider;
impl KernelProvider for Provider {
    fn kernels(&self) -> Vec<KernelRegistration> {
        (1..=2)
            .map(|id| KernelRegistration {
                backend: BackendKind::Cuda,
                id: KernelId::new(id).unwrap(),
                operation: Operation::MatMul,
                precision: PrecisionPlan::f32(),
                requirements: CapabilityRequirements {
                    backend: (id == 1).then_some(BackendRequirements::Cuda(CudaRequirements {
                        tma: true,
                        ..Default::default()
                    })),
                    ..Default::default()
                },
                max_shape_elements: 100,
                workspace_bytes: 0,
                priority: 1,
                estimated_ns: id,
                source: SourceLocation {
                    crate_name: "test".into(),
                    file: "test.rs".into(),
                    function: "matmul".into(),
                },
            })
            .collect()
    }
}
#[test]
fn filters_capability_and_shape() {
    let mut registry = KernelRegistry::default();
    registry.register(&Provider).unwrap();
    let mut op = OperationIr {
        id: OpId::new(1).unwrap(),
        operation: Operation::MatMul,
        shape: vec![10],
        layer: None,
        requirements: CapabilityRequirements::default(),
    };
    assert_eq!(
        registry
            .select(&op, &PrecisionPlan::f32(), &capabilities(), 0)
            .unwrap()
            .id
            .get(),
        2
    );
    op.shape = vec![101];
    assert!(
        registry
            .select(&op, &PrecisionPlan::f32(), &capabilities(), 0)
            .is_err()
    );
    assert!(registry.register(&Provider).is_err());
}
