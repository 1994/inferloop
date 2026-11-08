use infer_ir::*;

#[cfg(test)]
mod fixtures {
    use super::*;

    pub fn metal_device() -> DeviceCapabilities {
        DeviceCapabilities {
            backend: DeviceBackend::Metal(MetalCapabilities { simd_width: 32 }),
            device: infer_core::DeviceId::new(1).unwrap(),
            compute_dtypes: vec![DType::F32],
            memory_bytes: 1024,
            unified_memory: true,
            profiling: true,
            speculation: SpeculationCapability::default(),
        }
    }
}

use fixtures::*;

#[test]
fn backend_discriminator_owns_its_specific_capabilities() {
    let mut metal = metal_device();
    metal.backend = DeviceBackend::Metal(MetalCapabilities { simd_width: 32 });
    assert_eq!(metal.backend_kind(), BackendKind::Metal);
    assert!(
        metal
            .require(&CapabilityRequirements {
                backend: Some(BackendRequirements::Cuda(CudaRequirements {
                    tma: true,
                    ..Default::default()
                })),
                ..Default::default()
            })
            .is_err()
    );
    let bytes = serde_json::to_vec(&metal).unwrap();
    assert_eq!(
        serde_json::from_slice::<DeviceCapabilities>(&bytes).unwrap(),
        metal
    );
    let mut wrong = serde_json::to_value(&metal).unwrap();
    wrong["backend"]["capabilities"]["architecture"] = serde_json::json!({"compute_major":12});
    assert!(serde_json::from_value::<DeviceCapabilities>(wrong).is_err());
    let mut legacy = serde_json::to_value(&metal).unwrap();
    legacy["nvidia"] = serde_json::json!({"compute_major":12});
    assert!(serde_json::from_value::<DeviceCapabilities>(legacy).is_err());
    assert!(serde_json::from_value::<DeviceBackend>(serde_json::json!({"kind":"cuda"})).is_err());
}
#[test]
fn cuda_requirements_use_only_cuda_owned_attributes() {
    let mut cuda = metal_device();
    cuda.backend = DeviceBackend::Cuda(NvidiaCapabilities {
        architecture: NvidiaArchitecture {
            compute_major: 12,
            compute_minor: 0,
            multiprocessors: 170,
        },
        tensor_core_generation: Some(5),
        warp_size: 32,
        graphs: true,
        tma: true,
        clusters: true,
        pinned_transfer: true,
        cuda_ipc: true,
        nvlink: false,
        gpu_direct: false,
    });
    assert_eq!(cuda.backend_kind(), BackendKind::Cuda);
    assert!(
        cuda.require(&CapabilityRequirements {
            backend: Some(BackendRequirements::Cuda(CudaRequirements {
                graphs: true,
                tma: true,
                clusters: true
            })),
            ..Default::default()
        })
        .is_ok()
    );
}

/// E1 deleted the CPU test backends, so no production type may name one any more: this is the
/// guard that they cannot come back through a deserializer.
#[test]
fn production_architectures_reject_cpu_and_test_targets() {
    for name in ["cpu", "host", "test_cpu"] {
        assert!(serde_json::from_value::<BackendKind>(serde_json::json!(name)).is_err());
        assert!(serde_json::from_value::<DeviceBackend>(serde_json::json!({"kind":name})).is_err());
        assert!(
            serde_json::from_value::<BackendRequirements>(serde_json::json!({"kind":name}))
                .is_err()
        );
    }
}

/// Media prefill needs an explicit declaration before admission lets it run under a draft.
#[test]
fn media_prefill_requires_explicit_speculation_support() {
    assert!(SpeculationCapability::default().allows_media_prefill());
    let drafting = SpeculationCapability {
        draft_depth: 2,
        greedy_only: true,
        media_prefill: false,
    };
    assert!(!drafting.allows_media_prefill());
    let combined = SpeculationCapability {
        media_prefill: true,
        ..drafting
    };
    assert!(combined.allows_media_prefill());
}
