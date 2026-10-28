use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::{ProviderMetadata, SPI_VERSION, WorkloadProvider};
use infer_workloads::{NativeWorkloads, WorkloadRegistry};
use std::collections::BTreeMap;

mod support;

use infer_core::*;
use infer_ir::*;
struct CustomEmbedding;
impl WorkloadProvider for CustomEmbedding {
    fn identity(&self) -> &'static str {
        "custom-embedding-test-v1"
    }
    fn supports(&self, w: &Workload) -> bool {
        matches!(w, Workload::Embed { .. })
    }
    fn plan(&self, r: &CanonicalRequest, m: &ModelIr, p: ProgramId) -> Result<WorkloadPlan> {
        NativeWorkloads.plan(r, m, p)
    }
    fn postprocess(&self, _: &CanonicalRequest, o: &[ModelOutput]) -> Result<WorkloadOutput> {
        assert_ne!(o[0].hidden, [] as [Vec<f32>; 0]);
        Ok(WorkloadOutput::Embedding(vec![42.0].into()))
    }
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn providers() -> WorkloadRegistry {
    let mut registry = WorkloadRegistry::default();
    registry
        .register(
            ProviderMetadata {
                id: ProviderId::new(2).unwrap(),
                name: "custom".into(),
                spi_version: SPI_VERSION,
            },
            10,
            CustomEmbedding,
        )
        .unwrap();
    registry
        .register(
            ProviderMetadata {
                id: ProviderId::new(1).unwrap(),
                name: "native".into(),
                spi_version: SPI_VERSION,
            },
            0,
            NativeWorkloads,
        )
        .unwrap();
    registry
}
#[test]
fn registered_provider_runs_through_core_and_checkpoint_requires_same_registry() {
    let ir = support::model(ModelId::ONE);
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels).unwrap();
    let mut engine = Engine::new(
        support::ProtocolBackend::new(2, 2, &ir).unwrap(),
        ir.clone(),
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )
    .unwrap()
    .with_workloads(providers())
    .unwrap();
    let request = CanonicalRequest {
        id: RequestId::new(1).unwrap(),
        model: ModelId::ONE,
        session: None,
        input: RequestInput::Sequence {
            tokens: vec![1, 2].into(),
            media: vec![],
        },
        workload: Workload::Embed {
            pooling: Pooling::Mean,
            dimensions: None,
            normalize: false,
        },
        qos: Qos::default(),
        sampling: Sampling::default(),
        prefix_boundary: None,
        extensions: BTreeMap::new(),
    };
    engine.submit(request).unwrap();
    let snapshot = engine.snapshot().unwrap();
    assert!(
        Engine::restore(
            support::ProtocolBackend::new(2, 2, &ir).unwrap(),
            &kernels,
            snapshot.clone()
        )
        .is_err()
    );
    let mut restored = Engine::restore_with_workloads(
        support::ProtocolBackend::new(2, 2, &ir).unwrap(),
        &kernels,
        snapshot,
        providers(),
    )
    .unwrap();
    for now in 0..20 {
        if restored.is_idle() {
            break;
        }
        restored.tick(now).unwrap();
    }
    assert_eq!(
        restored
            .request(RequestId::new(1).unwrap())
            .unwrap()
            .completed
            .as_ref()
            .unwrap()
            .output,
        Some(WorkloadOutput::Embedding(vec![42.0].into()))
    );
    assert_eq!(restored.inspect().state.allocated_pages, 0);
}
