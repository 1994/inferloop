use infer_core::{Error, FinishReason, ModelId, ProgramId, RequestId, Result};
use infer_frontdoor::RuntimeHandle;
use infer_ir::{
    CanonicalRequest, ModelIr, ModelOutput, PrecisionPlan, Workload, WorkloadOutput, WorkloadPlan,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, EngineOutput, RuntimeConfig};
use infer_spi::WorkloadProvider;
use infer_test_support::fixtures as support;
use infer_workloads::NativeWorkloads;
use std::{sync::Arc, sync::Mutex, sync::mpsc, time::Duration};
use tokio::sync::mpsc::Receiver;
use tokio::time::sleep;
use tokio::time::timeout;

#[derive(Clone)]
struct ProjectionGate {
    gate: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
    entered: mpsc::Sender<()>,
}
impl WorkloadProvider for ProjectionGate {
    fn fork(&self) -> Option<Box<dyn WorkloadProvider + Send + Sync>> {
        Some(Box::new(self.clone()))
    }
    fn identity(&self) -> &'static str {
        "gated-projection-v1"
    }
    fn supports(&self, workload: &Workload) -> bool {
        NativeWorkloads.supports(workload)
    }
    fn plan(
        &self,
        request: &CanonicalRequest,
        model: &ModelIr,
        program: ProgramId,
    ) -> Result<WorkloadPlan> {
        NativeWorkloads.plan(request, model, program)
    }
    fn postprocess(
        &self,
        request: &CanonicalRequest,
        outputs: &[ModelOutput],
    ) -> Result<WorkloadOutput> {
        let gate = self
            .gate
            .lock()
            .map_err(|_| Error::invariant("projection gate poisoned"))?
            .take();
        if let Some(gate) = gate {
            self.entered
                .send(())
                .map_err(|e| Error::invariant(e.to_string()))?;
            gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
        }
        NativeWorkloads.postprocess(request, outputs)
    }
}
fn request(id: u64, embed: bool) -> Result<CanonicalRequest> {
    let workload = if embed {
        serde_json::json!({"Embed":{"pooling":"Mean","dimensions":null,"normalize":false}})
    } else {
        serde_json::json!({"Generate":{"max_new_tokens":5}})
    };
    serde_json::from_value(serde_json::json!({"id":id,"model":1,"input":{"Sequence":{"tokens":[1,2,3]}},"workload":workload,"sampling":{"temperature":0.7,"top_k":3,"seed":7,"eos_token":null}})).map_err(|e| Error::invalid(e.to_string()))
}
async fn terminal(stream: &mut Receiver<Result<EngineOutput>>) -> Result<FinishReason> {
    loop {
        let event = timeout(Duration::from_secs(3), stream.recv())
            .await
            .map_err(|e| Error::invariant(e.to_string()))?
            .ok_or_else(|| Error::invariant("stream closed before terminal"))??;
        if let EngineOutput::Finished(done) = event {
            return Ok(done.reason);
        }
    }
}
#[tokio::test]
async fn stalled_projection_does_not_block_sampling_control_or_terminal_delivery() -> Result<()> {
    for cancel in [true, false] {
        let ir = support::model(ModelId::ONE);
        let mut registry = KernelRegistry::default();
        registry.register(&support::DeclaredKernels)?;
        let (release, gate) = mpsc::channel();
        let (entered, start) = mpsc::channel();
        let provider = ProjectionGate {
            gate: Arc::new(Mutex::new(Some(gate))),
            entered,
        };
        let handle = RuntimeHandle::start(
            Engine::new(
                support::ProtocolBackend::new(16, 8, &ir)?,
                ir,
                PrecisionPlan::f32(),
                &registry,
                RuntimeConfig {
                    // The cancel branch has to win this deadline rather than race it: at 50 ms a
                    // loaded runner reports the output-stage timeout as the finish reason instead
                    // of the cancellation the scene is about. The other branch still needs the
                    // timeout to fire well inside the 3 s the terminal wait allows.
                    output_timeout_us: 500_000,
                    ..RuntimeConfig::default()
                },
            )?
            .with_workloads(provider)?,
        )?;
        let mut first = handle.submit(request(1, true)?).await?;
        let mut started = false;
        for _ in 0..1000 {
            if start.try_recv().is_ok() {
                started = true;
                break;
            }
            sleep(Duration::from_millis(1)).await;
        }
        assert!(started);
        let inspection = timeout(Duration::from_millis(25), handle.inspect())
            .await
            .map_err(|e| Error::invariant(e.to_string()))??;
        assert_eq!(inspection.output_waiters, 1);
        assert!(inspection.inflight_step.is_none());
        let mut second = handle.submit(request(2, false)?).await?;
        assert_eq!(terminal(&mut second).await?, FinishReason::Length);
        if cancel {
            handle.cancel(RequestId::ONE).await?;
        }
        let reason = terminal(&mut first).await?;
        if cancel {
            assert_eq!(reason, FinishReason::Cancelled);
        } else {
            assert!(matches!(reason, FinishReason::Failed(_)));
        }
        let inspection = handle.inspect().await?;
        assert_eq!(inspection.output_waiters, 1);
        assert!(inspection.resource_release_pending);
        release
            .send(())
            .map_err(|e| Error::invariant(e.to_string()))?;
        handle.shutdown().await?;
    }
    Ok(())
}
