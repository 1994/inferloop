use super::*;
use crate::stages::output::OutputShape;
use infer_core::ProgramId;
use infer_ir::{CanonicalRequest, ModelIr, ModelOutput, Workload, WorkloadOutput, WorkloadPlan};
use std::thread::sleep;
use std::{time::Duration, time::Instant};

struct GatedProjection {
    entered: mpsc::Sender<()>,
    gate: Mutex<mpsc::Receiver<()>>,
}
impl WorkloadProvider for GatedProjection {
    fn identity(&self) -> &'static str {
        "gated-output-test"
    }
    fn supports(&self, workload: &Workload) -> bool {
        infer_workloads::NativeWorkloads.supports(workload)
    }
    fn plan(
        &self,
        request: &CanonicalRequest,
        model: &ModelIr,
        program: ProgramId,
    ) -> Result<WorkloadPlan> {
        infer_workloads::NativeWorkloads.plan(request, model, program)
    }
    fn postprocess(&self, _: &CanonicalRequest, _: &[ModelOutput]) -> Result<WorkloadOutput> {
        self.entered
            .send(())
            .map_err(|error| Error::invariant(error.to_string()))?;
        self.gate
            .lock()
            .map_err(|error| Error::invariant(error.to_string()))?
            .recv()
            .map_err(|error| Error::invariant(error.to_string()))?;
        Ok(WorkloadOutput::Embedding(vec![1.0, 2.0].into()))
    }
}
fn job(id: u64, projection: bool) -> Result<OutputJob> {
    let request = serde_json::from_value(serde_json::json!({
        "id":id,"model":1,"input":{"Sequence":{"tokens":[1]}},
        "workload":{"Generate":{"max_new_tokens":1}}
    }))
    .map_err(|error| Error::invalid(error.to_string()))?;
    Ok(OutputJob {
        constraint: None,
        generated: Vec::new().into(),
        request: Arc::new(request),
        output: ModelOutput {
            next_token: None,
            logits: vec![0.0, 2.0, 1.0],
            hidden: vec![vec![1.0, 2.0]],

            tokens: Vec::new(),
        },
        shape: OutputShape {
            logits: 3,
            rows: 1,
            width: 2,
        },
        sample: (!projection).then_some(0),
        project: projection.then(Vec::new),
    })
}
fn deadline(limit: Instant) -> Result<()> {
    if Instant::now() >= limit {
        return Err(Error::invariant("CPU output acknowledgement timed out"));
    }
    sleep(Duration::from_micros(100));
    Ok(())
}
#[test]
fn one_batch_has_independent_completions_and_abandoned_tickets_retain_cpu_credit() -> Result<()> {
    let (release, gate) = mpsc::channel();
    let (entered, start) = mpsc::channel();
    let provider: Arc<dyn WorkloadProvider + Send + Sync> = Arc::new(GatedProjection {
        entered,
        gate: Mutex::new(gate),
    });
    let worker = OutputWorker::new(&provider, 2)?;
    let mut ticket = worker.submit(vec![job(1, true)?, job(2, false)?], worker.reserve()?)?;
    start
        .recv_timeout(Duration::from_secs(3))
        .map_err(|error| Error::invariant(error.to_string()))?;
    let limit = Instant::now() + Duration::from_secs(3);
    let sampled = loop {
        if let Some(reply) = ticket.poll(1) {
            break reply.result?;
        }
        deadline(limit)?;
    };
    let blocked = ticket.pending(0) && ticket.poll(0).is_none();
    drop(ticket);
    let other = worker.reserve()?;
    let retained = !worker.available();
    release
        .send(())
        .map_err(|error| Error::invariant(error.to_string()))?;
    while !worker.available() {
        deadline(limit)?;
    }
    drop(other);
    assert_eq!(sampled.tokens.as_slice(), &[1]);
    assert!(blocked && retained);
    Ok(())
}
