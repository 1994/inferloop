use crate::measure::{Measurement, measure};
use infer_core::{DecisionId, Error, ProgramId, RequestId, Result, StateId, StepId};
use infer_gpu_api::{BatchArena, BatchLease, Consumer, Producer, RingBuffer};
use infer_ir::{
    BackendKind, CostEstimate, CostQuery, ExecutionInput, ExecutionRole, ExecutionTask, ReadyWork,
    ResourceSnapshot, SchedulerConfig,
};
use infer_runtime::{CpuStepPool as StepPool, ReadyWorkBuffer as ReadyWindow};
use infer_scheduler::{
    CostAwarePolicy, FallbackCosts, PackingWorkspace, QueueRequest, QueueTiming, RequestQueue,
    ValidationScratch, validate_decision_reusing,
};
use infer_spi::{DecisionStorage, PlanningContext, SchedulingPolicy};
use serde::Serialize;
use std::{hint::black_box, sync::Arc, time::Instant};
#[derive(Serialize)]
pub struct Report {
    pub requests: usize,
    pub batch: usize,
    pub candidates: usize,
    pub context: usize,
    pub measurement: Measurement,
    pub stages: Vec<Stage>,
}
#[derive(Serialize)]
pub struct Stage {
    name: &'static str,
    p50_ns: u64,
    p99_ns: u64,
}
struct Cycle {
    samples: Vec<[u64; 5]>,
    queue: RequestQueue,
    rows: Vec<ReadyWork>,
    ready: ReadyWindow,
    candidates: Vec<RequestId>,
    members: Vec<RequestId>,
    tasks: Vec<ExecutionTask>,
    workspace: PackingWorkspace,
    output: DecisionStorage,
    validation: ValidationScratch,
    steps: StepPool,
    batches: BatchArena<()>,
    submissions: Producer<BatchLease>,
    applied: Consumer<BatchLease>,
    completions: Producer<BatchLease>,
    fenced: Consumer<BatchLease>,
    resources: ResourceSnapshot,
    tenant_finish: [u64; 32],
    iteration: u64,
    limit: usize,
}
fn query(context: usize) -> CostQuery {
    CostQuery::from_unit(
        ProgramId::ONE,
        BackendKind::Metal,
        ExecutionRole::Decode,
        1,
        context,
        CostEstimate {
            gpu_us: 1,
            ..CostEstimate::default()
        },
    )
}
fn row(id: u64, context: usize) -> Result<ReadyWork> {
    Ok(ReadyWork {
        request: RequestId::new(id)?,
        state: StateId::new(id)?,
        program: ProgramId::ONE,
        role: ExecutionRole::Decode,
        remaining_tokens: 1,
        tenant: format!("tenant{}", id % 32),
        weight: 1,
        deadline_us: None,
        virtual_finish: 0,
        cost_per_token: CostEstimate {
            gpu_us: 1,
            ..CostEstimate::default()
        },
        cost_query: query(context),
        latency_deadline_us: None,
        remaining_latency_us: 1,
        remaining_completion_us: 1,
        last_service_us: 0,
    })
}
impl Cycle {
    fn new(requests: usize, batch: usize, context: usize) -> Result<Self> {
        let limit = requests.min(256);
        let batch = batch.min(requests);
        let mut queue = RequestQueue::new(requests)?;
        let mut rows = Vec::with_capacity(requests);
        for id in 1..=requests {
            let row = row(id as u64, context)?;
            queue.enqueue(QueueRequest {
                request: row.request,
                phase: ExecutionRole::Decode,
                tenant: Arc::from(row.tenant.as_str()),
                virtual_finish: 0,
                last_service_us: 0,
                latest_start_us: None,
                hard_deadline_us: None,
                wait_deadline_us: u64::MAX,
            })?;
            rows.push(row);
        }
        let mut workspace = PackingWorkspace::default();
        CostAwarePolicy.reserve_workspace(&mut workspace, limit, batch)?;
        let (submissions, applied) = RingBuffer::new(2);
        let (completions, fenced) = RingBuffer::new(2);
        Ok(Self {
            samples: Vec::with_capacity(1128),
            queue,
            rows,
            ready: ReadyWindow::new(limit, query(context))?,
            candidates: Vec::with_capacity(limit),
            members: Vec::with_capacity(batch),
            tasks: Vec::with_capacity(batch),
            workspace,
            output: DecisionStorage::new(limit, batch)?,
            validation: ValidationScratch::new(limit, batch)?,
            steps: StepPool::new(batch, ProgramId::ONE)?,
            batches: BatchArena::new(2, batch)?,
            submissions,
            applied,
            completions,
            fenced,
            resources: ResourceSnapshot {
                max_num_seqs: batch,
                max_num_batched_tokens: batch,
                gpu_budget_us: 10000,
                workspace_bytes: 1 << 20,
                free_state_pages: usize::MAX,
                free_logical_pages: usize::MAX,
                graphs: vec![],
                free_state_bytes: u64::MAX,
                transfer_budget_us: 10000,
                encoder_budget_us: 10000,
                scheduler: SchedulerConfig::default(),
            },
            tenant_finish: [0; 32],
            iteration: 0,
            limit,
        })
    }
    fn tick(&mut self) -> Result<()> {
        let start = Instant::now();
        self.iteration += 1;
        self.queue.candidates_into(
            &mut self.candidates,
            self.limit,
            self.iteration,
            20000,
            10000,
        )?;
        self.ready.count = 0;
        for id in &self.candidates {
            self.ready.write(&self.rows[row_index(*id)?])?;
        }
        for row in self.ready.as_mut_slice() {
            row.virtual_finish = self.tenant_finish[(row_index(row.request)? + 1) % 32];
        }
        let projected = elapsed(start);
        let mut decision = CostAwarePolicy.plan_into(
            PlanningContext {
                ready: self.ready.as_slice(),
                resources: &self.resources,
                now_us: self.iteration,
                decision: DecisionId::new(self.iteration)?,
                step: StepId::new(self.iteration)?,
                costs: &FallbackCosts,
            },
            &mut self.workspace,
            &mut self.output,
        )?;
        let planned = elapsed(start);
        validate_decision_reusing(
            &decision,
            self.ready.as_slice(),
            &self.resources,
            ProgramId::ONE,
            &FallbackCosts,
            &mut self.validation,
        )?;
        let validated = elapsed(start);
        let step = decision
            .step
            .as_mut()
            .ok_or_else(|| Error::invariant("CPU benchmark stalled"))?;
        self.members.clear();
        self.tasks.clear();
        for work in &step.work {
            self.members.push(work.request);
            self.tasks.push(ExecutionTask {
                plain_greedy: false,
                request: work.request,
                state: work.state,
                tokens: ExecutionInput::Decode {
                    position: self.rows[row_index(work.request)?]
                        .cost_query
                        .context_tokens
                        - 1,
                    token: 1,
                },
                sampling: Some(infer_ir::Sampling::default()),
                media: None,
            });
        }
        self.queue.dispatch(step.id, &self.members)?;
        let sealed = self.steps.seal(step)?;
        self.handoff(sealed)?;
        let submitted = elapsed(start);
        for id in &self.members {
            self.queue.complete_ready(
                *id,
                StepId::new(self.iteration)?,
                QueueTiming {
                    phase: ExecutionRole::Decode,
                    last_service_us: self.iteration,
                    latest_start_us: None,
                    wait_deadline_us: u64::MAX,
                },
            )?;
            let index = (row_index(*id)? + 1) % 32;
            self.tenant_finish[index] += 1;
            self.queue.update_tenant(
                &self.rows[row_index(*id)?].tenant,
                self.tenant_finish[index],
            );
        }
        self.output.reclaim(decision);
        let committed = elapsed(start);
        self.samples.push([
            projected,
            planned - projected,
            validated - planned,
            submitted - validated,
            committed - submitted,
        ]);
        Ok(())
    }
    fn handoff(&mut self, step: Arc<infer_ir::StepPlan>) -> Result<()> {
        let handle = self.batches.seal(step, &self.tasks)?;
        self.submissions
            .push(handle)
            .map_err(|_| Error::invariant("submit ring full"))?;
        let handle = self
            .applied
            .pop()
            .map_err(|_| Error::invariant("submit ring empty"))?;
        self.batches.apply(handle, |step, tasks| {
            black_box(step.work.len());
            for task in tasks {
                black_box(task.tokens.delta(task.tokens.computed_frontier())?);
            }
            Ok(())
        })?;
        self.batches.acknowledge(handle, true)?;
        self.batches.complete(handle, ())?;
        self.completions
            .push(handle)
            .map_err(|_| Error::invariant("completion ring full"))?;
        let handle = self
            .fenced
            .pop()
            .map_err(|_| Error::invariant("completion ring empty"))?;
        if self.batches.take_completion(handle)?.is_none() {
            return Err(Error::invariant("lost CPU completion"));
        }
        Ok(())
    }
}
pub fn run(requests: usize, batch: usize, context: usize) -> Result<Report> {
    let mut cycle = Cycle::new(requests, batch, context)?;
    let measurement = measure(1000, || cycle.tick())?;
    cycle.queue.check_invariants()?;
    Ok(Report {
        requests,
        batch: batch.min(requests),
        candidates: requests.min(256),
        context,
        measurement,
        stages: [
            "queue_and_projection",
            "packing",
            "validation",
            "seal_handoff_ack_fence",
            "commit_and_requeue",
        ]
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let mut samples: Vec<_> = cycle.samples[128..].iter().map(|row| row[index]).collect();
            samples.sort_unstable();
            Stage {
                name,
                p50_ns: samples[500],
                p99_ns: samples[990],
            }
        })
        .collect(),
    })
}

fn row_index(id: RequestId) -> Result<usize> {
    usize::try_from(id.get() - 1).map_err(|_| Error::invalid("benchmark request index overflow"))
}

fn elapsed(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
