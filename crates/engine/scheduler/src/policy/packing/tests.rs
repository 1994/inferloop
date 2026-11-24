use super::*;
use crate::{CostAwarePolicy, FallbackCosts};
use infer_core::{ProgramId, RequestId, StateId};
use infer_ir::{BackendKind, SchedulerConfig};
use infer_spi::SchedulingPolicy;
use std::cell::Cell;

// Exhaustive reference selection evaluates every candidate before picking a winner.
// Keep this slow path in tests so shortcuts remain accountable to complete decision evidence.
fn exhaustive(
    mut builder: BatchBuilder<'_>,
    id: DecisionId,
    step: StepId,
) -> Result<SchedulingDecision> {
    while builder.used_tokens < builder.resources.max_num_batched_tokens {
        let mut within: Option<Quantum> = None;
        let mut atomic: Option<Quantum> = None;
        for index in 0..builder.ready.len() {
            let Some((tokens, cached_rank)) = builder.candidate(index) else {
                continue;
            };
            let item = &builder.ready[index];
            let rank = priority(
                item,
                builder.scratch.tenants[builder.scratch.progress[index].tenant_slot].finish,
                builder.costs.estimate(&[query(item, 1)])?.gpu_us,
                builder.now,
                builder.resources,
                builder.urgent_used,
            );
            assert_eq!(rank.key(), cached_rank.key());
            assert_eq!(rank.reason, cached_rank.reason);
            let Some(quantum) = builder.probe(index, tokens, rank)? else {
                continue;
            };
            let best = match quantum.fit {
                BudgetFit::WithinBudget => &mut within,
                BudgetFit::AtomicOverrun => &mut atomic,
            };
            if best
                .as_ref()
                .is_none_or(|old| quantum.priority.key() < old.priority.key())
            {
                *best = Some(quantum);
            }
        }
        let Some(quantum) = within.or(atomic) else {
            break;
        };
        builder.select(quantum)?;
        if builder.overrun {
            break;
        }
    }
    let mut output =
        infer_spi::DecisionStorage::new(builder.ready.len(), builder.resources.max_num_seqs)?;
    builder.into_decision(id, step, &mut output)
}
struct NonlinearCosts;
impl CostModelProvider for NonlinearCosts {
    fn identity(&self) -> &'static str {
        "test-nonlinear"
    }
    fn estimate(&self, work: &[CostQuery]) -> Result<CostEstimate> {
        let mut estimate = crate::aggregate(work)?;
        let tokens: usize = work.iter().map(|query| query.tokens).sum();
        // Neither additive nor monotone: packing cannot binary-search or sum singleton costs.
        if tokens.is_multiple_of(7) {
            estimate.gpu_us /= 3;
        }
        if work.len().is_multiple_of(3) {
            estimate.workspace_bytes += 100;
        }
        Ok(estimate)
    }
}
struct Random(usize);
impl Random {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 % bound
    }
}
fn items(random: &mut Random) -> Result<Vec<ReadyWork>> {
    (1..=random.next(24) + 1)
        .map(|id| {
            let program = ProgramId::new(if random.next(8) == 0 { 2 } else { 1 })?;
            let role = [
                ExecutionRole::Decode,
                ExecutionRole::Prefill,
                ExecutionRole::Forward,
            ][random.next(3)];
            let tenant = random.next(4);
            let unit = CostEstimate {
                gpu_us: random.next(80) as u64 + 1,
                workspace_bytes: random.next(180) as u64,
                num_gpu_blocks: random.next(3),
                ..Default::default()
            };
            Ok(ReadyWork {
                request: RequestId::new(id as u64)?,
                state: StateId::new(id as u64)?,
                program,
                role,
                remaining_tokens: random.next(20) + 1,
                tenant: format!("tenant{tenant}"),
                weight: u32::try_from(tenant).unwrap_or(0) + 1,
                deadline_us: (random.next(4) == 0).then_some(random.next(400) as u64),
                virtual_finish: random.next(20_000_000) as u64,
                cost_per_token: unit,
                cost_query: CostQuery::from_unit(
                    program,
                    BackendKind::Metal,
                    role,
                    1,
                    random.next(64) + 1,
                    unit,
                ),
                latency_deadline_us: (random.next(3) == 0).then_some(random.next(400) as u64),
                remaining_latency_us: random.next(400) as u64,
                remaining_completion_us: random.next(400) as u64,
                last_service_us: random.next(100) as u64,
            })
        })
        .collect()
}
#[test]
fn prioritized_packing_matches_exhaustive_search_with_nonlinear_costs() -> Result<()> {
    let mut random = Random(42);
    for _ in 0..512 {
        let ready = items(&mut random)?;
        let resources = ResourceSnapshot {
            max_num_seqs: random.next(16) + 1,
            max_num_batched_tokens: random.next(64) + 1,
            gpu_budget_us: random.next(256) as u64 + 1,
            workspace_bytes: random.next(200) as u64,
            free_state_pages: random.next(8),
            free_logical_pages: 8,
            free_state_bytes: 4096,
            transfer_budget_us: 100,
            encoder_budget_us: 100,
            graphs: vec![],
            scheduler: SchedulerConfig {
                fair_quantum_tokens: random.next(8) + 1,
                max_planning_probes: 65536,
                urgent_budget_percent: u32::try_from(random.next(101)).unwrap_or(0),
                max_wait_us: random.next(150) as u64 + 1,
                ..Default::default()
            },
        };
        let id = DecisionId::new(1)?;
        let step = StepId::new(2)?;
        for costs in [&FallbackCosts as &dyn CostModelProvider, &NonlinearCosts] {
            let expected = exhaustive(
                BatchBuilder::new(
                    &ready,
                    &resources,
                    100,
                    costs,
                    &mut PackingWorkspace::default(),
                )?,
                id,
                step,
            )?;
            let actual =
                CostAwarePolicy.plan_with_cost(&ready, &resources, 100, id, step, costs)?;
            assert_eq!(actual, expected);
        }
    }
    Ok(())
}
struct Counted(Cell<usize>);
impl CostModelProvider for Counted {
    fn identity(&self) -> &'static str {
        "counted"
    }
    fn estimate(&self, queries: &[CostQuery]) -> Result<CostEstimate> {
        self.0.set(self.0.get() + 1);
        FallbackCosts.estimate(queries)
    }
}
#[test]
fn unchanged_minimum_queries_are_not_repeated_for_each_quantum() -> Result<()> {
    let ready = items(&mut Random(7))?;
    let resources = ResourceSnapshot {
        max_num_seqs: 32,
        max_num_batched_tokens: 128,
        gpu_budget_us: 10_000,
        workspace_bytes: 10_000,
        free_state_pages: 1024,
        free_logical_pages: 1024,
        free_state_bytes: 1 << 20,
        transfer_budget_us: 1000,
        encoder_budget_us: 1000,
        graphs: vec![],
        scheduler: SchedulerConfig::default(),
    };
    let fast = Counted(Cell::new(0));
    let slow = Counted(Cell::new(0));
    let id = DecisionId::new(1)?;
    let step = StepId::new(2)?;
    let expected = exhaustive(
        BatchBuilder::new(
            &ready,
            &resources,
            0,
            &slow,
            &mut PackingWorkspace::default(),
        )?,
        id,
        step,
    )?;
    let actual = CostAwarePolicy.plan_with_cost(&ready, &resources, 0, id, step, &fast)?;
    assert_eq!(actual, expected);
    assert!(fast.0.get() < slow.0.get());
    Ok(())
}

#[test]
fn persistent_workspace_preserves_capacity_and_decision_evidence_across_steps() -> Result<()> {
    let ready = items(&mut Random(7))?;
    let resources = ResourceSnapshot {
        max_num_seqs: 32,
        max_num_batched_tokens: 128,
        gpu_budget_us: 10_000,
        workspace_bytes: 10_000,
        free_state_pages: 1024,
        free_logical_pages: 1024,
        free_state_bytes: 1 << 20,
        transfer_budget_us: 1000,
        encoder_budget_us: 1000,
        graphs: vec![],
        scheduler: SchedulerConfig::default(),
    };
    let mut workspace = PackingWorkspace::default();
    let id = DecisionId::ONE;
    let step = StepId::ONE;
    let expected =
        CostAwarePolicy.plan_with_cost(&ready, &resources, 100, id, step, &FallbackCosts)?;
    let context = || infer_spi::PlanningContext {
        ready: &ready,
        resources: &resources,
        now_us: 100,
        decision: id,
        step,
        costs: &FallbackCosts,
    };
    assert_eq!(
        CostAwarePolicy.plan_reusing(context(), &mut workspace)?,
        expected
    );
    let pointers = (
        workspace.progress.as_ptr(),
        workspace.tenants.as_ptr(),
        workspace.queries.as_ptr(),
        workspace.requests.as_ptr(),
        workspace.states.as_ptr(),
        workspace.tenant_order.as_ptr(),
    );
    let candidates_capacity = workspace.candidates.capacity();
    for _ in 0..64 {
        assert_eq!(
            CostAwarePolicy.plan_reusing(context(), &mut workspace)?,
            expected
        );
        assert_eq!(
            (
                workspace.progress.as_ptr(),
                workspace.tenants.as_ptr(),
                workspace.queries.as_ptr(),
                workspace.requests.as_ptr(),
                workspace.states.as_ptr(),
                workspace.tenant_order.as_ptr()
            ),
            pointers
        );
        assert_eq!(workspace.candidates.capacity(), candidates_capacity);
    }
    Ok(())
}
