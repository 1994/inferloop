//! Batch packing owns its queries, selections and per-request progress.
use super::{
    Priority, hard_limit, priority, query, service_charge, slack, target, validate_scheduler,
};
use crate::constants::PERCENT;
use crate::cost::update_summary;
use infer_core::{DecisionId, Error, Result, StepId};
use infer_ir::{
    CostEstimate, CostQuery, DeferReason, DeferredWork, ExecutionRole, PlannedWork, ReadyWork,
    ResourceSnapshot, SchedulingDecision, SelectionEvidence, SelectionReason, StepPlan,
};
use infer_spi::CostModelProvider;
use std::mem::replace;
use std::mem::take;
use std::{cmp::Ordering, collections::BinaryHeap};

struct RequestProgress {
    tokens: usize,
    selected_slot: Option<usize>,
    tenant_slot: usize,
    minimum: CostEstimate,
    urgent: Priority,
    regular: Priority,
}
struct TenantTag {
    weight: u32,
    finish: u64,
}
struct Selection {
    work: PlannedWork,
    evidence: SelectionEvidence,
}
#[derive(Clone, Copy)]
enum BudgetFit {
    WithinBudget,
    AtomicOverrun,
}
struct Quantum {
    index: usize,
    tokens: usize,
    priority: Priority,
    cost: CostEstimate,
    fit: BudgetFit,
}
// Reverse priority ordering makes the heap yield the best candidate first.
struct Candidate {
    index: usize,
    tokens: usize,
    priority: Priority,
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.key().cmp(&self.priority.key())
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
enum QueryTrial {
    Replaced { slot: usize, previous: CostQuery },
    Appended,
}
impl QueryTrial {
    fn restore(self, queries: &mut Vec<CostQuery>) {
        match self {
            Self::Replaced { slot, previous } => queries[slot] = previous,
            Self::Appended => {
                queries.pop();
            }
        }
    }
}
#[derive(Default)]
pub struct PackingWorkspace {
    progress: Vec<RequestProgress>,
    tenants: Vec<TenantTag>,
    selections: Vec<Selection>,
    queries: Vec<CostQuery>,
    candidates: BinaryHeap<Candidate>,
    tried: Vec<Candidate>,
    requests: Vec<infer_core::RequestId>,
    states: Vec<infer_core::StateId>,
    tenant_order: Vec<usize>,
}
impl PackingWorkspace {
    pub(super) fn reserve(&mut self, requests: usize, batch: usize) -> Result<()> {
        let batch = batch.min(requests);
        let reserve = |error| {
            Error::new(
                infer_core::ErrorCode::Capacity,
                format!("planner scratch allocation: {error}"),
            )
        };
        self.progress.try_reserve_exact(requests).map_err(reserve)?;
        self.tenants.try_reserve_exact(requests).map_err(reserve)?;
        self.selections.try_reserve_exact(batch).map_err(reserve)?;
        self.queries
            .try_reserve_exact(
                batch
                    .checked_add(1)
                    .ok_or_else(|| Error::invalid("planner query capacity overflow"))?,
            )
            .map_err(reserve)?;
        self.candidates
            .try_reserve_exact(requests)
            .map_err(reserve)?;
        self.tried.try_reserve_exact(requests).map_err(reserve)?;
        self.requests.try_reserve_exact(requests).map_err(reserve)?;
        self.states.try_reserve_exact(requests).map_err(reserve)?;
        self.tenant_order
            .try_reserve_exact(requests)
            .map_err(reserve)?;
        Ok(())
    }
}
pub(super) struct BatchBuilder<'a> {
    ready: &'a [ReadyWork],
    resources: &'a ResourceSnapshot,
    costs: &'a dyn CostModelProvider,
    now: u64,
    scratch: &'a mut PackingWorkspace,
    used_tokens: usize,
    urgent_used: u64,
    predicted: CostEstimate,
    overrun: bool,
    heap_urgent: Option<bool>,
    probes: usize,
    summary: Option<infer_spi::BatchCostSummary>,
}
impl<'a> BatchBuilder<'a> {
    pub(super) fn new(
        ready: &'a [ReadyWork],
        resources: &'a ResourceSnapshot,
        now: u64,
        costs: &'a dyn CostModelProvider,
        scratch: &'a mut PackingWorkspace,
    ) -> Result<Self> {
        validate_scheduler(&resources.scheduler)?;
        if resources.max_num_seqs == 0
            || resources.max_num_batched_tokens == 0
            || resources.gpu_budget_us == 0
        {
            return Err(Error::invalid("positive scheduling budgets required"));
        }
        validate_ready(ready, resources, now, costs, scratch)?;
        scratch.selections.clear();
        scratch.queries.clear();
        scratch.candidates.clear();
        scratch.tried.clear();
        Ok(Self {
            ready,
            resources,
            costs,
            now,
            scratch,
            used_tokens: 0,
            urgent_used: 0,
            predicted: CostEstimate::default(),
            overrun: false,
            heap_urgent: None,
            probes: 0,
            summary: costs
                .supports_batch_summary()
                .then(infer_spi::BatchCostSummary::default),
        })
    }
    pub(super) fn pack(self, id: DecisionId, step: StepId) -> Result<SchedulingDecision> {
        let mut output =
            infer_spi::DecisionStorage::new(self.ready.len(), self.resources.max_num_seqs)?;
        self.pack_into(id, step, &mut output)
    }
    pub(super) fn pack_into(
        mut self,
        id: DecisionId,
        step: StepId,
        output: &mut infer_spi::DecisionStorage,
    ) -> Result<SchedulingDecision> {
        let batch = self.resources.max_num_seqs.min(self.ready.len());
        if output.work.capacity() < batch
            || output.selected.capacity() < batch
            || output.deferred.capacity() < self.ready.len()
        {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "planner output exceeds reserved capacity",
            ));
        }
        while self.used_tokens < self.resources.max_num_batched_tokens
            && self.probes < self.resources.scheduler.max_planning_probes
        {
            let Some(quantum) = self.next_quantum()? else {
                break;
            };
            self.select(quantum)?;
            if self.overrun {
                break;
            }
        }
        self.into_decision(id, step, output)
    }
    fn compatible(&self, item: &ReadyWork) -> bool {
        self.scratch.queries.first().is_none_or(|first| {
            first.program == item.program && first.backend == item.cost_query.backend
        })
    }
    fn urgent_available(&self) -> bool {
        u128::from(self.urgent_used) * u128::from(PERCENT)
            < u128::from(self.resources.gpu_budget_us)
                * u128::from(self.resources.scheduler.urgent_budget_percent)
    }
    fn next_quantum(&mut self) -> Result<Option<Quantum>> {
        let urgent = self.urgent_available();
        if self.heap_urgent != Some(urgent) {
            self.scratch.candidates.clear();
            for index in 0..self.ready.len() {
                if let Some((tokens, priority)) = self.candidate(index) {
                    self.scratch.candidates.push(Candidate {
                        index,
                        tokens,
                        priority,
                    });
                }
            }
            self.heap_urgent = Some(urgent);
        }
        let result = self.pick_quantum();
        // A rejected full-batch query can become feasible after another selection: costs
        // need not be monotonic. Retain all probes for the next quantum, including errors.
        self.scratch.candidates.extend(self.scratch.tried.drain(..));
        result.map(|quantum| quantum.or_else(|| self.minimum_progress()))
    }
    fn minimum_progress(&self) -> Option<Quantum> {
        if !self.scratch.selections.is_empty()
            || self.probes < self.resources.scheduler.max_planning_probes
        {
            return None;
        }
        // Singleton forecasts were independently collected before trial packing. Exhaustion
        // cannot hide every feasible one-token request behind an expensive large quantum.
        self.ready
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                let (_, priority) = self.candidate(index)?;
                let cost = self.scratch.progress[index].minimum;
                let fit = self.budget_fit(cost, 1)?;
                Some(Quantum {
                    index,
                    tokens: 1,
                    priority,
                    cost,
                    fit,
                })
            })
            .min_by_key(|quantum| {
                (
                    matches!(quantum.fit, BudgetFit::AtomicOverrun),
                    quantum.priority.key(),
                )
            })
    }
    fn pick_quantum(&mut self) -> Result<Option<Quantum>> {
        let mut atomic = None;
        while let Some(candidate) = self.scratch.candidates.pop() {
            if self.probes == self.resources.scheduler.max_planning_probes {
                return Ok(atomic);
            }
            let Some((tokens, rank)) = self.candidate(candidate.index) else {
                continue;
            };
            if candidate.priority.key() != rank.key() || candidate.tokens != tokens {
                // Within an urgency epoch, tenant finish only increases. Stale keys are
                // lower bounds, so repair on pop without overlooking a better request.
                self.scratch.candidates.push(Candidate {
                    index: candidate.index,
                    tokens,
                    priority: rank,
                });
                continue;
            }
            let index = candidate.index;
            self.scratch.tried.push(candidate);
            let Some(quantum) = self.probe(index, tokens, rank)? else {
                continue;
            };
            match quantum.fit {
                BudgetFit::WithinBudget => return Ok(Some(quantum)),
                BudgetFit::AtomicOverrun if atomic.is_none() => atomic = Some(quantum),
                BudgetFit::AtomicOverrun => {}
            }
        }
        Ok(atomic)
    }
    fn candidate(&self, index: usize) -> Option<(usize, Priority)> {
        let item = &self.ready[index];
        let progress = &self.scratch.progress[index];
        if item
            .deadline_us
            .is_some_and(|deadline| deadline <= self.now)
            || !self.compatible(item)
            || (progress.selected_slot.is_none()
                && self.scratch.selections.len() >= self.resources.max_num_seqs)
        {
            return None;
        }
        let cap = match item.role {
            ExecutionRole::Decode => 1,
            ExecutionRole::Prefill => self.resources.scheduler.prefill_chunk_tokens,
            ExecutionRole::Forward => self.resources.scheduler.forward_chunk_tokens,
            ExecutionRole::Mixed => return None,
        };
        let chunk = item
            .remaining_tokens
            .min(cap)
            .saturating_sub(progress.tokens);
        let budget = self
            .resources
            .max_num_batched_tokens
            .saturating_sub(self.used_tokens);
        // A prompt chunk is atomic: one captured prompt graph replays at a fixed width, so a
        // chunk sliced into fair quanta pays a whole replay per slice for no extra progress.
        // Grant the whole chunk and let fairness order whole chunks between rounds instead of
        // trimming them. When decodes already hold part of the round, grant what remains if it
        // still funds at least half a chunk: the replay costs the same, and deferring instead
        // starved every prompt while any request decoded, serializing concurrent requests. A
        // round budget smaller than one chunk keeps the fair-quantum split, so a small budget
        // cannot starve prompt work.
        let atomic =
            item.role == ExecutionRole::Prefill && chunk <= self.resources.max_num_batched_tokens;
        let quantum = match item.role {
            ExecutionRole::Prefill if atomic && chunk > budget && budget * 2 >= chunk => {
                self.partial_prompt_grant(index, chunk, budget)
            }
            ExecutionRole::Prefill if atomic && chunk > budget => return None,
            ExecutionRole::Prefill if atomic => chunk,
            _ => chunk
                .min(budget)
                .min(self.resources.scheduler.fair_quantum_tokens),
        };
        if quantum == 0 {
            return None;
        }
        let mut rank = if self.urgent_available() {
            progress.urgent
        } else {
            progress.regular
        };
        rank.finish = self.scratch.tenants[progress.tenant_slot]
            .finish
            .saturating_add(rank.finish);
        Some((quantum, rank))
    }
    // Only trade a partial grant for alignment when it removes a full wide replay.
    // Saving just a narrow tail is not worth an additional mixed decode round. Use the
    // state's actual worker rather than the service's larger private prompt chunk.
    fn partial_prompt_grant(&self, index: usize, chunk: usize, budget: usize) -> usize {
        let item = &self.ready[index];
        // A short native prompt needs one replay. After another prompt has made
        // progress, keep it whole instead of creating a tiny tail with another
        // capture/normalization shape. Large chunks keep the half-chunk rule so
        // active decodes cannot starve prompt work.
        if item.cost_query.prefill_geometry.is_some_and(|geometry| {
            chunk <= geometry.width / 2 && chunk <= self.resources.max_num_batched_tokens / 2
        }) && self
            .scratch
            .queries
            .iter()
            .any(|query| query.role == ExecutionRole::Prefill)
        {
            return 0;
        }
        let Some(geometry) = item.cost_query.prefill_geometry.filter(|geometry| {
            geometry.width > 0
                && geometry.width < self.resources.scheduler.prefill_chunk_tokens
                && geometry.narrow_width < geometry.width
        }) else {
            return budget;
        };
        let aligned = budget / geometry.width * geometry.width;
        // Decodes alone must never starve a prompt, and a nonzero grant keeps the
        // existing half-chunk progress bound. A prior prompt may fill the round instead.
        if (aligned == 0
            && !self
                .scratch
                .queries
                .iter()
                .any(|q| q.role == ExecutionRole::Prefill))
            || (aligned > 0 && aligned < chunk.div_ceil(2))
        {
            return budget;
        }
        let remaining = item.remaining_tokens - self.scratch.progress[index].tokens;
        let wide_replays = |tokens: usize| {
            tokens / geometry.width + usize::from(tokens % geometry.width > geometry.narrow_width)
        };
        let greedy = wide_replays(budget) + wide_replays(remaining - budget);
        let whole = wide_replays(aligned) + wide_replays(remaining - aligned);
        if whole < greedy { aligned } else { budget }
    }

    fn probe(&mut self, index: usize, quantum: usize, rank: Priority) -> Result<Option<Quantum>> {
        // The descending search tries the whole prompt chunk first, so a round budget that funds
        // the chunk keeps it atomic. Its floor stays 1 so the singleton-overrun escape still
        // works when the device budget rejects even a full chunk: a round that can schedule
        // nothing else must still make progress.
        for tokens in (1..=quantum).rev() {
            if self.probes == self.resources.scheduler.max_planning_probes {
                return Ok(None);
            }
            self.probes += 1;
            let cost = self.preview(index, tokens)?;
            let Some(fit) = self.budget_fit(cost, tokens) else {
                continue;
            };
            return Ok(Some(Quantum {
                index,
                tokens,
                priority: rank,
                cost,
                fit,
            }));
        }
        Ok(None)
    }
    fn budget_fit(&self, cost: CostEstimate, tokens: usize) -> Option<BudgetFit> {
        if hard_limit(cost, self.resources).is_some() {
            return None;
        }
        if cost.gpu_us <= self.resources.gpu_budget_us {
            return Some(BudgetFit::WithinBudget);
        }
        (self.scratch.selections.is_empty()
            && tokens == 1
            && cost.gpu_us <= self.resources.scheduler.max_singleton_gpu_us)
            .then_some(BudgetFit::AtomicOverrun)
    }
    fn preview(&mut self, index: usize, tokens: usize) -> Result<CostEstimate> {
        if self.scratch.queries.is_empty() && tokens == 1 {
            return Ok(self.scratch.progress[index].minimum);
        }
        let q = query(
            &self.ready[index],
            self.scratch.progress[index].tokens + tokens,
        );
        if let Some(summary) = self.summary {
            let previous = self.scratch.progress[index]
                .selected_slot
                .map(|slot| self.scratch.queries[slot]);
            let summary = update_summary(summary, previous, q)?;
            return self.costs.estimate_summary(&summary)?.ok_or_else(|| {
                Error::invariant("cost provider advertised unavailable batch summary")
            });
        }
        let trial = if let Some(slot) = self.scratch.progress[index].selected_slot {
            QueryTrial::Replaced {
                slot,
                previous: replace(&mut self.scratch.queries[slot], q),
            }
        } else {
            self.scratch.queries.push(q);
            QueryTrial::Appended
        };
        let estimate = self.costs.estimate(&self.scratch.queries);
        trial.restore(&mut self.scratch.queries);
        estimate
    }
    fn select(&mut self, mut quantum: Quantum) -> Result<()> {
        let item = &self.ready[quantum.index];
        let progress = &mut self.scratch.progress[quantum.index];
        progress.tokens += quantum.tokens;
        self.used_tokens += quantum.tokens;
        let standalone = if quantum.tokens == 1 {
            progress.minimum.gpu_us
        } else {
            self.costs.estimate(&[query(item, quantum.tokens)])?.gpu_us
        };
        let tenant = &mut self.scratch.tenants[progress.tenant_slot];
        tenant.finish = tenant
            .finish
            .saturating_add(service_charge(standalone, item.weight));
        self.overrun = matches!(quantum.fit, BudgetFit::AtomicOverrun);
        if self.overrun {
            quantum.priority.reason = SelectionReason::MinimumQuantum;
        }
        if quantum.priority.reason == SelectionReason::UrgentSlo {
            self.urgent_used = self
                .urgent_used
                .saturating_add(quantum.cost.gpu_us.saturating_sub(self.predicted.gpu_us));
        }
        self.predicted = quantum.cost;
        let q = query(item, progress.tokens);
        if let Some(summary) = self.summary {
            let previous = progress
                .selected_slot
                .map(|slot| self.scratch.queries[slot]);
            self.summary = Some(update_summary(summary, previous, q)?);
        }
        if let Some(slot) = progress.selected_slot {
            let selection = &mut self.scratch.selections[slot];
            selection.work.token_count = progress.tokens;
            selection.evidence.tokens = progress.tokens;
            selection.evidence.virtual_finish = tenant.finish;
            selection.evidence.predicted_us = self.costs.estimate(&[q])?.gpu_us;
            self.scratch.queries[slot] = q;
            return Ok(());
        }
        progress.selected_slot = Some(self.scratch.selections.len());
        self.scratch.queries.push(q);
        self.scratch.selections.push(Selection {
            work: PlannedWork {
                request: item.request,
                state: item.state,
                token_count: quantum.tokens,
                role: item.role,
            },
            evidence: SelectionEvidence {
                request: item.request,
                role: item.role,
                tokens: quantum.tokens,
                target_us: target(item),
                slack_us: slack(item, self.now),
                waiting_us: self.now.saturating_sub(item.last_service_us),
                virtual_finish: tenant.finish,
                predicted_us: standalone,
                reason: quantum.priority.reason,
            },
        });
        Ok(())
    }
    fn deferral(&mut self, index: usize) -> Result<DeferredWork> {
        let item = &self.ready[index];
        let minimum = self.scratch.progress[index].minimum;
        let (reason, required, available) = if let Some(reason) = self.fixed_deferral(item, minimum)
        {
            reason
        } else {
            self.probes += 1;
            let combined = self.preview(index, 1)?;
            hard_limit(combined, self.resources).unwrap_or((
                DeferReason::GpuBudget,
                combined.gpu_us,
                self.resources.gpu_budget_us,
            ))
        };
        Ok(DeferredWork {
            request: item.request,
            reason,
            required,
            available,
        })
    }
    // Known limits do not need a hypothetical over-capacity batch cost query. Arbitrary
    // nonlinear providers still evaluate the complete batch whenever feasibility depends on it.
    fn fixed_deferral(
        &self,
        item: &ReadyWork,
        minimum: CostEstimate,
    ) -> Option<(DeferReason, u64, u64)> {
        if let Some(deadline) = item.deadline_us.filter(|deadline| *deadline <= self.now) {
            return Some((DeferReason::Deadline, self.now, deadline));
        }
        if !self.compatible(item) {
            return Some((
                DeferReason::IncompatibleProgram,
                item.program.get(),
                self.scratch
                    .queries
                    .first()
                    .map_or(0, |query| query.program.get()),
            ));
        }
        if let Some(limit) = hard_limit(minimum, self.resources) {
            return Some(limit);
        }
        if minimum.gpu_us > self.resources.scheduler.max_singleton_gpu_us {
            return Some((
                DeferReason::AtomicCostLimit,
                minimum.gpu_us,
                self.resources.scheduler.max_singleton_gpu_us,
            ));
        }
        if self.scratch.selections.len() >= self.resources.max_num_seqs {
            return Some((
                DeferReason::BatchLimit,
                self.scratch.selections.len() as u64 + 1,
                self.resources.max_num_seqs as u64,
            ));
        }
        if self.used_tokens >= self.resources.max_num_batched_tokens {
            return Some((DeferReason::TokenBudget, 1, 0));
        }
        (self.probes >= self.resources.scheduler.max_planning_probes).then_some((
            DeferReason::PlanningBudget,
            self.probes as u64 + 1,
            self.resources.scheduler.max_planning_probes as u64,
        ))
    }
    fn into_decision(
        mut self,
        id: DecisionId,
        step: StepId,
        output: &mut infer_spi::DecisionStorage,
    ) -> Result<SchedulingDecision> {
        output.deferred.clear();
        output.selected.clear();
        output.work.clear();
        for index in 0..self.ready.len() {
            if self.scratch.progress[index].tokens == 0 {
                output.deferred.push(self.deferral(index)?);
            }
        }
        output.deferred.sort_unstable_by_key(|work| work.request);
        let role = self.scratch.selections.first().map(|first| {
            if self
                .scratch
                .selections
                .iter()
                .all(|selection| selection.work.role == first.work.role)
            {
                first.work.role
            } else {
                ExecutionRole::Mixed
            }
        });
        let program = self.scratch.queries.first().map(|query| query.program);
        let graph = self
            .resources
            .graphs
            .iter()
            .filter(|_| role == Some(ExecutionRole::Decode) && !self.overrun)
            .filter(|graph| {
                graph.batch_capacity >= self.scratch.selections.len()
                    && graph.max_tokens >= self.used_tokens
                    && graph.workspace_bytes <= self.resources.workspace_bytes
                    && graph.program.is_none_or(|id| Some(id) == program)
            })
            .min_by_key(|graph| {
                (
                    graph.batch_capacity,
                    graph.max_tokens,
                    graph.workspace_bytes,
                )
            })
            .cloned();
        if let Some(graph) = &graph {
            self.predicted.workspace_bytes =
                self.predicted.workspace_bytes.max(graph.workspace_bytes);
        }
        for selection in self.scratch.selections.drain(..) {
            output.work.push(selection.work);
            output.selected.push(selection.evidence);
        }
        let planned = program
            .map(|program| -> Result<StepPlan> {
                Ok(StepPlan {
                    id: step,
                    decision: id,
                    program,
                    role: role
                        .ok_or_else(|| Error::invariant("scheduled batch has no execution role"))?,
                    work: take(&mut output.work),
                    cost: self.predicted,
                    graph,
                    quantum_overrun: self.overrun,
                })
            })
            .transpose()?;
        Ok(SchedulingDecision {
            window: None,
            id,
            step: planned,
            deferred: take(&mut output.deferred),
            selected: take(&mut output.selected),
        })
    }
}
fn validate_ready(
    ready: &[ReadyWork],
    resources: &ResourceSnapshot,
    now: u64,
    costs: &dyn CostModelProvider,
    scratch: &mut PackingWorkspace,
) -> Result<()> {
    scratch.requests.clear();
    scratch.states.clear();
    scratch.tenants.clear();
    scratch.progress.clear();
    for item in ready {
        if item.remaining_tokens == 0
            || item.weight == 0
            || item.tenant.is_empty()
            || item.role == ExecutionRole::Mixed
            || item.cost_query.program != item.program
            || item.cost_query.role != item.role
        {
            return Err(Error::invalid("invalid ready work"));
        }
        scratch.requests.push(item.request);
        scratch.states.push(item.state);
        let minimum = costs.estimate(&[query(item, 1)])?;
        scratch.progress.push(RequestProgress {
            tenant_slot: 0,
            minimum,
            tokens: 0,
            selected_slot: None,
            urgent: priority(item, 0, minimum.gpu_us, now, resources, 0),
            regular: priority(item, 0, minimum.gpu_us, now, resources, u64::MAX),
        });
    }
    scratch.tenant_order.clear();
    scratch.tenant_order.extend(0..ready.len());
    scratch
        .tenant_order
        .sort_unstable_by(|left, right| ready[*left].tenant.cmp(&ready[*right].tenant));
    let mut previous: Option<usize> = None;
    for &index in &scratch.tenant_order {
        let item = &ready[index];
        if previous.is_none_or(|previous| ready[previous].tenant != item.tenant) {
            scratch.tenants.push(TenantTag {
                weight: item.weight,
                finish: item.virtual_finish,
            });
        }
        let slot = scratch.tenants.len() - 1;
        let tenant = &mut scratch.tenants[slot];
        if tenant.weight != item.weight {
            return Err(Error::invalid("inconsistent tenant weights"));
        }
        tenant.finish = tenant.finish.max(item.virtual_finish);
        scratch.progress[index].tenant_slot = slot;
        previous = Some(index);
    }
    scratch.requests.sort_unstable();
    scratch.states.sort_unstable();
    if scratch.requests.windows(2).any(|pair| pair[0] == pair[1])
        || scratch.states.windows(2).any(|pair| pair[0] == pair[1])
    {
        return Err(Error::invalid("aliased ready work"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
