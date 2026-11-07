use infer_core::*;
use infer_ir::*;
use infer_scheduler::*;
use infer_spi::{CostModelProvider, SchedulingPolicy};
use std::collections::BTreeSet;

#[cfg(test)]
mod fixtures {
    use super::*;

    pub fn work(id: u64, role: ExecutionRole, tokens: usize) -> ReadyWork {
        let program = ProgramId::new(1).unwrap();
        let unit = CostEstimate {
            gpu_us: 1,
            ..Default::default()
        };
        ReadyWork {
            request: RequestId::new(id).unwrap(),
            state: StateId::new(id).unwrap(),
            program,
            role,
            remaining_tokens: tokens,
            tenant: format!("t{id}"),
            weight: 1,
            deadline_us: None,
            virtual_finish: 0,
            cost_per_token: unit,
            cost_query: CostQuery::from_unit(program, BackendKind::Metal, role, 1, 1, unit),
            latency_deadline_us: None,
            remaining_latency_us: tokens as u64,
            remaining_completion_us: tokens as u64,
            last_service_us: 0,
        }
    }

    pub fn plan(items: &[ReadyWork], r: &ResourceSnapshot, now: u64) -> SchedulingDecision {
        CostAwarePolicy
            .plan(
                items,
                r,
                now,
                DecisionId::new(1).unwrap(),
                StepId::new(2).unwrap(),
            )
            .unwrap()
    }
}

use fixtures::*;

fn resources() -> ResourceSnapshot {
    ResourceSnapshot {
        max_num_seqs: 8,
        max_num_batched_tokens: 16,
        gpu_budget_us: 100,
        workspace_bytes: 1024,
        free_state_pages: 4,
        free_logical_pages: 4,
        free_state_bytes: 4096,
        transfer_budget_us: 100,
        encoder_budget_us: 100,
        graphs: vec![],
        scheduler: SchedulerConfig {
            fair_quantum_tokens: 1,
            ..Default::default()
        },
    }
}
#[test]
fn slack_uses_remaining_service_instead_of_only_absolute_deadline() {
    let mut short = work(1, ExecutionRole::Prefill, 1);
    short.deadline_us = Some(20);
    let mut long = work(2, ExecutionRole::Prefill, 30);
    long.deadline_us = Some(40);
    let mut r = resources();
    r.max_num_batched_tokens = 1;
    let result = plan(&[short, long], &r, 0);
    assert_eq!(result.step.unwrap().work[0].request.get(), 2);
    assert_eq!(result.selected[0].slack_us, Some(10));
    assert_eq!(result.selected[0].reason, SelectionReason::UrgentSlo);
}
#[test]
fn mixed_batch_protects_decode_and_caps_long_prefill() {
    let mut r = resources();
    r.scheduler.prefill_chunk_tokens = 4;
    let result = plan(
        &[
            work(1, ExecutionRole::Prefill, 100),
            work(2, ExecutionRole::Decode, 1),
            work(3, ExecutionRole::Forward, 20),
        ],
        &r,
        0,
    );
    let step = result.step.unwrap();
    assert_eq!(step.role, ExecutionRole::Mixed);
    assert_eq!(step.work[0].role, ExecutionRole::Decode);
    assert_eq!(
        step.work
            .iter()
            .find(|w| w.request.get() == 1)
            .unwrap()
            .token_count,
        4
    );
    assert_eq!(
        step.work
            .iter()
            .find(|w| w.request.get() == 2)
            .unwrap()
            .token_count,
        1
    );
    assert!(step.work.iter().any(|w| w.role == ExecutionRole::Forward));
    assert_eq!(step.cost.gpu_us, 16);
}
#[test]
fn a_prompt_chunk_is_never_sliced_to_fit_a_round_that_funds_one() {
    let mut r = resources();
    r.max_num_batched_tokens = 64;
    r.scheduler.prefill_chunk_tokens = 64;
    r.scheduler.fair_quantum_tokens = 8;
    let step = plan(
        &[
            work(1, ExecutionRole::Prefill, 52),
            work(2, ExecutionRole::Prefill, 52),
        ],
        &r,
        0,
    )
    .step
    .unwrap();
    // One captured prompt graph replays at a fixed width, so a second chunk cannot ride along
    // in a round that only has 12 of 64 tokens left: it waits instead of paying a replay for
    // 12 tokens.
    assert_eq!(step.work.len(), 1);
    assert_eq!(step.work[0].role, ExecutionRole::Prefill);
    assert_eq!(step.work[0].token_count, 52);
}

#[test]
fn a_round_that_funds_two_chunks_carries_both_whole() {
    let mut r = resources();
    r.max_num_batched_tokens = 128;
    // The per-round device budget must fund both chunks; the fixture default caps the batch.
    r.gpu_budget_us = 1_000;
    r.scheduler.prefill_chunk_tokens = 64;
    r.scheduler.fair_quantum_tokens = 8;
    let step = plan(
        &[
            work(1, ExecutionRole::Prefill, 52),
            work(2, ExecutionRole::Prefill, 52),
        ],
        &r,
        0,
    )
    .step
    .unwrap();
    assert_eq!(step.work.len(), 2);
    assert!(step.work.iter().all(|w| w.token_count == 52));
}

#[test]
fn a_budget_below_one_chunk_still_slices_prompt_work() {
    let mut r = resources();
    r.max_num_batched_tokens = 4;
    r.scheduler.prefill_chunk_tokens = 64;
    r.scheduler.fair_quantum_tokens = 8;
    let step = plan(&[work(1, ExecutionRole::Prefill, 52)], &r, 0)
        .step
        .unwrap();
    assert_eq!(step.work[0].token_count, 4);
}

#[test]
fn wfq_updates_tenant_service_inside_a_batch() {
    let a = work(1, ExecutionRole::Prefill, 32);
    let mut b = work(2, ExecutionRole::Prefill, 32);
    b.weight = 3;
    let mut r = resources();
    r.max_num_batched_tokens = 12;
    let step = plan(&[a, b], &r, 0).step.unwrap();
    let count = |id| {
        step.work
            .iter()
            .find(|w| w.request.get() == id)
            .unwrap()
            .token_count
    };
    assert_eq!(count(1), 3);
    assert_eq!(count(2), 9);
}
#[test]
fn aging_prevents_starvation_even_under_urgent_decode_traffic() {
    let mut long = work(1, ExecutionRole::Forward, 100);
    long.virtual_finish = 1_000_000_000;
    let mut decode = work(2, ExecutionRole::Decode, 1);
    decode.last_service_us = 100;
    decode.latency_deadline_us = Some(101);
    let mut r = resources();
    r.max_num_batched_tokens = 1;
    r.scheduler.max_wait_us = 100;
    let decision = plan(&[long, decode], &r, 100);
    assert_eq!(decision.step.unwrap().work[0].request.get(), 1);
    assert_eq!(decision.selected[0].reason, SelectionReason::Aging);
}
#[test]
fn incompatible_or_resource_heavy_candidate_does_not_block_others() {
    for field in 0..4 {
        let mut bad = work(1, ExecutionRole::Prefill, 1);
        match field {
            0 => bad.cost_query.workspace_bytes = 1025,
            1 => bad.cost_query.num_gpu_blocks = 5,
            2 => bad.cost_query.transfer_us = 101,
            _ => bad.cost_query.encoder_us = 101,
        }
        let result = plan(&[bad, work(2, ExecutionRole::Prefill, 1)], &resources(), 0);
        assert_eq!(result.step.unwrap().work[0].request.get(), 2);
        assert_eq!(result.deferred.len(), 1);
        assert!(result.deferred[0].required > result.deferred[0].available);
    }
    let mut other = work(2, ExecutionRole::Prefill, 1);
    other.program = ProgramId::new(2).unwrap();
    other.cost_query.program = other.program;
    let result = plan(&[work(1, ExecutionRole::Decode, 1), other], &resources(), 0);
    assert_eq!(result.step.unwrap().work.len(), 1);
    assert_eq!(result.deferred[0].reason, DeferReason::IncompatibleProgram);
}
#[test]
fn multi_resource_cost_is_aggregated_for_the_complete_batch() {
    let mut a = work(1, ExecutionRole::Decode, 1);
    a.cost_query.state_bytes = 3000;
    a.cost_query.transfer_us = 60;
    let mut b = work(2, ExecutionRole::Decode, 1);
    b.cost_query.state_bytes = 3000;
    b.cost_query.transfer_us = 60;
    let decision = plan(&[a, b], &resources(), 0);
    assert_eq!(decision.deferred[0].reason, DeferReason::StateCapacity);
    assert_eq!(decision.deferred[0].required, 6000);
    assert_eq!(decision.deferred[0].available, 4096);
    let step = decision.step.unwrap();
    assert_eq!(step.work.len(), 1);
    assert_eq!(step.cost.state_bytes, 3000);
    assert_eq!(step.cost.transfer_us, 60);
}
#[test]
fn minimum_quantum_escape_is_singleton_and_never_relaxes_memory_limits() {
    let mut item = work(1, ExecutionRole::Prefill, 100);
    item.cost_query.fallback_per_token_us = 200;
    let step = plan(&[item.clone()], &resources(), 0).step.unwrap();
    assert!(step.quantum_overrun);
    assert_eq!(step.work[0].token_count, 1);
    assert_eq!(step.cost.gpu_us, 200);
    item.cost_query.state_bytes = 4097;
    assert!(plan(&[item], &resources(), 0).step.is_none());
}
#[test]
fn graph_descriptor_is_program_and_shape_compatible_and_decode_only() {
    let mut r = resources();
    r.graphs = vec![
        GraphVariant {
            batch_capacity: 2,
            max_tokens: 2,
            program: Some(ProgramId::new(2).unwrap()),
            workspace_bytes: 0,
        },
        GraphVariant {
            batch_capacity: 4,
            max_tokens: 4,
            program: Some(ProgramId::new(1).unwrap()),
            workspace_bytes: 500,
        },
    ];
    let step = plan(
        &[
            work(1, ExecutionRole::Decode, 1),
            work(2, ExecutionRole::Decode, 1),
        ],
        &r,
        0,
    )
    .step
    .unwrap();
    assert_eq!(step.graph.unwrap().batch_capacity, 4);
    assert_eq!(step.cost.workspace_bytes, 500);
    let step = plan(
        &[
            work(1, ExecutionRole::Decode, 1),
            work(2, ExecutionRole::Prefill, 2),
        ],
        &r,
        0,
    )
    .step
    .unwrap();
    assert!(step.graph.is_none());
}
#[test]
fn planning_is_input_order_invariant_and_checked_against_overflow() {
    let items = vec![
        work(1, ExecutionRole::Prefill, 32),
        work(2, ExecutionRole::Decode, 1),
        work(3, ExecutionRole::Forward, 32),
    ];
    let original = plan(&items, &resources(), 0);
    let mut reverse = items;
    reverse.reverse();
    assert_eq!(original, plan(&reverse, &resources(), 0));
    let mut item = work(1, ExecutionRole::Prefill, 4);
    item.cost_query.fallback_per_token_us = u64::MAX;
    let mut r = resources();
    r.scheduler.fair_quantum_tokens = 4;
    assert!(
        CostAwarePolicy
            .plan(
                &[item],
                &r,
                0,
                DecisionId::new(1).unwrap(),
                StepId::new(2).unwrap()
            )
            .is_err()
    );
}

#[test]
fn planning_budget_yields_partial_batch_with_explicit_cpu_deferrals() -> Result<()> {
    let ready = vec![
        work(1, ExecutionRole::Prefill, 256),
        work(2, ExecutionRole::Decode, 1),
    ];
    let mut r = resources();
    r.scheduler.max_planning_probes = 1;
    r.max_num_batched_tokens = 1024;
    r.gpu_budget_us = 10_000;
    let decision = CostAwarePolicy.plan(&ready, &r, 0, DecisionId::ONE, StepId::ONE)?;
    assert_eq!(decision.step.as_ref().map(|step| step.work.len()), Some(1));
    assert_eq!(decision.deferred.len(), 1);
    assert_eq!(decision.deferred[0].reason, DeferReason::PlanningBudget);
    validate_decision(&decision, &ready, &r, ProgramId::ONE, &FallbackCosts)
}

#[test]
fn exhausted_large_quantum_probe_still_dispatches_a_feasible_minimum() -> Result<()> {
    let ready = vec![work(1, ExecutionRole::Prefill, 100)];
    let mut r = resources();
    r.scheduler.max_planning_probes = 1;
    r.scheduler.prefill_chunk_tokens = 128;
    r.scheduler.fair_quantum_tokens = 128;
    r.max_num_batched_tokens = 128;
    r.gpu_budget_us = 2;
    let decision = CostAwarePolicy.plan(&ready, &r, 0, DecisionId::ONE, StepId::ONE)?;
    assert_eq!(
        decision
            .step
            .as_ref()
            .and_then(|step| step.work.first())
            .map(|work| work.token_count),
        Some(1)
    );
    validate_decision(&decision, &ready, &r, ProgramId::ONE, &FallbackCosts)
}

#[test]
fn fixed_cost_paging_fixture_keeps_its_mixed_batch_after_slow_gpu_feedback() -> Result<()> {
    let ready = [
        work(1, ExecutionRole::Decode, 1),
        work(2, ExecutionRole::Prefill, 2),
    ];
    let mut r = resources();
    r.gpu_budget_us = 10_000;
    let sample = CostObservation {
        step: StepId::ONE,
        work: vec![CostQuery::from_unit(
            ProgramId::ONE,
            BackendKind::Metal,
            ExecutionRole::Prefill,
            6,
            6,
            CostEstimate {
                gpu_us: 1,
                ..Default::default()
            },
        )]
        .into(),
        timing: ExecutionTiming {
            elapsed_us: 120_000,
            source: TimingSource::MetalGpu,
        },
    };
    for adaptive in [false, true] {
        let mut costs = CalibratedCosts::new(
            "paging-fixture".into(),
            CostModelConfig {
                adaptive,
                ..Default::default()
            },
        )?;
        costs.observe(&sample)?;
        let decision =
            CostAwarePolicy.plan_with_cost(&ready, &r, 0, DecisionId::ONE, StepId::ONE, &costs)?;
        validate_decision(&decision, &ready, &r, ProgramId::ONE, &costs)?;
        let step = decision.step.as_ref().unwrap();
        assert_eq!(step.work.len(), if adaptive { 1 } else { 2 });
        if !adaptive {
            assert_eq!(
                step.work.iter().map(|work| work.token_count).sum::<usize>(),
                3
            );
        }
    }
    Ok(())
}

#[test]
fn graph_forecast_passes_independent_validation_and_forged_graph_is_rejected() -> Result<()> {
    let mut r = resources();
    r.graphs = vec![GraphVariant {
        batch_capacity: 2,
        max_tokens: 2,
        program: Some(ProgramId::ONE),
        workspace_bytes: 500,
    }];
    let ready = [
        work(1, ExecutionRole::Decode, 1),
        work(2, ExecutionRole::Decode, 1),
    ];
    let mut decision = CostAwarePolicy.plan(&ready, &r, 0, DecisionId::ONE, StepId::ONE)?;
    validate_decision(&decision, &ready, &r, ProgramId::ONE, &FallbackCosts)?;
    decision
        .step
        .as_mut()
        .ok_or_else(|| Error::invariant("missing graph step"))?
        .graph
        .as_mut()
        .ok_or_else(|| Error::invariant("missing graph descriptor"))?
        .batch_capacity = 1;
    assert!(validate_decision(&decision, &ready, &r, ProgramId::ONE, &FallbackCosts).is_err());
    Ok(())
}

#[test]
fn independent_validator_rejects_foreign_programs_and_aliased_states() -> Result<()> {
    let r = resources();
    let mut foreign = work(1, ExecutionRole::Decode, 1);
    foreign.program = ProgramId::new(2)?;
    foreign.cost_query.program = foreign.program;
    let ready = [foreign];
    let mut decision = CostAwarePolicy.plan(&ready, &r, 0, DecisionId::ONE, StepId::ONE)?;
    decision
        .step
        .as_mut()
        .ok_or_else(|| Error::invariant("missing step"))?
        .program = ProgramId::ONE;
    assert!(validate_decision(&decision, &ready, &r, ProgramId::ONE, &FallbackCosts).is_err());
    let mut ready = [
        work(1, ExecutionRole::Decode, 1),
        work(2, ExecutionRole::Decode, 1),
    ];
    let decision = CostAwarePolicy.plan(&ready, &r, 0, DecisionId::ONE, StepId::ONE)?;
    ready[1].state = ready[0].state;
    assert!(validate_decision(&decision, &ready, &r, ProgramId::ONE, &FallbackCosts).is_err());
    Ok(())
}

/// The serving-derived budget must preserve decode priority without artificial prompt tails.
#[test]
fn a_full_prompt_replay_coexists_with_decodes_without_slicing_the_next_prompt() {
    let mut r = resources();
    r.max_num_seqs = 16;
    r.scheduler.prefill_chunk_tokens = 1024;
    r.max_num_batched_tokens = r.scheduler.prefill_chunk_tokens + r.max_num_seqs - 1;
    r.gpu_budget_us = 10_000;
    let mut items: Vec<_> = (1..=14)
        .map(|id| work(id, ExecutionRole::Decode, 1))
        .collect();
    items.extend([
        work(15, ExecutionRole::Prefill, 4096),
        work(16, ExecutionRole::Prefill, 4096),
    ]);
    let step = plan(&items, &r, 0).step.unwrap();
    assert_eq!(
        step.work
            .iter()
            .filter(|item| item.role == ExecutionRole::Decode)
            .count(),
        14
    );
    let prompts: Vec<_> = step
        .work
        .iter()
        .filter(|item| item.role == ExecutionRole::Prefill)
        .collect();
    assert_eq!(prompts.len(), 1);
    assert_eq!(prompts[0].token_count, 1024);
    assert_eq!(
        step.work.iter().map(|item| item.token_count).sum::<usize>(),
        1038
    );
}

fn replay_geometry_step(
    tail: usize,
    geometry: Option<PrefillGeometry>,
    budget: usize,
) -> SchedulingDecision {
    let mut r = resources();
    r.max_num_seqs = 32;
    r.max_num_batched_tokens = budget;
    r.gpu_budget_us = 1_000_000;
    r.scheduler.prefill_chunk_tokens = 2048;
    let mut items: Vec<_> = (1..=15)
        .map(|id| work(id, ExecutionRole::Decode, 1))
        .collect();
    if tail > 0 {
        items.push(work(16, ExecutionRole::Prefill, tail));
    }
    let mut prompt = work(17, ExecutionRole::Prefill, 4096);
    prompt.cost_query.prefill_geometry = geometry;
    items.push(prompt);
    plan(&items, &r, 0)
}

#[test]
fn actual_worker_geometry_avoids_a_wide_tail_without_changing_decode_selection() {
    let geometry = Some(PrefillGeometry {
        width: 1024,
        narrow_width: 64,
    });
    for (tail, expected) in [(0, 2033), (16, 2017), (64, 1024), (225, 1024)] {
        let result = replay_geometry_step(tail, geometry, 2048);
        let step = result.step.as_ref().unwrap();
        assert_eq!(
            step.work
                .iter()
                .filter(|w| w.role == ExecutionRole::Decode)
                .count(),
            15
        );
        let prompt = step.work.iter().find(|w| w.request.get() == 17).unwrap();
        assert_eq!(prompt.token_count, expected, "preceding tail {tail}");
        assert!(step.work.iter().map(|w| w.token_count).sum::<usize>() <= 2048);
    }
}

#[test]
fn single_private_replay_and_unknown_geometry_keep_the_existing_partial_grant() {
    for geometry in [
        None,
        Some(PrefillGeometry {
            width: 2048,
            narrow_width: 64,
        }),
        Some(PrefillGeometry {
            width: 0,
            narrow_width: 0,
        }),
        Some(PrefillGeometry {
            width: 1024,
            narrow_width: 1024,
        }),
    ] {
        let result = replay_geometry_step(225, geometry, 2048);
        let step = result.step.as_ref().unwrap();
        assert_eq!(
            step.work
                .iter()
                .find(|w| w.request.get() == 17)
                .unwrap()
                .token_count,
            1808
        );
    }
}

#[test]
fn replay_geometry_cannot_starve_prompt_work_when_only_decodes_occupy_a_round() {
    let geometry = Some(PrefillGeometry {
        width: 1024,
        narrow_width: 64,
    });
    // Only16 tokens remain after the15 ready decodes. A shape hint cannot turn this
    // explicitly tiny budget into decode-only execution until every decode finishes.
    let result = replay_geometry_step(0, geometry, 31);
    let step = result.step.as_ref().unwrap();
    assert_eq!(
        step.work
            .iter()
            .find(|w| w.request.get() == 17)
            .unwrap()
            .token_count,
        16
    );
    assert_eq!(step.work.iter().map(|w| w.token_count).sum::<usize>(), 31);
}

#[test]
fn replay_geometry_keeps_singleton_progress_at_every_subgraph_budget() {
    let mut prompt = work(1, ExecutionRole::Prefill, 4096);
    prompt.cost_query.prefill_geometry = Some(PrefillGeometry {
        width: 1024,
        narrow_width: 64,
    });
    let mut r = resources();
    r.gpu_budget_us = 1_000_000;
    r.scheduler.prefill_chunk_tokens = 2048;
    for budget in [1, 2, 8, 31, 64, 511, 1023] {
        r.max_num_batched_tokens = budget;
        let result = plan(&[prompt.clone()], &r, 0);
        let step = result.step.as_ref().unwrap();
        assert_eq!(step.work.len(), 1);
        assert_eq!(step.work[0].token_count, budget);
    }
}

#[test]
fn an_intervening_prompt_cannot_reextend_an_aligned_grant_into_a_wide_tail() {
    let mut r = resources();
    r.max_num_seqs = 32;
    r.max_num_batched_tokens = 2048;
    r.gpu_budget_us = 1_000_000;
    r.scheduler.prefill_chunk_tokens = 2048;
    let mut items: Vec<_> = (1..=15)
        .map(|id| work(id, ExecutionRole::Decode, 1))
        .collect();
    items.push(work(16, ExecutionRole::Prefill, 64));
    let mut prompt = work(17, ExecutionRole::Prefill, 4096);
    prompt.cost_query.prefill_geometry = Some(PrefillGeometry {
        width: 1024,
        narrow_width: 64,
    });
    items.push(prompt);
    items.push(work(18, ExecutionRole::Prefill, 900));
    let result = plan(&items, &r, 0);
    let step = result.step.as_ref().unwrap();
    assert_eq!(
        step.work
            .iter()
            .find(|w| w.request.get() == 17)
            .unwrap()
            .token_count,
        1024
    );
    assert_eq!(
        step.work
            .iter()
            .find(|w| w.request.get() == 18)
            .unwrap()
            .token_count,
        900
    );
    assert_eq!(
        step.work
            .iter()
            .filter(|w| w.role == ExecutionRole::Decode)
            .count(),
        15
    );
    assert_eq!(step.work.iter().map(|w| w.token_count).sum::<usize>(), 2003);
}

#[test]
fn resident_geometry_preserves_budget_progress_and_decode_protection_across_rounds() {
    let mut seed = 73usize;
    let mut next = |bound: usize| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed % bound
    };
    for _ in 0..512 {
        let decode_count = next(16);
        let decode_rounds = next(4);
        let budget = next(4096) + 1;
        let width = [32, 64, 128, 256, 512, 1024][next(6)];
        let narrow_width = [0, 1, 8, 16, 32, 64][next(6)].min(width - 1);
        let mut pending = vec![
            work(101, ExecutionRole::Prefill, next(8192) + 1),
            work(102, ExecutionRole::Prefill, next(8192) + 1),
        ];
        let original = pending
            .iter()
            .map(|w| w.remaining_tokens)
            .collect::<Vec<_>>();
        for item in &mut pending {
            item.cost_query.prefill_geometry = Some(PrefillGeometry {
                width,
                narrow_width,
            });
        }
        let mut totals = [0, 0];
        let mut r = resources();
        r.max_num_seqs = 32;
        r.max_num_batched_tokens = budget;
        r.gpu_budget_us = 1_000_000;
        r.scheduler.prefill_chunk_tokens = 2048;
        // Once finite decode work finishes, at least one prompt token must progress per round.
        for round in 0..=original.iter().sum::<usize>() + decode_rounds {
            if pending.iter().all(|w| w.remaining_tokens == 0) {
                break;
            }
            let mut ready: Vec<_> = if round < decode_rounds {
                (1..=decode_count)
                    .map(|id| work(u64::try_from(id).unwrap(), ExecutionRole::Decode, 1))
                    .collect()
            } else {
                Vec::new()
            };
            ready.extend(pending.iter().filter(|w| w.remaining_tokens > 0).cloned());
            let result = plan(&ready, &r, 0);
            let step = result.step.as_ref().unwrap();
            assert!(step.work.iter().map(|w| w.token_count).sum::<usize>() <= budget);
            let decodes = step
                .work
                .iter()
                .filter(|w| w.role == ExecutionRole::Decode)
                .count();
            assert_eq!(
                decodes,
                if round < decode_rounds {
                    decode_count.min(budget)
                } else {
                    0
                }
            );
            let mut selected_ids = BTreeSet::new();
            for selected in &step.work {
                assert!(selected_ids.insert(selected.request));
                if selected.role != ExecutionRole::Prefill {
                    continue;
                }
                let index = pending
                    .iter()
                    .position(|w| w.request == selected.request)
                    .unwrap();
                let item = &mut pending[index];
                let chunk = item.remaining_tokens.min(r.scheduler.prefill_chunk_tokens);
                assert!(selected.token_count > 0 && selected.token_count <= chunk);
                if chunk <= budget && selected.token_count < chunk {
                    assert!(selected.token_count >= chunk.div_ceil(2));
                }
                item.remaining_tokens -= selected.token_count;
                totals[index] += selected.token_count;
                item.cost_query.context_tokens += selected.token_count;
            }
        }
        assert!(pending.iter().all(|w| w.remaining_tokens == 0));
        assert_eq!(totals.as_slice(), original.as_slice());
    }
}

#[test]
fn native_short_prompt_does_not_leave_a_single_token_tail_behind_another_prompt() {
    for native in [false, true] {
        let mut r = resources();
        r.max_num_batched_tokens = 1024;
        r.scheduler.prefill_chunk_tokens = 1024;
        r.gpu_budget_us = 1_000_000;
        let mut first = work(1, ExecutionRole::Prefill, 897);
        first.deadline_us = Some(1);
        let mut short = work(2, ExecutionRole::Prefill, 128);
        if native {
            short.cost_query.prefill_geometry = Some(PrefillGeometry {
                width: 1024,
                narrow_width: 64,
            });
        }
        let filler = work(3, ExecutionRole::Prefill, 64);
        let decision = plan(&[first, short.clone(), filler], &r, 0);
        let step = decision.step.unwrap();
        assert_eq!(
            step.work
                .iter()
                .find(|item| item.request.get() == 1)
                .unwrap()
                .token_count,
            897
        );
        let selected = step.work.iter().find(|item| item.request.get() == 2);
        if native {
            assert!(selected.is_none());
            assert_eq!(
                step.work
                    .iter()
                    .find(|item| item.request.get() == 3)
                    .unwrap()
                    .token_count,
                64
            );
            let next = plan(&[short], &r, 0).step.unwrap();
            assert_eq!(next.work[0].token_count, 128);
        } else {
            assert_eq!(selected.unwrap().token_count, 127);
        }
        assert!(step.work.iter().map(|item| item.token_count).sum::<usize>() <= 1024);
    }
}
