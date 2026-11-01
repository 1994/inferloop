//! Batch draft steps across sequences, then return state to each request for verification.
use super::Sequence;
use crate::{device::CudaDevice, resident::slot_batch::SlotPool};
use infer_core::{Error, Result, StateId};
use infer_ir::{ExecutionInput, ExecutionTask};
use infer_workloads::{draft_budget, draft_may_propose};
use std::collections::BTreeMap;

pub(super) async fn propose(
    pool: &mut SlotPool,
    states: &mut BTreeMap<StateId, Sequence>,
    device: &CudaDevice,
    tasks: &[ExecutionTask],
    participants: &[(usize, usize)],
    width: usize,
) -> Result<Option<Vec<Vec<u32>>>> {
    if participants.len() < 2
        || participants.iter().any(|&(index, _)| {
            states
                .get(&tasks[index].state)
                .is_none_or(|state| !eligible(state, &tasks[index], width))
        })
    {
        return Ok(None);
    }
    let mut chains = Vec::with_capacity(participants.len());
    let mut hidden = Vec::with_capacity(participants.len());
    let mut counts = Vec::with_capacity(participants.len());
    for &(index, slot) in participants {
        let task = &tasks[index];
        let state = states
            .get_mut(&task.state)
            .ok_or_else(|| Error::invariant("draft state"))?;
        let ExecutionInput::Decode { token, position } = task.tokens else {
            return Err(Error::invariant("draft decode task"));
        };
        let spec = state
            .speculation
            .as_mut()
            .ok_or_else(|| Error::invariant("draft program"))?;
        spec.prompt_len.get_or_insert(position);
        pool.bind(
            slot,
            spec.program.states(),
            spec.program.fp8_states(),
            position - 1,
            device,
        )?;
        chains.push(vec![token]);
        hidden.push(spec.last_hidden.clone());
        counts.push(draft_budget(spec.depth, width, position, state.capacity));
    }
    let plain_greedy = participants
        .iter()
        .all(|&(index, _)| tasks[index].plain_greedy);
    validate_sampling(tasks, participants)?;
    for step in 0..counts.iter().copied().max().unwrap_or(0) {
        let active: Vec<_> = (0..participants.len())
            .filter(|&i| step < counts[i])
            .collect();
        let lanes: Vec<_> = active
            .iter()
            .map(|&i| {
                let (index, slot) = participants[i];
                let position = states[&tasks[index].state].history.len() + step;
                (slot, chains[i][step], position, position - 1)
            })
            .collect();
        let external: Vec<_> = active.iter().map(|&i| hidden[i].as_slice()).collect();
        let readout = pool
            .run_external_selected_async(device, &lanes, &external, plain_greedy)
            .await?;
        for (row, (&i, (next_hidden, logits))) in active.iter().zip(readout.rows).enumerate() {
            let (index, _) = participants[i];
            let task = &tasks[index];
            let state = states
                .get_mut(&task.state)
                .ok_or_else(|| Error::invariant("draft sampling state"))?;
            let spec = state
                .speculation
                .as_mut()
                .ok_or_else(|| Error::invariant("draft sampling program"))?;
            let token = readout.tokens.as_ref().map_or_else(
                || {
                    let prompt_len = spec
                        .prompt_len
                        .ok_or_else(|| Error::invariant("draft prompt length"))?;
                    let mut generated = state.history[prompt_len..].to_vec();
                    generated.extend_from_slice(&chains[i]);
                    infer_workloads::sample_with_history(
                        &logits,
                        task.sampling
                            .as_ref()
                            .ok_or_else(|| Error::invariant("draft sampling"))?,
                        task.request.get(),
                        state.history.len() + step + 1,
                        infer_workloads::SamplingHistory {
                            prompt: &state.history[..prompt_len],
                            generated: &generated,
                        },
                        &mut spec.scratch,
                    )
                },
                |tokens| tokens[row].ok_or_else(|| Error::invariant("draft selected token")),
            )?;
            chains[i].push(token);
            hidden[i] = next_hidden;
        }
    }
    Ok(Some(chains))
}

fn eligible(state: &Sequence, task: &ExecutionTask, width: usize) -> bool {
    let Some(spec) = &state.speculation else {
        return false;
    };
    let position = state.history.len();
    position > 0
        && position + 1 < state.capacity
        && spec.depth > 0
        && width > 1
        && spec.program.position() == position - 1
        && task
            .sampling
            .as_ref()
            .is_some_and(|sampling| draft_may_propose(sampling.temperature))
}

/// Restore accepted draft timelines using target hidden rows, including full acceptance.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the pooled execution transaction"
)]
pub(super) fn catch_up(
    pool: &mut SlotPool,
    states: &mut BTreeMap<StateId, Sequence>,
    device: &CudaDevice,
    tasks: &[ExecutionTask],
    participants: &[(usize, usize)],
    outputs: &[Option<infer_ir::TaskOutput>],
    rows: &[(Vec<f32>, Vec<f32>)],
    width: usize,
) -> Result<()> {
    let mut plans = Vec::with_capacity(participants.len());
    for &(index, slot) in participants {
        let ExecutionInput::Decode { position, .. } = tasks[index].tokens else {
            return Err(Error::invariant("draft catch-up decode"));
        };
        let accepted = &outputs[index]
            .as_ref()
            .ok_or_else(|| Error::invariant("draft catch-up output"))?
            .output
            .tokens;
        // Proposal KV beyond the submitted token was conditioned on draft hidden
        // rows. Acceptance validates tokens, not those hidden rows: refresh all of it.
        pool.rewind_external(slot, position)?;
        plans.push((index, slot, position, accepted.len()));
    }
    let steps = plans.iter().map(|p| p.3).max().unwrap_or(0);
    for step in 0..steps {
        let mut lanes = Vec::new();
        let mut hidden = Vec::new();
        for &(index, slot, position, count) in &plans {
            let offset = step;
            if offset >= count {
                continue;
            }
            let accepted = &outputs[index]
                .as_ref()
                .ok_or_else(|| Error::invariant("draft catch-up output"))?
                .output
                .tokens;
            lanes.push((
                slot,
                accepted[offset],
                position + offset + 1,
                position + offset,
            ));
            hidden.push(rows[slot * width + offset].0.as_slice());
        }
        pool.run_external_detached(device, &lanes, &hidden)?;
    }
    for &(index, slot) in participants {
        let spec = states
            .get_mut(&tasks[index].state)
            .and_then(|state| state.speculation.as_mut())
            .ok_or_else(|| Error::invariant("draft catch-up state"))?;
        pool.export_attention(slot, &mut spec.program, device)?;
    }
    Ok(())
}

fn validate_sampling(tasks: &[ExecutionTask], participants: &[(usize, usize)]) -> Result<()> {
    for &(index, _) in participants {
        tasks[index]
            .sampling
            .as_ref()
            .ok_or_else(|| Error::invariant("draft sampling"))?
            .validate()?;
    }
    Ok(())
}
