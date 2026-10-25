//! Non-blocking steady-state pooled decode; the backend retains compact commit records.
use super::{Completion, CudaBackend, CudaTicket, profiling::StepProfile};
use crate::resident::SlotLane;
use infer_core::{Error, RequestId, Result, StateId};
use infer_ir::{ExecutionInput, ExecutionTask, ModelOutput, OutputReadout, TaskOutput};
use std::sync::Arc;

struct DecodeTask {
    request: RequestId,
    state: StateId,
    position: usize,
    token: u32,
}

#[derive(Default)]
pub(super) struct DecodeFlight {
    tasks: Vec<DecodeTask>,
    lanes: Vec<SlotLane>,
    profile: Option<StepProfile>,
}

impl DecodeFlight {
    pub(super) const fn is_pending(&self) -> bool {
        !self.tasks.is_empty()
    }
}

impl CudaBackend {
    pub(super) fn can_submit_decode(&self, tasks: &[ExecutionTask]) -> bool {
        self.slots
            .as_ref()
            .is_some_and(|pool| pool.verify().is_none())
            && tasks.iter().all(|task| {
                self.states.get(&task.state).is_some_and(|state| {
                    state.readout == OutputReadout::Logits
                        && state.speculation.is_none()
                        && state.slot.as_ref().is_some_and(|lease| lease.batched)
                        && matches!(task.tokens, ExecutionInput::Decode { .. })
                })
            })
    }

    pub(super) fn submit_decode(&mut self, tasks: &[ExecutionTask]) -> Result<CudaTicket> {
        self.decode.tasks.clear();
        self.decode.lanes.clear();
        for task in tasks {
            let state = self
                .states
                .get(&task.state)
                .ok_or_else(|| Error::invariant("decode state"))?;
            let lease = state
                .slot
                .as_ref()
                .ok_or_else(|| Error::invariant("decode lease"))?;
            let ExecutionInput::Decode { token, position } = task.tokens else {
                return Err(Error::invariant("decode input"));
            };
            self.decode
                .lanes
                .push((lease.slot, token, position, position));
            self.decode.tasks.push(DecodeTask {
                request: task.request,
                state: task.state,
                position,
                token,
            });
        }
        self.decode.profile = StepProfile::start(self, tasks);
        let pool = self
            .slots
            .as_mut()
            .ok_or_else(|| Error::invariant("decode pool"))?;
        self.busy = true;
        let result = pool.submit_lanes(
            self.loaded.device(),
            &self.decode.lanes,
            tasks.iter().all(|task| task.plain_greedy),
        );
        let mut ticket = CudaTicket {
            owner: Arc::clone(&self.owner),
            completion: Completion::PendingDecode,
            drain_required: false,
        };
        if let Err(error) = result {
            self.poison_decode();
            ticket.completion = Completion::FailedDecode(error);
            ticket.drain_required = true;
        }
        Ok(ticket)
    }

    pub(super) fn poll_decode(&mut self) -> Result<Option<Vec<TaskOutput>>> {
        if !self.decode.is_pending() {
            return Err(Error::invariant("decode tasks missing"));
        }
        let Some(readout) = self
            .slots
            .as_mut()
            .ok_or_else(|| Error::invariant("decode pool"))?
            .poll_lanes()?
        else {
            return Ok(None);
        };
        if readout.rows.len() != self.decode.tasks.len()
            || readout
                .tokens
                .as_ref()
                .is_some_and(|tokens| tokens.len() != self.decode.tasks.len())
        {
            return Err(Error::invariant("decode output count"));
        }
        let mut outputs = Vec::with_capacity(self.decode.tasks.len());
        for (index, (task, (_, logits))) in self.decode.tasks.iter().zip(readout.rows).enumerate() {
            let next_token = readout
                .tokens
                .as_ref()
                .map(|tokens| {
                    tokens[index].ok_or_else(|| Error::invariant("decode selected token"))
                })
                .transpose()?;
            let state = self
                .states
                .get_mut(&task.state)
                .ok_or_else(|| Error::invariant("decode state"))?;
            ExecutionInput::Decode {
                position: task.position,
                token: task.token,
            }
            .commit(&mut state.history)?;
            let lease = state
                .slot
                .as_mut()
                .ok_or_else(|| Error::invariant("decode lease"))?;
            lease.position = state.history.len();
            outputs.push(TaskOutput {
                request: task.request,
                output: ModelOutput {
                    next_token,
                    logits,
                    hidden: Vec::new(),
                    tokens: Vec::new(),
                },
            });
        }
        self.decode.tasks.clear();
        if let Some(profile) = self.decode.profile.take() {
            profile.finish(self, Some(&outputs), Vec::new());
        }
        Ok(Some(outputs))
    }

    pub(super) fn poison_decode(&mut self) {
        for task in &self.decode.tasks {
            if let Some(state) = self.states.get_mut(&task.state) {
                state.poisoned = true;
            }
        }
        self.decode.tasks.clear();
        if let Some(profile) = self.decode.profile.take() {
            profile.finish(self, None, Vec::new());
        }
        // The ticket retains the failure until drain succeeds. Publishing fatal through
        // maintenance before then would prevent a direct runtime from polling its ticket.
    }
}
