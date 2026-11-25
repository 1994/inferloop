//! A host-only rank group: protocol doubles presented to the engine as one backend.
//!
//! It states the group protocol a tensor-parallel backend has to keep
//! (`docs/plans/engineering/parallelism.md`, section 5.1) so the engine-facing contract is pinned
//! before any device code exists: a group ticket completes only after every rank's fence, a
//! reservation is all-or-nothing, a reused prefix is the shortest any rank holds, and a failing
//! rank fails the group only once every other rank has fenced its work. It is a test double, not
//! a product backend.
use crate::fixtures::ProtocolBackend;
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::{
    BackendFormats, DataflowGraph, DeviceCapabilities, ExecutionProgram, ExecutionTask,
    FinishedState, ModelIr, OutputReadout, StepPlan, TaskOutput,
};
use infer_spi::BackendProvider;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Holds and faults a scene injects into one rank.
#[derive(Default)]
pub struct RankControl {
    /// The rank's work stays in flight while set.
    pub hold_fence: AtomicBool,
    /// The rank refuses its next reservations while set.
    pub fail_reserve: AtomicBool,
    /// The rank reports a device failure for its fenced work while set.
    pub fail_poll: AtomicBool,
}

/// One submission across every rank.
pub struct GroupTicket {
    /// Each rank's ticket until that rank fences.
    pending: Vec<Option<<ProtocolBackend as BackendProvider>::Ticket>>,
    /// Rank 0's outputs, which every other rank must match.
    primary: Option<Vec<TaskOutput>>,
    error: Option<Error>,
}

pub struct RankGroup {
    ranks: Vec<ProtocolBackend>,
    controls: Vec<Arc<RankControl>>,
    resources: infer_spi::ResourcePool,
    identity: String,
    in_flight: Arc<AtomicUsize>,
}

impl RankGroup {
    /// A group over `ranks`, each with its own control.
    ///
    /// # Errors
    /// Rejects an empty group or an unbounded resource pool.
    pub fn new(ranks: Vec<ProtocolBackend>) -> Result<Self> {
        if ranks.is_empty() {
            return Err(Error::invalid("a rank group needs a rank"));
        }
        Ok(Self {
            identity: format!("rank-group/{}", ranks.len()),
            controls: ranks.iter().map(|_| Arc::default()).collect(),
            resources: infer_spi::ResourcePool::new(64)?,
            in_flight: Arc::new(AtomicUsize::new(0)),
            ranks,
        })
    }

    /// The control of rank `index`.
    #[must_use]
    pub fn control(&self, index: usize) -> Arc<RankControl> {
        Arc::clone(&self.controls[index])
    }

    /// Rank work submitted and not yet fenced, across the group.
    #[must_use]
    pub fn in_flight(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.in_flight)
    }

    /// Rank `index` itself.
    #[must_use]
    pub fn rank(&self, index: usize) -> &ProtocolBackend {
        &self.ranks[index]
    }

    /// Apply `operation` to every rank, keeping the first failure but reaching every rank.
    fn each(
        &mut self,
        mut operation: impl FnMut(&mut ProtocolBackend) -> Result<()>,
    ) -> Result<()> {
        let mut first = None;
        for rank in &mut self.ranks {
            if let Err(error) = operation(rank) {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Poll rank `index` once, recording its fenced result in `ticket`.
    fn poll_rank(&mut self, index: usize, ticket: &mut GroupTicket) {
        let control = &self.controls[index];
        if control.hold_fence.load(Ordering::Acquire) {
            return;
        }
        let Some(rank_ticket) = ticket.pending[index].as_mut() else {
            return;
        };
        let outputs = if control.fail_poll.load(Ordering::Acquire) {
            Err(Error::new(
                ErrorCode::Backend,
                format!("rank {index} failed"),
            ))
        } else {
            self.ranks[index].poll(rank_ticket)
        };
        let fenced = match outputs {
            Ok(None) => return,
            Ok(Some(outputs)) => Some(outputs),
            Err(error) => {
                ticket.error.get_or_insert(error);
                None
            }
        };
        ticket.pending[index] = None;
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
        match (fenced, &ticket.primary) {
            (Some(outputs), None) if index == 0 => ticket.primary = Some(outputs),
            (Some(outputs), Some(primary)) if !same_outputs(&outputs, primary) => {
                ticket.error.get_or_insert_with(|| {
                    Error::invariant(format!("rank {index} diverged from rank 0"))
                });
            }
            _ => {}
        }
    }
}

fn same_outputs(left: &[TaskOutput], right: &[TaskOutput]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.request == b.request && a.output.logits == b.output.logits)
}

impl BackendProvider for RankGroup {
    type Ticket = GroupTicket;
    fn identity(&self) -> &str {
        &self.identity
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.ranks[0].capabilities()
    }
    fn formats(&self) -> BackendFormats {
        self.ranks[0].formats()
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.ranks
            .iter()
            .try_for_each(|rank| rank.validate_program(model, program))
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<DataflowGraph> {
        self.ranks[0].execution_graph(model)
    }
    fn begin_resource(
        &mut self,
        command: infer_spi::ResourceCommand,
    ) -> Result<infer_spi::ResourceTicket> {
        let (ticket, reply) = self.resources.channel()?;
        let _ = reply.send(infer_spi::execute_resource(self, command));
        Ok(ticket)
    }
    /// All or nothing: a rank that cannot reserve releases what the ranks before it reserved.
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        for index in 0..self.ranks.len() {
            let reserved = if self.controls[index].fail_reserve.load(Ordering::Acquire) {
                Err(Error::new(
                    ErrorCode::Backend,
                    format!("rank {index} cannot reserve"),
                ))
            } else {
                self.ranks[index].reserve_state_for(state, capacity, readout)
            };
            if let Err(error) = reserved {
                for rank in &mut self.ranks[..index] {
                    rank.release_state(state)?;
                }
                return Err(error);
            }
        }
        Ok(())
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.each(|rank| rank.reset_state(state))
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.each(|rank| rank.release_state(state))
    }
    fn finish_state(&mut self, state: StateId, kept: FinishedState) -> Result<()> {
        self.each(|rank| rank.finish_state(state, kept))
    }
    /// Only a prefix every rank holds can be reused.
    fn reusable_prefix(&self, tokens: &[u32], maximum: usize) -> usize {
        self.ranks
            .iter()
            .map(|rank| rank.reusable_prefix(tokens, maximum))
            .min()
            .unwrap_or(0)
    }
    fn reuse_prefix(&mut self, state: StateId, tokens: &[u32], maximum: usize) -> Result<usize> {
        let common = self.reusable_prefix(tokens, maximum);
        for (index, rank) in self.ranks.iter_mut().enumerate() {
            if rank.reuse_prefix(state, tokens, common)? != common {
                return Err(Error::invariant(format!(
                    "rank {index} attached a different prefix"
                )));
            }
        }
        Ok(common)
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        let mut pending = Vec::with_capacity(self.ranks.len());
        for rank in &mut self.ranks {
            pending.push(Some(rank.submit_borrowed(program, step, &tasks)?));
            self.in_flight.fetch_add(1, Ordering::AcqRel);
        }
        Ok(GroupTicket {
            pending,
            primary: None,
            error: None,
        })
    }
    /// Complete only once every rank fenced; a failure is reported after the last fence.
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        for index in 0..self.ranks.len() {
            self.poll_rank(index, ticket);
        }
        if ticket.pending.iter().any(Option::is_some) {
            return Ok(None);
        }
        if let Some(error) = ticket.error.take() {
            return Err(error);
        }
        ticket
            .primary
            .take()
            .map(Some)
            .ok_or_else(|| Error::invariant("group completed without rank 0"))
    }
}
