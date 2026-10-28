use super::{CudaExecutor, Sequence};
use crate::resident::DeviceProgram;
use crate::resident::dflash2::Engine;
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::OutputReadout;
use std::collections::BTreeSet;

impl CudaExecutor {
    pub(super) fn available(&self) -> Result<u64> {
        self.state_funding().map(|(free, _)| free)
    }

    pub(super) fn capacity_budget(&self) -> Result<u64> {
        if self.slots.is_some() {
            return Ok(self.budget);
        }
        self.state_funding().map(|(_, capacity)| capacity)
    }

    /// Price shared prompt storage before publishing private-state capacity.
    fn state_funding(&self) -> Result<(u64, u64)> {
        let active = self.active_budget()?;
        let reclaimable = self
            .pool
            .reclaimable_bytes
            .saturating_add(self.prefix.bytes());
        let scratch = if self.slots.is_none() {
            self.loaded.pending_prompt_scratch_bytes(true)?
        } else {
            0
        };
        let (free, capacity) = private_funding(
            self.budget,
            active,
            self.physical_available()?,
            reclaimable,
            scratch,
        );
        // Pooled homes already own their workspace; their capacity is the prepaid pool.
        Ok((
            free,
            if self.slots.is_some() {
                self.budget
            } else {
                capacity
            },
        ))
    }
    pub(super) fn physical_available(&self) -> Result<u64> {
        Ok(self
            .loaded
            .device()
            .allocatable_memory_bytes()?
            .saturating_sub(self.loaded.profile().device_headroom_bytes()))
    }
    /// Prefill width the state's captured program will use.
    ///
    /// A pooled state must keep its wide prefill path: silently downgrading chunked prefill to
    /// the verification width costs several times the prefill latency on every reused request.
    /// # Errors
    /// Rejects unknown state ids.
    pub fn prefill_width_for(&self, state: StateId) -> Result<usize> {
        self.prefill_geometry_for(state)
            .map(|geometry| geometry.width)
    }

    /// Actual prompt and tail shapes of the state's private or shared worker.
    /// # Errors
    /// Rejects unknown state ids.
    pub fn prefill_geometry_for(&self, state: StateId) -> Result<infer_ir::PrefillGeometry> {
        self.states
            .get(&state)
            .map(|sequence| {
                if sequence.home.is_some()
                    && let Some(worker) = &self.worker
                {
                    worker.program.prefill_geometry()
                } else {
                    sequence.program.prefill_geometry()
                }
            })
            .ok_or_else(|| Error::invalid("unknown CUDA state"))
    }

    /// Resident homes use prepaid pool storage or wait for a home. Every private
    /// sequence needs a funded verification graph: it can lose the slot/block race
    /// and must remain executable without capturing unquoted storage in decode.
    pub(super) fn private_verification_for(
        &self,
        capacity: usize,
        readout: OutputReadout,
        allow_speculation: bool,
    ) -> bool {
        !self.home_eligible(capacity, readout, allow_speculation)
    }

    /// Release a prewarmed pool only when no live sequence relies on it, the request
    /// cannot use it, and ordinary cache reclamation cannot satisfy the quote.
    pub(super) fn prepare_reservation_for(
        &mut self,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        let Some(pool) = &self.slots else {
            return Ok(());
        };
        if !self.states.is_empty()
            || (readout == OutputReadout::Logits && capacity <= pool.capacity())
            || pool.free_slots() != pool.width()
            || self
                .draft_slots
                .as_ref()
                .is_some_and(|draft| draft.free_slots() != draft.width())
        {
            return Ok(());
        }
        let needed = self.reservation_budget(capacity, readout, true)?;
        let scratch = self.loaded.pending_prompt_scratch_bytes(true)?;
        let reclaimable = self
            .pool
            .reclaimable_bytes
            .saturating_add(self.prefix.bytes());
        let physical = self.physical_available()?.saturating_add(reclaimable);
        if needed > self.budget
            || (needed <= self.available()? && needed.saturating_add(scratch) <= physical)
        {
            return Ok(());
        }
        self.idle()?;
        let pool_width = pool.width();
        let pool_capacity = pool.capacity();
        let before = self.physical_available()?;
        self.worker = None;
        self.homes.clear();
        if let Some(engine) = self.dflash.as_mut() {
            engine.detach_pool();
        }
        drop(self.draft_slots.take());
        drop(self.slots.take());
        self.pool_quote = None;
        // Subsequent solo requests remain private; actual concurrent retries may
        // rebuild a funded shared pool without retaining the old bucket geometry.
        self.slots_disabled = false;
        if let Err(error) = self.loaded.device().reclaim_barrier() {
            self.fatal = Some(error.clone());
            return Err(error);
        }
        self.loaded.reclaim_idle_prompt_scratch()?;
        let after = self.physical_available()?;
        tracing::info!(
            target: "infer::cuda_admission",
            capacity,
            needed_bytes = needed,
            before_bytes = before,
            after_bytes = after,
            pool_width,
            pool_capacity,
            "reclaimed idle incompatible CUDA slot pools before admission"
        );
        Ok(())
    }

    pub(super) fn reserve(
        &mut self,
        id: StateId,
        capacity: usize,
        readout: OutputReadout,
        allow_speculation: bool,
    ) -> Result<()> {
        self.reserve_inner(id, capacity, readout, allow_speculation, true, None)
    }

    /// Media prefill owns external per-lane embeddings; a slot/home program is not a valid
    /// substitute for the private prompt program. Reserve the private path directly.
    pub(super) fn reserve_media(
        &mut self,
        id: StateId,
        capacity: usize,
        readout: OutputReadout,
        media: &infer_ir::MediaPlan,
    ) -> Result<()> {
        self.reserve_inner(id, capacity, readout, false, false, Some(media))
    }

    fn reserve_inner(
        &mut self,
        id: StateId,
        capacity: usize,
        readout: OutputReadout,
        allow_speculation: bool,
        allow_home: bool,
        media: Option<&infer_ir::MediaPlan>,
    ) -> Result<()> {
        self.idle()?;
        if self.states.contains_key(&id) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "CUDA state already reserved",
            ));
        }
        if allow_home {
            self.resolve_request_pool(id, capacity, readout)?;
        }
        self.prepare_reservation_for(capacity, readout)?;
        if allow_home && self.reserve_home(id, capacity, readout, allow_speculation)? {
            return Ok(());
        }
        let private_verification = if allow_home {
            self.private_verification_for(capacity, readout, allow_speculation)
        } else {
            true
        };
        let media_bytes = media.map_or(Ok(0), |plan| {
            self.loaded
                .vision()
                .map_or(Ok(0), |vision| vision.footprint(plan))
        })?;
        let budget = self
            .reservation_budget(capacity, readout, private_verification)?
            .checked_add(media_bytes)
            .ok_or_else(|| Error::invalid("media reservation footprint overflow"))?;
        if self.states.len() >= self.maximum_states
            || self
                .active_budget()?
                .checked_add(budget)
                .is_none_or(|n| n > self.budget)
        {
            return Err(self.reservation_rejection(id, capacity, private_verification, budget)?);
        }
        if let Some(mut state) = self.pool.take(capacity, readout) {
            state.draft = self.lease_draft(allow_speculation);
            return self.reuse_pooled(id, state, capacity, budget, private_verification);
        }
        let scratch = self
            .loaded
            .pending_prompt_scratch_bytes(private_verification)?;
        self.make_room(budget, scratch)?;
        let program = match if private_verification {
            self.loaded.sequence(capacity)
        } else {
            self.loaded.sequence_pooled(capacity)
        } {
            Ok(program) => program,
            Err(error) => {
                if let Err(drain_error) = self.loaded.device().reclaim_barrier() {
                    self.fatal = Some(drain_error);
                }
                return Err(error);
            }
        };
        let speculation = if allow_speculation {
            match self.loaded.draft(capacity) {
                Ok(draft) => {
                    draft.map(|program| super::Speculation::new(program, self.loaded.mtp_depth()))
                }
                Err(error) => {
                    if let Err(drain_error) = self.loaded.device().reclaim_barrier() {
                        self.fatal = Some(drain_error);
                    }
                    return Err(error);
                }
            }
        } else {
            None
        };
        self.pool.allocations = self.pool.allocations.saturating_add(1);
        let draft = self.lease_draft(allow_speculation);
        self.states.insert(
            id,
            Sequence {
                draft,
                home: None,
                program,
                speculation,
                capacity,
                readout,
                budget,
                history: Vec::new(),
                hidden: Vec::new(),
                media: None,
                rotary_base: 0,
                media_prompt_len: 0,
                poisoned: false,
                slot: None,
                verification_priced: private_verification,
            },
        );
        Ok(())
    }

    fn reservation_rejection(
        &self,
        id: StateId,
        capacity: usize,
        private_verification: bool,
        budget: u64,
    ) -> Result<Error> {
        let active = self.active_budget()?;
        let (pool_slots, pool_free_slots) = self
            .slots
            .as_ref()
            .map_or((0, 0), |pool| (pool.width(), pool.free_slots()));
        tracing::debug!(
            target: "infer::cuda_admission",
            phase = "reservation_accounting",
            state = id.get(),
            capacity,
            private_verification,
            needed_bytes = budget,
            active_bytes = active,
            budget_bytes = self.budget,
            active_states = self.states.len(),
            maximum_states = self.maximum_states,
            pool_slots,
            pool_free_slots,
            "CUDA resident reservation rejected"
        );
        Ok(Error::new(
            ErrorCode::Capacity,
            format!(
                "CUDA resident state admission budget exhausted: needed={budget}, active={}, budget={}, states={}, maximum={}",
                active,
                self.budget,
                self.states.len(),
                self.maximum_states
            ),
        ))
    }

    /// Re-admit a pooled sequence shell under a freshly charged budget: reset program and
    /// speculation state, realign verification pricing with the new charge, and clear the
    /// shell's poison flag. A reset failure poisons the shell back into the pool; when even
    /// the drain after it fails, the backend records the drain as fatal.
    fn reuse_pooled(
        &mut self,
        id: StateId,
        mut state: Sequence,
        capacity: usize,
        budget: u64,
        private_verification: bool,
    ) -> Result<()> {
        state.poisoned = true;
        let reset = state.program.reset().and_then(|()| {
            state
                .speculation
                .as_mut()
                .map_or(Ok(()), super::Speculation::reset)
        });
        if let Err(error) = reset {
            if let (Some(draft), Some(engine)) = (state.draft.take(), self.dflash.as_mut()) {
                engine.release(&draft.cursor);
            }
            self.pool.retain(state);
            if let Err(drain_error) = self.loaded.device().drain() {
                self.fatal = Some(drain_error);
            }
            return Err(error);
        }
        if state.slot.is_some() {
            return Err(Error::invariant("pooled CUDA state holds a slot lease"));
        }
        state.capacity = capacity;
        // The reused program's verification storage must match the price just charged:
        // a serial-priced program may still hold one (a later slot lease credits its
        // release), while the pooled price never included one.
        if !private_verification {
            state.program.forfeit_verification();
        }
        state.verification_priced = private_verification;
        state.budget = budget;
        state.history.clear();
        state.hidden.clear();
        state.media = None;
        state.rotary_base = 0;
        state.media_prompt_len = 0;
        state.poisoned = false;
        self.pool.reuses = self.pool.reuses.saturating_add(1);
        self.states.insert(id, state);
        Ok(())
    }
    pub(super) fn reset(&mut self, id: StateId) -> Result<()> {
        self.idle()?;
        let restore = self
            .states
            .get(&id)
            .filter(|s| s.slot.is_some() && s.verification_priced)
            .map_or(0, |s| s.program.released_verification_bytes());
        if restore > 0 {
            self.make_room(restore, 0)?;
        }
        let state = self
            .states
            .get_mut(&id)
            .ok_or_else(|| Error::invalid("unknown CUDA state"))?;
        state.poisoned = true;
        state.budget = state
            .budget
            .checked_add(restore)
            .ok_or_else(|| Error::invariant("private verification reservation overflow"))?;
        state.program.reset()?;
        if let Some(speculation) = state.speculation.as_mut() {
            speculation.reset()?;
        }
        if let Some(draft) = state.draft.as_mut() {
            draft.reset();
        }
        // The slot keeps no rollback state: the next bind overwrites its covered prefix. A
        // slot-resident sequence keeps its slot leased; only the pooled replay forgets it.
        if let Some(lease) = state.slot.take()
            && let Some(pool) = self.slots.as_mut()
        {
            if state.home == Some(lease.slot) {
                pool.discard(lease.slot);
            } else {
                pool.release(lease.slot);
            }
        }
        // Reset has restored the original private charge and retired the slot lease.
        // Prepare before returning to compute; decode must never capture a graph.
        state.prepare_verification(&self.loaded)?;
        state.history.clear();
        state.hidden.clear();
        state.media = None;
        state.rotary_base = 0;
        state.media_prompt_len = 0;
        state.poisoned = false;
        Ok(())
    }
    pub(super) fn release(&mut self, id: StateId) -> Result<()> {
        self.idle()?;
        self.loaded.device().drain()?;
        let mut state = self
            .states
            .remove(&id)
            .ok_or_else(|| Error::invalid("unknown CUDA state"))?;
        if let (Some(draft), Some(engine)) = (state.draft.take(), self.dflash.as_mut()) {
            engine.release(&draft.cursor);
        }
        if let Some(lease) = state.slot.take()
            && let Some(pool) = self.slots.as_mut()
        {
            pool.release(lease.slot);
        }
        if let Some(slot) = state.home {
            if let Some(pool) = self.slots.as_mut() {
                pool.release(slot);
            }
            if let Some(home) = self.homes.get_mut(slot) {
                *home = Some(super::Home {
                    program: state.program,
                    draft: state.speculation.map(|speculation| speculation.program),
                });
            }
            return Ok(());
        }
        if !state.poisoned {
            state.history.clear();
            state.hidden.clear();
            self.pool.retain(state);
        }
        Ok(())
    }
    /// Lease a `DFlash2` ring to a speculating sequence; without a free ring it decodes plainly.
    fn lease_draft(&mut self, allow_speculation: bool) -> Option<super::DraftState> {
        allow_speculation
            .then(|| self.dflash.as_mut().and_then(Engine::acquire))
            .flatten()
            .map(super::DraftState::new)
    }

    /// Whether a request is served by slot-resident programs when one is free.
    fn home_eligible(
        &self,
        capacity: usize,
        readout: OutputReadout,
        allow_speculation: bool,
    ) -> bool {
        readout == OutputReadout::Logits
            && self
                .slots
                .as_ref()
                .is_some_and(|pool| capacity <= pool.capacity())
            && self.worker.is_some()
            && (self.states.values().any(|state| state.home.is_some())
                || self.homes.iter().flatten().any(|home| {
                    home.draft.is_none() && !allow_speculation
                        || home.draft.is_some() && allow_speculation
                        || self.loaded.mtp_depth() == 0
                }))
    }

    /// Admit a pooled request onto a free slot's resident programs: it prefills straight into
    /// the slot and then decodes from it, so admission captures nothing and the request owns no
    /// private state. False when no such slot is free; the caller admits a private program.
    fn reserve_home(
        &mut self,
        id: StateId,
        capacity: usize,
        readout: OutputReadout,
        allow_speculation: bool,
    ) -> Result<bool> {
        // A request that fits a slot waits for one rather than capturing a private program:
        // that capture holds the engine thread for seconds, and the slot is faster once free.
        let busy = if self.home_eligible(capacity, readout, allow_speculation) {
            Err(Error::new(
                ErrorCode::Capacity,
                "every slot-resident program is in use",
            ))
        } else {
            Ok(false)
        };
        let Some(pool) = self.slots.as_mut() else {
            return Ok(false);
        };
        if readout != OutputReadout::Logits
            || capacity > pool.capacity()
            || self.states.len() >= self.maximum_states
        {
            return Ok(false);
        }
        let Some(slot) = (0..self.homes.len()).find(|&slot| {
            self.homes[slot].as_ref().is_some_and(|home| {
                home.draft.is_none() && !allow_speculation
                    || home.draft.is_some() && allow_speculation
                    || self.loaded.mtp_depth() == 0
            })
        }) else {
            return busy;
        };
        if !pool.take_slot(slot) {
            return busy;
        }
        match pool.assign(slot, capacity, self.loaded.device()) {
            Ok(true) => {}
            Ok(false) => {
                pool.release(slot);
                return busy;
            }
            Err(error) => {
                pool.release(slot);
                return Err(error);
            }
        }
        let Some(mut home) = self.homes[slot].take() else {
            return Ok(false);
        };
        let reset = home
            .program
            .reset()
            .and_then(|()| home.draft.as_mut().map_or(Ok(()), DeviceProgram::reset));
        if let Err(error) = reset {
            pool.release(slot);
            self.homes[slot] = Some(home);
            return Err(error);
        }
        let depth = self.loaded.mtp_depth();
        let draft = self.lease_draft(allow_speculation);
        self.states.insert(
            id,
            Sequence {
                draft,
                home: Some(slot),
                program: home.program,
                speculation: home
                    .draft
                    .map(|program| super::Speculation::new(program, depth)),
                capacity,
                readout,
                // The pool paid for the slot's storage and programs when it was built.
                budget: 0,
                history: Vec::new(),
                hidden: Vec::new(),
                media: None,
                rotary_base: 0,
                media_prompt_len: 0,
                poisoned: false,
                slot: None,
                verification_priced: false,
            },
        );
        Ok(true)
    }

    pub(super) fn ownership(&self, states: &[(StateId, usize, usize)]) -> Result<()> {
        if states.len() != self.states.len()
            || states.iter().map(|s| s.0).collect::<BTreeSet<_>>().len() != states.len()
            || states.iter().any(|(id, capacity, cursor)| {
                !self.states.get(id).is_some_and(|s| {
                    !s.poisoned
                        && s.capacity == *capacity
                        && s.history.len() == *cursor
                        && self.cursor_matches(s, *cursor)
                })
            })
        {
            return Err(Error::invariant("CUDA state ownership mismatch"));
        }
        Ok(())
    }

    /// Device cursor agreement: slot-leased sequences track the pool cursor, unbound
    /// sequences track their own program's position.
    fn cursor_matches(&self, state: &Sequence, cursor: usize) -> bool {
        state.slot.as_ref().map_or_else(
            || state.program.position() == cursor,
            |lease| {
                lease.position == cursor
                    && self
                        .slots
                        .as_ref()
                        .is_some_and(|pool| pool.cursors().get(lease.slot) == Some(&cursor))
            },
        )
    }
}

/// Shared scratch consumes physical memory once, independently of per-state admission.
fn private_funding(
    budget: u64,
    active: u64,
    physical: u64,
    reclaimable: u64,
    scratch: u64,
) -> (u64, u64) {
    let headroom = physical.saturating_add(reclaimable).saturating_sub(scratch);
    (
        budget.saturating_sub(active).min(headroom),
        budget.min(active.saturating_add(headroom)),
    )
}

#[cfg(test)]
#[path = "../../tests/unit/executor_state.rs"]
mod tests;
