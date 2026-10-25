use super::{CachedMedia, CudaExecutor, ModelOutput, Sequence, SlotLease, Speculation, prefix};
use crate::constants::CB_DECODE_SLOTS;
use crate::constants::CB_SLOT_TOKENS;
use crate::constants::MIB;
use crate::constants::MTP_KV_OFFSET;
use crate::constants::PREFILL_LANES;
use crate::constants::WIDER_PREFILL_LANES;
use crate::device::CudaDevice;
use crate::loading::{LoadedModel, PooledServingQuote};
use crate::resident::slot_batch::SlotPool;
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::{ExecutionInput, ExecutionTask, MediaPlan, OutputReadout, TaskOutput};
use infer_models::mrope::ROTARY_AXES;
use infer_models::{PreparedPrompt, PromptImage, media_layout};
use std::collections::BTreeMap;
use std::mem::swap;
use std::mem::take;
use std::ops::Range;
/// Bound capture/replan work while adapting tensor quotes to observed driver storage.
const POOL_CAPTURE_ATTEMPTS: usize = 3;

/// Snapshot the KV rows a sequence has covered, when they form a reusable prompt prefix.
///
/// Returns `None` unless the sequence is a logits-readout generation that just consumed prefill
/// tokens at a cacheable granularity. Capture head-major KV rows and complete recurrent
/// and draft state so restoration resumes the same inference trajectory.
async fn cacheable_prefix(
    device: &CudaDevice,
    loaded: &LoadedModel,
    state: &mut Sequence,
    task: &ExecutionTask,
) -> Option<prefix::CachedPrefix> {
    if task.media.is_some()
        || state.poisoned
        || state.home.is_some()
        || state.readout != OutputReadout::Logits
        || !matches!(
            task.tokens,
            ExecutionInput::Full(_) | ExecutionInput::Prefill { .. }
        )
    {
        return None;
    }
    let covered = state.history.len();
    if covered == 0
        || covered > state.capacity
        || !covered.is_multiple_of(prefix::PREFIX_GRANULARITY)
    {
        return None;
    }
    let target = super::prefix_state::ProgramSnapshot::capture_async(
        device,
        &state.program,
        loaded.graph(),
        covered,
    )
    .await
    .ok()?;
    let draft = if let Some(spec) = &mut state.speculation {
        Some(super::prefix_state::DraftSnapshot {
            program: super::prefix_state::ProgramSnapshot::capture_async(
                device,
                &spec.program,
                loaded.draft_graph()?,
                covered.saturating_sub(MTP_KV_OFFSET),
            )
            .await
            .ok()?,
            hidden: spec.last_hidden.clone(),
        })
    } else {
        None
    };
    let bytes = target
        .bytes
        .saturating_add(draft.as_ref().map_or(0, |d| d.program.bytes));
    Some(prefix::CachedPrefix {
        tokens: state.history.clone(),
        value: prefix::Snapshot { target, draft },
        bytes,
    })
}

/// All storage needed before publishing a load-time pooled serving geometry.
struct PoolPlan {
    quote: PooledServingQuote,
}

impl CudaExecutor {
    pub(super) async fn execute(&mut self, tasks: &[ExecutionTask]) -> Result<Vec<TaskOutput>> {
        self.busy = true;
        let mut snapshots: Vec<prefix::CachedPrefix> = Vec::new();
        let result = self.run_grouped(tasks, &mut snapshots).await;
        let published = snapshots
            .into_iter()
            .try_for_each(|snapshot| self.prefix.publish(snapshot).map(drop));
        let result = result.and_then(|outputs| published.map(|()| outputs));
        if result.is_err() {
            for task in tasks {
                if let Some(state) = self.states.get_mut(&task.state) {
                    state.poisoned = true;
                }
            }
        }
        result
    }

    pub(super) fn prepare_execution(&mut self) -> Result<()> {
        self.ensure_slot_pool();
        for state in self.states.values_mut() {
            if !state.poisoned {
                state.prepare_verification(&self.loaded)?;
            }
        }
        Ok(())
    }

    /// Execute one step's tasks in order, replaying each consecutive run of eligible
    /// decode tasks through the shared slot graph and everything else serially.
    async fn run_grouped(
        &mut self,
        tasks: &[ExecutionTask],
        snapshots: &mut Vec<prefix::CachedPrefix>,
    ) -> Result<Vec<TaskOutput>> {
        let mut outputs: Vec<Option<TaskOutput>> = tasks.iter().map(|_| None).collect();
        let eligible: Vec<bool> = tasks
            .iter()
            .map(|task| self.decode_eligible(task))
            .collect();
        // Pool capture belongs to admission; submission only replays prepared graphs.
        let mut index = 0;
        while index < tasks.len() {
            if !eligible[index] {
                self.run_serial(tasks, index, &mut outputs, snapshots)
                    .await?;
                index += 1;
                continue;
            }
            let mut end = index + 1;
            while end < tasks.len() && eligible[end] {
                end += 1;
            }
            self.run_segment(tasks, index..end, &mut outputs, snapshots)
                .await?;
            index = end;
        }
        outputs
            .into_iter()
            .map(|output| output.ok_or_else(|| Error::invariant("CUDA task output missing")))
            .collect()
    }

    /// A task rides the slot graph when it is a decode at the sequence cursor, the
    /// sequence reads logits only, and it either already leases a
    /// slot or its KV fits the slot capacity. Sampling is irrelevant: the batched path
    /// leaves token decisions to the engine, exactly like a non-speculative serial step.
    fn decode_eligible(&self, task: &ExecutionTask) -> bool {
        let Some(state) = self.states.get(&task.state) else {
            return false;
        };
        if !matches!(task.tokens, ExecutionInput::Decode { position, .. } if position == state.history.len())
            || state.readout != OutputReadout::Logits
        {
            return false;
        }
        state.slot.as_ref().map_or_else(
            || {
                state.capacity
                    <= self
                        .slots
                        .as_ref()
                        .map_or(CB_SLOT_TOKENS, SlotPool::capacity)
            },
            |lease| lease.batched && lease.position == state.history.len(),
        )
    }

    /// Capture during admission only when two live requests can use pooled decode.
    /// Temporary funding failures remain retryable after reservation accounting changes.
    /// With a draft loaded the pool captures the pooled speculation graph instead: every
    /// sequence speculates from its first decode step, so a decode-only pool would sit
    /// unused while squeezing admission memory.
    fn ensure_slot_pool(&mut self) {
        if self.slots.is_some() || self.slots_disabled {
            return;
        }
        // Size shared KV storage from admitted requests, not the maximum context.
        // Long contexts outside this pool retain their private execution graph.
        let Some(capacity) = candidate_pool_capacity(
            self.states
                .values()
                .filter(|state| !state.poisoned)
                .map(|state| (state.capacity, state.readout)),
        ) else {
            return;
        };
        // Reserve the configured capture bucket, not the transient number of arrivals seen on
        // this tick. Otherwise a two-request first tick fixes the pool at two slots forever.
        let active = match self.active_budget() {
            Ok(bytes) => bytes,
            Err(error) => {
                self.pool_failure_reason = Some(error.to_string());
                return;
            }
        };
        if self.deferred_pool_plan == Some((capacity, active)) {
            return;
        }
        self.warm_slot_pool_at(capacity);
        let Some(pool) = self.slots.as_ref() else {
            // A live private request can temporarily consume the worker's budget.
            // Retry after reservation accounting changes, rather than on every decode
            // or disabling batching for the rest of a continuously busy server.
            self.deferred_pool_plan = Some((capacity, active));
            tracing::warn!(
                target: "infer::executor",
                capacity,
                last_failure = %self.pool_failure_reason.as_deref().unwrap_or("unknown"),
                "CUDA slot pool deferred; retaining private execution"
            );
            return;
        };
        let capacity = pool.capacity();
        for state in self.states.values_mut() {
            if state.capacity <= capacity
                && state.speculation.is_some()
                && state.readout == OutputReadout::Logits
            {
                state.program.discard_verification();
            }
        }
        if self.loaded.device().reclaim_barrier().is_err() {
            self.slots_disabled = true;
            self.clear_unbound_pool();
            tracing::warn!(
                target: "infer::executor",
                "CUDA slot pool disabled; reclaiming cached prefixes failed"
            );
        }
    }

    /// Capture the decode slot pool at load time so the first concurrent requests batch
    /// immediately — and so admission can price pooled sequences against a pool that
    /// actually exists, instead of charging every early request for a private
    /// verification graph the pool would discard on its first step.
    /// Failure leaves the lazy first-step path in charge.
    pub(super) fn warm_slot_pool(&mut self) {
        let capacity = CB_SLOT_TOKENS.min(self.loaded.model().max_sequence);
        self.warm_slot_pool_at(capacity);
    }

    pub(super) fn warm_slot_pool_at(&mut self, capacity: usize) {
        if self.maximum_states < 2 {
            return;
        }
        let maximum = self.maximum_states.min(CB_DECODE_SLOTS);
        let (envelope, baseline) = match self.loaded.shared_scratch_bytes().and_then(|scratch| {
            let physical = self.physical_available()?;
            Ok((
                pooled_envelope(physical, self.budget, self.active_budget()?, scratch),
                physical,
            ))
        }) {
            Ok(budget) => budget,
            Err(error) => {
                self.pool_failure_reason = Some(error.to_string());
                return;
            }
        };
        let mut funding = envelope;
        // Capture can materialize driver/graph storage absent from tensor quotes.
        // Retry against a smaller quote, never against a larger serving envelope.
        for attempt in 0..POOL_CAPTURE_ATTEMPTS {
            let selected = self.pooled_plan(capacity, maximum, funding);
            let plan = match selected {
                Ok(Some(plan)) => plan,
                Ok(None) => {
                    self.pool_failure_reason =
                        Some(format!("joint pool quote exceeds {funding} bytes"));
                    return;
                }
                Err(error) => {
                    self.pool_failure_reason = Some(error.to_string());
                    return;
                }
            };
            let quoted_bytes = plan.quote.total_bytes;
            // Keep the original baseline across retries: failed captures can leave
            // driver modules resident even after their tensor owners are reclaimed.
            let allocated = self
                .capture_pool_plan(attempt, capacity, envelope, &plan)
                .and_then(|()| {
                    self.physical_available()
                        .map(|free| baseline.saturating_sub(free))
                });
            if let Ok(allocated_bytes) = allocated
                && allocated_bytes <= envelope
            {
                super::profiling::capture_outcome(
                    attempt,
                    "accepted",
                    quoted_bytes,
                    Some(allocated_bytes),
                    envelope,
                    None,
                    None,
                );
                self.reclaimed_slot_capacity = None;
                self.deferred_pool_plan = None;
                self.slots_disabled = false;
                self.pool_failure_reason = None;
                tracing::info!(target: "infer::executor",
                    width = self.slots.as_ref().map_or(0, SlotPool::width),
                    capacity, quoted_bytes, allocated_bytes, envelope_bytes = envelope,
                    "CUDA slot pool captured within serving envelope");
                return;
            }
            let Some((allocated_bytes, next_funding)) =
                self.reject_pool_capture(attempt, allocated, quoted_bytes, envelope, funding)
            else {
                self.slots_disabled = true;
                return;
            };
            funding = next_funding;
            tracing::warn!(target: "infer::executor", attempt, allocated_bytes,
                envelope_bytes = envelope, next_quote_budget = funding,
                "pooled CUDA capture exceeded budget; replanning before publication");
        }
        self.slots_disabled = true;
    }

    fn reject_pool_capture(
        &mut self,
        attempt: usize,
        allocated: Result<u64>,
        quoted_bytes: u64,
        envelope: u64,
        funding: u64,
    ) -> Option<(u64, u64)> {
        self.pool_failure_reason = Some(match &allocated {
            Ok(bytes) => format!("pooled capture used {bytes} of {envelope} bytes"),
            Err(error) => error.to_string(),
        });
        self.clear_unbound_pool();
        if let Err(error) = self.loaded.device().reclaim_barrier() {
            let message = error.to_string();
            super::profiling::capture_outcome(
                attempt,
                "rejected",
                quoted_bytes,
                allocated.ok(),
                envelope,
                None,
                Some(&message),
            );
            self.pool_failure_reason = Some(message);
            return None;
        }
        let Ok(allocated_bytes) = allocated else {
            super::profiling::capture_outcome(
                attempt,
                "rejected",
                quoted_bytes,
                None,
                envelope,
                None,
                self.pool_failure_reason.as_deref(),
            );
            return None;
        };
        let next = pool_retry_budget(funding, quoted_bytes, allocated_bytes, self.kv_block_bytes)
            .min(self.physical_available().unwrap_or(0));
        super::profiling::capture_outcome(
            attempt,
            "rejected",
            quoted_bytes,
            Some(allocated_bytes),
            envelope,
            Some(next),
            None,
        );
        Some((allocated_bytes, next))
    }

    /// Construct unpublished owners and measure their complete allocation.
    fn capture_pool_plan(
        &mut self,
        attempt: usize,
        capacity: usize,
        envelope: u64,
        plan: &PoolPlan,
    ) -> Result<()> {
        let quote = plan.quote;
        let planned_width = quote.width;
        let prompt_width = quote.prompt_width;
        let kv_tokens = quote.kv_tokens;
        let quoted_bytes = quote.total_bytes;
        super::profiling::plan(attempt, envelope, &quote);
        tracing::info!(target: "infer::executor", planned_width, prompt_width, quoted_bytes,
            capacity, kv_tokens, "planned pooled CUDA serving storage");
        super::profiling::memory(attempt, "before_pool", planned_width, self.loaded.device());
        let pool = self
            .create_slot_pool(capacity, planned_width, kv_tokens)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::Capacity,
                    "could not construct pooled CUDA storage",
                )
            })?;
        let width = pool.width();
        super::profiling::memory(attempt, "target_pool", width, self.loaded.device());
        self.draft_slots = self.prepare_draft_pool(width, capacity);
        super::profiling::memory(attempt, "draft_pool", width, self.loaded.device());
        self.slots = Some(pool);
        self.build_homes(attempt, prompt_width);
        super::profiling::memory(attempt, "homes", width, self.loaded.device());
        if self.worker.is_none()
            || (self.loaded.mtp_depth() > 0 && self.draft_slots.is_none())
            || self.homes.iter().any(Option::is_none)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "incomplete pooled CUDA serving capture",
            ));
        }
        Ok(())
    }

    /// Only construction calls this: no live sequence has leased these new homes.
    fn clear_unbound_pool(&mut self) {
        self.worker = None;
        self.homes.clear();
        drop(self.draft_slots.take());
        drop(self.slots.take());
        let _ = self.loaded.reclaim_idle_prompt_scratch();
    }

    /// Quote all resources before capturing; retain the largest prompt rung at a width.
    fn pooled_plan(
        &self,
        capacity: usize,
        maximum: usize,
        envelope: u64,
    ) -> Result<Option<PoolPlan>> {
        let configured = self.loaded.pooled_prompt_width();
        // Preserve wide-prompt throughput: at most one halving, and never cross the
        // established 1024-row worker floor. Deeper shrinking saves memory but adds replays.
        let minimum_prompt = (configured / 2).max(WIDER_PREFILL_LANES).min(configured);
        for width in (2..=maximum).rev() {
            let mut prompt_width = configured;
            loop {
                if let Some(quote) = self.loaded.funded_pooled_serving(
                    width,
                    capacity,
                    paged_tokens(width, capacity),
                    prompt_width,
                    envelope,
                )? {
                    return Ok(Some(PoolPlan { quote }));
                }
                if prompt_width <= minimum_prompt {
                    break;
                }
                prompt_width = (prompt_width / 2).max(minimum_prompt);
            }
        }
        Ok(None)
    }

    /// Capture each slot's resident target and draft programs. A slot without them still
    /// serves sequences that bind a private program's state, so failures only log.
    #[expect(
        unsafe_code,
        reason = "Audited slot alias: homes are declared before the pools and cleared before \
                  either pool drops, and only the slot's lease holder replays them"
    )]
    fn build_homes(&mut self, attempt: usize, prompt_width: usize) {
        self.worker = None;
        self.homes.clear();
        let Some(pool) = self.slots.as_ref() else {
            return;
        };
        // SAFETY: the worker is cleared before the pool drops (see the expectation above).
        let kv = unsafe { pool.kv_views() };
        let worker = kv.and_then(|kv| self.loaded.prompt_worker(pool.capacity(), kv, prompt_width));
        match worker {
            Ok((program, draft)) => self.worker = Some(super::Worker { program, draft }),
            Err(error) => tracing::warn!(
                target: "infer::executor",
                %error,
                "wide prompt worker unavailable; slot-resident prompts use the narrow graph"
            ),
        }
        super::profiling::memory(attempt, "worker", pool.width(), self.loaded.device());
        for slot in 0..pool.width() {
            // SAFETY: see the expectation above.
            let built = unsafe {
                self.loaded
                    .slot_programs(pool, self.draft_slots.as_ref(), slot)
            };
            match built {
                Ok((program, draft)) => self.homes.push(Some(super::Home { program, draft })),
                Err(error) => {
                    tracing::warn!(
                        target: "infer::executor",
                        slot,
                        %error,
                        "slot-resident programs unavailable; this slot binds private state"
                    );
                    self.homes.push(None);
                }
            }
        }
    }

    /// Widest pool the device can fund, trying every width down to two.
    ///
    /// Each attempt that fails is followed by a reclaim so the next one is judged against
    /// memory that is actually free; a width that cannot fit is worth more to the server than
    /// no batching at all.
    fn create_slot_pool(
        &mut self,
        capacity: usize,
        maximum: usize,
        kv_tokens: usize,
    ) -> Option<SlotPool> {
        for width in (2..=maximum).rev() {
            loop {
                match self.loaded.slot_pool(width, capacity, Some(kv_tokens)) {
                    Ok(pool) => return Some(pool),
                    Err(error) => {
                        super::profiling::pool_failure("target", width, capacity, &error);
                        tracing::debug!(
                            target: "infer::executor",
                            width,
                            capacity,
                            reclaimable_mib = self
                                .loaded
                                .device()
                                .pool_reclaimable_bytes()
                                .unwrap_or(0)
                                / MIB as u64,
                            %error,
                            "slot pool attempt failed"
                        );
                        self.pool_failure_reason = Some(format!("width {width}: {error}"));
                        if !self.reclaim_cached_for_pool(&error) {
                            break;
                        }
                    }
                }
            }
        }
        None
    }

    fn prepare_draft_pool(&mut self, width: usize, capacity: usize) -> Option<SlotPool> {
        loop {
            match self.loaded.draft_slot_pool(width, capacity) {
                Ok(pool) => return pool,
                Err(error) => {
                    super::profiling::pool_failure("draft", width, capacity, &error);
                    self.pool_failure_reason = Some(format!("draft width {width}: {error}"));
                    if !self.reclaim_cached_for_pool(&error) {
                        return None;
                    }
                }
            }
        }
    }

    /// Idle programs and cached prefixes must not permanently disable active batching.
    /// Drain deferred frees before retrying the constructor's physical-memory admission.
    ///
    /// A driver out-of-memory is admitted here too. Only `Capacity` used to reclaim, so a
    /// width-4 OOM left the prefix cache in place and the width-3 and width-2 retries then
    /// failed on numbers that never reflected anything being freed — three refusals in a row
    /// and the pool was off for the life of the process, with the checkpoint snapshots as the
    /// only consumer that scales with the verification width.
    fn reclaim_cached_for_pool(&mut self, error: &Error) -> bool {
        matches!(error.code, ErrorCode::Capacity | ErrorCode::Backend)
            && (self.pool.evict() || self.prefix.evict_oldest().is_some())
            && self.loaded.device().reclaim_barrier().is_ok()
    }

    /// One consecutive run of eligible decode tasks: batched through the slot pool when
    /// one exists, serial otherwise.
    async fn run_segment(
        &mut self,
        tasks: &[ExecutionTask],
        segment: Range<usize>,
        outputs: &mut [Option<TaskOutput>],
        snapshots: &mut Vec<prefix::CachedPrefix>,
    ) -> Result<()> {
        if self.slots.is_none() {
            for index in segment {
                self.run_serial(tasks, index, outputs, snapshots).await?;
            }
            return Ok(());
        }
        let batched = self
            .execute_slot_segment(tasks, segment.clone(), outputs)
            .await?;
        for index in segment {
            if !batched.contains(&index) {
                self.run_serial(tasks, index, outputs, snapshots).await?;
            }
        }
        Ok(())
    }

    /// Assign slots to one eligible segment, replay it and commit the results.
    /// Returns the batched task indices; unassigned tasks are left for the serial path.
    /// Bound sequences always batch (their state lives in the slot), while unbound ones
    /// only bind when at least two lanes share a plain decode replay. An existing
    /// verification pool can accept one lane without recreating private checkpoints.
    async fn execute_slot_segment(
        &mut self,
        tasks: &[ExecutionTask],
        segment: Range<usize>,
        outputs: &mut [Option<TaskOutput>],
    ) -> Result<Vec<usize>> {
        let pool = self
            .slots
            .as_mut()
            .ok_or_else(|| Error::invariant("CUDA slot pool"))?;
        let mut participants: Vec<(usize, usize)> = Vec::new();
        let mut unbound: Vec<usize> = Vec::new();
        // Slot-resident sequences already hold their slot and wrote its state in place.
        let mut homed: Vec<(usize, usize)> = Vec::new();
        for index in segment {
            let state = self
                .states
                .get(&tasks[index].state)
                .ok_or_else(|| Error::invariant("validated CUDA state"))?;
            match (&state.slot, state.home) {
                (Some(lease), _) => participants.push((index, lease.slot)),
                (None, Some(slot)) => homed.push((index, slot)),
                (None, None) => unbound.push(index),
            }
        }
        let mut newly: Vec<(usize, usize)> = Vec::new();
        for &index in &unbound {
            if participants.len() + homed.len() + newly.len() >= pool.width() {
                break;
            }
            let Some(slot) = pool.take_free() else {
                break;
            };
            let tokens = self.states[&tasks[index].state].capacity;
            match pool.assign(slot, tokens, self.loaded.device()) {
                Ok(true) => {}
                Ok(false) => {
                    pool.release(slot);
                    break;
                }
                Err(error) => {
                    pool.release(slot);
                    return Err(error);
                }
            }
            newly.push((index, slot));
        }
        let minimum = if pool.verify().is_some() { 1 } else { 2 };
        if participants.is_empty() && homed.is_empty() && newly.len() < minimum {
            for (_, slot) in newly {
                pool.release(slot);
            }
            return Ok(Vec::new());
        }
        // Binding a slot-resident sequence copies nothing: its program's views are the slot.
        newly.extend_from_slice(&homed);
        participants.extend_from_slice(&newly);
        participants.sort_unstable_by_key(|&(index, _)| index);
        let device = self.loaded.device();
        if let Err(error) = slot_replay(
            (pool, self.draft_slots.as_mut()),
            &mut self.states,
            device,
            tasks,
            &participants,
            &newly,
            outputs,
        )
        .await
        {
            // A failed replay poisons the participating sequences, never the pool.
            for &(index, slot) in &participants {
                let Some(state) = self.states.get_mut(&tasks[index].state) else {
                    continue;
                };
                state.poisoned = true;
                // A home slot stays leased to its sequence until that sequence is released;
                // returning it here would let another request bind the same storage.
                if state.home == Some(slot) {
                    continue;
                }
                state.slot = None;
                pool.release(slot);
            }
            return Err(error);
        }
        Ok(participants.iter().map(|&(index, _)| index).collect())
    }

    /// Serial execution for one task; a slot-leased sequence has stale private state and
    /// must never reach here.
    async fn run_serial(
        &mut self,
        tasks: &[ExecutionTask],
        index: usize,
        outputs: &mut [Option<TaskOutput>],
        snapshots: &mut Vec<prefix::CachedPrefix>,
    ) -> Result<()> {
        let task = &tasks[index];
        let state = self
            .states
            .get_mut(&task.state)
            .ok_or_else(|| Error::invariant("validated CUDA state"))?;
        if state.slot.is_some() {
            return Err(Error::invariant(
                "slot-leased CUDA sequence cannot run serially",
            ));
        }
        // A slot-resident sequence's programs capture no prompt graph: every prompt chunk runs
        // on the worker.
        let wide = state.home.is_some() && !matches!(task.tokens, ExecutionInput::Decode { .. });
        let output = match self.worker.as_mut() {
            Some(worker) if wide => run_on_worker(state, worker, task, &self.loaded).await?,
            _ => state.run(task, &self.loaded).await?,
        };
        if let Some(snapshot) =
            cacheable_prefix(self.loaded.device(), &self.loaded, state, task).await
        {
            snapshots.push(snapshot);
        }
        outputs[index] = Some(TaskOutput {
            request: task.request,
            output,
        });
        Ok(())
    }
}

/// Bind newly assigned sequences into their slots, replay one decode step for every
/// participant and commit each lane's token, logits and cursor in task order.
/// Run a slot-resident sequence's prompt chunk on the wide worker: move its target and draft
/// state in, replay, and move the advanced state back into the slot.
async fn run_on_worker(
    state: &mut Sequence,
    worker: &mut super::Worker,
    task: &ExecutionTask,
    loaded: &LoadedModel,
) -> Result<ModelOutput> {
    super::profiling::phase("worker_load", || -> Result<()> {
        worker
            .program
            .copy_state_from(&state.program, state.program.position())?;
        if let (Some(spec), Some(draft)) = (state.speculation.as_ref(), worker.draft.as_mut()) {
            draft.copy_state_from(&spec.program, spec.program.position())?;
        }
        Ok(())
    })?;
    swap_worker(state, worker);
    let output = state.run(task, loaded).await;
    swap_worker(state, worker);
    let output = output?;
    super::profiling::phase("worker_store", || -> Result<()> {
        state
            .program
            .copy_state_from(&worker.program, worker.program.position())?;
        if let (Some(spec), Some(draft)) = (state.speculation.as_mut(), worker.draft.as_ref()) {
            spec.program.copy_state_from(draft, draft.position())?;
        }
        Ok(())
    })?;
    Ok(output)
}

const fn swap_worker(state: &mut Sequence, worker: &mut super::Worker) {
    swap(&mut state.program, &mut worker.program);
    if let (Some(spec), Some(draft)) = (state.speculation.as_mut(), worker.draft.as_mut()) {
        swap(&mut spec.program, draft);
    }
}

/// Numerator of the physical-KV share a paged pool reserves: three quarters of the unpaged
/// reservation, since a request holds blocks for what it can reach, not the slot's whole
/// capacity.
const PAGED_KV_SHARE_NUMERATOR: usize = 3;
/// Denominator of [`PAGED_KV_SHARE_NUMERATOR`].
const PAGED_KV_SHARE_DENOMINATOR: usize = 4;

/// Physical KV positions a pool of `width` slots of `capacity` tokens shares.
const fn paged_tokens(width: usize, capacity: usize) -> usize {
    width * capacity * PAGED_KV_SHARE_NUMERATOR / PAGED_KV_SHARE_DENOMINATOR
}

async fn slot_replay(
    pools: (&mut SlotPool, Option<&mut SlotPool>),
    states: &mut BTreeMap<StateId, Sequence>,
    device: &CudaDevice,
    tasks: &[ExecutionTask],
    participants: &[(usize, usize)],
    newly: &[(usize, usize)],
    outputs: &mut [Option<TaskOutput>],
) -> Result<()> {
    let (pool, draft) = pools;
    for &(index, slot) in newly {
        let state = states
            .get_mut(&tasks[index].state)
            .ok_or_else(|| Error::invariant("validated CUDA state"))?;
        let rows = state.history.len();
        pool.bind(
            slot,
            state.program.states(),
            state.program.fp8_states(),
            rows,
            device,
        )?;
        state.program.discard_verification();
        // Only a serial-priced admission charged for private verification storage; crediting
        // a pooled price with released bytes it never included would underflow the account.
        let credit = if state.verification_priced {
            state.program.released_verification_bytes()
        } else {
            0
        };
        state.budget = state
            .budget
            .checked_sub(credit)
            .ok_or_else(|| Error::invariant("private verification admission credit"))?;
        state.slot = Some(SlotLease {
            slot,
            position: rows,
            batched: true,
        });
    }
    if let Some(width) = pool.verify() {
        return slot_speculate(
            (pool, draft),
            states,
            device,
            tasks,
            participants,
            outputs,
            width,
        )
        .await;
    }
    let mut lanes = Vec::with_capacity(participants.len());
    for &(index, slot) in participants {
        let state = states
            .get(&tasks[index].state)
            .ok_or_else(|| Error::invariant("validated CUDA state"))?;
        let &ExecutionInput::Decode { token, .. } = &tasks[index].tokens else {
            return Err(Error::invariant("slot decode task"));
        };
        let position = state.history.len();
        lanes.push((slot, token, position, position));
    }
    let plain_greedy = participants
        .iter()
        .all(|&(index, _)| tasks[index].plain_greedy);
    let readout = pool
        .run_lanes_readout_async(device, &lanes, plain_greedy)
        .await?;
    for (row, ((_, logits), &(index, _))) in readout.rows.into_iter().zip(participants).enumerate()
    {
        let next_token = match readout.tokens.as_ref() {
            Some(tokens) => {
                Some(tokens[row].ok_or_else(|| Error::invariant("slot selected token"))?)
            }
            None => None,
        };
        let state = states
            .get_mut(&tasks[index].state)
            .ok_or_else(|| Error::invariant("validated CUDA state"))?;
        tasks[index].tokens.commit(&mut state.history)?;
        if let Some(lease) = &mut state.slot {
            lease.position += 1;
        }
        outputs[index] = Some(TaskOutput {
            request: tasks[index].request,
            output: ModelOutput {
                next_token,
                logits,
                hidden: Vec::new(),
                tokens: Vec::new(),
            },
        });
    }
    Ok(())
}

/// Batch compatible drafts, then verify all candidate chains in one target replay.
async fn slot_speculate(
    pools: (&mut SlotPool, Option<&mut SlotPool>),
    states: &mut BTreeMap<StateId, Sequence>,
    device: &CudaDevice,
    tasks: &[ExecutionTask],
    participants: &[(usize, usize)],
    outputs: &mut [Option<TaskOutput>],
    width: usize,
) -> Result<()> {
    let (pool, mut draft) = pools;
    let batched = match draft.as_deref_mut() {
        Some(draft) => {
            super::profiling::phase_async("draft_propose", async {
                super::drafting::propose(draft, states, device, tasks, participants, width).await
            })
            .await?
        }
        None => None,
    };
    let defer_draft = batched.is_some();
    let candidates = if let Some(candidates) = batched {
        candidates
    } else {
        let mut candidates = Vec::with_capacity(participants.len());
        for &(index, _) in participants {
            let state = states
                .get_mut(&tasks[index].state)
                .ok_or_else(|| Error::invariant("slot speculation state"))?;
            candidates.push(state.slot_candidates(&tasks[index], width).await?);
        }
        candidates
    };
    let mut lanes = Vec::new();
    for (&(index, slot), chain) in participants.iter().zip(&candidates) {
        let position = states[&tasks[index].state].history.len();
        for (offset, &token) in chain.iter().enumerate() {
            lanes.push((slot, offset, token, position + offset, position + offset));
        }
    }
    let plain_greedy = participants
        .iter()
        .all(|&(index, _)| tasks[index].plain_greedy);
    let mut readout = super::profiling::phase_async("verify", async {
        pool.run_verify_readout_async(device, &lanes, plain_greedy)
            .await
    })
    .await?;
    let rows = &mut readout.rows;
    let mut commits = Vec::with_capacity(participants.len());
    for (&(index, slot), chain) in participants.iter().zip(candidates) {
        let state = states
            .get_mut(&tasks[index].state)
            .ok_or_else(|| Error::invariant("slot speculation state"))?;
        let output = state
            .slot_decide(
                &tasks[index],
                &chain,
                &mut rows[slot * width..][..chain.len()],
                readout
                    .tokens
                    .as_ref()
                    .map(|tokens| &tokens[slot * width..][..chain.len()]),
                defer_draft,
            )
            .await?;
        commits.push((slot, output.tokens.len()));
        if let Some(lease) = &mut state.slot {
            lease.position = state.history.len();
        }
        outputs[index] = Some(TaskOutput {
            request: tasks[index].request,
            output,
        });
    }
    pool.commit_verify_many(device, &commits)?;
    if defer_draft {
        super::drafting::catch_up(
            draft.ok_or_else(|| Error::invariant("batched draft pool"))?,
            states,
            device,
            tasks,
            participants,
            outputs,
            rows,
            width,
        )?;
    }
    Ok(())
}

/// Greedy decision at one sequence position; the engine output stage uses the same sampler.
struct Sampler<'a> {
    sampling: &'a infer_ir::Sampling,
    request: u64,
    prompt: &'a [u32],
    scratch: &'a mut infer_workloads::SamplingWorkspace,
}
impl Sampler<'_> {
    fn pick(&mut self, logits: &[f32], index: usize, generated: &[u32]) -> Result<u32> {
        infer_workloads::sample_with_history(
            logits,
            self.sampling,
            self.request,
            index,
            infer_workloads::SamplingHistory {
                prompt: self.prompt,
                generated,
            },
            self.scratch,
        )
    }
}
impl Sequence {
    pub(super) fn prepare_verification(&mut self, loaded: &LoadedModel) -> Result<()> {
        // A slot lease already credited the released private storage. Only an
        // unbound private reservation funds this graph; homes use prepaid pool storage.
        if self.home.is_none()
            && self.slot.is_none()
            && self.verification_priced
            && self.speculation.is_some()
        {
            loaded.ensure_verification(&mut self.program)?;
        }
        Ok(())
    }

    /// Keep the draft cursor aligned even on non-greedy or capacity-limited steps.
    async fn slot_candidates(&mut self, task: &ExecutionTask, width: usize) -> Result<Vec<u32>> {
        let ExecutionInput::Decode { position, token } = task.tokens else {
            return Err(Error::invariant("slot speculation decode"));
        };
        let mut chain = vec![token];
        let Some(spec) = self.speculation.as_mut() else {
            return Ok(chain);
        };
        let prompt_len = *spec.prompt_len.get_or_insert(position);
        let count = spec.depth.min(width - 1).min(self.capacity - position - 1);
        if let Some(sampling) = task.sampling.as_ref().filter(|s| s.temperature == 0.0)
            && count > 0
            && position > 0
        {
            let mut generated = self.history[prompt_len..].to_vec();
            generated.push(token);
            let mut sampler = Sampler {
                sampling,
                request: task.request.get(),
                prompt: &self.history[..prompt_len],
                scratch: &mut spec.scratch,
            };
            chain.extend(
                propose(
                    &mut spec.program,
                    &mut sampler,
                    &mut generated,
                    (token, position),
                    &spec.last_hidden,
                    count,
                )
                .await?,
            );
        } else if position > 0 {
            spec.program
                .step_async(
                    token,
                    position,
                    position - MTP_KV_OFFSET,
                    Some(&spec.last_hidden),
                    false,
                )
                .await?;
        }
        Ok(chain)
    }

    /// Return only accepted proposals. On rejection the engine samples the replacement
    /// from the last accepted row and submits it next tick, avoiding an extra target pass.
    async fn slot_decide(
        &mut self,
        task: &ExecutionTask,
        chain: &[u32],
        rows: &mut [(Vec<f32>, Vec<f32>)],
        tokens: Option<&[Option<u32>]>,
        defer_draft: bool,
    ) -> Result<ModelOutput> {
        if let Some(sampling) = task.sampling.as_ref() {
            sampling.validate()?;
        }
        if tokens.is_some()
            && (!task.plain_greedy
                || task
                    .sampling
                    .as_ref()
                    .is_none_or(|sampling| !sampling.uses_plain_greedy()))
        {
            return Err(Error::invalid(
                "selected verification requires plain greedy",
            ));
        }
        if tokens
            .is_some_and(|tokens| tokens.len() < chain.len() || tokens.iter().any(Option::is_none))
        {
            return Err(Error::invariant("verification selected token rows"));
        }
        let position = self.history.len();
        task.tokens.commit(&mut self.history)?;
        let mut accepted = Vec::new();
        if let Some(spec) = self.speculation.as_mut() {
            let prompt_len = *spec.prompt_len.get_or_insert(position);
            if let Some(sampling) = task.sampling.as_ref() {
                let mut generated = self.history[prompt_len..].to_vec();
                let mut sampler = Sampler {
                    sampling,
                    request: task.request.get(),
                    prompt: &self.history[..prompt_len],
                    scratch: &mut spec.scratch,
                };
                for (offset, &candidate) in chain[1..].iter().enumerate() {
                    let selected = match tokens {
                        Some(tokens) => tokens[offset]
                            .ok_or_else(|| Error::invariant("verification selected token"))?,
                        None => sampler.pick(&rows[offset].1, position + offset + 1, &generated)?,
                    };
                    if selected != candidate {
                        break;
                    }
                    accepted.push(candidate);
                    generated.push(candidate);
                    if sampling.is_eos(candidate) {
                        break;
                    }
                }
            }
            if !defer_draft {
                // Even fully accepted proposals used draft hidden rows during drafting.
                // Replace their KV with target-conditioned rows before the next proposal.
                let hidden: Vec<_> = rows[..accepted.len()].iter().map(|r| r.0.clone()).collect();
                spec.replay(&accepted, &hidden, position + 1).await?;
            }
            spec.last_hidden.clone_from(&rows[accepted.len()].0);
        }
        self.history.extend_from_slice(&accepted);
        let next_token = tokens
            .map(|tokens| {
                tokens[accepted.len()].ok_or_else(|| Error::invariant("verification next token"))
            })
            .transpose()?;
        Ok(ModelOutput {
            next_token,
            logits: take(&mut rows[accepted.len()].1),
            hidden: Vec::new(),
            tokens: accepted,
        })
    }

    async fn run(&mut self, task: &ExecutionTask, loaded: &LoadedModel) -> Result<ModelOutput> {
        if let Some(plan) = task.media.as_ref() {
            return self.run_media(task, plan, loaded).await;
        }
        let tokens = task.tokens.delta(self.history.len())?;
        let readout = task.tokens.readout();
        if let Some(output) = self.speculate(task, readout).await? {
            return Ok(output);
        }
        let speculate = self.speculation.is_some();
        let mut offset = 0;
        let mut logits = Vec::new();
        let mut next_token = None;
        // Target hidden rows prime the draft; a chunked prefill spans several calls. Prompt-graph
        // chunks hand the draft the target's device rows directly; single-token steps keep the
        // host rows.
        let mut priming = Vec::new();
        let mut device_primed = false;
        while offset < tokens.len() {
            let count = self.chunk_len(tokens.len() - offset);
            let position = self.history.len() + offset;
            let last = offset + count == tokens.len();
            let read_logits = last && readout != OutputReadout::None;
            let device_prime = speculate && count > 1 && self.readout != OutputReadout::Full;
            let read_hidden = self.readout == OutputReadout::Full || (speculate && !device_prime);
            let results = if count > 1 {
                super::profiling::phase_async("prefill_chunk", async {
                    self.program
                        .prefill_batch_readout_async(
                            &tokens[offset..offset + count],
                            position,
                            read_logits,
                            read_hidden,
                        )
                        .await
                })
                .await?
            } else if read_logits
                && task.plain_greedy
                && readout == OutputReadout::Logits
                && self.readout != OutputReadout::Full
                && self.media_prompt_len == 0
            {
                let selected = super::profiling::phase_async("selected_step", async {
                    self.program
                        .step_selected_async(tokens[offset], position, read_hidden)
                        .await
                })
                .await?;
                next_token = selected
                    .tokens
                    .as_ref()
                    .map(|tokens| {
                        tokens[0].ok_or_else(|| Error::invariant("program selected token"))
                    })
                    .transpose()?;
                selected.rows
            } else {
                super::profiling::phase_async("prefill_step", async {
                    self.decode_step(tokens[offset], position, read_logits, read_hidden)
                        .await
                })
                .await?
            };
            if device_prime {
                self.prime_chunk(&tokens[offset..offset + count], position)
                    .await?;
                device_primed = true;
            }
            for (lane, (hidden, row_logits)) in results.into_iter().enumerate() {
                if speculate && !device_prime {
                    priming.push((position + lane, tokens[offset + lane], hidden.clone()));
                }
                if self.readout == OutputReadout::Full {
                    self.hidden.push(hidden);
                }
                if !row_logits.is_empty() {
                    logits = row_logits;
                }
            }
            offset += count;
        }
        if device_primed && let Some(spec) = self.speculation.as_mut() {
            spec.last_hidden =
                super::profiling::phase_async("prime_readback", spec.program.primed_hidden_async())
                    .await?;
        }
        if !matches!(task.tokens, ExecutionInput::Decode { .. }) {
            self.prime_draft(&priming).await?;
        }
        task.tokens.commit(&mut self.history)?;
        Ok(self.output(readout, next_token, logits))
    }

    /// Prefill one media prompt chunk with embeddings cached for the whole prompt.
    async fn run_media(
        &mut self,
        task: &ExecutionTask,
        plan: &MediaPlan,
        loaded: &LoadedModel,
    ) -> Result<ModelOutput> {
        if self.speculation.is_some() {
            return Err(Error::unsupported(
                "multimodal prefill with MTP speculation is not supported",
            ));
        }
        let tokens = task.tokens.delta(self.history.len())?;
        let readout = task.tokens.readout();
        if readout == OutputReadout::Full {
            return Err(Error::unsupported(
                "multimodal full-hidden readout is not supported",
            ));
        }
        let prompt_len = plan.prompt.len();
        if tokens.is_empty()
            || self
                .history
                .len()
                .checked_add(tokens.len())
                .is_none_or(|end| end > prompt_len)
        {
            return Err(Error::invalid("media prefill token range"));
        }
        if self.history.is_empty() && self.media.is_none() {
            self.prepare_media(plan, loaded)?;
        }
        let media = self
            .media
            .take()
            .ok_or_else(|| Error::invalid("media prefill did not start at position zero"))?;
        if media.bases.len() != plan.items.len() {
            self.media = Some(media);
            return Err(Error::invariant("cached media does not match its plan"));
        }
        let width = self.program.prefill_width();
        if width == 0 {
            return Err(Error::unsupported(
                "multimodal prefill requires a prompt graph",
            ));
        }
        let hidden = loaded.model().hidden_size;
        tracing::debug!(target: "infer::executor", prompt_len, tokens = tokens.len(), width,
            bases = ?media.bases, spans = ?plan.items.iter().map(|item| item.span.clone()).collect::<Vec<_>>(),
            "running CUDA media prefill");
        let mut offset = 0usize;
        let mut logits = Vec::new();
        while offset < tokens.len() {
            let count = width.min(tokens.len() - offset);
            let position = self.history.len() + offset;
            let chunk = &tokens[offset..offset + count];
            let last = self.history.len() + offset + count == prompt_len;
            let mut rows = Vec::new();
            for (index, item) in plan.items.iter().enumerate() {
                let start = item.span.start.max(position);
                let stop = item.span.end.min(position + count);
                for token in start..stop {
                    rows.push((
                        token - position,
                        media.bases[index] + (token - item.span.start),
                    ));
                }
            }
            let rotary = media
                .layout
                .positions
                .as_ref()
                .map(|positions| &positions[position..position + count]);
            let output = self
                .program
                .prefill_media_chunk_async(
                    chunk,
                    position,
                    &media.tensor,
                    &rows,
                    hidden,
                    rotary,
                    last,
                )
                .await?;
            tracing::debug!(target: "infer::executor", last, position, count, rows = output.len(),
                last_logits = output.last().map(|(_, row)| row.len()), "media chunk output");
            if last {
                logits = output
                    .into_iter()
                    .next_back()
                    .map(|(_, row)| row)
                    .ok_or_else(|| Error::invariant("media prefill logits"))?;
            }
            offset += count;
        }
        if self.history.len() + tokens.len() == prompt_len {
            self.rotary_base = media.layout.next_rotary;
            self.media_prompt_len = prompt_len;
        } else {
            self.media = Some(media);
        }
        task.tokens.commit(&mut self.history)?;
        Ok(self.output(readout, None, logits))
    }

    fn prepare_media(&mut self, plan: &MediaPlan, loaded: &LoadedModel) -> Result<()> {
        let tower = loaded
            .vision()
            .ok_or_else(|| Error::unsupported("model has no bound vision encoder"))?;
        let prepared = PreparedPrompt {
            tokens: plan.prompt.to_vec(),
            images: plan
                .items
                .iter()
                .map(|item| PromptImage {
                    pixels: item.pixels.to_vec(),
                    grid: item.grid,
                })
                .collect(),
            spans: plan.items.iter().map(|item| item.span.clone()).collect(),
        };
        let layout = media_layout(&prepared, loaded.imported(), &plan.modality)?;
        if layout.spans.len() != plan.items.len() {
            return Err(Error::invalid("media layout does not cover every item"));
        }
        let hidden = loaded.model().hidden_size;
        let mut data = Vec::new();
        let mut bases = Vec::with_capacity(prepared.images.len());
        for image in &prepared.images {
            let encoded = tower.encode(loaded.device(), image)?;
            bases.push(data.len() / hidden);
            data.extend(encoded);
        }
        let rows = data.len() / hidden;
        tracing::debug!(target: "infer::executor", rows, bases = ?bases,
            spans = ?layout.spans, positions = layout.positions.as_ref().map(Vec::len),
            "prepared CUDA media embeddings");
        let tensor = loaded.device().upload(data, &[rows, hidden])?;
        self.media = Some(CachedMedia {
            tensor,
            layout,
            bases,
        });
        Ok(())
    }

    async fn decode_step(
        &mut self,
        token: u32,
        position: usize,
        read_logits: bool,
        read_hidden: bool,
    ) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        if self.media_prompt_len == 0 {
            self.program
                .step_readout_async(token, position, position, None, read_logits, read_hidden)
                .await
                .map(|row| vec![row])
        } else {
            let rotary = self.rotary_at(position);
            self.program
                .step_positioned_async(token, [rotary; ROTARY_AXES], position, None, read_logits)
                .await
                .map(|row| vec![row])
        }
    }

    const fn rotary_at(&self, position: usize) -> usize {
        if self.media_prompt_len == 0 {
            position
        } else {
            self.rotary_base
                .saturating_add(position.saturating_sub(self.media_prompt_len))
        }
    }

    fn chunk_len(&self, remaining: usize) -> usize {
        let width = self.program.prefill_width();
        if width >= PREFILL_LANES || width > 1 && remaining >= width {
            width.min(remaining)
        } else {
            1
        }
    }

    fn output(
        &self,
        readout: OutputReadout,
        next_token: Option<u32>,
        logits: Vec<f32>,
    ) -> ModelOutput {
        ModelOutput {
            next_token,
            logits,
            hidden: if readout == OutputReadout::Full {
                self.hidden.clone()
            } else {
                Vec::new()
            },
            tokens: Vec::new(),
        }
    }

    /// Prime the draft with the device hidden rows the target's prompt chunk just wrote.
    async fn prime_chunk(&mut self, tokens: &[u32], position: usize) -> Result<()> {
        let rows = self
            .program
            .prompt_hidden(tokens.len())
            .ok_or_else(|| Error::invariant("target prompt hidden rows"))?;
        let spec = self
            .speculation
            .as_mut()
            .ok_or_else(|| Error::invariant("draft priming"))?;
        // A host-held row (from decode or a previous host-primed chunk) seeds the draft once;
        // afterwards the draft keeps the previous row on the device.
        let previous = take(&mut spec.last_hidden);
        super::profiling::phase_async("prime_draft", async {
            spec.program
                .prime_from_target_async(
                    tokens,
                    position,
                    &rows,
                    (!previous.is_empty()).then_some(previous.as_slice()),
                )
                .await
        })
        .await
    }

    /// Feed a prompt timeline through the draft: its KV mirrors the target from index one onward.
    async fn prime_draft(&mut self, rows: &[(usize, u32, Vec<f32>)]) -> Result<()> {
        let Some(spec) = self.speculation.as_mut() else {
            return Ok(());
        };
        let mut previous = take(&mut spec.last_hidden);
        let mut result = Ok(());
        // Chunk by the width the draft graph actually captured, not a fixed constant: the graph
        // is fixed-width, so a chunk wider than it is rejected and a narrower one wastes a replay.
        let width = spec.program.prefill_width().max(PREFILL_LANES);
        let mut start = 0;
        while start < rows.len() {
            let position = rows[start].0;
            let chunk = &rows[start..(start + width).min(rows.len())];
            // A fused draft batch needs contiguous positions; anything else replays token by
            // token through the ordinary single-step path.
            let contiguous = chunk
                .iter()
                .enumerate()
                .all(|(offset, (index, _, _))| *index == position + offset);
            if contiguous {
                let mut tokens = Vec::with_capacity(chunk.len());
                let mut externals = Vec::new();
                for (offset, (_, token, hidden)) in chunk.iter().enumerate() {
                    tokens.push(*token);
                    // MTP consumes the hidden of the position before the token it embeds; the
                    // prompt's first position has no predecessor and its state write is masked.
                    let source = if offset == 0 {
                        &previous
                    } else {
                        &chunk[offset - 1].2
                    };
                    externals.extend_from_slice(if source.len() == hidden.len() {
                        source
                    } else {
                        hidden
                    });
                }
                if let Err(error) = spec
                    .program
                    .prime_batch_async(&tokens, position, &externals)
                    .await
                {
                    result = Err(error);
                    break;
                }
                previous.clone_from(&chunk[chunk.len() - 1].2);
                start += chunk.len();
                continue;
            }
            let (index, token, hidden) = &rows[start];
            if *index > 0
                && let Err(error) = spec
                    .program
                    .step_async(
                        *token,
                        *index,
                        *index - MTP_KV_OFFSET,
                        Some(&previous),
                        false,
                    )
                    .await
            {
                result = Err(error);
                break;
            }
            previous.clone_from(hidden);
            start += 1;
        }
        if result.is_ok() {
            spec.last_hidden = previous;
        }
        result
    }

    /// Greedy MTP speculation. `None` keeps the ordinary single-step path.
    async fn speculate(
        &mut self,
        task: &ExecutionTask,
        readout: OutputReadout,
    ) -> Result<Option<ModelOutput>> {
        let Some(sampling) = task.sampling.as_ref() else {
            return Ok(None);
        };
        let ExecutionInput::Decode { position, .. } = task.tokens else {
            return Ok(None);
        };
        let Some(speculation) = &self.speculation else {
            return Ok(None);
        };
        let width = speculation.depth + 1;
        if readout != OutputReadout::Logits
            // Rejection under a non-zero temperature needs residual sampling the engine sampler
            // cannot reproduce from plain target logits, so speculation stays greedy-only.
            || sampling.temperature != 0.0
            || width < 2
            || position == 0
            || position != self.history.len()
            || position.saturating_add(width) > self.capacity
        {
            return Ok(None);
        }
        if self.program.batch_width() != width {
            return Err(Error::new(
                ErrorCode::Capacity,
                "MTP decode requires its reserved shared slot or prepared private verification graph",
            ));
        }
        // Use the same accept/rollback contract as pooled verification. On rejection
        // return the target logits at the accepted prefix; the output stage samples
        // the replacement and submits it on the next step. An extra scalar target
        // forward here repeats the rejected work and defeats weight sharing.
        let chain = self.slot_candidates(task, width).await?;
        let mut readout = self
            .program
            .step_batch_readout_async(&chain, position, task.plain_greedy)
            .await?;
        let output = self
            .slot_decide(
                task,
                &chain,
                &mut readout.rows,
                readout.tokens.as_deref(),
                false,
            )
            .await?;
        self.program
            .commit_batch_async(output.tokens.len() + 1)
            .await?;
        Ok(Some(output))
    }
}

impl Speculation {
    /// Rebuild the draft timeline after speculation: rewind, then replay the accepted prefix.
    async fn replay(
        &mut self,
        decided: &[u32],
        hidden_of: &[Vec<f32>],
        start: usize,
    ) -> Result<()> {
        self.program.rewind_attention(start - 1)?;
        for (offset, token) in decided.iter().enumerate() {
            let index = start + offset;
            self.program
                .step_async(
                    *token,
                    index,
                    index - MTP_KV_OFFSET,
                    Some(&hidden_of[offset]),
                    false,
                )
                .await?;
        }
        Ok(())
    }
}
/// Draft proposals for the positions after the submitted token. The first step consumes that
/// token itself, so it doubles as the draft's catch-up and proposes `position + 1`.
async fn propose(
    draft: &mut super::super::resident::DeviceProgram,
    sampler: &mut Sampler<'_>,
    proposed: &mut Vec<u32>,
    submitted: (u32, usize),
    previous: &[f32],
    count: usize,
) -> Result<Vec<u32>> {
    let mut proposals = Vec::new();
    let (mut token, start) = submitted;
    sampler.sampling.validate()?;
    let device_greedy = sampler.sampling.uses_plain_greedy() && draft.has_device_greedy();
    let mut hidden = if device_greedy {
        Vec::new()
    } else {
        previous.to_vec()
    };
    for index in (start..).take(count) {
        let candidate = if device_greedy {
            draft
                .step_greedy_async(
                    token,
                    index,
                    index - MTP_KV_OFFSET,
                    (index == start).then_some(previous),
                )
                .await?
        } else {
            let (next_hidden, logits) = draft
                .step_async(token, index, index - MTP_KV_OFFSET, Some(&hidden), true)
                .await?;
            hidden = next_hidden;
            sampler.pick(&logits, index + 1, proposed)?
        };
        proposed.push(candidate);
        proposals.push(candidate);
        token = candidate;
    }
    Ok(proposals)
}

/// Plan only for requests that fit the pool and request its logits readout.
fn candidate_pool_capacity(states: impl Iterator<Item = (usize, OutputReadout)>) -> Option<usize> {
    let (count, capacity) = states
        .filter(|&(capacity, readout)| {
            readout == OutputReadout::Logits && capacity <= CB_SLOT_TOKENS
        })
        .fold((0, 1), |(count, maximum), (capacity, _)| {
            (count + 1, maximum.max(capacity))
        });
    (count >= 2).then(|| capacity.next_power_of_two())
}

// Private programs keep their reservation and shared prompt scratch while a lazy
// pool is constructed. Only an empty backend can spend the scratch reservation
// on the pool's worker; physically reclaimable allocator storage is capped above.
fn pooled_envelope(physical: u64, budget: u64, active: u64, scratch: u64) -> u64 {
    let credit = if active == 0 { scratch } else { 0 };
    physical.min(budget.saturating_sub(active).saturating_add(credit))
}

/// Cover observed unquoted storage and one physical KV allocation quantum.
/// Saturation terminates retries when the envelope cannot fund any complete pool.
fn pool_retry_budget(funding: u64, quoted: u64, allocated: u64, kv_block: u64) -> u64 {
    funding.saturating_sub(allocated.saturating_sub(quoted).max(kv_block).max(1))
}

#[cfg(test)]
#[path = "../../tests/unit/executor_execution.rs"]
mod tests;
