use super::prefix_state::{DraftSnapshot, ProgramSnapshot};
use super::{CachedMedia, CudaExecutor, ModelOutput, Sequence, SlotLease, Speculation, prefix};
use crate::constants::CB_SLOT_TOKENS;
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::MAX_POOL_SLOTS;
use crate::constants::MIB;
use crate::constants::MTP_KV_OFFSET;
use crate::constants::PREFILL_LANES;
use crate::constants::WIDER_PREFILL_LANES;
use crate::device::CudaDevice;
use crate::loading::{LoadedModel, PooledServingQuote};
use crate::resident::DeviceProgram;
use crate::resident::dflash2::{Engine, VerifiedRows};
use crate::resident::prefill_burst::{self, PromptBurst, SEGMENT_ROWS};
use crate::resident::slot_batch::SlotPool;
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::{ExecutionInput, ExecutionTask, MediaPixels, MediaPlan, OutputReadout, TaskOutput};
use infer_models::media_plan_layout;
use infer_models::mrope::ROTARY_AXES;
use std::collections::BTreeMap;
use std::mem::swap;
use std::mem::take;
use std::ops::Range;
/// Bound capture/replan work while adapting tensor quotes to observed driver storage.
/// Spacing of automatic pool widths; at most this many lanes minus one go unused.
const POOL_WIDTH_STEP: usize = 4;
const POOL_CAPTURE_ATTEMPTS: usize = 3;
/// Prompts one step needs before any multi-sequence graph is worth its padded rows.
const MIN_BURST_PROMPTS: usize = 2;
/// A rung replays every row: at most this many segments per prompt still beats the worker.
const MAX_SEGMENTS_PER_PROMPT: usize = 4;

/// The prompt rows a just-finished chunk leaves as a reusable prefix, if any.
///
/// Only a logits-readout generation that consumed prefill tokens qualifies, at the regular
/// cadence or at the request's prefix boundary, and always on whole KV blocks.
fn reusable_prefix(state: &Sequence, task: &ExecutionTask) -> Option<usize> {
    if task.media.is_some()
        || state.poisoned
        || state.draft.is_some()
        || state.readout != OutputReadout::Logits
        || !matches!(
            task.tokens,
            ExecutionInput::Full(_) | ExecutionInput::Prefill { .. }
        )
    {
        return None;
    }
    let covered = state.history.len();
    (covered > 0
        && covered <= state.capacity
        && covered.is_multiple_of(KV_BLOCK_TOKENS)
        && (task.publish_prefix || covered.is_multiple_of(prefix::PREFIX_GRANULARITY)))
    .then_some(covered)
}

/// Snapshot a private sequence's covered prefix: head-major KV rows and complete recurrent and
/// draft state, so restoration resumes the same inference trajectory.
async fn cacheable_prefix(
    device: &CudaDevice,
    loaded: &LoadedModel,
    state: &mut Sequence,
    covered: usize,
) -> Option<prefix::CachedPrefix> {
    let target = ProgramSnapshot::capture_async(device, &state.program, loaded.graph(), covered)
        .await
        .ok()?;
    let draft = if let Some(spec) = &mut state.speculation {
        Some(DraftSnapshot {
            program: ProgramSnapshot::capture_async(
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
    Some(cached(state, target, draft))
}

/// Snapshot a slot-resident sequence's covered prefix by exporting it from its target and draft
/// slots. The copies are settled before the next chunk advances that state, and before any
/// destination is dropped: a snapshot that cannot be built is skipped, but a device fault while
/// settling it is reported.
fn slot_prefix(
    device: &CudaDevice,
    loaded: &LoadedModel,
    (target_pool, draft_pool): (Option<&SlotPool>, Option<&SlotPool>),
    state: &Sequence,
    (slot, covered): (usize, usize),
) -> Result<Option<prefix::CachedPrefix>> {
    let built = (|| -> Result<Option<(ProgramSnapshot, Option<DraftSnapshot>)>> {
        let Some(target_pool) = target_pool else {
            return Ok(None);
        };
        let (states, fp8) = target_pool.export(slot, covered, device)?;
        let target =
            ProgramSnapshot::from_parts(states, fp8, loaded.graph(), state.program.position())?;
        let draft = match (&state.speculation, draft_pool, loaded.draft_graph()) {
            (Some(spec), Some(draft_pool), Some(draft_graph)) => {
                let rows = covered.saturating_sub(MTP_KV_OFFSET);
                let (states, fp8) = draft_pool.export(slot, rows, device)?;
                Some(DraftSnapshot {
                    program: ProgramSnapshot::from_parts(
                        states,
                        fp8,
                        draft_graph,
                        spec.program.position(),
                    )?,
                    hidden: spec.last_hidden.clone(),
                })
            }
            (Some(_), _, _) => return Ok(None),
            (None, _, _) => None,
        };
        Ok(Some((target, draft)))
    })();
    device.drain()?;
    Ok(built
        .ok()
        .flatten()
        .map(|(target, draft)| cached(state, target, draft)))
}

fn cached(
    state: &Sequence,
    target: ProgramSnapshot,
    draft: Option<DraftSnapshot>,
) -> prefix::CachedPrefix {
    let bytes = target
        .bytes
        .saturating_add(draft.as_ref().map_or(0, |d| d.program.bytes));
    prefix::CachedPrefix {
        tokens: state.history.clone(),
        value: prefix::Snapshot { target, draft },
        bytes,
    }
}

/// All storage needed before publishing a load-time pooled serving geometry.
pub(super) struct PoolPlan {
    pub(super) quote: PooledServingQuote,
    pub(super) memory_bound: usize,
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
        self.run_burst(tasks, &mut outputs, snapshots).await?;
        let mut index = 0;
        while index < tasks.len() {
            if outputs[index].is_some() {
                index += 1;
                continue;
            }
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
        let request_tokens = self
            .states
            .values()
            .filter(|state| {
                !state.poisoned
                    && state.readout == OutputReadout::Logits
                    && state.capacity <= capacity
            })
            .map(|state| state.capacity)
            .max()
            .unwrap_or(capacity);
        self.warm_slot_pool_for(capacity, request_tokens);
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
        self.warm_slot_pool_for(capacity, 0);
    }

    pub(super) fn warm_slot_pool_for(&mut self, capacity: usize, request_tokens: usize) {
        if self.maximum_states < 2 && self.dflash.is_none() {
            return;
        }
        let maximum = self.maximum_states.min(MAX_POOL_SLOTS);
        let (envelope, baseline) = match self.pool_funding() {
            Ok(budget) => budget,
            Err(error) => {
                self.pool_failure_reason = Some(error.to_string());
                return;
            }
        };
        if self.serial_roofline(capacity, maximum, envelope) {
            return;
        }
        let mut funding = envelope;
        // Capture can materialize driver/graph storage absent from tensor quotes.
        // Retry against a smaller quote, never against a larger serving envelope.
        for attempt in 0..POOL_CAPTURE_ATTEMPTS {
            let selected = self.pooled_plan_for(capacity, maximum, funding, request_tokens);
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
                .capture_pool_plan(attempt, capacity, (baseline, envelope), &plan)
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
                self.memory_bound = plan.memory_bound;
                self.pool_quote = Some(plan.quote);
                self.pool_envelope = funding;
                self.pool_retry = None;
                self.deferred_pool_plan = None;
                self.slots_disabled = false;
                self.pool_failure_reason = None;
                self.log_pool_memory();
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

    fn serial_roofline(&mut self, capacity: usize, maximum: usize, envelope: u64) -> bool {
        if self.roofline_bound != Some(1) || self.dflash.is_some() {
            return false;
        }
        match self.memory_plan(capacity, maximum, envelope, 0) {
            Ok(plan) => self.memory_bound = plan.map_or(1, |plan| plan.quote.width),
            Err(error) => {
                self.pool_failure_reason = Some(error.to_string());
                tracing::warn!(target:"infer::executor",%error,"serial roofline memory quote unavailable");
            }
        }
        true
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
            tracing::warn!(target: "infer::executor", attempt,
                reason = self.pool_failure_reason.as_deref().unwrap_or("unknown"),
                "pooled CUDA capture failed; decoding serially");
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
        (baseline, envelope): (u64, u64),
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
        self.attach_dflash()?;
        self.build_homes(attempt, prompt_width);
        self.build_burst(baseline, envelope);
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
        self.pool_quote = None;
        self.worker = None;
        self.homes.clear();
        if let Some(engine) = self.dflash.as_mut() {
            engine.detach_pool();
        }
        drop(self.draft_slots.take());
        drop(self.slots.take());
        let _ = self.loaded.reclaim_idle_prompt_scratch();
    }

    /// Capture the requested draft's proposal before publishing the serving pool.
    fn attach_dflash(&mut self) -> Result<()> {
        let (Some(engine), Some(pool)) = (self.dflash.as_mut(), self.slots.as_ref()) else {
            return Ok(());
        };
        pool.verify_features()
            .ok_or_else(|| Error::invariant("pooled verification carries no draft features"))
            .and_then(|graphs| engine.attach_pool(self.loaded.device(), &graphs))?;
        tracing::info!(target: "infer::executor", width = pool.width(),
            "DFlash2 pooled proposal captured");
        Ok(())
    }

    /// Quote all resources before capturing; retain the largest prompt rung at a width.
    pub(super) fn pooled_plan_for(
        &self,
        capacity: usize,
        maximum: usize,
        envelope: u64,
        request_tokens: usize,
    ) -> Result<Option<PoolPlan>> {
        let Some(plan) = self.memory_plan(capacity, maximum, envelope, request_tokens)? else {
            return Ok(None);
        };
        let selected = self
            .roofline_bound
            .filter(|_| self.dflash.is_none())
            .unwrap_or(maximum)
            .min(plan.quote.width);
        if selected == plan.quote.width {
            return Ok(Some(plan));
        }
        let Some(mut narrower) = self.memory_plan(capacity, selected, envelope, request_tokens)?
        else {
            return Ok(None);
        };
        narrower.memory_bound = plan.memory_bound;
        Ok(Some(narrower))
    }
    fn memory_plan(
        &self,
        capacity: usize,
        maximum: usize,
        envelope: u64,
        request_tokens: usize,
    ) -> Result<Option<PoolPlan>> {
        let configured = self.loaded.pooled_prompt_width();
        // Preserve wide-prompt throughput: at most one halving, and never cross the
        // established 1024-row worker floor. Deeper shrinking saves memory but adds replays.
        let minimum_prompt = (configured / 2).max(WIDER_PREFILL_LANES).min(configured);
        let minimum = if self.dflash.is_some() { 1 } else { 2 };
        // Funded widths snap to a ladder: free memory differs between the tuning worker and the
        // server, and every distinct width is a distinct set of measured shapes. Below one step
        // and at the requested maximum the exact width is kept.
        let ladder = move |width: &usize| {
            *width == maximum || *width < POOL_WIDTH_STEP || width.is_multiple_of(POOL_WIDTH_STEP)
        };
        for width in (minimum..=maximum).rev().filter(ladder) {
            let mut prompt_width = configured;
            loop {
                if let Some(quote) = self.loaded.funded_pooled_serving(
                    width,
                    capacity,
                    paged_tokens(width, capacity),
                    request_tokens,
                    prompt_width,
                    envelope,
                )? {
                    return Ok(Some(PoolPlan {
                        quote,
                        memory_bound: width,
                    }));
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
            Ok((program, draft)) => {
                self.worker = Some(super::Worker {
                    program,
                    draft,
                    bursts: Vec::new(),
                });
            }
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

    /// Capture the multi-sequence prompt graph over the resident slots. Its absence only means
    /// concurrent prompts replay one by one on the worker.
    fn build_burst(&mut self, baseline: u64, envelope: u64) {
        let Some(pool) = self.slots.as_ref() else {
            return;
        };
        if self.worker.is_none()
            || pool.verify().is_some()
            || self.draft_slots.is_some()
            || self.dflash.is_some()
            || self.homes.iter().any(Option::is_none)
        {
            return;
        }
        let capacity = pool.capacity();
        let width = pool.width();
        let remaining = self.physical_available().map_or(0, |free| {
            envelope.saturating_sub(baseline.saturating_sub(free))
        });
        let mut programs: Vec<&mut DeviceProgram> = self
            .homes
            .iter_mut()
            .flatten()
            .map(|home| &mut home.program)
            .collect();
        // The quote funded these graphs; never let the serving pool shrink to pay for them.
        let rungs = prefill_burst::rungs(width);
        let Some(worker) = self.worker.as_mut() else {
            return;
        };
        let worker_rows = worker.program.prefill_width();
        let required = self
            .loaded
            .prompt_burst_bytes(width, worker_rows)
            .unwrap_or(u64::MAX);
        if required > remaining {
            tracing::info!(target: "infer::executor", required, remaining,
                "serving envelope leaves no room for multi-sequence prompt graphs");
            return;
        }
        tracing::info!(target: "infer::executor", required, remaining,
            "capturing multi-sequence prompt graphs");
        let bursts = capture_bursts(&self.loaded, capacity, &mut programs, worker, &rungs);
        worker.bursts = bursts;
        // An estimate can miss capture overhead; the serving pool must not pay for that.
        let free = self
            .loaded
            .device()
            .allocatable_memory_bytes()
            .map(|free| free.saturating_sub(self.loaded.profile().device_headroom_bytes()));
        if free.is_ok_and(|free| baseline.saturating_sub(free) > envelope) {
            tracing::info!(target: "infer::executor", envelope,
                "multi-sequence prompt graphs exceeded the envelope; prompts replay on the worker");
            worker.bursts.clear();
        }
    }

    /// Whether this task's whole input fits one segment of the multi-sequence prompt graph.
    fn burst_eligible(&self, task: &ExecutionTask) -> Option<usize> {
        if task.media.is_some() || matches!(task.tokens, ExecutionInput::Decode { .. }) {
            return None;
        }
        let state = self.states.get(&task.state)?;
        let home = state.home?;
        let count = task.tokens.delta(state.history.len()).ok()?.len();
        (state.slot.is_none()
            && !state.poisoned
            && state.speculation.is_none()
            && state.draft.is_none()
            && state.media_prompt_len == 0
            && state.readout == OutputReadout::Logits
            && task.tokens.readout() != OutputReadout::Full
            && (1..=SEGMENT_ROWS).contains(&count))
        .then_some(home)
    }

    /// Replay every eligible prompt of this step through one graph, reading each weight once.
    /// A step whose prompts would mostly pad their rung stays on the narrower worker graph.
    async fn run_burst(
        &mut self,
        tasks: &[ExecutionTask],
        outputs: &mut [Option<TaskOutput>],
        snapshots: &mut Vec<prefix::CachedPrefix>,
    ) -> Result<()> {
        if self.worker.as_ref().is_none_or(|w| w.bursts.is_empty()) {
            return Ok(());
        }
        let selected: Vec<(usize, usize)> = tasks
            .iter()
            .enumerate()
            .filter_map(|(index, task)| Some((index, self.burst_eligible(task)?)))
            .collect();
        let required = selected
            .iter()
            .map(|&(_, slot)| slot + 1)
            .max()
            .unwrap_or(0);
        let rung = self.worker.as_ref().and_then(|worker| {
            worker
                .bursts
                .iter()
                .position(|burst| burst.segments() >= required)
                .filter(|&rung| burst_pays(selected.len(), worker.bursts[rung].segments()))
        });
        let Some(rung) = rung else {
            return Ok(());
        };
        let mut owned = Vec::with_capacity(selected.len());
        for &(index, segment) in &selected {
            let state = self
                .states
                .get_mut(&tasks[index].state)
                .ok_or_else(|| Error::invariant("validated CUDA state"))?;
            let position = state.history.len();
            if position == 0 {
                state.program.reset_async().await?;
            }
            owned.push((
                segment,
                tasks[index].tokens.delta(position)?.to_vec(),
                position,
            ));
        }
        let lanes: Vec<_> = owned
            .iter()
            .map(|(segment, tokens, position)| prefill_burst::Lane {
                segment: *segment,
                tokens,
                position: *position,
            })
            .collect();
        let device = self.loaded.device();
        let burst = self
            .worker
            .as_mut()
            .and_then(|worker| worker.bursts.get_mut(rung))
            .ok_or_else(|| Error::invariant("prompt burst"))?;
        let logits =
            super::profiling::phase_async("prefill_burst", burst.run_async(device, &lanes)).await?;
        for ((&(index, slot), lane), logits) in selected.iter().zip(&lanes).zip(logits) {
            let task = &tasks[index];
            let state = self
                .states
                .get_mut(&task.state)
                .ok_or_else(|| Error::invariant("validated CUDA state"))?;
            state.program.note_burst(lane.position + lane.tokens.len());
            task.tokens.commit(&mut state.history)?;
            let output = state.output(task.tokens.readout(), None, logits);
            if let Some(covered) = reusable_prefix(state, task)
                && let Some(snapshot) = slot_prefix(
                    self.loaded.device(),
                    &self.loaded,
                    (self.slots.as_ref(), self.draft_slots.as_ref()),
                    state,
                    (slot, covered),
                )?
            {
                snapshots.push(snapshot);
            }
            outputs[index] = Some(TaskOutput {
                request: task.request,
                output,
            });
        }
        Ok(())
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
        let minimum = if self.dflash.is_some() { 1 } else { 2 };
        for width in (minimum..=maximum).rev() {
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
            (pool, self.draft_slots.as_mut(), self.dflash.as_mut()),
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
        let draft = self.dflash.as_mut();
        let output = match self.worker.as_mut() {
            Some(worker) if wide => run_on_worker(state, worker, task, &self.loaded, draft).await?,
            _ => state.run(task, &self.loaded, draft).await?,
        };
        let snapshot = match (reusable_prefix(state, task), state.home) {
            (Some(covered), Some(slot)) => slot_prefix(
                self.loaded.device(),
                &self.loaded,
                (self.slots.as_ref(), self.draft_slots.as_ref()),
                state,
                (slot, covered),
            )?,
            (Some(covered), None) => {
                cacheable_prefix(self.loaded.device(), &self.loaded, state, covered).await
            }
            (None, _) => None,
        };
        if let Some(snapshot) = snapshot {
            snapshots.push(snapshot);
        }
        outputs[index] = Some(TaskOutput {
            request: task.request,
            output,
        });
        Ok(())
    }
}

/// Every rung over the first programs, widest first so narrower rungs view its storage. Any
/// failure leaves no rung: prompts then replay on the worker.
#[expect(
    unsafe_code,
    reason = "The worker owning these graphs is cleared before the slot pool drops"
)]
fn capture_bursts(
    loaded: &LoadedModel,
    capacity: usize,
    programs: &mut [&mut DeviceProgram],
    worker: &mut super::Worker,
    rungs: &[usize],
) -> Vec<PromptBurst> {
    let mut bursts: Vec<PromptBurst> = Vec::new();
    for &segments in rungs.iter().rev() {
        let (nvfp4, worker_external) = worker.program.burst_shared();
        let shared = prefill_burst::Shared {
            nvfp4,
            wider: bursts.first(),
            worker_external,
        };
        // SAFETY: see the expectation above; every rung lives with the worker.
        let built = unsafe { loaded.prompt_burst(capacity, &mut programs[..segments], shared) };
        match built {
            Ok(burst) => bursts.push(burst),
            Err(error) => {
                tracing::info!(target: "infer::executor", segments, %error,
                    "multi-sequence prompt graph unavailable; prompts replay on the worker");
                return Vec::new();
            }
        }
    }
    bursts.reverse();
    bursts
}

const fn burst_pays(prompts: usize, segments: usize) -> bool {
    prompts >= MIN_BURST_PROMPTS && prompts * MAX_SEGMENTS_PER_PROMPT >= segments
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
    draft: Option<&mut Engine>,
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
    let output = state.run(task, loaded, draft).await;
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

/// Physical KV positions a pool of `width` slots of `capacity` tokens shares.
const fn paged_tokens(width: usize, capacity: usize) -> usize {
    width * capacity
}

async fn slot_replay(
    pools: (&mut SlotPool, Option<&mut SlotPool>, Option<&mut Engine>),
    states: &mut BTreeMap<StateId, Sequence>,
    device: &CudaDevice,
    tasks: &[ExecutionTask],
    participants: &[(usize, usize)],
    newly: &[(usize, usize)],
    outputs: &mut [Option<TaskOutput>],
) -> Result<()> {
    let (pool, draft, dflash) = pools;
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
            (pool, draft, dflash),
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
        // Media prompts use an independent rotary coordinate after their visual span.
        lanes.push((slot, token, state.rotary_at(position), position));
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
    pools: (&mut SlotPool, Option<&mut SlotPool>, Option<&mut Engine>),
    states: &mut BTreeMap<StateId, Sequence>,
    device: &CudaDevice,
    tasks: &[ExecutionTask],
    participants: &[(usize, usize)],
    outputs: &mut [Option<TaskOutput>],
    width: usize,
) -> Result<()> {
    let (pool, mut draft, dflash) = pools;
    let mut dflash = dflash.filter(|engine| engine.attached());
    let proposed = match dflash.as_deref_mut() {
        Some(engine) => Some(
            super::profiling::phase_async(
                "draft_propose",
                dflash_chains(engine, states, device, tasks, participants, width),
            )
            .await?,
        ),
        None => None,
    };
    let batched = match draft.as_deref_mut() {
        _ if proposed.is_some() => proposed,
        Some(draft) => {
            super::profiling::phase_async("draft_propose", async {
                super::drafting::propose(draft, states, device, tasks, participants, width).await
            })
            .await?
        }
        None => None,
    };
    let defer_draft = batched.is_some() && dflash.is_none();
    let candidates = match batched {
        Some(candidates) => candidates,
        None => serial_candidates(states, tasks, participants, width).await?,
    };
    let mut lanes = Vec::new();
    for (&(index, slot), chain) in participants.iter().zip(&candidates) {
        let state = &states[&tasks[index].state];
        let position = state.history.len();
        for (offset, &token) in chain.iter().enumerate() {
            let cursor = position + offset;
            lanes.push((slot, offset, token, state.rotary_at(cursor), cursor));
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
    let mut verified = Vec::with_capacity(participants.len());
    for (&(index, slot), chain) in participants.iter().zip(candidates) {
        let state = states
            .get_mut(&tasks[index].state)
            .ok_or_else(|| Error::invariant("slot speculation state"))?;
        let position = state.history.len();
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
        if let Some(run) = state.verified_run(slot * width, position) {
            verified.push(run);
        }
        outputs[index] = Some(TaskOutput {
            request: tasks[index].request,
            output,
        });
    }
    pool.commit_verify_many(device, &commits)?;
    if let Some(engine) = dflash
        && !verified.is_empty()
    {
        engine.ingest_verified(device, &verified, pool.verified_slots())?;
    }
    if defer_draft && draft.is_some() {
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

/// Each participant's own draft chain, proposed one sequence at a time.
async fn serial_candidates(
    states: &mut BTreeMap<StateId, Sequence>,
    tasks: &[ExecutionTask],
    participants: &[(usize, usize)],
    width: usize,
) -> Result<Vec<Vec<u32>>> {
    let mut candidates = Vec::with_capacity(participants.len());
    for &(index, _) in participants {
        let state = states
            .get_mut(&tasks[index].state)
            .ok_or_else(|| Error::invariant("slot speculation state"))?;
        candidates.push(state.slot_candidates(&tasks[index], width).await?);
    }
    Ok(candidates)
}

/// One `DFlash2` block per eligible participant: the submitted token followed by the draft's
/// proposals, cut to the sequence's remaining capacity. Other participants verify their
/// submitted token alone.
async fn dflash_chains(
    engine: &mut Engine,
    states: &mut BTreeMap<StateId, Sequence>,
    device: &CudaDevice,
    tasks: &[ExecutionTask],
    participants: &[(usize, usize)],
    width: usize,
) -> Result<Vec<Vec<u32>>> {
    let mut chains = Vec::with_capacity(participants.len());
    let mut requests = Vec::new();
    let mut lengths = Vec::new();
    for (row, &(index, slot)) in participants.iter().enumerate() {
        let task = &tasks[index];
        let ExecutionInput::Decode { token, position } = task.tokens else {
            return Err(Error::invariant("DFlash2 decode task"));
        };
        let state = states
            .get_mut(&task.state)
            .ok_or_else(|| Error::invariant("DFlash2 state"))?;
        chains.push(vec![token]);
        let capacity = state.capacity;
        let Some(draft) = state.draft.as_mut() else {
            continue;
        };
        draft.prompt_len.get_or_insert(position);
        let count = (width - 1).min(capacity.saturating_sub(position + 1));
        if draft.stale
            || draft.cursor.context != position
            || count == 0
            || !task
                .sampling
                .as_ref()
                .is_some_and(|sampling| sampling.temperature == 0.0)
        {
            continue;
        }
        requests.push((slot, draft.cursor.ring, token, position));
        lengths.push((row, count));
    }
    if requests.is_empty() {
        return Ok(chains);
    }
    let proposals = engine.propose(device, &requests).await?;
    for ((row, count), proposal) in lengths.into_iter().zip(proposals) {
        chains[row].extend_from_slice(&proposal[..count.min(proposal.len())]);
    }
    Ok(chains)
}

/// The longest candidate prefix the target selects itself, stopping after an end token.
fn accept(
    chain: &[u32],
    rows: &[(Vec<f32>, Vec<f32>)],
    tokens: Option<&[Option<u32>]>,
    sampler: &mut Sampler<'_>,
    position: usize,
    generated: &mut Vec<u32>,
) -> Result<Vec<u32>> {
    let mut accepted = Vec::new();
    for (offset, &candidate) in chain[1..].iter().enumerate() {
        let selected = match tokens {
            Some(tokens) => {
                tokens[offset].ok_or_else(|| Error::invariant("verification selected token"))?
            }
            None => sampler.pick(&rows[offset].1, position + offset + 1, generated)?,
        };
        if selected != candidate {
            break;
        }
        accepted.push(candidate);
        generated.push(candidate);
        if sampler.sampling.is_eos(candidate) {
            break;
        }
    }
    Ok(accepted)
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

    /// The submitted token and every accepted candidate of a verify step that began at
    /// `position` are now target history: their feature rows, from verify lane `lane` on,
    /// become draft context. A ring that missed a position goes stale instead.
    fn verified_run(&mut self, lane: usize, position: usize) -> Option<VerifiedRows> {
        let committed = self.history.len();
        let draft = self.draft.as_mut()?;
        if draft.stale || draft.cursor.context != position {
            draft.stale = true;
            return None;
        }
        draft.cursor.context = committed;
        Some((lane, draft.cursor.ring, position, committed - position))
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
        if let Some(draft) = self.draft.as_mut().filter(|_| chain.len() > 1) {
            let prompt_len = *draft.prompt_len.get_or_insert(position);
            if let Some(sampling) = task.sampling.as_ref() {
                let mut generated = self.history[prompt_len..].to_vec();
                let mut sampler = Sampler {
                    sampling,
                    request: task.request.get(),
                    prompt: &self.history[..prompt_len],
                    scratch: &mut draft.scratch,
                };
                accepted = accept(chain, rows, tokens, &mut sampler, position, &mut generated)?;
            }
        } else if let Some(spec) = self.speculation.as_mut() {
            let prompt_len = *spec.prompt_len.get_or_insert(position);
            if let Some(sampling) = task.sampling.as_ref() {
                let mut generated = self.history[prompt_len..].to_vec();
                let mut sampler = Sampler {
                    sampling,
                    request: task.request.get(),
                    prompt: &self.history[..prompt_len],
                    scratch: &mut spec.scratch,
                };
                accepted = accept(chain, rows, tokens, &mut sampler, position, &mut generated)?;
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

    async fn run(
        &mut self,
        task: &ExecutionTask,
        loaded: &LoadedModel,
        mut engine: Option<&mut Engine>,
    ) -> Result<ModelOutput> {
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
        // A DFlash2 ring follows a prompt only through the prompt graph's feature rows, so every
        // prompt chunk of a drafting sequence replays the prompt graph. A serial decode step
        // leaves a gap instead: the sequence then decodes without proposals.
        let ingest = engine.is_some()
            && !matches!(task.tokens, ExecutionInput::Decode { .. })
            && self.program.prefill_width() >= PREFILL_LANES
            && self.draft.as_ref().is_some_and(|draft| !draft.stale);
        while offset < tokens.len() {
            let count = self.chunk_len(tokens.len() - offset);
            let position = self.history.len() + offset;
            let last = offset + count == tokens.len();
            let read_logits = last && readout != OutputReadout::None;
            let device_prime = speculate && count > 1 && self.readout != OutputReadout::Full;
            let read_hidden = self.readout == OutputReadout::Full || (speculate && !device_prime);
            let results = if count > 1 || ingest {
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
            if ingest {
                self.ingest_chunk(engine.as_deref_mut(), loaded.device(), count, position)
                    .await?;
            } else if let Some(draft) = self.draft.as_mut() {
                draft.stale = true;
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

    /// Turn the prompt graph's feature rows of the chunk just replayed into draft context.
    async fn ingest_chunk(
        &mut self,
        engine: Option<&mut Engine>,
        device: &CudaDevice,
        count: usize,
        position: usize,
    ) -> Result<()> {
        let features = self.program.prompt_features(count);
        let (Some(engine), Some(draft), Some(features)) = (engine, self.draft.as_mut(), features)
        else {
            if let Some(draft) = self.draft.as_mut() {
                draft.stale = true;
            }
            return Ok(());
        };
        if draft.stale || draft.cursor.context != position {
            draft.stale = true;
            return Ok(());
        }
        super::profiling::phase_async(
            "draft_ingest",
            engine.ingest_prompt(device, &features, count, &mut draft.cursor, position),
        )
        .await
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
            self.prepare_media(plan, loaded).await?;
        }
        // On error, keep the source with the sequence until the submission fences or is isolated.
        let media = self
            .media
            .as_ref()
            .ok_or_else(|| Error::invalid("media prefill did not start at position zero"))?;
        if media.bases.len() != plan.items.len() {
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
            self.media = None;
        }
        task.tokens.commit(&mut self.history)?;
        Ok(self.output(readout, None, logits))
    }

    async fn prepare_media(&mut self, plan: &MediaPlan, loaded: &LoadedModel) -> Result<()> {
        let tower = loaded
            .vision()
            .ok_or_else(|| Error::unsupported("model has no bound vision encoder"))?;
        let layout = media_plan_layout(plan, loaded.imported())?;
        if layout.spans.len() != plan.items.len() {
            return Err(Error::invalid("media layout does not cover every item"));
        }
        let hidden = loaded.model().hidden_size;
        let mut data = Vec::new();
        let mut bases = Vec::with_capacity(plan.items.len());
        for image in &plan.items {
            let encoded = match &image.pixels {
                MediaPixels::Patches(pixels) => {
                    tower.encode_pixels(loaded.device(), image.grid, pixels)?
                }
                MediaPixels::Image(input) => {
                    tower
                        .encode_image(loaded.device(), image.grid, input)
                        .await?
                }
                MediaPixels::EncodedImage(input) => {
                    tower
                        .encode_encoded_image(loaded.device(), image.grid, input)
                        .await?
                }
            };
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
    draft: &mut DeviceProgram,
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
pub(super) fn pooled_envelope(physical: u64, budget: u64, active: u64, scratch: u64) -> u64 {
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

#[cfg(test)]
#[path = "../../tests/unit/executor_prompt_burst.rs"]
mod prompt_burst_tests;
