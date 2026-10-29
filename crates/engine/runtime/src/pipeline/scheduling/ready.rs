use crate::{Engine, RequestRecord};
use infer_core::event::CpuStage;
use infer_core::{Error, RequestId, Result};
use infer_ir::{ExecutionRole, PageGrowth, ReadyWork, Workload};
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::mem::take;
use std::time::Instant;

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    fn sync_ready_deltas(&mut self) -> Result<()> {
        let dirty = self.host.requests.take_dirty();
        let result = dirty.iter().copied().try_for_each(|slot| {
            self.host.requests.sync(slot, &self.state)?;
            self.host.ready_index.invalidate(slot)
        });
        self.host.requests.restore_dirty(dirty);
        result
    }
    pub(crate) fn ready_work_into(&mut self, ready: &mut super::index::ReadyWindow) -> Result<()> {
        let start = Instant::now();
        self.sync_ready_deltas()?;
        self.fill_candidate_ids()?;
        ready.count = 0;
        let candidates = take(&mut self.host.ready_index.candidates);
        let result = candidates.iter().copied().try_for_each(|id| {
            if self.preemption_focus.is_some_and(|focus| focus != id) {
                return Ok(());
            }
            let slot = self
                .host
                .requests
                .slot(id)
                .ok_or_else(|| Error::invariant("candidate slot missing"))?;
            if self.host.ready_index.versions[slot] != self.host.cost_epoch {
                let tenant = self.host.ready_index.rows[slot]
                    .as_mut()
                    .map(|r| take(&mut r.tenant))
                    .unwrap_or_default();
                let row = self.ready_request_with_tenant(
                    id,
                    self.host.requests.known(id)?,
                    Some(tenant),
                )?;
                self.host.ready_index.rows[slot] = Some(row);
                self.host.ready_index.versions[slot] = self.host.cost_epoch;
            }
            let row = self.host.ready_index.rows[slot]
                .as_mut()
                .ok_or_else(|| Error::invariant("candidate projection missing"))?;
            row.virtual_finish = self
                .tenants
                .get(row.tenant.as_str())
                .ok_or_else(|| Error::invariant("tenant missing"))?
                .virtual_finish;
            // Prefix/COW reference counts can change without mutating this request.
            row.cost_query.page_growth = self.backend.state_page_growth(row.state)?;
            row.cost_query.prefill_geometry = if row.role == ExecutionRole::Prefill {
                self.backend.state_prefill_geometry(row.state)?
            } else {
                None
            };
            row.cost_query.logical_growth = Some(PageGrowth {
                block_size: self.config.block_size,
                allocated_pages: self.state.get(row.state)?.pages.len(),
                bytes_per_page: 0,
                cow_tail: false,
            });
            ready.write(row)?;
            Ok(())
        });
        self.host.ready_index.candidates = candidates;
        self.cpu_stage(CpuStage::Ready, 0, start, ready.count);
        result
    }
    pub(crate) fn fill_candidate_ids(&mut self) -> Result<()> {
        self.host.queues.candidates_into(
            &mut self.host.ready_index.candidates,
            self.config.candidate_limit.min(self.config.max_requests),
            self.now_us,
            self.config.scheduler.max_wait_us,
            self.config.cpu.urgency_window_us,
        )?;
        if let Some(focus) = self.preemption_focus
            && self.host.queues.state(focus) == Some(infer_scheduler::QueueState::Ready)
            && !self.host.ready_index.candidates.contains(&focus)
        {
            self.host.ready_index.candidates.pop();
            self.host.ready_index.candidates.push(focus);
        }
        Ok(())
    }
    pub(crate) fn ready_request(&self, id: RequestId, r: &RequestRecord) -> Result<ReadyWork> {
        self.ready_request_with_tenant(id, r, Some(String::new()))
    }
    fn ready_request_with_tenant(
        &self,
        id: RequestId,
        r: &RequestRecord,
        tenant: Option<String>,
    ) -> Result<ReadyWork> {
        let tenant = tenant.map_or_else(String::new, |mut tenant| {
            tenant.clone_from(&r.request.qos.tenant);
            tenant
        });
        let generate = matches!(r.request.workload, Workload::Generate { .. });
        let role = if generate && r.prefill_done == r.prefill_target {
            ExecutionRole::Decode
        } else if generate {
            ExecutionRole::Prefill
        } else {
            ExecutionRole::Forward
        };
        let remaining = if role == ExecutionRole::Decode {
            1
        } else {
            r.prefill_end() - r.prefill_done
        };
        let context = if role == ExecutionRole::Decode {
            r.context.len()
        } else {
            r.prefill_done + 1
        };
        let state = r
            .state
            .ok_or_else(|| Error::invariant("runnable request has no state"))?;
        let mut cost_query = self.base_query(role, 1, context);
        cost_query.page_growth = self.backend.state_page_growth(state)?;
        cost_query.prefill_geometry = if role == ExecutionRole::Prefill {
            self.backend.state_prefill_geometry(state)?
        } else {
            None
        };
        cost_query.logical_growth = Some(PageGrowth {
            block_size: self.config.block_size,
            allocated_pages: self.state.get(state)?.pages.len(),
            bytes_per_page: 0,
            cow_tail: false,
        });
        let cost_per_token = self.costs.estimate(&[cost_query])?;
        let mut latency_query = cost_query;
        latency_query.tokens = remaining;
        latency_query.context_tokens = r.context.len();
        let latency = self.costs.estimate(&[latency_query])?.gpu_us;
        let complete = self.remaining_completion_cost(r, latency)?;
        let latency_deadline_us = if generate && r.first_token_us.is_none() {
            r.request
                .qos
                .ttft_slo_us
                .map(|s| r.accepted_us.saturating_add(s))
        } else if generate {
            r.request
                .qos
                .tpot_slo_us
                .and_then(|s| r.last_token_us.map(|last| last.saturating_add(s)))
        } else {
            None
        };
        Ok(ReadyWork {
            request: id,
            program: r.plan.program,
            state,
            role,
            remaining_tokens: remaining,
            tenant,
            weight: r.request.qos.weight,
            deadline_us: r.request.qos.deadline_us,
            virtual_finish: self
                .tenants
                .get(r.request.qos.tenant.as_str())
                .ok_or_else(|| Error::invariant("tenant missing"))?
                .virtual_finish,
            cost_per_token,
            cost_query,
            latency_deadline_us,
            remaining_latency_us: latency,
            remaining_completion_us: complete,
            last_service_us: r.last_service_us,
        })
    }
    pub(crate) fn remaining_completion_cost(&self, r: &RequestRecord, latency: u64) -> Result<u64> {
        let complete = if let Workload::Generate { max_new_tokens } = r.request.workload {
            let decode = self
                .costs
                .estimate(&[self.base_query(ExecutionRole::Decode, 1, r.plan.reserved_tokens)])?
                .gpu_us;
            latency.saturating_add(
                decode.saturating_mul(max_new_tokens.saturating_sub(r.generated.len() + 1) as u64),
            )
        } else {
            r.plan
                .units
                .iter()
                .skip(r.unit + 1)
                .try_fold(latency, |sum, unit| {
                    let c = self
                        .costs
                        .estimate(&[self.base_query(
                            ExecutionRole::Forward,
                            unit.len(),
                            unit.len(),
                        )])?
                        .gpu_us;
                    sum.checked_add(c)
                        .ok_or_else(|| Error::invalid("remaining latency overflow"))
                })?
        };
        Ok(complete)
    }
    fn queue_timing(&self, id: RequestId) -> Result<infer_scheduler::QueueTiming> {
        let r = self.host.requests.known(id)?;
        let generate = matches!(r.request.workload, Workload::Generate { .. });
        let phase = if !generate {
            ExecutionRole::Forward
        } else if r.prefill_done == r.prefill_target {
            ExecutionRole::Decode
        } else {
            ExecutionRole::Prefill
        };
        let has_latency_target = generate
            && if r.first_token_us.is_none() {
                r.request.qos.ttft_slo_us.is_some()
            } else {
                r.request.qos.tpot_slo_us.is_some() && r.last_token_us.is_some()
            };
        let latest_start_us = if r.request.qos.deadline_us.is_some() || has_latency_target {
            let forecast = self.ready_request_with_tenant(id, r, None)?;
            infer_scheduler::latest_start(
                forecast.deadline_us,
                forecast.remaining_completion_us,
                forecast.latency_deadline_us,
                forecast.remaining_latency_us,
            )
        } else {
            None
        };
        Ok(infer_scheduler::QueueTiming {
            phase,
            last_service_us: r.last_service_us,
            latest_start_us,
            wait_deadline_us: r
                .last_service_us
                .saturating_add(self.config.max_queue_wait_us),
        })
    }
    /// Refresh a bounded identity window after calibration changes. Newly ready requests
    /// already receive current forecasts in their lifecycle transition. New feedback
    /// does not restart an active sweep: every ready request gets refreshed even
    /// during continuous calibration. A changed epoch triggers another full sweep.
    pub(crate) fn refresh_queue_priorities(&mut self) -> Result<()> {
        if self.host.ready_index.priority_epoch == self.host.cost_epoch {
            return Ok(());
        }
        if self.host.ready_index.priority_cursor.is_none() {
            self.host.ready_index.priority_sweep_epoch = self.host.cost_epoch;
        }
        let mut ids = take(&mut self.host.wake_buffer);
        ids.clear();
        ids.extend(
            self.host
                .queues
                .ready_ids_after(self.host.ready_index.priority_cursor)
                .take(self.config.cpu.maintenance_items),
        );
        let result = ids.iter().copied().try_for_each(|id| {
            let timing = self.queue_timing(id)?;
            self.host.queues.update_priority(id, timing.latest_start_us)
        });
        if result.is_ok() {
            self.host.ready_index.priority_cursor = ids.last().copied();
            if ids.len() < self.config.cpu.maintenance_items {
                self.host.ready_index.priority_epoch = self.host.ready_index.priority_sweep_epoch;
                self.host.ready_index.priority_cursor = None;
            }
        }
        self.host.wake_buffer = ids;
        result
    }

    pub(crate) fn queue_request(&self, id: RequestId) -> Result<infer_scheduler::QueueRequest> {
        let r = self.host.requests.known(id)?;
        let timing = self.queue_timing(id)?;
        Ok(infer_scheduler::QueueRequest {
            request: id,
            phase: timing.phase,
            tenant: r.tenant.clone(),
            virtual_finish: self
                .tenants
                .get(r.request.qos.tenant.as_str())
                .ok_or_else(|| Error::invariant("tenant missing"))?
                .virtual_finish,
            last_service_us: timing.last_service_us,
            latest_start_us: timing.latest_start_us,
            hard_deadline_us: r.request.qos.deadline_us,
            wait_deadline_us: timing.wait_deadline_us,
        })
    }
    pub(crate) fn enqueue_request(&mut self, id: RequestId) -> Result<()> {
        self.host.queues.enqueue(self.queue_request(id)?)
    }
    pub(crate) fn refresh_queue(&mut self, id: RequestId) -> Result<()> {
        self.host.queues.refresh(id, self.queue_timing(id)?)
    }
}
