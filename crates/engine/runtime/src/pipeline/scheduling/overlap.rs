//! At most one provisional plan overlaps the current compute flight. Publication revalidates it.
use super::index::ReadyWindow;
use crate::{Engine, EngineOutput};
use infer_core::{Error, RequestStatus, Result};
use infer_ir::SchedulingDecision;
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::mem::take;

pub struct PreparedNext {
    ready: ReadyWindow,
    decision: SchedulingDecision,
    versions: Vec<u64>,
    cost_epoch: u64,
}
impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn discard_prepared(&mut self) {
        if let Some(mut prepared) = self.host.prepared_next.take() {
            self.host.decisions.reclaim(prepared.decision);
            self.host.ready_buffer = Some(prepared.ready);
            prepared.versions.clear();
            self.host.prepare_versions = prepared.versions;
        }
    }
    pub(crate) fn prepare_next(&mut self) -> Result<()> {
        if self.host.prepared_next.is_some() || self.preemption_focus.is_some() {
            return Ok(());
        }
        let mut ready = self
            .host
            .ready_buffer
            .take()
            .ok_or_else(|| Error::invariant("overlap ready storage missing"))?;
        let result = self.ready_work_into(&mut ready).and_then(|()| {
            if ready.is_empty() {
                return Ok(None);
            }
            let resources = self.resources()?;
            let mut decision = self.build_decision(&ready, &resources)?;
            if let Err(error) = self
                .validate_decision(&decision, &ready, &resources)
                .and_then(|()| self.append_focus_deferrals(&mut decision))
            {
                self.host.decisions.reclaim(decision);
                return Err(error);
            }
            Ok(Some(decision))
        });
        match result {
            Ok(Some(decision)) if decision.step.is_some() => {
                let mut versions = take(&mut self.host.prepare_versions);
                versions.clear();
                for row in ready.iter() {
                    versions.push(self.host.requests.known(row.request)?.progress_epoch);
                }
                self.host.prepared_next = Some(PreparedNext {
                    ready,
                    decision,
                    versions,
                    cost_epoch: self.host.cost_epoch,
                });
                Ok(())
            }
            result => {
                self.host.ready_buffer = Some(ready);
                result.map(|decision| {
                    if let Some(decision) = decision {
                        self.host.decisions.reclaim(decision);
                    }
                })
            }
        }
    }
    pub(crate) fn dispatch_prepared(&mut self, emitted: &mut Vec<EngineOutput>) -> Result<bool> {
        let Some(mut prepared) = self.host.prepared_next.take() else {
            return Ok(false);
        };
        let valid = self.host.cost_epoch == prepared.cost_epoch
            && self.preemption_focus.is_none()
            && prepared
                .ready
                .iter()
                .zip(&prepared.versions)
                .all(|(row, epoch)| {
                    self.host.requests.get(row.request).is_some_and(|record| {
                        record.status == RequestStatus::Runnable
                            && record.state == Some(row.state)
                            && record.progress_epoch == *epoch
                    })
                });
        let mut decision = Some(prepared.decision);
        let result = (|| {
            if !valid {
                return Ok(false);
            }
            for row in prepared.ready.as_mut_slice() {
                row.cost_query.page_growth = self.backend.state_page_growth(row.state)?;
                row.cost_query.prefill_geometry = if row.role == infer_ir::ExecutionRole::Prefill {
                    self.backend.state_prefill_geometry(row.state)?
                } else {
                    None
                };
                row.cost_query.logical_growth = Some(infer_ir::PageGrowth {
                    block_size: self.config.block_size,
                    allocated_pages: self.state.get(row.state)?.pages.len(),
                    bytes_per_page: 0,
                    cow_tail: false,
                });
            }
            let resources = self.resources()?;
            // The preceding fence may change COW and available pages. Stale forecasts are discarded.
            if self
                .validate_decision(
                    decision
                        .as_ref()
                        .ok_or_else(|| Error::invariant("prepared decision missing"))?,
                    &prepared.ready,
                    &resources,
                )
                .is_err()
            {
                return Ok(false);
            }
            self.commit_decision(
                decision
                    .take()
                    .ok_or_else(|| Error::invariant("prepared decision missing"))?,
                &prepared.ready,
                &resources,
                emitted,
            )
            .map(|()| true)
        })();
        if let Some(decision) = decision {
            self.host.decisions.reclaim(decision);
        }
        self.host.ready_buffer = Some(prepared.ready);
        prepared.versions.clear();
        self.host.prepare_versions = prepared.versions;
        match result {
            Ok(true) => self.host.prepared_dispatched += 1,
            Ok(false) => self.host.prepared_discarded += 1,
            Err(_) => {}
        }
        result
    }
}
