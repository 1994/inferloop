use super::index;
use crate::Engine;
use infer_core::{Error, Result, event::EventKind, event::ObjectKind};
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::mem::take;

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn preempt_for_progress(&mut self, ready: &mut index::ReadyWindow) -> Result<()> {
        if self.preemption_focus.is_some_and(|id| {
            self.host
                .requests
                .get(id)
                .is_none_or(|r| r.status.terminal())
        }) {
            self.preemption_focus = None;
        }
        self.ready_work_into(ready)?;
        if ready.is_empty() || !self.backend.supports_recompute_preemption() {
            return Ok(());
        }
        let resources = self.resources()?;
        let state = &self.state;
        let Some(focus) = infer_scheduler::recompute_plan_into(
            ready,
            &resources,
            self.now_us,
            self.preemption_focus,
            |id| {
                ready
                    .iter()
                    .find(|item| item.request == id)
                    .is_some_and(|item| {
                        state
                            .get(item.state)
                            .is_ok_and(|state| !state.pages.is_empty())
                    })
            },
            self.costs.as_ref(),
            &mut self.host.preemption,
        )?
        else {
            return Ok(());
        };
        let victims = take(&mut self.host.preemption.victims);
        let result = self.preempt_victims(focus, &victims, ready);
        self.host.preemption.victims = victims;
        result
    }
    fn preempt_victims(
        &mut self,
        focus: infer_core::RequestId,
        victims: &[infer_core::RequestId],
        ready: &mut index::ReadyWindow,
    ) -> Result<()> {
        for &victim in victims {
            let r = self
                .host
                .requests
                .get_mut(victim)
                .ok_or_else(|| Error::invariant("preemption victim disappeared"))?;
            let state = r
                .state
                .ok_or_else(|| Error::invariant("runnable request has no state"))?;
            let _ = r;
            self.reset_request_state(victim)?;
            let r = self
                .host
                .requests
                .get_mut(victim)
                .ok_or_else(|| Error::invariant("preemption victim disappeared"))?;
            r.prefill_done = 0;
            r.prefix_attempted = false;
            r.prefill_target = r.context.len();
            r.preemptions = r.preemptions.saturating_add(1);
            self.preemption_focus = Some(focus);
            if r.status == infer_core::RequestStatus::Runnable {
                self.refresh_queue(victim)?;
            }
            self.event(
                EventKind::Preempted,
                ObjectKind::Request,
                victim.get(),
                focus.get(),
                state.get(),
                0,
            );
            self.progress(victim)?;
            let item = self.ready_request(focus, self.host.requests.known(focus)?)?;
            if self.any_feasible(&[item], &self.resources()?)? {
                return self.ready_work_into(ready);
            }
        }
        self.ready_work_into(ready)
    }
}
