//! Delivery responsibilities.
use super::RuntimeActor;
use infer_core::{CancelSource, RequestId};
use infer_runtime::EngineOutput;
use infer_spi::BackendProvider;
use std::mem::take;

impl<B: BackendProvider> RuntimeActor<B> {
    /// Release a terminal request's retained result; a failure here is an ownership invariant.
    fn drain_completed(&mut self, id: RequestId) {
        if let Err(error) = self.engine.take_completed(id) {
            tracing::error!(target: "infer::request", %error, request = id.get(), "completed request drain failed");
        }
    }
    pub(super) fn deliver(&mut self, generated: &mut Vec<EngineOutput>) {
        for event in generated.drain(..) {
            let (id, terminal) = match &event {
                EngineOutput::Token { request, .. } => (*request, false),
                EngineOutput::Finished(done) => (done.request, true),
            };
            if terminal {
                self.drain_completed(id);
            }
            let Some(subscriber) = self.subscribers.get_mut(&id) else {
                continue;
            };
            if terminal {
                subscriber.terminal = Some(Ok(event));
                continue;
            }
            if subscriber.channel.try_send(Ok(event)).is_err() {
                self.engine.record_backpressure(id);
                // Retain one terminal record until the bounded queue drains or disconnects.
                if let Ok(events) = self.engine.cancel_from(id, CancelSource::Backpressure)
                    && let Some(done) = events
                        .into_iter()
                        .find(|e| matches!(e, EngineOutput::Finished(_)))
                {
                    subscriber.terminal = Some(Ok(done));
                    self.drain_completed(id);
                }
            }
        }
    }
    pub(super) fn deliver_pending_terminals(&mut self) {
        for (id, subscriber) in &mut self.subscribers {
            if subscriber.terminal.is_none()
                && let Some(done) = self.engine.pending_terminal(*id)
            {
                subscriber.terminal = Some(Ok(EngineOutput::Finished(done)));
            }
        }
    }
    pub(super) fn reap_completed(&mut self) {
        let mut scratch = take(&mut self.scratch);
        scratch.clear();
        scratch.extend(
            self.engine
                .request_records()
                .filter(|r| r.status.terminal())
                .map(|r| r.request.id),
        );
        for id in &scratch {
            self.drain_completed(*id);
        }
        self.scratch = scratch;
    }
}
