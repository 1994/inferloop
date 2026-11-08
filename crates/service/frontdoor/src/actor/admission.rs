//! Admission responsibilities.
use super::{OutputChannel, Quote, Quoting, RuntimeActor, STATE_BUDGET_RETRY_US, Subscriber};
use infer_core::{Error, ErrorCode, Result};
use infer_observe::trace::TraceContext;
use infer_runtime::EngineOutput;
use infer_spi::BackendProvider;
use std::mem::take;

/// Outcome of polling one sizing quote.
enum QuotePoll {
    /// The device owner has not replied yet.
    Pending,
    /// The owner priced the reservation; `None` means it priced no byte bound.
    Priced(Option<u64>),
}

impl<B: BackendProvider> RuntimeActor<B> {
    pub(super) fn poll_admissions(&mut self, generated: &mut Vec<EngineOutput>) {
        for _ in 0..self.quoting.len() {
            let Some(entry) = self.quoting.pop_front() else {
                break;
            };
            if entry.retry_at_us > self.now_us() {
                self.quoting.push_back(entry);
                continue;
            }
            if let Some(deferred) = self.poll_admission(entry, generated) {
                self.quoting.push_back(deferred);
            }
        }
    }
    /// Poll one queued submission; `Some` keeps it queued for a later pass.
    fn poll_admission(
        &mut self,
        entry: Quoting,
        generated: &mut Vec<EngineOutput>,
    ) -> Option<Quoting> {
        let Quoting {
            request,
            pending,
            trace,
            output,
            reply,
            mut ticket,
            mut quote,
            deadline_us,
            ..
        } = entry;
        let id = request.id();
        let abandoned = if self.stopping || reply.is_closed() || output.is_closed() {
            Some(Error::new(
                ErrorCode::Capacity,
                "admission abandoned or runtime stopping",
            ))
        } else {
            pending.check().err()
        };
        if let Some(error) = abandoned {
            self.engine.record_admission_rejection(id, &error);
            let _ = reply.send(Err(error));
            return None;
        }
        if matches!(quote, Quote::Pending) {
            match self.poll_quote(&request, &mut ticket) {
                Ok(QuotePoll::Pending) => {
                    return Some(Quoting {
                        request,
                        pending,
                        trace,
                        output,
                        reply,
                        ticket,
                        quote,
                        retry_at_us: 0,
                        deadline_us,
                    });
                }
                Ok(QuotePoll::Priced(priced)) => quote = Quote::Priced(priced),
                Err(error) => {
                    self.engine.record_admission_rejection(id, &error);
                    let _ = reply.send(Err(error));
                    return None;
                }
            }
        }
        let Quote::Priced(bytes) = quote else {
            return None;
        };
        match self.engine.admission_awaits_state(&request, bytes) {
            // The request fits the stable budget: wait for other leases to drain instead of
            // failing a turn that a momentary shortage blocks.
            Ok(true) if self.now_us() < deadline_us => {
                return Some(Quoting {
                    request,
                    pending,
                    trace,
                    output,
                    reply,
                    ticket: None,
                    quote,
                    retry_at_us: self.now_us().saturating_add(STATE_BUDGET_RETRY_US),
                    deadline_us,
                });
            }
            Ok(_) => {}
            Err(error) => {
                self.engine.record_admission_rejection(id, &error);
                let _ = reply.send(Err(error));
                return None;
            }
        }
        let admitted = self.admit(*request, bytes, trace, output);
        if let Err(error) = &admitted {
            self.engine.record_admission_rejection(id, error);
        }
        if reply.send(admitted).is_err()
            && let Ok(events) = self.engine.cancel(id)
        {
            generated.extend(events);
        }
        None
    }
    fn poll_quote(
        &mut self,
        request: &infer_runtime::PreparedRequest,
        ticket: &mut Option<infer_spi::ResourceTicket>,
    ) -> Result<QuotePoll> {
        let polled = match ticket.as_mut() {
            Some(issued) => issued.poll(),
            None => {
                // The lane was full when the quote was first issued; issue it again now.
                self.engine.try_admission_quote(request).map(|issued| {
                    *ticket = issued;
                    None
                })
            }
        };
        match polled {
            Ok(None) => Ok(QuotePoll::Pending),
            Ok(Some(infer_spi::ResourceReply::ReservationBytes(priced))) => {
                Ok(QuotePoll::Priced(priced))
            }
            Ok(Some(_)) => Err(Error::invariant("admission quote acknowledgement mismatch")),
            Err(error) => Err(error),
        }
    }
    pub(super) fn admit(
        &mut self,
        request: infer_runtime::PreparedRequest,
        bytes: Option<u64>,
        trace: Option<TraceContext>,
        output: OutputChannel,
    ) -> Result<()> {
        if self.stopping {
            return Err(Error::new(ErrorCode::Capacity, "runtime is shutting down"));
        }
        if self.subscribers.contains_key(&request.id()) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "request stream still exists",
            ));
        }
        if self.subscribers.len() >= self.engine.config().max_requests {
            return Err(Error::new(
                ErrorCode::Capacity,
                "output stream capacity exhausted",
            ));
        }
        let id = request.id();
        self.engine
            .submit_quoted_with_trace(request, bytes, trace)?;
        self.subscribers.insert(
            id,
            Subscriber {
                channel: output,
                terminal: None,
            },
        );
        Ok(())
    }
    pub(super) fn cancel_closed_streams(&mut self, generated: &mut Vec<EngineOutput>) {
        let mut scratch = take(&mut self.scratch);
        scratch.clear();
        scratch.extend(
            self.subscribers
                .iter()
                .filter(|(_, s)| s.terminal.is_none() && (self.stopping || s.channel.is_closed()))
                .map(|(id, _)| *id),
        );
        for id in &scratch {
            if let Ok(events) = self.engine.cancel(*id) {
                generated.extend(events);
            }
        }
        self.scratch = scratch;
    }
}
