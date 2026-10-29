//! Admission responsibilities.
use super::{Command, OutputChannel, RuntimeActor, Subscriber};
use infer_core::{Error, ErrorCode, Result};
use infer_observe::trace::TraceContext;
use infer_runtime::EngineOutput;
use infer_spi::BackendProvider;
use std::mem::take;

impl<B: BackendProvider> RuntimeActor<B> {
    pub(super) fn poll_admissions(&mut self, generated: &mut Vec<EngineOutput>) {
        for _ in 0..self.quoting.len() {
            let Some((command, mut ticket)) = self.quoting.pop_front() else {
                break;
            };
            let Command::Submit {
                request,
                pending,
                trace,
                output,
                reply,
            } = command
            else {
                continue;
            };
            let result = if self.stopping || reply.is_closed() || output.is_closed() {
                Err(Error::new(
                    ErrorCode::Capacity,
                    "admission abandoned or runtime stopping",
                ))
            } else if let Err(error) = pending.check() {
                Err(error)
            } else if let Some(issued) = ticket.as_mut() {
                issued.poll()
            } else {
                // The lane was full when the quote was first issued; issue it again now.
                self.engine.try_admission_quote(&request).map(|issued| {
                    ticket = issued;
                    None
                })
            };
            match result {
                Ok(None) => self.quoting.push_back((
                    Command::Submit {
                        request,
                        pending,
                        trace,
                        output,
                        reply,
                    },
                    ticket,
                )),
                result => {
                    let id = request.id();
                    if let Err(error) = &result {
                        self.engine.record_admission_rejection(id, error);
                    }
                    let admitted = result.and_then(|ack| match ack {
                        Some(infer_spi::ResourceReply::ReservationBytes(bytes)) => {
                            self.admit(*request, bytes, trace, output)
                        }
                        _ => Err(Error::invariant("admission quote acknowledgement mismatch")),
                    });
                    if reply.send(admitted).is_err()
                        && let Ok(events) = self.engine.cancel(id)
                    {
                        generated.extend(events);
                    }
                }
            }
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
