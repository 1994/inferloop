//! Commands responsibilities.
use super::{Command, DEFERRED_QUEUE_CAPACITY, RuntimeActor, STOP_WAITER_CAPACITY};
use infer_core::{Error, ErrorCode};
use infer_runtime::EngineOutput;
use infer_spi::BackendProvider;
use std::sync::mpsc::TryRecvError;

/// Maximum commands drained from one lane per scheduling pass.
const COMMAND_DRAIN_BATCH: usize = 64;

impl<B: BackendProvider> RuntimeActor<B> {
    pub(super) fn receive_commands(&mut self, generated: &mut Vec<EngineOutput>) {
        for _ in 0..COMMAND_DRAIN_BATCH {
            match self.control.try_recv() {
                Ok(command) => self.handle_command(command, generated),
                Err(_) => break,
            }
        }
        for _ in 0..COMMAND_DRAIN_BATCH {
            let command = self
                .deferred
                .pop_front()
                .or_else(|| self.receiver.try_recv().ok());
            let Some(command) = command else {
                break;
            };
            self.handle_command(command, generated);
        }
        // Move at most one bounded ingress batch out of the bulk lane; control remains independent.
        while self.deferred.len() < DEFERRED_QUEUE_CAPACITY {
            match self.receiver.try_recv() {
                Ok(command) => self.deferred.push_back(command),
                Err(TryRecvError::Disconnected) => {
                    self.stopping = true;
                    break;
                }
                Err(TryRecvError::Empty) => break,
            }
        }
        if self.stopping {
            for command in self.deferred.drain(..) {
                if let Command::Submit { reply, .. } = command {
                    let _ = reply.send(Err(Error::new(
                        ErrorCode::Capacity,
                        "runtime is shutting down",
                    )));
                }
            }
        }
    }
    pub(super) fn handle_command(&mut self, command: Command, generated: &mut Vec<EngineOutput>) {
        match command {
            Command::Submit {
                mut request,
                pending,
                trace,
                output,
                reply,
            } => {
                if reply.is_closed() || output.is_closed() {
                    return;
                }
                let result = pending.check().and_then(|()| {
                    if self.stopping {
                        return Err(Error::new(ErrorCode::Capacity, "runtime is shutting down"));
                    }
                    if self.engine.media_host_credit_ready(&mut request)? {
                        self.engine.try_admission_quote(&request)
                    } else {
                        Ok(None)
                    }
                });
                match result {
                    Ok(ticket) => self.quoting.push_back((
                        Command::Submit {
                            request,
                            pending,
                            trace,
                            output,
                            reply,
                        },
                        ticket,
                    )),
                    Err(error) => {
                        self.engine.record_admission_rejection(request.id(), &error);
                        let _ = reply.send(Err(error));
                    }
                }
            }
            Command::Cancel {
                request,
                preparing,
                reply,
            } => {
                let result = self
                    .engine
                    .cancel(request)
                    .map(|events| generated.extend(events));
                let result = match result {
                    Err(error) if preparing && error.code == ErrorCode::NotFound => Ok(()),
                    other => other,
                };
                let _ = reply.send(result);
            }
            Command::Inspect { reply } => {
                self.deliver(generated);
                let mut inspection = self.engine.inspect();
                inspection.ready &= !self.stopping;
                let _ = reply.send(inspection);
            }
            Command::Observe {
                query,
                credit,
                reply,
            } => {
                if !reply.is_closed() {
                    self.deliver(generated);
                    let result = self
                        .engine
                        .observation_snapshot(query)
                        .map(|snapshot| (snapshot, credit));
                    let _ = reply.send(result);
                }
            }
            Command::Stop { reply } => {
                self.stopping = true;
                if self.stop_replies.len() == STOP_WAITER_CAPACITY {
                    let _ = reply.send(Err(Error::new(
                        ErrorCode::Capacity,
                        "shutdown waiter capacity exhausted",
                    )));
                } else {
                    self.stop_replies.push(reply);
                }
            }
        }
    }
}
