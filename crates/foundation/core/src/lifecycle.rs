use crate::{Error, Result, StepId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaitReason {
    MediaReadiness,
    StateCapacity { required: usize, available: usize },
    BackendBusy,
    Transfer,
    Budget,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Length,
    Eos,
    Completed,
    Cancelled,
    Deadline,
    Failed(String),
}
/// Who stopped a request before it finished on its own.
///
/// `FinishReason::Cancelled` says that a request stopped early; this says why, so an operator can
/// tell a requester walking away from a runtime shutdown without turning on debug logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CancelSource {
    /// The requester stopped reading its response, so its output stream closed.
    ClientDisconnected,
    /// The runtime is shutting down and drained in-flight work.
    Shutdown,
    /// The requester called an explicit cancel endpoint.
    Requested,
    /// The requester read too slowly and its bounded output queue overflowed.
    Backpressure,
}
impl CancelSource {
    /// Every source, in the order metrics count them.
    pub const ALL: &[Self] = &[
        Self::ClientDisconnected,
        Self::Shutdown,
        Self::Requested,
        Self::Backpressure,
    ];
    /// Stable field and label value for structured logs and metrics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClientDisconnected => "client_disconnected",
            Self::Shutdown => "shutdown",
            Self::Requested => "requested",
            Self::Backpressure => "backpressure",
        }
    }
    /// Position in [`Self::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}
/// Where a request was when it ended, from preparation outside the engine to output delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestStage {
    /// Resolving remote media before any CPU work starts.
    Media,
    /// Encoding, building constraints or preparing the canonical request on the CPU pool.
    Preparing,
    /// Waiting for the engine to admit the prepared request.
    Admission,
    /// Admitted and waiting for its first dispatch.
    Queued,
    /// Dispatched at least once without producing a token yet.
    Prefill,
    /// Producing tokens.
    Decode,
    /// Its latest step is in the CPU output stage.
    Output,
}
impl RequestStage {
    /// Stable field value for structured logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Media => "media",
            Self::Preparing => "preparing",
            Self::Admission => "admission",
            Self::Queued => "queued",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Output => "output",
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestStatus {
    Runnable,
    Running { step: StepId },
    Waiting(WaitReason),
    Finished(FinishReason),
}
impl RequestStatus {
    #[must_use]
    pub const fn terminal(&self) -> bool {
        matches!(self, Self::Finished(_))
    }
    /// # Errors
    /// Returns an invariant error for an illegal request lifecycle transition.
    pub fn transition(&mut self, next: Self) -> Result<()> {
        if self.terminal() {
            return Err(Error::invariant("terminal request cannot transition"));
        }
        match (&*self, &next) {
            (Self::Running { .. }, Self::Running { .. }) => {
                Err(Error::invariant("request already running"))
            }
            (Self::Waiting(_), Self::Running { .. }) => {
                Err(Error::invariant("waiting request must become runnable"))
            }
            _ => {
                *self = next;
                Ok(())
            }
        }
    }
}
