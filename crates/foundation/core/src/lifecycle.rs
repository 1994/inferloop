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
/// Who stopped a request before it finished.
///
/// `FinishReason::Cancelled` records that a request stopped early; this records why an operator
/// should be able to tell a requester walking away from a runtime shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CancelSource {
    /// The requester stopped reading its response, so its output stream closed.
    ClientDisconnected,
    /// The runtime is shutting down and drained in-flight work.
    Shutdown,
    /// The requester called an explicit cancel endpoint.
    Requested,
}
impl CancelSource {
    /// Stable field value for structured logs and inspection.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClientDisconnected => "client_disconnected",
            Self::Shutdown => "shutdown",
            Self::Requested => "requested",
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
