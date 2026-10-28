//! Request-owned state and its immutable scheduling projection.
use crate::CompletedRequest;
use infer_core::credits::CreditLease;
use infer_core::{FinishReason, RequestStatus, StateId};
use infer_ir::{AdmissionDecision, CanonicalRequest, ModelOutput, WorkloadPlan};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    #[serde(default)]
    pub constraint: Option<infer_spi::ConstraintHandle>,
    pub request: Arc<CanonicalRequest>,
    pub tenant: Arc<str>,
    #[serde(skip)]
    pub host_lease: Option<CreditLease>,
    #[serde(skip)]
    pub byte_lease: Option<CreditLease>,
    /// Prepared media is device-bound request state; it is never serialized into a checkpoint.
    #[serde(skip)]
    pub media: Option<Arc<infer_ir::MediaPlan>>,
    pub plan: WorkloadPlan,
    pub state: Option<StateId>,
    pub status: RequestStatus,
    pub progress_epoch: u64,
    pub unit: usize,
    pub prefill_done: usize,
    pub prefill_target: usize,
    pub preemptions: u64,
    pub cached_tokens: usize,
    #[serde(default)]
    pub prefix_attempted: bool,
    /// The request's prefix boundary rounded down to a page: the prompt chunk that reaches it
    /// stops there so the backend can keep the covered prefix.
    #[serde(default)]
    pub prefix_stop: Option<usize>,
    pub context: TokenContext,
    pub generated: infer_ir::TokenBuffer,
    pub outputs: Vec<ModelOutput>,
    pub completed: Option<CompletedRequest>,
    pub pending_finish: Option<FinishReason>,
    pub accepted_us: u64,
    pub first_token_us: Option<u64>,
    pub last_token_us: Option<u64>,
    pub max_tpot_us: Option<u64>,
    #[serde(default)]
    pub last_service_us: u64,
    #[serde(default)]
    pub admission: AdmissionDecision,
}

impl RequestRecord {
    /// Where the next prompt chunk must stop: the prefix stop while it is still ahead, then the
    /// end of the prompt, then the end of the context.
    #[must_use]
    pub fn prefill_end(&self) -> usize {
        match self.prefix_stop {
            Some(stop) if self.prefill_done < stop => stop,
            _ if self.prefill_done < self.context.prompt.len() => self.context.prompt.len(),
            _ => self.context.len(),
        }
    }
    /// Whether a prompt chunk of `count` tokens ends exactly at the prefix stop.
    #[must_use]
    pub fn chunk_reaches_prefix_stop(&self, count: usize) -> bool {
        self.prefix_stop
            .is_some_and(|stop| self.prefill_done.checked_add(count) == Some(stop))
    }
}

/// Prompt storage is shared; generated payload has one mutable owner in `RequestRecord`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenContext {
    pub prompt: infer_ir::TokenBuffer,
    tail_len: usize,
}
impl TokenContext {
    #[must_use]
    pub const fn new(prompt: infer_ir::TokenBuffer) -> Self {
        Self {
            prompt,
            tail_len: 0,
        }
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.prompt.len() + self.tail_len
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// # Errors
    /// Rejects corrupted context length rather than wrapping its position.
    pub fn append(&mut self) -> infer_core::Result<()> {
        self.tail_len = self
            .tail_len
            .checked_add(1)
            .filter(|tail| self.prompt.len().checked_add(*tail).is_some())
            .ok_or_else(|| infer_core::Error::invariant("token context position overflow"))?;
        Ok(())
    }
    #[must_use]
    pub const fn generated_len(&self) -> usize {
        self.tail_len
    }
}
