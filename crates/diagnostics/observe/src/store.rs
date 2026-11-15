use crate::{Diagnostic, window::EventPages, window::ObservationWindow};
use infer_core::{
    Error, RequestId, Result, event::EventKind, event::ObjectKind, event::SemanticEvent,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, collections::VecDeque, sync::Arc};

/// Largest accepted event-retention capacity of an observation store.
const MAX_OBSERVATION_CAPACITY: usize = 1_048_576;
/// Retained diagnostics before the oldest is evicted.
const DIAGNOSTIC_RING_CAPACITY: usize = 256;
/// Wire code reported for a length-limited completion.
const FINISH_CODE_LENGTH: u64 = 0;
/// Wire code reported for an end-of-sequence completion.
const FINISH_CODE_EOS: u64 = 1;
/// Wire code reported for a completed request.
const FINISH_CODE_COMPLETED: u64 = 2;
/// Wire code reported for a cancelled request.
pub const FINISH_CODE_CANCELLED: u64 = 3;
/// Wire code reported for a deadline-exceeded request.
const FINISH_CODE_DEADLINE: u64 = 4;
/// Wire code reported for a failed request.
pub const FINISH_CODE_FAILED: u64 = 5;
/// Wire code reported for an invalid-input error.
const ERROR_CODE_INVALID_INPUT: u64 = 1;
/// Wire code reported for an unsupported operation.
const ERROR_CODE_UNSUPPORTED: u64 = 2;
/// Wire code reported for an exhausted capacity.
const ERROR_CODE_CAPACITY: u64 = 3;
/// Wire code reported for a missing resource.
const ERROR_CODE_NOT_FOUND: u64 = 4;
/// Wire code reported for a conflicting request.
const ERROR_CODE_CONFLICT: u64 = 5;
/// Wire code reported for a violated invariant.
const ERROR_CODE_INVARIANT: u64 = 6;
/// Wire code reported for a backend failure.
const ERROR_CODE_BACKEND: u64 = 7;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedEvent {
    pub sequence: u64,
    pub request: Option<RequestId>,
    pub event: SemanticEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventQuery {
    pub session: Option<String>,
    pub events: Vec<ObservedEvent>,
    pub next_cursor: u64,
    pub oldest_cursor: u64,
    pub cursor_gap: bool,
    pub ring_dropped: u64,
    pub history_evicted: u64,
}

/// A collector is an explicit cold-path consumer; exporters never run in event emission.
pub struct ObservationStore {
    events: EventPages,
    diagnostics: Arc<VecDeque<Diagnostic>>,
    requests: BTreeMap<RequestId, usize>,
    capacity: usize,
    next_diagnostic: u64,
    pub history_evicted: u64,
    pub diagnostics_evicted: u64,
}
impl ObservationStore {
    /// # Errors
    /// Returns invalid input for an empty or excessively large retention capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > MAX_OBSERVATION_CAPACITY {
            return Err(Error::invalid("observation capacity must be 1 to 1048576"));
        }
        Ok(Self {
            events: EventPages {
                next_sequence: 1,
                ..Default::default()
            },
            diagnostics: Arc::new(VecDeque::with_capacity(DIAGNOSTIC_RING_CAPACITY)),
            requests: BTreeMap::new(),
            capacity,
            next_diagnostic: 1,
            history_evicted: 0,
            diagnostics_evicted: 0,
        })
    }
    pub fn record(&mut self, event: SemanticEvent) {
        let _ = self.record_retirement(event);
    }
    /// Report an identity whose last retained event was evicted, after inserting the new event.
    pub fn record_retirement(&mut self, event: SemanticEvent) -> Option<RequestId> {
        let mut retired = None;
        let request = match event.object_kind {
            ObjectKind::Request => RequestId::new(event.object_id).ok(),
            ObjectKind::State => RequestId::new(event.correlation_id).ok(),
            _ => None,
        };
        if self.events.len == self.capacity {
            if let Some(id) = self.events.pop().and_then(|event| event.request)
                && let Some(count) = self.requests.get_mut(&id)
            {
                *count -= 1;
                if *count == 0 {
                    self.requests.remove(&id);
                    retired = Some(id);
                }
            }
            self.history_evicted = self.history_evicted.saturating_add(1);
        }
        if let Some(id) = request {
            *self.requests.entry(id).or_default() += 1;
        }
        self.events.push(ObservedEvent {
            sequence: self.events.next_sequence,
            request,
            event,
        });
        self.events.next_sequence = self.events.next_sequence.saturating_add(1);
        retired.filter(|id| !self.requests.contains_key(id))
    }
    pub fn diagnostic(&mut self, mut diagnostic: Diagnostic) {
        diagnostic.sequence = self.next_diagnostic;
        self.next_diagnostic = self.next_diagnostic.saturating_add(1);
        if self.diagnostics.len() == DIAGNOSTIC_RING_CAPACITY {
            Arc::make_mut(&mut self.diagnostics).pop_front();
            self.diagnostics_evicted = self.diagnostics_evicted.saturating_add(1);
        }
        Arc::make_mut(&mut self.diagnostics).push_back(diagnostic);
    }
    #[must_use]
    pub fn diagnostics(&self) -> Vec<Diagnostic> {
        self.diagnostics.iter().cloned().collect()
    }
    /// # Errors
    /// Returns invalid input for a query exceeding the per-response event budget.
    pub fn query(
        &self,
        after: u64,
        limit: usize,
        request: Option<RequestId>,
        ring_dropped: u64,
    ) -> Result<EventQuery> {
        self.events
            .query(after, limit, request, ring_dropped, self.history_evicted)
    }
    /// Share event pages and the bounded diagnostic window; no event payload is copied.
    #[must_use]
    pub fn snapshot(&self, include_events: bool) -> ObservationWindow {
        ObservationWindow {
            events: if include_events {
                self.events.clone()
            } else {
                EventPages::default()
            },
            diagnostics: self.diagnostics.clone(),
            retained: self.events.len,
            history_evicted: self.history_evicted,
            diagnostics_evicted: self.diagnostics_evicted,
        }
    }
    #[must_use]
    pub const fn retained(&self) -> usize {
        self.events.len
    }
    #[must_use]
    pub fn contains_request(&self, id: u64) -> bool {
        RequestId::new(id).is_ok_and(|id| self.requests.contains_key(&id))
    }
    #[must_use]
    pub fn timeline(&self) -> Vec<ObservedEvent> {
        self.events.iter().cloned().collect()
    }
    pub fn legacy_events(&self, cursor: &mut u64) -> Vec<SemanticEvent> {
        let events = self
            .events
            .after(*cursor)
            .map(|event| event.event)
            .collect();
        *cursor = self.events.next_sequence.saturating_sub(1);
        events
    }
}

#[must_use]
pub const fn finish_code(reason: &infer_core::FinishReason) -> u64 {
    match reason {
        infer_core::FinishReason::Length => FINISH_CODE_LENGTH,
        infer_core::FinishReason::Eos => FINISH_CODE_EOS,
        infer_core::FinishReason::Completed => FINISH_CODE_COMPLETED,
        infer_core::FinishReason::Cancelled => FINISH_CODE_CANCELLED,
        infer_core::FinishReason::Deadline => FINISH_CODE_DEADLINE,
        infer_core::FinishReason::Failed(_) => FINISH_CODE_FAILED,
    }
}

/// Who or what ended a request, packed into a `Finished` event's spare field.
///
/// A cancelled request carries its cancel source and a failed one its error code, each as its
/// position plus one; zero means the cause is unknown or the reason has none.
#[must_use]
pub fn finish_cause(
    reason: &infer_core::FinishReason,
    cancel: Option<infer_core::CancelSource>,
    error: Option<infer_core::ErrorCode>,
) -> u32 {
    let index = match reason {
        infer_core::FinishReason::Cancelled => cancel.map(infer_core::CancelSource::index),
        infer_core::FinishReason::Failed(_) => error.map(infer_core::ErrorCode::index),
        _ => None,
    };
    index
        .and_then(|index| index.checked_add(1))
        .and_then(|cause| u32::try_from(cause).ok())
        .unwrap_or(0)
}

#[must_use]
pub const fn error_code(code: infer_core::ErrorCode) -> u64 {
    match code {
        infer_core::ErrorCode::InvalidInput => ERROR_CODE_INVALID_INPUT,
        infer_core::ErrorCode::Unsupported => ERROR_CODE_UNSUPPORTED,
        infer_core::ErrorCode::Capacity => ERROR_CODE_CAPACITY,
        infer_core::ErrorCode::NotFound => ERROR_CODE_NOT_FOUND,
        infer_core::ErrorCode::Conflict => ERROR_CODE_CONFLICT,
        infer_core::ErrorCode::Invariant => ERROR_CODE_INVARIANT,
        infer_core::ErrorCode::Backend => ERROR_CODE_BACKEND,
    }
}

#[must_use]
pub const fn diagnostic_for_event(kind: EventKind) -> Option<crate::DiagnosticCode> {
    match kind {
        EventKind::Rejected => Some(crate::DiagnosticCode::AdmissionRejected),
        EventKind::InvariantViolation => Some(crate::DiagnosticCode::InvariantViolation),
        EventKind::Backpressure => Some(crate::DiagnosticCode::OutputBackpressure),
        _ => None,
    }
}
