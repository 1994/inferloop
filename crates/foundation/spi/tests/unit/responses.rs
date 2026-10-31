use super::*;
use std::sync::Arc;

struct FakeStore {
    capabilities: ResponseStoreCapabilities,
    version: u32,
}

impl ResponseStore for FakeStore {
    fn capabilities(&self) -> ResponseStoreCapabilities {
        self.capabilities
    }
    fn spi_version(&self) -> u32 {
        self.version
    }
    fn get<'a>(&'a self, _id: &'a ResponseId) -> ResponseFuture<'a, Option<StoredResponse>> {
        Box::pin(async { Ok(None) })
    }
    fn put(&self, _response: StoredResponse, _condition: WriteCondition) -> ResponseFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
    fn delete<'a>(&'a self, _id: &'a ResponseId) -> ResponseFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
    fn transition<'a>(
        &'a self,
        _id: &'a ResponseId,
        _allowed: &'a [ResponseStatus],
        _next: ResponseStatus,
    ) -> ResponseFuture<'a, StoredResponse> {
        Box::pin(async { Err(Error::unsupported("fake")) })
    }
}

fn capabilities() -> ResponseStoreCapabilities {
    ResponseStoreCapabilities {
        retrieve: true,
        previous_response_id: true,
        background: true,
        cancel: true,
        event_replay: false,
        persistent: false,
        shared: false,
        max_record_bytes: 1024,
    }
}

#[test]
fn response_ids_are_bounded_ascii_tokens() {
    let id = ResponseId::new("resp_7").unwrap();
    assert_eq!(id.as_str(), "resp_7");
    assert_eq!(id.to_string(), "resp_7");
    for invalid in ["", "resp 7", "resp/7", "resp\n7"] {
        assert!(ResponseId::new(invalid).is_err(), "{invalid:?}");
    }
    assert!(ResponseId::new("r".repeat(ResponseId::MAX_BYTES + 1)).is_err());
}

#[test]
fn statuses_round_trip_and_reject_backwards_transitions() {
    for status in [
        ResponseStatus::Queued,
        ResponseStatus::InProgress,
        ResponseStatus::Completed,
        ResponseStatus::Incomplete,
        ResponseStatus::Failed,
        ResponseStatus::Cancelled,
    ] {
        assert_eq!(status.as_str().parse::<ResponseStatus>().unwrap(), status);
    }
    assert!(ResponseStatus::Queued.can_transition_to(ResponseStatus::InProgress));
    assert!(ResponseStatus::InProgress.can_transition_to(ResponseStatus::Completed));
    assert!(!ResponseStatus::Completed.can_transition_to(ResponseStatus::InProgress));
    assert!(!ResponseStatus::Completed.can_transition_to(ResponseStatus::Cancelled));
    assert!(ResponseStatus::Completed.can_transition_to(ResponseStatus::Completed));
}

fn stored(status: ResponseStatus) -> StoredResponse {
    StoredResponse {
        id: ResponseId::new("resp_7").unwrap(),
        request_id: RequestId::new(7).unwrap(),
        model: "1".to_string(),
        created_at: 100,
        completed_at: None,
        status,
        incomplete: None,
        error: None,
        usage: None,
        previous_response_id: None,
        request_input: b"[]".to_vec(),
        response: b"{}".to_vec(),
        payload_version: RESPONSES_PAYLOAD_VERSION,
    }
}

#[test]
fn stored_records_validate_terminal_details() {
    let mut active = stored(ResponseStatus::InProgress);
    active.validate().unwrap();
    active.completed_at = Some(101);
    assert!(active.validate().is_err());

    let mut completed = stored(ResponseStatus::Completed);
    completed.completed_at = Some(101);
    completed.validate().unwrap();
    completed.incomplete = Some(IncompleteReason::MaxOutputTokens);
    assert!(completed.validate().is_err());

    let mut incomplete = stored(ResponseStatus::Incomplete);
    incomplete.incomplete = Some(IncompleteReason::MaxOutputTokens);
    incomplete.validate().unwrap();
    incomplete.error = Some(ResponseError::new("x", "y").unwrap());
    assert!(incomplete.validate().is_err());

    let mut failed = stored(ResponseStatus::Failed);
    failed.error = Some(ResponseError::new("backend_error", "boom").unwrap());
    failed.validate().unwrap();
    failed.incomplete = Some(IncompleteReason::MaxOutputTokens);
    assert!(failed.validate().is_err());

    let mut cancelled = stored(ResponseStatus::Cancelled);
    cancelled.validate().unwrap();
    cancelled.error = Some(ResponseError::new("x", "y").unwrap());
    assert!(cancelled.validate().is_err());
}

#[test]
fn stored_records_require_payloads_and_a_known_version() {
    let mut record = stored(ResponseStatus::InProgress);
    record.request_input.clear();
    assert!(record.validate().is_err());

    let mut record = stored(ResponseStatus::InProgress);
    record.payload_version = RESPONSES_PAYLOAD_VERSION + 1;
    assert_eq!(
        record.validate().unwrap_err().code,
        infer_core::ErrorCode::Unsupported
    );
}

#[test]
fn extensions_validate_the_installed_store() {
    let disabled = ResponsesExtensions::new();
    assert!(!disabled.is_enabled());
    assert!(disabled.capabilities().is_none());
    disabled.validate().unwrap();

    let enabled = ResponsesExtensions::new().with_store(Arc::new(FakeStore {
        capabilities: capabilities(),
        version: RESPONSES_SPI_VERSION,
    }));
    assert!(enabled.is_enabled());
    assert_eq!(enabled.capabilities(), Some(capabilities()));
    enabled.validate().unwrap();

    let mismatched = ResponsesExtensions::new().with_store(Arc::new(FakeStore {
        capabilities: capabilities(),
        version: RESPONSES_SPI_VERSION + 1,
    }));
    assert_eq!(
        mismatched.validate().unwrap_err().code,
        infer_core::ErrorCode::Unsupported
    );

    let invalid = ResponsesExtensions::new().with_store(Arc::new(FakeStore {
        capabilities: ResponseStoreCapabilities {
            retrieve: true,
            max_record_bytes: 0,
            ..capabilities()
        },
        version: RESPONSES_SPI_VERSION,
    }));
    assert_eq!(
        invalid.validate().unwrap_err().code,
        infer_core::ErrorCode::InvalidInput
    );
}

#[test]
fn usage_total_is_checked() {
    let usage = ResponseUsage {
        input_tokens: 2,
        output_tokens: 3,
        ..Default::default()
    };
    assert_eq!(usage.checked_total(), Some(5));
    let overflow = ResponseUsage {
        input_tokens: u64::MAX,
        output_tokens: 1,
        ..Default::default()
    };
    assert_eq!(overflow.checked_total(), None);
}
