//! Response lifecycles through `Responses` with a scripted upstream and the reference store.
use infer_agentic::{
    BackgroundStart, DocumentMetadata, ResponseDocument, Responses, Retention, Upstream,
};
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_spi::{
    RESPONSES_PAYLOAD_VERSION, ResponseError, ResponseId, ResponseStatus, ResponseStore,
    ResponsesExtensions, StoredResponse, WriteCondition,
};
use infer_storage::{MemoryRecordStore, RecordResponseStore, ResponseRetention};
use std::future::{Future, ready};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot::{Receiver, Sender, channel};
use tokio::time::{sleep, timeout};

const MODEL: &str = "model";
/// Record limit large enough for every body these tests store.
const RECORD_BYTES: usize = 1024;
/// Bound on waiting for a spawned task's store write.
const SETTLE: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(5);

/// A minimal protocol body: its status, an optional error, and a label as the payload.
struct Body {
    status: ResponseStatus,
    error: Option<(String, String)>,
    label: &'static str,
}

impl Body {
    const fn new(status: ResponseStatus, label: &'static str) -> Self {
        Self {
            status,
            error: None,
            label,
        }
    }
}

impl ResponseDocument for Body {
    fn metadata(&self) -> Result<DocumentMetadata> {
        Ok(DocumentMetadata {
            status: self.status,
            completed_at: (self.status == ResponseStatus::Completed).then_some(2),
            incomplete: None,
            error: self
                .error
                .as_ref()
                .map(|(code, message)| ResponseError::new(code, message))
                .transpose()?,
            usage: None,
        })
    }

    fn encode(&self) -> Result<Vec<u8>> {
        Ok(self.label.as_bytes().to_vec())
    }
}

/// Each admitted generation receives its scripted outcome; cancellations are recorded.
#[derive(Default)]
struct Scripted {
    cancelled: Arc<Mutex<Vec<RequestId>>>,
}

impl Scripted {
    fn cancelled(&self) -> Vec<RequestId> {
        self.cancelled
            .lock()
            .map(|ids| ids.clone())
            .unwrap_or_default()
    }
}

impl Upstream for Scripted {
    type Admitted = Receiver<Result<Body>>;
    type Document = Body;

    async fn complete(&self, admitted: Receiver<Result<Body>>) -> Result<Body> {
        admitted
            .await
            .map_err(|_| Error::new(ErrorCode::Backend, "generation abandoned"))?
    }

    fn cancel(&self, request: RequestId) -> impl Future<Output = Result<()>> + Send {
        ready(
            self.cancelled
                .lock()
                .map(|mut ids| ids.push(request))
                .map_err(|_| Error::invariant("cancel log poisoned")),
        )
    }
}

/// A generation that has already finished with `outcome`.
fn finished(outcome: Result<Body>) -> Receiver<Result<Body>> {
    let (sender, receiver) = channel();
    assert!(sender.send(outcome).is_ok());
    receiver
}

fn store(max_record_bytes: usize) -> Result<Arc<RecordResponseStore>> {
    let retention = ResponseRetention {
        max_record_bytes,
        ..ResponseRetention::default()
    };
    RecordResponseStore::new(Arc::new(MemoryRecordStore::default()), retention).map(Arc::new)
}

fn service(store: &Arc<RecordResponseStore>) -> Responses {
    let store: Arc<dyn ResponseStore> = store.clone();
    Responses::new(ResponsesExtensions::new().with_store(store))
}

fn retention(id: &str, request: u64) -> Result<Retention<Body>> {
    Ok(Retention {
        id: ResponseId::new(id)?,
        request: RequestId::new(request)?,
        model: MODEL.to_string(),
        created: 1,
        previous: None,
        request_input: b"[]".to_vec(),
        failed: Arc::new(|error: &Error| Body {
            status: ResponseStatus::Failed,
            error: Some(("backend_error".to_string(), error.message.clone())),
            label: "failed",
        }),
    })
}

const fn initial() -> Body {
    Body::new(ResponseStatus::InProgress, "initial")
}

async fn stored(store: &RecordResponseStore, id: &str) -> Result<StoredResponse> {
    store
        .get(&ResponseId::new(id)?)
        .await?
        .ok_or_else(|| Error::new(ErrorCode::NotFound, id.to_string()))
}

/// Wait until a spawned writer leaves the record in a terminal state.
async fn settled(store: &RecordResponseStore, id: &str) -> Result<StoredResponse> {
    timeout(SETTLE, async {
        loop {
            let record = stored(store, id).await?;
            if record.status.is_terminal() {
                return Ok(record);
            }
            sleep(POLL).await;
        }
    })
    .await
    .map_err(|_| Error::new(ErrorCode::Backend, "record did not settle"))?
}

async fn put(
    store: &RecordResponseStore,
    id: &str,
    status: ResponseStatus,
    model: &str,
) -> Result<()> {
    let record = StoredResponse {
        id: ResponseId::new(id)?,
        request_id: RequestId::new(9)?,
        model: model.to_string(),
        created_at: 1,
        completed_at: status.is_terminal().then_some(2),
        status,
        incomplete: None,
        error: None,
        usage: None,
        previous_response_id: None,
        request_input: b"[]".to_vec(),
        response: b"{}".to_vec(),
        payload_version: RESPONSES_PAYLOAD_VERSION,
    };
    store.put(record, WriteCondition::Always).await
}

/// Start one background response whose generation finishes when the returned sender fires.
async fn start(responses: &Responses, id: &str, request: u64) -> Result<Sender<Result<Body>>> {
    let (sender, receiver) = channel();
    let admit = async {
        Ok(BackgroundStart {
            admitted: receiver,
            retention: retention(id, request)?,
            initial: initial(),
        })
    };
    let body = responses
        .start_background(Scripted::default(), admit)
        .await?;
    assert_eq!(body.label, "initial");
    Ok(sender)
}

#[tokio::test]
async fn completion_replaces_the_initial_snapshot() -> Result<()> {
    let store = store(RECORD_BYTES)?;
    let responses = service(&store);
    let upstream = Scripted::default();
    let lease = responses
        .retain(&upstream, retention("resp_a", 1)?, &initial())
        .await?;
    assert_eq!(
        stored(&store, "resp_a").await?.status,
        ResponseStatus::InProgress
    );

    let done = finished(Ok(Body::new(ResponseStatus::Completed, "done")));
    let value = responses.complete(&upstream, done, Some(lease)).await?;

    assert_eq!(value.label, "done");
    let record = stored(&store, "resp_a").await?;
    assert_eq!(record.status, ResponseStatus::Completed);
    assert_eq!(record.response, b"done");
    assert_eq!(upstream.cancelled(), Vec::new());
    Ok(())
}

#[tokio::test]
async fn generation_failure_is_recorded_and_returned() -> Result<()> {
    let store = store(RECORD_BYTES)?;
    let responses = service(&store);
    let upstream = Scripted::default();
    let lease = responses
        .retain(&upstream, retention("resp_f", 1)?, &initial())
        .await?;

    let failed = finished(Err(Error::new(ErrorCode::Backend, "device lost")));
    let outcome = responses.complete(&upstream, failed, Some(lease)).await;

    assert_eq!(
        outcome.err().map(|error| error.code),
        Some(ErrorCode::Backend)
    );
    let record = stored(&store, "resp_f").await?;
    assert_eq!(record.status, ResponseStatus::Failed);
    assert_eq!(
        record.error.map(|error| error.message),
        Some("device lost".to_string())
    );
    Ok(())
}

#[tokio::test]
async fn an_initial_store_failure_cancels_the_generation() -> Result<()> {
    // The record limit admits nothing, so the in-progress write fails.
    let store = store(1)?;
    let responses = service(&store);
    let upstream = Scripted::default();

    let outcome = responses
        .retain(&upstream, retention("resp_s", 7)?, &initial())
        .await;

    assert_eq!(
        outcome.err().map(|error| error.code),
        Some(ErrorCode::Capacity)
    );
    assert_eq!(upstream.cancelled(), vec![RequestId::new(7)?]);
    Ok(())
}

#[tokio::test]
async fn cancellation_wins_over_a_later_completion() -> Result<()> {
    let store = store(RECORD_BYTES)?;
    let responses = service(&store);
    let upstream = Scripted::default();
    let lease = responses
        .retain(&upstream, retention("resp_c", 3)?, &initial())
        .await?;

    let cancelled = responses.cancel(&upstream, "resp_c").await?;
    assert_eq!(cancelled.status, ResponseStatus::Cancelled);
    assert_eq!(upstream.cancelled(), vec![RequestId::new(3)?]);

    let done = finished(Ok(Body::new(ResponseStatus::Completed, "done")));
    let outcome = responses.complete(&upstream, done, Some(lease)).await;
    assert_eq!(
        outcome.err().map(|error| error.code),
        Some(ErrorCode::Conflict)
    );
    assert_eq!(
        stored(&store, "resp_c").await?.status,
        ResponseStatus::Cancelled
    );

    let terminal = responses.cancel(&upstream, "resp_c").await;
    assert_eq!(
        terminal.err().map(|error| error.code),
        Some(ErrorCode::InvalidInput)
    );
    Ok(())
}

#[tokio::test]
async fn a_dropped_lease_records_the_abandoned_response() -> Result<()> {
    let store = store(RECORD_BYTES)?;
    let responses = service(&store);
    let upstream = Scripted::default();
    let lease = responses
        .retain(&upstream, retention("resp_d", 4)?, &initial())
        .await?;

    drop(lease);

    let record = settled(&store, "resp_d").await?;
    assert_eq!(record.status, ResponseStatus::Failed);
    assert_eq!(record.response, b"failed");
    Ok(())
}

#[tokio::test]
async fn background_runs_finish_into_the_store_and_release_capacity() -> Result<()> {
    let store = store(RECORD_BYTES)?;
    let responses = service(&store);
    let mut running = Vec::new();
    let exhausted = loop {
        let id = format!("resp_b{}", running.len());
        let request =
            u64::try_from(running.len()).map_err(|error| Error::invalid(error.to_string()))?;
        match start(&responses, &id, request + 1).await {
            Ok(sender) => running.push((id, sender)),
            Err(error) => break error,
        }
    };
    assert_eq!(exhausted.code, ErrorCode::Capacity);
    assert!(!running.is_empty());

    // A full deployment rejects before admission.
    let admitted = AtomicBool::new(false);
    let rejected = responses
        .start_background(Scripted::default(), async {
            admitted.store(true, Ordering::SeqCst);
            Err::<BackgroundStart<Scripted>, Error>(Error::invariant(
                "admitted past the capacity check",
            ))
        })
        .await;
    assert_eq!(
        rejected.err().map(|error| error.code),
        Some(ErrorCode::Capacity)
    );
    assert!(!admitted.load(Ordering::SeqCst));

    let (id, sender) = running.remove(0);
    assert_eq!(
        stored(&store, &id).await?.status,
        ResponseStatus::InProgress
    );
    assert!(
        sender
            .send(Ok(Body::new(ResponseStatus::Completed, "done")))
            .is_ok()
    );
    let record = settled(&store, &id).await?;
    assert_eq!(record.status, ResponseStatus::Completed);
    assert_eq!(record.response, b"done");

    // The finished run released its slot.
    let next = timeout(SETTLE, async {
        loop {
            if let Ok(sender) = start(&responses, "resp_next", 100).await {
                return sender;
            }
            sleep(POLL).await;
        }
    })
    .await;
    assert!(next.is_ok());
    Ok(())
}

#[tokio::test]
async fn chained_requests_need_a_finished_response_of_the_same_model() -> Result<()> {
    let store = store(RECORD_BYTES)?;
    let responses = service(&store);
    put(&store, "resp_done", ResponseStatus::Completed, MODEL).await?;
    put(&store, "resp_open", ResponseStatus::InProgress, MODEL).await?;
    put(&store, "resp_other", ResponseStatus::Completed, "other").await?;

    assert!(responses.prior(None, MODEL).await?.is_none());
    let prior = responses.prior(Some("resp_done"), MODEL).await?;
    assert_eq!(
        prior.map(|record| record.status),
        Some(ResponseStatus::Completed)
    );
    let cases = [
        ("resp_open", ErrorCode::Conflict),
        ("resp_other", ErrorCode::InvalidInput),
        ("resp_missing", ErrorCode::NotFound),
    ];
    for (id, code) in cases {
        let outcome = responses.prior(Some(id), MODEL).await;
        assert_eq!(outcome.err().map(|error| error.code), Some(code), "{id}");
    }
    Ok(())
}

#[tokio::test]
async fn state_features_without_a_store_are_unsupported() {
    let responses = Responses::new(ResponsesExtensions::new());
    let upstream = Scripted::default();

    assert!(!responses.has_store());
    let background = responses
        .start_background(Scripted::default(), async {
            Err::<BackgroundStart<Scripted>, Error>(Error::invariant("admitted without a store"))
        })
        .await;
    let codes = [
        responses.prior(Some("resp_x"), MODEL).await.err(),
        responses.retrieve("resp_x").await.err(),
        responses.cancel(&upstream, "resp_x").await.err(),
        background.err(),
    ]
    .map(|error| error.map(|error| error.code));
    assert_eq!(codes, [Some(ErrorCode::Unsupported); 4]);
}
