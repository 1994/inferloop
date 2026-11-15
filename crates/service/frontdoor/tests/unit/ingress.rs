use super::*;
use std::io::{Error as IoError, Result as IoResult, Write};
use std::time::Duration;
use tokio::time::timeout;
use tracing::subscriber::with_default;

#[tokio::test]
async fn cancellation_wakes_waiters_and_releases_dynamic_bytes() {
    let registry = Registry::new(1);
    let id = RequestId::new(1).unwrap();
    let pending = registry.reserve(id, 10).unwrap();
    pending.add_bytes(20).unwrap();
    assert_eq!(registry.usage.lock().unwrap().bytes, 30);
    let waiting = pending.clone();
    let waiter = tokio::spawn(async move { waiting.cancelled().await });
    assert!(registry.cancel(id));
    timeout(Duration::from_secs(1), waiter)
        .await
        .expect("cancelled request must wake")
        .unwrap();
    assert!(pending.check().is_err());
    drop(pending);
    assert_eq!(registry.usage.lock().unwrap().bytes, 0);
    assert!(registry.reserve(RequestId::new(2).unwrap(), 0).is_ok());
}

/// Captured `infer::request` lines, for asserting the terminal-line contract.
#[derive(Clone, Default)]
struct Lines(Arc<Mutex<Vec<u8>>>);

impl Write for Lines {
    fn write(&mut self, bytes: &[u8]) -> IoResult<usize> {
        self.0
            .lock()
            .map_err(|_| IoError::other("captured log poisoned"))?
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

impl Lines {
    fn during(&self, run: impl FnOnce()) {
        let writer = self.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        with_default(subscriber, run);
    }
    fn text(&self) -> String {
        self.0
            .lock()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_default()
    }
}

/// A submitter that goes away mid-preparation leaves no engine record, so the guard writes the
/// request's only terminal line, naming the stage it reached.
#[test]
fn an_abandoned_preparation_writes_its_terminal_line() {
    let lines = Lines::default();
    let registry = Registry::new(1);
    lines.during(|| {
        let pending = registry.reserve(RequestId::new(5).unwrap(), 64).unwrap();
        let mut guard = Abandon::new(pending.clone(), RequestStage::Media);
        guard.enter(RequestStage::Preparing);
        drop(guard);
        assert!(
            pending.check().is_err(),
            "dropping the guard cancels the work"
        );
    });
    let text = lines.text();
    assert_eq!(text.lines().count(), 1, "{text}");
    for field in [
        "request=5",
        "phase=\"abandoned\"",
        "cancel=\"client_disconnected\"",
        "stage=\"preparing\"",
        "bytes=64",
    ] {
        assert!(text.contains(field), "{field} in {text}");
    }
}

/// A preparation that returned is accounted by its result: success and admission rejections
/// belong to the engine, an explicit cancel and a preparation error end here.
#[test]
fn a_settled_preparation_reports_only_what_the_engine_cannot() {
    let lines = Lines::default();
    let registry = Registry::new(3);
    lines.during(|| {
        let admitted = registry.reserve(RequestId::new(1).unwrap(), 0).unwrap();
        let mut guard = Abandon::new(admitted, RequestStage::Preparing);
        guard.enter(RequestStage::Admission);
        guard.settle(&Ok(()));

        let refused = registry.reserve(RequestId::new(2).unwrap(), 0).unwrap();
        let mut guard = Abandon::new(refused, RequestStage::Preparing);
        guard.enter(RequestStage::Admission);
        guard.settle::<()>(&Err(Error::new(ErrorCode::Capacity, "engine full")));

        // Dropped after the engine received it: the engine writes the line.
        let submitted = registry.reserve(RequestId::new(6).unwrap(), 0).unwrap();
        let mut guard = Abandon::new(submitted, RequestStage::Preparing);
        guard.enter(RequestStage::Admission);
        drop(guard);

        let cancelled = registry.reserve(RequestId::new(3).unwrap(), 0).unwrap();
        let guard = Abandon::new(cancelled, RequestStage::Media);
        assert!(registry.cancel(RequestId::new(3).unwrap()));
        guard.settle::<()>(&Err(Error::new(ErrorCode::Capacity, "cancelled")));
    });
    let text = lines.text();
    assert!(!text.contains("request=1"), "{text}");
    assert!(!text.contains("request=2"), "{text}");
    assert!(!text.contains("request=6"), "{text}");
    let cancelled: Vec<_> = text.lines().filter(|l| l.contains("request=3")).collect();
    assert_eq!(cancelled.len(), 1, "{text}");
    assert!(cancelled[0].contains("phase=\"abandoned\""), "{text}");
    assert!(cancelled[0].contains("cancel=\"requested\""), "{text}");

    let lines = Lines::default();
    lines.during(|| {
        let invalid = registry.reserve(RequestId::new(4).unwrap(), 0).unwrap();
        let guard = Abandon::new(invalid, RequestStage::Preparing);
        guard.settle::<()>(&Err(Error::invalid("prompt does not encode")));
    });
    let text = lines.text();
    for field in [
        "request=4",
        "phase=\"rejected\"",
        "stage=\"preparing\"",
        "error_code=\"invalid_input\"",
    ] {
        assert!(text.contains(field), "{field} in {text}");
    }
}
