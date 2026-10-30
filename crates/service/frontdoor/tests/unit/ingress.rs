use super::*;
use std::time::Duration;
use tokio::time::timeout;

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
