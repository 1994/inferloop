use super::*;
use std::sync::Barrier;
use std::sync::atomic::AtomicUsize;
use std::thread;

#[test]
fn pending_checks_skip_work_until_a_completion_edge_or_fault_probe() {
    let mut notification = Notification::new();
    let start = notification.last_probe;
    assert!(notification.should_poll_at(start));
    for _ in 0..100 {
        assert!(!notification.should_poll_at(start));
    }
    assert!(!notification.should_poll_at(start + FAULT_PROBE_INTERVAL / 2));
    assert!(notification.should_poll_at(start + FAULT_PROBE_INTERVAL));
    assert!(!notification.should_poll_at(start + FAULT_PROBE_INTERVAL));
    notification.arm();
    assert!(notification.should_poll());
    assert!(!notification.should_poll_at(notification.last_probe));
}

#[test]
fn replacing_an_owner_forces_pending_work_to_register_the_new_waker() {
    let mut notification = Notification::new();
    let old = notification.waker().clone();
    assert!(notification.should_poll());
    let wakes = Arc::new(AtomicUsize::new(0));
    let target = Arc::clone(&wakes);
    notification.set_waker(Arc::new(move || {
        target.fetch_add(1, Ordering::Relaxed);
    }));
    assert_eq!(wakes.load(Ordering::Relaxed), 1);
    assert!(notification.should_poll());
    assert!(!notification.should_poll_at(notification.last_probe));
    // An old in-flight callback remains a valid edge after owner replacement.
    old.wake_by_ref();
    assert!(notification.should_poll());
    notification.waker().wake_by_ref();
    assert_eq!(wakes.load(Ordering::Relaxed), 2);
    assert!(notification.should_poll());
}

#[test]
fn completion_published_after_an_empty_check_is_not_lost() {
    let mut notification = Notification::new();
    let waker = notification.waker().clone();
    let barrier = Arc::new(Barrier::new(2));
    let publisher = Arc::clone(&barrier);
    let worker = thread::spawn(move || {
        for _ in 0..64 {
            publisher.wait();
            waker.wake_by_ref();
            publisher.wait();
        }
    });
    let now = notification.last_probe;
    assert!(notification.should_poll_at(now));
    for _ in 0..64 {
        assert!(!notification.should_poll_at(now));
        barrier.wait();
        barrier.wait();
        assert!(notification.should_poll_at(now));
    }
    worker.join().unwrap();
}
