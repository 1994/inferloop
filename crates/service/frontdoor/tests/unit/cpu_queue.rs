use super::{WorkQueue, WorkSender};
use std::error::Error;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn last_sender_drains_queued_jobs() -> Result<(), Box<dyn Error>> {
    let queue = Arc::new(WorkQueue::new(2, 4));
    let sender = WorkSender(queue.clone());
    let (finished, results) = mpsc::channel();
    for value in 0..4 {
        let finished = finished.clone();
        sender.submit(Box::new(move |available| {
            available().unwrap();
            finished.send(value).unwrap();
        }))?;
    }
    drop(finished);
    drop(sender);
    let (exited, exits) = mpsc::channel();
    let mut workers = Vec::new();
    for index in 0..2 {
        let queue = queue.clone();
        let exited = exited.clone();
        workers.push(thread::spawn(move || -> Result<(), String> {
            while let Some(job) = queue.next(index).map_err(|error| error.to_string())? {
                job(&|| queue.available(index));
            }
            exited.send(()).map_err(|error| error.to_string())
        }));
    }
    let mut completed = Vec::new();
    for _ in 0..4 {
        completed.push(results.recv_timeout(Duration::from_secs(5))?);
    }
    completed.sort_unstable();
    assert_eq!(completed, [0, 1, 2, 3]);
    for _ in 0..2 {
        exits.recv_timeout(Duration::from_secs(5))?;
    }
    for worker in workers {
        worker.join().map_err(|_| "worker panicked")??;
    }
    Ok(())
}

#[test]
fn last_sender_wakes_parked_workers() -> Result<(), Box<dyn Error>> {
    let queue = Arc::new(WorkQueue::new(2, 4));
    let sender = WorkSender(queue.clone());
    let (exited, done) = mpsc::channel();
    let mut workers = Vec::new();
    for index in 0..2 {
        let queue = queue.clone();
        let exited = exited.clone();
        workers.push(thread::spawn(move || -> Result<(), String> {
            if queue
                .next(index)
                .map_err(|error| error.to_string())?
                .is_some()
            {
                return Err("unexpected job".into());
            }
            exited.send(()).map_err(|error| error.to_string())
        }));
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let parked = queue
            .state
            .lock()
            .map_err(|error| error.to_string())?
            .workers
            .iter()
            .all(|worker| worker.idle);
        if parked {
            break;
        }
        assert!(Instant::now() < deadline, "workers did not park");
        thread::sleep(Duration::from_millis(1));
    }
    drop(sender);
    for _ in 0..2 {
        done.recv_timeout(Duration::from_secs(5))?;
    }
    for worker in workers {
        worker.join().map_err(|_| "worker panicked")??;
    }
    Ok(())
}
