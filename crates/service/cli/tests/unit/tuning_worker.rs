use super::*;
use std::env::{current_exe, temp_dir, var_os};
use std::error::Error as StdError;
use std::fs::{read_to_string, remove_dir_all, write};
use std::os::unix::process::ExitStatusExt;
use std::process::exit;
use std::result::Result as StdResult;
use std::thread::park;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const PROBE_ENV: &str = "INFER_TUNING_PARENT_PIPE_TEST";
const PROBE_TEST: &str = "commands::tuning_worker::tests::parent_link_survives_wait";
const PROBE_LIFETIME: Duration = Duration::from_millis(200);
const PARENT_LINK_TIMEOUT: Duration = Duration::from_secs(5);
const PARENT_LINK_POLL: Duration = Duration::from_millis(10);

fn probe_command() -> StdResult<Command, Box<dyn StdError>> {
    let mut command = Command::new(current_exe()?);
    command
        .args(["--exact", PROBE_TEST, "--test-threads=1"])
        .env(PROBE_ENV, "child")
        .env(WORKER_ENV, WORKER_VALUE)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    Ok(command)
}

#[test]
fn parent_link_survives_wait() -> StdResult<(), Box<dyn StdError>> {
    if var_os(PROBE_ENV).is_some() {
        assert!(worker_mode(var_os(WORKER_ENV).as_deref())?);
        monitor_parent()?;
        if var_os(PROBE_ENV).as_deref() == Some(OsStr::new("wait")) {
            loop {
                park();
            }
        }
        if var_os(PROBE_ENV).as_deref() == Some(OsStr::new("fail")) {
            exit(7);
        }
        sleep(PROBE_LIFETIME);
        return Ok(());
    }
    run_worker(&mut probe_command()?)?;
    Ok(())
}

#[test]
fn preparation_stops_when_parent_link_closes() -> StdResult<(), Box<dyn StdError>> {
    let mut child = probe_command()?
        .env(PROBE_ENV, "wait")
        .process_group(0)
        .stdin(Stdio::piped())
        .spawn()?;
    drop(child.stdin.take());
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= PARENT_LINK_TIMEOUT {
            kill_group(child.id());
            let _ = child.kill();
            let status = child.wait()?;
            return Err(
                format!("parent-link monitor timed out; child exited with {status}").into(),
            );
        }
        sleep(PARENT_LINK_POLL);
    };
    assert_eq!(status.signal(), Some(9));
    Ok(())
}

#[test]
fn preparation_marker_is_closed_and_explicit() -> Result<()> {
    assert!(!worker_mode(None)?);
    assert!(worker_mode(Some(OsStr::new(WORKER_VALUE)))?);
    assert!(worker_mode(Some(OsStr::new("unknown"))).is_err());
    Ok(())
}

#[test]
fn cache_probe_preserves_table_and_detects_unusable_parent() -> StdResult<(), Box<dyn StdError>> {
    let root = temp_dir().join(format!(
        "infer-tuning-{}-{}",
        id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    let table = root.join("nested/tuning.json");
    assert!(writable_cache(&table));
    assert!(!table.exists());
    write(&table, "existing table")?;
    assert!(writable_cache(&table));
    assert_eq!(read_to_string(&table)?, "existing table");
    assert!(!writable_cache(&table.join("blocked.json")));
    remove_dir_all(root)?;
    Ok(())
}

#[test]
fn preparation_timeout_and_nonzero_exit_are_bounded_failures() -> StdResult<(), Box<dyn StdError>> {
    let mut blocked = probe_command()?;
    blocked.env(PROBE_ENV, "wait");
    let error = run_worker_for(&mut blocked, Duration::from_millis(30))
        .err()
        .ok_or("preparation did not time out")?;
    assert!(error.to_string().contains("timed out"));
    let mut failed = probe_command()?;
    failed.env(PROBE_ENV, "fail");
    assert!(run_worker(&mut failed).is_err());
    Ok(())
}

#[cfg(cuda_backend)]
#[test]
fn preparation_stamp_needs_its_table_and_identity() -> StdResult<(), Box<dyn StdError>> {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = temp_dir().join(format!("infer-stamp-{}-{unique}", id()));
    create_dir_all(&root)?;
    let table = root.join("tuning.json");
    let stamp = stamp_path(&table, &[b"first".to_vec()]);
    assert_ne!(stamp, stamp_path(&table, &[b"second".to_vec()]));
    write_stamp(&stamp, 17)?;
    assert_eq!(read_stamp(&table, &stamp), None);
    write(&table, "{}")?;
    assert_eq!(read_stamp(&table, &stamp), Some(17));
    write(&stamp, "0")?;
    assert_eq!(read_stamp(&table, &stamp), None);
    write(&stamp, "garbage")?;
    assert_eq!(read_stamp(&table, &stamp), None);
    remove_dir_all(root)?;
    Ok(())
}
