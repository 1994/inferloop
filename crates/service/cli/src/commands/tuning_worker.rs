//! Keep preparation-only CUDA modules out of the serving process's memory budget.
#[cfg(cuda_backend)]
use crate::backend::{self, BackendChoice, Selection};
#[cfg(cuda_backend)]
use infer_backend_cuda::tuning::table_path;
use infer_core::{Error, ErrorCode, Result};
#[cfg(cuda_backend)]
use std::env::{args_os, current_exe, var_os};
use std::ffi::OsStr;
use std::fs::{OpenOptions, create_dir_all, remove_file};
use std::io::{self, Read};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio, abort, id};
use std::thread::{Builder, sleep};
use std::time::{Duration, Instant};

const WORKER_ENV: &str = "INFER_INTERNAL_CUDA_TUNING_WORKER";
const WORKER_VALUE: &str = "prepare";
const PREPARATION_TIMEOUT: Duration = Duration::from_secs(600);
const POLL_INTERVAL: Duration = Duration::from_millis(50);

#[cfg(cuda_backend)]
pub(super) fn prepare(mut selection: Selection) -> Result<(Selection, bool)> {
    let mode = var_os(WORKER_ENV);
    let worker = worker_mode(mode.as_deref())?;
    if worker {
        monitor_parent()?;
        return Ok((selection, true));
    }
    if !selection.autotune
        || selection.kind == BackendChoice::Metal
        || !backend::cuda_available()
        || !table_path().is_some_and(|path| writable_cache(&path))
    {
        return Ok((selection, false));
    }
    tracing::info!(target: "infer::tuning", "preparing CUDA tuning cache in an isolated process");
    let prepared = current_exe()
        .map_err(|error| io_error(&error))
        .and_then(|executable| {
            let mut command = Command::new(executable);
            command
                .args(args_os().skip(1))
                .env(WORKER_ENV, WORKER_VALUE);
            run_worker(&mut command)
        });
    if let Err(error) = prepared {
        tracing::warn!(target: "infer::tuning", %error,
            "CUDA preparation failed; serving retains validated cache or conservative defaults");
    }
    // The serving process still validates cached choices against its own live weights.
    selection.autotune = false;
    tracing::info!(target: "infer::tuning", "CUDA preparation exited; serving reuses validated tuning cache");
    Ok((selection, false))
}

fn worker_mode(value: Option<&OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(value) if value == WORKER_VALUE => Ok(true),
        Some(_) => Err(Error::invalid("invalid internal CUDA preparation mode")),
    }
}

/// Test atomic-cache replacement permissions without creating an empty tuning table.
fn writable_cache(path: &Path) -> bool {
    if path.file_name().is_none() {
        return false;
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if create_dir_all(parent).is_err() {
        return false;
    }
    let probe = parent.join(format!(".infer-tuning-{}.probe", id()));
    let Ok(file) = OpenOptions::new().write(true).create_new(true).open(&probe) else {
        return false;
    };
    drop(file);
    remove_file(probe).is_ok()
}

/// The pipe stays open through wait; closing it cancels an orphaned preparation group.
fn run_worker(command: &mut Command) -> Result<()> {
    run_worker_for(command, PREPARATION_TIMEOUT)
}

fn run_worker_for(command: &mut Command, timeout: Duration) -> Result<()> {
    let mut child = command
        .process_group(0)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|error| io_error(&error))?;
    let Some(keepalive) = child.stdin.take() else {
        kill_group(child.id());
        let _ = child.kill();
        let _ = child.wait();
        return Err(Error::invariant("CUDA preparation parent pipe missing"));
    };
    let started = Instant::now();
    let result = loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break Ok(()),
            Ok(Some(status)) => {
                break Err(Error::new(
                    ErrorCode::Backend,
                    format!("CUDA tuning preparation exited with {status}"),
                ));
            }
            Err(error) => break Err(io_error(&error)),
            Ok(None) if started.elapsed() >= timeout => {
                break Err(Error::new(
                    ErrorCode::Backend,
                    "CUDA tuning preparation timed out",
                ));
            }
            Ok(None) => sleep(POLL_INTERVAL),
        }
    };
    if result.is_err() {
        kill_group(child.id());
        let _ = child.kill();
        let _ = child.wait();
    }
    drop(keepalive);
    result
}

/// The child owns its process group, including any JIT compiler subprocesses.
fn kill_group(group: u32) {
    let _ = Command::new("/bin/kill")
        .args(["-KILL", "--", &format!("-{group}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn monitor_parent() -> Result<()> {
    let monitor = Builder::new()
        .name("cuda-tuning-parent".into())
        .spawn(|| {
            let _ = io::stdin().read_exact(&mut [0]);
            kill_group(id());
            // Fallback for systems without /bin/kill; do not run CUDA atexit handlers.
            abort();
        })
        .map_err(|error| io_error(&error))?;
    drop(monitor);
    Ok(())
}

fn io_error(error: &io::Error) -> Error {
    Error::new(ErrorCode::Backend, error.to_string())
}

#[cfg(test)]
#[path = "../../tests/unit/tuning_worker.rs"]
mod tests;
