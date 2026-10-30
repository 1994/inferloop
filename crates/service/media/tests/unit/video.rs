#![cfg(unix)]

use super::{SampledVideo, VideoMetadata};
use std::error::Error;
#[cfg(target_os = "linux")]
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;

type TestResult = Result<(), Box<dyn Error>>;

fn reader(mut command: Command) -> Result<SampledVideo, Box<dyn Error>> {
    let mut child = command
        .kill_on_drop(true)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let stdout = child.stdout.take().ok_or("missing child output")?;
    Ok(SampledVideo {
        child,
        stdout,
        metadata: VideoMetadata {
            width: 2,
            height: 2,
            fps: 1.0,
            frames: 1,
        },
        frame_bytes: 12,
        buffer: Vec::new(),
        filled: 0,
        remaining: 1,
        complete: false,
    })
}

#[cfg(unix)]
#[tokio::test]
async fn cancelled_partial_read_can_resume_without_losing_bytes() -> TestResult {
    let mut command = Command::new("sh");
    command.args(["-c", "printf abc; read -r line; printf defghijkl"]);
    let mut video = reader(command)?;
    assert!(
        timeout(Duration::from_secs(1), video.next_frame())
            .await
            .is_err()
    );
    assert_eq!(video.filled, 3);
    video
        .child
        .stdin
        .as_mut()
        .ok_or("missing child input")?
        .write_all(b"\n")
        .await?;
    let frame = timeout(Duration::from_secs(5), video.next_frame())
        .await??
        .ok_or("missing frame")?;
    assert_eq!(frame.into_parts().2, b"abcdefghijkl");
    assert!(
        timeout(Duration::from_secs(5), video.next_frame())
            .await??
            .is_none()
    );
    Ok(())
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn dropping_reader_stops_its_decoder() -> TestResult {
    let mut command = Command::new("sleep");
    command.arg("60");
    let video = reader(command)?;
    let pid = video.child.id().ok_or("missing child id")?;
    drop(video);
    let process = format!("/proc/{pid}");
    timeout(Duration::from_secs(5), async {
        while Path::new(&process).exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?;
    Ok(())
}
