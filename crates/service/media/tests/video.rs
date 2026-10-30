use infer_media_input::{SampledVideo, probe_video};
use std::env::temp_dir;
use std::error::Error;
use std::fs::{create_dir, remove_dir_all, write};
use std::path::PathBuf;
use std::process::{Command, id};
use std::time::{SystemTime, UNIX_EPOCH};

struct Fixture(PathBuf);
type ClipFixture = (Fixture, Vec<Vec<u8>>);
type TestResult<T> = Result<T, Box<dyn Error>>;

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = remove_dir_all(&self.0);
    }
}

fn lossless_clip(count: usize) -> TestResult<ClipFixture> {
    let path = temp_dir().join(format!(
        "infer-video-test-{}-{}",
        id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    create_dir(&path)?;
    let fixture = Fixture(path);
    let frames: Vec<Vec<u8>> = (0..count)
        .map(|frame| {
            (0..16 * 12 * 3)
                .map(|offset| u8::try_from((frame * 31 + offset * 11) % 256))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<_, _>>()?;
    write(fixture.0.join("frames.rgb"), frames.concat())?;
    let output = Command::new("ffmpeg")
        .args([
            "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "16x12", "-r", "3",
            "-i",
        ])
        .arg(fixture.0.join("frames.rgb"))
        .args(["-c:v", "ffv1", "-pix_fmt", "gbrp", "-an"])
        .arg(fixture.0.join("clip.mkv"))
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok((fixture, frames))
}

#[tokio::test]
async fn sampled_frames_preserve_exact_pixels_and_source_order() -> TestResult<()> {
    let (fixture, frames) = lossless_clip(7)?;
    let path = fixture.0.join("clip.mkv");
    let metadata = probe_video(&path).await.unwrap();
    assert_eq!(
        (metadata.width(), metadata.height(), metadata.frames()),
        (16, 12, 7)
    );
    assert_eq!(metadata.fps(), 3.0);
    let indices = [0, 2, 3, 6];
    let mut video = SampledVideo::start(&path, metadata, &indices).unwrap();
    for index in indices {
        let (width, height, pixels) = video.next_frame().await.unwrap().unwrap().into_parts();
        assert_eq!((width, height), (16, 12));
        assert_eq!(pixels, frames[index]);
    }
    assert!(video.next_frame().await.unwrap().is_none());
    assert!(video.next_frame().await.unwrap().is_none());
    assert!(SampledVideo::start(&path, metadata, &[]).is_err());
    assert!(SampledVideo::start(&path, metadata, &[2, 2]).is_err());
    assert!(SampledVideo::start(&path, metadata, &[7]).is_err());
    Ok(())
}

#[tokio::test]
async fn invalid_video_is_an_explicit_error() -> TestResult<()> {
    let (fixture, _) = lossless_clip(7)?;
    let path = fixture.0.join("broken.mkv");
    write(&path, b"not a video").unwrap();
    assert!(probe_video(&path).await.is_err());
    Ok(())
}

#[tokio::test]
async fn long_clips_decode_only_the_requested_frames() -> TestResult<()> {
    let (fixture, frames) = lossless_clip(800)?;
    let path = fixture.0.join("clip.mkv");
    let metadata = probe_video(&path).await.unwrap();
    assert_eq!(metadata.frames(), frames.len());
    let indices = [0, 101, 500, 799];
    let mut decoder = SampledVideo::start(&path, metadata, &indices).unwrap();
    for index in indices {
        let (_, _, pixels) = decoder.next_frame().await.unwrap().unwrap().into_parts();
        assert_eq!(pixels, frames[index]);
    }
    assert!(decoder.next_frame().await.unwrap().is_none());
    Ok(())
}
