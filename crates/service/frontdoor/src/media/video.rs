//! Bounded orchestration for sampled video-frame processing.
use super::{READ_CHUNK_BYTES, SourcePolicy, TempPath, require_local, source};
use crate::cpu::{CpuContext, CpuPool};
use futures_util::stream::{self, TryStreamExt};
use infer_core::{Error, ErrorCode, Result};
use infer_media_input::{SampledVideo, data_uri_reader, probe_video};
use infer_models::{PreprocessedVideo, RgbImage, VideoPlan, VideoProcessor};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

/// Decode only model-selected source frames and process temporal groups on the shared CPU pool.
pub(super) async fn resolve(
    url: String,
    policy: SourcePolicy,
    processor: VideoProcessor,
    pool: CpuPool,
) -> Result<Vec<PreprocessedVideo>> {
    let temp = if source::is_remote(&url) {
        download(&url, &pool).await?
    } else {
        let retained = url.len();
        pool.run_when_available(retained, move |context| materialize(&url, policy, context))
            .await?
    };
    let metadata = probe_video(&temp.path).await?;
    let temporal = processor.temporal_patch_size;
    let plan = Arc::new(
        pool.run_when_available(0, move |context| {
            context.check()?;
            let height =
                usize::try_from(metadata.height()).map_err(|_| Error::invalid("video height"))?;
            let width =
                usize::try_from(metadata.width()).map_err(|_| Error::invalid("video width"))?;
            processor.plan(metadata.frames(), metadata.fps(), height, width)
        })
        .await?,
    );
    let decoder_plan = plan.clone();
    let path = temp.path.clone();
    let decoder = pool
        .run_when_available(0, move |context| {
            context.check()?;
            let _runtime = context.runtime()?.enter();
            SampledVideo::start(&path, metadata, decoder_plan.indices())
        })
        .await?;
    let read_plan = plan.clone();
    let frames = stream::try_unfold((decoder, 0usize), move |(mut reader, group)| {
        let plan = read_plan.clone();
        async move {
            if group == plan.groups() {
                if reader.next_frame().await?.is_some() {
                    return Err(Error::invariant("video decoder returned an extra frame"));
                }
                return Ok(None);
            }
            let start = group
                .checked_mul(temporal)
                .ok_or_else(|| Error::invalid("video group overflow"))?;
            let count = (plan.indices().len() - start).min(temporal);
            let group_frames = read_frames(&mut reader, count).await?;
            Ok(Some(((group, group_frames), (reader, group + 1))))
        }
    });
    let workers = pool.inspect().workers;
    let processed = frames
        .map_ok(move |(group, frames)| {
            let pool = pool.clone();
            let plan = plan.clone();
            async move { process_group(&pool, plan, group, frames).await }
        })
        .try_buffered(workers)
        .try_collect()
        .await?;
    drop(temp);
    Ok(processed)
}

/// Keep the file ahead of its path so cancellation closes it before unlinking the temporary file.
struct VideoFile {
    file: File,
    temp: TempPath,
}

impl VideoFile {
    fn new() -> Result<Self> {
        let (temp, file) = TempPath::create("video", "media")?;
        Ok(Self { file, temp })
    }

    fn finish(self) -> TempPath {
        let Self { file, temp } = self;
        drop(file);
        temp
    }
}

async fn download(url: &str, pool: &CpuPool) -> Result<TempPath> {
    let mut response = source::stream(url, pool).await?;
    let mut target = pool
        .run_when_available(0, |context| {
            context.check()?;
            VideoFile::new()
        })
        .await?;
    while let Some(chunk) = response.next_chunk().await? {
        let retained = chunk.as_ref().len();
        target = pool
            .run_when_available(retained, move |context| {
                context.check()?;
                target
                    .file
                    .write_all(chunk.as_ref())
                    .map_err(|error| Error::invalid(format!("video file write: {error}")))?;
                Ok(target)
            })
            .await?;
    }
    pool.run_when_available(0, move |context| {
        context.check()?;
        Ok(target.finish())
    })
    .await
}

fn materialize(url: &str, policy: SourcePolicy, context: &CpuContext) -> Result<TempPath> {
    context.check()?;
    let mut target = VideoFile::new()?;
    if url.starts_with("data:") {
        copy_source(data_uri_reader(url)?, &mut target.file, context)?;
    } else {
        let path = if let Some(path) = url.strip_prefix("file://") {
            path
        } else if Path::new(url).is_absolute() {
            url
        } else if url.is_empty() {
            return Err(Error::invalid("media URL is empty"));
        } else {
            return Err(Error::unsupported(
                "media URL must be data:, file://, absolute path, http:// or https://",
            ));
        };
        require_local(policy, url)?;
        let file = File::open(path).map_err(|error| Error::invalid(error.to_string()))?;
        copy_source(file, &mut target.file, context)?;
    }
    Ok(target.finish())
}

fn copy_source(mut source: impl Read, target: &mut File, context: &CpuContext) -> Result<()> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(READ_CHUNK_BYTES).map_err(|error| {
        Error::new(
            ErrorCode::Capacity,
            format!("video source buffer allocation: {error}"),
        )
    })?;
    bytes.resize(READ_CHUNK_BYTES, 0);
    loop {
        context.check()?;
        let count = source
            .read(&mut bytes)
            .map_err(|error| Error::invalid(format!("video source read: {error}")))?;
        if count == 0 {
            return Ok(());
        }
        target
            .write_all(&bytes[..count])
            .map_err(|error| Error::invalid(format!("video file write: {error}")))?;
    }
}

async fn read_frames(reader: &mut SampledVideo, count: usize) -> Result<Vec<RgbImage>> {
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        let frame = reader
            .next_frame()
            .await?
            .ok_or_else(|| Error::invalid("video decoder ended before the sampling plan"))?;
        let (width, height, pixels) = frame.into_parts();
        frames.push(RgbImage {
            width: usize::try_from(width).map_err(|_| Error::invalid("video width"))?,
            height: usize::try_from(height).map_err(|_| Error::invalid("video height"))?,
            pixels,
        });
    }
    Ok(frames)
}

async fn process_group(
    pool: &CpuPool,
    plan: Arc<VideoPlan>,
    group: usize,
    frames: Vec<RgbImage>,
) -> Result<PreprocessedVideo> {
    let retained = frames.iter().try_fold(0usize, |total, frame| {
        total
            .checked_add(frame.pixels.len())
            .ok_or_else(|| Error::invalid("video frame size overflow"))
    })?;
    pool.run_when_available(retained, move |context| {
        context.check()?;
        plan.preprocess_group(group, &frames)
    })
    .await
}
