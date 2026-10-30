//! Bounded orchestration for sampled video-frame processing.
use super::{SourcePolicy, materialize, read_url};
use crate::cpu::CpuPool;
use futures_util::stream::{self, TryStreamExt};
use infer_core::{Error, Result};
use infer_media_input::{SampledVideo, probe_video};
use infer_models::{PreprocessedVideo, RgbImage, VideoPlan, VideoProcessor};
use std::sync::Arc;

/// Decode only model-selected source frames and process temporal groups on the shared CPU pool.
pub(super) async fn resolve(
    url: String,
    downloaded: Option<Vec<u8>>,
    policy: SourcePolicy,
    processor: VideoProcessor,
    pool: CpuPool,
) -> Result<Vec<PreprocessedVideo>> {
    let retained = url
        .len()
        .checked_add(downloaded.as_ref().map_or(0, Vec::len))
        .ok_or_else(|| Error::invalid("video input size overflow"))?;
    let temp = pool
        .run_when_available(retained, move |context| {
            context.check()?;
            let bytes = match downloaded {
                Some(bytes) => bytes,
                None => read_url(&url, None, policy, Some(context))?,
            };
            materialize(&bytes, "video")
        })
        .await?;
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
