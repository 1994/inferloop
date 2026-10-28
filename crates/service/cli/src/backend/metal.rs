use super::{SelectedBackend, Selection};
use crate::constants::DEFAULT_KV_PAGE_TOKENS;
use crate::constants::DEFAULT_PREFILL_CHUNK_TOKENS;
use crate::constants::DEFAULT_STAGING_MIB;
use crate::constants::MIB;
use crate::support::memory::utilization_bytes;
use infer_backend_metal::{MetalBackend, MetalConfig};
use infer_core::{Error, ModelId, Result};
use std::path::Path;

pub fn available() -> bool {
    MetalBackend::available()
}
pub fn load(path: &Path, memory_bytes: u64, selection: &Selection) -> Result<SelectedBackend> {
    // Metal already loads only the text graph; CUDA alone binds an optional vision tower.
    let _ = (selection.autotune, selection.text_only);
    let recommended = MetalBackend::recommended_memory_bytes()
        .ok_or_else(|| Error::unsupported("no Metal device available"))?;
    let share = utilization_bytes(recommended, selection.gpu_memory_utilization)?;
    let memory_bytes = if memory_bytes == 0 {
        share
    } else {
        memory_bytes.min(share)
    };
    if matches!(selection.num_speculative_tokens, Some(super::SpeculativeTokens::Depth(depth)) if depth != 0)
    {
        return Err(Error::unsupported(
            "MTP draft depth requires the CUDA resident backend",
        ));
    }
    if selection.draft_model.is_some() {
        return Err(Error::unsupported(
            "a separate draft model requires the CUDA resident backend",
        ));
    }
    let mut package = infer_models::ModelPackage::open(path, ModelId::new(1)?)?;
    Ok(SelectedBackend::Metal(Box::new(
        MetalBackend::from_package(
            &mut package,
            MetalConfig {
                memory_bytes,
                block_size: selection.block_size.unwrap_or(DEFAULT_KV_PAGE_TOKENS),
                kv_cache_blocks: selection.num_gpu_blocks_override,
                prefill_chunk_tokens: selection
                    .max_num_batched_tokens
                    .unwrap_or(DEFAULT_PREFILL_CHUNK_TOKENS),
                upload_staging_bytes: selection
                    .upload_staging_mib
                    .unwrap_or(DEFAULT_STAGING_MIB)
                    .checked_mul(MIB)
                    .ok_or_else(|| Error::invalid("upload staging size overflow"))?,
                ..Default::default()
            },
        )?,
    )))
}
