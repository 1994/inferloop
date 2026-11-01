//! Lossless FP8 staging reuses KV across two positions and six GQA heads.
// The combined loader lives under this unhashed module so the frozen vector/MMA tuning keys
// (which hash `attention_oxide.rs` and `vector.rs`) stay byte-identical.
mod combined;
use super::{Support, device_error};
use crate::device::CudaDevice;
use infer_core::Result;
use std::sync::{Arc, OnceLock};
const PTX: &str = include_str!("../../../kernels/shared-attention/attention-sm120.ptx");
pub(super) const LANES: [usize; 3] = [256, 1024, 2048];
const THREADS: u32 = 384;
const ROWS_PER_CTA: usize = 12;
pub(super) fn load(device: &CudaDevice) -> Result<Option<Support>> {
    if !device.target().has_simt_artifacts() {
        return Ok(None);
    }
    let module = device
        .stream
        .device()
        .load_module_from_ptx_src(PTX)
        .map_err(device_error)?;
    Ok(Some(Support {
        function: Arc::new(
            module
                .load_function("attention_fp8_256")
                .map_err(device_error)?,
        ),
        stream: device.stream.cu_stream() as usize,
        threads: THREADS,
        rows_per_cta: ROWS_PER_CTA,
        lanes: &LANES,
    }))
}
pub(super) fn artifact_identity() -> &'static str {
    use sha2::{Digest, Sha256};
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        let source = include_str!("../../../kernels/shared-attention/attention-sm120.cu");
        let provenance = include_str!("../../../kernels/shared-attention/provenance.json");
        format!(
            "{:x}:{:x}:{:x}",
            Sha256::digest(PTX),
            Sha256::digest(source),
            Sha256::digest(provenance)
        )
    })
}
