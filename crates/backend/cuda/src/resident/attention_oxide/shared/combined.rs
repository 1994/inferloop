//! Combined vector staging and PV consumer artifact loader.
use super::super::Support;
use crate::device::{CudaDevice, device_error};
use infer_core::Result;
use std::sync::{Arc, OnceLock};

const PTX: &str = include_str!("../../../../kernels/vector-attention-combined/v-combined.ptx");
const THREADS: u32 = 384;
const ROWS_PER_CTA: usize = 12;
const LANES: &[usize] = &[256];
/// Version of the host launch contract, as for the vector kernel: bump it only when a launch of
/// the same artifact would run differently.
const HOST_CONTRACT: &str = "vector-attention-combined-launch-v1";

impl Support {
    /// Load the SM120 combined vector key/consumer module.
    pub(crate) fn load_combined(device: &CudaDevice) -> Result<Option<Self>> {
        if !device.target().has_simt_artifacts() {
            return Ok(None);
        }
        let module = device.tile().load_ptx(PTX).map_err(device_error)?;
        Ok(Some(Self {
            function: Arc::new(
                module
                    .function("attention_fp8_256_vec_k_combined")
                    .map_err(device_error)?,
            ),
            stream: device.tile().stream_id(),
            threads: THREADS,
            rows_per_cta: ROWS_PER_CTA,
            lanes: LANES,
        }))
    }

    /// Artifact identity of the combined kernel: PTX, source, provenance and launch contract.
    pub(crate) fn combined_identity() -> &'static str {
        use sha2::{Digest, Sha256};
        static IDENTITY: OnceLock<String> = OnceLock::new();
        IDENTITY.get_or_init(|| {
            let source =
                include_str!("../../../../kernels/vector-attention-combined/v-combined.cu");
            let provenance =
                include_str!("../../../../kernels/vector-attention-combined/provenance.json");
            let launch = format!("{HOST_CONTRACT}:{THREADS}:{ROWS_PER_CTA}:{LANES:?}");
            format!(
                "{:x}:{:x}:{:x}:{:x}",
                Sha256::digest(PTX),
                Sha256::digest(source),
                Sha256::digest(provenance),
                Sha256::digest(launch)
            )
        })
    }
}
