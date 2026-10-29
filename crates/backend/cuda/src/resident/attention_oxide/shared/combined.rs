//! Combined vector staging and PV consumer artifact loader.
//!
//! Declared from `shared.rs` so the combined loader has its own artifact identity. The device
//! architecture classifier is part of that identity because it controls loader eligibility.
use super::super::Support;
use crate::device::architecture::is_sm120;
use crate::device::{CudaDevice, device_error};
use infer_core::Result;
use std::sync::{Arc, OnceLock};

const PTX: &str = include_str!("../../../../kernels/vector-attention-combined/v-combined.ptx");
const THREADS: u32 = 384;
const ROWS_PER_CTA: usize = 12;
const LANES: &[usize] = &[256];

impl Support {
    /// Load the SM120 combined vector key/consumer module.
    pub(crate) fn load_combined(device: &CudaDevice) -> Result<Option<Self>> {
        if !is_sm120(&device.profile()?.architecture) {
            return Ok(None);
        }
        let module = device
            .stream
            .device()
            .load_module_from_ptx_src(PTX)
            .map_err(device_error)?;
        Ok(Some(Self {
            function: Arc::new(
                module
                    .load_function("attention_fp8_256_vec_k_combined")
                    .map_err(device_error)?,
            ),
            stream: device.stream.cu_stream() as usize,
            threads: THREADS,
            rows_per_cta: ROWS_PER_CTA,
            lanes: LANES,
        }))
    }

    /// Artifact identity of the combined kernel: PTX, source, provenance and this loader.
    pub(crate) fn combined_identity() -> &'static str {
        use sha2::{Digest, Sha256};
        static IDENTITY: OnceLock<String> = OnceLock::new();
        IDENTITY.get_or_init(|| {
            let source =
                include_str!("../../../../kernels/vector-attention-combined/v-combined.cu");
            let provenance =
                include_str!("../../../../kernels/vector-attention-combined/provenance.json");
            format!(
                "{:x}:{:x}:{:x}:{:x}:{:x}",
                Sha256::digest(PTX),
                Sha256::digest(source),
                Sha256::digest(provenance),
                Sha256::digest(include_str!("combined.rs")),
                Sha256::digest(include_str!("../../../device/architecture.rs"))
            )
        })
    }
}
