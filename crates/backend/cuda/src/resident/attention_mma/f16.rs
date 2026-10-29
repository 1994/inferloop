//! Validated SM120 plain-FP16 FA4 target profile.
//!
//! This is deliberately narrower than the BF16 profile: only SM120, only the fixed
//! 24/4/256 full-causal geometry and only target FP8-KV prefill. Every other target shape and
//! the entire draft proposal path keep their existing arithmetic. The artifact identity and
//! tuning schema are independent from BF16 so cache entries can never cross profiles.
use crate::device::architecture::is_sm120;
use crate::device::platform::is_linux_x86_64;
use crate::{
    device::CudaDevice,
    tuning::attention::{Choice, Shape},
};
use infer_core::{Error, Result};
use sha2::{Digest, Sha256};

pub const PROFILE: &str = "fa4-f16-qkv:f32-acc:f16-output:gpu-seqlen-v1";
pub const SCHEMA: &str = "fa4-f16:plain-f16-qkv:f32-acc:f16-output:sm120-fixed:v1";
pub const VARIANT: &str = "sm120-f16";
pub const PTX: &str = include_str!("../../../kernels/fa4-attention-f16/bridge.ptx");
const BRIDGE_CU: &[u8] = include_bytes!("../../../kernels/fa4-attention-f16/bridge.cu");
const MANIFEST: &[u8] = include_bytes!("../../../kernels/fa4-attention-f16/manifest.json");
const PROVENANCE: &[u8] = include_bytes!("../../../kernels/fa4-attention-f16/provenance.json");
pub const SO: &[u8] = include_bytes!("../../../kernels/fa4-attention-f16/sm120/attention.so");

/// Whether this exact architecture and geometry has a validated FP16 artifact.
#[must_use]
pub fn supported(
    architecture: &str,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    lanes: usize,
    window: usize,
) -> bool {
    is_linux_x86_64()
        && is_sm120(architecture)
        && super::catalog::geometry(heads, kv_heads, dim, lanes, window)
}

/// Hash of every embedded FP16 source/artifact input. It is intentionally distinct from the
/// BF16 `artifact_identity` even though the host ABI is identical.
#[must_use]
pub fn artifact_identity() -> String {
    let mut hash = Sha256::new();
    hash.update(PTX);
    hash.update(BRIDGE_CU);
    hash.update(MANIFEST);
    hash.update(PROVENANCE);
    hash.update(SO);
    hash.update(PROFILE);
    format!("{:x}", hash.finalize())
}

/// Artifact, architecture and shape scoped identity for logs and tests. No tunable cache is
/// keyed without the artifact identity, so a BF16 entry cannot satisfy this profile.
pub fn fingerprint(device: &CudaDevice, shape: Shape) -> Result<String> {
    let profile = device.profile()?;
    let mut hash = Sha256::new();
    hash.update(artifact_identity());
    hash.update(&profile.architecture);
    hash.update(shape.heads.to_le_bytes());
    hash.update(shape.kv_heads.to_le_bytes());
    hash.update(shape.dim.to_le_bytes());
    hash.update(shape.lanes.to_le_bytes());
    hash.update(shape.window.to_le_bytes());
    Ok(format!("{:x}", hash.finalize()))
}

/// Deterministic target selection for the already validated SM120 artifact.
///
/// This is not a performance search: the probe task explicitly forbids a new performance matrix.
/// It records the distinct schema/artifact identity in the load log and returns the dedicated
/// FP16 choice. Capture still validates the bindings and falls through to canonical on mismatch.
pub fn select(device: &CudaDevice, shape: Shape) -> Result<Choice> {
    let architecture = &device.profile()?.architecture;
    if !supported(
        architecture,
        shape.heads,
        shape.kv_heads,
        shape.dim,
        shape.lanes,
        shape.window,
    ) {
        return Err(Error::invalid(
            "FP16 target attention requires validated SM120 geometry",
        ));
    }
    let identity = fingerprint(device, shape)?;
    tracing::info!(
        target: "infer::tuning",
        attention_profile = "fp16",
        schema = SCHEMA,
        profile = PROFILE,
        artifact_identity = %identity,
        variant = VARIANT,
        ?shape,
        "MMA F16 attention selected"
    );
    Ok(Choice::MmaF16)
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_attention_mma_f16.rs"]
mod tests;
