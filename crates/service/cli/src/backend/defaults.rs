//! Resolve the two omitted execution options together; preserve explicit numerical overrides.
use super::{CudaAttentionProfile, SpeculativeTokens};
pub(super) fn resolve(
    depth: Option<SpeculativeTokens>,
    profile: Option<CudaAttentionProfile>,
    fast_supported: bool,
) -> (SpeculativeTokens, CudaAttentionProfile) {
    if depth.is_none() && profile.is_none() && fast_supported {
        (SpeculativeTokens::Auto, CudaAttentionProfile::Bf16)
    } else {
        (
            depth.unwrap_or(SpeculativeTokens::Depth(0)),
            profile.unwrap_or(CudaAttentionProfile::Canonical),
        )
    }
}
#[cfg(test)]
mod tests;
