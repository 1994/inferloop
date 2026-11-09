//! Shape and evidence policy for the existing fused FP4 kernel.
use super::super::{GemmConfig, MainLoop, ResidentGemmKey};
use super::super::{TILE_M_SEARCH, TILE_N_SEARCH};
use crate::constants::{NVFP4_GROUP_SIZE, PROMPT_GEMM_TILE_ROWS, QUANT_GEMM_TILE};
use crate::tuning::projection_budget::ProjectionShape;
use crate::tuning::search::{Protocol, Record};

/// Conservative tile whose arithmetic and complete-model path are already qualified.
pub const fn baseline() -> GemmConfig {
    GemmConfig {
        tile_m: PROMPT_GEMM_TILE_ROWS,
        tile_n: QUANT_GEMM_TILE[1],
        main_loop: MainLoop::Plain,
    }
}
pub fn compatible(config: GemmConfig, shape: ProjectionShape) -> bool {
    let [m, n, k] = shape;
    config.valid()
        && config.main_loop == MainLoop::Plain
        && !shape.contains(&0)
        && m.is_multiple_of(config.tile_m)
        && n.is_multiple_of(config.tile_n)
        && k.is_multiple_of(NVFP4_GROUP_SIZE)
}
pub fn candidates(shape: ProjectionShape) -> impl Iterator<Item = GemmConfig> {
    TILE_M_SEARCH.into_iter().flat_map(move |tile_m| {
        TILE_N_SEARCH.into_iter().filter_map(move |tile_n| {
            let config = GemmConfig {
                tile_m,
                tile_n,
                main_loop: MainLoop::Plain,
            };
            compatible(config, shape).then_some(config)
        })
    })
}
/// Fused gate/up choices are distinct from standalone GEMM cache entries: the same record shape,
/// kept in its own family.
pub type Entry = Record<ResidentGemmKey, GemmConfig>;

/// Every compatible tile a search measures against the baseline.
pub(super) fn protocol(shape: ProjectionShape) -> Protocol<GemmConfig> {
    Protocol {
        baseline: baseline(),
        candidates: candidates(shape)
            .filter(|config| *config != baseline())
            .collect(),
    }
}

/// The record is a complete search over the shape's compatible tiles. A free function, because
/// standalone GEMM records share the type and define their own rules.
pub fn valid(entry: &Entry) -> bool {
    let shape = [entry.key.m, entry.key.n, entry.key.k];
    !entry.key.kernel_fingerprint.is_empty()
        && entry
            .key
            .driver_build
            .as_ref()
            .is_none_or(|value| !value.is_empty())
        && compatible(baseline(), shape)
        && entry
            .evidence
            .valid(&protocol(shape), |config| compatible(config, shape))
}
