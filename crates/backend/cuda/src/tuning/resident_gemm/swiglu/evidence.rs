//! Shape and evidence policy for the existing fused FP4 kernel.
use super::super::{GemmConfig, GemmTrial, MainLoop, MeasurementOrder, ResidentGemmKey, confirmed};
use super::super::{TILE_M_SEARCH, TILE_N_SEARCH};
use crate::constants::{NVFP4_GROUP_SIZE, PROMPT_GEMM_TILE_ROWS, QUANT_GEMM_TILE};
use crate::tuning::projection_budget::ProjectionShape;
use serde::{Deserialize, Serialize};

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
/// Fused gate/up choices are distinct from standalone GEMM cache entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: ResidentGemmKey,
    pub selected: GemmConfig,
    pub bitwise_verified: bool,
    pub trials: Vec<GemmTrial>,
    pub rejected: Vec<(GemmConfig, String)>,
    pub confirmation: GemmTrial,
}
impl Entry {
    pub(super) fn valid(&self) -> bool {
        let shape = [self.key.m, self.key.n, self.key.k];
        let valid = |trial: &GemmTrial| trial.valid() && compatible(trial.config, shape);
        self.bitwise_verified
            && !self.key.kernel_fingerprint.is_empty()
            && self
                .key
                .driver_build
                .as_ref()
                .is_none_or(|value| !value.is_empty())
            && compatible(self.selected, shape)
            && compatible(baseline(), shape)
            && self.confirmation.config == self.selected
            && self.confirmation.order == MeasurementOrder::CandidateFirst
            && valid(&self.confirmation)
            && self
                .trials
                .iter()
                .all(|trial| trial.order == MeasurementOrder::BaselineFirst && valid(trial))
            && self
                .rejected
                .iter()
                .all(|(config, reason)| compatible(*config, shape) && !reason.is_empty())
            && (self.selected == baseline()
                || confirmed(&self.confirmation)
                    && self
                        .trials
                        .iter()
                        .any(|trial| trial.config == self.selected && confirmed(trial)))
    }
}
