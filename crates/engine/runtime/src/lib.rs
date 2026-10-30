//! Backend-independent inference orchestration and bounded CPU ownership.
mod config;
mod constants;
mod cpu;
mod engine;
mod lengths;
mod observation;
mod persistence;
mod pipeline;
mod preparation;
mod requests;
mod resource;
pub mod runner;
mod stages;

pub use config::RuntimeConfig;
pub use cpu::plans::StepPool as CpuStepPool;
pub use cpu::{CpuRuntimeConfig, CpuRuntimeInspection, OwnerPlacement};
pub use engine::{
    CompletedRequest, Engine, EngineFault, EngineOutput, InFlight, RuntimeInspection, TenantService,
};
pub use lengths::{LengthLimitSources, LimitSource, ResolvedLengthLimits};
pub use observation::ObservationSnapshot;
pub use persistence::{ReplayAction, RuntimeSnapshot};
pub use pipeline::scheduling::index::ReadyWindow as ReadyWorkBuffer;
pub use preparation::{PreparedRequest, RequestPreparer};
pub use requests::{RequestRecord, TokenContext};
