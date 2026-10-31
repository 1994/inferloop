//! Native planning and output semantics for four orthogonal workloads.
mod constants;
mod sampling;
pub use sampling::{
    SamplingHistory, SamplingWorkspace, TokenMask, probabilities, probabilities_constrained,
    sample, sample_constrained, sample_reusing, sample_with_history, sampling_uniform,
};
mod registry;
pub use registry::WorkloadRegistry;
mod readouts;
pub use readouts::ProjectionWorkloads;
mod math;
mod native;
mod outputs;
#[cfg(test)]
#[path = "../tests/unit/recurrent_reference.rs"]
mod recurrent_reference;
#[cfg(test)]
#[path = "../tests/unit/cases.rs"]
mod tests;
pub use math::{pool, softmax};
pub use native::NativeWorkloads;
use outputs::{decisions, embedding, ranking};
mod recurrent;
pub use recurrent::{ConvGeometry, DeltaGeometry};
mod speculative;
pub use speculative::{
    Verification, accepted_prefix, draft_budget, draft_may_propose, draw_distribution, verify_draft,
};
