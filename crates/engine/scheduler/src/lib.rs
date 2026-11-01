//! Admission, measured batch costs and pure slack/WFQ micro scheduling.
mod admission;
pub(crate) mod constants;
mod cost;
mod policy;
mod pool_demand;
pub use admission::ResourceAdmission;
pub use cost::{CalibratedCosts, FallbackCosts, aggregate, update_summary};
pub use policy::{
    CostAwarePolicy, PackingWorkspace, latest_start, preemption_order, service_charge,
    validate_scheduler,
};
pub use pool_demand::{LiveStates, PoolAction, PoolDemand, PoolFacts, PoolRetry, decide_pool};

pub mod queue;
pub use queue::{BlockedOn, QueueInspection, QueueRequest, QueueState, QueueTiming, RequestQueue};

mod validation;
pub use validation::{
    ValidationScratch, any_feasible, step_cost_queries, step_cost_queries_into, validate_decision,
    validate_decision_reusing, validate_step, validate_step_reusing,
};

mod preemption;
pub use preemption::{PreemptionWorkspace, RecomputePlan, recompute_plan, recompute_plan_into};
