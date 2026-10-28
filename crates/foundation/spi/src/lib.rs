//! Planning-time extension contracts. Execution is monomorphized over a backend.
mod constraint;
mod generation;
pub use constraint::{ConstraintHandle, GenerationConstraint};
pub use generation::{GenerationPolicy, GenerationPreset, GreedyPreset};
mod planning;
mod resource;
pub use planning::DecisionStorage;
pub use resource::{
    ResourceAbandon, ResourceCommand, ResourcePool, ResourceReply, ResourceResponder,
    ResourceTicket, execute_resource,
};

pub const SPI_VERSION: u32 = 1;
pub mod backend;
pub mod cost;
pub mod extensions;
pub mod kernel;
pub mod model;
pub mod observer;
pub mod protocol;
pub mod provider;
pub mod scheduling;
pub mod state;
pub mod workload;
pub use backend::BackendProvider;
pub use cost::{BatchCostSummary, CostModelProvider};
pub use extensions::{
    FeatureProvider, FusionProvider, ModalityProvider, PrecisionProvider, PreparedMedia,
    SpeculationProvider,
};
pub use kernel::{KernelProvider, KernelRegistration, SourceLocation};
pub use model::{
    FusionPlan, HiddenActivation, ImportedModel, ModalityEncoder, ModalityPlan, ModelProvider,
    PrecisionPolicy, SpeculationPlan,
};
pub use observer::Observer;
pub use protocol::ProtocolAdapter;
pub use provider::ProviderMetadata;
pub use scheduling::{AdmissionPolicy, PlanningContext, SchedulingPolicy};
pub use state::{
    LocalOnlyTransfer, SequenceStateProvider, StateStorageProvider, StateTransferProvider,
    TransferTicket,
};
pub use workload::WorkloadProvider;
