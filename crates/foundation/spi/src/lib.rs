//! Planning-time extension contracts. Execution is monomorphized over a backend.
mod constraint;
pub use constraint::{ConstraintHandle, GenerationConstraint};
mod planning;
mod resource;
mod responses;
pub use planning::DecisionStorage;
pub use resource::{
    ResourceAbandon, ResourceCommand, ResourcePool, ResourceReply, ResourceResponder,
    ResourceTicket, execute_resource,
};
pub use responses::{
    IncompleteReason, RESPONSES_PAYLOAD_VERSION, RESPONSES_SPI_VERSION, ResponseError,
    ResponseEvent, ResponseFuture, ResponseId, ResponseStatus, ResponseStore,
    ResponseStoreCapabilities, ResponseUsage, ResponsesExtensions, StoredResponse, WriteCondition,
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
pub use backend::{BackendProvider, DeviceHostExecutor, ImagePreprocessing};
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
