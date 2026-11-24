//! Planning-time extension contracts. Execution is monomorphized over a backend.
mod constraint;
mod generation;
pub use constraint::{ConstraintHandle, GenerationConstraint};
pub use generation::{GenerationPolicy, GenerationPreset, GreedyPreset};
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
mod backend;
mod cost;
mod extensions;
mod kernel;
mod model;
mod observer;
mod protocol;
mod provider;
mod scheduling;
mod state;
mod workload;
pub use backend::{BackendProvider, DeviceHostExecutor, ImagePreprocessing};
pub use cost::{BatchCostSummary, CostModelProvider};
pub use extensions::{
    FeatureProvider, FusionProvider, ModalityProvider, PrecisionProvider, PreparedMedia,
    SpeculationProvider,
};
pub use kernel::{KernelProvider, KernelRegistration, SourceLocation};
pub use model::{
    DraftHead, FusionPlan, HiddenActivation, ImportedModel, ModalityEncoder, ModalityPlan,
    ModelProvider, PrecisionPolicy,
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
