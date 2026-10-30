//! Protocol-independent, serializable planning and execution representations.
pub mod dataflow;
pub mod diagnostics;
pub mod execution;
pub mod hardware;
mod image;
pub mod model;
pub mod multimodal;
pub mod precision;
pub mod request;
pub mod scheduling;
pub mod state_recipe;
pub mod tokens;
pub mod workload;
pub use dataflow::{
    BufferLifetime, DataflowGraph, TensorNode, TensorOp, TensorSpec, TensorStorage,
};
pub use diagnostics::{
    ConcurrencyInspection, ExecutionProfileInspection, ExecutionStats, KvCacheInspection,
    LayerProbe, OpTrace,
};
pub use execution::{
    CompiledOp, CostEstimate, DecisionWindow, DeferReason, DeferredWork, ExecutionIr,
    ExecutionProgram, ExecutionRole, ExecutionTask, GraphVariant, ModelOutput, Operation,
    OperationIr, PlannedWork, ReadyWork, ResourceSnapshot, SchedulingDecision, StepPlan,
    TaskOutput,
};
pub use hardware::{
    BackendKind, BackendRequirements, CapabilityRequirements, CudaRequirements, DeviceBackend,
    DeviceCapabilities, MetalCapabilities, MetalRequirements, NvidiaArchitecture,
    NvidiaCapabilities, SpeculationCapability,
};
pub use image::{ImageInput, ImageNormalization, ImagePatchLayout, ImageTransform};
pub use model::{
    BackboneKind, FeedForward, Head, Mixer, ModelIr, PositionSpec, StateKind, StateRequirement,
};
pub use multimodal::{
    MediaInput, MediaItem, MediaNode, MediaPixels, MediaPlan, MediaStage, Modality, MultimodalGraph,
};
pub use precision::{DType, PrecisionPlan, ScaleGranularity, ScaleSpec, TensorLayout};
pub use request::{CanonicalRequest, Qos, RequestInput, Sampling};
pub use scheduling::{
    AdmissionConfig, AdmissionDecision, AdmissionInput, AdmissionReason, CostModelConfig,
    CostModelInspection, CostObservation, CostQuery, ExecutionTiming, PageGrowth, PrefillGeometry,
    SchedulerConfig, SelectionEvidence, SelectionReason, TenantQuota, TimingSource,
};
pub use tokens::{ExecutionInput, OutputReadout, TokenBuffer, TokenSpan};
pub use workload::{
    DecisionAnswer, DecisionQuestion, DecisionReadout, DecisionSchema, FeaturePlan, Pooling,
    RankedDocument, SpeculationPlan, Workload, WorkloadOutput, WorkloadPlan,
};

mod buffer;
pub use buffer::SharedOutput;

pub use state_recipe::{
    StateExtent, StateLayout, StateMemory, StateRecipe, StateRegion, StateRegionKind,
    StateRegionLayout, StateReset,
};
