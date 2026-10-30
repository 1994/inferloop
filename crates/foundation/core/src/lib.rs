//! Stable identities, lifecycle, bounded storage and allocation-free semantic events.
pub mod arena;
pub(crate) mod constants;
pub mod credits;
mod error;
pub mod event;
pub mod id;
pub mod lifecycle;
pub mod map;
pub mod placement;
pub mod set;
pub use error::{Error, ErrorCode, Result};
pub use id::{
    AllocationId, BatchHandle, DecisionId, DeviceId, ExperimentId, IdAllocator, KernelId, MediaId,
    ModelId, OpId, OwnerId, ProgramId, ProviderId, RequestHandle, RequestId, SessionId, SnapshotId,
    StableId, StateId, StatePageId, StepId, TensorId, WorkloadId, new_owner_id,
};
pub use lifecycle::{FinishReason, RequestStatus, WaitReason};
