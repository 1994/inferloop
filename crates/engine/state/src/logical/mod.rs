//! Device-independent page ownership. Physical storage lives behind `StateStorageProvider`.
mod inspection;
mod lifecycle;
mod prefix;
mod reservation;
use infer_core::{IdAllocator, ModelId, RequestId, StateId, StatePageId, map::BoundedMap};
use infer_ir::StateKind;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PrefixKey {
    pub model: ModelId,
    pub revision: String,
    pub precision: String,
    pub layout: String,
    pub tokens: Vec<u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Page {
    references: usize,
    sealed: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceState {
    pub owner: RequestId,
    pub kind: StateKind,
    pub pages: Vec<StatePageId>,
    pub capacity_tokens: usize,
    pub committed_tokens: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub total_pages: usize,
    pub allocated_pages: usize,
    pub free_pages: usize,
    pub sequence_count: usize,
    pub cached_prefix_count: usize,
    pub references: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceStateManager {
    #[serde(skip)]
    spare_tables: Vec<Vec<StatePageId>>,
    total_pages: usize,
    block_size: usize,
    ids: IdAllocator,
    pages: BoundedMap<StatePageId, Page>,
    sequences: BoundedMap<StateId, SequenceState>,
    owners: BoundedMap<RequestId, StateId>,
    // A vector permits JSON checkpoints without lossy stringification of composite keys.
    prefixes: Vec<(PrefixKey, Vec<StatePageId>)>,
}
#[cfg(test)]
mod tests;
