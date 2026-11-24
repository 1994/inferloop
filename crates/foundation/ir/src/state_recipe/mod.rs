//! Cold-compiled physical state contract; sizing and region traversal allocate no memory.
mod compile;
mod layout;
#[cfg(test)]
mod tests;
use crate::{OutputReadout, StateKind};
use infer_core::TensorId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateRegionKind {
    Tensor { tensor: TensorId, kind: StateKind },
    Hidden,
    Logits,
    Tokens,
    PageTable,
    Probes,
    BlockLeases,
    ReadbackRows,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateMemory {
    DevicePrivate,
    HostMirror,
    /// One physical KV block, shared by sequences through generation-checked leases.
    KvBlock,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateExtent {
    Fixed(usize),
    Tokens(usize),
    Pages,
    Hidden(usize),
    FullTokens(usize),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateReset {
    Zero,
    InvalidPage,
    ClearCursor,
    /// Shared blocks may only be reclaimed after their last lease and execution fence.
    ReleaseLease,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRegion {
    pub kind: StateRegionKind,
    pub memory: StateMemory,
    pub extent: StateExtent,
    pub element_bytes: usize,
    pub reset: StateReset,
}
/// Every KV, recurrent/history, output, token, page-table and optional probe allocation.
/// Weights and per-batch activation scratch are deliberately separate program resources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRecipe {
    pub block_size: usize,
    pub max_tokens: usize,
    pub regions: Vec<StateRegion>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateLayout {
    pub capacity: usize,
    pub readout: OutputReadout,
    pub private_bytes: u64,
    pub host_bytes: u64,
    pub kv_block_bytes: u64,
    pub max_pages: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateRegionLayout {
    pub region: StateRegion,
    pub elements: usize,
    /// Offset in its memory class, not an assumption that separately allocated buffers alias.
    pub offset: u64,
    pub bytes: u64,
}
