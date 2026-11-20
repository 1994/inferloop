//! Backend-independent sealed batches, handle transport, and metadata ownership.
mod batches;
pub(crate) mod constants;
pub use batches::{BatchArena, BatchLease};
pub use rtrb::{Consumer, Producer, RingBuffer};

pub const MAX_SUBMISSION_BATCH: usize = 64;
pub const SUBMISSION_ABI_VERSION: u32 = 1;
mod descriptor;
#[cfg(test)]
mod tests;
pub use descriptor::{SubmissionDescriptor, WorkDescriptor};
mod slots;
pub use slots::SubmissionSlots;
