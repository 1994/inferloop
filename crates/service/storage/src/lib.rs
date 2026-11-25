//! Domain stores written once over the storage primitives, and the in-process backends.
//!
//! A consumer holds a domain trait such as [`infer_spi::ResponseStore`]; the adapters here
//! implement it over an [`infer_spi::RecordStore`], which is all a backend provides. The SQL
//! backends live in `infer-storage-sql`; other backends implement the same primitive.
mod clock;
mod envelope;
mod memory;
mod responses;
pub use memory::{MemoryRecordStore, MemoryRecordStoreConfig};
pub use responses::{RecordResponseStore, ResponseRetention};
