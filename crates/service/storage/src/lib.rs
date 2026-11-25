//! Domain stores written once over the storage primitives, and the in-process backends.
//!
//! A consumer holds a domain trait such as [`infer_spi::ResponseStore`]; the adapters here
//! implement it over an [`infer_spi::RecordStore`], which is all a backend provides. The SQL
//! backends live in `infer-storage-sql`; other backends implement the same primitive. A
//! deployment names a backend by a [`StorageLocation`], whose scheme selects an opener
//! registered in [`RecordBackends`].
mod backends;
mod clock;
mod envelope;
mod location;
mod memory;
mod responses;
pub use backends::{RecordBackends, RecordOpener};
pub use location::StorageLocation;
pub use memory::{MemoryRecordStore, MemoryRecordStoreConfig};
pub use responses::{RecordResponseStore, ResponseRetention};
