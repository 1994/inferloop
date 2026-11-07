//! Device work outside the kernel families: eager projections and probes, parameter packing, and
//! metadata writes before a graph's replay.
pub use super::projection::fp4_probe::initialize as initialize_fp4_probe;
pub use super::projection::matvec::{dense_matvec, fp8_matvec, nvfp4_matvec};
pub use super::recurrent::pack_parameters;
pub use super::state::{load_rotary, update as update_metadata, update_rotary};
