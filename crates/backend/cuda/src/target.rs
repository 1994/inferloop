//! Explicit hardware profiles. Architecture identifiers are not feature levels.
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum CudaTarget {
    HopperSm90,
    BlackwellSm120,
    Other(String),
}
