//! Acceptance cases shared by every cuTile kernel source.
//!
//! A source records a `Case` through the state contract and hands the output back to the case,
//! so the host tests and the AOT example compare the same inputs, tolerances and expected values
//! instead of keeping parallel copies.
pub mod elementwise;
pub mod norm;
pub mod pool_gemv;
pub mod recurrent;
pub mod rope;
