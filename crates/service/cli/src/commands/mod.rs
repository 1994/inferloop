//! Typed command options and their execution paths.
use crate::{backend, support::print, support::read_json};
#[cfg(native_backend)]
use crate::{
    host_quality, support::axum_serve, support::config, support::example, support::results,
    support::run_requests, support::run_to_idle, support::selected_engine, support::write_json,
};

mod options;
pub use options::*;

#[cfg(any(all(test, unix), cuda_backend))]
mod tuning_worker;

#[cfg(native_backend)]
mod server;

#[cfg(native_backend)]
mod inference;

mod verification;

#[cfg(native_backend)]
mod replay;

mod measurement;

mod models;

#[cfg(native_backend)]
pub use inference::run;
pub use measurement::compare;
#[cfg(native_backend)]
pub use measurement::{benchmark, profile};
pub use models::{inspect_model, inspect_package, tokenize};
#[cfg(native_backend)]
pub use replay::replay;
#[cfg(native_backend)]
pub use server::serve;
pub use verification::verify;
