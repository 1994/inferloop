//! Share backend availability with the CLI without adding a runtime dependency.
#[path = "../../../tools/build/backend_cfg.rs"]
mod backend_cfg;

use std::env::{self, VarError};

fn main() -> Result<(), VarError> {
    println!("cargo:rerun-if-changed=../../../tools/build/backend_cfg.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_AOT");
    // Either kernel source makes the backend available; the host code is the same for both.
    let cuda = ["CARGO_FEATURE_CUDA", "CARGO_FEATURE_CUDA_AOT"]
        .iter()
        .any(|feature| env::var_os(feature).is_some());
    backend_cfg::emit(&env::var("CARGO_CFG_TARGET_OS")?, cuda);
    Ok(())
}
