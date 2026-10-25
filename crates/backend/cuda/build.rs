//! Share backend availability with the CLI without adding a runtime dependency.
#[path = "../../../tools/build/backend_cfg.rs"]
mod backend_cfg;

use std::env::{self, VarError};

fn main() -> Result<(), VarError> {
    println!("cargo:rerun-if-changed=../../../tools/build/backend_cfg.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    backend_cfg::emit(
        &env::var("CARGO_CFG_TARGET_OS")?,
        env::var_os("CARGO_FEATURE_CUDA").is_some(),
    );
    Ok(())
}
