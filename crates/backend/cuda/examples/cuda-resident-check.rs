//! GPU numerical checks require Linux and an NVIDIA device.
#[cfg(cuda_backend)]
#[path = "resident_checks/mod.rs"]
mod checks;

#[cfg(cuda_backend)]
fn main() -> infer_core::Result<()> {
    checks::run()
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA resident checks require Linux");
    exit(1);
}
