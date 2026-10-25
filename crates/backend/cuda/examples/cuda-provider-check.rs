//! Actual-model acceptance through the engine SPI, including request ownership checks.
#[cfg(cuda_backend)]
use std::error::Error;
#[cfg(cuda_backend)]
#[path = "provider_check/mod.rs"]
mod check;
#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn Error>> {
    check::run()
}
#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA provider check requires Linux");
    exit(1);
}
