//! Command-line entry point; argument parsing and execution live in their own modules.
mod arguments;
mod backend;
mod commands;
mod constants;
mod dispatch;
#[cfg(native_backend)]
mod host_quality;
mod support;

// The executable selects the allocator for every backend's Rust host allocations.
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    dispatch::run();
}
