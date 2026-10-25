//! Keep explicit CUDA selection errors available when the implementation is not compiled.
use super::{SelectedBackend, Selection};
use infer_core::{Error, Result};
use std::path::Path;

pub fn load(_path: &Path, _memory_bytes: u64, _selection: &Selection) -> Result<SelectedBackend> {
    Err(Error::unsupported(
        "CUDA execution requires Linux and a build with --features cuda",
    ))
}
