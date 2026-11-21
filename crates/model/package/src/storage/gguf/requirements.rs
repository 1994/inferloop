//! Checkpoint capabilities a GGUF package requires.
use infer_core::{Error, Result};
use infer_format::GgufFile;
use infer_ir::{CheckpointFormat, ExecutableEncoding};
use std::path::Path;

/// Container and weight encodings a package needs from a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageRequirements {
    /// Checkpoint container.
    pub checkpoint: CheckpointFormat,
    /// Encodings used by at least one tensor.
    pub encodings: Vec<ExecutableEncoding>,
}

/// Inspect a GGUF file and report the backend capabilities it needs.
/// # Errors
/// Rejects a file whose tensor types are not named by the capability model.
pub fn gguf_requirements(path: impl AsRef<Path>) -> Result<PackageRequirements> {
    let file = GgufFile::open(path)?;
    let mut encodings = Vec::new();
    for tensor in file.tensors().values() {
        let encoding = encoding_for(tensor.ty()).map_err(|error| {
            Error::unsupported(format!(
                "GGUF tensor {} uses unsupported type {}: {error}",
                tensor.name(),
                tensor.ty().name()
            ))
        })?;
        if !encodings.contains(&encoding) {
            encodings.push(encoding);
        }
    }
    Ok(PackageRequirements {
        checkpoint: CheckpointFormat::Gguf,
        encodings,
    })
}

/// Capability encoding for one GGUF tensor type.
/// # Errors
/// Rejects a type the native kernels do not cover yet.
pub fn encoding_for(ty: infer_format::GgufType) -> Result<ExecutableEncoding> {
    Ok(match ty {
        infer_format::GgufType::F32
        | infer_format::GgufType::F16
        | infer_format::GgufType::Bf16 => ExecutableEncoding::Dense,
        other => ExecutableEncoding::GroupedInt(
            other
                .grouped_int()
                .ok_or_else(|| Error::unsupported(other.name()))?,
        ),
    })
}
