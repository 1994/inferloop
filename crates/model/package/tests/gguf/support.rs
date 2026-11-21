//! Metadata builders and a minimal GGUF writer for synthetic files.
use infer_format::GgufType;
use infer_format::{Metadata, MetadataValue};
use infer_models::default_registry;
use infer_models::gguf::{GgufArchitecture, GgufProjector, LayerPlan, MappedTensor, TensorSpec};
use std::env::temp_dir;
use std::fs::{create_dir_all, remove_dir_all};
use std::path::PathBuf;
use std::process::id;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// GGUF's default tensor-data alignment.
const ALIGNMENT: usize = 32;
/// GGUF metadata value type tags.
const STRING: u32 = 8;
const ARRAY: u32 = 9;

pub fn metadata(entries: Vec<(&str, MetadataValue)>) -> Metadata {
    Metadata::from_entries(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value)),
    )
}

/// `base` with `key` replaced by `value`, or removed when `value` is `None`.
pub fn with(base: &Metadata, key: &str, value: Option<MetadataValue>) -> Metadata {
    let mut entries: Vec<_> = base
        .keys()
        .filter(|name| *name != key)
        .map(|name| {
            (
                name.to_owned(),
                base.get(name).cloned().expect("listed key"),
            )
        })
        .collect();
    if let Some(value) = value {
        entries.push((key.to_owned(), value));
    }
    Metadata::from_entries(entries)
}

pub fn strings(values: &[&str]) -> MetadataValue {
    MetadataValue::Array(
        values
            .iter()
            .map(|value| MetadataValue::String((*value).into()))
            .collect(),
    )
}

pub fn spec(name: &str, shape: &[usize], ty: GgufType) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        shape: shape.to_vec(),
        ty,
    }
}

pub fn architecture(name: &str) -> &'static dyn GgufArchitecture {
    default_registry()
        .gguf()
        .architecture(name)
        .expect("built-in architecture")
}

pub fn projector(name: &str) -> &'static dyn GgufProjector {
    default_registry()
        .gguf()
        .projector(name)
        .expect("built-in projector")
}

/// Where `architecture` maps one stored tensor, with the layer plan `file` declares.
pub fn mapped(
    architecture: &dyn GgufArchitecture,
    file: &Metadata,
    tensor: &TensorSpec,
) -> Option<MappedTensor> {
    let layers = LayerPlan::of(file, architecture.name()).expect("layer plan");
    architecture.tensor(tensor, &layers)
}

/// A fresh directory under the system temporary directory.
pub fn scratch() -> PathBuf {
    let path = temp_dir().join(format!(
        "infer-gguf-{}-{}",
        id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = remove_dir_all(&path);
    create_dir_all(&path).expect("scratch directory");
    path
}

/// A GGUF v3 container holding `metadata` and no tensors.
pub fn container(metadata: &Metadata) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(metadata.keys().count() as u64).to_le_bytes());
    for key in metadata.keys() {
        push_str(&mut bytes, key);
        push_value(&mut bytes, metadata.get(key).expect("listed key"));
    }
    bytes.resize(bytes.len().next_multiple_of(ALIGNMENT), 0);
    bytes
}

fn push_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

const fn tag(value: &MetadataValue) -> u32 {
    match value {
        MetadataValue::U8(_) => 0,
        MetadataValue::I8(_) => 1,
        MetadataValue::U16(_) => 2,
        MetadataValue::I16(_) => 3,
        MetadataValue::U32(_) => 4,
        MetadataValue::I32(_) => 5,
        MetadataValue::F32(_) => 6,
        MetadataValue::Bool(_) => 7,
        MetadataValue::String(_) => STRING,
        MetadataValue::Array(_) => ARRAY,
        MetadataValue::U64(_) => 10,
        MetadataValue::I64(_) => 11,
        MetadataValue::F64(_) => 12,
    }
}

fn push_value(bytes: &mut Vec<u8>, value: &MetadataValue) {
    bytes.extend_from_slice(&tag(value).to_le_bytes());
    push_payload(bytes, value);
}

fn push_payload(bytes: &mut Vec<u8>, value: &MetadataValue) {
    match value {
        MetadataValue::U8(v) => bytes.push(*v),
        MetadataValue::I8(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::U16(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I16(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::U32(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I32(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::F32(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::Bool(v) => bytes.push(u8::from(*v)),
        MetadataValue::String(v) => push_str(bytes, v),
        MetadataValue::Array(values) => {
            let element = values.first().map_or(STRING, tag);
            bytes.extend_from_slice(&element.to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                push_payload(bytes, value);
            }
        }
        MetadataValue::U64(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I64(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::F64(v) => bytes.extend_from_slice(&v.to_le_bytes()),
    }
}
