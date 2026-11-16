#![expect(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    reason = "integration-test fixtures build tiny synthetic containers"
)]

use infer_core::ErrorCode;
use infer_format::{GgufFile, GgufLimits, GgufType, MetadataValue};
use std::env::temp_dir;
use std::fs::{create_dir_all, read, remove_dir_all, write};
use std::path::PathBuf;
use std::process::id;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

const ALIGNMENT: u32 = 32;

fn temp() -> PathBuf {
    let path = temp_dir().join(format!(
        "infer-format-{}-{}",
        id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    create_dir_all(&path).unwrap();
    path
}

enum Meta {
    Str(&'static str),
    U32(u32),
    Strings(&'static [&'static str]),
}

struct Tensor {
    name: &'static str,
    dims: &'static [u64],
    ty: u32,
    offset: u64,
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_str(bytes: &mut Vec<u8>, value: &str) {
    push_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn container(metadata: &[(&str, Meta)], tensors: &[Tensor], data: &[u8]) -> Vec<u8> {
    let mut header = b"GGUF".to_vec();
    push_u32(&mut header, 3);
    push_u64(&mut header, tensors.len() as u64);
    push_u64(&mut header, metadata.len() as u64);
    for (key, value) in metadata {
        push_str(&mut header, key);
        match value {
            Meta::Str(text) => {
                push_u32(&mut header, 8);
                push_str(&mut header, text);
            }
            Meta::U32(number) => {
                push_u32(&mut header, 4);
                push_u32(&mut header, *number);
            }
            Meta::Strings(values) => {
                push_u32(&mut header, 9);
                push_u32(&mut header, 8);
                push_u64(&mut header, values.len() as u64);
                for value in *values {
                    push_str(&mut header, value);
                }
            }
        }
    }
    for tensor in tensors {
        push_str(&mut header, tensor.name);
        push_u32(&mut header, tensor.dims.len() as u32);
        for dim in tensor.dims {
            push_u64(&mut header, *dim);
        }
        push_u32(&mut header, tensor.ty);
        push_u64(&mut header, tensor.offset);
    }
    let alignment = usize::try_from(ALIGNMENT).unwrap();
    let data_start = header.len().div_ceil(alignment) * alignment;
    header.resize(data_start, 0);
    header.extend_from_slice(data);
    header
}

fn base_metadata() -> Vec<(&'static str, Meta)> {
    vec![
        ("general.architecture", Meta::Str("test")),
        ("general.alignment", Meta::U32(ALIGNMENT)),
    ]
}

fn weights() -> Vec<u8> {
    let mut data = vec![0u8; 32 + 144];
    data[..4].copy_from_slice(&1.5f32.to_le_bytes());
    data[4..8].copy_from_slice(&(-2.0f32).to_le_bytes());
    data[8..12].copy_from_slice(&0.25f32.to_le_bytes());
    data[12..16].copy_from_slice(&3.0f32.to_le_bytes());
    data[32..32 + 144].fill(0x21);
    data
}

fn base_tensors() -> Vec<Tensor> {
    vec![
        Tensor {
            name: "weight",
            dims: &[4],
            ty: 0,
            offset: 0,
        },
        Tensor {
            name: "quant",
            dims: &[256],
            ty: 12,
            offset: 32,
        },
    ]
}

#[test]
fn parses_metadata_geometry_and_payload() {
    let root = temp();
    let path = root.join("model.gguf");
    write(
        &path,
        container(&base_metadata(), &base_tensors(), &weights()),
    )
    .unwrap();
    let file = GgufFile::open(&path).unwrap();
    assert_eq!(file.version(), 3);
    assert_eq!(file.architecture(), Some("test"));
    assert_eq!(file.data_start() % u64::from(ALIGNMENT), 0);
    let weight = file.tensor("weight").unwrap();
    assert_eq!(weight.ty(), GgufType::F32);
    assert_eq!(weight.shape().unwrap(), [4]);
    assert_eq!(weight.byte_len(), 16);
    let quant = file.tensor("quant").unwrap();
    assert_eq!(quant.ty(), GgufType::Q4K);
    assert_eq!(quant.elements().unwrap(), 256);
    assert_eq!(quant.byte_len(), 144);
    assert_eq!(
        file.read_tensor("weight", 16).unwrap(),
        [
            0, 0, 0xc0, 0x3f, 0, 0, 0, 0xc0, 0, 0, 0x80, 0x3e, 0, 0, 0x40, 0x40
        ]
    );
    assert_eq!(
        file.read_tensor("weight", 15).unwrap_err().code,
        ErrorCode::Capacity
    );
    assert_eq!(file.tensors().len(), 2);
    remove_dir_all(root).unwrap();
}

#[test]
fn parses_string_arrays() {
    let root = temp();
    let path = root.join("model.gguf");
    let mut metadata = base_metadata();
    metadata.push(("test.tags", Meta::Strings(&["alpha", "beta"])));
    write(&path, container(&metadata, &base_tensors(), &weights())).unwrap();
    let file = GgufFile::open(&path).unwrap();
    let value = file.metadata().get("test.tags").unwrap();
    let MetadataValue::Array(values) = value else {
        panic!("expected an array");
    };
    assert_eq!(values.len(), 2);
    assert_eq!(values[1].as_str(), Some("beta"));
    remove_dir_all(root).unwrap();
}

fn malformed_cases() -> Vec<(&'static str, Vec<u8>, ErrorCode)> {
    let bad_magic = {
        let mut bytes = container(&base_metadata(), &base_tensors(), &weights());
        bytes[..4].copy_from_slice(b"XXXX");
        bytes
    };
    let version_one = {
        let mut bytes = container(&base_metadata(), &base_tensors(), &weights());
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        bytes
    };
    let duplicate_metadata = {
        let mut metadata = base_metadata();
        metadata.push(("general.architecture", Meta::Str("other")));
        container(&metadata, &base_tensors(), &weights())
    };
    let duplicate_tensor = container(
        &base_metadata(),
        &[
            Tensor {
                name: "same",
                dims: &[4],
                ty: 0,
                offset: 0,
            },
            Tensor {
                name: "same",
                dims: &[4],
                ty: 0,
                offset: 16,
            },
        ],
        &weights(),
    );
    let unaligned = container(
        &base_metadata(),
        &[Tensor {
            name: "quant",
            dims: &[256],
            ty: 12,
            offset: 1,
        }],
        &weights(),
    );
    let overlapping = container(
        &base_metadata(),
        &[
            Tensor {
                name: "a",
                dims: &[4],
                ty: 0,
                offset: 0,
            },
            Tensor {
                name: "b",
                dims: &[4],
                ty: 0,
                offset: 0,
            },
        ],
        &weights(),
    );
    let past_end = container(
        &base_metadata(),
        &[Tensor {
            name: "quant",
            dims: &[256],
            ty: 12,
            offset: 32,
        }],
        &[0u8; 16],
    );
    let unknown_type = container(
        &base_metadata(),
        &[Tensor {
            name: "unknown",
            dims: &[256],
            ty: 255,
            offset: 0,
        }],
        &weights(),
    );
    vec![
        ("bad-magic", bad_magic, ErrorCode::InvalidInput),
        ("version-one", version_one, ErrorCode::Unsupported),
        (
            "duplicate-metadata",
            duplicate_metadata,
            ErrorCode::InvalidInput,
        ),
        (
            "duplicate-tensor",
            duplicate_tensor,
            ErrorCode::InvalidInput,
        ),
        ("unaligned", unaligned, ErrorCode::InvalidInput),
        ("overlapping", overlapping, ErrorCode::InvalidInput),
        ("past-end", past_end, ErrorCode::InvalidInput),
        ("unknown-type", unknown_type, ErrorCode::Unsupported),
    ]
}

#[test]
fn rejects_malformed_headers() {
    let root = temp();
    for (name, bytes, code) in malformed_cases() {
        let path = root.join(name);
        write(&path, bytes).unwrap();
        assert_eq!(GgufFile::open(&path).unwrap_err().code, code, "{name}");
    }
    remove_dir_all(root).unwrap();
}

#[test]
fn enforces_header_and_string_limits() {
    let root = temp();
    let path = root.join("model.gguf");
    let bytes = container(&base_metadata(), &base_tensors(), &weights());
    write(&path, &bytes).unwrap();
    let tiny_header = GgufLimits {
        header_bytes: 16,
        ..GgufLimits::default()
    };
    assert_eq!(
        GgufFile::open_with(&path, &tiny_header).unwrap_err().code,
        ErrorCode::Capacity
    );
    let tiny_string = GgufLimits {
        string_bytes: 4,
        ..GgufLimits::default()
    };
    assert_eq!(
        GgufFile::open_with(&path, &tiny_string).unwrap_err().code,
        ErrorCode::Capacity
    );
    let tiny_entries = GgufLimits {
        tensor_entries: 1,
        ..GgufLimits::default()
    };
    assert_eq!(
        GgufFile::open_with(&path, &tiny_entries).unwrap_err().code,
        ErrorCode::Capacity
    );
    assert_eq!(read(&path).unwrap(), bytes);
    remove_dir_all(root).unwrap();
}
