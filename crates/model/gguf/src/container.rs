//! Bounded GGUF container parsing.
use crate::metadata::{Metadata, MetadataValue};
use crate::tensor::{GgufType, elements, row_major_shape};
use infer_core::{Error, ErrorCode, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// GGUF file magic, little-endian.
const MAGIC: &[u8; 4] = b"GGUF";
/// Bytes of the GGUF magic.
const MAGIC_BYTES: usize = 4;
/// Smallest container version this parser accepts.
const VERSION_MIN: u32 = 2;
/// Largest container version this parser accepts.
const VERSION_MAX: u32 = 3;
/// Default tensor-data alignment when the file does not declare one.
const DEFAULT_ALIGNMENT: u64 = 32;
/// Largest accepted tensor-data alignment.
const MAX_ALIGNMENT: u64 = 64 * 1024 * 1024;
/// Bytes per `u8`.
const U8_BYTES: usize = 1;
/// Bytes per `u16`.
const U16_BYTES: usize = 2;
/// Bytes per `u32`.
const U32_BYTES: usize = 4;
/// Bytes per `u64`.
const U64_BYTES: usize = 8;
/// Bytes per `f32`.
const F32_BYTES: usize = 4;
/// Bytes per `f64`.
const F64_BYTES: usize = 8;
/// Default maximum bytes consumed by metadata and tensor headers.
const DEFAULT_HEADER_BYTES: u64 = 64 * 1024 * 1024;
/// Default maximum single string length.
const DEFAULT_STRING_BYTES: u64 = 16 * 1024 * 1024;
/// Default maximum metadata entries.
const DEFAULT_METADATA_ENTRIES: u64 = 1_000_000;
/// Default maximum tensor entries.
const DEFAULT_TENSOR_ENTRIES: u64 = 1_000_000;
/// Default maximum elements in one metadata array.
const DEFAULT_ARRAY_ELEMENTS: u64 = 4_000_000;
/// Default maximum tensor rank.
const DEFAULT_TENSOR_DIMS: u32 = 8;
/// Maximum nesting of metadata arrays.
const MAX_ARRAY_DEPTH: u32 = 4;
/// Metadata value type id for `u8`.
const KV_U8: u32 = 0;
/// Metadata value type id for `i8`.
const KV_I8: u32 = 1;
/// Metadata value type id for `u16`.
const KV_U16: u32 = 2;
/// Metadata value type id for `i16`.
const KV_I16: u32 = 3;
/// Metadata value type id for `u32`.
const KV_U32: u32 = 4;
/// Metadata value type id for `i32`.
const KV_I32: u32 = 5;
/// Metadata value type id for `f32`.
const KV_F32: u32 = 6;
/// Metadata value type id for `bool`.
const KV_BOOL: u32 = 7;
/// Metadata value type id for a string.
const KV_STRING: u32 = 8;
/// Metadata value type id for an array.
const KV_ARRAY: u32 = 9;
/// Metadata value type id for `u64`.
const KV_U64: u32 = 10;
/// Metadata value type id for `i64`.
const KV_I64: u32 = 11;
/// Metadata value type id for `f64`.
const KV_F64: u32 = 12;
/// The only valid true value of a boolean.
const BOOL_TRUE: u8 = 1;

/// Resource limits applied while reading one container.
#[derive(Debug, Clone, Copy)]
pub struct GgufLimits {
    /// Bytes consumed by metadata and tensor headers.
    pub header_bytes: u64,
    /// Largest accepted single string.
    pub string_bytes: u64,
    /// Largest accepted metadata entry count.
    pub metadata_entries: u64,
    /// Largest accepted tensor count.
    pub tensor_entries: u64,
    /// Largest accepted metadata array element count.
    pub array_elements: u64,
    /// Largest accepted tensor rank.
    pub tensor_dims: u32,
}

impl Default for GgufLimits {
    fn default() -> Self {
        Self {
            header_bytes: DEFAULT_HEADER_BYTES,
            string_bytes: DEFAULT_STRING_BYTES,
            metadata_entries: DEFAULT_METADATA_ENTRIES,
            tensor_entries: DEFAULT_TENSOR_ENTRIES,
            array_elements: DEFAULT_ARRAY_ELEMENTS,
            tensor_dims: DEFAULT_TENSOR_DIMS,
        }
    }
}

/// One tensor's validated descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    ty: GgufType,
    offset: u64,
    byte_len: u64,
}

impl TensorInfo {
    /// Stored tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// GGML dimensions, fastest axis first.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Row-major shape.
    /// # Errors
    /// Returns an invalid-input error when a dimension exceeds the address space.
    pub fn shape(&self) -> Result<Vec<usize>> {
        row_major_shape(&self.dims)
    }

    /// Element count.
    /// # Errors
    /// Returns an invalid-input error when the count overflows.
    pub fn elements(&self) -> Result<u64> {
        elements(&self.dims)
    }

    /// Stored tensor type.
    #[must_use]
    pub const fn ty(&self) -> GgufType {
        self.ty
    }

    /// Stored bytes relative to the tensor-data section.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Stored payload bytes.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// One opened GGUF container.
#[derive(Debug, Clone)]
pub struct GgufFile {
    path: PathBuf,
    version: u32,
    metadata: Metadata,
    tensors: BTreeMap<String, TensorInfo>,
    data_start: u64,
    file_len: u64,
}

impl GgufFile {
    /// Open a container with the default resource limits.
    /// # Errors
    /// Returns an I/O, invalid-input, capacity or unsupported error for a malformed or unsupported file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, &GgufLimits::default())
    }

    /// Open a container with explicit resource limits.
    /// # Errors
    /// Returns an I/O, invalid-input, capacity or unsupported error for a malformed or unsupported file.
    pub fn open_with(path: impl AsRef<Path>, limits: &GgufLimits) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file =
            File::open(&path).map_err(|e| Error::invalid(format!("{}: {e}", path.display())))?;
        let file_len = file
            .metadata()
            .map_err(|e| Error::invalid(e.to_string()))?
            .len();
        let mut reader = Reader::new(file, limits.header_bytes);
        let mut magic = [0u8; MAGIC_BYTES];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::invalid("GGUF magic mismatch"));
        }
        let version = reader.u32()?;
        if !(VERSION_MIN..=VERSION_MAX).contains(&version) {
            return Err(Error::unsupported(format!("GGUF version {version}")));
        }
        let tensor_count = reader.u64()?;
        let metadata_count = reader.u64()?;
        if metadata_count > limits.metadata_entries {
            return Err(Error::new(
                ErrorCode::Capacity,
                "GGUF metadata entry count exceeds budget",
            ));
        }
        if tensor_count > limits.tensor_entries {
            return Err(Error::new(
                ErrorCode::Capacity,
                "GGUF tensor entry count exceeds budget",
            ));
        }
        // Version 2 and 3 share the same metadata and tensor-info encoding.
        let metadata = read_metadata(&mut reader, metadata_count, limits)?;
        let alignment = alignment(&metadata)?;
        let (tensors, intervals) = read_tensors(&mut reader, tensor_count, limits)?;
        let data_start = align_up(reader.position(), alignment)?;
        if data_start > file_len {
            return Err(Error::invalid("GGUF tensor data starts past end of file"));
        }
        validate_ranges(&intervals, data_start, file_len, alignment)?;
        Ok(Self {
            path,
            version,
            metadata,
            tensors,
            data_start,
            file_len,
        })
    }

    /// Container version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Parsed metadata.
    #[must_use]
    pub const fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Declared model architecture.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.architecture()
    }

    /// Tensor descriptors, sorted by name.
    #[must_use]
    pub const fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    /// One tensor descriptor.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Absolute file offset of the tensor-data section.
    #[must_use]
    pub const fn data_start(&self) -> u64 {
        self.data_start
    }

    /// File size at open time.
    #[must_use]
    pub const fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Read one stored tensor within `budget` bytes.
    /// # Errors
    /// Returns an invalid-input error for an unknown tensor, or a capacity error when the payload
    /// exceeds `budget`.
    pub fn read_tensor(&self, name: &str, budget: u64) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::invalid(format!("missing GGUF tensor {name}")))?;
        if info.byte_len > budget {
            return Err(Error::new(
                ErrorCode::Capacity,
                "GGUF tensor exceeds read budget",
            ));
        }
        let start = self
            .data_start
            .checked_add(info.offset)
            .ok_or_else(|| Error::invalid("GGUF tensor offset overflow"))?;
        let mut file = File::open(&self.path)
            .map_err(|e| Error::invalid(format!("{}: {e}", self.path.display())))?;
        file.seek(SeekFrom::Start(start))
            .map_err(|e| Error::invalid(e.to_string()))?;
        let mut bytes = vec![
            0u8;
            usize::try_from(info.byte_len)
                .map_err(|_| Error::invalid("GGUF tensor exceeds address space"))?
        ];
        file.read_exact(&mut bytes)
            .map_err(|e| Error::invalid(e.to_string()))?;
        Ok(bytes)
    }
}

/// Read the metadata map, rejecting duplicate keys.
fn read_metadata(reader: &mut Reader, count: u64, limits: &GgufLimits) -> Result<Metadata> {
    let mut map = BTreeMap::new();
    for _ in 0..count {
        let key = reader.string(limits.string_bytes)?;
        let value = reader.value(limits, 0)?;
        if map.insert(key.clone(), value).is_some() {
            return Err(Error::invalid(format!("duplicate GGUF metadata key {key}")));
        }
    }
    Ok(Metadata::from_map(map))
}

/// One stored range: offset relative to the tensor-data section, byte length and name.
type Interval = (u64, u64, String);

/// Read every tensor descriptor, rejecting duplicate names and unsupported types.
fn read_tensors(
    reader: &mut Reader,
    count: u64,
    limits: &GgufLimits,
) -> Result<(BTreeMap<String, TensorInfo>, Vec<Interval>)> {
    let mut tensors = BTreeMap::new();
    let mut intervals = Vec::new();
    for _ in 0..count {
        let name = reader.string(limits.string_bytes)?;
        if name.is_empty() {
            return Err(Error::invalid("empty GGUF tensor name"));
        }
        let rank = reader.u32()?;
        if rank == 0 || rank > limits.tensor_dims {
            return Err(Error::invalid(format!("GGUF tensor {name} rank {rank}")));
        }
        let rank = usize::try_from(rank).map_err(|e| Error::invalid(e.to_string()))?;
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            let dim = reader.u64()?;
            if dim == 0 {
                return Err(Error::invalid(format!(
                    "GGUF tensor {name} has a zero dimension"
                )));
            }
            dims.push(dim);
        }
        let ty = GgufType::from_id(reader.u32()?);
        let offset = reader.u64()?;
        let byte_len = ty.byte_len(elements(&dims)?)?;
        let info = TensorInfo {
            name: name.clone(),
            dims,
            ty,
            offset,
            byte_len,
        };
        if tensors.insert(name.clone(), info).is_some() {
            return Err(Error::invalid(format!("duplicate GGUF tensor {name}")));
        }
        intervals.push((offset, byte_len, name));
    }
    Ok((tensors, intervals))
}

/// Check alignment, bounds and overlap of every stored tensor range.
fn validate_ranges(
    intervals: &[Interval],
    data_start: u64,
    file_len: u64,
    alignment: u64,
) -> Result<()> {
    let mut sorted = intervals.to_vec();
    sorted.sort_unstable();
    let mut end = 0u64;
    for (offset, bytes, name) in &sorted {
        if !offset.is_multiple_of(alignment) {
            return Err(Error::invalid(format!(
                "tensor {name} offset is not aligned"
            )));
        }
        let start = data_start
            .checked_add(*offset)
            .ok_or_else(|| Error::invalid("GGUF tensor offset overflow"))?;
        let tensor_end = start
            .checked_add(*bytes)
            .ok_or_else(|| Error::invalid("GGUF tensor size overflow"))?;
        if tensor_end > file_len {
            return Err(Error::invalid(format!(
                "tensor {name} ends past end of file"
            )));
        }
        if *offset < end {
            return Err(Error::invalid("GGUF tensors overlap"));
        }
        end = *offset + *bytes;
    }
    Ok(())
}

/// Declared tensor-data alignment.
fn alignment(metadata: &Metadata) -> Result<u64> {
    let alignment = metadata
        .u64("general.alignment")?
        .unwrap_or(DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_power_of_two() || alignment > MAX_ALIGNMENT {
        return Err(Error::invalid("invalid GGUF alignment"));
    }
    Ok(alignment)
}

/// Round `value` up to the next multiple of `alignment`.
fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| Error::invalid("GGUF alignment overflow"))
}

/// A bounded little-endian reader over the header region.
struct Reader {
    file: File,
    read: u64,
    limit: u64,
}

impl Reader {
    const fn new(file: File, limit: u64) -> Self {
        Self {
            file,
            read: 0,
            limit,
        }
    }

    const fn position(&self) -> u64 {
        self.read
    }

    fn read_exact(&mut self, bytes: &mut [u8]) -> Result<()> {
        let length = bytes.len() as u64;
        if self
            .read
            .checked_add(length)
            .is_none_or(|end| end > self.limit)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "GGUF header exceeds budget",
            ));
        }
        self.file
            .read_exact(bytes)
            .map_err(|e| Error::invalid(e.to_string()))?;
        self.read += length;
        Ok(())
    }

    fn bytes(&mut self, length: u64) -> Result<Vec<u8>> {
        let length = usize::try_from(length)
            .map_err(|_| Error::new(ErrorCode::Capacity, "GGUF string exceeds address space"))?;
        let mut bytes = vec![0u8; length];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8> {
        let mut bytes = [0u8; U8_BYTES];
        self.read_exact(&mut bytes)?;
        Ok(bytes[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let mut bytes = [0u8; U16_BYTES];
        self.read_exact(&mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32> {
        let mut bytes = [0u8; U32_BYTES];
        self.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut bytes = [0u8; U64_BYTES];
        self.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()?.cast_signed())
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()?.cast_signed())
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()?.cast_signed())
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()?.cast_signed())
    }

    fn f32(&mut self) -> Result<f32> {
        let mut bytes = [0u8; F32_BYTES];
        self.read_exact(&mut bytes)?;
        Ok(f32::from_le_bytes(bytes))
    }

    fn f64(&mut self) -> Result<f64> {
        let mut bytes = [0u8; F64_BYTES];
        self.read_exact(&mut bytes)?;
        Ok(f64::from_le_bytes(bytes))
    }

    fn string(&mut self, limit: u64) -> Result<String> {
        let length = self.u64()?;
        if length > limit {
            return Err(Error::new(
                ErrorCode::Capacity,
                "GGUF string exceeds budget",
            ));
        }
        String::from_utf8(self.bytes(length)?).map_err(|e| Error::invalid(e.to_string()))
    }

    /// Read one metadata value. `depth` bounds nested arrays.
    fn value(&mut self, limits: &GgufLimits, depth: u32) -> Result<MetadataValue> {
        let kind = self.u32()?;
        self.typed_value(kind, limits, depth)
    }

    fn typed_value(&mut self, kind: u32, limits: &GgufLimits, depth: u32) -> Result<MetadataValue> {
        Ok(match kind {
            KV_U8 => MetadataValue::U8(self.u8()?),
            KV_I8 => MetadataValue::I8(self.i8()?),
            KV_U16 => MetadataValue::U16(self.u16()?),
            KV_I16 => MetadataValue::I16(self.i16()?),
            KV_U32 => MetadataValue::U32(self.u32()?),
            KV_I32 => MetadataValue::I32(self.i32()?),
            KV_F32 => MetadataValue::F32(self.f32()?),
            KV_BOOL => MetadataValue::Bool(match self.u8()? {
                0 => false,
                BOOL_TRUE => true,
                _ => return Err(Error::invalid("invalid GGUF boolean")),
            }),
            KV_STRING => MetadataValue::String(self.string(limits.string_bytes)?),
            KV_ARRAY => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(Error::invalid("GGUF metadata array nesting"));
                }
                let element_kind = self.u32()?;
                let count = self.u64()?;
                if count > limits.array_elements {
                    return Err(Error::new(
                        ErrorCode::Capacity,
                        "GGUF metadata array exceeds budget",
                    ));
                }
                let mut values = Vec::with_capacity(
                    usize::try_from(count.min(limits.array_elements)).unwrap_or(0),
                );
                for _ in 0..count {
                    values.push(self.typed_value(element_kind, limits, depth + 1)?);
                }
                MetadataValue::Array(values)
            }
            KV_U64 => MetadataValue::U64(self.u64()?),
            KV_I64 => MetadataValue::I64(self.i64()?),
            KV_F64 => MetadataValue::F64(self.f64()?),
            other => {
                return Err(Error::invalid(format!(
                    "unknown GGUF metadata type {other}"
                )));
            }
        })
    }
}
