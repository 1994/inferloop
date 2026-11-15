//! Typed GGUF metadata values.
use infer_core::{Error, Result};
use std::collections::BTreeMap;

/// One GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Self>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    /// The integer value as `u64`, when it is an unsigned or non-negative signed integer.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(value) => Some(u64::from(*value)),
            Self::U16(value) => Some(u64::from(*value)),
            Self::U32(value) => Some(u64::from(*value)),
            Self::U64(value) => Some(*value),
            Self::I8(value) => u64::try_from(*value).ok(),
            Self::I16(value) => u64::try_from(*value).ok(),
            Self::I32(value) => u64::try_from(*value).ok(),
            Self::I64(value) => u64::try_from(*value).ok(),
            _ => None,
        }
    }

    /// The value as `u32`, when it fits.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        self.as_u64().and_then(|value| u32::try_from(value).ok())
    }

    /// The UTF-8 string value.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// The boolean value.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// The array elements.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(values) => Some(values),
            _ => None,
        }
    }
}

/// The metadata key/value map of one GGUF file.
#[derive(Debug, Clone, Default)]
pub struct Metadata(BTreeMap<String, MetadataValue>);

impl Metadata {
    pub(crate) const fn from_map(map: BTreeMap<String, MetadataValue>) -> Self {
        Self(map)
    }

    /// One raw metadata value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.0.get(key)
    }

    /// All keys, sorted.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    /// The declared model architecture (`general.architecture`).
    pub fn architecture(&self) -> Option<&str> {
        self.0
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
    }

    /// One unsigned integer, either signed or unsigned in the file.
    /// # Errors
    /// Returns an invalid-input error for a missing key or a non-integer value.
    pub fn require_u64(&self, key: &str) -> Result<u64> {
        self.get(key)
            .and_then(MetadataValue::as_u64)
            .ok_or_else(|| {
                Error::invalid(format!("GGUF metadata {key} is missing or not an integer"))
            })
    }

    /// One optional unsigned integer.
    /// # Errors
    /// Returns an invalid-input error when the key exists with a non-integer value.
    pub fn u64(&self, key: &str) -> Result<Option<u64>> {
        self.get(key).map_or(Ok(None), |value| {
            value
                .as_u64()
                .map(Some)
                .ok_or_else(|| Error::invalid(format!("GGUF metadata {key} is not an integer")))
        })
    }

    /// One required string value.
    /// # Errors
    /// Returns an invalid-input error for a missing or non-string value.
    pub fn require_str(&self, key: &str) -> Result<&str> {
        self.get(key)
            .and_then(MetadataValue::as_str)
            .ok_or_else(|| {
                Error::invalid(format!("GGUF metadata {key} is missing or not a string"))
            })
    }

    /// One string array value.
    /// # Errors
    /// Returns an invalid-input error when the value is not an array of strings.
    pub fn require_strings(&self, key: &str) -> Result<Vec<&str>> {
        let values = self
            .get(key)
            .and_then(MetadataValue::as_array)
            .ok_or_else(|| {
                Error::invalid(format!("GGUF metadata {key} is missing or not an array"))
            })?;
        values
            .iter()
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    Error::invalid(format!("GGUF metadata {key} has a non-string element"))
                })
            })
            .collect()
    }
}
