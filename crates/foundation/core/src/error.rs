//! Typed error codes and results shared by engine layers.
use serde::{Deserialize, Serialize};
use std::result::Result as StdResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    InvalidInput,
    Unsupported,
    Capacity,
    NotFound,
    Conflict,
    Invariant,
    Backend,
}

impl ErrorCode {
    /// Every code, in the order metrics count them.
    pub const ALL: &[Self] = &[
        Self::InvalidInput,
        Self::Unsupported,
        Self::Capacity,
        Self::NotFound,
        Self::Conflict,
        Self::Invariant,
        Self::Backend,
    ];
    /// Stable field and label value for structured logs and metrics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidInput => "invalid_input",
            Self::Unsupported => "unsupported",
            Self::Capacity => "capacity",
            Self::NotFound => "not_found",
            Self::Conflict => "conflict",
            Self::Invariant => "invariant",
            Self::Backend => "backend",
        }
    }
    /// Position in [`Self::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[error("{code:?}: {message}")]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
}

impl Error {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidInput, message)
    }
    pub fn invariant(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Invariant, message)
    }
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Unsupported, message)
    }
}
pub type Result<T> = StdResult<T, Error>;
