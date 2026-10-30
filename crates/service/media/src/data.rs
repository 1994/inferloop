//! Decoding of inline media sources.
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use infer_core::{Error, ErrorCode, Result};

/// Encoded characters per three decoded bytes, before padding.
const BASE64_EXPANSION_NUMERATOR: usize = 4;
const BASE64_EXPANSION_DENOMINATOR: usize = 3;
/// Allowance for an inline source's MIME header when a caller sets a byte limit.
const DATA_URI_HEADER_BYTES: usize = 4096;

/// Decode a base64 data URI into owned media bytes.
///
/// This operation is CPU-bound and callers should run it on their bounded worker pool, not an
/// asynchronous executor. A supplied limit bounds the decoded bytes; `None` adds no byte cap.
///
/// # Errors
/// Rejects malformed or non-base64 data URIs and inputs exceeding the caller's limit.
pub fn decode_data_uri(url: &str, limit: Option<usize>) -> Result<Vec<u8>> {
    let exceeds = limit.is_some_and(|limit| {
        let encoded_limit = (limit.saturating_mul(BASE64_EXPANSION_NUMERATOR)
            / BASE64_EXPANSION_DENOMINATOR)
            .saturating_add(DATA_URI_HEADER_BYTES);
        url.len() > encoded_limit
    });
    if exceeds {
        return Err(Error::new(
            ErrorCode::Capacity,
            "media data URI text exceeds the bounded limit",
        ));
    }
    let (header, payload) = url
        .split_once(',')
        .ok_or_else(|| Error::invalid("malformed data URI"))?;
    if !header.starts_with("data:") {
        return Err(Error::invalid("malformed data URI"));
    }
    if !header.ends_with(";base64") {
        return Err(Error::unsupported("only base64 data URIs are supported"));
    }
    let bytes = STANDARD
        .decode(payload)
        .map_err(|error| Error::invalid(format!("data URI base64: {error}")))?;
    if limit.is_some_and(|limit| bytes.len() > limit) {
        return Err(Error::new(
            ErrorCode::Capacity,
            "media data URI exceeds the bounded limit",
        ));
    }
    Ok(bytes)
}
