//! Decoding of inline media sources.
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use base64::read::DecoderReader;
use infer_core::{Error, ErrorCode, Result};
use std::io::Read;

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
    let payload = data_uri_payload(url)?;
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

/// Read decoded bytes from a base64 data URI without allocating a second full input buffer.
///
/// Reading is CPU-bound and must run on a bounded worker. Invalid base64 is reported by the
/// reader as an I/O error; the URI must remain alive until reading finishes.
///
/// # Errors
/// Rejects malformed or non-base64 data URI headers.
pub fn data_uri_reader(url: &str) -> Result<impl Read + Send + '_> {
    let payload = data_uri_payload(url)?;
    Ok(DecoderReader::new(payload.as_bytes(), &STANDARD))
}

fn data_uri_payload(url: &str) -> Result<&str> {
    let (header, payload) = url
        .split_once(',')
        .ok_or_else(|| Error::invalid("malformed data URI"))?;
    if !header.starts_with("data:") {
        return Err(Error::invalid("malformed data URI"));
    }
    if !header.ends_with(";base64") {
        return Err(Error::unsupported("only base64 data URIs are supported"));
    }
    Ok(payload)
}
