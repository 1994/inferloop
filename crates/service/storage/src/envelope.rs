//! The record value of one stored response.
//!
//! `[format u16][metadata length u32][metadata JSON][request_input length u32][request_input]
//! [response]`, little-endian. The two payloads are already serialized by the frontdoor and are
//! kept as they are, without a second encoding; the metadata is small and stays readable when
//! someone inspects a database by hand.
use infer_core::{Error, RequestId, Result};
use infer_spi::{
    IncompleteReason, ResponseError, ResponseId, ResponseStatus, ResponseUsage, StoredResponse,
};
use serde::{Deserialize, Serialize};

/// Layout of the value; a different one is refused rather than guessed at.
const FORMAT: u16 = 1;

#[derive(Serialize, Deserialize)]
struct Metadata {
    id: ResponseId,
    request_id: RequestId,
    model: String,
    created_at: u64,
    completed_at: Option<u64>,
    status: ResponseStatus,
    incomplete: Option<IncompleteReason>,
    error: Option<ResponseError>,
    usage: Option<ResponseUsage>,
    previous_response_id: Option<ResponseId>,
    payload_version: u32,
}

fn length(bytes: usize) -> Result<[u8; size_of::<u32>()]> {
    u32::try_from(bytes)
        .map(u32::to_le_bytes)
        .map_err(|_| Error::invalid("stored response section exceeds 4 GiB"))
}

/// Encode a response as one record value.
///
/// # Errors
/// Returns an invalid-input error for a section too large to frame.
pub fn encode(response: &StoredResponse) -> Result<Vec<u8>> {
    let metadata = serde_json::to_vec(&Metadata {
        id: response.id.clone(),
        request_id: response.request_id,
        model: response.model.clone(),
        created_at: response.created_at,
        completed_at: response.completed_at,
        status: response.status,
        incomplete: response.incomplete,
        error: response.error.clone(),
        usage: response.usage,
        previous_response_id: response.previous_response_id.clone(),
        payload_version: response.payload_version,
    })
    .map_err(|error| Error::invariant(format!("response metadata encoding failed: {error}")))?;
    let mut value = Vec::with_capacity(
        size_of::<u16>()
            + 2 * size_of::<u32>()
            + metadata.len()
            + response.request_input.len()
            + response.response.len(),
    );
    value.extend_from_slice(&FORMAT.to_le_bytes());
    value.extend_from_slice(&length(metadata.len())?);
    value.extend_from_slice(&metadata);
    value.extend_from_slice(&length(response.request_input.len())?);
    value.extend_from_slice(&response.request_input);
    value.extend_from_slice(&response.response);
    Ok(value)
}

/// Split `count` bytes off the front of `rest`.
fn take<'a>(rest: &mut &'a [u8], count: usize) -> Result<&'a [u8]> {
    if rest.len() < count {
        return Err(Error::invariant("stored response record is truncated"));
    }
    let (head, tail) = rest.split_at(count);
    *rest = tail;
    Ok(head)
}

fn take_length(rest: &mut &[u8]) -> Result<usize> {
    let bytes: [u8; size_of::<u32>()] = take(rest, size_of::<u32>())?
        .try_into()
        .map_err(|_| Error::invariant("stored response length is truncated"))?;
    usize::try_from(u32::from_le_bytes(bytes))
        .map_err(|_| Error::invariant("stored response length overflow"))
}

/// Decode a record value written by [`encode`] and validate it.
///
/// # Errors
/// Returns an unsupported error for another format, and an invariant error for a truncated or
/// inconsistent record.
pub fn decode(value: &[u8]) -> Result<StoredResponse> {
    let mut rest = value;
    let format: [u8; size_of::<u16>()] = take(&mut rest, size_of::<u16>())?
        .try_into()
        .map_err(|_| Error::invariant("stored response format is truncated"))?;
    let format = u16::from_le_bytes(format);
    if format != FORMAT {
        return Err(Error::unsupported(format!(
            "stored response record format {format} is not {FORMAT}"
        )));
    }
    let metadata_length = take_length(&mut rest)?;
    let metadata: Metadata = serde_json::from_slice(take(&mut rest, metadata_length)?)
        .map_err(|error| Error::invariant(format!("stored response metadata: {error}")))?;
    let input_length = take_length(&mut rest)?;
    let request_input = take(&mut rest, input_length)?.to_vec();
    let response = StoredResponse {
        id: metadata.id,
        request_id: metadata.request_id,
        model: metadata.model,
        created_at: metadata.created_at,
        completed_at: metadata.completed_at,
        status: metadata.status,
        incomplete: metadata.incomplete,
        error: metadata.error,
        usage: metadata.usage,
        previous_response_id: metadata.previous_response_id,
        request_input,
        response: rest.to_vec(),
        payload_version: metadata.payload_version,
    };
    response.validate()?;
    Ok(response)
}
