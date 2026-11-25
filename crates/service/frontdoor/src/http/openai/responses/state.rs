//! Wire-side persistence glue: the Responses body as an [`infer_agentic::ResponseDocument`].
//!
//! Lifecycles, conditional writes and the disconnect guard live in `infer-agentic`; this module
//! only maps the typed `OpenAI` response object to store metadata and back.
use super::super::execution::Prepared as GenerationPrepared;
use super::super::request::OpaqueJson;
use super::output;
use super::output::{
    IncompleteDetails, ResponseBody, ResponseErrorBody, ResponseUsageBody, Snapshot,
};
use super::request::{PriorState, Spec};
use infer_agentic::{DocumentMetadata, ResponseDocument, Retention};
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_spi::{
    IncompleteReason, ResponseError, ResponseId, ResponseStatus, ResponseUsage, StoredResponse,
};
use std::sync::Arc;

impl ResponseDocument for ResponseBody {
    fn metadata(&self) -> Result<DocumentMetadata> {
        Ok(DocumentMetadata {
            status: self.status.parse::<ResponseStatus>()?,
            completed_at: self.completed_at,
            incomplete: self
                .incomplete_details
                .as_ref()
                .map(|details| incomplete_reason(&details.reason))
                .transpose()?,
            error: self
                .error
                .as_ref()
                .map(|error| ResponseError::new(&error.code, &error.message))
                .transpose()?,
            usage: self.usage.as_ref().map(response_usage),
        })
    }

    fn encode(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|error| Error::invalid(format!("response serialization: {error}")))
    }
}

/// The parts of one admitted response that every lifecycle snapshot repeats.
#[derive(Clone)]
struct Envelope {
    request: RequestId,
    created: u64,
    model: String,
    temperature: f32,
    top_p: f32,
    spec: Spec,
}

impl Envelope {
    fn new(prepared: &GenerationPrepared, spec: &Spec) -> Self {
        Self {
            request: prepared.request,
            created: prepared.created,
            model: prepared.model_name.clone(),
            temperature: prepared.temperature,
            top_p: prepared.top_p,
            spec: spec.clone(),
        }
    }

    fn body(&self, status: &str, error: Option<ResponseErrorBody>) -> ResponseBody {
        Snapshot {
            request: self.request,
            created: self.created,
            completed_at: None,
            model: &self.model,
            status,
            incomplete: None,
            error,
            output: &[],
            spec: &self.spec,
            temperature: self.temperature,
            top_p: self.top_p,
            usage: None,
        }
        .value()
    }

    fn failed(&self, error: &Error) -> ResponseBody {
        self.body(
            "failed",
            Some(ResponseErrorBody {
                code: error_code(error.code).to_string(),
                message: error.message.clone(),
            }),
        )
    }
}

/// The in-progress snapshot written when an admitted response is first retained.
pub(super) fn initial(prepared: &GenerationPrepared, spec: &Spec) -> ResponseBody {
    Envelope::new(prepared, spec).body("in_progress", None)
}

/// Bind one admitted generation to its store identity.
///
/// # Errors
/// Returns an invalid-input error when the generated identity or a chained identifier cannot
/// be represented in the store contract.
pub(super) fn retention(
    prepared: &GenerationPrepared,
    spec: &Spec,
    request_input: Vec<u8>,
) -> Result<Retention<ResponseBody>> {
    let id = match spec.response_id.as_deref() {
        Some(id) => ResponseId::new(id)?,
        None => ResponseId::new(format!("resp_{}", prepared.request.get()))?,
    };
    let previous = spec
        .previous_response_id
        .as_deref()
        .map(ResponseId::new)
        .transpose()?;
    let envelope = Envelope::new(prepared, spec);
    Ok(Retention {
        id,
        request: prepared.request,
        model: prepared.model_name.clone(),
        created: prepared.created,
        previous,
        request_input,
        failed: Arc::new(move |error| envelope.failed(error)),
    })
}

/// Reconstruct the prior conversation of a validated `previous_response_id` record.
///
/// # Errors
/// Returns an invariant error when the stored payloads do not decode.
pub(super) fn prior_state(stored: &StoredResponse) -> Result<PriorState> {
    let input: Vec<OpaqueJson> = serde_json::from_slice(&stored.request_input)
        .map_err(|error| Error::invariant(format!("stored input payload is invalid: {error}")))?;
    let response: ResponseBody = serde_json::from_slice(&stored.response).map_err(|error| {
        Error::invariant(format!("stored response payload is invalid: {error}"))
    })?;
    let output = response
        .output()
        .iter()
        .map(|item| {
            serde_json::to_value(item)
                .map(OpaqueJson)
                .map_err(|error| Error::invariant(format!("stored output is invalid: {error}")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PriorState {
        input,
        output,
        model: stored.model.clone(),
    })
}

/// Decode a stored response and overlay the metadata that may be newer than the body.
///
/// A cancel transition intentionally does not rewrite the opaque body; the overlay keeps the
/// returned status, error, incomplete reason and usage authoritative.
///
/// # Errors
/// Returns an invariant error if the stored body is not a JSON object.
pub(super) fn retrieved(stored: &StoredResponse) -> Result<ResponseBody> {
    let mut value: ResponseBody = serde_json::from_slice(&stored.response).map_err(|error| {
        Error::invariant(format!("stored response payload is invalid: {error}"))
    })?;
    value.status = stored.status.as_str().to_string();
    value.completed_at = stored.completed_at;
    value.error = stored.error.as_ref().map(|error| ResponseErrorBody {
        code: error.code.clone(),
        message: error.message.clone(),
    });
    value.incomplete_details = stored.incomplete.map(|reason| IncompleteDetails {
        reason: reason.as_str().to_string(),
    });
    value.usage = stored.usage.map(response_usage_body);
    Ok(value)
}

fn incomplete_reason(reason: &str) -> Result<IncompleteReason> {
    Ok(match reason {
        "max_output_tokens" => IncompleteReason::MaxOutputTokens,
        "max_messages" => IncompleteReason::MaxMessages,
        "content_filter" => IncompleteReason::ContentFilter,
        other => {
            return Err(Error::invariant(format!(
                "unknown incomplete reason {other:?}"
            )));
        }
    })
}

fn response_usage(usage: &ResponseUsageBody) -> ResponseUsage {
    ResponseUsage {
        input_tokens: u64::try_from(usage.input_tokens).unwrap_or(u64::MAX),
        output_tokens: u64::try_from(usage.output_tokens).unwrap_or(u64::MAX),
        reasoning_tokens: u64::try_from(usage.output_tokens_details.reasoning_tokens)
            .unwrap_or(u64::MAX),
        cached_tokens: u64::try_from(usage.input_tokens_details.cached_tokens).unwrap_or(u64::MAX),
        cache_write_tokens: u64::try_from(usage.input_tokens_details.cache_write_tokens)
            .unwrap_or(u64::MAX),
    }
}

fn response_usage_body(usage: ResponseUsage) -> ResponseUsageBody {
    ResponseUsageBody {
        input_tokens: usize::try_from(usage.input_tokens).unwrap_or(usize::MAX),
        input_tokens_details: output::InputTokensDetails {
            cached_tokens: usize::try_from(usage.cached_tokens).unwrap_or(usize::MAX),
            cache_write_tokens: usize::try_from(usage.cache_write_tokens).unwrap_or(usize::MAX),
        },
        output_tokens: usize::try_from(usage.output_tokens).unwrap_or(usize::MAX),
        output_tokens_details: output::OutputTokensDetails {
            reasoning_tokens: usize::try_from(usage.reasoning_tokens).unwrap_or(usize::MAX),
        },
        total_tokens: usage
            .checked_total()
            .and_then(|total| usize::try_from(total).ok())
            .unwrap_or(usize::MAX),
    }
}

/// Storage-safe spelling of an error code for a failed response object.
pub(super) const fn error_code(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::InvalidInput => "invalid_request",
        ErrorCode::Unsupported => "unsupported_feature",
        ErrorCode::NotFound => "response_not_found",
        ErrorCode::Conflict => "request_conflict",
        ErrorCode::Capacity => "capacity_exceeded",
        ErrorCode::Backend => "backend_error",
        ErrorCode::Invariant => "internal_error",
    }
}
