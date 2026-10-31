//! Persistence glue between the Responses routes and an installed [`infer_spi::ResponseStore`].
use super::super::execution::Prepared as GenerationPrepared;
use super::output::Snapshot;
use super::request::{PriorState, Spec};
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_spi::{
    IncompleteReason, RESPONSES_PAYLOAD_VERSION, ResponseError, ResponseId, ResponseStatus,
    ResponseStore, ResponseUsage, StoredResponse, WriteCondition,
};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::runtime::Handle;

/// One request's retained state handles. Cloneable so a background task can finish the store
/// write after the HTTP response has already returned.
#[derive(Clone)]
pub(super) struct StoreSink {
    store: Arc<dyn ResponseStore>,
    id: ResponseId,
    request_id: RequestId,
    model: String,
    created: u64,
    temperature: f32,
    top_p: f32,
    request_input: Vec<u8>,
    previous_response_id: Option<ResponseId>,
    spec: Spec,
}

impl StoreSink {
    /// Bind a store to one admitted generation.
    ///
    /// # Errors
    /// Returns an invariant or invalid-input error when the generated identity or a chained
    /// identifier cannot be represented in the store contract.
    pub(super) fn new(
        store: Arc<dyn ResponseStore>,
        prepared: &GenerationPrepared,
        spec: &Spec,
        request_input: Vec<u8>,
    ) -> Result<Self> {
        let id = match spec.response_id.as_deref() {
            Some(id) => ResponseId::new(id)?,
            None => ResponseId::new(format!("resp_{}", prepared.request.get()))?,
        };
        let previous_response_id = spec
            .previous_response_id
            .as_deref()
            .map(ResponseId::new)
            .transpose()?;
        Ok(Self {
            store,
            id,
            request_id: prepared.request,
            model: prepared.model_name.clone(),
            created: prepared.created,
            temperature: prepared.temperature,
            top_p: prepared.top_p,
            request_input,
            previous_response_id,
            spec: spec.clone(),
        })
    }

    /// Persist the initial in-progress snapshot and return it for a background response.
    ///
    /// # Errors
    /// Returns a capacity or backend error from the store.
    pub(super) async fn start(&self) -> Result<Value> {
        let value = self.initial_value();
        self.put(&value, WriteCondition::Always).await?;
        Ok(value)
    }

    /// Persist the terminal response unless it was cancelled while generating.
    ///
    /// # Errors
    /// Returns a capacity or backend error from the store.
    pub(super) async fn finish(&self, value: &Value) -> Result<()> {
        if let Err(error) = self
            .put(
                value,
                WriteCondition::UnlessStatus(ResponseStatus::Cancelled),
            )
            .await
        {
            let persist_error = Error::new(
                ErrorCode::Backend,
                format!("failed to persist a completed response: {error}"),
            );
            if self.fail(&persist_error).await.is_err() {
                let _ = self.store.delete(&self.id).await;
            }
            return Err(error);
        }
        // The conditional write is a no-op when another writer already recorded a terminal
        // state (cancel, or a disconnect cleanup). Report that to the stream so it does not
        // publish response.completed for a response another writer terminated.
        if self.store.get(&self.id).await?.is_some_and(|stored| {
            matches!(
                stored.status,
                ResponseStatus::Cancelled | ResponseStatus::Failed
            )
        }) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "response was terminated while completing",
            ));
        }
        Ok(())
    }

    /// Persist a failed response unless it was cancelled while generating.
    ///
    /// # Errors
    /// Returns a capacity or backend error from the store.
    pub(super) async fn fail(&self, error: &Error) -> Result<()> {
        let value = Snapshot {
            request: self.request_id,
            created: self.created,
            completed_at: None,
            model: &self.model,
            status: "failed",
            incomplete: None,
            error: Some(json!({
                "code": error_code(error.code),
                "message": error.message,
            })),
            output: &[],
            spec: &self.spec,
            temperature: self.temperature,
            top_p: self.top_p,
            usage: None,
        }
        .value();
        match self
            .put(
                &value,
                WriteCondition::UnlessStatus(ResponseStatus::Cancelled),
            )
            .await
        {
            Ok(()) => Ok(()),
            Err(error) => {
                let _ = self.store.delete(&self.id).await;
                Err(error)
            }
        }
    }

    fn initial_value(&self) -> Value {
        Snapshot {
            request: self.request_id,
            created: self.created,
            completed_at: None,
            model: &self.model,
            status: "in_progress",
            incomplete: None,
            error: None,
            output: &[],
            spec: &self.spec,
            temperature: self.temperature,
            top_p: self.top_p,
            usage: None,
        }
        .value()
    }

    async fn put(&self, value: &Value, condition: WriteCondition) -> Result<()> {
        let record = self.record(value)?;
        let limit = self.store.capabilities().max_record_bytes;
        let bytes = record
            .request_input
            .len()
            .checked_add(record.response.len())
            .ok_or_else(|| Error::invalid("stored response size overflow"))?;
        if bytes > limit {
            return Err(Error::new(
                ErrorCode::Capacity,
                "stored response exceeds the store record limit",
            ));
        }
        self.store.put(record, condition).await
    }

    fn record(&self, value: &Value) -> Result<StoredResponse> {
        let status = value
            .get("status")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::invariant("response object has no status"))?
            .parse::<ResponseStatus>()?;
        let response = serde_json::to_vec(value)
            .map_err(|error| Error::invalid(format!("response serialization: {error}")))?;
        let record = StoredResponse {
            id: self.id.clone(),
            request_id: self.request_id,
            model: self.model.clone(),
            created_at: self.created,
            completed_at: value.get("completed_at").and_then(Value::as_u64),
            status,
            incomplete: incomplete_reason(value)?,
            error: response_error(value)?,
            usage: response_usage(value)?,
            previous_response_id: self.previous_response_id.clone(),
            request_input: self.request_input.clone(),
            response,
            payload_version: RESPONSES_PAYLOAD_VERSION,
        };
        record.validate()?;
        Ok(record)
    }
}

/// Reconstruct the prior conversation for `previous_response_id`.
///
/// # Errors
/// Returns not-found for a missing response, conflict for a response that is still active, or
/// invalid-input for a model mismatch or payloads that do not decode.
pub(super) fn prior_state(stored: &StoredResponse, model: &str) -> Result<PriorState> {
    if !matches!(
        stored.status,
        ResponseStatus::Completed | ResponseStatus::Incomplete
    ) {
        return Err(Error::new(
            ErrorCode::Conflict,
            "previous response must have completed or ended incomplete",
        ));
    }
    if stored.model != model {
        return Err(Error::invalid(
            "previous_response_id belongs to a different model",
        ));
    }
    let input: Vec<Value> = serde_json::from_slice(&stored.request_input)
        .map_err(|error| Error::invariant(format!("stored input payload is invalid: {error}")))?;
    let response: Value = serde_json::from_slice(&stored.response).map_err(|error| {
        Error::invariant(format!("stored response payload is invalid: {error}"))
    })?;
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| Error::invariant("stored response has no output list"))?;
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
pub(super) fn retrieved(stored: &StoredResponse) -> Result<Value> {
    let mut value: Value = serde_json::from_slice(&stored.response).map_err(|error| {
        Error::invariant(format!("stored response payload is invalid: {error}"))
    })?;
    if !value.is_object() {
        return Err(Error::invariant("stored response is not an object"));
    }
    value["status"] = json!(stored.status.as_str());
    value["completed_at"] = json!(stored.completed_at);
    value["error"] = stored.error.as_ref().map_or(
        Value::Null,
        |error| json!({"code":error.code, "message":error.message}),
    );
    value["incomplete_details"] = stored
        .incomplete
        .map_or(Value::Null, |reason| json!({"reason":reason.as_str()}));
    value["usage"] = stored.usage.map_or(Value::Null, usage_json);
    Ok(value)
}

fn incomplete_reason(value: &Value) -> Result<Option<IncompleteReason>> {
    let Some(reason) = value
        .get("incomplete_details")
        .and_then(|details| details.get("reason"))
        .and_then(Value::as_str)
    else {
        return Ok(None);
    };
    Ok(Some(match reason {
        "max_output_tokens" => IncompleteReason::MaxOutputTokens,
        "max_messages" => IncompleteReason::MaxMessages,
        "content_filter" => IncompleteReason::ContentFilter,
        other => {
            return Err(Error::invariant(format!(
                "unknown incomplete reason {other:?}"
            )));
        }
    }))
}

fn response_error(value: &Value) -> Result<Option<ResponseError>> {
    let Some(error) = value.get("error").filter(|error| !error.is_null()) else {
        return Ok(None);
    };
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::invariant("response error has no code"))?;
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::invariant("response error has no message"))?;
    ResponseError::new(code, message).map(Some)
}

fn response_usage(value: &Value) -> Result<Option<ResponseUsage>> {
    let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) else {
        return Ok(None);
    };
    let field = |name: &str| -> Result<u64> {
        usage
            .get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::invariant(format!("response usage has no {name}")))
    };
    let details = |name: &str, field_name: &str| -> u64 {
        usage
            .get(name)
            .and_then(|details| details.get(field_name))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    Ok(Some(ResponseUsage {
        input_tokens: field("input_tokens")?,
        output_tokens: field("output_tokens")?,
        reasoning_tokens: details("output_tokens_details", "reasoning_tokens"),
        cached_tokens: details("input_tokens_details", "cached_tokens"),
        cache_write_tokens: details("input_tokens_details", "cache_write_tokens"),
    }))
}

pub(super) fn usage_json(usage: ResponseUsage) -> Value {
    json!({
        "input_tokens": usage.input_tokens,
        "input_tokens_details": {
            "cached_tokens": usage.cached_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
        },
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
        "total_tokens": usage.checked_total().unwrap_or(u64::MAX),
    })
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

/// Armed persistence guard for one response.
///
/// If the HTTP future or SSE stream is dropped before a terminal write, the guard spawns a
/// best-effort failure write. The request store never keeps an active record forever merely
/// because a client disconnected.
pub(super) struct StoreLease {
    sink: StoreSink,
    armed: bool,
}

impl StoreLease {
    pub(super) const fn new(sink: StoreSink) -> Self {
        Self { sink, armed: false }
    }

    /// Write the initial snapshot and arm the disconnect cleanup.
    ///
    /// # Errors
    /// Returns a capacity or backend error from the store.
    pub(super) async fn start(&mut self) -> Result<Value> {
        self.armed = true;
        match self.sink.start().await {
            Ok(value) => Ok(value),
            Err(error) => {
                self.armed = false;
                Err(error)
            }
        }
    }

    /// Write the terminal snapshot and disarm cleanup.
    ///
    /// # Errors
    /// Returns a capacity or backend error from the store.
    pub(super) async fn finish(&mut self, value: &Value) -> Result<()> {
        let result = self.sink.finish(value).await;
        self.armed = false;
        result
    }

    /// Write a failed snapshot and disarm cleanup.
    ///
    /// # Errors
    /// Returns a capacity or backend error from the store.
    pub(super) async fn fail(&mut self, error: &Error) -> Result<()> {
        let result = self.sink.fail(error).await;
        self.armed = false;
        result
    }
}

impl Drop for StoreLease {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let sink = self.sink.clone();
        if let Ok(handle) = Handle::try_current() {
            handle.spawn(async move {
                let error = Error::new(
                    ErrorCode::Backend,
                    "response was abandoned before completion",
                );
                if let Err(store_error) = sink.fail(&error).await {
                    tracing::error!(
                        target: "infer::openai",
                        error = %store_error,
                        "failed to retain an abandoned response"
                    );
                }
            });
        }
    }
}
