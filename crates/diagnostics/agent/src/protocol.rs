//! JSON-RPC envelopes carry no engine types; command parameters are decoded separately.
use crate::constants::JSONRPC_INVALID_PARAMS;
use crate::constants::JSONRPC_INVALID_REQUEST;
use crate::constants::JSONRPC_SERVER_ERROR;
use infer_core::{Error, ErrorCode};
use serde::Serialize;
use serde_json::{Value, json};

pub const PROTOCOL_VERSION: &str = "1.0";

pub struct RpcRequest {
    pub id: Option<Value>,
    pub method: String,
    pub params: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    #[must_use]
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }
}

impl From<Error> for RpcError {
    fn from(error: Error) -> Self {
        Self {
            code: if error.code == ErrorCode::InvalidInput {
                JSONRPC_INVALID_PARAMS
            } else {
                JSONRPC_SERVER_ERROR
            },
            message: error.message,
            data: Some(json!({"code": error.code})),
        }
    }
}

impl RpcRequest {
    /// # Errors
    /// Returns Invalid Request for malformed envelopes; params remain command input.
    pub fn parse(value: Value) -> Result<Self, RpcError> {
        let Value::Object(mut object) = value else {
            return Err(RpcError::new(
                JSONRPC_INVALID_REQUEST,
                "invalid JSON-RPC request",
            ));
        };
        let version = object.remove("jsonrpc");
        let method = object.remove("method");
        let id = object.remove("id");
        if version.as_ref().and_then(Value::as_str) != Some("2.0")
            || !method.as_ref().is_some_and(Value::is_string)
            || id
                .as_ref()
                .is_some_and(|id| !id.is_null() && !id.is_string() && !id.is_number())
        {
            return Err(RpcError::new(
                JSONRPC_INVALID_REQUEST,
                "invalid JSON-RPC request",
            ));
        }
        let Some(Value::String(method)) = method else {
            return Err(RpcError::new(
                JSONRPC_INVALID_REQUEST,
                "invalid JSON-RPC method",
            ));
        };
        Ok(Self {
            id,
            method,
            params: object.remove("params").unwrap_or_else(|| json!({})),
        })
    }
}

#[must_use]
pub fn error_response(id: Value, error: &RpcError) -> Value {
    let mut response = serde_json::Map::new();
    response.insert("jsonrpc".into(), Value::String("2.0".into()));
    response.insert("id".into(), id);
    response.insert("error".into(), json!(error));
    Value::Object(response)
}

#[must_use]
pub fn response(id: Value, outcome: Result<Value, RpcError>) -> Value {
    match outcome {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err(error) => error_response(id, &error),
    }
}
