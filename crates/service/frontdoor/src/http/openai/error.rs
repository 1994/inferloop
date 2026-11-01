use axum::extract::rejection::JsonRejection;
use axum::{Json, http::StatusCode, response::IntoResponse, response::Response};
use infer_core::{Error, ErrorCode};
use serde::Serialize;

pub(super) struct ApiError {
    status: StatusCode,
    kind: &'static str,
    code: &'static str,
    message: String,
}

#[derive(Serialize)]
pub(super) struct ErrorEnvelope<'a> {
    pub(super) error: ErrorBody<'a>,
}

#[derive(Serialize)]
pub(super) struct ErrorBody<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
    param: Option<&'a str>,
    code: &'a str,
}

impl ApiError {
    pub(super) fn envelope(&self) -> ErrorEnvelope<'_> {
        ErrorEnvelope {
            error: ErrorBody {
                message: &self.message,
                kind: self.kind,
                param: None,
                code: self.code,
            },
        }
    }
}

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        let (status, kind, code) = match error.code {
            ErrorCode::InvalidInput => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_request",
            ),
            ErrorCode::Unsupported => (
                StatusCode::NOT_IMPLEMENTED,
                "invalid_request_error",
                "unsupported_feature",
            ),
            ErrorCode::NotFound => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "model_not_found",
            ),
            ErrorCode::Conflict => (
                StatusCode::CONFLICT,
                "invalid_request_error",
                "request_conflict",
            ),
            ErrorCode::Capacity => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "capacity_exceeded",
            ),
            ErrorCode::Backend => (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "backend_error",
            ),
            ErrorCode::Invariant => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "internal_error",
            ),
        };
        Self {
            status,
            kind,
            code,
            message: error.message,
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        Self {
            status: if rejection.status() == StatusCode::UNPROCESSABLE_ENTITY {
                StatusCode::BAD_REQUEST
            } else {
                rejection.status()
            },
            kind: "invalid_request_error",
            code: "invalid_request",
            message: rejection.body_text(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.envelope())).into_response()
    }
}
