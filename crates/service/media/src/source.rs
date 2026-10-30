//! Reusable asynchronous HTTP source with pooled connections.
use infer_core::{Error, ErrorCode, Result};
use reqwest::{Client, Response, redirect::Policy};
use std::time::Duration;

/// Match the previously supported redirect chain.
const MAX_REDIRECTS: usize = 50;
/// Overall timeout for one remote media source.
const HTTP_TIMEOUT_SECONDS: u64 = 20;

/// A reusable, connection-pooled HTTP(S) media source.
#[derive(Debug, Clone)]
pub struct HttpSource {
    client: Client,
}

impl HttpSource {
    /// Create a client using system trust roots and proxy configuration.
    ///
    /// # Errors
    /// Returns a backend error when TLS or the HTTP client cannot initialize.
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECONDS))
            .redirect(Policy::limited(MAX_REDIRECTS))
            .build()
            .map_err(|error| Error::new(ErrorCode::Backend, error.without_url().to_string()))?;
        Ok(Self { client })
    }

    /// Download one remote media source without blocking the calling executor.
    ///
    /// The returned bytes belong to the caller. Dropping the future cancels the download.
    /// A supplied limit bounds the decoded HTTP response bytes; `None` imposes no fixed byte cap.
    ///
    /// # Errors
    /// Returns an input error for network, status or protocol failures, and capacity errors for a
    /// configured limit or failed allocation. URLs are removed from transport error messages.
    pub async fn fetch(&self, url: &str, limit: Option<usize>) -> Result<Vec<u8>> {
        let mut response = self
            .client
            .get(url)
            .send()
            .await
            .and_then(Response::error_for_status)
            .map_err(download_error)?;
        if limit.is_some_and(|limit| {
            response
                .content_length()
                .is_some_and(|length| u64::try_from(limit).is_ok_and(|limit| length > limit))
        }) {
            return Err(Error::new(
                ErrorCode::Capacity,
                "media download exceeds the bounded limit",
            ));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(download_error)? {
            let total = bytes
                .len()
                .checked_add(chunk.len())
                .ok_or_else(|| Error::invalid("media download size overflow"))?;
            if limit.is_some_and(|limit| total > limit) {
                return Err(Error::new(
                    ErrorCode::Capacity,
                    "media download exceeds the bounded limit",
                ));
            }
            bytes.try_reserve(chunk.len()).map_err(|error| {
                Error::new(
                    ErrorCode::Capacity,
                    format!("media download allocation: {error}"),
                )
            })?;
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }
}

fn download_error(error: reqwest::Error) -> Error {
    Error::invalid(format!("media download failed; {}", error.without_url()))
}
