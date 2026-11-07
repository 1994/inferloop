//! Reusable asynchronous HTTP source with pooled connections.
use infer_core::{Error, ErrorCode, Result};
use reqwest::{Client, Response, redirect::Policy};
use std::time::Duration;

/// Match the previously supported redirect chain.
const MAX_REDIRECTS: usize = 50;
/// Timeout for connection setup, stalled reads, and a complete buffered image download.
const HTTP_TIMEOUT_SECONDS: u64 = 20;

/// A reusable, connection-pooled HTTP(S) media source.
#[derive(Debug, Clone)]
pub struct HttpSource {
    client: Client,
}

/// An in-progress HTTP response whose chunks are consumed asynchronously.
///
/// Dropping this value cancels the response. A caller-supplied byte limit applies across chunks.
pub struct HttpDownload {
    response: Response,
    received: usize,
    limit: Option<usize>,
    failed: bool,
}

impl HttpSource {
    /// Create a client using system trust roots and proxy configuration.
    ///
    /// # Errors
    /// Returns a backend error when TLS or the HTTP client cannot initialize.
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(HTTP_TIMEOUT_SECONDS))
            .read_timeout(Duration::from_secs(HTTP_TIMEOUT_SECONDS))
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
            .open(url, limit, Some(Duration::from_secs(HTTP_TIMEOUT_SECONDS)))
            .await?;
        let mut bytes = Vec::new();
        while let Some(chunk) = response.next_chunk().await? {
            let chunk = chunk.as_ref();
            bytes.try_reserve(chunk.len()).map_err(|error| {
                Error::new(
                    ErrorCode::Capacity,
                    format!("media download allocation: {error}"),
                )
            })?;
            bytes.extend_from_slice(chunk);
        }
        Ok(bytes)
    }

    /// Open a remote source without buffering its complete response.
    ///
    /// Connection and stalled-read timeouts apply, but an actively progressing stream has no
    /// overall duration limit. The caller controls when to read the next chunk.
    ///
    /// # Errors
    /// Returns an input error for network, status or protocol failures, or a capacity error when
    /// the advertised response size exceeds the optional limit.
    pub async fn stream(&self, url: &str, limit: Option<usize>) -> Result<HttpDownload> {
        self.open(url, limit, None).await
    }

    async fn open(
        &self,
        url: &str,
        limit: Option<usize>,
        deadline: Option<Duration>,
    ) -> Result<HttpDownload> {
        let request = self.client.get(url);
        let request = if let Some(deadline) = deadline {
            request.timeout(deadline)
        } else {
            request
        };
        let response = request
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
        Ok(HttpDownload {
            response,
            received: 0,
            limit,
            failed: false,
        })
    }
}

impl HttpDownload {
    /// Read the next response chunk, or `None` after the response completes.
    ///
    /// The returned bytes remain owned by the caller. Errors terminate the stream.
    ///
    /// # Errors
    /// Returns an input error for a failed read or a capacity error when the optional limit is
    /// exceeded. Errors remain terminal on subsequent calls.
    pub async fn next_chunk(&mut self) -> Result<Option<impl AsRef<[u8]> + Send + 'static>> {
        if self.failed {
            return Err(Error::invalid("media download already failed"));
        }
        let chunk = match self.response.chunk().await {
            Ok(chunk) => chunk,
            Err(error) => {
                self.failed = true;
                return Err(download_error(error));
            }
        };
        if let Some(chunk) = &chunk {
            let Some(total) = self.received.checked_add(chunk.len()) else {
                self.failed = true;
                return Err(Error::invalid("media download size overflow"));
            };
            if self.limit.is_some_and(|limit| total > limit) {
                self.failed = true;
                return Err(Error::new(
                    ErrorCode::Capacity,
                    "media download exceeds the bounded limit",
                ));
            }
            self.received = total;
        }
        Ok(chunk)
    }
}

fn download_error(error: reqwest::Error) -> Error {
    Error::invalid(format!("media download failed; {}", error.without_url()))
}
