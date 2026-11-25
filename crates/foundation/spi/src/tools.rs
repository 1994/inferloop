//! Engine-owned tool execution contracts.
//!
//! A deployment installs a [`ToolExecutor`] next to the response store. The tools it declares are
//! offered to the model alongside the request's own function tools; when the model calls one,
//! the serving layer runs it in-process and feeds the output back without a client round trip.
//!
//! Like [`crate::ResponseStore`], the executor is object-safe and returns boxed futures, so it can
//! bridge to any runtime without the engine depending on one. Dropping a returned future cancels
//! the call; an implementation must not leave a partial side effect behind or spawn unbounded
//! work of its own.
use infer_core::{Error, Result};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// Version of the tool execution contract.
pub const TOOLS_SPI_VERSION: u32 = 1;

/// A boxed tool call. Failures are reported as [`ToolOutput::Failed`], not as an error, because
/// they are fed back to the model.
pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>>;

/// Name of one callable tool.
///
/// Validated with the `OpenAI` function-name rule (1 to 64 ASCII letters, digits, underscores
/// or dashes) so a name can be offered to the model and matched against its calls verbatim.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ToolName(Box<str>);

impl ToolName {
    /// Largest accepted name, in bytes.
    pub const MAX_BYTES: usize = 64;

    /// Validate and wrap a tool name.
    ///
    /// # Errors
    /// Returns an invalid-input error for an empty, oversized, or non-token name.
    pub fn new(value: &str) -> Result<Self> {
        if value.is_empty() || value.len() > Self::MAX_BYTES {
            return Err(Error::invalid(format!(
                "tool name must be 1..{} bytes",
                Self::MAX_BYTES
            )));
        }
        if !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(Error::invalid(format!(
                "tool name {value:?} must contain only ASCII letters, digits, underscores or dashes"
            )));
        }
        Ok(Self(value.into()))
    }

    /// The name as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ToolName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Whether running a tool can change state outside the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEffect {
    /// Reads only; the engine may run it as soon as the model closes the call.
    ReadOnly,
    /// Writes, sends, or spends something; it needs an approval flow before it runs.
    SideEffect,
}

/// One tool an executor offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    name: ToolName,
    description: Box<str>,
    parameters: Box<str>,
    effect: ToolEffect,
    timeout: Duration,
    max_output_bytes: usize,
}

impl ToolSpec {
    /// Describe one tool. `parameters` is the JSON Schema of its arguments, as JSON text the
    /// serving layer passes to the model unchanged. `timeout` bounds one call and
    /// `max_output_bytes` bounds its output; the serving layer may apply tighter limits.
    ///
    /// # Errors
    /// Returns an invalid-input error for empty parameters, a zero timeout, or a zero output limit.
    pub fn new(
        name: ToolName,
        description: &str,
        parameters: &str,
        effect: ToolEffect,
        timeout: Duration,
        max_output_bytes: usize,
    ) -> Result<Self> {
        if parameters.trim().is_empty() {
            return Err(Error::invalid(format!(
                "tool {name} needs a parameters schema"
            )));
        }
        if timeout.is_zero() || max_output_bytes == 0 {
            return Err(Error::invalid(format!(
                "tool {name} needs a non-zero timeout and output limit"
            )));
        }
        Ok(Self {
            name,
            description: description.into(),
            parameters: parameters.into(),
            effect,
            timeout,
            max_output_bytes,
        })
    }

    #[must_use]
    pub const fn name(&self) -> &ToolName {
        &self.name
    }

    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The JSON Schema of the arguments, as JSON text.
    #[must_use]
    pub fn parameters(&self) -> &str {
        &self.parameters
    }

    #[must_use]
    pub const fn effect(&self) -> ToolEffect {
        self.effect
    }

    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    #[must_use]
    pub const fn max_output_bytes(&self) -> usize {
        self.max_output_bytes
    }
}

/// One call the model made to an executor tool.
#[derive(Debug, Clone, Copy)]
pub struct ToolInvocation<'a> {
    pub name: &'a ToolName,
    /// The model's arguments, as JSON text.
    pub arguments: &'a str,
}

/// What one tool call returns to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolOutput {
    /// The tool's text output.
    Text(String),
    /// Why the call failed, stated for the model; never presented as success.
    Failed(String),
}

/// Runs the tools a deployment owns.
pub trait ToolExecutor: Send + Sync {
    /// The contract version the implementation was built against.
    fn spi_version(&self) -> u32 {
        TOOLS_SPI_VERSION
    }

    /// The tools this executor runs. Names must be unique and stable for the executor's lifetime.
    fn tools(&self) -> &[ToolSpec];

    /// Run one call of a tool listed by [`Self::tools`].
    fn call<'a>(&'a self, invocation: ToolInvocation<'a>) -> ToolFuture<'a>;
}
