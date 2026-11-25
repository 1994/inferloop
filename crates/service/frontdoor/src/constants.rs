//! Named staging-budget and request-limit constants shared by the actor and HTTP adapters.

/// Estimated staging bytes reserved per token when preparing a request or delivering decoded output.
pub const TOKEN_STAGING_BYTES: usize = 8;

/// Estimated staging bytes reserved per generated output token.
pub const GENERATED_TOKEN_STAGING_BYTES: usize = 4;

/// Multiplier applied to raw request text bytes when reserving preparation staging bytes.
pub const TEXT_STAGING_EXPANSION: usize = 16;

/// Fixed preparation staging overhead added to every text request, in bytes.
pub const TEXT_STAGING_OVERHEAD_BYTES: usize = 4096;

/// Maximum accepted HTTP request body size, in bytes.
pub const MAX_REQUEST_BODY_BYTES: usize = 2 * 1024 * 1024;
