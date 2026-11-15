//! Shared unit conversions and W3C traceparent identifier lengths.

/// Nanoseconds in one microsecond, for OTLP nanosecond timestamps.
pub const NANOS_PER_MICROSECOND: u64 = 1000;
/// Microseconds in one second, for Prometheus second-valued samples.
pub const MICROSECONDS_PER_SECOND: f64 = 1e6;
/// Dash-separated fields of a W3C version-00 traceparent header.
pub const TRACEPARENT_FIELDS: usize = 4;
/// Hex characters of a W3C trace identifier.
pub const TRACE_ID_HEX_LEN: usize = 32;
/// Hex characters of a W3C span identifier.
pub const SPAN_ID_HEX_LEN: usize = 16;
/// Hex characters of the traceparent trace-flags field.
pub const TRACEPARENT_FLAGS_HEX_LEN: usize = 2;
/// Radix of the traceparent version and trace-flags hex fields.
pub const HEX_RADIX: u32 = 16;
