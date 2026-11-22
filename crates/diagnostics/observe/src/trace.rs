use crate::ObservedEvent;
use crate::constants::HEX_RADIX;
use crate::constants::NANOS_PER_MICROSECOND;
use crate::constants::SPAN_ID_HEX_LEN;
use crate::constants::TRACE_ID_HEX_LEN;
use crate::constants::TRACEPARENT_FIELDS;
use crate::constants::TRACEPARENT_FLAGS_HEX_LEN;
use infer_core::digest::lower_hex;
use infer_core::{Error, Result, event::EventKind, event::ObjectKind};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceContext {
    pub trace_id: String,
    pub parent_span_id: String,
    pub sampled: bool,
}
impl TraceContext {
    #[must_use]
    pub fn valid(&self) -> bool {
        valid_hex(&self.trace_id, TRACE_ID_HEX_LEN)
            && valid_hex(&self.parent_span_id, SPAN_ID_HEX_LEN)
    }
    /// # Errors
    /// Returns invalid input for malformed W3C version-00 traceparent identifiers.
    pub fn parse(parent: &str) -> Result<Self> {
        let parts: Vec<_> = parent.split('-').collect();
        if parts.len() != TRACEPARENT_FIELDS
            || parts[0] != "00"
            || !valid_hex(parts[1], TRACE_ID_HEX_LEN)
            || !valid_hex(parts[2], SPAN_ID_HEX_LEN)
            || parts[3].len() != TRACEPARENT_FLAGS_HEX_LEN
            || !parts[3]
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::invalid("invalid version-00 traceparent"));
        }
        let flags =
            u8::from_str_radix(parts[3], HEX_RADIX).map_err(|e| Error::invalid(e.to_string()))?;
        Ok(Self {
            trace_id: parts[1].into(),
            parent_span_id: parts[2].into(),
            sampled: flags & 1 == 1,
        })
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct TraceCoverage {
    pub complete_requests: usize,
    pub retained_incomplete_requests: usize,
    pub observed_request_identities: usize,
}
#[must_use]
pub fn coverage(events: &[ObservedEvent]) -> TraceCoverage {
    let mut requests = BTreeMap::<u64, (bool, bool)>::new();
    for record in events {
        if let Some(request) = record.request {
            let entry = requests.entry(request.get()).or_default();
            entry.0 |= record.event.kind == EventKind::Accepted;
            entry.1 |= record.event.kind == EventKind::Finished;
        }
    }
    let complete = requests
        .values()
        .filter(|(accepted, finished)| *accepted && *finished)
        .count();
    TraceCoverage {
        complete_requests: complete,
        retained_incomplete_requests: requests.len() - complete,
        observed_request_identities: requests.len(),
    }
}
fn valid_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        && value.bytes().any(|b| b != b'0')
}

/// Semantic timestamps are the caller's runtime clock, separate from GPU/CPU clock domains.
#[must_use]
pub fn timeline(events: &[ObservedEvent], ring_dropped: u64, history_evicted: u64) -> Value {
    let records:Vec<_>=events.iter().map(|record|json!({"name":record.event.kind.label(),"cat":"runtime_semantic","ph":"i","s":"t","ts":record.event.timestamp_us,"pid":1,"tid":1,"args":record})).collect();
    json!({"traceEvents":records,"displayTimeUnit":"us","clock":"runtime_logical","ring_dropped":ring_dropped,"history_evicted":history_evicted})
}

struct RequestSpan {
    start: u64,
    end: Option<u64>,
    failed: bool,
    events: Vec<Value>,
}

/// Export complete retained request spans in OTLP JSON; partial histories remain explicit.
#[must_use]
pub fn otlp(
    events: &[ObservedEvent],
    parents: &BTreeMap<u64, TraceContext>,
    namespace: u64,
    unix_origin_us: u64,
) -> Value {
    let mut spans = BTreeMap::<u64, RequestSpan>::new();
    for record in events {
        if record.event.object_kind != ObjectKind::Request {
            continue;
        }
        let event = record.event;
        if event.kind == EventKind::Accepted {
            spans.insert(
                event.object_id,
                RequestSpan {
                    start: event.timestamp_us,
                    end: None,
                    failed: false,
                    events: Vec::new(),
                },
            );
        }
        if let Some(span) = spans.get_mut(&event.object_id) {
            span.events.push(json!({"name":event.kind.label(),"timeUnixNano":unix_origin_us.saturating_add(event.timestamp_us).saturating_mul(NANOS_PER_MICROSECOND).to_string(),"attributes":[{"key":"infer.object_id","value":{"stringValue":event.object_id.to_string()}},{"key":"infer.correlation_id","value":{"stringValue":event.correlation_id.to_string()}}]}));
            if event.kind == EventKind::Finished {
                span.end = Some(event.timestamp_us);
                span.failed = event.arg0 > 2;
            }
        }
    }
    let spans:Vec<_>=spans.into_iter().filter_map(|(id,span)| {
        let end=span.end?;
        let hash=lower_hex(&Sha256::digest(format!("{namespace}:{id}")));
        let parent=parents.get(&id);
        Some(json!({"traceId":parent.map_or_else(||hash[..TRACE_ID_HEX_LEN].to_string(),|p|p.trace_id.clone()),"spanId":&hash[TRACE_ID_HEX_LEN..TRACE_ID_HEX_LEN + SPAN_ID_HEX_LEN],"parentSpanId":parent.map_or("",|p|p.parent_span_id.as_str()),"flags":u32::from(parent.is_none_or(|p|p.sampled)),"name":"infer.request","kind":2,"startTimeUnixNano":unix_origin_us.saturating_add(span.start).saturating_mul(NANOS_PER_MICROSECOND).to_string(),"endTimeUnixNano":unix_origin_us.saturating_add(end).saturating_mul(NANOS_PER_MICROSECOND).to_string(),"attributes":[{"key":"infer.request_id","value":{"stringValue":id.to_string()}},{"key":"infer.clock","value":{"stringValue":"runtime_logical"}}],"events":span.events,"status":{"code":if span.failed {2}else{1}}}))
    }).collect();
    json!({"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"infer"}}]},"scopeSpans":[{"scope":{"name":"infer.runtime","version":"0.1.0"},"spans":spans}]}]})
}
