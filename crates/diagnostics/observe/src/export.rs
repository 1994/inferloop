use crate::constants::{MICROSECONDS_PER_SECOND, NANOSECONDS_PER_SECOND};
use crate::{
    DeviceTimingMetrics, LATENCY_BOUNDS_US, LatencyHistogram, ObservationStore, RuntimeMetrics,
};
use infer_core::event::EventKind;
use infer_core::{CancelSource, ErrorCode};
use infer_ir::{OpClass, TimedPhase};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ResourceGauges {
    pub backend: String,
    pub ready: bool,
    pub active_requests: usize,
    pub inflight_steps: usize,
    pub resource_waiters: usize,
    pub output_waiters: usize,
    pub output_batches: usize,
    pub logical_pages: usize,
    pub free_logical_pages: usize,
    pub active_kv_blocks: usize,
    pub cached_kv_blocks: usize,
    pub available_kv_blocks: usize,
    pub kv_pool_bytes: u64,
    pub event_ring_dropped: u64,
    pub collector_metadata_dropped: u64,
}

/// Prometheus labels have fixed event/backend cardinality; no request/token/prompt labels.
#[must_use]
pub fn prometheus(
    metrics: &RuntimeMetrics,
    gauges: &ResourceGauges,
    store: &ObservationStore,
) -> String {
    prometheus_retention(
        metrics,
        gauges,
        store.history_evicted,
        store.diagnostics_evicted,
    )
}
/// Export from immutable counters without borrowing the live collector.
#[must_use]
pub fn prometheus_retention(
    metrics: &RuntimeMetrics,
    gauges: &ResourceGauges,
    history_evicted: u64,
    diagnostics_evicted: u64,
) -> String {
    let backend = gauges
        .backend
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    let mut lines = vec![
        "# HELP infer_semantic_events_total Events emitted before bounded ring delivery.".into(),
        "# TYPE infer_semantic_events_total counter".into(),
    ];
    for kind in EventKind::ALL {
        lines.push(format!(
            "infer_semantic_events_total{{backend=\"{backend}\",kind=\"{}\"}} {}",
            kind.label(),
            metrics.count(kind)
        ));
    }
    lines.push("# TYPE infer_deferrals_total counter".into());
    for (index, reason) in infer_ir::DeferReason::LABELS.iter().enumerate() {
        lines.push(format!(
            "infer_deferrals_total{{backend=\"{backend}\",reason=\"{reason}\"}} {}",
            metrics.deferred_reasons[index]
        ));
    }
    lines.push("# TYPE infer_requests_cancelled_total counter".into());
    for source in CancelSource::ALL {
        lines.push(format!(
            "infer_requests_cancelled_total{{backend=\"{backend}\",cause=\"{}\"}} {}",
            source.as_str(),
            metrics.cancelled_by[source.index()]
        ));
    }
    lines.push("# TYPE infer_request_failures_total counter".into());
    for code in ErrorCode::ALL {
        lines.push(format!(
            "infer_request_failures_total{{backend=\"{backend}\",code=\"{}\"}} {}",
            code.as_str(),
            metrics.failed_by[code.index()]
        ));
    }
    for (name, value) in [
        (
            "infer_requests_successful_total",
            metrics.successful_requests,
        ),
        ("infer_requests_failed_total", metrics.failed_requests),
        (
            "infer_prefix_tokens_reused_total",
            metrics.prefix_tokens_reused,
        ),
        ("infer_event_ring_dropped_total", gauges.event_ring_dropped),
        (
            "infer_collector_metadata_dropped_total",
            gauges.collector_metadata_dropped,
        ),
        ("infer_event_history_evicted_total", history_evicted),
        ("infer_diagnostics_evicted_total", diagnostics_evicted),
    ] {
        lines.push(format!("# TYPE {name} counter"));
        lines.push(format!("{name}{{backend=\"{backend}\"}} {value}"));
    }
    gauge_lines(&mut lines, &backend, gauges);
    for (name, histogram) in [
        ("infer_request_e2e_seconds", &metrics.e2e),
        ("infer_request_ttft_seconds", &metrics.ttft),
        ("infer_token_tpot_seconds", &metrics.tpot),
        ("infer_cpu_execution_seconds", &metrics.cpu_execution),
        ("infer_cpu_ready_seconds", &metrics.cpu_stages[0]),
        ("infer_cpu_planning_seconds", &metrics.cpu_stages[1]),
        ("infer_cpu_dispatch_seconds", &metrics.cpu_stages[2]),
        ("infer_cpu_completion_seconds", &metrics.cpu_stages[3]),
        ("infer_cpu_owner_seconds", &metrics.cpu_stages[4]),
        ("infer_gpu_execution_seconds", &metrics.gpu_execution),
        ("infer_cpu_output_seconds", &metrics.cpu_output),
        (
            "infer_resource_preparation_seconds",
            &metrics.resource_preparation,
        ),
    ] {
        lines.push(format!("# TYPE {name} histogram"));
        histogram_lines(
            &mut lines,
            name,
            &format!("backend=\"{backend}\""),
            histogram,
        );
    }
    device_lines(&mut lines, &backend, &metrics.device);
    lines.push(String::new());
    lines.join("\n")
}

/// Device time by phase, operator class and implementation. Labels come from fixed enums and the
/// backend's declared variant list, so cardinality stays bounded; layers are JSON-only.
#[expect(
    clippy::cast_precision_loss,
    reason = "Prometheus duration counters are conventionally floating-point seconds"
)]
fn device_lines(lines: &mut Vec<String>, backend: &str, device: &DeviceTimingMetrics) {
    let name = "infer_device_graph_seconds";
    lines.push(format!("# TYPE {name} histogram"));
    for (phase, timing) in TimedPhase::ALL.iter().zip(&device.phases) {
        let labels = format!("backend=\"{backend}\",phase=\"{}\"", phase.label());
        histogram_lines(lines, name, &labels, &timing.graphs);
    }
    let name = "infer_device_node_seconds_total";
    lines.push(format!("# TYPE {name} counter"));
    for (phase, timing) in TimedPhase::ALL.iter().zip(&device.phases) {
        for (op, total) in OpClass::ALL.iter().zip(&timing.ops) {
            if total.segments == 0 {
                continue;
            }
            lines.push(format!(
                "{name}{{backend=\"{backend}\",phase=\"{}\",op=\"{}\"}} {}",
                phase.label(),
                op.label(),
                total.device_ns as f64 / NANOSECONDS_PER_SECOND
            ));
        }
    }
    let name = "infer_device_variant_seconds_total";
    lines.push(format!("# TYPE {name} counter"));
    for (variant, total) in device.variant_names.iter().zip(&device.variants) {
        if total.segments == 0 {
            continue;
        }
        let variant = variant
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n");
        lines.push(format!(
            "{name}{{backend=\"{backend}\",variant=\"{variant}\"}} {}",
            total.device_ns as f64 / NANOSECONDS_PER_SECOND
        ));
    }
}

fn gauge_lines(lines: &mut Vec<String>, backend: &str, gauges: &ResourceGauges) {
    for (name, value) in [
        ("infer_ready", u64::from(gauges.ready)),
        (
            "infer_active_requests",
            u64::try_from(gauges.active_requests).unwrap_or(u64::MAX),
        ),
        (
            "infer_inflight_steps",
            u64::try_from(gauges.inflight_steps).unwrap_or(u64::MAX),
        ),
        (
            "infer_resource_waiters",
            u64::try_from(gauges.resource_waiters).unwrap_or(u64::MAX),
        ),
        (
            "infer_output_waiters",
            u64::try_from(gauges.output_waiters).unwrap_or(u64::MAX),
        ),
        (
            "infer_output_batches",
            u64::try_from(gauges.output_batches).unwrap_or(u64::MAX),
        ),
        (
            "infer_logical_pages",
            u64::try_from(gauges.logical_pages).unwrap_or(u64::MAX),
        ),
        (
            "infer_free_logical_pages",
            u64::try_from(gauges.free_logical_pages).unwrap_or(u64::MAX),
        ),
        (
            "infer_active_kv_blocks",
            u64::try_from(gauges.active_kv_blocks).unwrap_or(u64::MAX),
        ),
        (
            "infer_cached_kv_blocks",
            u64::try_from(gauges.cached_kv_blocks).unwrap_or(u64::MAX),
        ),
        (
            "infer_available_kv_blocks",
            u64::try_from(gauges.available_kv_blocks).unwrap_or(u64::MAX),
        ),
        ("infer_kv_pool_bytes", gauges.kv_pool_bytes),
    ] {
        lines.push(format!("# TYPE {name} gauge"));
        lines.push(format!("{name}{{backend=\"{backend}\"}} {value}"));
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Prometheus histogram boundaries and cumulative duration sums are conventionally represented as floating-point seconds"
)]
/// One histogram's series under `labels`; the caller writes the `# TYPE` line once per name.
fn histogram_lines(
    lines: &mut Vec<String>,
    name: &str,
    labels: &str,
    histogram: &LatencyHistogram,
) {
    let mut cumulative = 0_u64;
    for (index, bound) in LATENCY_BOUNDS_US.iter().enumerate() {
        cumulative = cumulative.saturating_add(histogram.buckets[index]);
        lines.push(format!(
            "{name}_bucket{{{labels},le=\"{}\"}} {cumulative}",
            *bound as f64 / MICROSECONDS_PER_SECOND
        ));
    }
    lines.push(format!(
        "{name}_bucket{{{labels},le=\"+Inf\"}} {}",
        histogram.count
    ));
    lines.push(format!(
        "{name}_sum{{{labels}}} {}",
        histogram.sum_us as f64 / MICROSECONDS_PER_SECOND
    ));
    lines.push(format!("{name}_count{{{labels}}} {}", histogram.count));
}
