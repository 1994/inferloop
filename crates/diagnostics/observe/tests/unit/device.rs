use super::*;
use crate::{ObservationStore, RuntimeMetrics, export};
use infer_ir::{TimedGraph, TimedSegment};

fn segment(op: OpClass, layer: Option<u16>, variant: Option<u16>, device_ns: u64) -> TimedSegment {
    TimedSegment {
        graph: TimedGraph::Target,
        node: Some(0),
        op,
        layer,
        variant,
        device_ns,
    }
}

fn sample(phase: TimedPhase, total_ns: u64, segments: Vec<TimedSegment>) -> DeviceTimingSample {
    DeviceTimingSample {
        phase,
        rows: 16,
        total_ns,
        segments,
    }
}

#[test]
fn samples_aggregate_by_phase_operator_layer_and_variant() {
    let mut device = DeviceTimingMetrics::with_variants(vec!["fa4".into(), "nvfp4".into()]);
    device.record(&sample(
        TimedPhase::Verify,
        40_000_000,
        vec![
            segment(OpClass::Attention, Some(3), Some(0), 2_000_000),
            segment(OpClass::Projection, Some(3), Some(1), 30_000_000),
            segment(OpClass::Runtime, None, None, 1_000_000),
        ],
    ));
    device.record(&sample(TimedPhase::Verify, 38_000_000, Vec::new()));
    device.record(&sample(TimedPhase::Decode, 9_000_000, Vec::new()));

    assert_eq!(device.samples, 3);
    let verify = device.phase(TimedPhase::Verify).unwrap();
    assert_eq!(verify.graphs.count, 2);
    assert_eq!(verify.graphs.sum_us, 78_000);
    assert_eq!(verify.rows, 32);
    assert_eq!(
        verify.ops[OpClass::Projection.index()],
        DeviceTotal {
            device_ns: 30_000_000,
            segments: 1
        }
    );
    assert_eq!(verify.ops[OpClass::Runtime.index()].device_ns, 1_000_000);
    assert_eq!(verify.layers[3].device_ns, 32_000_000);
    assert_eq!(device.variants[0].device_ns, 2_000_000);
    assert_eq!(device.variants[1].device_ns, 30_000_000);
    assert_eq!(device.phase(TimedPhase::Decode).unwrap().graphs.count, 1);
}

#[test]
fn out_of_range_layers_and_variants_fold_into_the_last_slot() {
    let mut device = DeviceTimingMetrics::default();
    let layers = device.phases[0].layers.len();
    let variants = device.variants.len();
    device.record(&sample(
        TimedPhase::Prefill,
        1_000,
        vec![segment(OpClass::Norm, Some(u16::MAX), Some(u16::MAX), 7)],
    ));
    let prefill = device.phase(TimedPhase::Prefill).unwrap();
    assert_eq!(prefill.layers.len(), layers);
    assert_eq!(prefill.layers[MAX_TIMED_LAYERS - 1].device_ns, 7);
    assert_eq!(device.variants.len(), variants);
    assert_eq!(device.variants[MAX_TIMED_VARIANTS - 1].device_ns, 7);
}

#[test]
fn prometheus_exports_device_time_with_bounded_labels() {
    let mut metrics = RuntimeMetrics {
        device: DeviceTimingMetrics::with_variants(vec!["fa4 \"bf16\"".into()]),
        ..RuntimeMetrics::default()
    };
    metrics.device.record(&sample(
        TimedPhase::Verify,
        2_000_000,
        vec![segment(OpClass::Attention, Some(1), Some(0), 500_000_000)],
    ));
    let gauges = export::ResourceGauges {
        backend: "cuda".into(),
        ready: true,
        active_requests: 0,
        inflight_steps: 0,
        resource_waiters: 0,
        output_waiters: 0,
        output_batches: 0,
        logical_pages: 0,
        free_logical_pages: 0,
        active_kv_blocks: 0,
        cached_kv_blocks: 0,
        available_kv_blocks: 0,
        kv_pool_bytes: 0,
        event_ring_dropped: 0,
        collector_metadata_dropped: 0,
    };
    let output = export::prometheus(&metrics, &gauges, &ObservationStore::new(2).unwrap());
    assert!(output.contains("# TYPE infer_device_graph_seconds histogram\n"));
    assert!(
        output.contains("infer_device_graph_seconds_count{backend=\"cuda\",phase=\"verify\"} 1\n")
    );
    assert!(output.contains(
        "infer_device_node_seconds_total{backend=\"cuda\",phase=\"verify\",op=\"attention\"} 0.5\n"
    ));
    assert!(output.contains(
        "infer_device_variant_seconds_total{backend=\"cuda\",variant=\"fa4 \\\"bf16\\\"\"} 0.5\n"
    ));
    // Unmeasured operator classes add no series, and layers never become labels.
    assert!(!output.contains("op=\"norm\""));
    assert!(!output.contains("layer="));
    assert_eq!(
        output.matches("# TYPE infer_device_graph_seconds ").count(),
        1
    );
}
