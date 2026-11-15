use infer_ir::{
    DeviceTimingConfig, DeviceTimingMode, DeviceTimingSample, OpClass, TensorNode, TensorOp,
    TimedGraph, TimedInterval, TimedPhase, TimedSegment,
};

#[test]
fn sampling_reads_one_of_every_n_measured_executions() {
    let every_four = DeviceTimingConfig {
        mode: DeviceTimingMode::Graph,
        sample_every: 4,
    };
    let sampled: Vec<u64> = (0..12).filter(|&n| every_four.samples(n)).collect();
    assert_eq!(sampled, [0, 4, 8]);

    let zero_means_every = DeviceTimingConfig {
        mode: DeviceTimingMode::Node,
        sample_every: 0,
    };
    assert!((0..5).all(|n| zero_means_every.samples(n)));

    let off = DeviceTimingConfig {
        mode: DeviceTimingMode::Off,
        sample_every: 1,
    };
    assert!((0..5).all(|n| !off.samples(n)));
}

#[test]
fn the_default_measures_graphs_and_samples() {
    let config = DeviceTimingConfig::default();
    assert_eq!(config.mode, DeviceTimingMode::Graph);
    assert!(config.sample_every > 1);
    let parsed: DeviceTimingConfig = serde_json::from_str(r#"{"mode":"node"}"#).unwrap();
    assert_eq!(parsed.mode, DeviceTimingMode::Node);
    assert_eq!(parsed.sample_every, config.sample_every);
}

#[test]
fn a_node_segment_takes_its_class_and_layer_from_the_graph() {
    let node = TensorNode {
        id: infer_core::OpId::new(7).unwrap(),
        layer: Some(12),
        op: TensorOp::Attention {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            window: None,
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        states: Vec::new(),
    };
    let segment = TimedSegment::of_node(
        TimedPhase::Verify,
        TimedGraph::Target,
        (41, &node),
        (2_000, 5_000),
    );
    assert_eq!(segment.phase, TimedPhase::Verify);
    assert_eq!(segment.node, Some(41));
    assert_eq!(segment.op, OpClass::Attention);
    assert_eq!(segment.layer, Some(12));
    assert_eq!(segment.variant, None);
    assert_eq!((segment.start_ns, segment.duration_ns), (2_000, 5_000));
}

#[test]
fn a_sample_spans_its_intervals_and_round_trips() {
    let sample = DeviceTimingSample {
        rows: 4,
        intervals: vec![
            TimedInterval {
                phase: TimedPhase::Draft,
                start_ns: 0,
                duration_ns: 3_000,
            },
            TimedInterval {
                phase: TimedPhase::Verify,
                start_ns: 3_000,
                duration_ns: 9_000,
            },
        ],
        segments: Vec::new(),
        host: Vec::new(),
    };
    assert_eq!(sample.total_ns(), 12_000);
    assert_eq!(DeviceTimingSample::default().total_ns(), 0);
    let json = serde_json::to_string(&sample).unwrap();
    assert_eq!(
        serde_json::from_str::<DeviceTimingSample>(&json).unwrap(),
        sample
    );
}

#[test]
fn classes_index_their_own_table_positions() {
    for (position, class) in OpClass::ALL.iter().enumerate() {
        assert_eq!(class.index(), position, "{}", class.label());
    }
    assert_eq!(OpClass::of(&TensorOp::Linear), OpClass::Projection);
    assert_eq!(
        OpClass::of(&TensorOp::ExpertLinear {
            experts: 8,
            active: 2,
            input: 64,
            output: 64,
            routed_input: true,
        }),
        OpClass::Projection
    );
}
