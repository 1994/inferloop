use super::*;
use crate::decoder::lower;
use infer_ir::{BackboneKind, FeedForward, Head, Mixer, Modality, ModelIr, PositionSpec};

fn model(layers: usize) -> ModelIr {
    ModelIr {
        id: infer_core::ModelId::ONE,
        backbone: BackboneKind::Decoder,
        vocab_size: 32,
        hidden_size: 8,
        max_sequence: 64,
        mixers: vec![
            Mixer::Attention {
                query_heads: 2,
                kv_heads: 1,
                head_dim: 4,
                sliding_window: None,
                output_gate: false,
                qk_norm: false,
            };
            layers
        ],
        feed_forward: FeedForward::Dense { intermediate: 16 },
        position: PositionSpec {
            rope_theta: 10_000.0,
            rotary_fraction: 1.0,
            multimodal_sections: vec![],
            interleaved: false,
        },
        norm_epsilon: 1e-6,
        norm_weight_offset: 0.0,
        heads: vec![Head::LanguageModel],
        modalities: vec![Modality::Text],
        state: vec![],
        tied_embeddings: false,
        media_features: 0,
    }
}

fn last_of_layer(graph: &DataflowGraph, layer: usize) -> usize {
    graph
        .nodes
        .iter()
        .rposition(|node| node.layer == Some(layer) && node.op == TensorOp::Add)
        .unwrap()
}

#[test]
fn each_tap_projects_the_residual_its_layer_returns() {
    let mut graph = lower(&model(4)).unwrap();
    let streams: Vec<_> = [1, 3]
        .iter()
        .map(|&layer| graph.nodes[last_of_layer(&graph, layer)].outputs[0])
        .collect();
    attach_tap_projection(&mut graph, &[1, 3], 8).unwrap();
    graph.validate().unwrap();
    let projections: Vec<_> = graph
        .nodes
        .iter()
        .filter(|node| {
            node.op == TensorOp::Linear
                && graph.tensors.iter().any(|tensor| {
                    tensor.id == node.inputs[1]
                        && matches!(&tensor.storage, TensorStorage::Weight { slot } if slot.starts_with("draft.fc."))
                })
        })
        .collect();
    assert_eq!(projections.len(), 2);
    for (projection, stream) in projections.iter().zip(&streams) {
        assert_eq!(projection.inputs[0], *stream);
    }
    // The second tap's sum is the graph result, and it adds the first tap's projection.
    let features = graph.features.unwrap();
    let total = graph
        .nodes
        .iter()
        .find(|node| node.outputs == [features])
        .unwrap();
    assert_eq!(total.op, TensorOp::Add);
    assert_eq!(
        total.inputs,
        vec![projections[0].outputs[0], projections[1].outputs[0]]
    );
}

#[test]
fn features_stay_alive_until_the_graph_returns() {
    let mut graph = lower(&model(3)).unwrap();
    attach_tap_projection(&mut graph, &[0], 8).unwrap();
    let features = graph.features.unwrap();
    let lifetime = graph
        .lifetimes
        .iter()
        .find(|lifetime| lifetime.tensor == features)
        .unwrap();
    assert_eq!(lifetime.last_node, graph.nodes.len());
}

#[test]
fn invalid_taps_are_rejected() {
    let base = lower(&model(3)).unwrap();
    for taps in [vec![], vec![2, 1], vec![1, 1], vec![3]] {
        let mut graph = base.clone();
        assert!(
            attach_tap_projection(&mut graph, &taps, 8).is_err(),
            "{taps:?}"
        );
    }
    let mut graph = base.clone();
    assert!(attach_tap_projection(&mut graph, &[1], 16).is_err());
    let mut graph = base;
    attach_tap_projection(&mut graph, &[1], 8).unwrap();
    assert!(attach_tap_projection(&mut graph, &[2], 8).is_err());
}

#[test]
fn media_features_follow_their_leading_layers_in_order() {
    let mut with_media = model(4);
    with_media.media_features = 2;
    let graph = lower(&with_media).unwrap();
    let features: Vec<_> = graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(position, node)| match node.op {
            TensorOp::MediaFeature { index } => Some((position, index, node.layer)),
            _ => None,
        })
        .collect();
    assert_eq!(graph.media_features(), 2);
    for &(position, index, layer) in &features {
        assert_eq!(layer, Some(index));
        // Each feature is added to the stream its layer's last residual returns.
        assert_eq!(position, last_of_layer(&graph, index) + 1);
        assert_eq!(
            graph.nodes[position].inputs,
            graph.nodes[position - 1].outputs
        );
    }
    assert_eq!(features.len(), 2);
    assert_eq!(lower(&model(4)).unwrap().media_features(), 0);
}

#[test]
fn media_features_beyond_the_layers_or_out_of_order_are_rejected() {
    let mut too_many = model(2);
    too_many.media_features = 3;
    assert!(lower(&too_many).is_err());
    let mut with_media = model(3);
    with_media.media_features = 2;
    let mut graph = lower(&with_media).unwrap();
    for node in &mut graph.nodes {
        if let TensorOp::MediaFeature { index } = &mut node.op {
            *index = 1 - *index;
        }
    }
    assert!(graph.validate().is_err());
}

#[test]
fn a_tap_reads_the_stream_after_its_media_feature() {
    let mut with_media = model(4);
    with_media.media_features = 2;
    let mut graph = lower(&with_media).unwrap();
    let returned = graph
        .nodes
        .iter()
        .rfind(|node| node.layer == Some(1))
        .map(|node| node.outputs[0])
        .unwrap();
    attach_tap_projection(&mut graph, &[1, 3], 8).unwrap();
    graph.validate().unwrap();
    assert!(
        graph
            .nodes
            .iter()
            .any(|node| node.op == TensorOp::Linear && node.inputs[0] == returned)
    );
}
