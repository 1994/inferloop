use super::*;
use infer_core::ModelId;
use infer_models::{TensorDtype, WeightEncoding, WeightSource, default_registry};

const HIDDEN: usize = 16;
const THETA: f64 = 10_000.0;
const EPSILON: f64 = 1e-6;
const POSITION: f64 = 7.0;

/// A Llama-shaped configuration with four 96-wide query heads over two key heads.
fn graph(extra: &str) -> DataflowGraph {
    let bytes = format!(
        r#"{{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":384,
        "vocab_size":256,"num_hidden_layers":1,"num_attention_heads":4,
        "num_key_value_heads":2,"intermediate_size":512,"max_position_embeddings":4096{extra}}}"#
    )
    .into_bytes();
    let provider = default_registry().resolve(&bytes).unwrap();
    let model = provider.import(ModelId::ONE, &bytes).unwrap().model;
    provider.graph(&model).unwrap()
}

fn sources(graph: &DataflowGraph) -> BTreeMap<String, DeviceWeight> {
    graph
        .tensors
        .iter()
        .filter_map(|spec| match &spec.storage {
            TensorStorage::Weight { slot } => Some((
                slot.clone(),
                DeviceWeight {
                    shape: spec.shape.clone(),
                    encoding: WeightEncoding::Float,
                    data: WeightSource {
                        name: slot.clone(),
                        shard: String::new(),
                        dtype: TensorDtype::BF16,
                        shape: spec.shape.clone(),
                        bytes: 0,
                        view: None,
                    },
                    scale: None,
                    global_scale: None,
                    input_global_scale: None,
                    weight_block: None,
                    fp8_token_input: false,
                },
            )),
            _ => None,
        })
        .collect()
}

#[test]
fn a_narrow_head_is_stored_in_the_next_power_of_two() {
    let mut graph = graph("");
    let mut weights = sources(&graph);
    let padding = pad_heads(&mut graph, &mut weights, false, false).unwrap();
    let attention = graph
        .nodes
        .iter()
        .find_map(|node| match node.op {
            TensorOp::Attention { head_dim, .. } => Some(head_dim),
            _ => None,
        })
        .unwrap();
    assert_eq!(attention, 128);
    // q, k, v rows and the output projection's columns.
    assert_eq!(padding.weights.len(), 4);
    // Full rotary over 96 lanes pads each 48-lane half to 64.
    assert!(padding.rotary.values().all(|&real| real == 96));
    assert_eq!(
        weights["layers.0.self_attn.q_proj.weight"].shape,
        [4 * 128, 384]
    );
    assert_eq!(
        weights["layers.0.self_attn.o_proj.weight"].shape,
        [384, 4 * 128]
    );
    crate::resident::require_kernel_geometry(&graph).unwrap();
}

#[test]
fn power_of_two_heads_and_drafts_are_left_alone_or_refused() {
    let mut graph = graph(r#","head_dim":128"#);
    let mut weights = sources(&graph);
    assert!(
        pad_heads(&mut graph, &mut weights, true, false)
            .unwrap()
            .weights
            .is_empty()
    );
    let mut narrow = self::graph("");
    let mut weights = sources(&narrow);
    let error = pad_heads(&mut narrow, &mut weights, true, false).unwrap_err();
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
}

/// Deterministic values in [-1, 1).
fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn project(weight: &[f32], rows: usize, x: &[f32]) -> Vec<f64> {
    (0..rows)
        .map(|row| {
            weight[row * x.len()..][..x.len()]
                .iter()
                .zip(x)
                .map(|(&w, &v)| f64::from(w) * f64::from(v))
                .sum()
        })
        .collect()
}

fn norm(x: &mut [f64], gain: &[f32], epsilon: f64, offset: f64) {
    let mean = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inverse = (mean + epsilon).sqrt().recip();
    for (v, &g) in x.iter_mut().zip(gain) {
        *v *= inverse * (f64::from(g) + offset);
    }
}

fn rotate(x: &mut [f64], frequencies: &[f64]) {
    let half = x.len() / 2;
    for (lane, &frequency) in frequencies.iter().enumerate().take(half) {
        let (sin, cos) = (POSITION * frequency).sin_cos();
        let (a, b) = (x[lane], x[lane + half]);
        x[lane] = a.mul_add(cos, -b * sin);
        x[lane + half] = a.mul_add(sin, b * cos);
    }
}

/// One query head against one key head, as a reference decoder computes it.
fn score(
    (query, key): (&[f32], &[f32]),
    gains: Option<(&[f32], &[f32], f64, f64)>,
    width: usize,
    frequencies: &[f64],
) -> f64 {
    let x = values(HIDDEN, 1);
    let mut q = project(query, width, &x);
    let mut k = project(key, width, &x);
    if let Some((query_gain, key_gain, epsilon, offset)) = gains {
        norm(&mut q, query_gain, epsilon, offset);
        norm(&mut k, key_gain, epsilon, offset);
    }
    rotate(&mut q, frequencies);
    rotate(&mut k, frequencies);
    q.iter().zip(&k).map(|(a, b)| a * b).sum::<f64>() / (width as f64).sqrt()
}

fn real_frequencies(real: usize) -> Vec<f64> {
    (0..real / 2)
        .map(|lane| THETA.powf(-((2 * lane) as f64) / real as f64))
        .collect()
}

/// The padded head computes the real score: zero lanes stay zero, the norm and score
/// corrections cancel, and the rotary table keeps the real frequencies.
fn assert_padding_preserves_scores(real: usize, normed: bool, offset: f32) {
    let stored = (real / 2).next_power_of_two() * 2;
    let lanes = layout(real, stored, true);
    let ratio = width_ratio(real, stored);
    let (query, key) = (values(real * HIDDEN, 2), values(real * HIDDEN, 3));
    let (query_gain, key_gain) = (values(real, 4), values(real, 5));
    let rows = |scale| Pad::Rows {
        heads: 1,
        parts: 1,
        real,
        layout: lanes.clone(),
        scale,
    };
    let gain = |scale| Pad::Gain {
        layout: lanes.clone(),
        scale,
        offset,
    };
    let query_scale = if normed { 1.0 } else { ratio.recip().sqrt() };
    let (padded_query, _) = rows(query_scale)
        .apply(&query, (real, HIDDEN), 0.0, |v, s| v * s)
        .unwrap();
    let (padded_key, _) = rows(1.0)
        .apply(&key, (real, HIDDEN), 0.0, |v, s| v * s)
        .unwrap();
    let padded_gains = (
        gain(1.0).gain(&query_gain).unwrap(),
        gain(ratio.sqrt()).gain(&key_gain).unwrap(),
    );
    let offset = f64::from(offset);
    let expected = score(
        (&query, &key),
        normed.then_some((&query_gain[..], &key_gain[..], EPSILON, offset)),
        real,
        &real_frequencies(real),
    );
    let frequencies: Vec<f64> = rotary_frequencies(stored / 2, real, THETA)
        .into_iter()
        .map(f64::from)
        .collect();
    let actual = score(
        (&padded_query, &padded_key),
        normed.then_some((
            &padded_gains.0[..],
            &padded_gains.1[..],
            EPSILON * f64::from(ratio),
            offset,
        )),
        stored,
        &frequencies,
    );
    // The table is stored in f32, the reference rotates in f64.
    assert!(
        (expected - actual).abs() <= 1e-4 * expected.abs().max(1.0),
        "{real} {normed} {offset}: {expected} vs {actual}"
    );
}

#[test]
fn padding_preserves_attention_scores() {
    for real in [48, 96, 100] {
        assert_padding_preserves_scores(real, false, 0.0);
        assert_padding_preserves_scores(real, true, 0.0);
        assert_padding_preserves_scores(real, true, 1.0);
    }
}

#[test]
fn output_columns_and_gated_query_rows_keep_their_heads_in_place() {
    let lanes = layout(6, 8, false);
    let columns = Pad::Columns {
        heads: 2,
        real: 6,
        layout: lanes.clone(),
    };
    let weight: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let (padded, shape) = columns.apply(&weight, (1, 12), -1.0, |v, _| v).unwrap();
    assert_eq!(shape, [1, 16]);
    assert_eq!(
        padded,
        [
            0., 1., 2., 3., 4., 5., -1., -1., 6., 7., 8., 9., 10., 11., -1., -1.
        ]
    );
    // A gated query stacks its query and gate per head; only the query is scaled.
    let rows = Pad::Rows {
        heads: 1,
        parts: 2,
        real: 6,
        layout: lanes,
        scale: 2.0,
    };
    let (padded, shape) = rows.apply(&weight, (12, 1), 0.0, |v, s| v * s).unwrap();
    assert_eq!(shape, [16, 1]);
    assert_eq!(&padded[..8], &[0., 2., 4., 6., 8., 10., 0., 0.]);
    assert_eq!(&padded[8..], &[6., 7., 8., 9., 10., 11., 0., 0.]);
}
