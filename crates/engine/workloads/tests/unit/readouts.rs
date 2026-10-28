use super::*;
use crate::{NativeWorkloads, softmax};
use infer_core::{ErrorCode, ModelId, RequestId};
use infer_ir::{
    BackboneKind, DecisionQuestion, DecisionReadout, DecisionSchema, FeedForward, Head, Mixer,
    Modality, Pooling, PositionSpec, Qos, RequestInput, Sampling,
};
use std::f64::consts::E;
use std::slice::from_ref;

fn readouts() -> ProjectionWorkloads {
    let weights = [
        ("embedding.weight", vec![2, 2], vec![2.0, 0.0, 0.0, 3.0]),
        ("rank.weight", vec![1, 2], vec![1.0, -1.0]),
        (
            "decision.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        ),
    ]
    .into_iter()
    .map(|(name, shape, data)| (name.into(), HostTensor { shape, data }))
    .collect();
    ProjectionWorkloads {
        weights,
        hidden: 2,
        vocab: 8,
        identity: "test-readouts".into(),
    }
}

fn model() -> ModelIr {
    ModelIr {
        id: ModelId::ONE,
        backbone: BackboneKind::Recurrent,
        vocab_size: 8,
        hidden_size: 2,
        max_sequence: 16,
        mixers: vec![Mixer::Recurrent { state_width: 2 }],
        feed_forward: FeedForward::Dense { intermediate: 4 },
        position: PositionSpec {
            rope_theta: 10_000.0,
            rotary_fraction: 1.0,
            multimodal_sections: vec![],
            interleaved: false,
        },
        norm_epsilon: 1e-5,
        norm_weight_offset: 0.0,
        heads: vec![
            Head::LanguageModel,
            Head::Embedding,
            Head::Rank,
            Head::Decision,
        ],
        modalities: vec![Modality::Text],
        state: vec![],
        tied_embeddings: false,
    }
}

fn request(workload: Workload) -> CanonicalRequest {
    let input = if matches!(workload, Workload::Rerank { .. }) {
        RequestInput::Pairs {
            query: vec![1].into(),
            documents: vec![vec![2].into(), vec![3].into()].into(),
        }
    } else {
        RequestInput::Sequence {
            tokens: vec![1, 2].into(),
            media: vec![],
        }
    };
    CanonicalRequest {
        id: RequestId::ONE,
        model: ModelId::ONE,
        session: None,
        input,
        workload,
        qos: Qos::default(),
        sampling: Sampling::default(),
        extensions: BTreeMap::new(),
    }
}

fn decision(question: DecisionQuestion) -> Workload {
    Workload::Decision(DecisionSchema {
        questions: vec![question],
        calibration_temperature: 1.0,
        abstain_below: None,
        readout: DecisionReadout::Projection,
    })
}

#[test]
fn decision_plans_check_vocabulary_and_projection_channels_for_every_question() {
    let readouts = readouts();
    let model = model();
    for token in [1, 3, 4, 7, 8, u32::MAX] {
        for question in [
            DecisionQuestion::Binary {
                negative_token: 0,
                positive_token: token,
            },
            DecisionQuestion::Binary {
                negative_token: token,
                positive_token: 0,
            },
            DecisionQuestion::Categorical {
                options: vec![0, token],
            },
            DecisionQuestion::Ordinal {
                options: vec![token, 0],
                values: vec![0.0, 1.0],
            },
            DecisionQuestion::Continuous {
                token,
                min: 0.0,
                max: 1.0,
            },
        ] {
            let request = request(decision(question));
            let original = request.clone();
            let native = NativeWorkloads.plan(&request, &model, ProgramId::ONE);
            let projected = readouts.plan(&request, &model, ProgramId::ONE);
            assert_eq!(native.is_ok(), token < 8, "native token {token}");
            assert_eq!(projected.is_ok(), token < 4, "projected token {token}");
            if let Err(error) = projected {
                assert_eq!(error.code, ErrorCode::InvalidInput);
                assert_eq!(
                    error.message,
                    if token < 8 {
                        "decision option exceeds installed readout channels"
                    } else {
                        "decision option exceeds vocabulary"
                    }
                );
            }
            assert_eq!(
                request, original,
                "planning must preserve the borrowed request"
            );
        }
    }
}

fn output(hidden: Vec<Vec<f32>>) -> ModelOutput {
    ModelOutput {
        next_token: Some(7),
        logits: vec![999.0; 128],
        hidden,
        tokens: vec![6, 7],
    }
}

#[test]
fn projected_embedding_ranking_and_decisions_preserve_outputs_and_borrowed_inputs() {
    let readouts = readouts();
    let first = output(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    for (pooling, expected) in [
        (Pooling::Mean, vec![4.0, 9.0]),
        (Pooling::Last, vec![6.0, 12.0]),
    ] {
        let request = request(Workload::Embed {
            pooling,
            dimensions: None,
            normalize: false,
        });
        assert_eq!(
            readouts.postprocess(&request, from_ref(&first)).unwrap(),
            WorkloadOutput::Embedding(expected.into())
        );
    }
    let outputs = [first, output(vec![vec![4.0, 1.0]])];
    let original = outputs.clone();
    let ranking = readouts
        .postprocess(&request(Workload::Rerank { top_k: 2 }), &outputs)
        .unwrap();
    let WorkloadOutput::Ranking(ranking) = ranking else {
        panic!("expected ranking")
    };
    assert_eq!(
        ranking
            .iter()
            .map(|r| (r.index, r.score))
            .collect::<Vec<_>>(),
        vec![(1, 3.0), (0, -1.0)]
    );
    let decision = request(decision(DecisionQuestion::Categorical {
        options: vec![0, 1, 2, 3],
    }));
    let answers = readouts.postprocess(&decision, &outputs[..1]).unwrap();
    let WorkloadOutput::Decisions(answers) = answers else {
        panic!("expected decisions")
    };
    assert_eq!(
        answers[0].probabilities,
        softmax(&[3.0, 4.0, -3.0, -4.0], 1.0).unwrap()
    );
    assert_eq!(
        outputs, original,
        "postprocessing must preserve borrowed backbone data"
    );
}

#[test]
fn projected_readouts_keep_shape_and_output_count_failures() {
    let readouts = readouts();
    let embed = request(Workload::Embed {
        pooling: Pooling::Mean,
        dimensions: None,
        normalize: false,
    });
    let empty = output(vec![]);
    let malformed = output(vec![vec![1.0]]);
    let nonfinite = output(vec![vec![f32::NAN, 0.0]]);
    for bad in [empty.clone(), malformed, nonfinite] {
        assert!(readouts.postprocess(&embed, &[bad]).is_err());
    }
    let valid = output(vec![vec![1.0, 2.0]]);
    assert!(readouts.postprocess(&embed, &[]).is_err());
    assert!(
        readouts
            .postprocess(&embed, &[valid.clone(), valid])
            .is_err()
    );
    let ranking = request(Workload::Rerank { top_k: 1 });
    assert_eq!(
        readouts
            .postprocess(&ranking, &[empty])
            .unwrap_err()
            .message,
        "readout has no hidden row"
    );
}

#[test]
fn language_model_decisions_read_vocabulary_logits_without_a_projection() {
    let mut readouts = readouts();
    readouts.weights.remove("decision.weight");
    let workload = Workload::Decision(DecisionSchema {
        questions: vec![DecisionQuestion::Categorical {
            options: vec![1, 2],
        }],
        calibration_temperature: 1.0,
        abstain_below: None,
        readout: DecisionReadout::LanguageModel,
    });
    let mut model = model();
    model.heads = vec![Head::LanguageModel];
    let request = request(workload);
    assert!(readouts.supports(&request.workload));
    readouts.plan(&request, &model, ProgramId::ONE).unwrap();
    NativeWorkloads
        .plan(&request, &model, ProgramId::ONE)
        .unwrap();
    // A projection decision still needs its head on a model that only generates.
    let projection = self::request(decision(DecisionQuestion::Categorical {
        options: vec![1, 2],
    }));
    assert_eq!(
        NativeWorkloads
            .plan(&projection, &model, ProgramId::ONE)
            .unwrap_err()
            .code,
        ErrorCode::Unsupported
    );

    let mut logits = vec![0.0; 8];
    logits[1] = 1.0;
    logits[2] = 1.0;
    let output = ModelOutput {
        next_token: None,
        logits,
        hidden: vec![],
        tokens: vec![],
    };
    let WorkloadOutput::Decisions(answers) =
        readouts.postprocess(&request, from_ref(&output)).unwrap()
    else {
        panic!("expected decisions")
    };
    assert_eq!(answers[0].probabilities, vec![0.5, 0.5]);
    let mass = f64::from(answers[0].label_mass.unwrap());
    assert!((mass - 2.0 * E / 2.0f64.mul_add(E, 6.0)).abs() < 1e-6);
}
