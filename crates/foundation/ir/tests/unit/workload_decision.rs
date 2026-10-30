use super::*;
use serde_json::json;

fn decision(readout: DecisionReadout) -> Workload {
    Workload::Decision(DecisionSchema {
        questions: vec![DecisionQuestion::Categorical {
            options: vec![1, 2],
        }],
        calibration_temperature: 1.0,
        abstain_below: None,
        readout,
    })
}

#[test]
fn a_decision_reads_the_head_and_final_rows_its_readout_names() {
    let projection = decision(DecisionReadout::Projection);
    assert_eq!(projection.head(), Some(Head::Decision));
    assert_eq!(projection.final_readout(), OutputReadout::Full);
    let language = decision(DecisionReadout::LanguageModel);
    assert_eq!(language.head(), Some(Head::LanguageModel));
    assert_eq!(language.final_readout(), OutputReadout::Logits);
    assert_eq!(
        Workload::Generate { max_new_tokens: 1 }.final_readout(),
        OutputReadout::Logits
    );
    assert_eq!(Workload::Reward.final_readout(), OutputReadout::Full);
}

#[test]
fn decisions_written_before_the_readout_field_keep_reading_a_projection() {
    let earlier = json!({"Decision": {
        "questions": [{"Binary": {"negative_token": 0, "positive_token": 1}}],
        "calibration_temperature": 1.0,
        "abstain_below": null,
    }});
    let Workload::Decision(schema) = serde_json::from_value(earlier).expect("decision parses")
    else {
        panic!("expected a decision")
    };
    assert_eq!(schema.readout, DecisionReadout::Projection);
    let answer: DecisionAnswer = serde_json::from_value(
        json!({"probabilities": [0.5, 0.5], "expected": null, "abstained": false}),
    )
    .expect("answer parses");
    assert_eq!(answer.label_mass, None);
    assert_eq!(
        serde_json::to_value(decision(DecisionReadout::LanguageModel)).expect("serializes")["Decision"]
            ["readout"],
        "language_model"
    );
}
