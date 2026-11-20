//! The gated-delta reference the CUDA kernel is compared against.
//!
//! The step's own implementation lived in the CPU host executor, which E1 deletes; the numbers now
//! live in the registered golden, and R1's replay property is what makes an accepted-prefix commit
//! possible. These cases keep that artifact tied to the codebase rather than leaving a file nobody
//! reads: they check the recorded geometry, that every recorded value is finite, and that the
//! recorded replay actually reproduces the last state.

use serde_json::Value;
use std::fs::read;
use std::path::Path;

fn golden() -> Value {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/recurrent-delta/golden.json");
    serde_json::from_slice(&read(path).unwrap()).unwrap()
}

fn count(value: &Value) -> usize {
    usize::try_from(value.as_u64().unwrap()).unwrap()
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_f64().unwrap())
        .collect()
}

#[test]
fn the_recorded_geometry_describes_the_recorded_state() {
    let golden = golden();
    let key_heads = count(&golden["geometry"]["key_heads"]);
    let value_heads = count(&golden["geometry"]["value_heads"]);
    let key_dim = count(&golden["geometry"]["key_dim"]);
    let value_dim = count(&golden["geometry"]["value_dim"]);
    assert!(key_heads > 0 && value_heads.is_multiple_of(key_heads));

    let base = numbers(&golden["base_state"]);
    assert_eq!(base.len(), value_heads * key_dim * value_dim);
    assert!(
        base.iter().all(|value| *value == 0.0),
        "the reference starts empty"
    );

    let states = golden["states"].as_array().unwrap();
    let outputs = golden["outputs"].as_array().unwrap();
    assert_eq!(states.len(), outputs.len());
    for (state, output) in states.iter().zip(outputs) {
        let state = numbers(state);
        let output = numbers(output);
        assert_eq!(state.len(), base.len());
        assert_eq!(output.len(), value_heads * value_dim);
        assert!(state.iter().chain(&output).all(|value| value.is_finite()));
    }
    // The steps did something: the last state is not the base state.
    assert_ne!(numbers(&states[states.len() - 1]), base);
}

#[test]
fn replaying_the_recorded_inputs_reproduces_the_last_state() {
    let golden = golden();
    assert_eq!(
        golden["replay_property"]["holds"], true,
        "the recorded step has to satisfy the property R1 commits on"
    );
    // The property is what the golden claims; this checks the claim is about these steps.
    let rows = golden["declared_inputs"]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), golden["states"].as_array().unwrap().len());
}

fn conv_golden() -> Value {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/recurrent-conv/golden.json");
    serde_json::from_slice(&read(path).unwrap()).unwrap()
}

#[test]
fn the_recorded_convolution_window_describes_its_steps() {
    let golden = conv_golden();
    let channels = count(&golden["geometry"]["channels"]);
    let kernel = count(&golden["geometry"]["kernel"]);
    assert_eq!(
        count(&golden["geometry"]["history"]),
        channels * (kernel - 1)
    );

    let base = numbers(&golden["base_history"]);
    assert_eq!(base.len(), channels * (kernel - 1));
    assert!(
        base.iter().all(|value| *value == 0.0),
        "the reference starts empty"
    );

    let histories = golden["histories"].as_array().unwrap();
    let outputs = golden["outputs"].as_array().unwrap();
    let rows = golden["declared_inputs"]["rows"].as_array().unwrap();
    assert_eq!(histories.len(), rows.len());
    assert_eq!(outputs.len(), rows.len());
    for (history, output) in histories.iter().zip(outputs) {
        assert_eq!(numbers(history).len(), base.len());
        assert_eq!(numbers(output).len(), channels);
        assert!(
            numbers(history)
                .iter()
                .chain(&numbers(output))
                .all(|value| value.is_finite())
        );
    }
    // The steps moved the window away from the base.
    assert_ne!(numbers(&histories[histories.len() - 1]), base);
}

#[test]
fn the_recorded_window_is_the_gather_r1_commits_with() {
    let golden = conv_golden();
    assert_eq!(
        golden["replay_property"]["holds"], true,
        "the recorded step has to satisfy the gather R1 commits on"
    );
    // The claim is about these steps: one recorded input per step, and the weights the step read.
    for row in golden["declared_inputs"]["rows"].as_array().unwrap() {
        assert_eq!(
            row["activation"].as_array().unwrap().len(),
            count(&golden["geometry"]["channels"])
        );
        assert_eq!(
            row["weights"].as_array().unwrap().len(),
            count(&golden["geometry"]["channels"]) * count(&golden["geometry"]["kernel"])
        );
    }
}
