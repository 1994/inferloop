use super::*;

fn job() -> Result<OutputJob> {
    let request = serde_json::from_value(serde_json::json!({
        "id": 1, "model": 1, "input": {"Sequence": {"tokens": [1]}},
        "workload": {"Generate": {"max_new_tokens": 8}}
    }))
    .map_err(|error| Error::invalid(error.to_string()))?;
    Ok(OutputJob {
        constraint: None,
        request: Arc::new(request),
        output: ModelOutput {
            next_token: Some(5),
            logits: Vec::new(),
            hidden: Vec::new(),
            tokens: vec![2, 3],
        },
        shape: OutputShape {
            logits: 8,
            rows: 0,
            width: 2,
        },
        sample: Some(0),
        generated: Vec::new().into(),
        project: None,
    })
}
fn process(job: OutputJob) -> Result<ProcessedOutput> {
    job.process(
        &infer_workloads::NativeWorkloads,
        &mut infer_workloads::SamplingWorkspace::default(),
    )
}

#[test]
fn device_token_preserves_accepted_prefix_and_final_token() -> Result<()> {
    let result = process(job()?)?;
    assert_eq!(result.tokens.as_slice(), &[2, 3, 5]);
    let output = result
        .output
        .ok_or_else(|| Error::invariant("processed output"))?;
    assert_eq!(output.next_token, None);
    assert_eq!(output.logits.as_slice(), &[] as &[f32]);
    Ok(())
}
#[test]
fn backend_token_does_not_bypass_sampling_scope_or_shape_validation() -> Result<()> {
    let mut cases = Vec::new();
    let mut value = job()?;
    value.sample = None;
    cases.push(value);
    let mut value = job()?;
    value.project = Some(Vec::new());
    cases.push(value);
    let mut value = job()?;
    Arc::make_mut(&mut value.request).sampling.temperature = 1.0;
    cases.push(value);
    let mut value = job()?;
    Arc::make_mut(&mut value.request).sampling.presence_penalty = 0.5;
    cases.push(value);
    let mut value = job()?;
    value.output.next_token = Some(8);
    cases.push(value);
    let mut value = job()?;
    value.output.logits = vec![0.0; 8];
    cases.push(value);
    let mut value = job()?;
    value.output.tokens.push(8);
    cases.push(value);
    let mut value = job()?;
    value.output.tokens = vec![1; DECIDED_TOKEN_CAPACITY];
    cases.push(value);
    let mut value = job()?;
    value.shape.rows = 1;
    value.output.hidden = vec![vec![0.0, f32::NAN]];
    cases.push(value);
    for value in cases {
        assert!(process(value).is_err());
    }
    Ok(())
}
#[test]
fn legacy_output_serialization_and_host_sampling_remain_valid() -> Result<()> {
    let old: ModelOutput = serde_json::from_value(serde_json::json!({
        "logits": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "hidden": [], "tokens": []
    }))
    .map_err(|error| Error::invalid(error.to_string()))?;
    assert_eq!(old.next_token, None);
    let wire = serde_json::to_value(&old).map_err(|error| Error::invalid(error.to_string()))?;
    assert!(wire.get("next_token").is_none());
    let mut value = job()?;
    value.output = old;
    assert_eq!(process(value)?.tokens.as_slice(), &[7]);
    Ok(())
}

struct OnlyTokenTwo;
impl infer_spi::GenerationConstraint for OnlyTokenTwo {
    fn mask(&mut self) -> Result<&[u32]> {
        Ok(&[4])
    }
    fn commit(&mut self, token: u32) -> Result<()> {
        if token == 2 {
            Ok(())
        } else {
            Err(Error::invariant("illegal token committed"))
        }
    }
}
fn constrained_job() -> Result<OutputJob> {
    let mut value = job()?;
    value.constraint = Some(infer_spi::ConstraintHandle::new(
        Box::new(OnlyTokenTwo),
        8,
        128,
    ));
    value.output.tokens.clear();
    value.output.next_token = None;
    value.output.logits = vec![0.0, 100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    Ok(value)
}
#[test]
fn constrained_output_masks_the_peak_and_transfers_position_state() -> Result<()> {
    let first = process(constrained_job()?)?;
    assert_eq!(first.tokens.as_slice(), &[2]);
    let mut second = constrained_job()?;
    second.constraint = first.constraint;
    second.sample = Some(1);
    second.generated.push(2);
    let mut result = process(second)?;
    assert_eq!(result.tokens.as_slice(), &[2]);
    let handle = result.constraint.as_mut().unwrap();
    assert!(handle.select(1, |_| Ok(2)).is_err());
    // Diagnostic clones cannot deserialize into a matcher that silently restarts at position 0.
    let wire = serde_json::to_string(handle).unwrap();
    assert!(serde_json::from_str::<infer_spi::ConstraintHandle>(&wire).is_err());
    let mut clone = handle.clone();
    assert_eq!(clone.select(2, |_| Ok(2))?, 2);
    assert!(handle.select(2, |_| Ok(2)).is_err());
    Ok(())
}
#[test]
fn constrained_output_rejects_device_tokens_and_speculative_prefixes() -> Result<()> {
    let mut device = constrained_job()?;
    device.output.next_token = Some(1);
    device.output.logits.clear();
    assert!(process(device).is_err());
    let mut speculative = constrained_job()?;
    speculative.output.tokens.push(1);
    assert!(process(speculative).is_err());
    Ok(())
}
