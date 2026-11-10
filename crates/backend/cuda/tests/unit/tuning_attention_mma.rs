use super::*;
use crate::device::device_error;
use crate::tuning::attention::SCHEMA as ATTENTION_SCHEMA;
fn entry() -> Entry {
    let shape = Shape {
        heads: 24,
        kv_heads: 4,
        dim: 256,
        lanes: 256,
        window: 0,
    };
    let case = |history| Case {
        case: history,
        cold: Times {
            baseline_ms: vec![1.0; 10],
            candidate_ms: vec![0.5; 10],
        },
        warm: Times {
            baseline_ms: vec![1.0; 10],
            candidate_ms: vec![0.5; 10],
        },
    };
    let trial = |variant| Trial {
        variant,
        fixtures: reference::FIXTURES,
        elements: reference::FIXTURES * shape.lanes * shape.heads * shape.dim,
        max_f64_error: 0.002,
        forward: histories(shape, capacity(shape))
            .into_iter()
            .map(case)
            .collect(),
        reverse: histories(shape, capacity(shape))
            .into_iter()
            .map(case)
            .collect(),
    };
    Entry {
        key: Key {
            hardware: r#"{"architecture":"sm_100"}"#.into(),
            driver_api: 13040,
            driver_build: Some("test-driver".into()),
            fingerprint: "a".repeat(64),
            schema: SCHEMA.into(),
            shape,
        },
        architecture: "sm_100".into(),
        profile: catalog::PROFILE.into(),
        baseline: Choice::Simt,
        selected: Choice::Mma {
            variant: Variant::Sm100,
        },
        trials: vec![trial(Variant::Sm100), trial(Variant::Sm100Pair)],
    }
}
#[test]
fn bf16_profile_has_its_own_tuning_identity() {
    // The BF16 FA4 choice reuses its independent schema/table section, never the canonical
    // FP8-KV cache. A canonical entry therefore cannot satisfy this profile.
    assert_ne!(SCHEMA, ATTENTION_SCHEMA);
    assert!(SCHEMA.contains("fa4-bf16"));
    assert!(ATTENTION_SCHEMA.contains("prefill-fp8-kv"));
    assert!(catalog::PROFILE.contains("fa4-bf16-qkv"));
    // FP16 has its own schema and artifact identity; a BF16 entry can never satisfy it.
    assert_ne!(attention_mma::f16::SCHEMA, SCHEMA);
    assert!(attention_mma::f16::SCHEMA.contains("fa4-f16"));
    assert_ne!(
        attention_mma::f16::artifact_identity(),
        attention_mma::artifact_identity()
    );
    let mut entry = entry();
    entry.key.schema = ATTENTION_SCHEMA.into();
    assert!(!entry.valid());
}

#[test]
fn cluster_selection_requires_complete_independent_profile_evidence() {
    assert!(entry().valid());
    let mut e = entry();
    e.trials.pop();
    assert!(!e.valid());
    let mut e = entry();
    e.trials[1].variant = Variant::Sm103Pair;
    assert!(!e.valid());
    let mut e = entry();
    e.trials[0].reverse.pop();
    assert!(!e.valid());
    let mut e = entry();
    e.trials[0].fixtures -= 1;
    assert!(!e.valid());
    let mut e = entry();
    e.trials[0].max_f64_error = f64::NAN;
    assert!(!e.valid());
    let mut e = entry();
    e.profile = "canonical-f32".into();
    assert!(!e.valid());
}
#[test]
fn fast_long_case_cannot_hide_a_short_regression() {
    let mut e = entry();
    e.trials[0].forward[0].cold.candidate_ms.fill(1.2);
    assert!(!e.valid());
    e.selected = Choice::Mma {
        variant: Variant::Sm100Pair,
    };
    assert!(e.valid());
}

#[test]
#[ignore = "requires CUDA and safe-run; AOT ABI, full BF16-profile F64 sanity and measured architecture/cluster selection"]
fn hardware_mma_aot_graph_and_tuning() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let shape = Shape {
        heads: 24,
        kv_heads: 4,
        dim: 256,
        lanes: 256,
        window: 0,
    };
    let session = ResidentTuningSession::default();
    let selected = select(&device, shape, true, &session, Choice::Simt);
    let entry = memo()
        .lock()
        .map_err(|_| Error::invariant("test MMA memo poisoned"))?
        .values()
        .find(|e| e.key.shape == shape)
        .cloned()
        .ok_or_else(|| Error::invalid("MMA numerical and cluster search did not complete"))?;
    assert!(entry.valid());
    assert_eq!(
        select(&device, shape, false, &session, Choice::Simt),
        selected
    );
    println!(
        "MMA_ATTENTION_TUNING {}",
        serde_json::to_string(&entry).map_err(device_error)?
    );
    Ok(())
}
#[test]
fn stored_history_cases_read_as_shared_cases() -> Result<()> {
    let times = r#"{"baseline_ms":[1.0,1.0],"candidate_ms":[0.5,0.5]}"#;
    let stored = format!(r#"{{"history":8192,"cold":{times},"warm":{times}}}"#);
    let case: Case = serde_json::from_str(&stored).map_err(device_error)?;
    assert_eq!(case.case, 8192);
    assert!(accepted(&case));
    Ok(())
}
