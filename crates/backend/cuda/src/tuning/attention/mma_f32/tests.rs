use super::*;
fn entry() -> Entry {
    let shape = Shape {
        heads: 24,
        kv_heads: 4,
        dim: 256,
        lanes: 128,
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
    let histories = mma::histories(shape, mma::capacity(shape));
    Entry {
        key: Key {
            hardware: r#"{"architecture":"sm_120"}"#.into(),
            driver_api: 13040,
            driver_build: Some("test-driver".into()),
            fingerprint: "b".repeat(64),
            schema: SCHEMA.into(),
            shape,
        },
        architecture: "sm_120".into(),
        profile: PROFILE.into(),
        selected: Choice::Mma {
            variant: Variant::Sm120,
        },
        trials: vec![Trial {
            variant: Variant::Sm120,
            fixtures: 4,
            elements: 4 * 128 * 24 * 256,
            max_f64_error: 0.002,
            forward: histories.into_iter().map(case).collect(),
            reverse: histories.into_iter().map(case).collect(),
        }],
        max_relative_l2: 0.003,
        history_f64_rows: 72,
    }
}
#[test]
fn float_cache_evidence_cannot_reuse_fp8_or_short_only_proofs() {
    assert!(entry().valid());
    let mut e = entry();
    e.key.schema = "fp8-cache-proof".into();
    assert!(!e.valid());
    let mut e = entry();
    e.history_f64_rows = 0;
    assert!(!e.valid());
    let mut e = entry();
    e.max_relative_l2 = f64::NAN;
    assert!(!e.valid());
    let mut e = entry();
    e.max_relative_l2 = 0.021;
    assert!(!e.valid());
    let mut e = entry();
    e.trials[0].reverse.pop();
    assert!(!e.valid());
    let mut e = entry();
    e.selected = Choice::Mma {
        variant: Variant::Hopper,
    };
    assert!(!e.valid());
}
#[test]
fn float_cache_profile_requires_short_and_long_cost_confirmation() {
    let mut e = entry();
    e.trials[0].forward[0].cold.candidate_ms.fill(1.2);
    assert!(!e.valid());
    let mut e = entry();
    e.trials[0].reverse[2].warm.candidate_ms.fill(1.1);
    assert!(!e.valid());
    e.selected = Choice::Simt;
    assert!(e.valid());
}
#[test]
fn float_reference_keeps_fractional_cache_values_and_negative_offset_mask() -> Result<()> {
    let s = entry().key.shape;
    let f = Fixture::short(s, 0)?;
    assert!(f.keys.iter().any(|v| v.to_bits() & 0xffff != 0));
    let f = Fixture::short(s, 2)?;
    assert_eq!(f.reference(0, 0)?, vec![0.0; s.dim]);
    assert_eq!(f.reference(s.lanes - 1, s.heads - 1)?, vec![0.0; s.dim]);
    Ok(())
}
#[test]
#[ignore = "requires CUDA and safe-run; F32 paged KV, complete short F64 outputs, long F64 rows, paired cold/warm and cache reuse"]
fn hardware_float_kv_mma_full_oracle_and_cache() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let shape = entry().key.shape;
    let session = ResidentTuningSession::default();
    let selected = select(&device, shape, true, &session);
    let result = memo()
        .lock()
        .map_err(|_| Error::invariant("F32 MMA test memo poisoned"))?
        .values()
        .find(|e| e.key.shape == shape)
        .cloned()
        .ok_or_else(|| Error::invalid("F32 MMA search did not finish"))?;
    assert!(result.valid());
    assert_eq!(select(&device, shape, false, &session), selected);
    println!(
        "F32_MMA_TUNING {}",
        serde_json::to_string(&result).map_err(device_error)?
    );
    Ok(())
}
