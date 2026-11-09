use super::*;
use crate::constants::PROMPT_GEMM_TILE_ROWS;
use crate::constants::QUANT_GEMM_TILE;
use crate::mlp::Matrix;
use crate::tuning::resident_gemm::prompt_tile_rows;
use std::env::temp_dir;
use std::fs::remove_file;
use std::fs::write;
use std::process::id as process_id;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

fn key() -> Key {
    Key {
        hardware: "complete-test-device-profile".into(),
        driver_api: 13000,
        driver_build: Some("test-driver-build".into()),
        kernel_fingerprint: "all-six-frontend-identities".into(),
        library: Some(LibraryIdentity {
            lt_version: 130_001,
            cudart_version: 13000,
        }),
        contract: CONTRACT.into(),
        vendor_candidates: false,
        m: 64,
        n: 67,
        k: 272,
    }
}
fn times(speedup: f32) -> PairedTimes {
    PairedTimes {
        baseline_ms: vec![speedup; 10],
        candidate_ms: vec![1.0; 10],
        baseline_median_ms: speedup,
        candidate_median_ms: 1.0,
        speedup,
    }
}
fn entry() -> Entry {
    let selected = Choice::Native(NativeConfig {
        tile_m: 64,
        tile_n: 128,
    });
    Entry {
        key: key(),
        evidence: search::Evidence {
            selected,
            exact_candidates_verified: true,
            trials: vec![Trial {
                choice: selected,
                order: MeasurementOrder::BaselineFirst,
                cold: times(1.03),
                warm: times(1.03),
            }],
            rejected: None,
            confirmation: Trial {
                choice: selected,
                order: MeasurementOrder::CandidateFirst,
                cold: times(1.03),
                warm: times(1.03),
            },
        },
    }
}
#[test]
fn fp8_typed_keys_isolate_library_driver_compiler_contract_and_geometry() -> Result<()> {
    let original = key();
    assert!(original.persistent());
    let mut variants = Vec::new();
    let mut other = original.clone();
    other.hardware.push_str("-different-SM-L2-capacity");
    variants.push(other);
    let mut other = original.clone();
    other.driver_api += 1;
    variants.push(other);
    let mut other = original.clone();
    other.driver_build = Some("changed-release-build".into());
    variants.push(other);
    let mut other = original.clone();
    other
        .kernel_fingerprint
        .push_str("-changed-compiler-source");
    variants.push(other);
    let mut other = original.clone();
    other.library.as_mut().unwrap().lt_version += 1;
    variants.push(other);
    let mut other = original.clone();
    other.library.as_mut().unwrap().cudart_version += 1;
    variants.push(other);
    let mut other = original.clone();
    other.vendor_candidates = true;
    variants.push(other);
    let mut other = original.clone();
    other.contract.push_str("-different-accumulation");
    variants.push(other);
    for axis in 0..3 {
        let mut other = original.clone();
        match axis {
            0 => other.m += 1,
            1 => other.n += 1,
            _ => other.k += 1,
        }
        variants.push(other);
    }
    assert!(variants.into_iter().all(|changed| changed != original));
    let mut unknown = original.clone();
    unknown.driver_build = None;
    assert!(!unknown.persistent());
    unknown.driver_build = Some("known".into());
    unknown.library = None;
    unknown.vendor_candidates = true;
    assert!(!unknown.persistent());
    let encoded = serde_json::to_string(&entry()).map_err(device_error)?;
    let decoded: Entry = serde_json::from_str(&encoded).map_err(device_error)?;
    assert_eq!(decoded.key, original);
    assert!(decoded.valid());
    Ok(())
}
#[test]
fn fp8_confirmation_requires_exact_complete_finite_and_two_percent_both_modes() {
    let mut valid = entry();
    assert!(valid.valid());
    valid.evidence.confirmation.order = MeasurementOrder::BaselineFirst;
    assert!(!valid.valid());
    valid.evidence.confirmation.order = MeasurementOrder::CandidateFirst;
    valid.evidence.trials[0].order = MeasurementOrder::CandidateFirst;
    assert!(!valid.valid());
    valid.evidence.trials[0].order = MeasurementOrder::BaselineFirst;
    valid.evidence.trials[0].warm = times(1.019);
    assert!(!valid.valid());
    valid.evidence.trials[0].warm = times(1.03);
    valid.evidence.exact_candidates_verified = false;
    assert!(!valid.valid());
    valid.evidence.exact_candidates_verified = true;
    valid.evidence.confirmation.warm = times(1.019);
    assert!(!valid.valid());
    valid.evidence.confirmation.warm = times(1.03);
    valid.evidence.confirmation.cold.baseline_ms[0] = f32::NAN;
    assert!(!valid.valid());
    valid.evidence.confirmation.cold = times(1.03);
    valid.evidence.confirmation.cold.candidate_ms.pop();
    assert!(!valid.valid());
    valid.evidence.confirmation.cold = times(1.03);
    valid.evidence.confirmation.cold.speedup = 1.04;
    assert!(!valid.valid());
    valid.evidence.confirmation.cold = times(1.03);
    valid.evidence.selected = Choice::Native(NativeConfig {
        tile_m: 32,
        tile_n: 128,
    });
    assert!(!valid.valid());
    let config = AlgorithmConfig {
        id: 7,
        tile: 1,
        split_k: 0,
        reduction: 0,
        swizzle: 0,
        custom: 0,
        stages: 1,
        inner: 0,
        cluster: 0,
    };
    // Nonnegative split counts remain subject to SDK AlgoCheck and actual zero-workspace warmup.
    assert!(Choice::Lt(config).valid());
    let mut invalid = config;
    invalid.split_k = -1;
    assert!(!Choice::Lt(invalid).valid());
    let encoded = serde_json::to_string(&config).unwrap();
    assert!(!encoded.contains("opaque"));
    assert_eq!(
        serde_json::from_str::<AlgorithmConfig>(&encoded).unwrap(),
        config
    );
}
#[test]
fn fp8_records_from_the_former_section_keep_their_choice_and_rules() -> Result<()> {
    let current = entry();
    let mut stored = serde_json::to_value(&current).map_err(device_error)?;
    let record = stored
        .as_object_mut()
        .ok_or_else(|| Error::invariant("record object"))?;
    record.remove("exact_candidates_verified");
    record.insert("finite_bitwise".into(), true.into());
    record.insert("fixtures".into(), FIXTURES.into());
    let reverse = |trial: &mut Value| {
        if let Some(trial) = trial.as_object_mut() {
            let reverse = trial.remove("order") == Some("CandidateFirst".into());
            trial.insert("reverse".into(), reverse.into());
        }
    };
    record["trials"]
        .as_array_mut()
        .into_iter()
        .flatten()
        .for_each(reverse);
    reverse(&mut record["confirmation"]);
    let mut short = stored.clone();
    short["fixtures"] = 1.into();
    let decode = |stored| serde_json::from_value::<Entry>(legacy(stored)).map_err(device_error);
    let migrated = decode(stored)?;
    assert_eq!(migrated.evidence.selected, current.evidence.selected);
    assert_eq!(migrated.evidence.rejected, None);
    assert_eq!(
        migrated.evidence.confirmation.order,
        MeasurementOrder::CandidateFirst
    );
    assert!(migrated.valid());
    assert!(!decode(short)?.valid());
    Ok(())
}
#[test]
fn fp8_exact_gate_detects_single_bit_signed_zero_and_nonfinite_outputs() -> Result<()> {
    let values = [1.0, -2.5, 0.0];
    bitwise(&values, &values)?;
    assert!(bitwise(&[], &[]).is_err());
    assert!(bitwise(&values[..2], &values).is_err());
    assert!(bitwise(&[1.0, -2.5, -0.0], &values).is_err());
    assert!(bitwise(&[1.0, f32::NAN, 0.0], &values).is_err());
    assert!(bitwise(&[f32::from_bits(1.0_f32.to_bits() + 1), -2.5, 0.0], &values).is_err());
    let first = inputs(3, 272, 0)?;
    let second = inputs(3, 272, 1)?;
    let alternate = inputs(3, 272, 2)?;
    let random = inputs(3, 272, FIXTURE_RANDOM)?;
    assert_ne!(first, alternate);
    assert_ne!(alternate, random);
    assert!(
        alternate
            .iter()
            .chain(&random)
            .all(|value| value.is_finite())
    );
    assert!(first.iter().chain(&second).all(|value| value.is_finite()));
    assert_ne!(first, second);
    assert!(first[..272].iter().all(|value| *value == 0.0));
    Ok(())
}
#[test]
fn fp8_table_atomic_updates_preserve_scalar_and_nvfp4_sections() -> Result<()> {
    let path = temp_dir().join(format!(
        "fp8-tuning-test-{}-{}.json",
        process_id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(device_error)?
            .as_nanos()
    ));
    let initial = TilingTable::default();
    write(
        &path,
        serde_json::to_string(&initial).map_err(device_error)?,
    )
    .map_err(device_error)?;
    assert!(super::super::update_table(Some(&path), |table| {
        table.devices.insert("old-device".into(), vec![]);
        let stored = serde_json::to_value(entry()).unwrap_or_default();
        table.records.entry(FAMILY.into()).or_default().push(stored);
    }));
    assert!(super::super::update_table(Some(&path), |table| {
        table.devices.insert("new-device".into(), vec![]);
    }));
    let loaded = TilingTable::load(&path)?;
    assert_eq!(loaded.devices.len(), 2);
    let stored = search::lookup::<Key, Choice>(&loaded, FAMILY, &key(), |_| true);
    assert!(stored.is_some_and(|entry| entry.valid()));
    remove_file(path).map_err(device_error)?;
    Ok(())
}
#[test]
#[ignore = "requires CUDA hardware; root runs release serially"]
fn fp8_native_candidates_and_lt_exact_gate_cover_tails_and_weight_owners() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let mut support = vendor_fp8::Support::new_with(&device, true);
    for shape in [
        [1, 67, 272],
        [3, 65, 528],
        [17, 129, 256],
        [65, 67, 272],
        [129, 129, 528],
    ] {
        let [m, n, k] = shape;
        let codes = (0..n * k)
            .map(|i| {
                f8e4m3fn(
                    24 + u8::try_from((i * 13 + i / 7) % 56).unwrap_or_default()
                        + if i % 5 < 2 { 128 } else { 0 },
                )
            })
            .collect();
        let scales = (0..n)
            .map(|i| f32::from(u16::try_from(i % 13 + 1).unwrap_or_default()) / 113.0)
            .collect();
        let weight = ProjectionWeight::Fp8(
            Matrix::upload(&device, codes, &[n, k])?,
            device.runtime().upload(scales)?,
        );
        for fixture in 0..FIXTURES {
            let base = Captured::new(
                &device,
                &mut support,
                &weight,
                shape,
                Choice::baseline(m),
                fixture,
            )?;
            base.run(&device)?;
            let expected = device.runtime().read(&base.output)?;
            for tile_m in ROW_TILES {
                for tile_n in COLUMN_TILES {
                    let choice = Choice::Native(NativeConfig { tile_m, tile_n });
                    let candidate =
                        Captured::new(&device, &mut support, &weight, shape, choice, fixture)?;
                    candidate.run(&device)?;
                    bitwise(&device.runtime().read(&candidate.output)?, &expected)?;
                }
            }
        }
        let choice = Choice::Native(NativeConfig {
            tile_m: 64,
            tile_n: 128,
        });
        let mut identity = key();
        identity.m = m;
        identity.n = n;
        identity.k = k;
        let expired = ResidentTuningSession::without_search_budget();
        assert!(
            verify(
                &device,
                &mut support,
                &weight,
                shape,
                choice,
                Some(&expired)
            )?
            .is_none()
        );
        assert!(tune(&device, &mut support, &weight, identity.clone(), &expired)?.is_none());
        assert_eq!(
            cached_gate(&device, &mut support, &weight, &identity, choice)?,
            choice
        );
        assert_eq!(remembered(&weight, &identity, choice)?, Some(true));
        let ProjectionWeight::Fp8(codes, scales) = &weight else {
            return Err(Error::invariant("FP8 test channel"));
        };
        let second = ProjectionWeight::Fp8(
            codes.clone(),
            device.runtime().upload(device.read(scales.tensor())?)?,
        );
        assert_eq!(remembered(&second, &identity, choice)?, None);
        assert_eq!(
            cached_gate(&device, &mut support, &second, &identity, choice)?,
            choice
        );
        let invalid =
            ProjectionWeight::Fp8(codes.clone(), device.runtime().upload(vec![f32::NAN; n])?);
        assert_eq!(
            cached_gate(&device, &mut support, &invalid, &identity, choice)?,
            Choice::baseline(m)
        );
        assert_eq!(remembered(&invalid, &identity, choice)?, Some(false));
        assert_eq!(remembered(&weight, &identity, choice)?, Some(true));
    }
    Ok(())
}

#[test]
fn fp8_row_tile_128_is_accepted_and_enumerated_after_the_contract_bump() {
    // `Key::contract` is part of the persisted FP8 table key: bumping it is what
    // forces a stored 64-wide selection to be re-searched with the 128 row tile.
    assert!(CONTRACT.contains("fp8-channel-exact-v2"));
    assert!(CONTRACT.contains(":tile-m128"));
    for tile_n in COLUMN_TILES {
        assert!(
            NativeConfig {
                tile_m: 128,
                tile_n
            }
            .valid()
        );
    }
    let mut choices = Vec::new();
    for tile_m in ROW_TILES {
        for tile_n in COLUMN_TILES {
            choices.push(Choice::Native(NativeConfig { tile_m, tile_n }));
        }
    }
    assert!(choices.iter().any(|choice| {
        matches!(choice, Choice::Native(config)
            if config.tile_m == 128
                && config.tile_n == QUANT_GEMM_TILE[1])
    }));
    assert_ne!(choices, [] as [Choice; 0]);
    // The 64-row tile stays a search candidate even though the wide tile is now the prompt
    // baseline, so an autotune can still demote it on a shape where 64 measures faster.
    assert!(ROW_TILES.contains(&PROMPT_GEMM_TILE_ROWS));
    // The baseline is row-dependent: decode and tails keep 16/64, only full chunks take 128.
    assert_eq!(prompt_tile_rows(1), 16);
    assert_eq!(prompt_tile_rows(16), 16);
    assert_eq!(prompt_tile_rows(17), 64);
    assert_eq!(prompt_tile_rows(64), 64);
    assert_eq!(prompt_tile_rows(127), 64);
    assert_eq!(prompt_tile_rows(128), 128);
    assert_eq!(prompt_tile_rows(256), 128);
    assert_eq!(NativeConfig::baseline(256).tile_m, 128);
    assert_eq!(NativeConfig::baseline(64).tile_m, 64);
}
