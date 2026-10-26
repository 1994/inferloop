use super::*;
use crate::constants::QUANT_GEMM_TILE;
use crate::resident::ProgramGeometry;
use crate::strategy::LinearTiling;
use std::env::temp_dir;
use std::fs::remove_file;
use std::fs::write;
use std::process::id;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

fn profile() -> DeviceProfile {
    DeviceProfile {
        name: "test-device".into(),
        architecture: "sm_test".into(),
        multiprocessors: 12,
        total_memory_bytes: 8 << 30,
        memory_clock_khz: 1000,
        memory_bus_width_bits: 128,
        l2_cache_bytes: 1 << 20,
    }
}

fn key() -> Result<ResidentGemmKey> {
    ResidentGemmKey::new(
        &profile(),
        13000,
        [64, 67, 272],
        Some("test-driver-build".into()),
        "test-kernel-fingerprint".into(),
    )
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

fn entry(key: ResidentGemmKey) -> ResidentGemmEntry {
    let config = GemmConfig {
        tile_m: 64,
        tile_n: 128,
        main_loop: MainLoop::Prefetch8,
    };
    ResidentGemmEntry {
        key,
        selected: config,
        bitwise_verified: true,
        trials: vec![GemmTrial {
            config,
            order: MeasurementOrder::BaselineFirst,
            cold: times(1.03),
            warm: times(1.03),
        }],
        confirmation: GemmTrial {
            config,
            order: MeasurementOrder::CandidateFirst,
            cold: times(1.03),
            warm: times(1.03),
        },
    }
}

#[test]
fn resident_keys_isolate_full_geometry_hardware_and_numeric_contract() -> Result<()> {
    let original = key()?;
    assert!(original.kernel_schema.contains("search-ab:confirm-ba"));
    let mut table = TilingTable::default();
    table.resident_gemm.push(entry(original.clone()));
    assert!(matches!(
        decision(Some(&table), &original, false)?,
        Decision::Cached(_)
    ));
    let mut variants = Vec::new();
    let mut changed = original.clone();
    changed.m += 1;
    variants.push(changed);
    let mut changed = original.clone();
    changed.n += 1;
    variants.push(changed);
    let mut changed = original.clone();
    changed.k += 16;
    variants.push(changed);
    let mut changed = original.clone();
    changed.driver_version += 1;
    variants.push(changed);
    let mut changed = original.clone();
    changed.driver_build = Some("different-release-build".into());
    variants.push(changed);
    let mut changed = original.clone();
    changed.driver_build = None;
    variants.push(changed);
    let mut changed = original.clone();
    changed
        .kernel_fingerprint
        .push_str("-changed-frontend-or-assembler");
    variants.push(changed);
    let mut changed = original.clone();
    changed.kernel_schema.push_str("-changed");
    variants.push(changed);
    let mut changed = original.clone();
    assert!(changed.kernel_schema.contains("nvfp4-v3"));
    changed.kernel_schema = changed.kernel_schema.replace("nvfp4-v3", "nvfp4-v1");
    variants.push(changed);
    let mut changed = original.clone();
    changed.activation_mode = "different-scale-mode".into();
    variants.push(changed);
    let mut hardware = profile();
    hardware.multiprocessors += 1;
    variants.push(ResidentGemmKey::new(
        &hardware,
        13000,
        [64, 67, 272],
        original.driver_build.clone(),
        original.kernel_fingerprint.clone(),
    )?);
    let mut hardware = profile();
    hardware.l2_cache_bytes += 1;
    variants.push(ResidentGemmKey::new(
        &hardware,
        13000,
        [64, 67, 272],
        original.driver_build.clone(),
        original.kernel_fingerprint,
    )?);
    for variant in variants {
        assert_eq!(decision(Some(&table), &variant, false)?, Decision::Baseline);
        assert_eq!(decision(Some(&table), &variant, true)?, Decision::Measure);
    }
    Ok(())
}

#[test]
fn autotune_disabled_never_requests_measurement_but_keeps_valid_cached_choices() -> Result<()> {
    let key = key()?;
    assert_eq!(decision(None, &key, false)?, Decision::Baseline);
    assert_eq!(decision(None, &key, true)?, Decision::Measure);
    let mut table = TilingTable::default();
    table.resident_gemm.push(entry(key.clone()));
    assert!(matches!(
        decision(Some(&table), &key, false)?,
        Decision::Cached(_)
    ));
    table.resident_gemm[0].bitwise_verified = false;
    assert!(decision(Some(&table), &key, true).is_err());
    let expired = ResidentTuningSession::without_search_budget();
    let fresh = ResidentTuningSession::default();
    assert!(!expired.permits_gemm_search(
        SearchFamily::Nvfp4,
        [64, 128, 256],
        nvfp4_search_trials()
    )?);
    assert!(fresh.permits_gemm_search(
        SearchFamily::Nvfp4,
        [64, 128, 256],
        nvfp4_search_trials()
    )?);
    assert!(!expired.permits_norm_search());
    assert!(fresh.permits_norm_search());
    assert_eq!(
        ProgramGeometry::from_flags(3, 256, 64, false, false).widths(),
        vec![1, 64]
    );
    assert_eq!(
        ProgramGeometry::from_flags(3, 256, 64, true, true).widths(),
        vec![1, 3, 64, 256]
    );
    assert_eq!(
        ProgramGeometry::from_flags(0, 128, 0, true, true).widths(),
        vec![1, 128]
    );
    assert_eq!(
        ProgramGeometry::from_flags(3, 256, 0, false, true).widths(),
        vec![1, 256]
    );
    assert_eq!(
        ProgramGeometry::from_flags(3, 3, 0, true, true).widths(),
        vec![1, 3]
    );
    Ok(())
}

#[test]
fn resident_cache_rejects_incomplete_noisy_or_illegal_evidence() -> Result<()> {
    let mut valid = entry(key()?);
    assert!(valid.valid());
    valid.confirmation.warm.speedup = 1.019;
    assert!(!valid.valid());
    valid.confirmation.warm.speedup = 1.03;
    valid.confirmation.cold.candidate_ms = vec![];
    assert!(!valid.valid());
    valid.confirmation.cold = times(1.03);
    valid.selected.tile_n = 256;
    assert!(!valid.valid());
    valid.selected.tile_n = 128;
    valid.confirmation.cold.baseline_ms[0] = f32::NAN;
    assert!(!valid.valid());
    valid.confirmation.cold = times(1.03);
    valid.confirmation.cold.speedup = 1.04;
    assert!(!valid.valid());
    valid.confirmation.cold = times(1.03);
    valid.confirmation.cold.baseline_ms.fill(1.04);
    assert!(!valid.valid());
    valid.confirmation.cold = times(1.03);
    valid.confirmation.cold.candidate_ms.pop();
    assert!(!valid.valid());
    valid.confirmation.cold = times(1.03);
    valid.confirmation.cold.baseline_ms = vec![1.04, 1.03, 1.03, 1.02];
    valid.confirmation.cold.candidate_ms = vec![1.0; 4];
    assert!(valid.valid());
    let complete = entry(key()?);
    for (cold, warm) in [(1.019, 1.03), (1.03, 1.019)] {
        let mut search_failure = complete.clone();
        search_failure.trials[0].cold = times(cold);
        search_failure.trials[0].warm = times(warm);
        assert!(!search_failure.valid());
        let mut confirmation_failure = complete.clone();
        confirmation_failure.confirmation.cold = times(cold);
        confirmation_failure.confirmation.warm = times(warm);
        assert!(!confirmation_failure.valid());
    }
    let mut same_order = complete.clone();
    same_order.confirmation.order = MeasurementOrder::BaselineFirst;
    assert!(!same_order.valid());
    let mut missing_search = complete.clone();
    missing_search.trials.clear();
    assert!(!missing_search.valid());
    let mut legacy = serde_json::to_value(&complete).map_err(device_error)?;
    legacy["confirmation"]
        .as_object_mut()
        .ok_or_else(|| Error::invariant("trial fixture object"))?
        .remove("order");
    let legacy: ResidentGemmEntry = serde_json::from_value(legacy).map_err(device_error)?;
    assert!(!legacy.valid());
    // Timing records keep operator names even when the first measured arm is reversed.
    let ab = ordered_times(&[8.0, 10.0], &[2.0, 4.0], MeasurementOrder::BaselineFirst)?;
    let ba = ordered_times(&[2.0, 4.0], &[8.0, 10.0], MeasurementOrder::CandidateFirst)?;
    for evidence in [ab, ba] {
        assert_eq!(evidence.baseline_ms, vec![8.0, 10.0]);
        assert_eq!(evidence.candidate_ms, vec![2.0, 4.0]);
        assert_eq!(evidence.baseline_median_ms.to_bits(), 9.0_f32.to_bits());
        assert_eq!(evidence.candidate_median_ms.to_bits(), 3.0_f32.to_bits());
        assert_eq!(evidence.speedup.to_bits(), 3.0_f32.to_bits());
        assert!(times_valid(&evidence));
    }
    assert!(ordered_times(&[1.0], &[2.0], MeasurementOrder::Unspecified).is_err());
    Ok(())
}

#[test]
fn bitwise_gate_checks_all_outputs_lengths_finiteness_and_signed_zero() -> Result<()> {
    assert!(verify_bitwise(&[], &[]).is_err());
    let expected = [1.0, -2.5, 0.0];
    verify_bitwise(&expected, &expected)?;
    assert!(verify_bitwise(&expected[..2], &expected).is_err());
    assert!(verify_bitwise(&[1.0, -2.5, -0.0], &expected).is_err());
    assert!(verify_bitwise(&[1.0, -2.5, f32::INFINITY], &expected).is_err());
    assert!(verify_bitwise(&[1.0, f32::NAN, 0.0], &expected).is_err());
    assert!(
        verify_bitwise(
            &[f32::from_bits(1.0_f32.to_bits() + 1), -2.5, 0.0],
            &expected
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn scalar_and_resident_sections_round_trip_and_share_atomic_updates() -> Result<()> {
    let path = temp_dir().join(format!(
        "resident-tuning-test-{}-{}.json",
        id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(device_error)?
            .as_nanos()
    ));
    let mut scalar = TilingTable::default();
    scalar.devices.insert(
        "scalar-device".into(),
        vec![super::super::TuningEntry {
            dtype: "nvfp4".into(),
            rows: 67,
            columns: 272,
            tile_rows: 4,
            tile_columns: 128,
        }],
    );
    let encoded = serde_json::to_string(&scalar).map_err(device_error)?;
    write(&path, encoded).map_err(device_error)?;
    let resident = entry(key()?);
    assert!(super::super::update_table(Some(&path), |table| table
        .resident_gemm
        .push(resident.clone())));
    let loaded = TilingTable::load(&path)?;
    assert_eq!(loaded.devices.len(), 1);
    assert_eq!(loaded.resident_gemm.len(), 1);
    assert_eq!(loaded.resident_gemm[0].key, resident.key);
    assert_eq!(loaded.resident_gemm[0].selected, resident.selected);
    assert!(loaded.resident_gemm[0].valid());
    // A scalar update must retain the resident section, using the very same writer lock.
    assert!(super::super::update_table(Some(&path), |table| {
        table.devices.insert("second-scalar-device".into(), vec![]);
    }));
    let loaded = TilingTable::load(&path)?;
    assert_eq!(loaded.devices.len(), 2);
    assert_eq!(loaded.resident_gemm.len(), 1);
    assert_eq!(
        loaded.tiling_for("scalar-device", "nvfp4", 67, 272)?,
        Some(LinearTiling::new(4, 128)?)
    );
    remove_file(&path).map_err(device_error)?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; root runs release serially"]
fn resident_gemm_candidates_preserve_quantization_and_output_bits_on_tails() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for shape in [
        [1, 67, 272],
        [3, 65, 528],
        [17, 129, 256],
        [65, 67, 272],
        [129, 129, 528],
    ] {
        let [m, n, k] = shape;
        let packed = (0..n * k / 2)
            .map(|i| {
                f4e2m1fnx2::from_bits(u8::try_from((i * 13 + i / 9) % 256).unwrap_or_default())
            })
            .collect();
        let scales = (0..n * k / 16)
            .map(|i| f8e4m3fn(48 + u8::try_from(i % 16).unwrap_or_default()))
            .collect();
        let weight = ProjectionWeight::Fp4(
            device.upload(packed, &[n, k / 2])?,
            device.upload(scales, &[n, k / 16])?,
            0.75,
        );
        let base = Captured::new(&device, &weight, 1.25, shape, GemmConfig::baseline(m))?;
        base.run(&device)?;
        let expected = device.read(&base.output)?;
        for tile_m in TILE_M_SEARCH {
            for tile_n in TILE_N_SEARCH {
                for main_loop in LOOP_SEARCH {
                    let config = GemmConfig {
                        tile_m,
                        tile_n,
                        main_loop,
                    };
                    let candidate = Captured::new(&device, &weight, 1.25, shape, config)?;
                    candidate.run(&device)?;
                    verify_bitwise(&device.read(&candidate.output)?, &expected)?;
                }
            }
        }
        let choice = GemmConfig {
            tile_m: 64,
            tile_n: 128,
            main_loop: MainLoop::Plain,
        };
        let key = ResidentGemmKey::new(
            device.profile()?,
            CudaDevice::driver_version()?,
            shape,
            driver_build(),
            kernel_fingerprint(&device, shape)?,
        )?;
        memo()
            .lock()
            .map_err(|_| Error::invariant("GPU test tuning memo poisoned"))?
            .insert(
                key.clone(),
                MemoDecision {
                    config: choice,
                    covered: true,
                },
            );
        let expired = ResidentTuningSession::without_search_budget();
        assert!(!expired.permits_gemm_search(SearchFamily::Nvfp4, shape, nvfp4_search_trials())?);
        // Resolving a cached choice must still prove this previously unseen weight after
        // another projection has consumed the entire new-search allowance.
        assert_eq!(
            resolve(&device, &weight, 1.25, shape, true, &expired)?,
            choice
        );
        let ProjectionWeight::Fp4(codes, scales, _) = &weight else {
            return Err(Error::invariant("GPU test FP4 weight"));
        };
        let invalid_weight = ProjectionWeight::Fp4(Arc::clone(codes), Arc::clone(scales), 0.0);
        assert_eq!(
            verified_config(&device, &invalid_weight, 1.25, &key, choice)?,
            GemmConfig::baseline(m)
        );
        assert_eq!(
            verified_config(&device, &weight, 0.0, &key, choice)?,
            GemmConfig::baseline(m)
        );
        // A failed scalar/weight gate must leave the previously validated projection usable.
        assert_eq!(
            verified_config(&device, &weight, 1.25, &key, choice)?,
            choice
        );
    }
    Ok(())
}

#[test]
fn tile_m_128_is_accepted_and_enumerated_after_the_schema_bump() {
    // The schema is part of the table-cache key: without this bump a stored 64x128
    // selection keeps short-circuiting the search and the new row tile never runs.
    assert!(KERNEL_SCHEMA.contains(":tile-m128"));
    for tile_n in TILE_N_SEARCH {
        assert!(
            GemmConfig {
                tile_m: 128,
                tile_n,
                main_loop: MainLoop::Plain,
            }
            .valid()
        );
    }
    let mut candidates = Vec::new();
    for tile_m in TILE_M_SEARCH {
        for tile_n in TILE_N_SEARCH {
            for main_loop in LOOP_SEARCH {
                candidates.push(GemmConfig {
                    tile_m,
                    tile_n,
                    main_loop,
                });
            }
        }
    }
    assert!(candidates.iter().any(|config| {
        config.tile_m == 128
            && config.tile_n == QUANT_GEMM_TILE[1]
            && config.main_loop == MainLoop::Plain
    }));
    assert_ne!(candidates, [] as [GemmConfig; 0]);
}

#[test]
fn cache_memo_coverage_requires_proof_or_a_legal_baseline() {
    let rows = 64;
    let baseline = GemmConfig::baseline(rows);
    let candidate = GemmConfig {
        tile_m: 64,
        tile_n: 128,
        main_loop: MainLoop::Plain,
    };
    assert!(memo_covers(
        rows,
        MemoDecision {
            config: candidate,
            covered: true
        },
        candidate
    ));
    assert!(!memo_covers(
        rows,
        MemoDecision {
            config: candidate,
            covered: true
        },
        baseline
    ));
    assert!(memo_covers(
        rows,
        MemoDecision {
            config: baseline,
            covered: true
        },
        baseline
    ));
    assert!(!memo_covers(
        rows,
        MemoDecision {
            config: baseline,
            covered: false
        },
        baseline
    ));
}

#[test]
fn measured_compute_flops_uses_full_gemm_work() -> Result<()> {
    assert_eq!(measured_flops(1024, 5120, 5120, 2.0)?, 26_843_545_600_000);
    assert_eq!(measured_flops(1, 1, 1, 0.25)?, 8_000);
    assert_eq!(measured_flops(1, 1, 1, 0.125)?, 16_000);
    assert_eq!(measured_flops(1, 1, 1, 1.5)?, 1_333);
    assert!(measured_flops(1, 1, 1, 0.0).is_err());
    assert!(measured_flops(1, 1, 1, f32::NAN).is_err());
    assert!(measured_flops(1, 1, 1, f32::INFINITY).is_err());
    assert!(measured_flops(usize::MAX, usize::MAX, usize::MAX, 1.0).is_err());
    Ok(())
}
