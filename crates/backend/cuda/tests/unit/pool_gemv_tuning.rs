use super::{Entry, Key, legacy_choice};
use crate::constants::CB_DECODE_SLOTS;
use crate::resident::pool_gemv::kernels;
use crate::resident::small_gemm::kernels as blocked_kernels;
use crate::tuning::pool_gemv::Choice;
use cutile::DType;
use cutile::cutile_compiler::compile_api::KernelCompiler;
use cutile::half::bf16;
use std::error::Error;

const DEVICE: &str = "dev|sm_120|24576MiB";
const DRIVER_BUILD: &str = "NVRM version: test";
const HARDWARE: &str = "hardware";
const FINGERPRINT: &str = "fp";
const PROOF: &str = "abc";

fn key() -> Key<'static> {
    Key {
        device: DEVICE,
        driver: 12_000,
        driver_build: DRIVER_BUILD,
        hardware: HARDWARE,
        fingerprint: FINGERPRINT,
        rows: 16,
        columns: 5120,
        outputs: 128,
        proof: PROOF,
    }
}

fn entry() -> Entry {
    Entry {
        device: DEVICE.into(),
        driver: 12_000,
        driver_build: DRIVER_BUILD.into(),
        hardware: HARDWARE.into(),
        kernel: concat!("pool-gemv-vs-blocked-v1/", env!("CARGO_PKG_VERSION")).into(),
        kernel_fingerprint: FINGERPRINT.into(),
        rows: 16,
        columns: 5120,
        outputs: 128,
        weight_proof: PROOF.into(),
        choice: "gemv".into(),
        exact: false,
        cold_gemv_us: 10.0,
        cold_blocked_us: 12.0,
        warm_gemv_us: 8.0,
        warm_blocked_us: 10.0,
        cold_ab_blocked_samples: vec![0.012, 0.012],
        cold_ab_gemv_samples: vec![0.010, 0.010],
        cold_ba_blocked_samples: vec![0.012, 0.012],
        cold_ba_gemv_samples: vec![0.010, 0.010],
        warm_ab_blocked_samples: vec![0.010, 0.010],
        warm_ab_gemv_samples: vec![0.008, 0.008],
        warm_ba_blocked_samples: vec![0.010, 0.010],
        warm_ba_gemv_samples: vec![0.008, 0.008],
        gemv_faster: true,
    }
}

fn matches(entry: &Entry) -> bool {
    entry.matches(&key()) && entry.driver_build == DRIVER_BUILD && entry.hardware == HARDWARE
}

#[test]
fn legacy_fallback_remains_the_slot_width_bound() {
    assert_eq!(legacy_choice(1), Choice::Gemv);
    assert_eq!(legacy_choice(CB_DECODE_SLOTS), Choice::Gemv);
    assert_eq!(legacy_choice(CB_DECODE_SLOTS + 1), Choice::Blocked);
}

#[test]
fn pool_entry_rejects_stale_or_nonfinite_evidence() {
    let entry = entry();
    assert!(matches(&entry));
    assert!(entry.geometry_matches(&key()));
    assert!(entry.geometry_matches(&Key {
        proof: "other",
        ..key()
    }));
    for key in [
        Key {
            device: "other",
            ..key()
        },
        Key {
            driver: 12_001,
            ..key()
        },
        Key {
            proof: "other",
            ..key()
        },
        Key { rows: 17, ..key() },
        Key {
            fingerprint: "other",
            ..key()
        },
    ] {
        assert!(!entry.matches(&key));
    }
    let mut bad = entry.clone();
    bad.driver_build.clear();
    assert!(!matches(&bad));
    let mut bad = entry.clone();
    bad.hardware.clear();
    assert!(!matches(&bad));
    let mut bad = entry.clone();
    bad.choice = "other".into();
    assert!(!matches(&bad));
    let mut bad = entry.clone();
    bad.cold_gemv_us = f64::NAN;
    assert!(!matches(&bad));
    let mut bad = entry.clone();
    bad.cold_ab_gemv_samples = vec![0.020, 0.020];
    assert!(!matches(&bad));
    let mut bad = entry;
    bad.choice = "blocked".into();
    bad.gemv_faster = false;
    assert!(!matches(&bad));
}

#[test]
fn gemv_and_blocked_kernels_compile_with_legal_cta_strides() -> Result<(), Box<dyn Error>> {
    for (bn, bk) in [(16, 256), (32, 256)] {
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "dense")
            .target("sm_120")
            .generics(vec![
                bf16::DTYPE.as_str().into(),
                bn.to_string(),
                bk.to_string(),
                5120.to_string(),
            ])
            .strides(&[
                ("out", &[bn, 1]),
                ("input", &[5120, 1]),
                ("weight", &[5120, 1]),
            ])
            .compile()?;
    }
    KernelCompiler::new(blocked_kernels::__module_ast_self, "kernels", "dense")
        .target("sm_120")
        .generics(vec![bf16::DTYPE.as_str().into(), "5120".into()])
        .strides(&[
            ("out", &[32, 1]),
            ("input", &[5120, 1]),
            ("weight", &[5120, 1]),
        ])
        .compile()?;
    Ok(())
}
