use super::{Entry, legacy_choice};
use crate::constants::CB_DECODE_SLOTS;
use crate::resident::pool_gemv::kernels;
use crate::resident::small_gemm::kernels as blocked_kernels;
use crate::tuning::pool_gemv::Choice;
use cutile::DType;
use cutile::cutile_compiler::compile_api::KernelCompiler;
use cutile::half::bf16;
use std::error::Error;

#[test]
fn legacy_fallback_remains_the_slot_width_bound() {
    assert_eq!(legacy_choice(1), Choice::Gemv);
    assert_eq!(legacy_choice(CB_DECODE_SLOTS), Choice::Gemv);
    assert_eq!(legacy_choice(CB_DECODE_SLOTS + 1), Choice::Blocked);
}

#[test]
fn pool_entry_rejects_stale_or_nonfinite_evidence() {
    let entry = Entry {
        device: "dev|sm_120|24576MiB".into(),
        driver: 12_000,
        kernel: concat!("pool-gemv-vs-blocked-v1/", env!("CARGO_PKG_VERSION")).into(),
        rows: 16,
        columns: 5120,
        outputs: 128,
        weight_proof: "abc".into(),
        choice: "gemv".into(),
        exact: false,
        cold_gemv_us: 10.0,
        cold_blocked_us: 12.0,
        warm_gemv_us: 8.0,
        warm_blocked_us: 10.0,
        gemv_faster: true,
    };
    assert!(entry.matches("dev|sm_120|24576MiB", 12_000, 16, 5120, 128, "abc"));
    assert!(!entry.matches("other", 12_000, 16, 5120, 128, "abc"));
    assert!(!entry.matches("dev|sm_120|24576MiB", 12_001, 16, 5120, 128, "abc"));
    assert!(!entry.matches("dev|sm_120|24576MiB", 12_000, 16, 5120, 128, "other"));
    assert!(!entry.matches("dev|sm_120|24576MiB", 12_000, 17, 5120, 128, "abc"));
    let mut bad = entry;
    bad.cold_gemv_us = f64::NAN;
    assert!(!bad.matches("dev|sm_120|24576MiB", 12_000, 16, 5120, 128, "abc"));
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
