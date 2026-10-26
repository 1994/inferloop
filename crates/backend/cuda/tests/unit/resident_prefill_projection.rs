use super::{DENSE_REFERENCE, gemm, narrow_dense_layout, with_dense_reference};
use std::error::Error as StdError;
use std::panic::catch_unwind;

#[test]
fn narrow_dense_scope_matches_the_small_projection_structure() {
    for (rows, columns) in [(1, 5120), (32, 5120), (40, 4096), (48, 8192)] {
        assert!(narrow_dense_layout("sm_120", rows, columns));
    }
    for (rows, columns) in [(0, 5120), (64, 5120), (65, 5120), (48, 5121)] {
        assert!(!narrow_dense_layout("sm_120", rows, columns));
    }
    for architecture in ["sm_100", "sm_90", "sm_121", ""] {
        assert!(!narrow_dense_layout(architecture, 48, 5120));
    }
}

#[test]
fn dense_reference_restores_after_nested_scope_and_unwind() {
    assert!(!DENSE_REFERENCE.get());
    with_dense_reference(|| {
        assert!(DENSE_REFERENCE.get());
        with_dense_reference(|| assert!(DENSE_REFERENCE.get()));
        assert!(DENSE_REFERENCE.get());
    });
    assert!(!DENSE_REFERENCE.get());
    let failure = catch_unwind(|| with_dense_reference(|| panic!("reference unwind")));
    assert!(failure.is_err());
    assert!(!DENSE_REFERENCE.get());
}

#[test]
fn dense_n16_compiles_for_alternate_role_cardinalities() -> Result<(), Box<dyn StdError>> {
    use cutile::cutile_compiler::compile_api::KernelCompiler;
    for rows in [32_i32, 40, 48, 64] {
        KernelCompiler::new(gemm::__module_ast_self, "gemm", "dense_n16")
            .target("sm_120")
            .generics(vec!["5120".into()])
            .strides(&[
                ("out", &[rows, 1]),
                ("input", &[5120, 1]),
                ("weight", &[5120, 1]),
            ])
            .compile()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA; role-selected N16 numeric check, run inside safe-run"]
fn role_selected_n16_matches_reference_at_40_and_48_without_precision_change()
-> Result<(), Box<dyn StdError>> {
    use super::{GdnRole, records_narrow_dense};
    use crate::device::CudaDevice;
    use cutile::half::bf16;
    use cutile::prelude::*;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let profile = device.profile()?.clone();
    let k = 5120_usize;
    let width = 32_usize;
    let run =
        |rows: usize, input: &[f32], values: &[bf16]| -> Result<Vec<f32>, Box<dyn StdError>> {
            let x = device.upload(input.to_vec(), &[width, k])?;
            let weight = device.upload(values.to_vec(), &[rows, k])?;
            let mut output = api::zeros::<f32>(&[width, rows]).sync_on(&device.stream)?;
            let graph = CudaGraph::scope(&device.stream, |scope| {
                let selected = records_narrow_dense(
                    scope,
                    &profile,
                    Some(GdnRole::Beta),
                    &x.view(&[width, k])?,
                    &weight,
                    &mut output,
                    (width, rows, k),
                )?;
                assert!(selected);
                Ok(())
            })?;
            graph.launch().sync_on(&device.stream)?;
            Ok(device.read(&Arc::new(output))?)
        };
    // Exact dyadic operands: BF16 conversion and F32 partial sums are exact, so the
    // independent F64 reference must match bit-for-bit at both cardinalities.
    let exact_input: Vec<f32> = (0..width * k)
        .map(|i| match i % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        })
        .collect();
    let exact_weight: Vec<bf16> = (0..48 * k)
        .map(|i| bf16::from_f32((f32::from(u16::try_from(i % 5).unwrap_or_default()) - 2.0) / 4.0))
        .collect();
    for rows in [40_usize, 48] {
        let actual = run(rows, &exact_input, &exact_weight[..rows * k])?;
        for row in 0..width {
            for column in 0..rows {
                let expected: f64 = (0..k)
                    .map(|index| {
                        f64::from(exact_input[row * k + index])
                            * f64::from(f32::from(exact_weight[column * k + index]))
                    })
                    .sum();
                assert_eq!(
                    f64::from(actual[row * rows + column]),
                    expected,
                    "exact row{row} column{column}"
                );
            }
        }
    }
    // Adversarial regression: the alternate cardinality must reproduce the same N16
    // arithmetic as the 48-row control, without a test-only tolerance relaxation.
    let input: Vec<f32> = (0..width * k)
        .map(|i| (f32::from(u16::try_from(i % 251).unwrap_or_default()) - 125.0) / 97.0)
        .collect();
    let values: Vec<bf16> = (0..48 * k)
        .map(|i| {
            bf16::from_f32((f32::from(u16::try_from(i % 127).unwrap_or_default()) - 63.0) / 41.0)
        })
        .collect();
    let control = run(48, &input, &values)?;
    let alternate = run(40, &input, &values[..40 * k])?;
    for row in 0..width {
        for column in 0..40 {
            assert_eq!(
                alternate[row * 40 + column].to_bits(),
                control[row * 48 + column].to_bits(),
                "adversarial row{row} column{column}"
            );
        }
    }
    // A non-role caller must not take the N16 path.
    let x = device.upload(input, &[width, k])?;
    let weight = device.upload(values, &[48, k])?;
    let mut untouched = api::zeros::<f32>(&[width, 48]).sync_on(&device.stream)?;
    let decoy = CudaGraph::scope(&device.stream, |scope| {
        let selected = records_narrow_dense(
            scope,
            &profile,
            None,
            &x.view(&[width, k])?,
            &weight,
            &mut untouched,
            (width, 48, k),
        )?;
        assert!(!selected);
        Ok(())
    })?;
    decoy.launch().sync_on(&device.stream)?;
    Ok(())
}
