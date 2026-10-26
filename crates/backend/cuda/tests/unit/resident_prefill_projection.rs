use super::{DENSE_REFERENCE, gemm, narrow_dense_layout, with_dense_reference};
use std::error::Error;
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
fn dense_n16_compiles_for_alternate_role_cardinalities() -> Result<(), Box<dyn Error>> {
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
fn role_selected_n16_matches_independent_f32_reference_at_40_and_48() -> Result<(), Box<dyn Error>>
{
    use super::{GdnRole, records_narrow_dense};
    use crate::device::CudaDevice;
    use cutile::half::bf16;
    use cutile::prelude::*;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let profile = device.profile()?.clone();
    let k = 5120_usize;
    let width = 32_usize;
    let input: Vec<f32> = (0..width * k)
        .map(|i| (f32::from(u16::try_from(i % 251).unwrap_or_default()) - 125.0) / 97.0)
        .collect();
    let x = device.upload(input.clone(), &[width, k])?;
    for rows in [40_usize, 48] {
        let values: Vec<bf16> = (0..rows * k)
            .map(|i| {
                bf16::from_f32(
                    (f32::from(u16::try_from(i % 127).unwrap_or_default()) - 63.0) / 41.0,
                )
            })
            .collect();
        let weight = device.upload(values.clone(), &[rows, k])?;
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
        let actual = device.read(&Arc::new(output))?;
        for row in 0..width {
            for column in 0..rows {
                let mut expected = 0.0_f64;
                for index in 0..k {
                    let xv = f32::from(bf16::from_f32(input[row * k + index]));
                    let wv = f32::from(values[column * k + index]);
                    expected = f64::mul_add(f64::from(xv), f64::from(wv), expected);
                }
                let got = f64::from(actual[row * rows + column]);
                assert!(
                    (got - expected).abs() <= expected.abs().max(1.0) * 2.0e-3,
                    "row{row} column{column} got {got} expected {expected}"
                );
            }
        }
        // A non-role caller must not take the N16 path.
        let mut untouched = api::zeros::<f32>(&[width, rows]).sync_on(&device.stream)?;
        let decoy = CudaGraph::scope(&device.stream, |scope| {
            let selected = records_narrow_dense(
                scope,
                &profile,
                None,
                &x.view(&[width, k])?,
                &weight,
                &mut untouched,
                (width, rows, k),
            )?;
            assert!(!selected);
            Ok(())
        })?;
        decoy.launch().sync_on(&device.stream)?;
    }
    Ok(())
}
