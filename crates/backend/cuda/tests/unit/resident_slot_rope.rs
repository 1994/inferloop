use super::kernels;
use crate::constants::METADATA_FIELDS;
use crate::device::CudaDevice;
use crate::resident::kernels::aux;
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error;
use std::slice::from_ref;

type TestResult = Result<(), Box<dyn Error>>;

#[test]
fn pooled_rotary_kernel_compiles_without_a_device() -> TestResult {
    use cutile::cutile_compiler::compile_api::KernelCompiler;
    for (heads, dim, half) in [(4, 64, 16), (4, 256, 32), (24, 256, 32), (24, 256, 128)] {
        let stride = i32::try_from(dim)?;
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "rope")
            .target("sm_120")
            .generics(
                [dim, half, heads, METADATA_FIELDS]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .strides(&[
                ("out", &[stride, 1]),
                ("x", &[stride, 1]),
                ("frequencies", &[1]),
                ("metadata", &[1]),
            ])
            .compile()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn pooled_rotary_matches_per_lane_bits() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for slots in [2, 5, 16] {
        for (heads, dim, half) in [(4, 64, 16), (4, 256, 32), (24, 256, 32), (24, 256, 128)] {
            check_geometry(&device, slots * 3, heads, dim, half)?;
        }
    }
    Ok(())
}

fn check_geometry(
    device: &CudaDevice,
    lanes: usize,
    heads: usize,
    dim: usize,
    half: usize,
) -> TestResult {
    let input = device.upload(
        (0..lanes * heads * dim)
            .map(|i| (f32::from(u16::try_from(i % 131).unwrap()) - 65.0) / 37.0)
            .collect(),
        &[lanes * heads, dim],
    )?;
    let frequencies = device.upload(
        (0..half)
            .map(|i| {
                10000.0_f32.powf(
                    -f32::from(u16::try_from(i).unwrap()) / f32::from(u16::try_from(half).unwrap()),
                )
            })
            .collect(),
        &[half],
    )?;
    let metadata = device.upload(
        (0..lanes)
            .flat_map(|lane| {
                let position = [0, 1, 4095, 32767][lane % 4];
                [position, 0, if lane % 5 == 0 { -1 } else { position }, 0]
            })
            .collect(),
        &[lanes * METADATA_FIELDS],
    )?;
    let mut original = (0..lanes)
        .map(|_| api::zeros::<f32>(&[heads, dim]).sync_on(&device.stream))
        .collect::<Result<Vec<_>, _>>()?;
    let mut candidate = api::zeros::<f32>(&[lanes * heads, dim]).sync_on(&device.stream)?;
    let reference = Graph::scope(&device.stream, |scope| {
        for (lane, out) in original.iter_mut().enumerate() {
            let input_range = lane * heads..(lane + 1) * heads;
            let metadata_range = lane * METADATA_FIELDS..(lane + 1) * METADATA_FIELDS;
            scope.record(
                aux::rope(
                    out.partition([1, half]),
                    &input.slice(from_ref(&input_range))?.view(&[heads, dim])?,
                    &frequencies,
                    &metadata.slice(from_ref(&metadata_range))?,
                )
                .generics([dim, half].iter().map(ToString::to_string).collect()),
            )?;
        }
        Ok(())
    })?;
    let batched = Graph::scope(&device.stream, |scope| {
        scope.record(
            kernels::rope(
                (&mut candidate).partition([1, half]),
                &input,
                &frequencies,
                &metadata,
            )
            .generics(
                [dim, half, heads, METADATA_FIELDS]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            ),
        )?;
        Ok(())
    })?;
    reference.launch().sync_on(&device.stream)?;
    batched.launch().sync_on(&device.stream)?;
    let actual = device.read_borrowed(&candidate)?;
    for (lane, out) in original.iter().enumerate() {
        let expected = device.read_borrowed(out)?;
        for (column, (&a, &b)) in actual[lane * heads * dim..(lane + 1) * heads * dim]
            .iter()
            .zip(&expected)
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "lanes={lanes} heads={heads} dim={dim} half={half} lane={lane} column={column}: {a} != {b}"
            );
        }
    }
    Ok(())
}
