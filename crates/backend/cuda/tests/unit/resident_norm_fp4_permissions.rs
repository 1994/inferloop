//! Capture-boundary invalidation and exact buffer binding, including the error path.
use super::{CaptureQuantReuse, Workspace, vendor_fp8};
use crate::resident::norm_fp4;
use crate::tuning::resident_gemm::ResidentTuningSession;
use crate::{device::CudaDevice, mlp::ProjectionWeight};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::collections::BTreeMap;
use std::error::Error as StdError;
type TestResult<T = ()> = Result<T, Box<dyn StdError>>;

fn workspace(device: &CudaDevice) -> Workspace {
    let vendor_fp8 = vendor_fp8::Support::new_with(device, true);
    let vendor_fp4 = vendor_fp8::nvfp4::Support::new(
        &vendor_fp8,
        &Arc::new(ResidentTuningSession::default()),
        false,
    );
    Workspace {
        quant_reuse: CaptureQuantReuse::default(),
        norm_maximum: None,
        norm_fp4: None,
        strategies: BTreeMap::new(),
        swiglu_strategies: BTreeMap::new(),
        pool_dense_strategies: BTreeMap::new(),
        prompt_dense_strategies: BTreeMap::new(),
        buffers: BTreeMap::new(),
        fp8: BTreeMap::new(),
        fp8_strategies: BTreeMap::new(),
        vendor_fp8,
        vendor_fp4,
        vendor_diagnostic: false,
    }
}

fn invalidate(workspace: &mut Workspace, action: &str) {
    match action {
        "begin" => workspace.begin_quant_capture(),
        "end" => workspace.end_quant_capture(),
        "clear" => workspace.clear_quant_reuse(),
        "skip" => workspace.enter_quant_node(2, None),
        "valid" | "error" => {}
        _ => {
            let binding = &mut workspace.norm_fp4.as_mut().unwrap().binding;
            match action {
                "pointer" => binding.pointer ^= 1,
                "rows" => binding.rows += 1,
                "columns" => binding.columns += 16,
                "scale" => binding.scale_bits ^= 1,
                "codes" => binding.codes ^= 1,
                "scales" => binding.scales ^= 1,
                _ => panic!("unknown fixture"),
            }
        }
    }
    workspace.enter_quant_node(1, None);
}

fn run_case(device: &CudaDevice, action: &str) -> TestResult<(Vec<u8>, Vec<u8>)> {
    let (rows, columns, hidden) = (64, 512, 64);
    let input = device.upload(vec![0.5f32; rows * columns], &[rows, columns])?;
    let weight = device.upload(vec![1.0f32; columns], &[columns])?;
    let mut dense = api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?;
    let mut workspace = workspace(device);
    for columns in [columns, hidden] {
        workspace.buffers.insert(
            (rows, columns),
            (
                api::zeros::<f4e2m1fnx2>(&[rows, columns / 2]).sync_on(&device.stream)?,
                api::zeros::<f8e4m3fn>(&[rows, columns / 16]).sync_on(&device.stream)?,
            ),
        );
    }
    let projection = ProjectionWeight::Fp4(
        device.upload(
            vec![f4e2m1fnx2::from_bits(0x21); hidden * columns / 2],
            &[hidden, columns / 2],
        )?,
        device.upload(
            vec![f8e4m3fn(56); hidden * columns / 16],
            &[hidden, columns / 16],
        )?,
        1.0,
    );
    let before = norm_fp4::consumptions();
    workspace.begin_quant_capture();
    let graph = Graph::scope(&device.stream, |scope| {
        workspace.record_norm_fp4(
            scope,
            1,
            &input.view(&[rows, columns])?,
            &weight,
            &mut dense,
            (1e-5, 1.0, 1.0),
        )?;
        assert!(workspace.norm_fp4.is_some());
        invalidate(&mut workspace, action);
        let pointer = dense.device_pointer().cu_deviceptr();
        let view = dense.view(&[rows, columns])?;
        if action == "error" {
            assert!(
                workspace
                    .record_gate_up_swiglu(
                        scope,
                        (&projection, &projection),
                        (1.0, 1.0),
                        (&view, pointer),
                        (rows - 1, columns, hidden)
                    )
                    .is_err()
            );
            assert!(workspace.norm_fp4.is_none());
        }
        workspace.record_gate_up_swiglu(
            scope,
            (&projection, &projection),
            (1.0, 1.0),
            (&view, pointer),
            (rows, columns, hidden),
        )?;
        assert!(workspace.norm_fp4.is_none());
        assert_eq!(
            norm_fp4::consumptions() - before,
            usize::from(action == "valid")
        );
        workspace.record_gate_up_swiglu(
            scope,
            (&projection, &projection),
            (1.0, 1.0),
            (&view, pointer),
            (rows, columns, hidden),
        )?;
        assert_eq!(
            norm_fp4::consumptions() - before,
            usize::from(action == "valid")
        );
        Ok(())
    })?;
    workspace.end_quant_capture();
    graph.launch().sync_on(&device.stream)?;
    let (codes, scales) = workspace.quantized_fp4(rows, hidden).unwrap();
    Ok((
        device
            .read_borrowed(codes)?
            .into_iter()
            .map(f4e2m1fnx2::to_bits)
            .collect(),
        device
            .read_borrowed(scales)?
            .into_iter()
            .map(|v| v.0)
            .collect(),
    ))
}

#[test]
#[ignore = "CUDA release and protected runner"]
fn norm_fp4_capture_permissions_expire_and_match_all_bindings() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let expected = run_case(&device, "clear")?;
    assert_eq!(expected.0.len(), 64 * 64 / 2);
    assert_eq!(expected.1.len(), 64 * 64 / 16);
    for action in [
        "valid", "begin", "end", "skip", "pointer", "rows", "columns", "scale", "codes", "scales",
        "error",
    ] {
        assert_eq!(run_case(&device, action)?, expected, "{action} output");
    }
    Ok(())
}
