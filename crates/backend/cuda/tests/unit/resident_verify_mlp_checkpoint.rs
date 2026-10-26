//! Full observable MLP outputs against the original padded projection path.
use super::{GateUpMode, MlpPlan, ProgramWeights};
use crate::device::CudaDevice;
use crate::loading::{LoadOptions, LoadedModel};
use crate::mlp::ProjectionWeight;
use crate::resident::ProgramGeometry;
use crate::resident::nvfp4_gemm::{self, Workspace};
use crate::resident::program::ActivationQuantization;
use cutile::prelude::*;
use infer_core::ModelId;
use serde_json::json;
use std::env::var;
use std::error::Error as StdError;
struct Mlp {
    gate: ProjectionWeight,
    up: ProjectionWeight,
    down: ProjectionWeight,
    input_scale: f32,
    down_scale: f32,
    columns: usize,
    hidden: usize,
    outputs: usize,
}
fn fp4(
    weights: &ProgramWeights,
    id: infer_core::TensorId,
) -> Result<ProjectionWeight, Box<dyn StdError>> {
    let Some(ProjectionWeight::Fp4(c, s, g)) = weights.projections.get(&id) else {
        return Err("not FP4".into());
    };
    Ok(ProjectionWeight::Fp4(c.clone(), s.clone(), *g))
}
fn triples(
    loaded: &LoadedModel,
    ws: &Workspace,
    rows: usize,
) -> Result<Vec<Mlp>, Box<dyn StdError>> {
    let plan = MlpPlan::for_verify(loaded.graph(), loaded.weights(), (rows, ws));
    let mut result = Vec::new();
    for &(g, u, d) in plan.fused.values() {
        let n = &loaded.graph().nodes;
        let (gate, up, down) = (
            fp4(loaded.weights(), n[g].inputs[1])?,
            fp4(loaded.weights(), n[u].inputs[1])?,
            fp4(loaded.weights(), n[d].inputs[1])?,
        );
        let (ProjectionWeight::Fp4(gc, ..), ProjectionWeight::Fp4(dc, ..)) = (&gate, &down) else {
            return Err("not FP4".into());
        };
        let input_scale = *loaded
            .weights()
            .input_scales
            .get(&n[g].inputs[1])
            .ok_or("input scale")?;
        let down_scale = *loaded
            .weights()
            .input_scales
            .get(&n[d].inputs[1])
            .ok_or("down scale")?;
        let (columns, hidden, outputs) = (
            usize::try_from(gc.shape()[1])? * 2,
            usize::try_from(gc.shape()[0])?,
            usize::try_from(dc.shape()[0])?,
        );
        result.push(Mlp {
            gate,
            up,
            down,
            input_scale,
            down_scale,
            columns,
            hidden,
            outputs,
        });
    }
    assert!(
        !result.is_empty(),
        "no structural FP4 MLPs with equal gate/up input scales"
    );
    Ok(result)
}
struct Temps {
    gate: Tensor<f32>,
    up: Tensor<f32>,
}
fn compare(
    device: &CudaDevice,
    a: &[Tensor<f32>],
    b: &[Tensor<f32>],
) -> Result<usize, Box<dyn StdError>> {
    let mut count = 0;
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        let a = device.read_borrowed(a)?;
        let b = device.read_borrowed(b)?;
        assert!(a.len() >= b.len());
        for (i, (a, b)) in a.iter().take(b.len()).zip(&b).enumerate() {
            assert!(a.is_finite() && b.is_finite());
            assert_eq!(a.to_bits(), b.to_bits(), "full MLP bitwise mismatch {i}");
        }
        count += b.len();
    }
    Ok(count)
}
fn capture(
    device: &CudaDevice,
    ws: &mut Workspace,
    mlps: &[Mlp],
    input: &Tensor<f32>,
    outputs: &mut [Tensor<f32>],
    tmp: &mut Temps,
    fused: bool,
) -> Result<CudaGraph<()>, DeviceError> {
    let rows = usize::try_from(input.shape()[0]).unwrap();
    let columns = usize::try_from(input.shape()[1]).unwrap();
    let view = input.view(&[rows, columns])?;
    ws.begin_quant_capture();
    let graph = CudaGraph::scope(&device.stream, |scope| {
        for (mlp, out) in mlps.iter().zip(outputs) {
            if fused {
                ws.record_gate_up_swiglu(
                    scope,
                    (&mlp.gate, &mlp.up),
                    (mlp.input_scale, mlp.down_scale),
                    &view,
                    (GateUpMode::Verify, rows, mlp.columns, mlp.hidden),
                )?;
                ws.record_prequantized(scope, &mlp.down, mlp.down_scale, out, mlp.hidden)?;
            } else {
                assert!(nvfp4_gemm::record(
                    scope,
                    ws,
                    &mlp.gate,
                    Some(ActivationQuantization::Fp4(mlp.input_scale)),
                    &view,
                    &mut tmp.gate,
                    mlp.columns
                )?);
                assert!(nvfp4_gemm::record(
                    scope,
                    ws,
                    &mlp.up,
                    Some(ActivationQuantization::Fp4(mlp.input_scale)),
                    &view,
                    &mut tmp.up,
                    mlp.columns
                )?);
                let gate = tmp.gate.view(&[rows, mlp.hidden])?;
                let up = tmp.up.view(&[rows, mlp.hidden])?;
                ws.record_swiglu_input(
                    scope,
                    &mlp.down,
                    mlp.down_scale,
                    &gate,
                    &up,
                    out,
                    mlp.hidden,
                )?;
            }
        }
        Ok(())
    });
    ws.end_quant_capture();
    graph
}
#[test]
#[ignore = "real CUDA checkpoint; all eligible MLPs and partial row tile vs padded original"]
fn verify_mlp_matches_padded_original() -> Result<(), Box<dyn StdError>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let loaded = LoadedModel::open(
        device.clone(),
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            autotune: true,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    for rows in [48, 45] {
        let baseline_rows = 48;
        let mut old = Workspace::program(
            &device,
            loaded.graph(),
            loaded.weights(),
            ProgramGeometry {
                verification: baseline_rows,
                prompts: [0, 0],
            },
        )?;
        let mut new = Workspace::program(
            &device,
            loaded.graph(),
            loaded.weights(),
            ProgramGeometry {
                verification: rows,
                prompts: [0, 0],
            },
        )?;
        let mlps = triples(&loaded, &new, rows)?;
        assert_eq!(mlps.len(), 56);
        let (columns, hidden) = (mlps[0].columns, mlps[0].hidden);
        assert!(
            mlps.iter()
                .all(|m| m.columns == columns && m.hidden == hidden)
        );
        let seed = |count| {
            (0..count * columns)
                .map(|i| (f32::from(u16::try_from(i % 211).unwrap()) - 105.0) / 37.0)
                .collect::<Vec<_>>()
        };
        let original = device.upload(seed(baseline_rows), &[baseline_rows, columns])?;
        let input = device.upload(seed(rows), &[rows, columns])?;
        let mut tmp = Temps {
            gate: api::zeros::<f32>(&[baseline_rows, hidden]).sync_on(&device.stream)?,
            up: api::zeros::<f32>(&[baseline_rows, hidden]).sync_on(&device.stream)?,
        };
        let mut a = Vec::new();
        let mut b = Vec::new();
        for m in &mlps {
            a.push(api::zeros::<f32>(&[baseline_rows, m.outputs]).sync_on(&device.stream)?);
            b.push(api::zeros::<f32>(&[rows, m.outputs]).sync_on(&device.stream)?);
        }
        let ga = capture(&device, &mut old, &mlps, &original, &mut a, &mut tmp, false)?;
        let gb = capture(&device, &mut new, &mlps, &input, &mut b, &mut tmp, true)?;
        ga.launch().sync_on(&device.stream)?;
        gb.launch().sync_on(&device.stream)?;
        let elements = compare(&device, &a, &b)?;
        println!(
            "{}",
            json!({"verify_mlp_padded_original":true,"rows":rows,"baseline_rows":baseline_rows,"mlps":mlps.len(),"full_finite_elements":elements,"new_scratch_bytes":0})
        );
    }
    Ok(())
}
