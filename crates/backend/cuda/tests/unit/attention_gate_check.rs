//! The performance gate exercises the production tile selection against an independent oracle.
use super::DenseAttentionPlan;
use crate::device::{CudaDevice, device_error};
use crate::vision::program::{AttentionMode, attention_kernel, attention_tiles};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Graph;
use infer_kernel_api::attention::{AttentionDescriptor, AttentionMask};
use serde::Deserialize;
use std::env::var;
use std::fs::read as fs_read;
use std::path::Path;

#[derive(Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}

#[derive(Deserialize, serde::Serialize)]
struct Case {
    id: String,
    tokens: usize,
    heads: usize,
    kv_tokens: usize,
    kv_heads: usize,
    mask: String,
    query_start: i64,
    left: usize,
    right: usize,
    rope: bool,
    head_dim: usize,
    frame_tokens: usize,
    elements: usize,
}

#[test]
#[ignore = "requires GPU and INFER_ATTENTION_GATE; use make check-attention"]
fn attention_native_gate() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::invalid("attention gate requires release"));
    }
    let path = var("INFER_ATTENTION_GATE")
        .map_err(|_| Error::invalid("INFER_ATTENTION_GATE safetensors path required"))?;
    let manifest_path = Path::new(&path).with_extension("json");
    let bytes = fs_read(manifest_path).map_err(device_error)?;
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(device_error)?;
    let mut file = infer_models::SafetensorsFile::open(path)?;
    let device = CudaDevice::new(0)?;
    for case in manifest.cases {
        run(&device, &mut file, &case)?;
    }
    Ok(())
}

fn run(device: &CudaDevice, file: &mut infer_models::SafetensorsFile, case: &Case) -> Result<()> {
    let mut read = |name| {
        file.read_f32(&format!("{}/{name}", case.id), 1 << 30)
            .map(|tensor| tensor.data)
    };
    let q = read("q")?;
    let k = read("k")?;
    let v = read("v")?;
    let cos = read("cos")?;
    let sin = read("sin")?;
    let expected = read("output")?;
    let half = (case.head_dim / 2).next_power_of_two();
    let actual = if case.rope {
        attention_kernel(
            device,
            &q,
            &k,
            &v,
            &cos,
            &sin,
            case.tokens,
            case.heads,
            half,
            AttentionMode::Online,
            case.frame_tokens,
            case.head_dim,
            attention_tiles(case.tokens),
            true,
        )?
    } else {
        dense(device, case, &q, &k, &v)?
    };
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(&expected).all(|x| x.is_finite()));
    let mut error = 0.0f32;
    let mut scale = 0.0f32;
    for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
        if index % half < case.head_dim / 2 {
            error = error.max((a - b).abs());
            scale = scale.max(b.abs());
        } else {
            assert!(a.abs() <= 1e-6, "nonzero padding in {}", case.id);
        }
    }
    println!(
        "ATTENTION_GATE {}",
        serde_json::json!({
            "case": case.id, "shape": case, "warmup": 100,
            "max_abs_error": error, "reference_max_abs": scale,
            "elements": case.elements, "finite": true, "replays_per_sample": super::benchmark::REPEATS,
        })
    );
    assert!(
        error <= 1e-4f32.mul_add(scale, 1e-6),
        "{}: error {error}, scale {scale}",
        case.id
    );
    Ok(())
}

fn dense(device: &CudaDevice, case: &Case, q: &[f32], k: &[f32], v: &[f32]) -> Result<Vec<f32>> {
    let mask = match case.mask.as_str() {
        "none" => AttentionMask::None,
        "causal" => AttentionMask::Causal {
            query_start: case.query_start,
        },
        "window" => AttentionMask::Window {
            query_start: case.query_start,
            left: case.left,
            right: case.right,
        },
        "segments" => AttentionMask::Segments {
            tokens: case.frame_tokens,
        },
        _ => return Err(Error::invalid("unknown fixture mask")),
    };
    let dim = u16::try_from(case.head_dim).map_err(device_error)?;
    let plan = DenseAttentionPlan::new(AttentionDescriptor {
        queries: case.tokens,
        keys: case.kv_tokens,
        query_heads: case.heads,
        kv_heads: case.kv_heads,
        qk_dim: case.head_dim,
        value_dim: case.head_dim,
        scale: 1.0 / f32::from(dim).sqrt(),
        dtype: infer_ir::DType::F32,
        mask,
    })?;
    let shapes = plan.shapes();
    let upload = |data: &[f32], shape: [usize; 2]| {
        let mut padded = vec![0.0f32; shape[0] * shape[1]];
        padded[..data.len()].copy_from_slice(data);
        device.upload(padded, &shape)
    };
    let q = upload(q, shapes[0])?;
    let k = upload(k, shapes[1])?;
    let v = upload(v, shapes[2])?;
    let mut out = api::zeros::<f32>(&shapes[3])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = Graph::scope(&device.stream, |scope| {
        for _ in 0..super::benchmark::REPEATS {
            plan.record(scope, &q, &k, &v, &mut out)?;
        }
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    super::benchmark::measure(
        device,
        &graph,
        i32::try_from(case.tokens).map_err(device_error)?,
        case.heads,
        case.head_dim,
        [plan.tile(), plan.tile(), 0],
    )?;
    let output = device.read(&Arc::new(out))?;
    Ok(output[..case.tokens * shapes[3][1]].to_vec())
}
