use super::DenseAttentionPlan;
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::Result;
use infer_cuda_kernels::tile::runtime::Graph;
use infer_kernel_api::attention::{AttentionDescriptor, AttentionMask};

fn descriptor() -> AttentionDescriptor {
    AttentionDescriptor {
        queries: 2,
        keys: 3,
        query_heads: 8,
        kv_heads: 2,
        qk_dim: 48,
        value_dim: 80,
        scale: 0.125,
        dtype: infer_ir::DType::F32,
        mask: AttentionMask::None,
    }
}

#[test]
fn provider_limits_do_not_restrict_the_semantic_contract() -> Result<()> {
    let mut spec = descriptor();
    spec.dtype = infer_ir::DType::Bf16;
    spec.validate()?;
    assert!(DenseAttentionPlan::new(spec).is_err());
    spec = descriptor();
    spec.qk_dim = 512;
    spec.validate()?;
    assert!(DenseAttentionPlan::new(spec).is_err());
    Ok(())
}

#[test]
fn dense_plan_checks_device_index_and_mask_overflow() {
    let mut spec = descriptor();
    spec.keys = usize::try_from(i32::MAX).unwrap();
    assert!(DenseAttentionPlan::new(spec).is_err());
    spec = descriptor();
    spec.mask = AttentionMask::Window {
        query_start: i64::from(i32::MAX),
        left: 0,
        right: 1,
    };
    assert!(DenseAttentionPlan::new(spec).is_err());
}

#[test]
#[ignore = "requires NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn dense_attention_supports_distinct_qk_and_value_dimensions() -> Result<()> {
    let spec = descriptor();
    let plan = DenseAttentionPlan::new(spec)?;
    let shapes = plan.shapes();
    let device = CudaDevice::new(0)?;
    let q = device.upload(vec![0.0f32; shapes[0][0] * shapes[0][1]], &shapes[0])?;
    let k = device.upload(vec![0.0f32; shapes[1][0] * shapes[1][1]], &shapes[1])?;
    let width = shapes[2][1] / spec.kv_heads;
    let mut values = vec![0.0f32; shapes[2][0] * shapes[2][1]];
    for token in 0..spec.keys {
        for head in 0..spec.kv_heads {
            for lane in 0..spec.value_dim {
                values[token * shapes[2][1] + head * width + lane] =
                    f32::from(u16::try_from(token + head + lane).map_err(device_error)?);
            }
        }
    }
    let v = device.upload(values, &shapes[2])?;
    let mut out = api::zeros::<f32>(&shapes[3])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = Graph::scope(&device.stream, |scope| {
        plan.record(scope, &q, &k, &v, &mut out)
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let actual = device.read(&Arc::new(out))?;
    for token in 0..shapes[3][0] {
        for head in 0..spec.query_heads {
            for lane in 0..width {
                let expected = if token < spec.queries && lane < spec.value_dim {
                    f32::from(
                        u16::try_from(1 + head / (spec.query_heads / spec.kv_heads) + lane)
                            .map_err(device_error)?,
                    )
                } else {
                    0.0
                };
                let observed = actual[token * shapes[3][1] + head * width + lane];
                assert!(observed.is_finite() && (observed - expected).abs() < 1e-3);
            }
        }
    }
    Ok(())
}
