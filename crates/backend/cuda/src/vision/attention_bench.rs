//! Reproducible device timings against independent SDPA fixture outputs.
use super::program::{AttentionMode, attention_kernel};
use crate::device::{CudaDevice, device_error};
use cutile::jit_cache::enable_default;
use infer_core::{Error, Result};
use std::env::var;

#[test]
#[ignore = "requires GPU and INFER_ATTENTION_BENCH fixtures; run under tools/bench/safe-run.sh"]
fn attention_benchmark_against_sdpa() -> Result<()> {
    let path = var("INFER_ATTENTION_BENCH")
        .map_err(|_| Error::invalid("INFER_ATTENTION_BENCH safetensors path required"))?;
    let mut file = infer_models::SafetensorsFile::open(path)?;
    enable_default().map_err(device_error)?;
    let device = CudaDevice::new(0)?;
    for index in 0..6 {
        let prefix = format!("c{index}");
        let q = file.read_f32(&format!("{prefix}/q"), 1 << 30)?;
        let k = file.read_f32(&format!("{prefix}/k"), 1 << 30)?.data;
        let v = file.read_f32(&format!("{prefix}/v"), 1 << 30)?.data;
        let cos = file.read_f32(&format!("{prefix}/cos"), 1 << 30)?.data;
        let sin = file.read_f32(&format!("{prefix}/sin"), 1 << 30)?.data;
        let expected = file.read_f32(&format!("{prefix}/output"), 1 << 30)?.data;
        let head = file.read_f32(&format!("{prefix}/head_dim"), 4096)?.shape[0];
        let (tokens, heads, half) = (q.shape[0], q.shape[1], q.shape[3]);
        for tiles in [[32, 32, 0], [64, 64, 0], [64, 64, 1], [128, 64, 1]] {
            let actual = attention_kernel(
                &device,
                &q.data,
                &k,
                &v,
                &cos,
                &sin,
                tokens,
                heads,
                half,
                AttentionMode::Online,
                tokens,
                head,
                tiles,
                true,
            )?;
            let scale = expected.iter().fold(0.0f32, |s, x| s.max(x.abs()));
            let error = actual
                .iter()
                .zip(&expected)
                .fold(0.0f32, |e, (a, b)| e.max((a - b).abs()));
            assert_eq!(actual.len(), expected.len());
            assert!(
                error / scale < 1e-4,
                "case {index}, {tiles:?}: {}",
                error / scale
            );
        }
    }
    Ok(())
}
