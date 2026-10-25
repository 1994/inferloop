//! Small numerical test only: no model load and no throughput measurement.
#[cfg(cuda_backend)]
use std::env::args;
#[cfg(cuda_backend)]
fn main() -> infer_core::Result<()> {
    use cuda_core::{f4e2m1fnx2, f8e4m3fn};
    use cutile::half::bf16;
    use infer_backend_cuda::{
        device::CudaDevice,
        mlp::{MlpConfig, MlpGraph, MlpWeights, ProjectionWeight},
        strategy::LinearTiling,
    };
    let device = CudaDevice::new(0)?;
    let d = 48;
    for (encoding, (n, down_scale)) in (0..3)
        .flat_map(|encoding| [(80, 40.0_f32), (560, 280.0_f32)].map(|shape| (encoding, shape)))
    {
        let matrix = |rows, columns| -> infer_core::Result<ProjectionWeight> {
            Ok(match encoding {
                0 => ProjectionWeight::Dense(
                    device.upload(vec![bf16::from_f32(0.5); rows * columns], &[rows, columns])?,
                ),
                1 => ProjectionWeight::Fp8(
                    device.upload(vec![f8e4m3fn(0x38); rows * columns], &[rows, columns])?,
                    device.upload(vec![0.5; rows], &[rows])?,
                ),
                _ => ProjectionWeight::Fp4(
                    device.upload(
                        vec![f4e2m1fnx2::from_bits(0x22); rows * columns / 2],
                        &[rows, columns / 2],
                    )?,
                    device.upload(
                        vec![f8e4m3fn(0x38); rows * columns / 16],
                        &[rows, columns / 16],
                    )?,
                    2.0,
                ),
            })
        };
        let weights = MlpWeights {
            norm: device.upload(vec![0.0; d], &[d])?,
            gate: matrix(n, d)?,
            up: matrix(n, d)?,
            down: matrix(d, n)?,
        };
        let config = MlpConfig {
            hidden: d,
            intermediate: n,
            epsilon: 1e-6,
            norm_offset: 1.0,
            tiling: LinearTiling::new(16, 256)?,
            pdl: args().any(|arg| arg == "--pdl"),
        };
        let mut graph = MlpGraph::new(&device, &config, &weights)?;
        // Drop caller weight handles to check graph-owned storage lifetimes.
        drop(weights);
        for value in [1.0_f32, -0.1, 0.3, 0.0].into_iter().cycle().take(32) {
            let output = graph.apply(&vec![value; d])?;
            let normalized = value / value.mul_add(value, config.epsilon).sqrt();
            let projected = normalized * 24.0;
            let expected =
                (projected / (1.0 + (-projected).exp()) * projected).mul_add(down_scale, value);
            if output
                .iter()
                .any(|v| !v.is_finite() || (v - expected).abs() > 1e-4 * expected.abs().max(1.0))
            {
                return Err(infer_core::Error::invariant(format!(
                    "MLP graph mismatch encoding={encoding}, input={value}, expected={expected}, actual={}",
                    output[0]
                )));
            }
        }
    }
    println!(
        "MLP graph: BF16/FP8/NVFP4, padded dimensions, changed inputs and retained storage passed"
    );
    Ok(())
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA correctness checks require Linux");
    exit(1);
}
