//! Independent, nonuniform fixtures exercise row/column layout and block scale indexing.
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use infer_backend_cuda::mlp::Matrix;
use infer_backend_cuda::nvfp4::to_bf16;
use infer_backend_cuda::{device::CudaDevice, mlp::ProjectionWeight};
use infer_core::{Error, Result};

pub fn make(
    device: &CudaDevice,
    encoding: usize,
    rows: usize,
    columns: usize,
) -> Result<(ProjectionWeight, Vec<f32>)> {
    let scale = [0.25, 0.5, 1.0, 2.0];
    let scale_bits = [0x28, 0x30, 0x38, 0x40];
    let small = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut decoded = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            let code = (row * 7 + column * 3) % 16;
            let value = small[code % 8] * if code < 8 { 1.0 } else { -1.0 };
            let factor = match encoding {
                0 => 0.5,
                1 => scale[row % 4],
                3 => scale[(row + column / 16) % 4] / 3.0,
                _ => scale[(row + column / 16) % 4] * 0.5,
            };
            let value = value * factor;
            decoded.push(if encoding == 3 {
                bf16::from_f32(value).to_f32()
            } else {
                value
            });
        }
    }
    let projection = match encoding {
        0 => ProjectionWeight::Dense(Matrix::upload(
            device,
            decoded.iter().copied().map(bf16::from_f32).collect(),
            &[rows, columns],
        )?),
        1 => {
            let bits = [0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c];
            let values = (0..rows)
                .flat_map(|r| {
                    (0..columns).map(move |c| {
                        let code = (r * 7 + c * 3) % 16;
                        f8e4m3fn(bits[code % 8] | if code < 8 { 0 } else { 0x80 })
                    })
                })
                .collect();
            ProjectionWeight::Fp8(
                Matrix::upload(device, values, &[rows, columns])?,
                device
                    .runtime()
                    .upload((0..rows).map(|r| scale[r % 4]).collect())?,
            )
        }
        _ => {
            let mut values = Vec::new();
            for r in 0..rows {
                for c in 0..columns / 2 {
                    let lo = (r * 7 + c * 6) % 16;
                    let hi = (r * 7 + c * 6 + 3) % 16;
                    let byte =
                        u8::try_from(lo | (hi << 4)).map_err(|e| Error::invalid(e.to_string()))?;
                    values.push(byte);
                }
            }
            let scales: Vec<u8> = (0..rows)
                .flat_map(|r| (0..columns / 16).map(move |g| scale_bits[(r + g) % 4]))
                .collect();
            if encoding == 3 {
                let converted = to_bf16(&values, &scales, 3.0, rows, columns, rows * columns * 2)?;
                ProjectionWeight::Dense(Matrix::upload(
                    device,
                    converted.into_iter().map(bf16::from_bits).collect(),
                    &[rows, columns],
                )?)
            } else {
                ProjectionWeight::Fp4(
                    Matrix::upload(
                        device,
                        values.into_iter().map(f4e2m1fnx2::from_bits).collect(),
                        &[rows, columns / 2],
                    )?,
                    Matrix::upload(
                        device,
                        scales.into_iter().map(f8e4m3fn).collect(),
                        &[rows, columns / 16],
                    )?,
                    2.0,
                )
            }
        }
    };
    Ok((projection, decoded))
}
