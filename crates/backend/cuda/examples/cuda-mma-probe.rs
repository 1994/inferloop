//! Capability/accuracy probe, deliberately not a performance benchmark.
#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use infer_backend_cuda::device::CudaDevice;
#[cfg(cuda_backend)]
use std::error::Error;
#[cfg(cuda_backend)]
#[cutile::module]
mod probe {
    use cutile::core::{
        Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_2, UnpackF4e2m1fnx2Tile, constant,
        f4e2m1fnx2, f8e4m3fn, get_tile_block_id, mmaf_scaled, num_tiles,
    };
    #[cutile::entry()]
    fn scaled<const BM: i32, const BN: i32, const BK: i32, const BP: i32, const BS: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        a: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        b: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        sa: &Tensor<f8e4m3fn, { [-1, -1] }>,
        sb: &Tensor<f8e4m3fn, { [-1, -1] }>,
    ) {
        let id = get_tile_block_id();
        let ap = a.partition(shape![BM, BP]);
        let bp = b.partition(shape![BN, BP]);
        let sap = sa.partition(shape![BM, BS]);
        let sbp = sb.partition(shape![BN, BS]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0, shape![BM, BN]);
        for k in 0i32..num_tiles(&ap, 1) {
            let a = ap.load([id.0, k]).unpack(shape![BM, BK]);
            let b = bp.load([id.1, k]).unpack(shape![BN, BK]).transpose();
            let sa = sap.load([id.0, k]);
            let sb = sbp.load([id.1, k]).transpose();
            acc = mmaf_scaled(a, b, acc, sa, sb);
        }
        out.store(acc);
    }
}

#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn Error>> {
    use cutile::prelude::*;
    enable_default()?;
    let target = CudaDevice::new(0)?.target().name();
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let mut results = Vec::new();
    for (m, n, k, bm, bn, bk) in [
        (16, 16, 128, 16, 16, 64),
        (17, 19, 128, 16, 16, 64),
        (32, 64, 256, 32, 64, 128),
        (64, 64, 256, 64, 64, 128),
        (128, 128, 256, 128, 128, 128),
    ] {
        let (a, af, sa) = operand(m, k, 1);
        let (b, bf, sb) = operand(n, k, 3);
        let a = api::copy_host_vec_to_device(&Arc::new(a))
            .reshape(&[m, k / 2])
            .sync_on(&stream)?;
        let b = api::copy_host_vec_to_device(&Arc::new(b))
            .reshape(&[n, k / 2])
            .sync_on(&stream)?;
        let sa = api::copy_host_vec_to_device(&Arc::new(sa))
            .reshape(&[m, k / 16])
            .sync_on(&stream)?;
        let sb = api::copy_host_vec_to_device(&Arc::new(sb))
            .reshape(&[n, k / 16])
            .sync_on(&stream)?;
        let out = probe::scaled(api::zeros::<f32>(&[m, n]).partition([bm, bn]), a, b, sa, sb)
            .generics(
                [bm, bn, bk, bk / 2, bk / 16]
                    .map(|v| v.to_string())
                    .to_vec(),
            )
            .first()
            .unpartition()
            .sync_on(&stream)?;
        let values = out.to_host_vec().sync_on(&stream)?;
        let mut max_error = 0.0_f64;
        for row in 0..m {
            for col in 0..n {
                let expected: f64 = (0..k).map(|i| af[row * k + i] * bf[col * k + i]).sum();
                let actual = f64::from(values[row * n + col]);
                let error = (expected - actual).abs();
                if !actual.is_finite() || error > 1e-5 * expected.abs().max(1.0) {
                    return Err(format!(
                        "mmaf_scaled mismatch ({row},{col}): {actual} vs {expected}"
                    )
                    .into());
                }
                max_error = max_error.max(error);
            }
        }
        results.push(
            serde_json::json!({"m":m,"n":n,"k":k,"tile":[bm,bn,bk],"max_absolute_error":max_error}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "operation":"cuda_tile.mmaf_scaled", "precision":"NVFP4 E2M1, E4M3 block16 scales, F32 accumulation",
            "cutile":"0.4.0", "passed":true, "cases":results, "performance_measured":false
            ,"target": target
        }))?
    );
    Ok(())
}

#[cfg(cuda_backend)]
fn operand(
    rows: usize,
    k: usize,
    seed: usize,
) -> (
    Vec<cuda_core::f4e2m1fnx2>,
    Vec<f64>,
    Vec<cuda_core::f8e4m3fn>,
) {
    use cuda_core::{f4e2m1fnx2, f8e4m3fn};
    let table = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let codes: Vec<u8> = (0_u8..16).collect();
    let mut values = Vec::new();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    for row in 0..rows {
        for group in 0..k / 16 {
            let scale = (row + group + seed) % 3;
            scales.push(f8e4m3fn([0x30, 0x38, 0x40][scale]));
            for pair in 0..8 {
                let low = codes[(row * 3 + group * 5 + pair * seed) % 16];
                let high = codes[(row * 7 + group + pair + seed) % 16];
                packed.push(f4e2m1fnx2::from_bits(low | high << 4));
                values.push(table[usize::from(low)] * [0.5, 1.0, 2.0][scale]);
                values.push(table[usize::from(high)] * [0.5, 1.0, 2.0][scale]);
            }
        }
    }
    (packed, values, scales)
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA capability probe requires Linux");
    exit(1);
}
