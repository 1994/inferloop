//! Draft block attention against a host reference over BF16-rounded operands: blocks that
//! straddle a key tile, wrap their ring, or sit idle.
use super::draft;
use crate::device::CudaDevice;
use cutile::{half::bf16, prelude::*};
use std::error::Error;
use std::sync::Arc;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const D: usize = 128;
const GROUP: usize = 4;
const KV: usize = 2;
const BLOCK: usize = 8;
const RING: usize = 256;
const WINDOW: usize = 100;
/// Tolerance for BF16 operands and probabilities against the F64 reference.
const TOLERANCE: f64 = 2e-2;

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (f32::from(u16::try_from((i * 37 + salt * 11) % 97).unwrap()) - 48.0) / 64.0)
        .collect()
}

fn rounded(value: f32) -> f64 {
    f64::from(bf16::from_f32(value).to_f32())
}

fn upload<T: DType>(device: &CudaDevice, data: Vec<T>, shape: &[usize]) -> TestResult<Tensor<T>> {
    Arc::try_unwrap(device.upload(data, shape)?).map_err(|_| "test upload still shared".into())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn block_attention_reads_every_tile_of_its_window() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    // Slot 0 straddles a key tile, slot 1 wraps, slot 2 is idle, and slot 3
    // crosses a tile boundary at the earliest query's sliding-window start.
    let bases = [60i32, 300, -1, 156];
    let slot_rings = [1i32, 0, 1, 0];
    let slots = bases.len();
    let rows = slots * BLOCK;
    let gd = GROUP * D;
    let query = values(rows * KV * gd, 1);
    let keys = values(KV * 2 * RING * D, 2);
    let value_rows = values(KV * 2 * RING * D, 3);
    let to_bf16 = |data: &[f32]| data.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>();
    let mut out = upload(&device, vec![0.0f32; rows * KV * gd], &[rows, KV, gd])?;
    draft::block_attention(
        (&mut out).partition([BLOCK, 1, gd]),
        &upload(&device, query.clone(), &[rows, KV, gd])?,
        &upload(&device, to_bf16(&keys), &[KV, 2 * RING, D])?,
        &upload(&device, to_bf16(&value_rows), &[KV, 2 * RING, D])?,
        &upload(&device, bases.to_vec(), &[slots])?,
        &upload(&device, slot_rings.to_vec(), &[slots])?,
    )
    .generics(
        [D, GROUP, gd, BLOCK, BLOCK * GROUP, RING, WINDOW]
            .iter()
            .map(ToString::to_string)
            .collect(),
    )
    .sync_on(&device.stream)?;
    let actual = device.read_borrowed(&out)?;
    // The kernel scales the F32 query by `rsqrt(D)` before rounding it to BF16.
    #[expect(clippy::cast_precision_loss, reason = "a small head dimension")]
    let scale = (D as f32).sqrt().recip();
    for (slot, (&base, &ring)) in bases.iter().zip(&slot_rings).enumerate() {
        for head in 0..KV {
            for token in 0..BLOCK {
                for g in 0..GROUP {
                    let row = ((slot * BLOCK + token) * KV + head) * gd + g * D;
                    let got = &actual[row..row + D];
                    let Ok(base) = usize::try_from(base) else {
                        assert!(
                            got.iter().all(|&v| v == 0.0),
                            "idle slot {slot} wrote output"
                        );
                        continue;
                    };
                    let position = base + token;
                    let last = base + BLOCK - 1;
                    let earliest = (position + 1).saturating_sub(WINDOW);
                    let ring = usize::try_from(ring)?;
                    let key_row = |p: usize| (head * 2 * RING + ring * RING + p % RING) * D;
                    let q: Vec<f64> = query[row..row + D]
                        .iter()
                        .map(|&v| rounded(v * scale))
                        .collect();
                    let scores: Vec<f64> = (earliest..=last)
                        .map(|p| {
                            let k = &keys[key_row(p)..key_row(p) + D];
                            q.iter().zip(k).map(|(a, &b)| a * rounded(b)).sum()
                        })
                        .collect();
                    let maximum = scores.iter().copied().fold(f64::MIN, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for d in 0..D {
                        let expected: f64 = (earliest..=last)
                            .zip(&weights)
                            .map(|(p, w)| w * rounded(value_rows[key_row(p) + d]))
                            .sum::<f64>()
                            / total;
                        let error = (f64::from(got[d]) - expected).abs();
                        assert!(
                            error <= TOLERANCE * (1.0 + expected.abs()),
                            "slot {slot} head {head} token {token} group {g} dim {d}: \
                             {} != {expected}",
                            got[d]
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
