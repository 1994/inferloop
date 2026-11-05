//! Independent F64 attention from actual checkpoint Q/K/V, including every appended cache byte.
use super::{LoadOptions, LoadedModel};
use crate::constants::KV_BLOCK_TOKENS;
use crate::resident::fp8_cache::KvState;
use crate::tuning::table_path as tuning_table_path;
use crate::{
    device::CudaDevice,
    resident::{DeviceProgram, attention_prefill_taps as taps},
    tuning::attention::Choice,
};
use cuda_core::f8e4m3fn;
use infer_core::{Error, ModelId, Result, TensorId};
use std::collections::BTreeMap;
use std::env::var;
use std::fs::read as fs_read;
use std::sync::OnceLock;
type Cache = BTreeMap<TensorId, (Vec<f8e4m3fn>, Vec<f8e4m3fn>)>;
const TOLERANCE: f64 = 1e-4;
fn decode(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let fraction = code & 7;
    let value = if exponent == 0 {
        f32::from(fraction) / 512.0
    } else {
        (1.0 + f32::from(fraction) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if code & 128 == 0 { value } else { -value }
}
fn encode(value: f32) -> u8 {
    static LEVELS: OnceLock<Vec<f32>> = OnceLock::new();
    assert!(value.is_finite());
    let value = value.clamp(-448.0, 448.0);
    let sign = if value.is_sign_negative() { 128 } else { 0 };
    let value = value.abs();
    let levels = LEVELS.get_or_init(|| (0u8..127).map(decode).collect());
    let hi = levels.partition_point(|&x| x < value).min(126);
    let lo = hi.saturating_sub(1);
    let left = value - levels[lo];
    let right = levels[hi] - value;
    let code = if left < right || left.total_cmp(&right).is_eq() && lo.is_multiple_of(2) {
        lo
    } else {
        hi
    };
    u8::try_from(code).unwrap() | sign
}
fn cache(device: &CudaDevice, program: &DeviceProgram) -> Result<Cache> {
    program
        .fp8_states()
        .iter()
        .map(
            |(
                &id,
                KvState {
                    keys: k, values: v, ..
                },
            )| Ok((id, (device.runtime().read(k)?, device.runtime().read(v)?))),
        )
        .collect()
}
struct Chunk {
    query: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    output: Vec<f32>,
    metadata: Vec<i32>,
    table: Vec<i32>,
}
impl Chunk {
    fn read(device: &CudaDevice, tap: &taps::Entry) -> Result<Self> {
        Ok(Self {
            query: device.runtime().read(&tap.query)?,
            keys: device.runtime().read(&tap.keys)?,
            values: device.runtime().read(&tap.values)?,
            output: device.runtime().read(&tap.output)?,
            metadata: device.runtime().read(&tap.metadata)?,
            table: device.runtime().read(&tap.table)?,
        })
    }
    fn position(&self, lane: usize) -> Result<Option<usize>> {
        let p = self.metadata[0]
            + self.metadata[2]
            + i32::try_from(lane).map_err(|e| Error::invalid(e.to_string()))?;
        Ok(usize::try_from(p).ok())
    }
    fn physical(&self, p: usize) -> Result<usize> {
        let b = KV_BLOCK_TOKENS;
        Ok(
            usize::try_from(self.table[p / b]).map_err(|e| Error::invalid(e.to_string()))? * b
                + p % b,
        )
    }
    fn append(
        &self,
        tap: &taps::Entry,
        before: &(Vec<f8e4m3fn>, Vec<f8e4m3fn>),
        after: &(Vec<f8e4m3fn>, Vec<f8e4m3fn>),
    ) -> Result<()> {
        let count = usize::try_from(self.metadata[1]).map_err(|e| Error::invalid(e.to_string()))?;
        for (raw, previous, actual, scale) in [
            (&self.keys, &before.0, &after.0, tap.scales[0]),
            (&self.values, &before.1, &after.1, tap.scales[1]),
        ] {
            let mut expected = previous.iter().map(|x| x.0).collect::<Vec<_>>();
            for lane in 0..count {
                let Some(p) = self.position(lane)? else {
                    continue;
                };
                let p = self.physical(p)?;
                for h in 0..tap.kv_heads {
                    for d in 0..tap.dim {
                        let source = (lane * tap.kv_heads + h) * tap.dim + d;
                        expected[(h * tap.capacity + p) * tap.dim + d] =
                            encode(raw[source] / scale);
                    }
                }
            }
            assert_eq!(expected.len(), actual.len());
            for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(a.0, *b, "FP8 append index={index}");
            }
        }
        Ok(())
    }
    fn reference(
        &self,
        tap: &taps::Entry,
        kv: &(Vec<f8e4m3fn>, Vec<f8e4m3fn>),
    ) -> Result<Vec<f64>> {
        let keys =
            kv.0.iter()
                .map(|v| f64::from(decode(v.0) * tap.scales[0]))
                .collect::<Vec<_>>();
        let values =
            kv.1.iter()
                .map(|v| f64::from(decode(v.0) * tap.scales[1]))
                .collect::<Vec<_>>();
        let count = usize::try_from(self.metadata[1]).map_err(|e| Error::invalid(e.to_string()))?;
        let scale =
            f64::from(u32::try_from(tap.dim).map_err(|e| Error::invalid(e.to_string()))?).sqrt();
        let mut expected = vec![0.0; self.output.len()];
        for lane in 0..count {
            let Some(p) = self.position(lane)? else {
                continue;
            };
            let start = if tap.window == 0 {
                0
            } else {
                (p + 1).saturating_sub(tap.window)
            };
            for h in 0..tap.heads {
                let q0 = (lane * tap.heads + h) * tap.dim;
                let kh = h / (tap.heads / tap.kv_heads);
                let offsets = (start..=p)
                    .map(|pos| Ok((kh * tap.capacity + self.physical(pos)?) * tap.dim))
                    .collect::<Result<Vec<_>>>()?;
                let mut scores = offsets
                    .iter()
                    .map(|&k0| {
                        self.query[q0..q0 + tap.dim]
                            .iter()
                            .zip(&keys[k0..k0 + tap.dim])
                            .map(|(&q, &k)| f64::from(q) * k)
                            .sum::<f64>()
                            / scale
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let sum = scores.iter().sum::<f64>();
                for (&k0, s) in offsets.iter().zip(scores) {
                    for d in 0..tap.dim {
                        expected[q0 + d] = (s / sum).mul_add(values[k0 + d], expected[q0 + d]);
                    }
                }
            }
        }
        Ok(expected)
    }
}
fn check(
    device: &CudaDevice,
    program: &DeviceProgram,
    before: &Cache,
    width: usize,
    choice: Choice,
) -> Result<()> {
    let after = cache(device, program)?;
    let mut taps_seen = 0;
    for ((output, lanes), tap) in &program.attention_taps().buffers {
        if *lanes != width {
            continue;
        }
        let chunk = Chunk::read(device, tap)?;
        chunk.append(tap, &before[&tap.state], &after[&tap.state])?;
        let reference = chunk.reference(tap, &after[&tap.state])?;
        let mut max = 0.0_f64;
        for (i, (&a, &b)) in chunk.output.iter().zip(&reference).enumerate() {
            assert!(a.is_finite() && b.is_finite());
            let error = (f64::from(a) - b).abs() / b.abs().max(1.0);
            max = max.max(error);
            assert!(
                error < TOLERANCE,
                "{choice:?} width={width} output={output:?} base={} element={i} error={error} actual={a} expected={b}",
                chunk.metadata[0]
            );
        }
        taps_seen += 1;
        eprintln!(
            "ATTENTION_CHECKPOINT {}",
            serde_json::json!({"choice":choice,"width":width,"output":output,"metadata":chunk.metadata,"elements":reference.len(),"cache_elements":after[&tap.state].0.len()*2,"full_f64":true,"full_append_codes":true,"max_normalized_error":max,"tolerance":TOLERANCE})
        );
    }
    assert!(taps_seen > 0, "no actual checkpoint attention taps");
    Ok(())
}
#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL, frozen 511-token INFER_TEST_TOKENS; safe-run"]
fn checkpoint_attention_full_f64_and_original_append() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens_path = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&fs_read(tokens_path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(tokens.len(), 511);
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 256,
            verification_width: 3,
            autotune: false,
            chunked_recurrent: true,
            ..LoadOptions::default()
        },
    )?;
    let table_path =
        tuning_table_path().ok_or_else(|| Error::invalid("frozen tuning cache required"))?;
    let original_cache = fs_read(&table_path).map_err(|e| Error::invalid(e.to_string()))?;
    for width in [64, 256] {
        loaded.weights.prefill_width = width;
        loaded.weights.narrow_prefill_width = 0;
        for choice in [
            Choice::Simt,
            Choice::Tiled { query_tile: 16 },
            Choice::Tiled { query_tile: 32 },
        ] {
            let mut program = taps::with_choice(choice, || loaded.sequence(1024))?;
            assert!(!program.attention_taps().buffers.is_empty());
            for length in [511, 51] {
                for (chunk, input) in tokens[..length].chunks(width).enumerate() {
                    let before = cache(loaded.device(), &program)?;
                    program.prefill_batch(input, chunk * width, true)?;
                    check(loaded.device(), &program, &before, width, choice)?;
                }
            }
        }
    }
    assert_eq!(
        fs_read(table_path).map_err(|e| Error::invalid(e.to_string()))?,
        original_cache
    );
    Ok(())
}
