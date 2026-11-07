//! Independent F64 attention on BF16-rounded genuine F32 cache values.
use super::super::Shape;
use crate::constants::KV_BLOCK_TOKENS;
use crate::device::device_error;
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::bf16;
const SEED_STEP: u64 = 0x9e37_79b9_7f4a_7c15;
const MIX_FIRST: u64 = 0xbf58_476d_1ce4_e5b9;
const MIX_SECOND: u64 = 0x94d0_49bb_1331_11eb;
const SHIFT_FIRST: u32 = 30;
const SHIFT_SECOND: u32 = 27;
const SHIFT_FINAL: u32 = 31;
const HIGH_WORD: u32 = 48;
const CENTER: f32 = 32_760.0;
const DIVISOR: f32 = 32_749.0;
const Q_PHASE: usize = 3;
const K_PHASE: usize = 17;
const V_PHASE: usize = 29;
const AK_PHASE: usize = 41;
const AV_PHASE: usize = 59;
const QUERY_SCALE: f32 = 0.125;
const TABLE_SHIFT: usize = 3;
const UNIFORM: usize = 3;
pub(super) const FIXTURES: usize = 4;
pub(super) const MAX_ERROR: f64 = 0.01;
pub(super) const MAX_RELATIVE_L2: f64 = 0.02;
pub(super) struct Fixture {
    pub shape: Shape,
    pub capacity: usize,
    pub query: Vec<f32>,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub append_keys: Vec<f32>,
    pub append_values: Vec<f32>,
    pub metadata: Vec<i32>,
    pub table: Vec<i32>,
    pub expected_keys: Vec<f32>,
    pub expected_values: Vec<f32>,
    rounded_keys: Vec<f32>,
    rounded_values: Vec<f32>,
}
fn values(count: usize, phase: usize) -> Result<Vec<f32>> {
    (0..count)
        .map(|i| {
            let mut value = u64::try_from(i).map_err(device_error)?.wrapping_add(
                u64::try_from(phase)
                    .map_err(device_error)?
                    .wrapping_mul(SEED_STEP),
            );
            value = (value ^ (value >> SHIFT_FIRST)).wrapping_mul(MIX_FIRST);
            value = (value ^ (value >> SHIFT_SECOND)).wrapping_mul(MIX_SECOND);
            value ^= value >> SHIFT_FINAL;
            let code = value >> HIGH_WORD;
            Ok((f32::from(u16::try_from(code).map_err(device_error)?) - CENTER) / DIVISOR)
        })
        .collect()
}
fn rounded(value: f32) -> f64 {
    f64::from(bf16::from_f32(value).to_f32())
}
impl Fixture {
    pub fn short(shape: Shape, index: usize) -> Result<Self> {
        let capacity = (shape.lanes * 2).next_multiple_of(KV_BLOCK_TOKENS);
        let (history, count, offset) = match index {
            0 => (0, shape.lanes, 0),
            1 => (shape.lanes, shape.lanes - 1, -1),
            2 => (0, 1, -1),
            UNIFORM => (shape.lanes / 2, shape.lanes / 2, 0),
            _ => return Err(Error::invalid("F32 attention fixture index")),
        };
        let mut fixture = Self::new(shape, history, count, offset, capacity)?;
        if index == UNIFORM {
            fixture.query.fill(0.0);
        }
        Ok(fixture)
    }
    pub fn history(shape: Shape, history: usize, capacity: usize) -> Result<Self> {
        Self::new(shape, history, shape.lanes - 1, 0, capacity)
    }
    fn new(
        shape: Shape,
        history: usize,
        count: usize,
        offset: i32,
        capacity: usize,
    ) -> Result<Self> {
        let block = KV_BLOCK_TOKENS;
        if !capacity.is_multiple_of(block)
            || history.checked_add(count).is_none_or(|end| end > capacity)
            || count > shape.lanes
        {
            return Err(Error::invalid("F32 attention fixture bounds"));
        }
        let mut query = values(shape.lanes * shape.heads * shape.dim, Q_PHASE)?;
        for value in &mut query {
            *value *= QUERY_SCALE;
        }
        let size = shape.kv_heads * capacity * shape.dim;
        let keys = values(size, K_PHASE)?;
        let cache_values = values(size, V_PHASE)?;
        let raw = shape.lanes * shape.kv_heads * shape.dim;
        let blocks = capacity / block;
        let mut f = Self {
            shape,
            capacity,
            query,
            expected_keys: keys.clone(),
            expected_values: cache_values.clone(),
            keys,
            values: cache_values,
            append_keys: values(raw, AK_PHASE)?,
            append_values: values(raw, AV_PHASE)?,
            metadata: vec![
                i32::try_from(history).map_err(device_error)?,
                i32::try_from(count).map_err(device_error)?,
                offset,
                0,
            ],
            table: (0..blocks)
                .map(|i| i32::try_from((i + TABLE_SHIFT) % blocks).map_err(device_error))
                .collect::<Result<Vec<_>>>()?,
            rounded_keys: Vec::new(),
            rounded_values: Vec::new(),
        };
        for row in 0..count {
            let position = f.metadata[0] + offset + i32::try_from(row).map_err(device_error)?;
            let Ok(position) = usize::try_from(position) else {
                continue;
            };
            let physical = f.physical(position)?;
            for head in 0..shape.kv_heads {
                let src = (row * shape.kv_heads + head) * shape.dim;
                let dst = (head * capacity + physical) * shape.dim;
                f.expected_keys[dst..dst + shape.dim]
                    .copy_from_slice(&f.append_keys[src..src + shape.dim]);
                f.expected_values[dst..dst + shape.dim]
                    .copy_from_slice(&f.append_values[src..src + shape.dim]);
            }
        }
        f.rounded_keys = f
            .expected_keys
            .iter()
            .map(|&v| bf16::from_f32(v).to_f32())
            .collect();
        f.rounded_values = f
            .expected_values
            .iter()
            .map(|&v| bf16::from_f32(v).to_f32())
            .collect();
        Ok(f)
    }
    fn physical(&self, position: usize) -> Result<usize> {
        let block = KV_BLOCK_TOKENS;
        Ok(
            usize::try_from(self.table[position / block]).map_err(device_error)? * block
                + position % block,
        )
    }
    pub fn reference(&self, row: usize, head: usize) -> Result<Vec<f64>> {
        let s = self.shape;
        if row >= usize::try_from(self.metadata[1]).map_err(device_error)? {
            return Ok(vec![0.0; s.dim]);
        }
        let position =
            self.metadata[0] + self.metadata[2] + i32::try_from(row).map_err(device_error)?;
        let Ok(position) = usize::try_from(position) else {
            return Ok(vec![0.0; s.dim]);
        };
        let q0 = (row * s.heads + head) * s.dim;
        let kv_head = head / (s.heads / s.kv_heads);
        let query = self.query[q0..q0 + s.dim]
            .iter()
            .copied()
            .map(rounded)
            .collect::<Vec<_>>();
        let mut scores = Vec::with_capacity(position + 1);
        let scale = f64::from(u32::try_from(s.dim).map_err(device_error)?).sqrt();
        for p in 0..=position {
            let start = (kv_head * self.capacity + self.physical(p)?) * s.dim;
            scores.push(
                query
                    .iter()
                    .zip(&self.rounded_keys[start..start + s.dim])
                    .map(|(&q, &k)| q * f64::from(k))
                    .sum::<f64>()
                    / scale,
            );
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for value in &mut scores {
            *value = (*value - maximum).exp();
        }
        let denominator = scores.iter().sum::<f64>();
        let mut output = vec![0.0; s.dim];
        for (p, &weight) in scores.iter().enumerate() {
            let start = (kv_head * self.capacity + self.physical(p)?) * s.dim;
            for (out, &v) in output
                .iter_mut()
                .zip(&self.rounded_values[start..start + s.dim])
            {
                *out += weight * f64::from(v) / denominator;
            }
        }
        Ok(output)
    }
}
