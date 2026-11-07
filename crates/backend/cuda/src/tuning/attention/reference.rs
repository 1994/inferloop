//! Independent full-output softmax and FP8 cache reference for pre-capture search.
mod history;
use super::Shape;
use crate::constants::KV_BLOCK_TOKENS;
use crate::device::device_error;
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::bf16;
use infer_cuda_kernels::tile::runtime::f8e4m3fn;
use std::mem::replace;
pub(super) const FIXTURES: usize = 4;
pub(super) const KEY_SCALE: f32 = 0.5;
pub(super) const VALUE_SCALE: f32 = 2.0;
const PERIOD: usize = 197;
const DIVISOR: f32 = 137.0;
const CODE_START: u8 = 32;
const CODE_STRIDE: usize = 11;
const CODE_PERIOD: usize = 31;
const SIGN_PERIOD: usize = 3;
const SIGN_BIT: u8 = 128;
const EXPONENT_MASK: u8 = 127;
const MANTISSA_BITS: u32 = 3;
const MANTISSA_MASK: u8 = 7;
const SUBNORMAL_DIVISOR: f32 = 512.0;
const MANTISSA_DIVISOR: f32 = 8.0;
const EXPONENT_BIAS: i32 = 7;
const UNIFORM_FIXTURE: usize = 3;
const QUERY_STRIDE: usize = 23;
const QUERY_PHASE: usize = 7;
const QUERY_CENTER: f32 = 98.0;
const APPEND_KEY_PHASE: usize = 7;
const APPEND_VALUE_PHASE: usize = 19;
const HISTORY_KEY_PHASE: usize = 3;
const HISTORY_VALUE_PHASE: usize = 13;
const SHORT_WINDOW: usize = 7;

pub(super) struct Fixture {
    pub shape: Shape,
    pub capacity: usize,
    pub metadata: Vec<i32>,
    pub table: Vec<i32>,
    pub query: Vec<f32>,
    pub append_keys: Vec<f32>,
    pub append_values: Vec<f32>,
    pub keys: Vec<f8e4m3fn>,
    pub values: Vec<f8e4m3fn>,
    pub expected_keys: Vec<f8e4m3fn>,
    pub expected_values: Vec<f8e4m3fn>,
    pub expected: Vec<f64>,
    pub window: usize,
}
fn code(index: usize, phase: usize) -> Result<f8e4m3fn> {
    let magnitude = CODE_START
        + u8::try_from((index * CODE_STRIDE + phase) % CODE_PERIOD).map_err(device_error)?;
    Ok(f8e4m3fn(
        magnitude
            + if (index + phase).is_multiple_of(SIGN_PERIOD) {
                SIGN_BIT
            } else {
                0
            },
    ))
}
pub(super) fn decode(value: f8e4m3fn) -> f32 {
    let code = value.0;
    let exponent = (code & EXPONENT_MASK) >> MANTISSA_BITS;
    let mantissa = code & MANTISSA_MASK;
    let magnitude = if exponent == 0 {
        f32::from(mantissa) / SUBNORMAL_DIVISOR
    } else {
        (1.0 + f32::from(mantissa) / MANTISSA_DIVISOR)
            * 2.0_f32.powi(i32::from(exponent) - EXPONENT_BIAS)
    };
    if code & SIGN_BIT == 0 {
        magnitude
    } else {
        -magnitude
    }
}
impl Fixture {
    pub fn new(shape: Shape, index: usize) -> Result<Self> {
        let capacity = (shape.lanes * 2).next_multiple_of(KV_BLOCK_TOKENS);
        let (base, count, offset) = match index {
            0 => (0, shape.lanes - 1, 0),
            1 => (shape.lanes, shape.lanes - 1, -1),
            2 => (0, 1, -1),
            UNIFORM_FIXTURE => (shape.lanes / 2, shape.lanes / 2, 0),
            _ => return Err(Error::invalid("attention fixture index")),
        };
        let blocks = capacity / KV_BLOCK_TOKENS;
        let table = (0..blocks)
            .map(|i| {
                i32::try_from(if index == 1 { i ^ 1 } else { i })
                    .map_err(|e| Error::invalid(e.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let query = (0..shape.lanes * shape.heads * shape.dim)
            .map(|i| {
                Ok(if index == UNIFORM_FIXTURE {
                    0.0
                } else {
                    (f32::from(
                        u16::try_from((i * QUERY_STRIDE + index * QUERY_PHASE) % PERIOD)
                            .map_err(device_error)?,
                    ) - QUERY_CENTER)
                        / DIVISOR
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let raw_len = shape.lanes * shape.kv_heads * shape.dim;
        let append_keys = (0..raw_len)
            .map(|i| code(i, APPEND_KEY_PHASE).map(|c| decode(c) * KEY_SCALE))
            .collect::<Result<Vec<_>>>()?;
        let append_values = (0..raw_len)
            .map(|i| code(i, APPEND_VALUE_PHASE).map(|c| decode(c) * VALUE_SCALE))
            .collect::<Result<Vec<_>>>()?;
        let size = shape.kv_heads * capacity * shape.dim;
        let keys = (0..size)
            .map(|i| code(i, HISTORY_KEY_PHASE))
            .collect::<Result<Vec<_>>>()?;
        let values = (0..size)
            .map(|i| code(i, HISTORY_VALUE_PHASE))
            .collect::<Result<Vec<_>>>()?;
        let metadata = vec![
            i32::try_from(base).map_err(|e| Error::invalid(e.to_string()))?,
            i32::try_from(count).map_err(|e| Error::invalid(e.to_string()))?,
            offset,
            0,
        ];
        let mut fixture = Self {
            shape,
            capacity,
            metadata,
            table,
            query,
            append_keys,
            append_values,
            expected_keys: keys.clone(),
            expected_values: values.clone(),
            keys,
            values,
            expected: Vec::new(),
            window: if index == UNIFORM_FIXTURE {
                SHORT_WINDOW
            } else {
                shape.window
            },
        };
        fixture.append_reference()?;
        fixture.expected = fixture.reference()?;
        Ok(fixture)
    }
    pub fn mma(shape: Shape, index: usize) -> Result<Self> {
        let mut fixture = Self::new(shape, index)?;
        fixture.window = 0;
        let rounded = fixture
            .query
            .iter()
            .map(|&q| bf16::from_f32(q).to_f32())
            .collect();
        let original = replace(&mut fixture.query, rounded);
        // Fixture KV codes times power-of-two scales are exactly representable in BF16.
        fixture.expected = fixture.reference()?;
        fixture.query = original;
        Ok(fixture)
    }
    fn physical(&self, position: usize) -> Result<usize> {
        let block = KV_BLOCK_TOKENS;
        Ok(
            usize::try_from(self.table[position / block]).map_err(device_error)? * block
                + position % block,
        )
    }
    fn append_reference(&mut self) -> Result<()> {
        let count = usize::try_from(self.metadata[1]).map_err(device_error)?;
        for lane in 0..count {
            let position =
                i32::try_from(lane).map_err(device_error)? + self.metadata[0] + self.metadata[2];
            let Ok(position) = usize::try_from(position) else {
                continue;
            };
            let physical = self.physical(position)?;
            for head in 0..self.shape.kv_heads {
                for d in 0..self.shape.dim {
                    let input = (lane * self.shape.kv_heads + head) * self.shape.dim + d;
                    let output = (head * self.capacity + physical) * self.shape.dim + d;
                    // Sources are decoded finite codes times exact powers of two.
                    self.expected_keys[output] = code(input, APPEND_KEY_PHASE)?;
                    self.expected_values[output] = code(input, APPEND_VALUE_PHASE)?;
                }
            }
        }
        Ok(())
    }
    fn scores(
        &self,
        query: &[f32],
        keys: &[f64],
        head: usize,
        start: usize,
        position: usize,
    ) -> Result<Vec<f64>> {
        let dim = self.shape.dim;
        let scale = f64::from(u32::try_from(dim).map_err(device_error)?).sqrt();
        (start..=position)
            .map(|p| {
                let offset = (head * self.capacity + self.physical(p)?) * dim;
                Ok(query
                    .iter()
                    .zip(&keys[offset..offset + dim])
                    .map(|(&q, &k)| f64::from(q) * k)
                    .sum::<f64>()
                    / scale)
            })
            .collect()
    }
    fn reference(&self) -> Result<Vec<f64>> {
        let Shape {
            heads,
            kv_heads,
            dim,
            lanes,
            ..
        } = self.shape;
        let keys = self
            .expected_keys
            .iter()
            .map(|&c| f64::from(decode(c) * KEY_SCALE))
            .collect::<Vec<_>>();
        let values = self
            .expected_values
            .iter()
            .map(|&c| f64::from(decode(c) * VALUE_SCALE))
            .collect::<Vec<_>>();
        let mut output = vec![0.0; lanes * heads * dim];
        for lane in 0..usize::try_from(self.metadata[1]).map_err(device_error)? {
            let position =
                i32::try_from(lane).map_err(device_error)? + self.metadata[0] + self.metadata[2];
            let Ok(position) = usize::try_from(position) else {
                continue;
            };
            let start = if self.window == 0 {
                0
            } else {
                (position + 1).saturating_sub(self.window)
            };
            for head in 0..heads {
                let q0 = (lane * heads + head) * dim;
                let kv_head = head / (heads / kv_heads);
                let mut scores =
                    self.scores(&self.query[q0..q0 + dim], &keys, kv_head, start, position)?;
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for score in &mut scores {
                    *score = (*score - max).exp();
                }
                let denominator = scores.iter().sum::<f64>();
                for (i, probability) in scores.iter().enumerate() {
                    let offset = (kv_head * self.capacity + self.physical(start + i)?) * dim;
                    for d in 0..dim {
                        output[q0 + d] += probability * values[offset + d] / denominator;
                    }
                }
            }
        }
        Ok(output)
    }
}
