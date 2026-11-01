//! Independent host F64 scalar recurrence, from original F32 inputs.
use super::{CHUNK_ROWS, MAX_F64_ERROR, SHAPE_RANK, SUPPORTED_DIMS};
use crate::constants::METADATA_FIELDS;
use infer_core::{Error, Result};

const FIXTURE_OFFSET: usize = 3;
const FIXTURE_PERIOD: usize = 101;
const FIXTURE_CENTER: f32 = 50.0;
const FIXTURE_SCALE: f32 = 100.0;
const QKV_SALT: usize = 17;
const BETA_SALT: usize = 13;
const ALPHA_SALT: usize = 7;
const DECAY_SALT: usize = 3;
const BIAS_SALT: usize = 5;
const INITIAL_SALT: usize = 23;
const CORRELATED_BETA: f32 = 8.0;
const CORRELATED_DECAY_LOG: f32 = -10.0;
const NORMALIZATION_EPSILON: f64 = 1e-6;
const SOFTPLUS_LINEAR_THRESHOLD: f64 = 20.0;

pub(super) struct Fixture {
    pub shape: [usize; SHAPE_RANK],
    pub metadata: [i32; METADATA_FIELDS],
    pub qkv: Vec<f32>,
    pub beta: Vec<f32>,
    pub alpha: Vec<f32>,
    pub a_log: Vec<f32>,
    pub bias: Vec<f32>,
    pub initial: Vec<f32>,
    pub state: Vec<f64>,
    pub output: Vec<f64>,
}
fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            (f32::from(
                u16::try_from((i * salt + FIXTURE_OFFSET) % FIXTURE_PERIOD).unwrap_or_default(),
            ) - FIXTURE_CENTER)
                / FIXTURE_SCALE
        })
        .collect()
}
/// Widest prompt graph the exact tuner measures; the reference covers every one of its rungs.
const MAX_REFERENCE_LANES: usize = 2048;

impl Fixture {
    pub fn new(shape: [usize; SHAPE_RANK], correlated: bool) -> Result<Self> {
        Self::new_case(
            shape,
            correlated,
            [
                0,
                i32::try_from(shape[3]).map_err(|_| Error::invalid("reference lanes"))?,
                0,
                0,
            ],
        )
    }
    pub fn new_case(
        shape: [usize; SHAPE_RANK],
        correlated: bool,
        metadata: [i32; METADATA_FIELDS],
    ) -> Result<Self> {
        let [kh, vh, dim, lanes] = shape;
        if kh == 0
            || !vh.is_multiple_of(kh)
            || !SUPPORTED_DIMS.contains(&dim)
            || lanes == 0
            || lanes > MAX_REFERENCE_LANES
            || !lanes.is_multiple_of(CHUNK_ROWS)
        {
            return Err(Error::invalid("unsupported WY reference geometry"));
        }
        let mut qkv = values(lanes * (2 * kh + vh) * dim, QKV_SALT);
        if correlated {
            for lane in 0..lanes {
                qkv[lane * (2 * kh + vh) * dim..][..2 * kh * dim].fill(1.0);
            }
        }
        let beta = if correlated {
            vec![CORRELATED_BETA; lanes * vh]
        } else {
            values(lanes * vh, BETA_SALT)
        };
        let alpha = if correlated {
            vec![0.0; lanes * vh]
        } else {
            values(lanes * vh, ALPHA_SALT)
        };
        let a_log = if correlated {
            vec![CORRELATED_DECAY_LOG; vh]
        } else {
            values(vh, DECAY_SALT)
        };
        let bias = if correlated {
            vec![0.0; vh]
        } else {
            values(vh, BIAS_SALT)
        };
        let initial = values(vh * dim * dim, INITIAL_SALT);
        let mut state: Vec<f64> = initial.iter().copied().map(f64::from).collect();
        let mut output = vec![0.0; lanes * vh * dim];
        let dim_f64 =
            f64::from(u32::try_from(dim).map_err(|_| Error::invalid("reference dimension"))?);
        for lane in 0..lanes {
            if !active(lane, metadata)? {
                continue;
            }
            for head in 0..vh {
                let base = lane * (2 * kh + vh) * dim;
                let key = head / (vh / kh);
                let q: Vec<f64> = qkv[base + key * dim..][..dim]
                    .iter()
                    .copied()
                    .map(f64::from)
                    .collect();
                let k: Vec<f64> = qkv[base + (kh + key) * dim..][..dim]
                    .iter()
                    .copied()
                    .map(f64::from)
                    .collect();
                let qnorm = (q.iter().map(|x| x * x).sum::<f64>() + NORMALIZATION_EPSILON).sqrt()
                    * dim_f64.sqrt();
                let knorm = (k.iter().map(|x| x * x).sum::<f64>() + NORMALIZATION_EPSILON).sqrt();
                let b = 1.0 / (1.0 + (-f64::from(beta[lane * vh + head])).exp());
                let a = f64::from(alpha[lane * vh + head]) + f64::from(bias[head]);
                let soft = if a > SOFTPLUS_LINEAR_THRESHOLD {
                    a
                } else {
                    a.exp().ln_1p()
                };
                let decay = (-f64::from(a_log[head]).exp() * soft).exp();
                let st = &mut state[head * dim * dim..][..dim * dim];
                for x in st.iter_mut() {
                    *x *= decay;
                }
                for col in 0..dim {
                    let prediction = (0..dim)
                        .map(|row| st[row * dim + col] * k[row] / knorm)
                        .sum::<f64>();
                    let delta =
                        (f64::from(qkv[base + (2 * kh + head) * dim + col]) - prediction) * b;
                    for row in 0..dim {
                        st[row * dim + col] = (k[row] / knorm).mul_add(delta, st[row * dim + col]);
                    }
                    output[(lane * vh + head) * dim + col] = (0..dim)
                        .map(|row| st[row * dim + col] * q[row] / qnorm)
                        .sum();
                }
            }
        }
        Ok(Self {
            shape,
            metadata,
            qkv,
            beta,
            alpha,
            a_log,
            bias,
            initial,
            state,
            output,
        })
    }
}
pub(super) fn maximum_error(actual: &[f32], expected: &[f64]) -> Result<f64> {
    if actual.is_empty()
        || actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !a.is_finite() || !b.is_finite())
    {
        return Err(Error::invalid(
            "WY reference result shape or nonfinite value",
        ));
    }
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0.0_f64, f64::max);
    if error >= MAX_F64_ERROR {
        return Err(Error::invalid(format!(
            "WY independent F64 error {error} exceeds original 2e-5 gate"
        )));
    }
    Ok(error)
}

fn active(lane: usize, metadata: [i32; METADATA_FIELDS]) -> Result<bool> {
    let lane = i32::try_from(lane).map_err(|_| Error::invalid("reference lane"))?;
    Ok(lane < metadata[1] && metadata[0] + lane + metadata[2] >= 0)
}
